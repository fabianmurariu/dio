# SQL Query Execution Engine using rust-lms

## Overview

This document provides a concrete implementation plan for building a SQL query execution engine that:
- Reads Parquet files using the `parquet` and `arrow` crates
- Parses SQL using `datafusion-sql` and `datafusion-expr` (logical plan + expressions)
- Compiles query pipelines using `rust-lms` (Cranelift JIT)
- Operates **zero-copy** on Arrow RecordBatches via staged array wrappers
- Fuses operators until pipeline breakers (join/group by)
- Uses **runtime Rust code** for complex operations (hash tables)

**Key Principles:**
1. **Zero-copy**: Staged arrays wrap pointers directly into Arrow buffers - no data copying
2. **Hybrid execution**: Simple operations (scan, filter, project) are JIT-compiled; complex operations (hash tables) remain in Rust and are called via external functions
3. **Pipeline segments**: Each segment compiles to one function; Rust orchestrates between segments

---

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  SQL Query  │────▶│  datafusion-sql │────▶│  LogicalPlan│
│   String    │     │    + expr       │     │  (from DF)  │
└─────────────┘     └─────────────────┘     └─────────────┘
                                                   │
                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────────┐
│   Result    │◀────│   Pipeline  │◀────│ Physical Plan   │
│  (Arrow)    │     │ Orchestrator│     │ (our pipelines) │
└─────────────┘     └─────────────┘     └─────────────────┘
                          │
           ┌──────────────┼──────────────┐
           ▼              ▼              ▼
    ┌────────────┐ ┌────────────┐ ┌────────────┐
    │ Pipeline 1 │ │ Pipeline 2 │ │ Pipeline N │
    │  (JIT fn)  │ │  (JIT fn)  │ │  (JIT fn)  │
    └────────────┘ └────────────┘ └────────────┘
           │              │              │
           └──────────────┼──────────────┘
                          ▼
                 ┌─────────────────┐
                 │  Runtime Rust   │
                 │  (hash tables,  │
                 │   allocators)   │
                 └─────────────────┘
```

### Execution Model

1. **Pipeline 1** (build side): `scan → filter → project → hash_table_insert()`
   - JIT-compiled loop over input RecordBatch
   - Calls external Rust function to insert into hash table

2. **Pipeline 2** (probe side): `scan → filter → hash_table_probe() → project → output`
   - JIT-compiled loop over input RecordBatch
   - Calls external Rust function to probe hash table
   - Continues JIT-compiled processing of matches

---

## Phase 1: Staged Array Types (Zero-Copy Arrow Integration)

This is the foundation. Staged arrays wrap Arrow array buffers with zero copying.

### 1.1 Core Staged Array Traits

```rust
// src/staged_arrays/mod.rs

use rust_lms::prelude::*;

/// Marker trait for types that can be elements of staged arrays
pub trait StagedElement: StagedType {
    /// The Rust type this represents
    type Native;
}

impl StagedElement for I64Type {
    type Native = i64;
}

impl StagedElement for F64Type {
    type Native = f64;
}

impl StagedElement for BoolType {
    type Native = bool;
}

// StringView is represented as i128 (16 bytes)
impl StagedElement for I128Type {
    type Native = i128;
}
```

### 1.2 Primitive Staged Array (Int64, Float64)

For primitive arrays, Arrow stores values contiguously.

```rust
// src/staged_arrays/primitive.rs

use rust_lms::prelude::*;

/// A staged primitive array - wraps Arrow PrimitiveArray<T> zero-copy
///
/// Arrow layout for PrimitiveArray<Int64>:
/// - data_ptr: *const i64 (values buffer)
/// - length: usize
/// - validity_ptr: *const u8 (null bitmap, optional)
///
/// All pointers come directly from Arrow buffers - NO COPYING
#[derive(Clone)]
pub struct StagedPrimitiveArray<T: StagedElement> {
    /// Pointer to values buffer (from Arrow)
    pub data_ptr: Var<SPtr<T>>,
    /// Number of elements
    pub len: Var<U64Type>,
    /// Pointer to validity bitmap (null = bit is 0)
    /// If all values are valid, this can be null pointer
    pub validity_ptr: Var<SPtr<U8Type>>,
    /// Whether validity bitmap is present
    pub has_validity: bool,  // Static - known at compile time
}

impl<T: StagedElement> StagedPrimitiveArray<T> {
    /// Create from Arrow array pointers (called at staging time)
    ///
    /// The pointers are passed as function arguments at runtime,
    /// but the structure (which pointers exist) is known at staging time
    pub fn from_arrow_ptrs(
        data_ptr: Var<SPtr<T>>,
        len: Var<U64Type>,
        validity_ptr: Var<SPtr<U8Type>>,
        has_validity: bool,
    ) -> Self {
        Self { data_ptr, len, validity_ptr, has_validity }
    }

    /// Get element at index (unchecked)
    /// Returns: Var<T> - the value at position idx
    pub fn get(&self, idx: Var<U64Type>) -> Var<T> {
        // Generated code: *(data_ptr + idx)
        ptr_index(self.data_ptr, idx)
    }

    /// Check if element at index is valid (not null)
    /// Returns: Var<BoolType>
    pub fn is_valid(&self, idx: Var<U64Type>) -> Var<BoolType> {
        if !self.has_validity {
            // No validity bitmap = all values valid
            Const::new(true).into()
        } else {
            // Bitmap check: validity_ptr[idx / 8] & (1 << (idx % 8)) != 0
            let byte_idx = div(idx, Const::new(8u64));
            let bit_idx = rem(idx, Const::new(8u64));
            let byte = ptr_index(self.validity_ptr, byte_idx);
            let mask = shl(Const::new(1u8), bit_idx.cast::<U8Type>());
            ne(and(byte, mask), Const::new(0u8))
        }
    }

    /// Length of the array
    pub fn length(&self) -> Var<U64Type> {
        self.len
    }
}
```

### 1.3 Boolean Staged Array

Booleans in Arrow are bit-packed (8 values per byte).

```rust
// src/staged_arrays/boolean.rs

/// A staged boolean array - wraps Arrow BooleanArray zero-copy
///
/// Arrow layout for BooleanArray:
/// - data_ptr: *const u8 (bit-packed values, 8 bools per byte)
/// - length: usize
/// - validity_ptr: *const u8 (null bitmap)
#[derive(Clone)]
pub struct StagedBooleanArray {
    /// Pointer to bit-packed values buffer
    pub data_ptr: Var<SPtr<U8Type>>,
    /// Number of elements (not bytes!)
    pub len: Var<U64Type>,
    /// Pointer to validity bitmap
    pub validity_ptr: Var<SPtr<U8Type>>,
    pub has_validity: bool,
}

impl StagedBooleanArray {
    /// Get boolean at index
    pub fn get(&self, idx: Var<U64Type>) -> Var<BoolType> {
        // data_ptr[idx / 8] & (1 << (idx % 8)) != 0
        let byte_idx = div(idx, Const::new(8u64));
        let bit_idx = rem(idx, Const::new(8u64));
        let byte = ptr_index(self.data_ptr, byte_idx);
        let mask = shl(Const::new(1u8), bit_idx.cast::<U8Type>());
        ne(and(byte, mask), Const::new(0u8))
    }

    /// Check validity (same as primitive)
    pub fn is_valid(&self, idx: Var<U64Type>) -> Var<BoolType> {
        if !self.has_validity {
            Const::new(true).into()
        } else {
            let byte_idx = div(idx, Const::new(8u64));
            let bit_idx = rem(idx, Const::new(8u64));
            let byte = ptr_index(self.validity_ptr, byte_idx);
            let mask = shl(Const::new(1u8), bit_idx.cast::<U8Type>());
            ne(and(byte, mask), Const::new(0u8))
        }
    }

    pub fn length(&self) -> Var<U64Type> {
        self.len
    }
}
```

### 1.4 StringView Staged Array

StringView uses a 16-byte (i128) view per element. Small strings (≤12 bytes) are inlined; larger strings reference external buffers.

```rust
// src/staged_arrays/string_view.rs

/// StringView encoding (16 bytes / i128):
///
/// For strings ≤ 12 bytes (inlined):
/// ┌──────────┬────────────────────────────┐
/// │ length   │  string data (padded)      │
/// │ (4 bytes)│  (12 bytes)                │
/// └──────────┴────────────────────────────┘
///   bits 0-31   bits 32-127
///
/// For strings > 12 bytes (external):
/// ┌──────────┬──────────┬───────────┬──────────┐
/// │ length   │ prefix   │ buf_index │  offset  │
/// │ (4 bytes)│ (4 bytes)│ (4 bytes) │ (4 bytes)│
/// └──────────┴──────────┴───────────┴──────────┘
///   bits 0-31  bits 32-63  bits 64-95  bits 96-127

/// A staged StringView array - wraps Arrow StringViewArray zero-copy
#[derive(Clone)]
pub struct StagedStringViewArray {
    /// Pointer to views buffer (array of i128)
    pub views_ptr: Var<SPtr<I128Type>>,
    /// Number of strings
    pub len: Var<U64Type>,
    /// Pointers to variadic data buffers (for non-inlined strings)
    /// At staging time, we know how many buffers there are
    pub data_buffers: Vec<Var<SPtr<U8Type>>>,
    /// Validity bitmap
    pub validity_ptr: Var<SPtr<U8Type>>,
    pub has_validity: bool,
}

/// A single StringView value (the i128 view)
#[derive(Clone)]
pub struct StagedStringView {
    pub view: Var<I128Type>,
}

impl StagedStringViewArray {
    /// Get the view (i128) at index
    pub fn get_view(&self, idx: Var<U64Type>) -> StagedStringView {
        let view = ptr_index(self.views_ptr, idx);
        StagedStringView { view }
    }

    /// Check validity
    pub fn is_valid(&self, idx: Var<U64Type>) -> Var<BoolType> {
        if !self.has_validity {
            Const::new(true).into()
        } else {
            let byte_idx = div(idx, Const::new(8u64));
            let bit_idx = rem(idx, Const::new(8u64));
            let byte = ptr_index(self.validity_ptr, byte_idx);
            let mask = shl(Const::new(1u8), bit_idx.cast::<U8Type>());
            ne(and(byte, mask), Const::new(0u8))
        }
    }

    pub fn length(&self) -> Var<U64Type> {
        self.len
    }
}

impl StagedStringView {
    /// Extract length from view (first 4 bytes)
    pub fn len(&self) -> Var<U32Type> {
        // length is in bits 0-31
        self.view.cast::<U32Type>()
    }

    /// Check if string is inlined (length <= 12)
    pub fn is_inlined(&self) -> Var<BoolType> {
        le(self.len(), Const::new(12u32))
    }

    /// Get pointer to string data and length
    /// For inlined: returns pointer to bytes 4-15 of the view
    /// For external: returns pointer into data buffer at offset
    ///
    /// This requires the data_buffers from the parent array
    pub fn get_data_ptr(
        &self,
        view_addr: Var<SPtr<I128Type>>,  // Address of this view in memory
        data_buffers: &[Var<SPtr<U8Type>>],
    ) -> (Var<SPtr<U8Type>>, Var<U32Type>) {
        let length = self.len();

        // We need if_then_else to handle inlined vs external
        // For now, return the components and let caller decide
        //
        // Inlined: data starts at view_addr + 4 bytes
        // External:
        //   buf_index = (view >> 64) & 0xFFFFFFFF
        //   offset = (view >> 96) & 0xFFFFFFFF
        //   data_ptr = data_buffers[buf_index] + offset

        // This is complex - see detailed implementation below
        todo!("Implement with if_then_else for inlined vs external")
    }

    /// For string comparisons, extract the 4-byte prefix (for fast rejection)
    /// Prefix is in bits 32-63 for external strings
    pub fn prefix(&self) -> Var<U32Type> {
        // (view >> 32) as u32
        shr(self.view, Const::new(32i128)).cast::<U32Type>()
    }
}
```

### 1.5 Unified StagedColumn Enum

```rust
// src/staged_arrays/column.rs

/// A staged column - union of all supported array types
pub enum StagedColumn {
    Int64(StagedPrimitiveArray<I64Type>),
    Float64(StagedPrimitiveArray<F64Type>),
    Boolean(StagedBooleanArray),
    StringView(StagedStringViewArray),
}

/// A staged scalar value extracted from a column
pub enum StagedValue {
    Int64(Var<I64Type>),
    Float64(Var<F64Type>),
    Boolean(Var<BoolType>),
    StringView(StagedStringView),
}

impl StagedColumn {
    /// Get value at index (returns appropriate StagedValue variant)
    pub fn get(&self, idx: Var<U64Type>) -> StagedValue {
        match self {
            StagedColumn::Int64(arr) => StagedValue::Int64(arr.get(idx)),
            StagedColumn::Float64(arr) => StagedValue::Float64(arr.get(idx)),
            StagedColumn::Boolean(arr) => StagedValue::Boolean(arr.get(idx)),
            StagedColumn::StringView(arr) => StagedValue::StringView(arr.get_view(idx)),
        }
    }

    /// Check validity at index
    pub fn is_valid(&self, idx: Var<U64Type>) -> Var<BoolType> {
        match self {
            StagedColumn::Int64(arr) => arr.is_valid(idx),
            StagedColumn::Float64(arr) => arr.is_valid(idx),
            StagedColumn::Boolean(arr) => arr.is_valid(idx),
            StagedColumn::StringView(arr) => arr.is_valid(idx),
        }
    }

    pub fn length(&self) -> Var<U64Type> {
        match self {
            StagedColumn::Int64(arr) => arr.length(),
            StagedColumn::Float64(arr) => arr.length(),
            StagedColumn::Boolean(arr) => arr.length(),
            StagedColumn::StringView(arr) => arr.length(),
        }
    }
}
```

### 1.6 Creating Staged Arrays from Arrow (Runtime Bridge)

```rust
// src/staged_arrays/from_arrow.rs

use arrow::array::*;
use arrow::buffer::Buffer;

/// Descriptor for passing Arrow array to JIT code
/// This is a plain Rust struct that holds the raw pointers
#[repr(C)]
pub struct ArrayDescriptor {
    pub data_ptr: *const u8,
    pub len: u64,
    pub validity_ptr: *const u8,  // null if no validity bitmap
}

/// For StringViewArray, we need additional buffer pointers
#[repr(C)]
pub struct StringViewDescriptor {
    pub views_ptr: *const u8,      // Pointer to i128 views
    pub len: u64,
    pub validity_ptr: *const u8,
    pub num_buffers: u32,
    pub buffer_ptrs: [*const u8; 8],  // Up to 8 data buffers (can extend)
}

/// Extract descriptor from Arrow Int64Array
pub fn int64_array_to_descriptor(arr: &Int64Array) -> ArrayDescriptor {
    ArrayDescriptor {
        data_ptr: arr.values().as_ptr() as *const u8,
        len: arr.len() as u64,
        validity_ptr: arr.nulls()
            .map(|n| n.buffer().as_ptr())
            .unwrap_or(std::ptr::null()),
    }
}

/// Extract descriptor from Arrow Float64Array
pub fn float64_array_to_descriptor(arr: &Float64Array) -> ArrayDescriptor {
    ArrayDescriptor {
        data_ptr: arr.values().as_ptr() as *const u8,
        len: arr.len() as u64,
        validity_ptr: arr.nulls()
            .map(|n| n.buffer().as_ptr())
            .unwrap_or(std::ptr::null()),
    }
}

/// Extract descriptor from Arrow BooleanArray
pub fn boolean_array_to_descriptor(arr: &BooleanArray) -> ArrayDescriptor {
    ArrayDescriptor {
        data_ptr: arr.values().as_ptr(),
        len: arr.len() as u64,
        validity_ptr: arr.nulls()
            .map(|n| n.buffer().as_ptr())
            .unwrap_or(std::ptr::null()),
    }
}

/// Extract descriptor from Arrow StringViewArray
pub fn string_view_array_to_descriptor(arr: &StringViewArray) -> StringViewDescriptor {
    let views = arr.views();
    let data_buffers = arr.data_buffers();

    let mut buffer_ptrs = [std::ptr::null(); 8];
    for (i, buf) in data_buffers.iter().enumerate().take(8) {
        buffer_ptrs[i] = buf.as_ptr();
    }

    StringViewDescriptor {
        views_ptr: views.as_ptr() as *const u8,
        len: arr.len() as u64,
        validity_ptr: arr.nulls()
            .map(|n| n.buffer().as_ptr())
            .unwrap_or(std::ptr::null()),
        num_buffers: data_buffers.len() as u32,
        buffer_ptrs,
    }
}
```

---

## Phase 2: Project Setup and Dependencies

### 2.1 Project Structure

```
dio-sql/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── staged_arrays/
│   │   ├── mod.rs
│   │   ├── primitive.rs
│   │   ├── boolean.rs
│   │   ├── string_view.rs
│   │   ├── column.rs
│   │   └── from_arrow.rs
│   ├── plan/
│   │   ├── mod.rs
│   │   ├── physical.rs         # Our physical plan
│   │   └── pipeline.rs         # Pipeline representation
│   ├── compile/
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   ├── operators.rs        # Staged operators
│   │   └── expr.rs             # Expression compilation
│   ├── runtime/
│   │   ├── mod.rs
│   │   ├── hash_table.rs       # Runtime hash table (Rust)
│   │   ├── extern_fns.rs       # External functions for JIT
│   │   └── orchestrator.rs     # Pipeline orchestration
│   └── catalog/
│       ├── mod.rs
│       └── schema_provider.rs  # For datafusion-sql
└── tests/
    ├── staged_arrays.rs
    └── integration.rs
```

### 2.2 Dependencies (Cargo.toml)

```toml
[package]
name = "dio-sql"
version = "0.1.0"
edition = "2021"

[dependencies]
# rust-lms for staged computation / JIT
rust-lms = { path = "../rust-lms" }

# SQL parsing and logical plan (NOT full datafusion!)
datafusion-sql = "44"
datafusion-expr = "44"
datafusion-common = "44"

# Arrow/Parquet for data
arrow = { version = "53", features = ["ffi"] }
parquet = { version = "53", features = ["arrow"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"

# Hash table (we might use hashbrown or std)
hashbrown = "0.15"
```

---

## Phase 3: SQL Parsing with datafusion-sql

We use `datafusion-sql` to parse SQL and get a `LogicalPlan`, then convert to our physical plan.

### 3.1 Schema Provider for datafusion-sql

```rust
// src/catalog/schema_provider.rs

use datafusion_sql::planner::ContextProvider;
use datafusion_sql::TableReference;
use datafusion_expr::{TableSource, Expr};
use datafusion_common::{DataFusionError, Result};
use arrow::datatypes::SchemaRef;
use std::sync::Arc;
use std::collections::HashMap;

/// Our simple catalog - maps table names to schemas and file paths
pub struct SimpleCatalog {
    tables: HashMap<String, TableInfo>,
}

pub struct TableInfo {
    pub schema: SchemaRef,
    pub file_path: String,
}

impl SimpleCatalog {
    pub fn new() -> Self {
        Self { tables: HashMap::new() }
    }

    /// Register a Parquet file as a table
    pub fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        let schema = read_parquet_schema(path)?;
        self.tables.insert(name.to_string(), TableInfo {
            schema: Arc::new(schema),
            file_path: path.to_string(),
        });
        Ok(())
    }
}

/// Implement ContextProvider for datafusion-sql
impl ContextProvider for SimpleCatalog {
    fn get_table_source(&self, name: TableReference) -> Result<Arc<dyn TableSource>> {
        let table_name = name.table();
        let info = self.tables.get(table_name)
            .ok_or_else(|| DataFusionError::Plan(format!("Table not found: {}", table_name)))?;

        Ok(Arc::new(SimpleTableSource {
            schema: info.schema.clone(),
            file_path: info.file_path.clone(),
        }))
    }

    // ... other required methods with default implementations
}

/// Simple table source that just holds schema and path
struct SimpleTableSource {
    schema: SchemaRef,
    file_path: String,
}

impl TableSource for SimpleTableSource {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// Read schema from Parquet file
fn read_parquet_schema(path: &str) -> Result<arrow::datatypes::Schema> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let file = File::open(path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    Ok(builder.schema().as_ref().clone())
}
```

### 3.2 Parsing SQL to LogicalPlan

```rust
// src/plan/mod.rs

use datafusion_sql::planner::SqlToRel;
use datafusion_sql::parser::DFParser;
use datafusion_expr::LogicalPlan;
use crate::catalog::SimpleCatalog;

pub fn parse_sql(sql: &str, catalog: &SimpleCatalog) -> Result<LogicalPlan> {
    // Parse SQL to AST
    let dialect = datafusion_sql::parser::DFParser::new_with_dialect(
        sql,
        &sqlparser::dialect::GenericDialect {}
    )?;
    let statements = dialect.parse_statements()?;

    if statements.len() != 1 {
        return Err(anyhow!("Expected exactly one statement"));
    }

    // Convert AST to LogicalPlan
    let planner = SqlToRel::new(catalog);
    let plan = planner.statement_to_plan(statements[0].clone())?;

    Ok(plan)
}
```

---

## Phase 4: Physical Plan and Pipelines

We convert datafusion's LogicalPlan to our own physical representation with explicit pipelines.

### 4.1 Pipeline Representation

```rust
// src/plan/pipeline.rs

use datafusion_expr::Expr as DFExpr;
use arrow::datatypes::SchemaRef;

/// A pipeline is a sequence of operators that can be fused into one JIT function
/// Pipelines are separated by "pipeline breakers" (hash table operations)
#[derive(Debug, Clone)]
pub struct Pipeline {
    pub id: usize,
    pub source: PipelineSource,
    pub operators: Vec<PipelineOp>,
    pub sink: PipelineSink,
    pub input_schema: SchemaRef,
    pub output_schema: SchemaRef,
}

/// Where a pipeline reads from
#[derive(Debug, Clone)]
pub enum PipelineSource {
    /// Scan a Parquet file
    TableScan {
        table_name: String,
        file_path: String,
        projection: Vec<usize>,  // Column indices to read
    },
    /// Probe a hash table (from previous pipeline)
    HashTableProbe {
        hash_table_id: usize,
        probe_keys: Vec<usize>,
    },
}

/// Operations within a pipeline (all fused)
#[derive(Debug, Clone)]
pub enum PipelineOp {
    /// Filter rows
    Filter { predicate: DFExpr },
    /// Project/compute columns
    Project { exprs: Vec<(DFExpr, String)> },
}

/// Where a pipeline writes to
#[derive(Debug, Clone)]
pub enum PipelineSink {
    /// Output results (terminal)
    Output,
    /// Build a hash table (for join build side)
    HashTableBuild {
        hash_table_id: usize,
        key_indices: Vec<usize>,
    },
    /// Insert into aggregate hash table
    HashAggregateBuild {
        hash_table_id: usize,
        group_by_indices: Vec<usize>,
        aggregates: Vec<AggregateExpr>,
    },
}

#[derive(Debug, Clone)]
pub struct AggregateExpr {
    pub func: AggregateFunc,
    pub input_idx: usize,
}

#[derive(Debug, Clone, Copy)]
pub enum AggregateFunc {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}
```

### 4.2 Converting LogicalPlan to Pipelines

```rust
// src/plan/physical.rs

use datafusion_expr::LogicalPlan;
use crate::plan::pipeline::*;

/// The complete physical plan - a DAG of pipelines
pub struct PhysicalPlan {
    pub pipelines: Vec<Pipeline>,
    /// Hash tables shared between pipelines
    pub hash_tables: Vec<HashTableDef>,
}

pub struct HashTableDef {
    pub id: usize,
    pub key_schema: SchemaRef,
    pub value_schema: SchemaRef,
}

/// Convert datafusion LogicalPlan to our pipelines
pub fn logical_to_physical(plan: &LogicalPlan) -> Result<PhysicalPlan> {
    let mut converter = PlanConverter::new();
    converter.convert(plan)?;
    Ok(PhysicalPlan {
        pipelines: converter.pipelines,
        hash_tables: converter.hash_tables,
    })
}

struct PlanConverter {
    pipelines: Vec<Pipeline>,
    hash_tables: Vec<HashTableDef>,
    next_pipeline_id: usize,
    next_ht_id: usize,
}

impl PlanConverter {
    fn new() -> Self {
        Self {
            pipelines: Vec::new(),
            hash_tables: Vec::new(),
            next_pipeline_id: 0,
            next_ht_id: 0,
        }
    }

    fn convert(&mut self, plan: &LogicalPlan) -> Result<()> {
        match plan {
            LogicalPlan::TableScan(scan) => {
                // Simple scan - one pipeline
                self.create_scan_pipeline(scan)
            }
            LogicalPlan::Filter(filter) => {
                // Add filter op to current pipeline
                self.convert(&filter.input)?;
                self.add_filter(&filter.predicate)
            }
            LogicalPlan::Projection(proj) => {
                // Add project op to current pipeline
                self.convert(&proj.input)?;
                self.add_projection(&proj.expr)
            }
            LogicalPlan::Join(join) => {
                // Pipeline breaker!
                // 1. Build side pipeline -> hash table
                // 2. Probe side pipeline reads from hash table
                self.handle_join(join)
            }
            LogicalPlan::Aggregate(agg) => {
                // Pipeline breaker!
                // 1. Input pipeline -> aggregate hash table
                // 2. New pipeline reads aggregate results
                self.handle_aggregate(agg)
            }
            _ => Err(anyhow!("Unsupported plan node: {:?}", plan))
        }
    }

    fn handle_join(&mut self, join: &datafusion_expr::Join) -> Result<()> {
        // Create hash table for build side
        let ht_id = self.next_ht_id;
        self.next_ht_id += 1;

        // Pipeline 1: Build side -> hash table
        self.convert(&join.left)?;
        let build_pipeline_idx = self.pipelines.len() - 1;
        self.pipelines[build_pipeline_idx].sink = PipelineSink::HashTableBuild {
            hash_table_id: ht_id,
            key_indices: extract_join_keys(&join.on, true),
        };

        // Pipeline 2: Probe side, reading from hash table
        self.convert(&join.right)?;
        let probe_pipeline_idx = self.pipelines.len() - 1;

        // Insert hash table probe at the start of probe pipeline ops
        // (After source, before other ops)
        let probe_source = PipelineSource::HashTableProbe {
            hash_table_id: ht_id,
            probe_keys: extract_join_keys(&join.on, false),
        };

        // Merge the probe with existing pipeline...
        // (Complex logic to restructure pipeline)

        Ok(())
    }

    // ... other conversion methods
}
```

---

## Phase 5: Runtime Hash Tables (Rust, not JIT)

Hash tables are implemented in Rust and called via external functions.

### 5.1 Runtime Hash Table

```rust
// src/runtime/hash_table.rs

use hashbrown::raw::RawTable;
use std::alloc::{alloc, dealloc, Layout};

/// A runtime hash table for join operations
/// Key and value are stored as raw bytes for flexibility
pub struct RuntimeHashTable {
    /// The underlying hash table
    table: RawTable<Entry>,
    /// Schema info for key/value interpretation
    key_size: usize,
    value_size: usize,
}

struct Entry {
    hash: u64,
    key_value: Box<[u8]>,  // key bytes followed by value bytes
}

impl RuntimeHashTable {
    pub fn new(key_size: usize, value_size: usize) -> Self {
        Self {
            table: RawTable::new(),
            key_size,
            value_size,
        }
    }

    /// Insert a key-value pair
    /// Called from JIT code via external function
    pub fn insert(&mut self, hash: u64, key: &[u8], value: &[u8]) {
        debug_assert_eq!(key.len(), self.key_size);
        debug_assert_eq!(value.len(), self.value_size);

        let mut kv = vec![0u8; self.key_size + self.value_size].into_boxed_slice();
        kv[..self.key_size].copy_from_slice(key);
        kv[self.key_size..].copy_from_slice(value);

        let entry = Entry { hash, key_value: kv };

        self.table.insert(hash, entry, |e| e.hash);
    }

    /// Probe for a key, returns iterator over matching values
    pub fn probe<'a>(&'a self, hash: u64, key: &'a [u8]) -> impl Iterator<Item = &'a [u8]> + 'a {
        self.table
            .iter()
            .filter(move |entry| {
                entry.hash == hash && &entry.key_value[..self.key_size] == key
            })
            .map(move |entry| &entry.key_value[self.key_size..])
    }
}

/// Handle for hash table, passed to JIT code
pub type HashTableHandle = *mut RuntimeHashTable;
```

### 5.2 External Functions for JIT

```rust
// src/runtime/extern_fns.rs

use rust_lms::extern_fn;
use crate::runtime::hash_table::{RuntimeHashTable, HashTableHandle};

/// Create a new hash table
/// Returns handle (pointer) to the hash table
#[extern_fn]
#[no_mangle]
pub extern "C" fn ht_create(key_size: u64, value_size: u64) -> HashTableHandle {
    let ht = Box::new(RuntimeHashTable::new(key_size as usize, value_size as usize));
    Box::into_raw(ht)
}

/// Insert into hash table
/// key_ptr and value_ptr point to the key/value bytes
#[extern_fn]
#[no_mangle]
pub extern "C" fn ht_insert(
    ht: HashTableHandle,
    hash: u64,
    key_ptr: *const u8,
    key_size: u64,
    value_ptr: *const u8,
    value_size: u64,
) {
    let ht = unsafe { &mut *ht };
    let key = unsafe { std::slice::from_raw_parts(key_ptr, key_size as usize) };
    let value = unsafe { std::slice::from_raw_parts(value_ptr, value_size as usize) };
    ht.insert(hash, key, value);
}

/// Probe hash table - returns iterator handle
/// The iterator is used with ht_probe_next
#[extern_fn]
#[no_mangle]
pub extern "C" fn ht_probe_start(
    ht: HashTableHandle,
    hash: u64,
    key_ptr: *const u8,
    key_size: u64,
) -> ProbeIterHandle {
    // Create iterator state and return handle
    // ... implementation details
    todo!()
}

/// Get next match from probe iterator
/// Returns 0 if no more matches, 1 if match found (value written to out_ptr)
#[extern_fn]
#[no_mangle]
pub extern "C" fn ht_probe_next(
    iter: ProbeIterHandle,
    out_value_ptr: *mut u8,
) -> u64 {
    // Get next value from iterator
    // ... implementation details
    todo!()
}

/// Destroy hash table
#[extern_fn]
#[no_mangle]
pub extern "C" fn ht_destroy(ht: HashTableHandle) {
    unsafe {
        drop(Box::from_raw(ht));
    }
}

pub type ProbeIterHandle = *mut ProbeIterator;

pub struct ProbeIterator {
    // Internal state for iteration
    // ...
}
```

---

## Phase 6: Pipeline Compilation

Each pipeline compiles to a single JIT function.

### 6.1 Compilation Context

```rust
// src/compile/context.rs

use rust_lms::prelude::*;
use crate::staged_arrays::*;
use crate::runtime::extern_fns::*;

/// Context for compiling a single pipeline
pub struct PipelineCompileContext<'a> {
    pub compiler: &'a mut Compiler,
    /// Registered external functions
    pub ht_create: ExternRef<HtCreateExtern>,
    pub ht_insert: ExternRef<HtInsertExtern>,
    pub ht_probe_start: ExternRef<HtProbeStartExtern>,
    pub ht_probe_next: ExternRef<HtProbeNextExtern>,
    pub ht_destroy: ExternRef<HtDestroyExtern>,
}

impl<'a> PipelineCompileContext<'a> {
    pub fn new(compiler: &'a mut Compiler) -> Self {
        Self {
            ht_create: compiler.extern_fn::<HtCreateExtern>(),
            ht_insert: compiler.extern_fn::<HtInsertExtern>(),
            ht_probe_start: compiler.extern_fn::<HtProbeStartExtern>(),
            ht_probe_next: compiler.extern_fn::<HtProbeNextExtern>(),
            ht_destroy: compiler.extern_fn::<HtDestroyExtern>(),
            compiler,
        }
    }
}
```

### 6.2 Compiling a Pipeline

```rust
// src/compile/operators.rs

use rust_lms::prelude::*;
use crate::plan::pipeline::*;
use crate::staged_arrays::*;
use crate::compile::context::*;
use crate::compile::expr::compile_expr;

/// Compile a pipeline to a JIT function
///
/// The generated function signature depends on the pipeline:
/// - Scan pipeline: fn(columns: &[ArrayDescriptor], output: &mut OutputBuffer)
/// - Probe pipeline: fn(columns: &[ArrayDescriptor], ht: HashTableHandle, output: &mut OutputBuffer)
pub fn compile_pipeline(
    ctx: &mut PipelineCompileContext,
    pipeline: &Pipeline,
) -> CompiledFunction {
    // For a scan + filter + project + hash_build pipeline:
    //
    // fn pipeline(col0_ptr, col0_len, col0_validity, col1_ptr, ..., ht_handle) {
    //     for i in 0..col0_len {
    //         // Load values from columns
    //         let val0 = col0_ptr[i];
    //         let val1 = col1_ptr[i];
    //         ...
    //
    //         // Apply filter
    //         if predicate(val0, val1, ...) {
    //             // Apply projection
    //             let out0 = expr0(val0, val1, ...);
    //             let out1 = expr1(val0, val1, ...);
    //
    //             // Sink: insert into hash table
    //             let key = serialize_key(out0);
    //             let value = serialize_value(out0, out1, ...);
    //             let hash = compute_hash(key);
    //             ht_insert(ht_handle, hash, key_ptr, key_size, value_ptr, value_size);
    //         }
    //     }
    // }

    let schema = &pipeline.input_schema;
    let num_columns = schema.fields().len();

    // Build function with column parameters
    // Each column needs: data_ptr, len, validity_ptr
    // Plus any hash table handles

    ctx.compiler.fun_n("pipeline", num_columns * 3 + 1, |vctx, params| {
        // Parse parameters into StagedColumns
        let columns: Vec<StagedColumn> = build_staged_columns(schema, params);

        // Get length from first column
        let len = columns[0].length();

        // Loop variable
        let i = vctx.let_var(0u64);

        // Main loop
        while_loop(
            lt(*i, len),
            {
                // Load current row values
                let row_values: Vec<StagedValue> = columns.iter()
                    .map(|col| col.get(*i))
                    .collect();

                // Apply operators (filter, project, etc.)
                let body = compile_operators(
                    vctx,
                    ctx,
                    &pipeline.operators,
                    &row_values,
                    &pipeline.sink,
                );

                // Increment and continue
                seq(body, assign(i, add(*i, Const::new(1u64))))
            }
        )
    })
}

/// Compile the operator chain for one row
fn compile_operators(
    vctx: &mut VarContext,
    ctx: &PipelineCompileContext,
    operators: &[PipelineOp],
    values: &[StagedValue],
    sink: &PipelineSink,
) -> impl Staged<Out = UnitType> {
    // Start with sink as innermost callback
    let mut callback: Box<dyn FnOnce(&[StagedValue]) -> _> = Box::new(|vals| {
        compile_sink(vctx, ctx, sink, vals)
    });

    // Wrap with operators from inside out (reverse order)
    for op in operators.iter().rev() {
        match op {
            PipelineOp::Filter { predicate } => {
                let pred = predicate.clone();
                let inner = callback;
                callback = Box::new(move |vals| {
                    let cond = compile_expr(&pred, vals);
                    if_then(cond, inner(vals))
                });
            }
            PipelineOp::Project { exprs } => {
                let exprs = exprs.clone();
                let inner = callback;
                callback = Box::new(move |vals| {
                    let projected: Vec<StagedValue> = exprs.iter()
                        .map(|(e, _)| compile_expr(e, vals))
                        .collect();
                    inner(&projected)
                });
            }
        }
    }

    callback(values)
}

/// Compile the sink (output or hash table insert)
fn compile_sink(
    vctx: &mut VarContext,
    ctx: &PipelineCompileContext,
    sink: &PipelineSink,
    values: &[StagedValue],
) -> impl Staged<Out = UnitType> {
    match sink {
        PipelineSink::Output => {
            // Write to output buffer
            // ... output buffer writing logic
            unit()
        }
        PipelineSink::HashTableBuild { hash_table_id, key_indices } => {
            // Serialize key and value
            // Call ht_insert external function
            let ht_handle = vctx.get_param::<HashTableHandle>("ht_handle");

            // Compute hash of key columns
            let hash = compute_hash(values, key_indices);

            // Serialize key bytes
            let (key_ptr, key_size) = serialize_to_bytes(values, key_indices);

            // Serialize all values
            let (value_ptr, value_size) = serialize_all_to_bytes(values);

            // Call external insert function
            call_extern6(
                ctx.ht_insert,
                ht_handle,
                hash,
                key_ptr,
                key_size,
                value_ptr,
                value_size,
            )
        }
        PipelineSink::HashAggregateBuild { .. } => {
            // Similar but with aggregate update logic
            todo!()
        }
    }
}
```

### 6.3 Expression Compilation

```rust
// src/compile/expr.rs

use datafusion_expr::Expr as DFExpr;
use datafusion_expr::Operator as DFOp;
use rust_lms::prelude::*;
use crate::staged_arrays::StagedValue;

/// Compile a datafusion expression to staged code
pub fn compile_expr(expr: &DFExpr, row: &[StagedValue]) -> StagedValue {
    match expr {
        DFExpr::Column(col) => {
            // Column reference by index
            let idx = col.index.unwrap_or(0);  // Should have index from planning
            row[idx].clone()
        }

        DFExpr::Literal(lit) => {
            use datafusion_expr::ScalarValue;
            match lit {
                ScalarValue::Int64(Some(v)) => StagedValue::Int64(Const::new(*v).into()),
                ScalarValue::Float64(Some(v)) => StagedValue::Float64(Const::new(*v).into()),
                ScalarValue::Boolean(Some(v)) => StagedValue::Boolean(Const::new(*v).into()),
                // ... other literals
                _ => todo!("Unsupported literal: {:?}", lit),
            }
        }

        DFExpr::BinaryExpr(binary) => {
            let left = compile_expr(&binary.left, row);
            let right = compile_expr(&binary.right, row);

            compile_binary_op(&left, &binary.op, &right)
        }

        DFExpr::Not(inner) => {
            let val = compile_expr(inner, row);
            match val {
                StagedValue::Boolean(b) => StagedValue::Boolean(not(b)),
                _ => panic!("NOT requires boolean"),
            }
        }

        // ... other expression types
        _ => todo!("Unsupported expression: {:?}", expr),
    }
}

fn compile_binary_op(left: &StagedValue, op: &DFOp, right: &StagedValue) -> StagedValue {
    match (left, right, op) {
        // Integer arithmetic
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::Plus) => {
            StagedValue::Int64(add(*l, *r))
        }
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::Minus) => {
            StagedValue::Int64(sub(*l, *r))
        }
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::Multiply) => {
            StagedValue::Int64(mul(*l, *r))
        }
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::Divide) => {
            StagedValue::Int64(div(*l, *r))
        }

        // Integer comparisons
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::Lt) => {
            StagedValue::Boolean(lt(*l, *r))
        }
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::LtEq) => {
            StagedValue::Boolean(le(*l, *r))
        }
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::Gt) => {
            StagedValue::Boolean(gt(*l, *r))
        }
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::GtEq) => {
            StagedValue::Boolean(ge(*l, *r))
        }
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::Eq) => {
            StagedValue::Boolean(eq(*l, *r))
        }
        (StagedValue::Int64(l), StagedValue::Int64(r), DFOp::NotEq) => {
            StagedValue::Boolean(ne(*l, *r))
        }

        // Float arithmetic (similar pattern)
        (StagedValue::Float64(l), StagedValue::Float64(r), DFOp::Plus) => {
            StagedValue::Float64(add(*l, *r))
        }
        // ... more float ops

        // Boolean logic
        (StagedValue::Boolean(l), StagedValue::Boolean(r), DFOp::And) => {
            StagedValue::Boolean(and(*l, *r))
        }
        (StagedValue::Boolean(l), StagedValue::Boolean(r), DFOp::Or) => {
            StagedValue::Boolean(or(*l, *r))
        }

        _ => todo!("Unsupported binary op: {:?} {:?} {:?}", left, op, right),
    }
}
```

---

## Phase 7: Pipeline Orchestrator

Rust code orchestrates execution of multiple pipelines.

### 7.1 Orchestrator

```rust
// src/runtime/orchestrator.rs

use arrow::record_batch::RecordBatch;
use crate::plan::physical::PhysicalPlan;
use crate::plan::pipeline::*;
use crate::compile::compile_pipeline;
use crate::staged_arrays::from_arrow::*;
use crate::runtime::hash_table::RuntimeHashTable;

/// Executes a physical plan by orchestrating pipelines
pub struct PipelineOrchestrator {
    /// Compiled pipeline functions
    compiled_pipelines: Vec<CompiledPipeline>,
    /// Runtime hash tables
    hash_tables: Vec<RuntimeHashTable>,
}

struct CompiledPipeline {
    /// The JIT-compiled function
    func: CompiledFunction,
    /// Pipeline metadata
    pipeline: Pipeline,
}

impl PipelineOrchestrator {
    pub fn new(plan: &PhysicalPlan) -> Result<Self> {
        let mut compiler = Compiler::new();
        let mut ctx = PipelineCompileContext::new(&mut compiler);

        // Compile all pipelines
        let compiled_pipelines: Vec<_> = plan.pipelines.iter()
            .map(|p| {
                let func = compile_pipeline(&mut ctx, p);
                CompiledPipeline { func, pipeline: p.clone() }
            })
            .collect();

        // Create hash tables
        let hash_tables: Vec<_> = plan.hash_tables.iter()
            .map(|def| RuntimeHashTable::new(
                compute_key_size(&def.key_schema),
                compute_value_size(&def.value_schema),
            ))
            .collect();

        Ok(Self { compiled_pipelines, hash_tables })
    }

    /// Execute the plan on input data
    pub fn execute(&mut self, tables: &HashMap<String, Vec<RecordBatch>>) -> Result<Vec<RecordBatch>> {
        let mut output = OutputBuffer::new();

        // Execute pipelines in order
        // (Topologically sorted by dependencies)
        for compiled in &self.compiled_pipelines {
            self.execute_pipeline(compiled, tables, &mut output)?;
        }

        output.to_record_batches()
    }

    fn execute_pipeline(
        &mut self,
        compiled: &CompiledPipeline,
        tables: &HashMap<String, Vec<RecordBatch>>,
        output: &mut OutputBuffer,
    ) -> Result<()> {
        let pipeline = &compiled.pipeline;

        match &pipeline.source {
            PipelineSource::TableScan { table_name, file_path, projection } => {
                // Get RecordBatches for this table
                let batches = tables.get(table_name)
                    .ok_or_else(|| anyhow!("Table not found: {}", table_name))?;

                // Process each batch
                for batch in batches {
                    // Extract array descriptors (zero-copy pointers)
                    let descriptors: Vec<ArrayDescriptor> = projection.iter()
                        .map(|&idx| array_to_descriptor(batch.column(idx)))
                        .collect();

                    // Get hash table handle if needed
                    let ht_handle = match &pipeline.sink {
                        PipelineSink::HashTableBuild { hash_table_id, .. } => {
                            &mut self.hash_tables[*hash_table_id] as *mut _
                        }
                        _ => std::ptr::null_mut(),
                    };

                    // Call compiled function
                    // The function signature varies - use appropriate calling convention
                    unsafe {
                        compiled.func.call(&descriptors, ht_handle, output);
                    }
                }
            }

            PipelineSource::HashTableProbe { hash_table_id, probe_keys } => {
                // Probe pipeline - reads from hash table
                // Process input batches while probing
                // ... similar pattern with hash table as input
                todo!()
            }
        }

        Ok(())
    }
}

/// Buffer for collecting output rows
pub struct OutputBuffer {
    // Column builders for output
    // ...
}

impl OutputBuffer {
    pub fn new() -> Self {
        todo!()
    }

    pub fn to_record_batches(self) -> Result<Vec<RecordBatch>> {
        todo!()
    }
}
```

---

## Phase 8: Integration and Testing

### 8.1 Main API

```rust
// src/lib.rs

pub mod staged_arrays;
pub mod plan;
pub mod compile;
pub mod runtime;
pub mod catalog;

use arrow::record_batch::RecordBatch;
use crate::catalog::SimpleCatalog;
use crate::plan::{parse_sql, logical_to_physical};
use crate::runtime::orchestrator::PipelineOrchestrator;

/// Main entry point for SQL query execution
pub struct QueryEngine {
    catalog: SimpleCatalog,
}

impl QueryEngine {
    pub fn new() -> Self {
        Self {
            catalog: SimpleCatalog::new(),
        }
    }

    /// Register a Parquet file as a table
    pub fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        self.catalog.register_parquet(name, path)
    }

    /// Execute a SQL query
    pub fn execute(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        // 1. Parse SQL to LogicalPlan (using datafusion-sql)
        let logical_plan = parse_sql(sql, &self.catalog)?;

        // 2. Convert to physical plan with pipelines
        let physical_plan = logical_to_physical(&logical_plan)?;

        // 3. Create orchestrator (compiles pipelines)
        let mut orchestrator = PipelineOrchestrator::new(&physical_plan)?;

        // 4. Load data and execute
        let tables = self.load_tables(&physical_plan)?;
        let results = orchestrator.execute(&tables)?;

        Ok(results)
    }

    fn load_tables(&self, plan: &PhysicalPlan) -> Result<HashMap<String, Vec<RecordBatch>>> {
        // Load Parquet files into RecordBatches
        // (The actual data stays in Arrow buffers - zero copy!)
        todo!()
    }
}
```

### 8.2 Example Usage

```rust
// examples/simple_query.rs

use dio_sql::QueryEngine;

fn main() -> Result<()> {
    let mut engine = QueryEngine::new();

    // Register tables
    engine.register_parquet("orders", "data/orders.parquet")?;
    engine.register_parquet("customers", "data/customers.parquet")?;

    // Simple filter + project (single pipeline, fully JIT)
    let results = engine.execute(
        "SELECT order_id, amount * 1.1 as with_tax
         FROM orders
         WHERE amount > 100"
    )?;

    // Join query (two pipelines: build + probe)
    let results = engine.execute(
        "SELECT c.name, o.amount
         FROM orders o
         JOIN customers c ON o.customer_id = c.id
         WHERE o.amount > 100"
    )?;

    // Aggregate query (two pipelines: aggregate + output)
    let results = engine.execute(
        "SELECT customer_id, SUM(amount) as total
         FROM orders
         GROUP BY customer_id"
    )?;

    for batch in &results {
        println!("{:?}", batch);
    }

    Ok(())
}
```

### 8.3 Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::*;

    #[test]
    fn test_staged_primitive_array() {
        // Test that StagedPrimitiveArray correctly wraps Arrow arrays
        let arr = Int64Array::from(vec![1, 2, 3, 4, 5]);
        let desc = int64_array_to_descriptor(&arr);

        // Create staged array and compile a simple loop
        let mut compiler = Compiler::new();
        let sum_fn = compiler.fun3("sum", |vctx, data_ptr, len, _validity| {
            let staged_arr = StagedPrimitiveArray::<I64Type>::from_arrow_ptrs(
                data_ptr, len, Const::new(std::ptr::null()).into(), false
            );

            let sum = vctx.let_var(0i64);
            let i = vctx.let_var(0u64);

            while_loop(
                lt(*i, staged_arr.length()),
                seq(
                    assign(sum, add(*sum, staged_arr.get(*i))),
                    assign(i, add(*i, Const::new(1u64)))
                )
            );

            *sum
        });

        let compiled = compiler.compile(sum_fn).unwrap();
        let f = compiled.as_fn();

        let result = f(desc.data_ptr as *const i64, desc.len, desc.validity_ptr);
        assert_eq!(result, 15);  // 1+2+3+4+5
    }

    #[test]
    fn test_filter_pipeline() {
        // Test a simple scan + filter pipeline
        // ...
    }

    #[test]
    fn test_hash_join() {
        // Test hash join with two pipelines
        // ...
    }
}
```

---

## Implementation Order

1. **Phase 1: Staged Arrays** (Foundation)
   - Implement `StagedPrimitiveArray<I64Type>` and `<F64Type>`
   - Implement `StagedBooleanArray`
   - Implement `StagedStringViewArray`
   - Write tests for each array type
   - Implement `from_arrow` descriptors

2. **Phase 2: Project Setup**
   - Create project structure
   - Set up Cargo.toml with dependencies
   - Verify rust-lms integration works

3. **Phase 3: Basic Pipeline Compilation**
   - Implement simple scan pipeline (no filter/project)
   - Test end-to-end: RecordBatch → JIT code → loop over values

4. **Phase 4: Operators**
   - Add filter operator
   - Add project operator
   - Test fused scan + filter + project

5. **Phase 5: SQL Integration**
   - Implement SimpleCatalog
   - Integrate datafusion-sql parsing
   - Convert LogicalPlan to physical Pipeline

6. **Phase 6: Hash Tables**
   - Implement RuntimeHashTable
   - Implement external functions
   - Add hash table sink to pipelines

7. **Phase 7: Join Support**
   - Implement build pipeline
   - Implement probe pipeline
   - Test hash join end-to-end

8. **Phase 8: Aggregates**
   - Implement aggregate hash table
   - Add aggregate functions
   - Test GROUP BY queries

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Data Access | Zero-copy via staged array wrappers | No copying = maximum performance |
| SQL Parsing | datafusion-sql + datafusion-expr | Mature SQL parser, avoid reinventing |
| Hash Tables | Runtime Rust (not JIT) | Complex dynamic structures; external function calls |
| Compilation Unit | One JIT function per pipeline | Clear boundaries at pipeline breakers |
| String Encoding | StringView (i128) | Modern Arrow format, efficient comparisons |
| Null Handling | Validity bitmaps | Standard Arrow approach |

---

## Open Questions

1. **Output buffer format**: Should output be written to pre-allocated Arrow arrays, or to a dynamic buffer that's converted to Arrow after?

2. **String comparison in JIT**: For filter predicates on strings, should we:
   - Call external Rust function for comparison
   - Generate inline comparison code (complex for variable-length)
   - Use prefix comparison for fast rejection, then external call

3. **Memory management for hash tables**: Who owns the memory?
   - Hash tables live in Rust heap
   - JIT code only holds handles (pointers)
   - Orchestrator manages lifecycle

4. **Error handling in JIT code**: How to handle runtime errors (null dereference, division by zero)?
   - Trap and return error code
   - Check before operation (overhead)
   - Ignore (undefined behavior)
