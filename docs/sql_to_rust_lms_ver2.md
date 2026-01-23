# SQL Query Execution Engine using rust-lms

## Overview

This document provides a concrete implementation plan for building a SQL query execution engine that:
- Reads Parquet files using the `parquet` and `arrow` crates
- Parses SQL queries using `sqlparser`
- Compiles query execution pipelines using `rust-lms` (Cranelift JIT)
- Fuses operators until pipeline breakers (join/group by)
- Processes data in chunks (Arrow RecordBatches)

The approach follows the **First Futamura Projection**: we write a query *interpreter* using rust-lms staged types, and through specialization, it becomes a query *compiler* that generates optimized native code.

---

## Architecture

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  SQL Query  │────▶│   Parser    │────▶│  Logical    │────▶│  Physical   │
│   String    │     │ (sqlparser) │     │    Plan     │     │    Plan     │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
                                                                   │
                                                                   ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Result    │◀────│   Execute   │◀────│  Compiled   │◀────│  rust-lms   │
│  (Arrow)    │     │   (JIT)     │     │   Code      │     │  Compiler   │
└─────────────┘     └─────────────┘     └─────────────┘     └─────────────┘
```

---

## Phase 1: Project Setup and Dependencies

### 1.1 Create Project Structure

```
dio-sql/
├── Cargo.toml
├── src/
│   ├── lib.rs
│   ├── parser/
│   │   ├── mod.rs
│   │   └── sql_to_plan.rs
│   ├── plan/
│   │   ├── mod.rs
│   │   ├── logical.rs
│   │   └── physical.rs
│   ├── types/
│   │   ├── mod.rs
│   │   ├── schema.rs
│   │   └── value.rs
│   ├── operators/
│   │   ├── mod.rs
│   │   ├── scan.rs
│   │   ├── filter.rs
│   │   ├── project.rs
│   │   ├── hash_join.rs
│   │   └── aggregate.rs
│   ├── compile/
│   │   ├── mod.rs
│   │   ├── context.rs
│   │   └── codegen.rs
│   └── execution/
│       ├── mod.rs
│       └── runtime.rs
└── examples/
    └── simple_query.rs
```

### 1.2 Dependencies (Cargo.toml)

```toml
[package]
name = "dio-sql"
version = "0.1.0"
edition = "2021"

[dependencies]
# rust-lms for staged computation
rust-lms = { path = "../rust-lms" }

# SQL parsing
sqlparser = "0.40"

# Arrow/Parquet for data
arrow = { version = "50", features = ["ffi"] }
parquet = { version = "50", features = ["arrow"] }

# Error handling
thiserror = "1.0"
anyhow = "1.0"
```

---

## Phase 2: Type System and Schema

### 2.1 SQL Type Representation

Create types that bridge SQL types to rust-lms staged types:

```rust
// src/types/schema.rs

use rust_lms::prelude::*;

/// SQL column types we support
#[derive(Debug, Clone, PartialEq)]
pub enum SqlType {
    Int64,
    Float64,
    Utf8,      // String
    Boolean,
    Date32,    // Days since epoch
}

/// A column in a schema
#[derive(Debug, Clone)]
pub struct Column {
    pub name: String,
    pub data_type: SqlType,
    pub nullable: bool,
}

/// Table schema - known at compile time (static)
#[derive(Debug, Clone)]
pub struct Schema {
    pub columns: Vec<Column>,
}

impl Schema {
    pub fn field_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|c| c.name == name)
    }

    pub fn field_type(&self, name: &str) -> Option<&SqlType> {
        self.columns.iter()
            .find(|c| c.name == name)
            .map(|c| &c.data_type)
    }
}
```

### 2.2 Mixed-Stage Record Type

The key insight from the papers: **schema is static (compile-time), field values are dynamic (runtime)**.

```rust
// src/types/value.rs

use rust_lms::prelude::*;

/// A staged value - the actual data is dynamic (Rep[T] / Var<T>)
/// but the type is known statically
pub enum StagedValue {
    Int64(Var<I64Type>),
    Float64(Var<F64Type>),
    // For strings, we store pointer + length
    Utf8 { ptr: Var<SPtr<U8Type>>, len: Var<U64Type> },
    Boolean(Var<BoolType>),
}

/// A record with static schema but dynamic field values
/// This is the "mixed-stage data structure" from the paper
pub struct StagedRecord {
    /// Schema is known at staging time (compile time)
    pub schema: Schema,
    /// Field values are Var<T> - they become variables in generated code
    pub fields: Vec<StagedValue>,
}

impl StagedRecord {
    /// Field lookup - index computed at staging time, value is dynamic
    pub fn get(&self, field_name: &str) -> Option<&StagedValue> {
        let idx = self.schema.field_index(field_name)?;
        self.fields.get(idx)
    }

    /// Get field by index (faster, no string lookup at staging time)
    pub fn get_idx(&self, idx: usize) -> Option<&StagedValue> {
        self.fields.get(idx)
    }
}
```

---

## Phase 3: Query Plan Representation

### 3.1 Logical Plan (SQL-level)

```rust
// src/plan/logical.rs

/// Expression in a query
#[derive(Debug, Clone)]
pub enum Expr {
    /// Column reference: "column_name"
    Column(String),
    /// Literal value
    Literal(LiteralValue),
    /// Binary operation: expr op expr
    BinaryOp {
        left: Box<Expr>,
        op: BinaryOperator,
        right: Box<Expr>,
    },
    /// Function call: func(args...)
    Function {
        name: String,
        args: Vec<Expr>,
    },
    /// Aggregate function: SUM, COUNT, etc.
    Aggregate {
        func: AggregateFunc,
        arg: Box<Expr>,
    },
}

#[derive(Debug, Clone)]
pub enum LiteralValue {
    Int64(i64),
    Float64(f64),
    Utf8(String),
    Boolean(bool),
    Null,
}

#[derive(Debug, Clone)]
pub enum BinaryOperator {
    Eq, Ne, Lt, Le, Gt, Ge,
    Add, Sub, Mul, Div,
    And, Or,
}

#[derive(Debug, Clone)]
pub enum AggregateFunc {
    Count,
    Sum,
    Avg,
    Min,
    Max,
}

/// Logical query plan - tree of relational operators
#[derive(Debug, Clone)]
pub enum LogicalPlan {
    /// Scan a table from a Parquet file
    Scan {
        table_name: String,
        file_path: String,
        schema: Schema,
        projection: Option<Vec<String>>, // columns to read
    },
    /// Filter rows: WHERE clause
    Filter {
        predicate: Expr,
        input: Box<LogicalPlan>,
    },
    /// Project columns: SELECT clause
    Project {
        exprs: Vec<(Expr, String)>, // (expression, alias)
        input: Box<LogicalPlan>,
    },
    /// Hash join two inputs
    HashJoin {
        left: Box<LogicalPlan>,
        right: Box<LogicalPlan>,
        left_keys: Vec<String>,
        right_keys: Vec<String>,
        join_type: JoinType,
    },
    /// Aggregate with grouping
    Aggregate {
        input: Box<LogicalPlan>,
        group_by: Vec<Expr>,
        aggregates: Vec<(AggregateFunc, Expr, String)>, // (func, arg, alias)
    },
    /// Sort results
    Sort {
        input: Box<LogicalPlan>,
        order_by: Vec<(Expr, bool)>, // (expr, ascending)
    },
    /// Limit results
    Limit {
        input: Box<LogicalPlan>,
        limit: usize,
    },
}

#[derive(Debug, Clone)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}
```

### 3.2 Physical Plan (Execution-level)

The physical plan is closer to execution - it specifies HOW to execute:

```rust
// src/plan/physical.rs

/// Physical operators - these map to actual code generation
#[derive(Debug, Clone)]
pub enum PhysicalPlan {
    /// Read chunks from Parquet file
    ParquetScan {
        file_path: String,
        schema: Schema,
        projection: Vec<usize>, // column indices to read
    },
    /// Fused pipeline: filter + project operations
    /// These are fused into a single loop
    Pipeline {
        input: Box<PhysicalPlan>,
        operations: Vec<PipelineOp>,
        output_schema: Schema,
    },
    /// Hash join - pipeline breaker
    HashJoin {
        build_side: Box<PhysicalPlan>,  // smaller table
        probe_side: Box<PhysicalPlan>,  // larger table
        build_keys: Vec<usize>,
        probe_keys: Vec<usize>,
        output_schema: Schema,
    },
    /// Hash aggregate - pipeline breaker
    HashAggregate {
        input: Box<PhysicalPlan>,
        group_by_indices: Vec<usize>,
        aggregates: Vec<PhysicalAggregate>,
        output_schema: Schema,
    },
}

/// Operations that can be fused into a pipeline
#[derive(Debug, Clone)]
pub enum PipelineOp {
    Filter(CompiledExpr),
    Project(Vec<CompiledExpr>),
}

/// Compiled expression ready for code generation
#[derive(Debug, Clone)]
pub enum CompiledExpr {
    ColumnRef(usize),  // index into input record
    Literal(LiteralValue),
    BinaryOp {
        left: Box<CompiledExpr>,
        op: BinaryOperator,
        right: Box<CompiledExpr>,
    },
}

#[derive(Debug, Clone)]
pub struct PhysicalAggregate {
    pub func: AggregateFunc,
    pub input_idx: usize,  // column index for input
    pub output_type: SqlType,
}
```

---

## Phase 4: SQL Parser Integration

### 4.1 Parse SQL to Logical Plan

```rust
// src/parser/sql_to_plan.rs

use sqlparser::ast::*;
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

pub struct QueryPlanner {
    /// Known table schemas (loaded from Parquet metadata)
    pub table_schemas: HashMap<String, (String, Schema)>, // name -> (path, schema)
}

impl QueryPlanner {
    pub fn new() -> Self {
        Self {
            table_schemas: HashMap::new(),
        }
    }

    /// Register a Parquet file as a table
    pub fn register_parquet(&mut self, name: &str, path: &str) -> Result<()> {
        let schema = Self::read_parquet_schema(path)?;
        self.table_schemas.insert(
            name.to_string(),
            (path.to_string(), schema)
        );
        Ok(())
    }

    /// Read schema from Parquet file metadata
    fn read_parquet_schema(path: &str) -> Result<Schema> {
        use parquet::file::reader::{FileReader, SerializedFileReader};
        use std::fs::File;

        let file = File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        let parquet_schema = reader.metadata().file_metadata().schema();

        // Convert Parquet schema to our Schema type
        let columns = parquet_schema.get_fields().iter().map(|field| {
            Column {
                name: field.name().to_string(),
                data_type: parquet_type_to_sql_type(field),
                nullable: !field.is_required(),
            }
        }).collect();

        Ok(Schema { columns })
    }

    /// Parse SQL and create logical plan
    pub fn plan_sql(&self, sql: &str) -> Result<LogicalPlan> {
        let dialect = GenericDialect {};
        let statements = Parser::parse_sql(&dialect, sql)?;

        if statements.len() != 1 {
            return Err(anyhow!("Expected exactly one statement"));
        }

        match &statements[0] {
            Statement::Query(query) => self.plan_query(query),
            _ => Err(anyhow!("Only SELECT queries are supported")),
        }
    }

    fn plan_query(&self, query: &Query) -> Result<LogicalPlan> {
        // Start with FROM clause
        let mut plan = self.plan_from(&query.body)?;

        // Add WHERE clause if present
        if let SetExpr::Select(select) = &*query.body {
            if let Some(selection) = &select.selection {
                let predicate = self.plan_expr(selection, plan.output_schema())?;
                plan = LogicalPlan::Filter {
                    predicate,
                    input: Box::new(plan),
                };
            }

            // Add GROUP BY if present
            if !select.group_by.is_empty() {
                plan = self.plan_group_by(plan, select)?;
            }

            // Add projection (SELECT clause)
            plan = self.plan_projection(plan, select)?;
        }

        // Add ORDER BY if present
        if !query.order_by.is_empty() {
            plan = self.plan_order_by(plan, &query.order_by)?;
        }

        // Add LIMIT if present
        if let Some(limit) = &query.limit {
            plan = self.plan_limit(plan, limit)?;
        }

        Ok(plan)
    }

    // ... additional helper methods for each clause type
}
```

---

## Phase 5: Core Compilation - The Staged Interpreter

This is the heart of the system. Following the **First Futamura Projection**, we write an interpreter using staged types. When executed on a concrete query, it generates specialized code.

### 5.1 Compilation Context

```rust
// src/compile/context.rs

use rust_lms::prelude::*;

/// Context for code generation
/// Tracks the rust-lms Compiler and current variables
pub struct CompileContext<'a> {
    pub compiler: &'a mut Compiler<'a>,
    /// Maps table names to their buffer references
    pub table_buffers: HashMap<String, Var<SRef<'a, Slice<U8Type>>>>,
}

impl<'a> CompileContext<'a> {
    pub fn new(compiler: &'a mut Compiler<'a>) -> Self {
        Self {
            compiler,
            table_buffers: HashMap::new(),
        }
    }
}
```

### 5.2 The Callback-Based Operator Interface

The key pattern from the papers: each operator has an `exec` method that takes a callback. This enables operator fusion.

```rust
// src/operators/mod.rs

use rust_lms::prelude::*;

/// The callback type - receives a record, produces staged code
/// This is the "yld" function from the paper
pub type RecordCallback<'a> = Box<dyn FnOnce(StagedRecord) -> Box<dyn Staged<Out = UnitType>> + 'a>;

/// Trait for all operators - the staged interpreter interface
///
/// Key insight: `exec` doesn't return records directly.
/// Instead, it takes a callback that it invokes for each record.
/// This enables fusion: the callback can contain the next operator's logic.
pub trait Operator {
    /// Execute this operator, calling `callback` for each output record
    ///
    /// Returns staged code that, when compiled and run, will:
    /// 1. Iterate over input data
    /// 2. Apply this operator's logic
    /// 3. Call the callback for each result record
    fn exec<'a>(
        &self,
        ctx: &mut CompileContext<'a>,
        callback: RecordCallback<'a>,
    ) -> Box<dyn Staged<Out = UnitType>>;

    /// Output schema of this operator
    fn output_schema(&self) -> &Schema;
}
```

### 5.3 Scan Operator (Parquet/Arrow)

```rust
// src/operators/scan.rs

use rust_lms::prelude::*;
use arrow::array::*;
use arrow::datatypes::*;

/// Scans data from a Parquet file chunk by chunk
///
/// At compile time: generates a loop over chunks
/// At runtime: the compiled loop iterates over actual data
pub struct ScanOperator {
    pub file_path: String,
    pub schema: Schema,
    pub projection: Vec<usize>,
}

impl Operator for ScanOperator {
    fn exec<'a>(
        &self,
        ctx: &mut CompileContext<'a>,
        callback: RecordCallback<'a>,
    ) -> Box<dyn Staged<Out = UnitType>> {
        // The schema is static - known at compile time
        let schema = self.schema.clone();
        let projection = self.projection.clone();

        // At compile time, we generate code that:
        // 1. Gets a pointer to the data buffer (passed at runtime)
        // 2. Loops over each row
        // 3. Extracts fields and calls the callback

        ctx.compiler.fun1("scan", move |vctx, data_ptr: Var<SRef<Slice<U8Type>>>| {
            // Get the number of rows (stored at start of buffer)
            let num_rows = vctx.let_var(/* load row count from buffer */);

            // Loop counter
            let i = vctx.let_var(0u64);

            // Main scan loop
            let loop_body = while_loop(
                lt(*i, *num_rows),
                {
                    // Build a StagedRecord for row i
                    // Each field is a Var<T> loaded from the appropriate offset
                    let fields: Vec<StagedValue> = projection.iter().map(|&col_idx| {
                        let col = &schema.columns[col_idx];
                        match col.data_type {
                            SqlType::Int64 => {
                                // Calculate offset and load i64 value
                                let offset = /* compute offset for column col_idx, row i */;
                                let val = array_index(data_ptr, offset);
                                StagedValue::Int64(vctx.let_var(val))
                            }
                            SqlType::Float64 => {
                                // Similar for f64
                                StagedValue::Float64(vctx.let_var(/* load f64 */))
                            }
                            // ... other types
                        }
                    }).collect();

                    let record = StagedRecord {
                        schema: schema.clone(),
                        fields,
                    };

                    // CRITICAL: Call the callback with this record
                    // This is where operator fusion happens!
                    // The callback contains the next operator's code
                    let callback_result = callback(record);

                    // Increment loop counter
                    (callback_result, assign(*i, add(*i, 1u64)))
                }
            );

            loop_body
        })
    }

    fn output_schema(&self) -> &Schema {
        &self.schema
    }
}
```

### 5.4 Filter Operator (Fused)

```rust
// src/operators/filter.rs

/// Filter operator - selects rows matching a predicate
///
/// This operator FUSES with its parent: instead of materializing
/// intermediate results, it just wraps the callback in a conditional.
pub struct FilterOperator {
    pub input: Box<dyn Operator>,
    pub predicate: CompiledExpr,
    pub output_schema: Schema,
}

impl Operator for FilterOperator {
    fn exec<'a>(
        &self,
        ctx: &mut CompileContext<'a>,
        callback: RecordCallback<'a>,
    ) -> Box<dyn Staged<Out = UnitType>> {
        let predicate = self.predicate.clone();

        // Create a new callback that:
        // 1. Evaluates the predicate
        // 2. If true, calls the original callback
        // 3. If false, does nothing
        let filter_callback: RecordCallback<'a> = Box::new(move |record: StagedRecord| {
            // Evaluate predicate on this record - returns Var<BoolType>
            let pred_result: Var<BoolType> = compile_expr(&predicate, &record);

            // Generate: if (predicate) { callback(record) }
            let then_branch = callback(record);

            if_then(pred_result, then_branch).boxed()
        });

        // Execute input operator with our wrapped callback
        // The filter logic is now FUSED into the scan loop!
        self.input.exec(ctx, filter_callback)
    }

    fn output_schema(&self) -> &Schema {
        &self.output_schema
    }
}
```

### 5.5 Project Operator (Fused)

```rust
// src/operators/project.rs

/// Project operator - computes new columns from expressions
/// Also fuses with its input - no intermediate materialization
pub struct ProjectOperator {
    pub input: Box<dyn Operator>,
    pub expressions: Vec<CompiledExpr>,
    pub output_schema: Schema,
}

impl Operator for ProjectOperator {
    fn exec<'a>(
        &self,
        ctx: &mut CompileContext<'a>,
        callback: RecordCallback<'a>,
    ) -> Box<dyn Staged<Out = UnitType>> {
        let expressions = self.expressions.clone();
        let output_schema = self.output_schema.clone();

        let project_callback: RecordCallback<'a> = Box::new(move |input_record: StagedRecord| {
            // Compute each output expression
            let output_fields: Vec<StagedValue> = expressions.iter().map(|expr| {
                compile_expr(expr, &input_record)
            }).collect();

            // Create output record with new fields
            let output_record = StagedRecord {
                schema: output_schema.clone(),
                fields: output_fields,
            };

            // Pass to next operator
            callback(output_record)
        });

        self.input.exec(ctx, project_callback)
    }

    fn output_schema(&self) -> &Schema {
        &self.output_schema
    }
}
```

### 5.6 Expression Compilation

```rust
// src/compile/codegen.rs

/// Compile an expression to staged code
///
/// Takes a CompiledExpr (known at compile time) and a StagedRecord
/// (with dynamic Var<T> fields), returns a StagedValue
pub fn compile_expr(expr: &CompiledExpr, record: &StagedRecord) -> StagedValue {
    match expr {
        CompiledExpr::ColumnRef(idx) => {
            // Column reference - just get the field from the record
            record.get_idx(*idx).unwrap().clone()
        }

        CompiledExpr::Literal(lit) => {
            match lit {
                LiteralValue::Int64(v) => StagedValue::Int64(Const::new(*v).into()),
                LiteralValue::Float64(v) => StagedValue::Float64(Const::new(*v).into()),
                LiteralValue::Boolean(v) => StagedValue::Boolean(Const::new(*v).into()),
                // ...
            }
        }

        CompiledExpr::BinaryOp { left, op, right } => {
            let left_val = compile_expr(left, record);
            let right_val = compile_expr(right, record);

            match (left_val, right_val, op) {
                // Integer operations
                (StagedValue::Int64(l), StagedValue::Int64(r), BinaryOperator::Add) => {
                    StagedValue::Int64(add(l, r))
                }
                (StagedValue::Int64(l), StagedValue::Int64(r), BinaryOperator::Lt) => {
                    StagedValue::Boolean(lt(l, r))
                }
                // Float operations
                (StagedValue::Float64(l), StagedValue::Float64(r), BinaryOperator::Mul) => {
                    StagedValue::Float64(mul(l, r))
                }
                // Comparison generates boolean
                (StagedValue::Int64(l), StagedValue::Int64(r), BinaryOperator::Eq) => {
                    StagedValue::Boolean(eq(l, r))
                }
                // ... more combinations
            }
        }
    }
}

/// Compile a predicate to a boolean Var
pub fn compile_predicate(expr: &CompiledExpr, record: &StagedRecord) -> Var<BoolType> {
    match compile_expr(expr, record) {
        StagedValue::Boolean(b) => b,
        _ => panic!("Predicate must evaluate to boolean"),
    }
}
```

---

## Phase 6: Pipeline Breakers

Pipeline breakers (join, group by) require materializing intermediate results. They use specialized hash tables.

### 6.1 Staged Hash Table

```rust
// src/compile/hash_table.rs

/// A compile-time hash table abstraction
/// Schema is static, entries are dynamic Var<T> values
///
/// Following the paper: this is a generation-time abstraction
/// that produces specialized array operations in generated code
pub struct StagedHashTable {
    /// Schema of keys
    pub key_schema: Schema,
    /// Schema of values
    pub value_schema: Schema,
    /// Size (power of 2 for fast modulo)
    pub size: usize,
}

impl StagedHashTable {
    /// Generate hash table allocation code
    pub fn allocate<'a>(&self, ctx: &mut VarBuilder<'a>) -> HashTableVars<'a> {
        // Allocate arrays for each key column
        let key_arrays: Vec<_> = self.key_schema.columns.iter().map(|col| {
            match col.data_type {
                SqlType::Int64 => {
                    let arr = /* allocate i64 array of self.size */;
                    ctx.let_var(arr)
                }
                // ... other types
            }
        }).collect();

        // Allocate arrays for each value column
        let value_arrays: Vec<_> = self.value_schema.columns.iter().map(|col| {
            // ... similar
        }).collect();

        // Track which slots are used
        let used = ctx.let_var(/* allocate bool array */);
        let next_slot = ctx.let_var(0u64);

        HashTableVars {
            key_arrays,
            value_arrays,
            used,
            next_slot,
            size: self.size,
        }
    }

    /// Generate code to insert a key-value pair
    pub fn insert<'a>(
        &self,
        ht: &HashTableVars<'a>,
        key: &StagedRecord,
        value: &StagedRecord,
    ) -> impl Staged<Out = UnitType> {
        // Compute hash of key
        let hash = self.compute_hash(key);
        let slot = mod_op(hash, ht.size as u64);

        // Linear probing to find empty slot
        // ... generate probing loop

        // Store key and value in slot
        // ... generate store operations
    }

    /// Generate code to probe for a key
    pub fn probe<'a, F>(
        &self,
        ht: &HashTableVars<'a>,
        key: &StagedRecord,
        on_match: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(StagedRecord) -> Box<dyn Staged<Out = UnitType>>,
    {
        // Compute hash, probe, call on_match for each matching entry
        // ... generate probing loop with callback
    }
}
```

### 6.2 Hash Join Operator

```rust
// src/operators/hash_join.rs

/// Hash join - a pipeline breaker
///
/// Build phase: materialize the build side into a hash table
/// Probe phase: scan probe side, look up matches in hash table
pub struct HashJoinOperator {
    pub build_input: Box<dyn Operator>,
    pub probe_input: Box<dyn Operator>,
    pub build_keys: Vec<usize>,
    pub probe_keys: Vec<usize>,
    pub output_schema: Schema,
}

impl Operator for HashJoinOperator {
    fn exec<'a>(
        &self,
        ctx: &mut CompileContext<'a>,
        callback: RecordCallback<'a>,
    ) -> Box<dyn Staged<Out = UnitType>> {
        let build_keys = self.build_keys.clone();
        let probe_keys = self.probe_keys.clone();
        let output_schema = self.output_schema.clone();

        // === BUILD PHASE ===
        // Create hash table for build side
        let ht = StagedHashTable {
            key_schema: extract_key_schema(&self.build_input.output_schema(), &build_keys),
            value_schema: self.build_input.output_schema().clone(),
            size: DEFAULT_HT_SIZE,
        };

        let ht_vars = ht.allocate(&mut ctx.compiler.var_builder());

        // Build side callback: insert each record into hash table
        let build_callback: RecordCallback<'a> = Box::new(|record: StagedRecord| {
            let key_record = extract_key(&record, &build_keys);
            ht.insert(&ht_vars, &key_record, &record).boxed()
        });

        // Execute build side
        let build_code = self.build_input.exec(ctx, build_callback);

        // === PROBE PHASE ===
        // Probe side callback: look up each record in hash table
        let probe_callback: RecordCallback<'a> = Box::new(move |probe_record: StagedRecord| {
            let key_record = extract_key(&probe_record, &probe_keys);

            // Probe returns all matches via callback
            ht.probe(&ht_vars, &key_record, |build_record| {
                // Merge build and probe records
                let joined = merge_records(&build_record, &probe_record, &output_schema);
                callback(joined)
            }).boxed()
        });

        // Execute probe side
        let probe_code = self.probe_input.exec(ctx, probe_callback);

        // Sequence: build first, then probe
        (build_code, probe_code).boxed()
    }

    fn output_schema(&self) -> &Schema {
        &self.output_schema
    }
}
```

### 6.3 Hash Aggregate Operator

```rust
// src/operators/aggregate.rs

/// Hash aggregate - groups rows and computes aggregates
/// Another pipeline breaker
pub struct HashAggregateOperator {
    pub input: Box<dyn Operator>,
    pub group_by_indices: Vec<usize>,
    pub aggregates: Vec<PhysicalAggregate>,
    pub output_schema: Schema,
}

impl Operator for HashAggregateOperator {
    fn exec<'a>(
        &self,
        ctx: &mut CompileContext<'a>,
        callback: RecordCallback<'a>,
    ) -> Box<dyn Staged<Out = UnitType>> {
        let group_indices = self.group_by_indices.clone();
        let aggregates = self.aggregates.clone();
        let output_schema = self.output_schema.clone();

        // Hash table: key = group by columns, value = aggregate accumulators
        let ht = StagedHashTable {
            key_schema: extract_key_schema(&self.input.output_schema(), &group_indices),
            value_schema: make_accumulator_schema(&aggregates),
            size: DEFAULT_HT_SIZE,
        };

        let ht_vars = ht.allocate(&mut ctx.compiler.var_builder());

        // === ACCUMULATE PHASE ===
        let accum_callback: RecordCallback<'a> = Box::new(move |record: StagedRecord| {
            let key = extract_key(&record, &group_indices);

            // Update or insert aggregate accumulators
            ht.update_or_insert(&ht_vars, &key,
                // Init function for new group
                |_| init_accumulators(&aggregates),
                // Update function for existing group
                |accums| update_accumulators(&aggregates, &record, accums)
            ).boxed()
        });

        let accum_code = self.input.exec(ctx, accum_callback);

        // === OUTPUT PHASE ===
        // Iterate over hash table and emit final results
        let output_code = ht.foreach(&ht_vars, |key, accums| {
            let final_values = finalize_accumulators(&aggregates, accums);
            let output_record = merge_records(&key, &final_values, &output_schema);
            callback(output_record)
        });

        (accum_code, output_code).boxed()
    }

    fn output_schema(&self) -> &Schema {
        &self.output_schema
    }
}
```

---

## Phase 7: Runtime Interface

### 7.1 Data Buffer Format

Define how Arrow RecordBatches are passed to compiled code:

```rust
// src/execution/runtime.rs

use arrow::array::*;
use arrow::record_batch::RecordBatch;

/// Convert Arrow RecordBatch to flat buffer for compiled code
///
/// Layout: [num_rows: u64] [col0_data...] [col1_data...] ...
/// Each column is stored contiguously
pub fn record_batch_to_buffer(batch: &RecordBatch) -> Vec<u8> {
    let mut buffer = Vec::new();

    // Write row count
    buffer.extend_from_slice(&(batch.num_rows() as u64).to_le_bytes());

    // Write each column
    for col in batch.columns() {
        match col.data_type() {
            DataType::Int64 => {
                let arr = col.as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..arr.len() {
                    buffer.extend_from_slice(&arr.value(i).to_le_bytes());
                }
            }
            DataType::Float64 => {
                let arr = col.as_any().downcast_ref::<Float64Array>().unwrap();
                for i in 0..arr.len() {
                    buffer.extend_from_slice(&arr.value(i).to_le_bytes());
                }
            }
            // ... other types
        }
    }

    buffer
}

/// Read Parquet file and get RecordBatches
pub fn read_parquet_batches(path: &str) -> impl Iterator<Item = RecordBatch> {
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use std::fs::File;

    let file = File::open(path).unwrap();
    let builder = ParquetRecordBatchReaderBuilder::try_new(file).unwrap();
    let reader = builder.with_batch_size(8192).build().unwrap();

    reader.map(|r| r.unwrap())
}
```

### 7.2 Query Executor

```rust
// src/execution/mod.rs

use rust_lms::prelude::*;

/// Main query execution entry point
pub struct QueryExecutor {
    planner: QueryPlanner,
}

impl QueryExecutor {
    pub fn new() -> Self {
        Self {
            planner: QueryPlanner::new(),
        }
    }

    /// Register a Parquet file as a table
    pub fn register_table(&mut self, name: &str, path: &str) -> Result<()> {
        self.planner.register_parquet(name, path)
    }

    /// Execute a SQL query and return results
    pub fn execute(&self, sql: &str) -> Result<Vec<RecordBatch>> {
        // 1. Parse SQL to logical plan
        let logical_plan = self.planner.plan_sql(sql)?;

        // 2. Optimize and convert to physical plan
        let physical_plan = optimize_and_physicalize(logical_plan)?;

        // 3. Compile the plan
        let compiled = self.compile_plan(&physical_plan)?;

        // 4. Execute compiled code on data
        let results = self.execute_compiled(&compiled, &physical_plan)?;

        Ok(results)
    }

    fn compile_plan(&self, plan: &PhysicalPlan) -> Result<CompiledQuery> {
        let mut compiler = Compiler::new();
        let mut ctx = CompileContext::new(&mut compiler);

        // Create the root operator
        let root_op = build_operator_tree(plan);

        // Collect results callback - stores output records
        let results_buffer = /* allocate output buffer */;
        let collect_callback: RecordCallback = Box::new(move |record| {
            // Store record in results buffer
            store_record_to_buffer(&results_buffer, &record).boxed()
        });

        // Generate code by executing the staged interpreter
        let query_code = root_op.exec(&mut ctx, collect_callback);

        // Compile to native code
        let compiled = compiler.compile(query_code)?;

        Ok(CompiledQuery {
            module: compiled,
            // ... metadata
        })
    }

    fn execute_compiled(
        &self,
        compiled: &CompiledQuery,
        plan: &PhysicalPlan,
    ) -> Result<Vec<RecordBatch>> {
        let mut results = Vec::new();

        // Get the compiled function
        let query_fn = compiled.module.as_fn();

        // Read data and execute
        match plan {
            PhysicalPlan::ParquetScan { file_path, .. } => {
                for batch in read_parquet_batches(file_path) {
                    let buffer = record_batch_to_buffer(&batch);

                    // Call compiled code with data buffer
                    let output = query_fn(&buffer);

                    // Convert output to RecordBatch
                    let result_batch = buffer_to_record_batch(output, plan.output_schema());
                    results.push(result_batch);
                }
            }
            // Handle other plan types (joins need multiple inputs)
            _ => { /* ... */ }
        }

        Ok(results)
    }
}
```

---

## Phase 8: Integration and Testing

### 8.1 Example Usage

```rust
// examples/simple_query.rs

use dio_sql::QueryExecutor;

fn main() -> Result<()> {
    let mut executor = QueryExecutor::new();

    // Register tables
    executor.register_table("orders", "data/orders.parquet")?;
    executor.register_table("customers", "data/customers.parquet")?;

    // Simple filter query
    let results = executor.execute(
        "SELECT order_id, amount FROM orders WHERE amount > 100"
    )?;

    // Join query
    let results = executor.execute(
        "SELECT c.name, o.amount
         FROM orders o
         JOIN customers c ON o.customer_id = c.id
         WHERE o.amount > 100"
    )?;

    // Aggregate query
    let results = executor.execute(
        "SELECT customer_id, SUM(amount) as total
         FROM orders
         GROUP BY customer_id"
    )?;

    // Print results
    for batch in &results {
        println!("{:?}", batch);
    }

    Ok(())
}
```

### 8.2 Test Cases

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_scan() {
        // Generate test parquet file
        let path = create_test_parquet(&[
            ("id", vec![1i64, 2, 3]),
            ("value", vec![10i64, 20, 30]),
        ]);

        let mut executor = QueryExecutor::new();
        executor.register_table("test", &path).unwrap();

        let results = executor.execute("SELECT * FROM test").unwrap();
        assert_eq!(results[0].num_rows(), 3);
    }

    #[test]
    fn test_filter() {
        let path = create_test_parquet(&[
            ("id", vec![1i64, 2, 3, 4, 5]),
            ("value", vec![10i64, 20, 30, 40, 50]),
        ]);

        let mut executor = QueryExecutor::new();
        executor.register_table("test", &path).unwrap();

        let results = executor.execute(
            "SELECT id FROM test WHERE value > 25"
        ).unwrap();

        // Should return rows 3, 4, 5
        assert_eq!(results[0].num_rows(), 3);
    }

    #[test]
    fn test_join() {
        // ... test hash join
    }

    #[test]
    fn test_aggregate() {
        // ... test group by and aggregations
    }
}
```

---

## Implementation Order

Follow this order to build incrementally:

1. **Week 1-2: Foundation**
   - Set up project structure
   - Implement type system (Schema, SqlType, StagedValue, StagedRecord)
   - Create basic logical/physical plan structures
   - Write Parquet schema reader

2. **Week 3-4: Core Operators**
   - Implement Scan operator (Parquet → staged record loop)
   - Implement Filter operator (with fusion)
   - Implement Project operator (with fusion)
   - Test fused scan-filter-project pipeline

3. **Week 5-6: SQL Parser**
   - Integrate sqlparser crate
   - Implement SQL → LogicalPlan conversion
   - Add expression compilation

4. **Week 7-8: Pipeline Breakers**
   - Implement StagedHashTable
   - Implement Hash Join operator
   - Implement Hash Aggregate operator
   - Test multi-stage queries

5. **Week 9-10: Integration**
   - Build QueryExecutor
   - Handle chunk-based execution
   - Add error handling
   - Write comprehensive tests

---

## Key Design Decisions Summary

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Execution Model | Push-based (callbacks) | Better for compilation; enables operator fusion |
| Schema Treatment | Static (compile-time) | Enables type specialization; no runtime dispatch |
| Field Values | Staged (Var<T>) | Generates optimized native access code |
| Hash Tables | Generation-time abstraction | Compiles to flat arrays; no library calls |
| Data Format | Arrow RecordBatch | Industry standard; efficient columnar layout |
| JIT Backend | Cranelift (via rust-lms) | Fast compilation; good code quality |

---

## Questions for Clarification

Before proceeding, I'd like to clarify:

1. **String handling**: Should strings be handled as:
   - Direct pointers into Arrow buffers (zero-copy but requires careful lifetime management)
   - Copied to temporary buffers (simpler but more overhead)
   - Dictionary-encoded for common values (as in the LB2 paper)

2. **Null handling**: How should NULL values be handled?
   - Generate explicit null checks in compiled code
   - Use Arrow's validity bitmaps
   - Ignore nulls initially (simplest)

3. **Memory management**: For pipeline breakers (hash tables):
   - Pre-allocate fixed size and fail if exceeded
   - Dynamically grow (requires external function calls)
   - Estimate size from statistics

4. **Chunk processing**: Should we:
   - Compile once and reuse for all chunks (current plan)
   - Compile per-chunk with potentially different optimizations
   - Support both modes

5. **Error handling**: What should happen on:
   - Runtime errors (division by zero, overflow)
   - Data errors (malformed parquet file)
   - Type mismatches at parse time
