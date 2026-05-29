# SQL to rust-lms: Implementation Plan

This document outlines what needs to be added to `rust-lms` to support compiled SQL execution using the push-based operator model described in the "SQL to C" paper, integrated with Apache Arrow via the arrow-parquet crate and datafusion-sql for query parsing.

## Architecture Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌────────────────┐     ┌─────────────┐
│  Arrow Parquet  │────>│  DataFusion SQL  │────>│  Push Operator │────>│  rust-lms   │
│  (RecordBatch)  │     │  (Logical Plan)  │     │  Translation   │     │  Compiler   │
└─────────────────┘     └──────────────────┘     └────────────────┘     └─────────────┘
                                                                               │
                                                                               v
                                                                        ┌─────────────┐
                                                                        │  Cranelift  │
                                                                        │  JIT Code   │
                                                                        └─────────────┘
```

## Current rust-lms Capabilities

**Types**: `i64`, `U64Type`, `F64Type`, `BoolType`, `UnitType`
**Values**: `VarRef<T>`, `Const<T>`
**Operations**: `Add`, `Sub`, `Mul`, `Div`, `Lt`, `Eq`
**Control**: `Seq`, `IfThenElse`, `IfThen`, `While`
**Functions**: `fun1`, `fun1_rec`, `call1`, `FunRef`

---

## Phase 1: Pointer and Memory Types

The foundation for all columnar data access. Arrow arrays are represented as pointers with lengths.

### 1.1 New Types (`types.rs`)

```rust
/// Raw pointer type - the base for all memory access
#[derive(Clone, Copy, Debug)]
pub struct PtrType<T: StagedType> {
    _phantom: PhantomData<T>,
}

impl<T: StagedType> StagedType for PtrType<T> {
    type RuntimeValue = *const T::RuntimeValue;
    fn cranelift_type() -> Type { types::I64 } // pointer-sized
}

/// Slice type - (pointer, length) pair for array access
#[derive(Clone, Copy, Debug)]
pub struct SliceType<T: StagedType> {
    _phantom: PhantomData<T>,
}

// For SliceType, RuntimeValue could be (*const T, usize)
// But in Cranelift we pass as two separate i64 values
```

### 1.2 Pointer Operations (`ptr.rs` - new module)

```rust
/// Load value from pointer: *ptr
pub struct Load<P> { ptr: P }

impl<P, T> Staged for Load<P>
where
    P: Staged<Out = PtrType<T>>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.ptr.codegen(ctx);
        // Cranelift: load T from memory at ptr
        ctx.builder.ins().load(T::cranelift_type(), MemFlags::trusted(), ptr, 0)
    }
}

/// Store value to pointer: *ptr = val
pub struct Store<P, V> { ptr: P, val: V }

/// Pointer offset: ptr + offset (with scaling by element size)
pub struct PtrOffset<P, I> { ptr: P, index: I }

impl<P, I, T> Staged for PtrOffset<P, I>
where
    P: Staged<Out = PtrType<T>>,
    I: Staged<Out = i64>,
    T: StagedType,
{
    type Out = PtrType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);
        let element_size = std::mem::size_of::<T::RuntimeValue>() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let offset = ctx.builder.ins().imul(idx, scale);
        ctx.builder.ins().iadd(ptr, offset)
    }
}

/// Array indexing: ptr[index]
pub struct ArrayIndex<P, I> { ptr: P, index: I }
// Combines PtrOffset + Load
```

### 1.3 Stack Allocation

```rust
/// Allocate stack space for a value
pub struct StackSlot<T: StagedType> {
    slot_id: usize, // tracked in CompilationContext
    _phantom: PhantomData<T>,
}

// CompilationContext needs:
pub struct CompilationContext<'a, 'b> {
    // ... existing fields ...
    pub stack_slots: &'b mut HashMap<usize, StackSlot>,
}
```

---

## Phase 2: External Function Calls

Critical for calling Rust runtime functions (memory allocation, string operations, hash functions).

### 2.1 External Function Registration

```rust
// In Compiler:
impl Compiler {
    /// Register an external Rust function that can be called from JIT code
    pub fn extern_fn<A, R>(&mut self, name: &str, fn_ptr: fn(A) -> R) -> ExternFunRef<A, R>
    where
        A: ExternParam,
        R: ExternReturn,
    {
        // Store the function pointer and signature
        // During compilation, use JITBuilder::symbol to expose it
    }
}

/// Reference to an external function
#[derive(Clone, Copy)]
pub struct ExternFunRef<A, R> {
    id: usize,
    _phantom: PhantomData<(A, R)>,
}

/// Call an external function
pub struct ExternCall<F, ARGS> {
    func: F,
    args: ARGS,
}
```

### 2.2 JIT Symbol Registration

```rust
// During module creation in compile():
let mut builder = JITBuilder::new(default_libcall_names())?;

// Register external symbols
for (name, ptr) in &self.extern_symbols {
    builder.symbol(name, *ptr);
}
```

### 2.3 Multi-Argument Functions

Extend beyond `fun1` to support multiple arguments:

```rust
pub fn fun2<A1, A2, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef2<A1, A2, OUT>
pub fn fun3<A1, A2, A3, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef3<A1, A2, A3, OUT>

// Or use a tuple-based approach:
pub fn fun<ARGS: TupleArgs, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef<ARGS, OUT>
```

---

## Phase 3: Arrow Integration Types

Types that map directly to Arrow array representations.

### 3.1 Arrow Buffer Type

```rust
/// Represents an Arrow array buffer (data pointer + length + null bitmap)
pub struct ArrowBuffer<T: StagedType> {
    data_ptr: VarRef<PtrType<T>>,
    length: VarRef<i64>,
    null_bitmap: VarRef<PtrType<U8Type>>, // Optional
}

/// Arrow string array representation (using string view format)
pub struct ArrowStringArray {
    data_ptr: VarRef<PtrType<U8Type>>,    // String data
    offsets_ptr: VarRef<PtrType<I32Type>>, // Offset array
    length: VarRef<i64>,
}
```

### 3.2 Loop Over Arrow Array

```rust
/// For-each loop over array elements (push-style iteration)
pub struct ForEach<ARRAY, BODY> {
    array: ARRAY,
    body: BODY, // Takes VarRef<T> for current element and VarRef<i64> for index
}

// Example usage:
let sum = compiler.var::<i64>();
compiler.for_each(arr, |elem: VarRef<i64>, idx| {
    assign(sum, add(sum, elem))
})
```

### 3.3 Null Handling

```rust
/// Nullable wrapper type
pub struct NullableType<T: StagedType> {
    _phantom: PhantomData<T>,
}

/// Check if null at index
pub struct IsNull<BITMAP, IDX> {
    bitmap: BITMAP,
    index: IDX,
}

/// If-null-then-else pattern
pub struct IfNull<COND, VALUE, DEFAULT> {
    is_null: COND,
    value: VALUE,
    default: DEFAULT,
}
```

---

## Phase 4: String Operations

Strings require external Rust functions since they involve variable-length data.

### 4.1 String Type

```rust
/// Arrow string view: (ptr, len) pair
pub struct StringViewType;

impl StagedType for StringViewType {
    type RuntimeValue = (*const u8, usize); // pointer and length
    fn cranelift_type() -> Type {
        // Two i64 values packed, or use struct type
        types::I64 // For pointer; length passed separately
    }
}
```

### 4.2 String Runtime Functions

These are Rust functions registered as external symbols:

```rust
// Runtime library (runtime.rs)
#[no_mangle]
pub extern "C" fn str_eq(a_ptr: *const u8, a_len: usize, b_ptr: *const u8, b_len: usize) -> bool {
    if a_len != b_len { return false; }
    unsafe {
        let a = std::slice::from_raw_parts(a_ptr, a_len);
        let b = std::slice::from_raw_parts(b_ptr, b_len);
        a == b
    }
}

#[no_mangle]
pub extern "C" fn str_hash(ptr: *const u8, len: usize) -> u64 {
    // FNV-1a or similar fast hash
}

#[no_mangle]
pub extern "C" fn str_lt(a_ptr: *const u8, a_len: usize, b_ptr: *const u8, b_len: usize) -> bool {
    // Lexicographic comparison
}
```

### 4.3 Staged String Operations

```rust
/// String equality comparison (calls external function)
pub struct StrEq<L, R> { left: L, right: R }

impl<L, R> Staged for StrEq<L, R>
where
    L: Staged<Out = StringViewType>,
    R: Staged<Out = StringViewType>,
{
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate call to str_eq runtime function
    }
}
```

---

## Phase 5: Hash Table Operations

Critical for hash joins and aggregations.

### 5.1 Hash Table Type

```rust
/// Opaque hash table handle
pub struct HashTableType<K: StagedType, V: StagedType> {
    _phantom: PhantomData<(K, V)>,
}

// Runtime: Use Rust's HashMap or custom implementation
type HashTablePtr = *mut std::collections::HashMap<i64, Vec<usize>>;
```

### 5.2 Hash Table Operations (External Functions)

```rust
// Runtime library
#[no_mangle]
pub extern "C" fn ht_create() -> HashTablePtr { ... }

#[no_mangle]
pub extern "C" fn ht_probe(ht: HashTablePtr, key: i64) -> *const usize { ... }

#[no_mangle]
pub extern "C" fn ht_insert(ht: HashTablePtr, key: i64, value: usize) { ... }

#[no_mangle]
pub extern "C" fn ht_destroy(ht: HashTablePtr) { ... }
```

### 5.3 Staged Hash Operations

```rust
/// Create hash table
pub struct HashTableCreate<K, V> { _phantom: PhantomData<(K, V)> }

/// Probe hash table
pub struct HashTableProbe<HT, KEY> { table: HT, key: KEY }

/// Insert into hash table
pub struct HashTableInsert<HT, KEY, VAL> { table: HT, key: KEY, value: VAL }

/// Iterate over hash table bucket
pub struct HashTableForEach<HT, KEY, BODY> { table: HT, key: KEY, body: BODY }
```

---

## Phase 6: Tuple/Record Types

For handling SQL rows and intermediate results.

### 6.1 Tuple Type

```rust
/// Tuple type for multi-value records
pub struct TupleType2<A: StagedType, B: StagedType> {
    _phantom: PhantomData<(A, B)>,
}

pub struct TupleType3<A, B, C> { ... }
// etc.

// Or use a macro/builder pattern:
tuple_type!(RecordType, a: i64, b: F64Type, c: StringViewType);
```

### 6.2 Tuple Operations

```rust
/// Create a tuple from values
pub struct MakeTuple<A, B> { a: A, b: B }

/// Extract first element
pub struct TupleFirst<T> { tuple: T }

/// Extract second element
pub struct TupleSecond<T> { tuple: T }

// For larger tuples, use index-based access
pub struct TupleGet<T, const N: usize> { tuple: T }
```

### 6.3 Memory Layout

Tuples need to be laid out in memory for hash table storage:

```rust
trait TupleLayout: StagedType {
    fn field_offsets() -> &'static [usize];
    fn total_size() -> usize;
}
```

---

## Phase 7: Push Operator Framework

The core pattern from the "SQL to C" paper.

### 7.1 Consumer Trait

```rust
/// A consumer that receives records from upstream operators
pub trait Consumer<T: StagedType> {
    /// Process one record, returning staged code
    fn consume(&self, record: VarRef<T>, compiler: &mut Compiler) -> impl Staged<Out = UnitType>;
}

/// Operator that produces records
pub trait Operator {
    type Output: StagedType;

    /// Generate code that calls consumer for each output record
    fn produce(&self, consumer: &dyn Consumer<Self::Output>, compiler: &mut Compiler)
        -> impl Staged<Out = UnitType>;
}
```

### 7.2 Scan Operator

```rust
pub struct TableScan {
    columns: Vec<ColumnDesc>,
}

impl Operator for TableScan {
    type Output = RecordType;

    fn produce(&self, consumer: &dyn Consumer<RecordType>, compiler: &mut Compiler)
        -> impl Staged<Out = UnitType>
    {
        // Generate: for i in 0..length { consumer.consume(record_at(i)) }
        let i = compiler.var::<i64>();
        seq(
            assign(i, Const::new(0)),
            while_loop(
                lt(i, self.length),
                seq(
                    consumer.consume(self.record_at(i), compiler),
                    assign(i, add(i, Const::new(1))),
                ),
            ),
        )
    }
}
```

### 7.3 Filter Operator

```rust
pub struct Filter<UPSTREAM, PRED> {
    upstream: UPSTREAM,
    predicate: PRED,
}

impl<U: Operator, P> Operator for Filter<U, P>
where
    P: Fn(VarRef<U::Output>) -> impl Staged<Out = BoolType>,
{
    type Output = U::Output;

    fn produce(&self, consumer: &dyn Consumer<U::Output>, compiler: &mut Compiler)
        -> impl Staged<Out = UnitType>
    {
        // Create a filtering consumer that wraps the downstream consumer
        let filtering_consumer = FilteringConsumer {
            downstream: consumer,
            predicate: &self.predicate,
        };
        self.upstream.produce(&filtering_consumer, compiler)
    }
}
```

### 7.4 Project Operator

```rust
pub struct Project<UPSTREAM, PROJ> {
    upstream: UPSTREAM,
    projection: PROJ, // Maps input record to output record
}
```

### 7.5 Hash Join Operator

```rust
pub struct HashJoin<LEFT, RIGHT, KEY_LEFT, KEY_RIGHT> {
    left: LEFT,
    right: RIGHT,
    left_key: KEY_LEFT,
    right_key: KEY_RIGHT,
}

impl Operator for HashJoin<...> {
    fn produce(&self, consumer: &dyn Consumer<JoinedRecord>, compiler: &mut Compiler) {
        // Phase 1: Build hash table from left side
        let ht = compiler.var::<HashTableType>();
        seq(
            assign(ht, hash_table_create()),
            seq(
                // Scan left, insert into hash table
                self.left.produce(&BuildPhaseConsumer { ht }, compiler),
                // Phase 2: Probe with right side
                self.right.produce(&ProbePhaseConsumer { ht, consumer }, compiler),
            ),
        )
    }
}
```

### 7.6 Aggregation Operator

```rust
pub struct Aggregate<UPSTREAM, KEY, AGG> {
    upstream: UPSTREAM,
    group_key: KEY,
    aggregates: Vec<AGG>, // SUM, COUNT, AVG, etc.
}

/// Aggregation accumulator types
pub enum AggregateKind {
    Sum,
    Count,
    Min,
    Max,
    Avg, // Tracks sum and count
}
```

---

## Phase 8: Memory Management

### 8.1 Arena Allocator

For intermediate results during query execution:

```rust
// Runtime library
pub struct Arena {
    chunks: Vec<Vec<u8>>,
    current: *mut u8,
    remaining: usize,
}

#[no_mangle]
pub extern "C" fn arena_alloc(arena: *mut Arena, size: usize, align: usize) -> *mut u8 {
    // Fast bump allocation
}

#[no_mangle]
pub extern "C" fn arena_reset(arena: *mut Arena) {
    // Reset for next query/batch
}
```

### 8.2 Staged Allocation

```rust
/// Allocate from arena
pub struct ArenaAlloc<SIZE> {
    arena: VarRef<PtrType<Arena>>,
    size: SIZE,
}

/// Allocate and initialize a tuple
pub struct AllocTuple<T: TupleLayout> {
    arena: VarRef<PtrType<Arena>>,
    values: T,
}
```

---

## Phase 9: DataFusion Integration

### 9.1 Logical Plan Translation

```rust
use datafusion::logical_plan::LogicalPlan;

pub fn translate_plan(plan: &LogicalPlan) -> Box<dyn Operator> {
    match plan {
        LogicalPlan::TableScan(scan) => Box::new(translate_scan(scan)),
        LogicalPlan::Filter(filter) => Box::new(translate_filter(filter)),
        LogicalPlan::Projection(proj) => Box::new(translate_project(proj)),
        LogicalPlan::Join(join) => Box::new(translate_join(join)),
        LogicalPlan::Aggregate(agg) => Box::new(translate_aggregate(agg)),
        // ...
    }
}
```

### 9.2 Expression Translation

```rust
use datafusion::logical_plan::Expr;

pub fn translate_expr<T: StagedType>(
    expr: &Expr,
    row: VarRef<RecordType>,
    compiler: &mut Compiler,
) -> Box<dyn Staged<Out = T>> {
    match expr {
        Expr::Column(col) => Box::new(column_access(row, col.index)),
        Expr::Literal(lit) => Box::new(translate_literal(lit)),
        Expr::BinaryExpr { left, op, right } => {
            let l = translate_expr(left, row, compiler);
            let r = translate_expr(right, row, compiler);
            match op {
                Operator::Plus => Box::new(add(l, r)),
                Operator::Lt => Box::new(lt(l, r)),
                // ...
            }
        }
        // ...
    }
}
```

---

## Phase 10: Arrow RecordBatch Integration

### 10.1 RecordBatch to Staged

```rust
use arrow::record_batch::RecordBatch;

/// Compile a query for a specific RecordBatch schema
pub struct CompiledQuery {
    func: fn(batch: &RecordBatch, output: &mut Vec<RecordBatch>),
}

impl CompiledQuery {
    pub fn compile(plan: &LogicalPlan, schema: &Schema) -> Result<Self> {
        let mut compiler = Compiler::new();

        // Register RecordBatch accessor functions
        compiler.extern_fn("get_column_ptr", get_column_ptr);
        compiler.extern_fn("get_batch_length", get_batch_length);

        // Generate query code
        let operator = translate_plan(plan);
        let output_consumer = OutputConsumer::new();
        let code = operator.produce(&output_consumer, &mut compiler);

        compiler.compile(code)
    }
}
```

### 10.2 Column Access Runtime

```rust
#[no_mangle]
pub extern "C" fn get_i64_column(batch: *const RecordBatch, col_idx: usize) -> *const i64 {
    unsafe {
        let batch = &*batch;
        let array = batch.column(col_idx).as_any().downcast_ref::<Int64Array>().unwrap();
        array.values().as_ptr()
    }
}

#[no_mangle]
pub extern "C" fn get_batch_length(batch: *const RecordBatch) -> i64 {
    unsafe { (*batch).num_rows() as i64 }
}
```

---

## Implementation Priority

### Must Have (Core Functionality)

1. **Phase 1**: Pointer and memory types - foundation for everything
2. **Phase 2**: External function calls - required for runtime functions
3. **Phase 3**: Arrow integration types - basic array access
4. **Phase 7.1-7.3**: Basic operators (Scan, Filter, Project)

### Should Have (Feature Complete)

5. **Phase 4**: String operations - needed for real SQL queries
6. **Phase 5**: Hash table operations - for joins
7. **Phase 7.5-7.6**: Hash Join and Aggregation
8. **Phase 8**: Memory management

### Nice to Have (Production Ready)

9. **Phase 6**: Tuple types - cleaner record handling
10. **Phase 9-10**: Full DataFusion integration

---

## Tricky Implementation Details

### Pointer Safety

The main challenge is safely interfacing between Rust's ownership and raw pointers in JIT code:

```rust
// Problem: How do we ensure the Arrow array lives long enough?
// Solution: The compiled function takes &RecordBatch, which guarantees lifetime

pub struct CompiledQuery<'a> {
    _phantom: PhantomData<&'a ()>, // Lifetime of input data
    func: unsafe extern "C" fn(*const RecordBatch, *mut OutputBuffer),
}
```

### Calling Convention

Cranelift uses the system ABI by default. For external Rust functions:

```rust
// Ensure extern "C" ABI
#[no_mangle]
pub extern "C" fn my_runtime_fn(arg: i64) -> i64 { ... }

// In Cranelift signature:
let mut sig = module.make_signature();
sig.call_conv = CallConv::SystemV; // or WindowsFastcall on Windows
```

### String Views in Registers

Arrow StringViewArray uses 16-byte string views. In Cranelift:

```rust
// Option 1: Two i64 values (ptr, len)
sig.params.push(AbiParam::new(types::I64)); // ptr
sig.params.push(AbiParam::new(types::I64)); // len

// Option 2: Single i128 (if supported)
sig.params.push(AbiParam::new(types::I128));

// Option 3: Pass by reference (pointer to StringView struct)
sig.params.push(AbiParam::new(types::I64)); // *StringView
```

### SIMD Opportunities

For future optimization, Cranelift supports SIMD:

```rust
// Example: 4x i64 addition
let v1 = builder.ins().vload(types::I64X4, ptr1, offset);
let v2 = builder.ins().vload(types::I64X4, ptr2, offset);
let sum = builder.ins().iadd(v1, v2);
builder.ins().vstore(sum, out_ptr, offset);
```

### Error Handling

SQL queries can fail (division by zero, null dereference). Options:

1. **Panic/Abort**: Simplest, call Rust panic handler
2. **Error codes**: Return Result-like struct
3. **Exception-like**: Use Cranelift's trap mechanism

```rust
// Trap on division by zero
let zero = builder.ins().iconst(types::I64, 0);
let is_zero = builder.ins().icmp(IntCC::Equal, divisor, zero);
builder.ins().trapnz(is_zero, TrapCode::IntegerDivisionByZero);
```

---

## File Structure

```
rust-lms/src/
├── lib.rs              (add: mod ptr, mod string, mod arrow, mod hash, mod tuple, mod operator)
├── types.rs            (add: PtrType, SliceType, StringViewType)
├── staged.rs           (no changes needed)
├── control.rs          (add: ForEach, ForRange)
├── func.rs             (add: extern_fn, fun2, fun3, ExternCall)
├── ptr.rs              (new: Load, Store, PtrOffset, ArrayIndex, StackSlot)
├── string.rs           (new: StrEq, StrLt, StrHash, string runtime)
├── arrow.rs            (new: ArrowBuffer, arrow accessor wrappers)
├── hash.rs             (new: HashTable operations)
├── tuple.rs            (new: TupleType, MakeTuple, TupleGet)
├── operator/
│   ├── mod.rs          (Consumer, Operator traits)
│   ├── scan.rs         (TableScan)
│   ├── filter.rs       (Filter)
│   ├── project.rs      (Project)
│   ├── join.rs         (HashJoin)
│   └── aggregate.rs    (Aggregate)
├── runtime/
│   ├── mod.rs          (runtime function registration)
│   ├── string.rs       (str_eq, str_hash, etc.)
│   ├── hash_table.rs   (hash table implementation)
│   └── arena.rs        (arena allocator)
└── datafusion/
    ├── mod.rs          (DataFusion integration)
    ├── translate.rs    (LogicalPlan -> Operator)
    └── expr.rs         (Expr -> Staged)
```

---

## Testing Strategy

1. **Unit tests per module**: Test each new type and operation in isolation
2. **Integration tests**: Small SQL queries compiled and executed
3. **Correctness tests**: Compare results against DataFusion's interpreted execution
4. **Performance benchmarks**: TPC-H style queries comparing compiled vs interpreted

---

## Open Questions

1. **Null Handling Strategy**: Sentinel values vs explicit null checks?
2. **Memory Ownership**: Who owns hash table memory during query execution?
3. **Multi-threaded Execution**: Partition data and compile parallel operators?
4. **Dynamic Dispatch**: When do we need `Box<dyn Operator>` vs monomorphization?
5. **Code Caching**: Cache compiled queries based on plan hash?
