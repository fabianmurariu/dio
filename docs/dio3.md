# Dio v3: JIT Compiled Columnar Expression Evaluator

Dio is a high-performance JIT (Just-In-Time) compiled columnar expression evaluator that transforms arithmetic expressions written in a Lisp-like DSL into optimized machine code using a multi-stage compilation pipeline. It's designed for efficient columnar data processing with Apache Arrow arrays.

## Project Overview

Dio enables you to write mathematical expressions in a simple Lisp syntax and have them compiled to optimized machine code that operates directly on Apache Arrow arrays with zero-copy semantics. The system is particularly well-suited for analytical workloads that require high-performance vectorized computations.

### Key Features

- **Multi-stage Compilation Pipeline**: Lisp → ByteCode → SSA → Cranelift → Native Machine Code
- **Zero-copy Execution**: Direct pointer passing to compiled functions without data copying
- **Expression Caching**: Compiled functions are cached by expression hash to avoid recompilation
- **Unified Code Generation**: Both elementwise and reduction operations use consistent patterns
- **Apache Arrow Integration**: Native support for Arrow array formats

## Architecture: The New Pipeline

### Pipeline Overview

The Dio v3 compilation pipeline consists of four distinct phases:

```
Lisp Expression → ByteCode IR → SSA v2 IR → Cranelift IR → Native Machine Code
```

Each stage serves a specific purpose in the compilation process:

1. **Lisp Expression**: Human-readable mathematical expressions
2. **ByteCode IR**: C-like imperative intermediate representation  
3. **SSA v2 IR**: Static Single Assignment form with control flow
4. **Cranelift IR**: Low-level machine-independent assembly
5. **Native Code**: Platform-specific optimized machine instructions

### Phase 1: Expression DSL (Lisp)

Dio expressions are written in a typed Lisp dialect:

```lisp
;; Elementwise addition
(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))

;; Elementwise operations with multiple operands
(lambda ([U64Array a] [U64Array b] [U64Array c] U64Array) (+ a b c))

;; Mixed operations
(lambda ([U64Array x] [U64Array y] U64Array) (+ (* x 2) y))

;; Reduction operations
(lambda ([U64Array a] U64) (sum a))
(lambda ([U64Array a] [U64Array b] U64) (sum (+ a b)))
```

**Supported Types:**
- `U64Array`, `I64Array` - Array types for unsigned/signed 64-bit integers
- `U64`, `I64` - Scalar types (primarily for reduction results)

**Supported Operations:**
- **Elementwise**: `+`, `-`, `*`, `/` - operate element-by-element on arrays
- **Reductions**: `sum`, `count` - reduce arrays to scalars

### Phase 2: ByteCode Intermediate Representation

The ByteCode IR represents expressions as C-like imperative code with explicit loops and array indexing:

```rust
// Elementwise addition ByteCode
function(u64[] a, u64[] b, u64 length) -> u64[] {
  // Local variables:
  u64 i;

  for (i = 0; i < length; i += 1) {
    output[i] = (a[i] + b[i]);
  }
  return;
}

// Reduction ByteCode  
function(u64[] a, u64 length) -> u64 {
  // Local variables:
  u64 acc;
  u64 i;

  acc = 0;
  for (i = 0; i < length; i += 1) {
    acc = (acc + a[i]);
  }
  output[0] = acc;  // Unified: reductions write to length-1 arrays
  return;
}
```

**Key Design Decisions:**
- **Unified Output Pattern**: Both elementwise and reductions write to output arrays
- **Explicit Indexing**: All array accesses use explicit `array[index]` notation
- **C-like Syntax**: Familiar imperative style for easier debugging

### Phase 3: SSA v2 Intermediate Representation

SSA (Static Single Assignment) form ensures each variable is assigned exactly once, enabling advanced optimizations:

```
Entry block: BlockId(0)
Block BlockId(0):
  Parameters: SsaValue(0): U64, SsaValue(1): U64, SsaValue(2): U64, SsaValue(3): U64
  Constant { dest: SsaValue(5), value: 0, data_type: I64 }
  GetElementPtr { dest: SsaValue(6), address: SsaValue(0), index: SsaValue(5), element_size: 8 }
  Load { dest: SsaValue(7), address: SsaValue(6), offset: 0, data_type: U64 }
  Jump { target_block: BlockId(1), args: [SsaValue(11)] }
  
Block BlockId(1):  // Loop condition
  Parameters: SsaValue(12): U64
  BinaryOp { dest: SsaValue(14), op: Lt, lhs: SsaValue(12), rhs: SsaValue(3) }
  Branch { condition: SsaValue(14), true_block: BlockId(2), false_block: BlockId(3) }
  
Block BlockId(2):  // Loop body
  Parameters: SsaValue(13): U64
  GetElementPtr { dest: SsaValue(15), address: SsaValue(7), index: SsaValue(13), element_size: 8 }
  Load { dest: SsaValue(16), address: SsaValue(15), offset: 0, data_type: U64 }
  // ... arithmetic operations ...
  Store { address: SsaValue(20), offset: 0, value: SsaValue(19) }
  Jump { target_block: BlockId(1), args: [SsaValue(22)] }
```

**SSA v2 Features:**
- **Block-based Control Flow**: Explicit basic blocks with jumps and branches
- **Static Single Assignment**: Each SSA value assigned exactly once
- **Typed Instructions**: All operations have explicit data types
- **Memory Operations**: Explicit loads/stores with pointer arithmetic

### Phase 4: Cranelift IR Generation

Cranelift IR is a low-level, machine-independent assembly-like representation:

```
function u0:0(i64, i64, i64, i64) apple_aarch64 {
block0(v0: i64, v1: i64, v2: i64, v3: i64):
    v6 = iconst.i64 0
    v7 = iconst.i64 8
    v8 = imul v6, v7
    v9 = iadd v0, v8
    v10 = load.i64 notrap aligned v9
    jump block1(v16)

block1(v4: i64):
    v17 = icmp slt v4, v3
    brif v17, block2(v4), block3

block2(v5: i64):
    v18 = iconst.i64 8
    v19 = imul v5, v18
    v20 = iadd.i64 v10, v19
    v21 = load.i64 notrap aligned v20
    // ... computation ...
    v31 = iadd v5, v30
    jump block1(v31)

block3:
    return
}
```

### Phase 5: Native Machine Code

Cranelift generates platform-specific optimized machine code that executes directly on the CPU with no interpretation overhead.

## Key Simplifications in v3

### 1. Direct Lambda Return Type Usage

**Before v3:** Complex type inference using `coerce_nary_op_types()`
**v3:** Direct extraction from `Lambda.return_type`

```rust
// Old approach
let dio_types: Result<Vec<_>, _> = input_arrays
    .iter()
    .map(|array| arrow_type_to_dio_array(array.data_type()))
    .collect();
let output_type = coerce_nary_op_types(&dio_types)?;

// New approach  
let output_type = match expr {
    Expr::Lambda { return_type, .. } => return_type.clone(),
    _ => return Err(DioError::Runtime("Expected Lambda expression".to_string())),
};
```

### 2. Unified Vector Approach

**Before v3:** Reductions returned scalars, elementwise returned arrays
**v3:** Both write to output arrays (reductions use length-1 arrays)

```rust
// Elementwise: output[i] = a[i] + b[i]
for (i = 0; i < length; i += 1) {
  output[i] = (a[i] + b[i]);
}

// Reduction: output[0] = sum(a)  
for (i = 0; i < length; i += 1) {
  acc = (acc + a[i]);
}
output[0] = acc;  // Unified approach!
```

This unification enables:
- **Consistent Code Generation**: Same patterns for both operation types
- **Simplified Function Signatures**: No special cases for scalar returns
- **Easier Optimization**: Uniform memory access patterns

## Function Execution Model

### Memory Layout

Compiled functions expect this parameter layout:
1. `inputs_ptr: *mut *mut u64` - Pointer to array of input array pointers
2. `input_count: u64` - Number of input arrays
3. `output_ptr: *mut u64` - Pointer to output array buffer
4. `length: u64` - Number of elements to process

### Zero-Copy Semantics

- **Input arrays**: Passed as pointers to existing Arrow array data
- **Output arrays**: Pre-allocated buffers that compiled functions write into
- **No data copying**: All operations work directly on Arrow memory

### Caching Strategy

```rust
// Function signatures are hashed for caching
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FunctionSignature {
    input_types: Vec<Type>,
    output_type: Type, 
    expression_hash: u64,
}

// Cache lookup avoids recompilation
let signature = FunctionSignature::new(dio_types, output_type.clone(), expr);
let code_ptr = {
    let mut cache = FUNCTION_CACHE.lock().unwrap();
    if let Some(ptr) = cache.get(&signature) {
        ptr.as_ptr()  // Cache hit!
    } else {
        // Cache miss - compile and store
        let ssa_program_v2 = ast_to_ssa_v2_via_bytecode(expr)?;
        let mut backend = CraneliftBackend::new()?;
        let code_ptr = backend.compile_v2(&ssa_program_v2)?;
        cache.insert(signature, CompiledFunction::new(code_ptr));
        code_ptr
    }
};
```

## Development Workflow

### Debugging Pipeline Transformations

Enable debug tracing to see transformations at each stage:

```bash
DIO_DEBUG_PIPELINE=1 cargo test test_name -- --nocapture
```

Example output:
```
=== PIPELINE DEBUG ===
--- Input AST ---
(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))

--- ByteCode (C-like) ---
function(u64[] a, u64[] b, u64 length) -> u64[] {
  for (i = 0; i < length; i += 1) {
    output[i] = (a[i] + b[i]);
  }
  return;
}

--- SSA v2 ---
Entry block: BlockId(0)
Block BlockId(0):
  Parameters: SsaValue(0): U64, ...
  
--- Cranelift IR (SSA v2) ---
function u0:0(i64, i64, i64, i64) apple_aarch64 {
  block0(v0: i64, v1: i64, v2: i64, v3: i64):
    ...
}
```

### Testing Strategy

The project includes comprehensive tests at each pipeline stage:

- **Integration Tests**: Full pipeline tests using `execute_generic_bytecode()`
- **Unit Tests**: Individual stage transformations
- **Performance Tests**: Benchmarking against reference implementations

### Build Commands

```bash
# Build the project
cargo build

# Run all tests  
cargo test

# Run specific test with debug output
DIO_DEBUG_PIPELINE=1 cargo test test_generic_execute_binary_u64 -- --nocapture

# Check compilation without building
cargo check
```

## Current Status

### ✅ Working Features

- **Elementwise Operations**: `+`, `-`, `*`, `/` with proper array indexing
- **Debug Pipeline**: Complete visibility into all transformation stages  
- **Type System**: Robust typed Lambda expressions
- **Function Caching**: Avoid recompilation of identical expressions
- **Zero-Copy Execution**: Direct Arrow array pointer passing

### 🚧 Known Issues  

- **Reduction SSA Generation**: Variable scoping across loop boundaries needs phi nodes
- **Limited Type Support**: Currently only U64/I64 integers
- **Error Handling**: Some compilation errors could be more descriptive

### 📋 Future Enhancements

- **Additional Types**: F64 floating point support
- **More Operations**: Comparison operators, conditional expressions
- **Advanced Optimizations**: Loop unrolling, vectorization hints
- **SIMD Support**: Explicit vector instruction generation

## Getting Started

1. **Clone and build** the project:
   ```bash
   git clone <repo-url>
   cd dio
   cargo build
   ```

2. **Run the test suite** to verify everything works:
   ```bash
   cargo test
   ```

3. **Explore with debug output** to understand the pipeline:
   ```bash
   DIO_DEBUG_PIPELINE=1 cargo test test_generic_execute_binary_u64 -- --nocapture
   ```

4. **Study the main API** in `src/execution.rs`:
   ```rust
   pub fn execute_generic_bytecode(
       expr: &Expr,
       input_arrays: &[ArrayRef],
   ) -> Result<ArrayRef, DioError>
   ```

The new pipeline represents a significant architectural improvement, providing better debuggability, more consistent code generation, and a solid foundation for future enhancements.