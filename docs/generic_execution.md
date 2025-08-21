# Generic Execution Architecture

## Problem Statement

As Dio's type system grows, we face two critical scaling challenges:

1. **Type Explosion**: With N types, binary operations require N² specialized execution functions (execute_add_u64_u64, execute_add_u64_i64, etc.)
2. **Parameter Variance**: Current implementation hardcodes binary operations, but lambda expressions should support arbitrary arity

## Industry Context

### Type Erasure Patterns

**Type erasure** is a fundamental technique in systems programming where concrete types are "erased" behind a common interface, allowing generic code to operate on heterogeneous data without sacrificing performance.

Common approaches:
- **C-style**: `void*` pointers with runtime type tags
- **OOP-style**: Virtual inheritance with dynamic dispatch  
- **Trait objects**: Rust's `Arc<dyn Trait>` pattern (used by Arrow)
- **Template specialization**: Compile-time code generation per type

### Columnar Processing Engines

**DuckDB Approach**:
- Uses C++ templates for compile-time type specialization
- Avoids LLVM/JIT to maintain simplicity and embeddability
- Processes fixed-size vectors (2048 tuples) with specialized kernels
- `UnifiedVectorFormat` provides type-erased interface

**DataFusion Approach**:
- Uses Arrow's `ArrayRef` (Arc<dyn Array>) for type erasure
- Downcast pattern: `array.as_any().downcast_ref::<T>()`
- Exploring JIT with Cranelift for performance-critical paths
- Vectorized execution with specialized kernels per type combination

**Arrow Ecosystem Pattern**:
- `ArrayRef = Arc<dyn Array>` provides reference-counted type erasure
- Runtime type discovery via `DataType` enum matching
- Zero-copy operations through raw pointer extraction
- Generic functions: `fn(inputs: &[ArrayRef]) -> Result<ArrayRef>`

## Dio's Hybrid Approach

### Core Innovation: Type Erasure + JIT Specialization

Unlike DuckDB (pure templates) or DataFusion (pure interpretation), Dio combines:
- **Arrow's type erasure** for interface simplicity
- **Cranelift JIT specialization** for maximum performance
- **Function signature generation** based on lambda type information

### Architecture Overview

```
Lambda Type Info → Function Signature Generation → Cranelift JIT → Cached Function Pointer
       ↓                        ↓                       ↓                    ↓
([U64Array x] [I64Array y])  fn(*const u8, *const u8,  Native Machine     execute_generic()
                             u64, *mut u8)             Code               calls via function pointer
```

## Technical Design

### 1. Array Type Erasure with ArrayRef

```rust
use arrow_array::ArrayRef;
use std::sync::Arc;

pub fn execute_generic(
    expr: &Expr,           // Parsed lambda with type information
    inputs: &[ArrayRef]    // Type-erased Arrow arrays
) -> Result<ArrayRef, DioError>
```

**Benefits**:
- Single function handles all type combinations
- Arrow compatibility and zero-copy operations
- Automatic memory management via `Arc<>`
- Built-in null handling and metadata

### 2. ArrayMetadata Extraction

```rust
#[derive(Debug, Clone)]
pub struct ArrayMetadata {
    pub data_type: arrow_schema::DataType,
    pub length: usize,
    pub data_ptr: *const u8,        // Raw data pointer for Cranelift
    pub null_bitmap: Option<*const u8>, // Null handling (future)
}

impl ArrayMetadata {
    pub fn from_array_ref(array: &ArrayRef) -> Result<Self, DioError> {
        // Extract type, length, and raw data pointer from Arrow array
    }
}
```

### 3. Variadic Function Signature Generation

Instead of fixed binary signatures:
```rust
// Old: Fixed binary
type BinaryFunc = extern "C" fn(*const u64, *const u64, u64, *mut u64);

// New: Variadic with pointer array
type VariadicFunc = extern "C" fn(
    inputs: *const *const u8,  // Array of input pointers
    input_count: u32,          // Number of inputs
    output: *mut u8,           // Output array pointer
    length: u64                // Vector length
);
```

### 4. Optimized Cranelift IR Generation

**Single-Loop Pattern**: The Cranelift backend generates efficient single-loop IR that directly accesses array elements:

```
Desired Pattern: for i in 0..length: out[i] = input[0][i] + input[1][i] + input[2][i] + ...
```

**Generated IR Structure**:
```cranelift
function u0:0(i64, i32, i64, i64) apple_aarch64 {
block0(v0: i64, v1: i32, v2: i64, v3: i64):  // inputs_ptr, input_count, output_ptr, length
    v5 = iconst.i64 0
    jump block1(v5)  ; Initialize loop with i = 0

block1(v4: i64):  // Loop header
    v6 = icmp ult v4, v3  ; i < length
    brif v6, block2, block3  ; if true: loop body, else: exit

block2:  // Loop body: out[i] = input[0][i] + input[1][i] + ...
    v7 = iconst.i64 8
    v8 = imul.i64 v4, v7    ; element_offset = i * 8
    
    ; Load input[0][i]
    v9 = load.i64 notrap aligned v0      ; first_array_ptr = inputs[0]
    v10 = iadd v9, v8                    ; element_addr = array + offset
    v11 = load.i64 notrap aligned v10    ; accumulator = first_element
    
    ; Load input[1][i] and add
    v12 = iconst.i64 8
    v13 = iadd.i64 v0, v12              ; array_ptr_addr = inputs + 8
    v14 = load.i64 notrap aligned v13    ; second_array_ptr = inputs[1]
    v15 = iadd v14, v8                   ; element_addr = array + offset
    v16 = load.i64 notrap aligned v15    ; second_element = input[1][i]
    v17 = iadd v11, v16                  ; accumulator += second_element
    
    ; Load input[2][i] and add (for ternary operations)
    v18 = iconst.i64 16
    v19 = iadd.i64 v0, v18              ; array_ptr_addr = inputs + 16
    v20 = load.i64 notrap aligned v19    ; third_array_ptr = inputs[2]
    v21 = iadd v20, v8                   ; element_addr = array + offset
    v22 = load.i64 notrap aligned v21    ; third_element = input[2][i]
    v23 = iadd v17, v22                  ; accumulator += third_element
    
    ; Store result: out[i] = accumulator
    v24 = iadd.i64 v2, v8               ; output_addr = output + offset
    store notrap aligned v23, v24        ; output[i] = accumulator
    
    v25 = iadd_imm.i64 v4, 1            ; i++
    jump block1(v25)                     ; Continue loop

block3:  // Exit
    return
}
```

**Key Optimizations**:
- **Single loop**: No nested iteration over input arrays
- **Direct element access**: `input[j][i]` pattern with compile-time array indexing
- **Unrolled operations**: Each input array gets dedicated load/compute instructions
- **No runtime loops**: Number of inputs known at compile time

This eliminates the nested loop pattern that was previously generated and provides optimal performance for vectorized operations.

### 5. Function Caching Strategy

```rust
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TypeSignature {
    pub operation: String,           // "add", "multiply", etc.
    pub input_types: Vec<DataType>,  // [U64Array, I64Array]  
    pub output_type: DataType,       // I64Array
}

pub struct FunctionCache {
    cache: HashMap<TypeSignature, *const u8>, // Compiled function pointers
}
```

**Cache Key Generation**:
1. Extract operation from lambda body AST
2. Extract input/output types from lambda signature
3. Hash type signature for O(1) lookup
4. Compile on cache miss, store function pointer

### 5. Runtime Execution Flow

```rust
pub fn execute_generic(expr: &Expr, inputs: &[ArrayRef]) -> Result<ArrayRef, DioError> {
    // 1. Validate inputs match lambda parameter types
    let type_signature = extract_type_signature(expr, inputs)?;
    
    // 2. Check function cache
    let func_ptr = FUNCTION_CACHE.get_or_compile(&type_signature, expr)?;
    
    // 3. Extract raw data pointers from Arrow arrays
    let metadata: Vec<ArrayMetadata> = inputs.iter()
        .map(ArrayMetadata::from_array_ref)
        .collect::<Result<Vec<_>, _>>()?;
    
    // 4. Prepare input pointer array for Cranelift
    let input_ptrs: Vec<*const u8> = metadata.iter()
        .map(|m| m.data_ptr)
        .collect();
    
    // 5. Allocate output array
    let output_array = allocate_output_array(&type_signature.output_type, metadata[0].length)?;
    let output_ptr = extract_data_ptr_mut(&output_array)?;
    
    // 6. Call JIT-compiled function
    unsafe {
        let func: VariadicFunc = std::mem::transmute(func_ptr);
        func(
            input_ptrs.as_ptr(),
            input_ptrs.len() as u32,
            output_ptr,
            metadata[0].length as u64
        );
    }
    
    Ok(output_array)
}
```

## Implementation Status

### ✅ Phase 1: Arrow Integration Foundation (COMPLETED)
- [x] Add arrow dependencies to Cargo.toml
- [x] Create `src/array_support.rs` with ArrayMetadata
- [x] Implement Arrow ↔ raw pointer conversion
- [x] Basic ArrayRef → execute_generic() integration

### ✅ Phase 2: Variadic Lambda Support (COMPLETED)
- [x] Extend parser: `([Type var]*)` instead of fixed arity
- [x] Update AST for variable parameter lists
- [x] Modify SSA IR generation for dynamic parameter counts
- [x] Update type coercion for N-ary operations

### ✅ Phase 3: Generic Cranelift Backend (COMPLETED)
- [x] Implement variadic function signature generation
- [x] Update loop generation for optimized single-loop pattern
- [x] Add input pointer array handling
- [x] Generic operation dispatching (add, subtract)
- [x] Environment variable `DIO_DEBUG_JIT` for IR debugging

### ✅ Phase 4: Function Caching & Performance (COMPLETED)
- [x] Implement TypeSignature hashing
- [x] Add FunctionCache with thread-safe access
- [x] Compile-on-demand with cache warming strategies
- [x] Comprehensive test coverage (78 passing tests)

### 🔄 Phase 5: Advanced Features (FUTURE WORK)
- [ ] Null handling integration with Arrow
- [ ] Memory alignment optimizations
- [ ] SIMD instruction generation hints
- [ ] Error propagation from JIT code
- [ ] Support for additional operations (multiply, divide, etc.)
- [ ] Floating-point array support

## Performance Analysis

### Costs
- **Runtime type checking**: ~5-10 cycles per input array
- **Function pointer indirection**: ~1 cycle overhead  
- **Cache lookup**: O(1) hash table access
- **Arrow metadata extraction**: ~10-20 cycles per array

### Benefits  
- **Eliminated type explosion**: O(1) functions vs O(N²) 
- **Cranelift specialization**: Zero runtime branching in hot loops
- **SIMD-friendly memory layout**: Arrow's alignment preserved
- **Vectorized operations**: Amortizes overhead over 1000s of elements

### Expected Performance
- **Small arrays (<100 elements)**: 10-20% overhead from indirection
- **Medium arrays (1K-10K elements)**: Performance parity
- **Large arrays (>10K elements)**: Potential 5-10% improvement from better cache locality

## Alternative Approaches Considered

### 1. Pure Template Specialization (DuckDB-style)
**Pros**: Zero runtime overhead, compile-time optimization
**Cons**: Binary size explosion, longer compile times, not extensible

### 2. Pure Interpretation (DataFusion-style)  
**Pros**: Simple implementation, good for prototyping
**Cons**: Runtime branching overhead, limited optimization

### 3. LLVM JIT (Traditional approach)
**Pros**: Maximum optimization potential
**Cons**: Large dependency, slower compile times, deployment complexity

### 4. Rust Trait Specialization
**Pros**: Zero-cost abstractions when available
**Cons**: Unstable feature, limited expressiveness

## Risk Mitigation

### Memory Safety
- Arrow arrays provide automatic bounds checking
- Raw pointer access isolated to unsafe blocks
- Function pointer validation before execution

### Type Safety  
- Lambda type signatures validated at parse time
- Runtime type checking before function calls
- Comprehensive error handling for type mismatches

### Performance Regression
- Benchmarking suite comparing all approaches
- Feature flags for fallback to current implementation
- Gradual migration path for existing code

## Success Metrics

1. **Scalability**: Support 10+ types with O(1) function count
2. **Performance**: Within 5% of specialized implementations
3. **Usability**: Single `execute_generic()` API
4. **Extensibility**: Easy addition of new types/operations  
5. **Compatibility**: Full Arrow ecosystem integration

This architecture positions Dio as a unique hybrid system combining the performance benefits of JIT compilation with the simplicity and interoperability of Arrow's columnar format.