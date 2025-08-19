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

### 4. Function Caching Strategy

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

## Implementation Phases

### Phase 1: Arrow Integration Foundation
- [ ] Add arrow dependencies to Cargo.toml
- [ ] Create `src/array_support.rs` with ArrayMetadata
- [ ] Implement Arrow ↔ raw pointer conversion
- [ ] Basic ArrayRef → execute_generic() integration

### Phase 2: Variadic Lambda Support  
- [ ] Extend parser: `([Type var]*)` instead of fixed arity
- [ ] Update AST for variable parameter lists
- [ ] Modify SSA IR generation for dynamic parameter counts
- [ ] Update type coercion for N-ary operations

### Phase 3: Generic Cranelift Backend
- [ ] Implement variadic function signature generation
- [ ] Update loop generation for N inputs instead of 2
- [ ] Add input pointer array handling
- [ ] Generic operation dispatching (add, multiply, etc.)

### Phase 4: Function Caching & Performance
- [ ] Implement TypeSignature hashing
- [ ] Add FunctionCache with thread-safe access
- [ ] Compile-on-demand with cache warming strategies
- [ ] Performance benchmarks vs current approach

### Phase 5: Advanced Features
- [ ] Null handling integration with Arrow
- [ ] Memory alignment optimizations
- [ ] SIMD instruction generation hints
- [ ] Error propagation from JIT code

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