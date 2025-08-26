# Generic Execution System - Dio v3

This document provides detailed information about Dio's generic execution system, focusing on the new ByteCode pipeline that transforms Lisp expressions into optimized machine code.

## Overview

The generic execution system is the core runtime component of Dio v3, responsible for:

1. **Expression Compilation**: Converting Lisp expressions to native machine code via the ByteCode pipeline
2. **Memory Management**: Zero-copy integration with Apache Arrow arrays  
3. **Function Caching**: Avoiding recompilation through intelligent caching
4. **Type Safety**: Ensuring type correctness throughout the pipeline

## Main Execution Function

### `execute_generic_bytecode()`

The primary entry point for the new pipeline:

```rust
pub fn execute_generic_bytecode(
    expr: &Expr,
    input_arrays: &[ArrayRef],
) -> Result<ArrayRef, DioError>
```

**Parameters:**
- `expr`: A typed Lambda expression (e.g., `(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))`)
- `input_arrays`: Apache Arrow arrays containing the input data

**Returns:**
- `ArrayRef`: Result array containing computed values
- For reductions with scalar return types, returns a length-1 array internally

### Execution Flow

```rust
// 1. Input validation
if input_arrays.is_empty() {
    return Err(DioError::Runtime("Must provide at least one input array".to_string()));
}

// 2. Array length consistency check
let array_length = input_arrays[0].len();
for array in input_arrays.iter().skip(1) {
    if array.len() != array_length {
        return Err(DioError::Runtime("All input arrays must have the same length".to_string()));
    }
}

// 3. Direct type extraction from Lambda (Simplification 1)
let (output_type, output_length) = match expr {
    Expr::Lambda { return_type, body, .. } => {
        // Simplification 2: Treat reductions as length-1 vectors  
        let is_reduction = matches!(**body, Expr::Sum(_) | Expr::Count(_));
        if is_reduction && return_type.is_scalar() {
            let array_type = match return_type {
                Type::U64 => Type::U64Array,
                Type::I64 => Type::I64Array,
                Type::F64 => Type::F64Array,
                _ => return_type.clone(),
            };
            (array_type, 1)  // Length-1 array for reductions
        } else {
            (return_type.clone(), array_length)
        }
    },
    _ => return Err(DioError::Runtime("execute_generic_bytecode expects a Lambda expression".to_string())),
};

// 4. Create output buffer
let output_arrow_type = dio_type_to_arrow(&output_type)?;
let mut output_buffer = create_output_buffer(&output_arrow_type, output_length)?;

// 5. Compile via ByteCode pipeline
let ssa_program_v2 = ast_to_ssa_v2_via_bytecode(expr)?;
let mut backend = CraneliftBackend::new()?;
let code_ptr = backend.compile_v2(&ssa_program_v2)?;

// 6. Execute compiled function
let compiled_fn = CompiledFunction::new(code_ptr);
unsafe {
    compiled_fn.call_nary_op(input_arrays, &mut output_buffer)?;
}

// 7. Return result
buffer_to_array_ref(output_buffer, &output_arrow_type)
```

## Key Simplifications in v3

### Simplification 1: Direct Type Extraction

**Before v3:**
```rust
let dio_types: Result<Vec<_>, _> = input_arrays
    .iter()
    .map(|array| arrow_type_to_dio_array(array.data_type()))
    .collect();
let dio_types = dio_types?;
let output_type = coerce_nary_op_types(&dio_types)?;
```

**v3 Approach:**
```rust
let output_type = match expr {
    Expr::Lambda { return_type, .. } => return_type.clone(),
    _ => return Err(DioError::Runtime("Expected Lambda expression".to_string())),
};
```

**Benefits:**
- **Simpler Logic**: No complex type coercion inference
- **More Direct**: Lambda expressions have explicit return types
- **Less Error-Prone**: Eliminates a source of type mismatches

### Simplification 2: Unified Vector Approach

**Before v3:**
- Elementwise operations: Return arrays of length `n`
- Reductions: Return scalar values
- Different code generation patterns

**v3 Approach:**
- Elementwise operations: Write to `output[i]` for `i = 0..n-1`
- Reductions: Write to `output[0]` (length-1 array)
- Unified code generation pattern

**Implementation:**
```rust
let is_reduction = matches!(**body, Expr::Sum(_) | Expr::Count(_));
if is_reduction && return_type.is_scalar() {
    // Convert scalar return type to array type for internal processing
    let array_type = match return_type {
        Type::U64 => Type::U64Array,
        Type::I64 => Type::I64Array, 
        Type::F64 => Type::F64Array,
        _ => return_type.clone(),
    };
    (array_type, 1)  // Create length-1 output buffer
} else {
    (return_type.clone(), array_length)
}
```

## Pipeline Integration: `ast_to_ssa_v2_via_bytecode()`

This function orchestrates the multi-stage compilation:

```rust
pub fn ast_to_ssa_v2_via_bytecode(expr: &Expr) -> Result<SsaProgramV2, DioError> {
    // Debug tracing
    if std::env::var("DIO_DEBUG_PIPELINE").is_ok() {
        println!("=== PIPELINE DEBUG ===");
        println!("--- Input AST ---");
        println!("{}", expr);
    }

    // Stage 1: AST → ByteCode
    let bytecode = ast_to_bytecode(expr)?;
    if std::env::var("DIO_DEBUG_PIPELINE").is_ok() {
        println!("--- ByteCode (C-like) ---");
        println!("{}", bytecode);
    }

    // Stage 2: ByteCode → SSA v2
    let ssa_program = bytecode_to_ssa_v2(&bytecode)?;
    if std::env::var("DIO_DEBUG_PIPELINE").is_ok() {
        println!("--- SSA v2 ---");
        println!("{}", ssa_program);
        println!("======================");
    }

    Ok(ssa_program)
}
```

## Function Signature and Calling Convention

### Compiled Function Signature

All compiled functions follow this signature:
```rust
fn compiled_function(
    inputs_ptr: *mut *mut u64,    // Pointer to array of input pointers
    input_count: u64,             // Number of input arrays  
    output_ptr: *mut u64,         // Pointer to output buffer
    length: u64,                  // Number of elements to process
)
```

### Parameter Mapping

The SSA v2 generation creates entry block parameters that map to this signature:

```rust
// Create entry block parameters to match Cranelift backend signature
let mut entry_params = Vec::new();
let mut param_mapping = std::collections::HashMap::new();

// 1. inputs*: *mut *mut u64
let inputs_ptr = ssa_program.new_value(SsaDataType::U64);
entry_params.push((inputs_ptr, SsaDataType::U64));
param_mapping.insert("inputs".to_string(), inputs_ptr);

// 2. input_count: u64  
let input_count = ssa_program.new_value(SsaDataType::U64);
entry_params.push((input_count, SsaDataType::U64));
param_mapping.insert("input_count".to_string(), input_count);

// 3. output*: *mut u64 (always needed for unified approach)
let output_ptr = ssa_program.new_value(SsaDataType::U64);
entry_params.push((output_ptr, SsaDataType::U64));
param_mapping.insert("output".to_string(), output_ptr);

// 4. length: u64
let length = ssa_program.new_value(SsaDataType::U64);
entry_params.push((length, SsaDataType::U64));
param_mapping.insert("length".to_string(), length);
```

### Memory Access Patterns

**Input Array Access:**
```rust
// Load array pointer from inputs array
let index_val = /* array index */;
let array_ptr_addr = GetElementPtr {
    address: inputs_ptr,
    index: index_val, 
    element_size: 8  // sizeof(pointer)
};
let array_ptr = Load {
    address: array_ptr_addr,
    data_type: U64
};

// Access element within array
let element_addr = GetElementPtr {
    address: array_ptr,
    index: loop_index,
    element_size: 8  // sizeof(u64)
};
let element_value = Load {
    address: element_addr,
    data_type: U64
};
```

**Output Array Access:**
```rust
// Store result to output array
let output_addr = GetElementPtr {
    address: output_ptr,
    index: loop_index,  // or constant 0 for reductions
    element_size: 8
};
Store {
    address: output_addr,
    value: computed_result
};
```

## Caching System

### Cache Key Generation

Functions are cached using a hash of their signature:

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FunctionSignature {
    input_types: Vec<Type>,
    output_type: Type,
    expression_hash: u64,
}

impl FunctionSignature {
    fn new(input_types: Vec<Type>, output_type: Type, expr: &Expr) -> Self {
        let mut hasher = DefaultHasher::new();
        expr.hash(&mut hasher);
        let expression_hash = hasher.finish();
        
        Self {
            input_types,
            output_type,
            expression_hash,
        }
    }
}
```

### Cache Lookup Strategy

```rust
static FUNCTION_CACHE: LazyLock<Mutex<HashMap<FunctionSignature, CompiledFunction>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// In execute_generic_cached():
let signature = FunctionSignature::new(dio_types, output_type.clone(), expr);
let code_ptr = {
    let mut cache = FUNCTION_CACHE.lock().unwrap();
    if let Some(ptr) = cache.get(&signature) {
        ptr.as_ptr()  // Cache hit - reuse compiled function
    } else {
        // Cache miss - compile and store
        let ssa_program = ast_to_ssa(expr)?;
        let mut backend = CraneliftBackend::new()?;
        let code_ptr = backend.compile(&ssa_program)?;
        let compiled_fn = CompiledFunction::new(code_ptr);
        cache.insert(signature, compiled_fn);
        code_ptr
    }
};
```

## Type System Integration

### Dio Type to Arrow Type Mapping

```rust
pub fn dio_type_to_arrow(dio_type: &Type) -> Result<ArrowDataType, DioError> {
    match dio_type {
        Type::U64 => Ok(ArrowDataType::UInt64),
        Type::U64Array => Ok(ArrowDataType::UInt64),
        Type::I64 => Ok(ArrowDataType::Int64), 
        Type::I64Array => Ok(ArrowDataType::Int64),
        Type::F64 => Ok(ArrowDataType::Float64),
        Type::F64Array => Ok(ArrowDataType::Float64),
    }
}
```

### Buffer Creation

```rust
pub fn create_output_buffer(
    data_type: &ArrowDataType,
    length: usize,
) -> Result<MutableBuffer, DioError> {
    let byte_width = match data_type {
        ArrowDataType::UInt64 | ArrowDataType::Int64 | ArrowDataType::Float64 => 8,
        _ => return Err(DioError::Runtime(format!("Unsupported output type: {:?}", data_type))),
    };
    
    let buffer_size = length * byte_width;
    let buffer = MutableBuffer::with_capacity(buffer_size).with_bitset(buffer_size, false);
    Ok(buffer)
}
```

## Error Handling

### Common Error Cases

1. **Empty Input Arrays**
   ```rust
   if input_arrays.is_empty() {
       return Err(DioError::Runtime("Must provide at least one input array".to_string()));
   }
   ```

2. **Length Mismatches**
   ```rust
   for array in input_arrays.iter().skip(1) {
       if array.len() != array_length {
           return Err(DioError::Runtime("All input arrays must have the same length".to_string()));
       }
   }
   ```

3. **Invalid Expression Type**
   ```rust
   let output_type = match expr {
       Expr::Lambda { return_type, .. } => return_type.clone(),
       _ => return Err(DioError::Runtime("execute_generic_bytecode expects a Lambda expression".to_string())),
   };
   ```

4. **Compilation Failures**
   ```rust
   let ssa_program_v2 = ast_to_ssa_v2_via_bytecode(expr)
       .map_err(|e| DioError::Compilation(format!("ByteCode pipeline failed: {}", e)))?;
   ```

## Performance Characteristics

### Zero-Copy Guarantees

- **Input Data**: Arrow arrays are passed as raw pointers - no copying
- **Output Data**: Results written directly to pre-allocated buffers
- **Compilation**: Cached functions eliminate recompilation overhead

### Memory Layout

```
Input Arrays in Memory:
┌─────────────┬─────────────┬─────────────┐
│   Array A   │   Array B   │   Array C   │  
│ [1,2,3,4,5] │ [6,7,8,9,0] │ [a,b,c,d,e] │
└─────────────┴─────────────┴─────────────┘

Compiled Function Receives:
inputs_ptr → [ptr_to_A, ptr_to_B, ptr_to_C]
output_ptr → [allocated_buffer]

Generated Loop (Elementwise):
for i in 0..length {
    output[i] = input_A[i] + input_B[i] + input_C[i]
}
```

### Optimization Opportunities

- **Loop Vectorization**: Cranelift can auto-vectorize simple loops
- **Bounds Check Elimination**: Controlled loop bounds enable optimization
- **Register Allocation**: Cranelift optimizes register usage for the target architecture
- **Constant Folding**: Known-constant expressions are evaluated at compile time

## Testing and Debugging

### Debug Environment Variables

- `DIO_DEBUG_PIPELINE=1`: Show all pipeline transformations
- `DIO_DEBUG_JIT=1`: Show Cranelift IR generation details
- `DIO_DEBUG_SSA=1`: Show SSA construction details

### Example Test Pattern

```rust
#[test]
fn test_generic_execute_binary_u64() {
    let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
    let a = create_u64_array_from_vec(vec![1, 2, 3, 4, 5]).unwrap();
    let b = create_u64_array_from_vec(vec![10, 20, 30, 40, 50]).unwrap();
    
    let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
    
    let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
    assert_eq!(result_u64.values(), &[11, 22, 33, 44, 55]);
}
```

This comprehensive generic execution system provides the foundation for high-performance columnar expression evaluation while maintaining type safety and zero-copy semantics throughout the pipeline.