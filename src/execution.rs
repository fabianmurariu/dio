use crate::array_support::{
    buffer_to_array_ref, create_output_buffer, dio_type_to_arrow, ArrayMetadata,
};
use crate::ast::Expr;
use crate::ast::Type;
use crate::casting::coerce_nary_op_types;
use crate::cranelift_backend::CraneliftBackend;
use crate::error::DioError;
use crate::ssa::ast_to_ssa;
use arrow::array::ArrayRef;
use arrow::buffer::MutableBuffer;
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

/// Function signature hash for caching compiled functions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct FunctionSignature {
    /// Input types in order
    input_types: Vec<Type>,
    /// Output type
    output_type: Type,
    /// AST hash (simple string representation for now)
    ast_hash: String,
}

impl FunctionSignature {
    fn new(input_types: Vec<Type>, output_type: Type, expr: &Expr) -> Self {
        Self {
            input_types,
            output_type,
            ast_hash: expr.to_string(), // Simple hash using Display representation
        }
    }
}

/// Wrapper for function pointer that can be safely sent between threads
/// The function pointers from Cranelift are actually safe to share since they point to JIT'd code
#[derive(Debug, Clone, Copy)]
struct SafeFunctionPtr(*const u8);

unsafe impl Send for SafeFunctionPtr {}
unsafe impl Sync for SafeFunctionPtr {}

impl SafeFunctionPtr {
    fn new(ptr: *const u8) -> Self {
        SafeFunctionPtr(ptr)
    }

    fn as_ptr(&self) -> *const u8 {
        self.0
    }
}

/// Global function cache for compiled Cranelift functions
/// Uses type signatures as keys to avoid recompilation of identical functions
static FUNCTION_CACHE: LazyLock<Mutex<HashMap<FunctionSignature, SafeFunctionPtr>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Compiled function wrapper
pub struct CompiledFunction {
    code_ptr: *const u8,
}

impl CompiledFunction {
    pub fn new(code_ptr: *const u8) -> Self {
        Self { code_ptr }
    }

    /// Execute the compiled function with raw u64 arrays (legacy binary interface)
    /// Safety: This is unsafe because we're calling JIT-compiled code
    pub unsafe fn call_u64_add(&self, a: &[u64], b: &[u64]) -> Result<Vec<u64>, DioError> {
        if a.len() != b.len() {
            return Err(DioError::Runtime(
                "Arrays must have the same length".to_string(),
            ));
        }

        let length = a.len();
        if length == 0 {
            return Ok(Vec::new());
        }

        // Prepare output array
        let mut output = vec![0u64; length];

        // Get raw pointers to array data
        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let output_ptr = output.as_mut_ptr();

        // Cast function pointer and call
        type AddFunction = extern "C" fn(*const u64, *const u64, u64, *mut u64);
        let func: AddFunction = std::mem::transmute(self.code_ptr);

        func(a_ptr, b_ptr, length as u64, output_ptr);

        Ok(output)
    }

    /// Execute the compiled function with variadic Arrow arrays
    /// Function signature: fn(inputs: *const *const u8, input_count: u32, output: *mut u8, length: u64)
    /// Safety: This is unsafe because we're calling JIT-compiled code and working with raw pointers
    pub unsafe fn call_nary_add(
        &self,
        input_arrays: &[ArrayRef],
        output_buffer: &mut MutableBuffer,
        array_length: usize,
    ) -> Result<(), DioError> {
        if input_arrays.is_empty() {
            return Err(DioError::Runtime(
                "Must provide at least one input array".to_string(),
            ));
        }

        // Verify all arrays have the same length
        for (i, array) in input_arrays.iter().enumerate() {
            if array.len() != array_length {
                return Err(DioError::Runtime(format!(
                    "Array {} length ({}) doesn't match expected length ({})",
                    i,
                    array.len(),
                    array_length
                )));
            }
        }

        // Extract metadata from all input arrays
        let input_metadata: Result<Vec<_>, _> = input_arrays
            .iter()
            .map(ArrayMetadata::from_array_ref)
            .collect();
        let input_metadata = input_metadata?;

        // Create array of input data pointers
        let input_ptrs: Vec<*const u8> = input_metadata.iter().map(|meta| meta.data_ptr).collect();

        // Get output buffer data pointer
        let output_ptr = output_buffer.as_mut_ptr();

        // Cast function pointer and call with variadic signature
        type NaryAddFunction = extern "C" fn(*const *const u8, u32, *mut u8, u64);
        let func: NaryAddFunction = std::mem::transmute(self.code_ptr);

        func(
            input_ptrs.as_ptr(),       // inputs: *const *const u8
            input_arrays.len() as u32, // input_count: u32
            output_ptr,                // output: *mut u8
            array_length as u64,       // length: u64
        );

        Ok(())
    }
}

/// High-level function to compile and execute (+ a b) expression with U64 arrays
/// Now uses the generic execution system for compatibility
pub fn execute_add_u64(expr: &Expr, a: &[u64], b: &[u64]) -> Result<Vec<u64>, DioError> {
    use crate::array_support::create_u64_array_from_vec;
    use arrow::array::UInt64Array;

    // Convert to Arrow arrays
    let a_array = create_u64_array_from_vec(a.to_vec())?;
    let b_array = create_u64_array_from_vec(b.to_vec())?;

    // Use generic execution
    let result_array = execute_generic(expr, &[a_array, b_array])?;

    // Convert back to Vec<u64>
    let result_u64 = result_array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| DioError::Runtime("Expected UInt64Array result".to_string()))?;

    Ok(result_u64.values().to_vec())
}

/// High-level function to compile and execute (+ a b) expression with I64 arrays
/// Now uses the generic execution system for compatibility
pub fn execute_add_i64(expr: &Expr, a: &[i64], b: &[i64]) -> Result<Vec<i64>, DioError> {
    use crate::array_support::create_i64_array_from_vec;
    use arrow::array::Int64Array;

    // Convert to Arrow arrays
    let a_array = create_i64_array_from_vec(a.to_vec())?;
    let b_array = create_i64_array_from_vec(b.to_vec())?;

    // Use generic execution
    let result_array = execute_generic(expr, &[a_array, b_array])?;

    // Convert back to Vec<i64>
    let result_i64 = result_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DioError::Runtime("Expected Int64Array result".to_string()))?;

    Ok(result_i64.values().to_vec())
}

/// High-level function to compile and execute (+ a b) expression with mixed U64/I64 arrays
/// Returns I64 array as per casting rules (signed takes precedence)
/// Now uses the generic execution system for compatibility
pub fn execute_add_mixed_u64_i64(expr: &Expr, a: &[u64], b: &[i64]) -> Result<Vec<i64>, DioError> {
    use crate::array_support::{create_i64_array_from_vec, create_u64_array_from_vec};
    use arrow::array::Int64Array;

    // Convert to Arrow arrays
    let a_array = create_u64_array_from_vec(a.to_vec())?;
    let b_array = create_i64_array_from_vec(b.to_vec())?;

    // Use generic execution (will coerce to I64Array)
    let result_array = execute_generic(expr, &[a_array, b_array])?;

    // Convert back to Vec<i64>
    let result_i64 = result_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or_else(|| DioError::Runtime("Expected Int64Array result".to_string()))?;

    Ok(result_i64.values().to_vec())
}

/// Generic execute function using Arrow ArrayRef with N-ary operations and type erasure
/// This is the main execution function that should be used for all operations
pub fn execute_generic(expr: &Expr, input_arrays: &[ArrayRef]) -> Result<ArrayRef, DioError> {
    if input_arrays.is_empty() {
        return Err(DioError::Runtime(
            "Must provide at least one input array".to_string(),
        ));
    }

    // Verify all arrays have the same length
    let array_length = input_arrays[0].len();
    for (i, array) in input_arrays.iter().enumerate() {
        if array.len() != array_length {
            return Err(DioError::Runtime(format!(
                "Array {} length ({}) doesn't match expected length ({})",
                i,
                array.len(),
                array_length
            )));
        }
    }

    // Extract Dio types from Arrow arrays for type coercion
    let dio_types: Result<Vec<_>, _> = input_arrays
        .iter()
        .map(|array| crate::array_support::arrow_type_to_dio_array(array.data_type()))
        .collect();
    let dio_types = dio_types?;

    // Determine the output type using N-ary coercion rules
    let output_type = coerce_nary_op_types(&dio_types)?;
    let output_arrow_type = dio_type_to_arrow(&output_type)?;

    // Create output buffer
    let mut output_buffer = create_output_buffer(&output_arrow_type, array_length)?;

    // AST -> SSA IR
    let ssa_program = ast_to_ssa(expr)?;

    // SSA IR -> Cranelift -> Machine Code
    let mut backend = CraneliftBackend::new()?;
    let code_ptr = backend.compile(&ssa_program)?;

    // Execute with Arrow arrays
    let compiled_fn = CompiledFunction::new(code_ptr);
    unsafe {
        compiled_fn.call_nary_add(input_arrays, &mut output_buffer, array_length)?;
    }

    // Convert buffer to ArrayRef
    let result_array = buffer_to_array_ref(output_buffer, &output_arrow_type)?;
    Ok(result_array)
}

/// Cached generic execute function using Arrow ArrayRef with function caching
/// Uses a global cache to avoid recompiling identical function signatures
pub fn execute_generic_cached(
    expr: &Expr,
    input_arrays: &[ArrayRef],
) -> Result<ArrayRef, DioError> {
    if input_arrays.is_empty() {
        return Err(DioError::Runtime(
            "Must provide at least one input array".to_string(),
        ));
    }

    // Verify all arrays have the same length
    let array_length = input_arrays[0].len();
    for (i, array) in input_arrays.iter().enumerate() {
        if array.len() != array_length {
            return Err(DioError::Runtime(format!(
                "Array {} length ({}) doesn't match expected length ({})",
                i,
                array.len(),
                array_length
            )));
        }
    }

    // Extract Dio types from Arrow arrays for type coercion
    let dio_types: Result<Vec<_>, _> = input_arrays
        .iter()
        .map(|array| crate::array_support::arrow_type_to_dio_array(array.data_type()))
        .collect();
    let dio_types = dio_types?;

    // Determine the output type using N-ary coercion rules
    let output_type = coerce_nary_op_types(&dio_types)?;
    let output_arrow_type = dio_type_to_arrow(&output_type)?;

    // Create function signature for cache lookup
    let signature = FunctionSignature::new(dio_types, output_type.clone(), expr);

    // Check cache first
    let code_ptr = {
        let cache = FUNCTION_CACHE.lock().unwrap();
        cache.get(&signature).map(|ptr| ptr.as_ptr())
    };

    let code_ptr = match code_ptr {
        Some(cached_ptr) => {
            // Cache hit - use cached function
            cached_ptr
        }
        None => {
            // Cache miss - compile and cache
            let ssa_program = ast_to_ssa(expr)?;
            let mut backend = CraneliftBackend::new()?;
            let new_code_ptr = backend.compile(&ssa_program)?;

            // Add to cache
            {
                let mut cache = FUNCTION_CACHE.lock().unwrap();
                cache.insert(signature, SafeFunctionPtr::new(new_code_ptr));
            }

            new_code_ptr
        }
    };

    // Create output buffer
    let mut output_buffer = create_output_buffer(&output_arrow_type, array_length)?;

    // Execute with Arrow arrays using cached or newly compiled function
    let compiled_fn = CompiledFunction::new(code_ptr);
    unsafe {
        compiled_fn.call_nary_add(input_arrays, &mut output_buffer, array_length)?;
    }

    // Convert buffer to ArrayRef
    let result_array = buffer_to_array_ref(output_buffer, &output_arrow_type)?;
    Ok(result_array)
}

/// Clear the function cache (useful for testing or memory management)
pub fn clear_function_cache() {
    let mut cache = FUNCTION_CACHE.lock().unwrap();
    cache.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_expr;

    #[test]
    fn test_end_to_end_simple_addition() {
        // Test the complete vertical slice: AST -> SSA -> Cranelift -> Execution
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();

        // Create test arrays
        let a = vec![1, 2, 3, 4, 5];
        let b = vec![10, 20, 30, 40, 50];

        // Execute
        let result = execute_add_u64(&expr, &a, &b).unwrap();

        // Verify results
        let expected = vec![11, 22, 33, 44, 55];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_empty_arrays() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = vec![];
        let b = vec![];

        let result = execute_add_u64(&expr, &a, &b).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_single_element() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = vec![42];
        let b = vec![58];

        let result = execute_add_u64(&expr, &a, &b).unwrap();
        assert_eq!(result, vec![100]);
    }

    #[test]
    fn test_large_arrays() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();

        // Create large test arrays
        let size = 10000;
        let a: Vec<u64> = (0..size).collect();
        let b: Vec<u64> = (0..size).map(|i| i * 2).collect();

        let result = execute_add_u64(&expr, &a, &b).unwrap();

        // Verify first few and last few elements
        assert_eq!(result[0], 0); // 0 + 0 = 0
        assert_eq!(result[1], 3); // 1 + 2 = 3
        assert_eq!(result[2], 6); // 2 + 4 = 6
        assert_eq!(result[9999], 29997); // 9999 + 19998 = 29997
    }

    #[test]
    fn test_mismatched_array_lengths() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = vec![1, 2, 3];
        let b = vec![4, 5];

        let result = execute_add_u64(&expr, &a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_zero_values() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = vec![0, 0, 0];
        let b = vec![1, 2, 3];

        let result = execute_add_u64(&expr, &a, &b).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_max_values() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = vec![u64::MAX / 2, 100];
        let b = vec![u64::MAX / 2, 200];

        let result = execute_add_u64(&expr, &a, &b).unwrap();

        // First element: should be close to u64::MAX (with potential overflow)
        // Second element: should be 300
        assert_eq!(result[1], 300);
    }

    #[test]
    fn test_end_to_end_i64_addition() {
        // Test with I64Array types
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (+ a b))").unwrap();

        let a = vec![-1i64, -2, -3, 4, 5];
        let b = vec![10i64, 20, 30, -40, -50];

        let result = execute_add_i64(&expr, &a, &b).unwrap();

        // Verify results with negative numbers
        let expected = vec![9i64, 18, 27, -36, -45];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_mixed_type_addition() {
        // Test mixed U64Array + I64Array -> I64Array (coercion to signed)
        let expr = parse_expr("(lambda ([U64Array a] [I64Array b] I64Array) (+ a b))").unwrap();

        let a = vec![1u64, 2, 3, 4, 5];
        let b = vec![-1i64, -2, -3, 4, 5];

        let result = execute_add_mixed_u64_i64(&expr, &a, &b).unwrap();

        // Verify results: 1-1=0, 2-2=0, 3-3=0, 4+4=8, 5+5=10
        let expected = vec![0i64, 0, 0, 8, 10];
        assert_eq!(result, expected);
    }

    #[test]
    fn test_i64_large_values() {
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (+ a b))").unwrap();

        let a = vec![i64::MAX / 2, -100];
        let b = vec![i64::MAX / 2, 50];

        let result = execute_add_i64(&expr, &a, &b).unwrap();

        // Second element should be -50
        assert_eq!(result[1], -50);
    }

    #[test]
    fn test_generic_execute_binary_u64() {
        use crate::array_support::create_u64_array_from_vec;
        use arrow::array::UInt64Array;

        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();

        let a = create_u64_array_from_vec(vec![1, 2, 3, 4, 5]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20, 30, 40, 50]).unwrap();

        let result = execute_generic(&expr, &[a, b]).unwrap();

        // Verify result type and values
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let expected = vec![11, 22, 33, 44, 55];
        assert_eq!(result_u64.values(), expected.as_slice());
    }

    #[test]
    fn test_generic_execute_binary_i64() {
        use crate::array_support::create_i64_array_from_vec;
        use arrow::array::Int64Array;

        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (+ a b))").unwrap();

        let a = create_i64_array_from_vec(vec![-1, -2, 3, 4, 5]).unwrap();
        let b = create_i64_array_from_vec(vec![10, 20, -30, 40, 50]).unwrap();

        let result = execute_generic(&expr, &[a, b]).unwrap();

        // Verify result type and values
        let result_i64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        let expected = vec![9, 18, -27, 44, 55];
        assert_eq!(result_i64.values(), expected.as_slice());
    }

    #[test]
    fn test_generic_execute_mixed_types() {
        use crate::array_support::{create_i64_array_from_vec, create_u64_array_from_vec};
        use arrow::array::Int64Array;

        // Mixed U64Array + I64Array should coerce to I64Array
        let expr = parse_expr("(lambda ([U64Array a] [I64Array b] I64Array) (+ a b))").unwrap();

        let a = create_u64_array_from_vec(vec![1, 2, 3, 4, 5]).unwrap();
        let b = create_i64_array_from_vec(vec![-1, -2, -3, 4, 5]).unwrap();

        let result = execute_generic(&expr, &[a, b]).unwrap();

        // Should return I64Array (coerced to signed)
        let result_i64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        let expected = vec![0, 0, 0, 8, 10];
        assert_eq!(result_i64.values(), expected.as_slice());
    }

    #[test]
    fn test_generic_execute_ternary_addition() {
        use crate::array_support::create_u64_array_from_vec;
        use arrow::array::UInt64Array;

        let expr =
            parse_expr("(lambda ([U64Array a] [U64Array b] [U64Array c] U64Array) (+ a b c))")
                .unwrap();

        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20, 30]).unwrap();
        let c = create_u64_array_from_vec(vec![100, 200, 300]).unwrap();

        let result = execute_generic(&expr, &[a, b, c]).unwrap();

        // Verify N-ary addition: 1+10+100=111, 2+20+200=222, 3+30+300=333
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let expected = vec![111, 222, 333];
        assert_eq!(result_u64.values(), expected.as_slice());
    }

    #[test]
    fn test_generic_execute_quaternary_mixed() {
        use crate::array_support::{create_i64_array_from_vec, create_u64_array_from_vec};
        use arrow::array::Int64Array;

        // Mixed types: U64Array + I64Array + U64Array + I64Array -> I64Array
        let expr = parse_expr(
            "(lambda ([U64Array w] [I64Array x] [U64Array y] [I64Array z] I64Array) (+ w x y z))",
        )
        .unwrap();

        let w = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let x = create_i64_array_from_vec(vec![10, 20]).unwrap();
        let y = create_u64_array_from_vec(vec![100, 200]).unwrap();
        let z = create_i64_array_from_vec(vec![1000, 2000]).unwrap();

        let result = execute_generic(&expr, &[w, x, y, z]).unwrap();

        // Should return I64Array (coerced to signed)
        // 1+10+100+1000=1111, 2+20+200+2000=2222
        let result_i64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        let expected = vec![1111, 2222];
        assert_eq!(result_i64.values(), expected.as_slice());
    }

    #[test]
    fn test_generic_execute_empty_arrays() {
        use crate::array_support::create_u64_array_from_vec;
        use arrow::array::UInt64Array;

        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();

        let a = create_u64_array_from_vec(vec![]).unwrap();
        let b = create_u64_array_from_vec(vec![]).unwrap();

        let result = execute_generic(&expr, &[a, b]).unwrap();

        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result_u64.len(), 0);
    }

    #[test]
    fn test_generic_execute_mismatched_lengths() {
        use crate::array_support::create_u64_array_from_vec;

        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();

        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![4, 5]).unwrap();

        let result = execute_generic(&expr, &[a, b]);
        assert!(result.is_err());
    }

    #[test]
    fn test_cached_execute_basic() {
        use crate::array_support::create_u64_array_from_vec;
        use arrow::array::UInt64Array;

        // Clear cache before test
        clear_function_cache();

        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();

        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![4, 5, 6]).unwrap();

        // First call should compile and cache
        let result1 = execute_generic_cached(&expr, &[a.clone(), b.clone()]).unwrap();
        let result1_u64 = result1.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result1_u64.values(), &[5, 7, 9]);

        // Second call should use cache (same result)
        let result2 = execute_generic_cached(&expr, &[a, b]).unwrap();
        let result2_u64 = result2.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result2_u64.values(), &[5, 7, 9]);
    }

    #[test]
    fn test_cached_execute_different_signatures() {
        use crate::array_support::{create_i64_array_from_vec, create_u64_array_from_vec};
        use arrow::array::{Int64Array, UInt64Array};

        clear_function_cache();

        // Two different lambda expressions should cache separately
        let expr1 = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let expr2 = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (+ a b))").unwrap();

        let a_u64 = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let b_u64 = create_u64_array_from_vec(vec![3, 4]).unwrap();
        let a_i64 = create_i64_array_from_vec(vec![-1, -2]).unwrap();
        let b_i64 = create_i64_array_from_vec(vec![3, 4]).unwrap();

        // Execute first signature
        let result1 = execute_generic_cached(&expr1, &[a_u64, b_u64]).unwrap();
        let result1_u64 = result1.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result1_u64.values(), &[4, 6]);

        // Execute second signature (different cache entry)
        let result2 = execute_generic_cached(&expr2, &[a_i64, b_i64]).unwrap();
        let result2_i64 = result2.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result2_i64.values(), &[2, 2]);
    }

    #[test]
    fn test_cached_execute_ternary() {
        use crate::array_support::create_u64_array_from_vec;
        use arrow::array::UInt64Array;

        clear_function_cache();

        let expr =
            parse_expr("(lambda ([U64Array a] [U64Array b] [U64Array c] U64Array) (+ a b c))")
                .unwrap();

        let a = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20]).unwrap();
        let c = create_u64_array_from_vec(vec![100, 200]).unwrap();

        // First execution should compile
        let result1 = execute_generic_cached(&expr, &[a.clone(), b.clone(), c.clone()]).unwrap();
        let result1_u64 = result1.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result1_u64.values(), &[111, 222]);

        // Second execution should use cache
        let result2 = execute_generic_cached(&expr, &[a, b, c]).unwrap();
        let result2_u64 = result2.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result2_u64.values(), &[111, 222]);
    }

    #[test]
    fn test_cache_clear() {
        use crate::array_support::create_u64_array_from_vec;

        clear_function_cache();

        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = create_u64_array_from_vec(vec![1]).unwrap();
        let b = create_u64_array_from_vec(vec![2]).unwrap();

        // Execute to populate cache
        let _result = execute_generic_cached(&expr, &[a.clone(), b.clone()]).unwrap();

        // Clear cache
        clear_function_cache();

        // Should still work (will recompile)
        let result = execute_generic_cached(&expr, &[a, b]).unwrap();
        let result_u64 = result
            .as_any()
            .downcast_ref::<arrow::array::UInt64Array>()
            .unwrap();
        assert_eq!(result_u64.values(), &[3]);
    }
}
