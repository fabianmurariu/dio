use crate::array_support::{
    buffer_to_array_ref, create_output_buffer, dio_type_to_arrow, ArrayMetadata,
};
use crate::ast::Expr;
use crate::ast::Type;
use crate::casting::coerce_nary_op_types;
use crate::cranelift_backend::CraneliftBackend;
use crate::error::DioError;
use crate::ssa::ast_to_ssa;
use arrow::array::{ArrayRef, Scalar};
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

    /// Executes the JIT-compiled function.
    /// # Safety
    /// This function is unsafe because it calls a raw function pointer from JIT-compiled code.
    /// The caller must ensure the pointer is valid and has the correct n-ary signature.
    pub unsafe fn call_nary_op(
        &self,
        input_arrays: &[ArrayRef],
        output_buffer: &mut MutableBuffer,
    ) -> Result<(), DioError> {
        let array_length = input_arrays[0].len();
        let meta: Result<Vec<_>, _> = input_arrays
            .iter()
            .map(ArrayMetadata::from_array_ref)
            .collect();
        let meta = meta?;
        let ptrs: Vec<*const u8> = meta.iter().map(|m| m.data_ptr).collect();

        type NaryOpFn = extern "C" fn(*const *const u8, u32, *mut u8, u64);
        let func: NaryOpFn = std::mem::transmute(self.code_ptr);

        func(
            ptrs.as_ptr(),
            input_arrays.len() as u32,
            output_buffer.as_mut_ptr(),
            array_length as u64,
        );

        Ok(())
    }

    /// Executes a reduction JIT-compiled function.
    /// # Safety
    /// This function is unsafe because it calls a raw function pointer from JIT-compiled code.
    /// The caller must ensure the pointer is valid and has the correct reduction signature.
    pub unsafe fn call_reduction_op(&self, input_arrays: &[ArrayRef]) -> Result<i64, DioError> {
        let array_length = input_arrays[0].len();
        let meta: Result<Vec<_>, _> = input_arrays
            .iter()
            .map(ArrayMetadata::from_array_ref)
            .collect();
        let meta = meta?;
        let ptrs: Vec<*const u8> = meta.iter().map(|m| m.data_ptr).collect();

        // Reduction function signature: (inputs_ptr, input_count, length) -> result
        type ReductionFn = extern "C" fn(*const *const u8, u32, u64) -> i64;
        let func: ReductionFn = std::mem::transmute(self.code_ptr);

        let result = func(
            ptrs.as_ptr(),
            input_arrays.len() as u32,
            array_length as u64,
        );

        Ok(result)
    }
}

/// Generic execute function using Arrow ArrayRef with N-ary operations and type erasure
///
/// # Deprecated
/// This function is deprecated in favor of `execute_generic_bytecode()` which uses the new
/// Lisp -> ByteCode -> SSA v2 -> Cranelift pipeline. The old pipeline (Lisp -> SSA v1 -> Cranelift)
/// has been superseded by the more maintainable and debuggable ByteCode intermediate representation.
#[deprecated(since = "0.2.0", note = "Use `execute_generic_bytecode()` instead")]
pub fn execute_generic(expr: &Expr, input_arrays: &[ArrayRef]) -> Result<ArrayRef, DioError> {
    if input_arrays.is_empty() {
        return Err(DioError::Runtime(
            "Must provide at least one input array".to_string(),
        ));
    }
    let array_length = input_arrays[0].len();
    for array in input_arrays.iter().skip(1) {
        if array.len() != array_length {
            return Err(DioError::Runtime(
                "All input arrays must have the same length".to_string(),
            ));
        }
    }

    let dio_types: Result<Vec<_>, _> = input_arrays
        .iter()
        .map(|array| crate::array_support::arrow_type_to_dio_array(array.data_type()))
        .collect();
    let dio_types = dio_types?;

    let output_type = coerce_nary_op_types(&dio_types)?;
    let output_arrow_type = dio_type_to_arrow(&output_type)?;

    let mut output_buffer = create_output_buffer(&output_arrow_type, array_length)?;
    let ssa_program = ast_to_ssa(expr)?;
    let mut backend = CraneliftBackend::new()?;
    let code_ptr = backend.compile(&ssa_program)?;

    let compiled_fn = CompiledFunction::new(code_ptr);
    unsafe {
        compiled_fn.call_nary_op(input_arrays, &mut output_buffer)?;
    }

    buffer_to_array_ref(output_buffer, &output_arrow_type)
}

/// Execute function using the new ByteCode pipeline: AST -> ByteCode -> SSA v2 -> Cranelift
pub fn execute_generic_bytecode(
    expr: &Expr,
    input_arrays: &[ArrayRef],
) -> Result<ArrayRef, DioError> {
    if input_arrays.is_empty() {
        return Err(DioError::Runtime(
            "Must provide at least one input array".to_string(),
        ));
    }
    let array_length = input_arrays[0].len();
    for array in input_arrays.iter().skip(1) {
        if array.len() != array_length {
            return Err(DioError::Runtime(
                "All input arrays must have the same length".to_string(),
            ));
        }
    }

    // Simplification 1: Extract output type directly from Lambda return_type
    let (output_type, output_length) = match expr {
        Expr::Lambda {
            return_type, body, ..
        } => {
            // Simplification 2: Treat reductions as length-1 vectors instead of scalars
            let is_reduction = matches!(**body, Expr::Sum(_) | Expr::Count(_));
            if is_reduction && return_type.is_scalar() {
                // For reductions with scalar return type, create a length-1 array internally
                let array_type = match return_type {
                    Type::U64 => Type::U64Array,
                    Type::I64 => Type::I64Array,
                    Type::F64 => Type::F64Array,
                    _ => return_type.clone(),
                };
                (array_type, 1)
            } else {
                (return_type.clone(), array_length)
            }
        }
        _ => {
            return Err(DioError::Runtime(
                "execute_generic_bytecode expects a Lambda expression".to_string(),
            ))
        }
    };
    let output_arrow_type = dio_type_to_arrow(&output_type)?;

    let mut output_buffer = create_output_buffer(&output_arrow_type, output_length)?;

    // Use the new ByteCode pipeline: AST -> ByteCode -> SSA v2
    let ssa_program_v2 = crate::bytecode::ast_to_ssa_v2_via_bytecode(expr)?;
    let mut backend = CraneliftBackend::new()?;
    let code_ptr = backend.compile_v2(&ssa_program_v2)?;

    let compiled_fn = CompiledFunction::new(code_ptr);
    unsafe {
        compiled_fn.call_nary_op(input_arrays, &mut output_buffer)?;
    }

    buffer_to_array_ref(output_buffer, &output_arrow_type)
}

/// Cached generic execute function using the new ByteCode pipeline with function caching
pub fn execute_generic_cached(
    expr: &Expr,
    input_arrays: &[ArrayRef],
) -> Result<ArrayRef, DioError> {
    if input_arrays.is_empty() {
        return Err(DioError::Runtime(
            "Must provide at least one input array".to_string(),
        ));
    }
    let array_length = input_arrays[0].len();
    for array in input_arrays.iter().skip(1) {
        if array.len() != array_length {
            return Err(DioError::Runtime(
                "All input arrays must have the same length".to_string(),
            ));
        }
    }

    // Simplification 1: Extract output type directly from Lambda return_type
    let (output_type, output_length) = match expr {
        Expr::Lambda {
            return_type, body, ..
        } => {
            // Simplification 2: Treat reductions as length-1 vectors instead of scalars
            let is_reduction = matches!(**body, Expr::Sum(_) | Expr::Count(_));
            if is_reduction && return_type.is_scalar() {
                // For reductions with scalar return type, create a length-1 array internally
                let array_type = match return_type {
                    Type::U64 => Type::U64Array,
                    Type::I64 => Type::I64Array,
                    Type::F64 => Type::F64Array,
                    _ => return_type.clone(),
                };
                (array_type, 1)
            } else {
                (return_type.clone(), array_length)
            }
        }
        _ => {
            return Err(DioError::Runtime(
                "execute_generic_cached expects a Lambda expression".to_string(),
            ))
        }
    };
    let output_arrow_type = dio_type_to_arrow(&output_type)?;

    // Create cache signature using extracted types
    let dio_types: Result<Vec<_>, _> = input_arrays
        .iter()
        .map(|array| crate::array_support::arrow_type_to_dio_array(array.data_type()))
        .collect();
    let dio_types = dio_types?;
    let signature = FunctionSignature::new(dio_types, output_type.clone(), expr);

    // Check cache or compile using new ByteCode pipeline
    let code_ptr = {
        let mut cache = FUNCTION_CACHE.lock().unwrap();
        if let Some(ptr) = cache.get(&signature) {
            ptr.as_ptr()
        } else {
            // Use the new ByteCode pipeline: AST -> ByteCode -> SSA v2
            let ssa_program_v2 = crate::bytecode::ast_to_ssa_v2_via_bytecode(expr)?;
            let mut backend = CraneliftBackend::new()?;
            let new_code_ptr = backend.compile_v2(&ssa_program_v2)?;
            cache.insert(signature, SafeFunctionPtr::new(new_code_ptr));
            new_code_ptr
        }
    };

    let mut output_buffer = create_output_buffer(&output_arrow_type, output_length)?;
    let compiled_fn = CompiledFunction::new(code_ptr);
    unsafe {
        compiled_fn.call_nary_op(input_arrays, &mut output_buffer)?;
    }

    buffer_to_array_ref(output_buffer, &output_arrow_type)
}

/// Execute a reduction operation, returning a scalar value
///
/// # Deprecated
/// This function is deprecated in favor of `execute_generic_bytecode()` which treats reductions
/// as length-1 arrays internally, providing a unified code generation approach for both
/// elementwise operations and reductions.
#[deprecated(
    since = "0.2.0",
    note = "Use `execute_generic_bytecode()` with length-1 arrays instead"
)]
pub fn execute_reduction(expr: &Expr, input_arrays: &[ArrayRef]) -> Result<i64, DioError> {
    if input_arrays.is_empty() {
        return Err(DioError::Runtime(
            "Must provide at least one input array".to_string(),
        ));
    }
    let array_length = input_arrays[0].len();
    for array in input_arrays.iter().skip(1) {
        if array.len() != array_length {
            return Err(DioError::Runtime(
                "All input arrays must have the same length".to_string(),
            ));
        }
    }

    let ssa_program = ast_to_ssa(expr)?;
    let mut backend = CraneliftBackend::new()?;
    let code_ptr = backend.compile(&ssa_program)?;

    let compiled_fn = CompiledFunction::new(code_ptr);
    unsafe { compiled_fn.call_reduction_op(input_arrays) }
}

/// Execute a reduction operation with caching
///
/// # Deprecated
/// This function is deprecated in favor of `execute_generic_bytecode()` which treats reductions
/// as length-1 arrays internally, providing a unified code generation approach.
#[deprecated(
    since = "0.2.0",
    note = "Use `execute_generic_bytecode()` with length-1 arrays instead"
)]
pub fn execute_reduction_cached(expr: &Expr, input_arrays: &[ArrayRef]) -> Result<i64, DioError> {
    if input_arrays.is_empty() {
        return Err(DioError::Runtime(
            "Must provide at least one input array".to_string(),
        ));
    }
    let array_length = input_arrays[0].len();
    for array in input_arrays.iter().skip(1) {
        if array.len() != array_length {
            return Err(DioError::Runtime(
                "All input arrays must have the same length".to_string(),
            ));
        }
    }

    let dio_types: Result<Vec<_>, _> = input_arrays
        .iter()
        .map(|array| crate::array_support::arrow_type_to_dio_array(array.data_type()))
        .collect();
    let dio_types = dio_types?;

    // For reductions, the output type should be scalar
    let output_type = if dio_types.iter().all(|t| t.is_i64()) {
        Type::I64
    } else {
        Type::U64
    };

    let signature = FunctionSignature::new(dio_types, output_type, expr);

    let code_ptr = {
        let mut cache = FUNCTION_CACHE.lock().unwrap();
        if let Some(cached_ptr) = cache.get(&signature) {
            cached_ptr.as_ptr()
        } else {
            let ssa_program = ast_to_ssa(expr)?;
            let mut backend = CraneliftBackend::new()?;
            let new_code_ptr = backend.compile(&ssa_program)?;
            cache.insert(signature, SafeFunctionPtr::new(new_code_ptr));
            new_code_ptr
        }
    };

    let compiled_fn = CompiledFunction::new(code_ptr);
    unsafe { compiled_fn.call_reduction_op(input_arrays) }
}

/// Clear the function cache (useful for testing or memory management)
pub fn clear_function_cache() {
    let mut cache = FUNCTION_CACHE.lock().unwrap();
    cache.clear();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::array_support::{create_i64_array_from_vec, create_u64_array_from_vec};
    use crate::parse_expr;
    use arrow::array::{Int64Array, UInt64Array};

    #[test]
    fn test_generic_execute_binary_u64() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3, 4, 5]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20, 30, 40, 50]).unwrap();

        // Use the new ByteCode pipeline: AST -> ByteCode -> SSA v2 -> Cranelift
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result_u64.values(), &[11, 22, 33, 44, 55]);
    }

    #[test]
    fn test_generic_execute_binary_i64() {
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (+ a b))").unwrap();
        let a = create_i64_array_from_vec(vec![-1, -2, 3, 4, 5]).unwrap();
        let b = create_i64_array_from_vec(vec![10, 20, -30, 40, 50]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_i64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result_i64.values(), &[9, 18, -27, 44, 55]);
    }

    #[test]
    fn test_generic_execute_subtraction_i64() {
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (- a b))").unwrap();
        let a = create_i64_array_from_vec(vec![-1, 20, 3, 40, 5]).unwrap();
        let b = create_i64_array_from_vec(vec![10, -2, -30, 4, 50]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_i64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result_i64.values(), &[-11, 22, 33, 36, -45]);
    }

    #[test]
    fn test_generic_execute_mixed_types() {
        let expr = parse_expr("(lambda ([U64Array a] [I64Array b] I64Array) (+ a b))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3, 4, 5]).unwrap();
        let b = create_i64_array_from_vec(vec![-1, -2, -3, 4, 5]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_i64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result_i64.values(), &[0, 0, 0, 8, 10]);
    }

    #[test]
    fn test_generic_execute_empty_arrays() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = create_u64_array_from_vec(vec![]).unwrap();
        let b = create_u64_array_from_vec(vec![]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_generic_execute_mismatched_lengths() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![4, 5]).unwrap();
        assert!(execute_generic_bytecode(&expr, &[a, b]).is_err());
    }

    #[test]
    fn test_cached_execute_basic() {
        clear_function_cache();
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![4, 5, 6]).unwrap();
        let result1 = execute_generic_cached(&expr, &[a.clone(), b.clone()]).unwrap();
        let result1_u64 = result1.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result1_u64.values(), &[5, 7, 9]);
        let result2 = execute_generic_cached(&expr, &[a, b]).unwrap();
        let result2_u64 = result2.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result2_u64.values(), &[5, 7, 9]);
    }

    #[test]
    fn test_cached_execute_different_signatures() {
        clear_function_cache();
        let expr1 = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let expr2 = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (+ a b))").unwrap();
        let a_u64 = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let b_u64 = create_u64_array_from_vec(vec![3, 4]).unwrap();
        let a_i64 = create_i64_array_from_vec(vec![-1, -2]).unwrap();
        let b_i64 = create_i64_array_from_vec(vec![3, 4]).unwrap();
        let result1 = execute_generic_cached(&expr1, &[a_u64, b_u64]).unwrap();
        let result1_u64 = result1.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result1_u64.values(), &[4, 6]);
        let result2 = execute_generic_cached(&expr2, &[a_i64, b_i64]).unwrap();
        let result2_i64 = result2.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result2_i64.values(), &[2, 2]);
    }

    #[test]
    fn test_cache_clear() {
        clear_function_cache();
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = create_u64_array_from_vec(vec![1]).unwrap();
        let b = create_u64_array_from_vec(vec![2]).unwrap();
        let _ = execute_generic_cached(&expr, &[a.clone(), b.clone()]).unwrap();
        clear_function_cache();
        let result = execute_generic_cached(&expr, &[a, b]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result_u64.values(), &[3]);
    }

    #[test]
    fn test_generic_execute_ternary_addition() {
        let expr =
            parse_expr("(lambda ([U64Array a] [U64Array b] [U64Array c] U64Array) (+ a b c))")
                .unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20, 30]).unwrap();
        let c = create_u64_array_from_vec(vec![100, 200, 300]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b, c]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result_u64.values(), &[111, 222, 333]);
    }

    #[test]
    fn test_generic_execute_quaternary_mixed() {
        let expr = parse_expr(
            "(lambda ([U64Array w] [I64Array x] [U64Array y] [I64Array z] I64Array) (+ w x y z))",
        )
        .unwrap();
        let w = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let x = create_i64_array_from_vec(vec![10, 20]).unwrap();
        let y = create_u64_array_from_vec(vec![100, 200]).unwrap();
        let z = create_i64_array_from_vec(vec![1000, 2000]).unwrap();
        let result = execute_generic_bytecode(&expr, &[w, x, y, z]).unwrap();
        let result_i64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result_i64.values(), &[1111, 2222]);
    }

    #[test]
    fn test_cached_execute_ternary() {
        clear_function_cache();
        let expr =
            parse_expr("(lambda ([U64Array a] [U64Array b] [U64Array c] U64Array) (+ a b c))")
                .unwrap();
        let a = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20]).unwrap();
        let c = create_u64_array_from_vec(vec![100, 200]).unwrap();
        let result1 = execute_generic_cached(&expr, &[a.clone(), b.clone(), c.clone()]).unwrap();
        let result1_u64 = result1.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result1_u64.values(), &[111, 222]);
        let result2 = execute_generic_cached(&expr, &[a, b, c]).unwrap();
        let result2_u64 = result2.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result2_u64.values(), &[111, 222]);
    }

    #[test]
    fn test_execute_reduction_sum_single_array() {
        let expr = parse_expr("(lambda ([U64Array a] U64) (sum a))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3, 4]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 10); // 1 + 2 + 3 + 4 = 10
    }

    #[test]
    fn test_execute_reduction_sum_binary_addition() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64) (sum (+ a b)))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20, 30]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 66); // (1+10) + (2+20) + (3+30) = 11 + 22 + 33 = 66
    }

    #[test]
    fn test_execute_reduction_sum_ternary_addition() {
        let expr =
            parse_expr("(lambda ([U64Array a] [U64Array b] [U64Array c] U64) (sum (+ a b c)))")
                .unwrap();
        let a = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20]).unwrap();
        let c = create_u64_array_from_vec(vec![100, 200]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b, c]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 333); // (1+10+100) + (2+20+200) = 111 + 222 = 333
    }

    #[test]
    fn test_execute_reduction_sum_mixed_types() {
        let expr = parse_expr("(lambda ([U64Array a] [I64Array b] I64) (sum (+ a b)))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_i64_array_from_vec(vec![10, 20, 30]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 66); // (1+10) + (2+20) + (3+30) = 11 + 22 + 33 = 66
    }

    #[test]
    fn test_execute_reduction_sum_subtraction() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64) (sum (- a b)))").unwrap();
        let a = create_u64_array_from_vec(vec![10, 20, 30]).unwrap();
        let b = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 54); // (10-1) + (20-2) + (30-3) = 9 + 18 + 27 = 54
    }

    #[test]
    fn test_execute_reduction_cached() {
        clear_function_cache();
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64) (sum (+ a b)))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20]).unwrap();

        // First call
        let result1 = execute_generic_cached(&expr, &[a.clone(), b.clone()]).unwrap();
        let result1_u64 = result1.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result1 = result1_u64.value(0);
        assert_eq!(result1, 33); // (1+10) + (2+20) = 11 + 22 = 33

        // Second call should use cached version
        let result2 = execute_generic_cached(&expr, &[a, b]).unwrap();
        let result2_u64 = result2.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result2 = result2_u64.value(0);
        assert_eq!(result2, 33);
    }

    #[test]
    fn test_execute_reduction_empty_arrays() {
        let expr = parse_expr("(lambda ([U64Array a] U64) (sum a))").unwrap();
        let a = create_u64_array_from_vec(vec![]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 0); // Sum of empty array is 0
    }

    #[test]
    fn test_generic_execute_subtraction_u64() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (- a b))").unwrap();
        let a = create_u64_array_from_vec(vec![10, 20, 30, 40, 50]).unwrap();
        let b = create_u64_array_from_vec(vec![1, 2, 3, 4, 5]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        assert_eq!(result_u64.values(), &[9, 18, 27, 36, 45]);
    }

    #[test]
    fn test_generic_execute_subtraction_mixed_types() {
        let expr = parse_expr("(lambda ([U64Array a] [I64Array b] I64Array) (- a b))").unwrap();
        let a = create_u64_array_from_vec(vec![10, 20, 30]).unwrap();
        let b = create_i64_array_from_vec(vec![-5, 15, -10]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_i64 = result.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result_i64.values(), &[15, 5, 40]); // 10-(-5)=15, 20-15=5, 30-(-10)=40
    }

    #[test]
    fn test_cached_execute_subtraction() {
        clear_function_cache();
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (- a b))").unwrap();
        let a = create_i64_array_from_vec(vec![100, 50, 0]).unwrap();
        let b = create_i64_array_from_vec(vec![25, 60, -10]).unwrap();
        let result1 = execute_generic_cached(&expr, &[a.clone(), b.clone()]).unwrap();
        let result1_i64 = result1.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result1_i64.values(), &[75, -10, 10]); // 100-25=75, 50-60=-10, 0-(-10)=10
                                                          // Second call should use cache
        let result2 = execute_generic_cached(&expr, &[a, b]).unwrap();
        let result2_i64 = result2.as_any().downcast_ref::<Int64Array>().unwrap();
        assert_eq!(result2_i64.values(), &[75, -10, 10]);
    }

    #[test]
    fn test_execute_let_binding_single_variable() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64) (let [U64 s (sum a) U64 t (sum b)] (+ s t)))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20, 30]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 66); // sum(a) + sum(b) = 6 + 60 = 66
    }

    #[test]
    fn test_execute_let_binding_multiple_variables() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64) (let [U64 s (sum a) U64 c (count b)] (+ s c)))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20, 30]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 9); // sum(a) + count(b) = 6 + 3 = 9
    }

    #[test]
    fn test_execute_let_binding_with_complex_expression() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] [U64Array c] U64) (let [U64 s (sum (+ a b)) U64 t (sum c)] (+ s t)))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2]).unwrap();
        let b = create_u64_array_from_vec(vec![10, 20]).unwrap();
        let c = create_u64_array_from_vec(vec![100, 200]).unwrap();
        let result = execute_generic_bytecode(&expr, &[a, b, c]).unwrap();
        let result_u64 = result.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result = result_u64.value(0);
        assert_eq!(result, 333); // sum(a+b) + sum(c) = (11+22) + (100+200) = 33 + 300 = 333
    }

    #[test]
    fn test_execute_let_binding_cached() {
        clear_function_cache();
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64) (let [U64 s (sum a) U64 c (count b)] (+ s c)))").unwrap();
        let a = create_u64_array_from_vec(vec![5, 10]).unwrap();
        let b = create_u64_array_from_vec(vec![1, 2]).unwrap();
        
        // First call
        let result1 = execute_generic_cached(&expr, &[a.clone(), b.clone()]).unwrap();
        let result1_u64 = result1.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result1 = result1_u64.value(0);
        assert_eq!(result1, 17); // sum(a) + count(b) = 15 + 2 = 17
        
        // Second call should use cached version
        let result2 = execute_generic_cached(&expr, &[a, b]).unwrap();
        let result2_u64 = result2.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result2 = result2_u64.value(0);
        assert_eq!(result2, 17);
    }
}
