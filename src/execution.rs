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
    use serde::{Deserialize, Serialize};
    use std::fs;

    #[derive(Deserialize, Serialize, Debug)]
    struct TestConfig {
        exec_tests: Vec<ExecutionTest>,
    }

    #[derive(Deserialize, Serialize, Debug)]
    struct ExecutionTest {
        name: String,
        expr: String,
        inputs: Vec<Vec<i64>>,
        output: Vec<i64>,
    }

    /// Create an ArrayRef from a vector of i64 values based on the type
    fn create_array_from_values(values: Vec<i64>, expected_type: &Type) -> Result<ArrayRef, DioError> {
        match expected_type {
            Type::U64Array => {
                let u64_values: Vec<u64> = values.into_iter().map(|v| v as u64).collect();
                create_u64_array_from_vec(u64_values)
            }
            Type::I64Array => {
                create_i64_array_from_vec(values)
            }
            _ => Err(DioError::Runtime(format!("Unsupported array type: {:?}", expected_type)))
        }
    }

    /// Extract output values from an ArrayRef for comparison
    fn extract_output_values(array: &ArrayRef) -> Vec<i64> {
        if let Some(u64_array) = array.as_any().downcast_ref::<UInt64Array>() {
            u64_array.values().iter().map(|&v| v as i64).collect()
        } else if let Some(i64_array) = array.as_any().downcast_ref::<Int64Array>() {
            i64_array.values().to_vec()
        } else {
            panic!("Unsupported array type for output extraction");
        }
    }

    /// Extract lambda parameter types from an expression
    fn extract_parameter_types(expr: &crate::ast::Expr) -> Vec<Type> {
        match expr {
            crate::ast::Expr::Lambda { params, .. } => {
                params.iter().map(|param| param.type_.clone()).collect()
            }
            _ => panic!("Expected Lambda expression"),
        }
    }

    #[test]
    fn test_json_driven_execution() {
        // Load the JSON test configuration
        let test_data_path = concat!(env!("CARGO_MANIFEST_DIR"), "/test_data/execution_tests.json");
        let json_content = fs::read_to_string(test_data_path)
            .expect("Failed to read test_data/execution_tests.json");
        
        let config: TestConfig = serde_json::from_str(&json_content)
            .expect("Failed to parse JSON test configuration");

        // Run each test
        for test_case in config.exec_tests {
            println!("Running test: {}", test_case.name);
            
            // Parse the expression
            let expr = match parse_expr(&test_case.expr) {
                Ok(e) => e,
                Err(e) => {
                    panic!("Test '{}': Failed to parse expression '{}': {:?}", 
                           test_case.name, test_case.expr, e);
                }
            };

            // Extract parameter types from the lambda
            let param_types = extract_parameter_types(&expr);
            
            // Validate input count matches parameter count
            if test_case.inputs.len() != param_types.len() {
                panic!("Test '{}': Input count {} doesn't match parameter count {}", 
                       test_case.name, test_case.inputs.len(), param_types.len());
            }

            // Create input arrays based on parameter types
            let input_arrays: Result<Vec<_>, _> = test_case.inputs.iter()
                .zip(&param_types)
                .map(|(values, param_type)| create_array_from_values(values.clone(), param_type))
                .collect();
            
            let input_arrays = match input_arrays {
                Ok(arrays) => arrays,
                Err(e) => {
                    panic!("Test '{}': Failed to create input arrays: {:?}", test_case.name, e);
                }
            };

            // Execute the expression
            let result = match execute_generic_bytecode(&expr, &input_arrays) {
                Ok(r) => r,
                Err(e) => {
                    panic!("Test '{}': Execution failed for expr '{}' with inputs {:?}: {:?}", 
                           test_case.name, test_case.expr, test_case.inputs, e);
                }
            };

            // Extract and compare output values
            let actual_output = extract_output_values(&result);
            
            if actual_output != test_case.output {
                // For debugging: print extra info about arrays returning unexpected length
                if actual_output.len() != test_case.output.len() {
                    println!("  Length mismatch - Expected: {}, Got: {} elements", test_case.output.len(), actual_output.len());
                    println!("  Result array type: {:?}", result.data_type());
                    println!("  Result array length: {}", result.len());
                }
                panic!("Test '{}': Output mismatch for expr '{}' with inputs {:?}\n  Expected: {:?}\n  Actual: {:?}", 
                       test_case.name, test_case.expr, test_case.inputs, test_case.output, actual_output);
            }

            println!("✓ Test '{}' passed", test_case.name);
        }
    }

    #[test]
    fn test_generic_execute_mismatched_lengths() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let a = create_u64_array_from_vec(vec![1, 2, 3]).unwrap();
        let b = create_u64_array_from_vec(vec![4, 5]).unwrap();
        assert!(execute_generic_bytecode(&expr, &[a, b]).is_err());
    }
}
