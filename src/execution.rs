use crate::ast::Expr;
use crate::ssa::ast_to_ssa;
use crate::cranelift_backend::CraneliftBackend;
use crate::error::DioError;

/// Compiled function wrapper
pub struct CompiledFunction {
    code_ptr: *const u8,
}

impl CompiledFunction {
    pub fn new(code_ptr: *const u8) -> Self {
        Self { code_ptr }
    }
    
    /// Execute the compiled function with raw u64 arrays
    /// Safety: This is unsafe because we're calling JIT-compiled code
    pub unsafe fn call_u64_add(
        &self,
        a: &[u64],
        b: &[u64],
    ) -> Result<Vec<u64>, DioError> {
        if a.len() != b.len() {
            return Err(DioError::Runtime("Arrays must have the same length".to_string()));
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
}

/// High-level function to compile and execute (+ a b) expression
pub fn execute_add_u64(
    expr: &Expr,
    a: &[u64],
    b: &[u64],
) -> Result<Vec<u64>, DioError> {
    // AST -> SSA IR
    let ssa_program = ast_to_ssa(expr)?;
    
    // SSA IR -> Cranelift -> Machine Code
    let mut backend = CraneliftBackend::new()?;
    let code_ptr = backend.compile(&ssa_program)?;
    
    // Execute with raw arrays
    let compiled_fn = CompiledFunction::new(code_ptr);
    unsafe { compiled_fn.call_u64_add(a, b) }
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
        assert_eq!(result[0], 0);      // 0 + 0 = 0
        assert_eq!(result[1], 3);      // 1 + 2 = 3  
        assert_eq!(result[2], 6);      // 2 + 4 = 6
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
}