use std::collections::HashMap;
use crate::ast::{Expr, Type, TypedParam};
use crate::casting::coerce_nary_op_types;
use crate::error::DioError;

/// SSA Value identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaValue(pub u32);

/// Basic Block identifier  
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

/// Simplified data types for our vertical slice
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    U64,
    ArrayU64,
}

/// SSA Instructions - simplified for (+ a b) vertical slice
#[derive(Debug, Clone)]
pub enum SsaInstruction {
    /// Load array parameter: %0 = LoadArrayParam { param_index: 0 }
    LoadArrayParam { dest: SsaValue, param_index: u32, data_type: DataType },
    
    /// Load length parameter: %1 = LoadLengthParam { param_index: 2 }  
    LoadLengthParam { dest: SsaValue, param_index: u32 },
    
    /// Load scalar constant: %2 = LoadScalar { value: 0 }
    LoadScalar { dest: SsaValue, value: u64 },
    
    /// Array element access: %3 = ArrayAccess { array: %0, index: %loop_var }
    ArrayAccess { dest: SsaValue, array: SsaValue, index: SsaValue },
    
    /// Addition: %4 = Add { lhs: %3, rhs: %5 }
    Add { dest: SsaValue, lhs: SsaValue, rhs: SsaValue },
    
    /// Store to output array: StoreArrayElement { array: %output, index: %loop_var, value: %4 }
    StoreArrayElement { array: SsaValue, index: SsaValue, value: SsaValue },
    
    /// Loop: Loop { index_var: %loop_var, start: %start, end: %length, body: block1 }
    Loop { index_var: SsaValue, start: SsaValue, end: SsaValue, body: BlockId },
    
    /// Jump to another block: Jump { target: block0 }
    Jump { target: BlockId },
    
    /// Return: Return { value: %output_array }
    Return { value: Option<SsaValue> },
}

/// Basic block in SSA form
#[derive(Debug, Clone)]
pub struct SsaBlock {
    pub id: BlockId,
    pub instructions: Vec<SsaInstruction>,
}

/// Complete SSA program
#[derive(Debug, Clone)]
pub struct SsaProgram {
    pub blocks: HashMap<BlockId, SsaBlock>,
    pub entry_block: BlockId,
    pub value_types: HashMap<SsaValue, DataType>,
    pub next_value_id: u32,
    pub next_block_id: u32,
}

impl SsaProgram {
    pub fn new() -> Self {
        Self {
            blocks: HashMap::new(),
            entry_block: BlockId(0),
            value_types: HashMap::new(),
            next_value_id: 0,
            next_block_id: 0,
        }
    }
    
    pub fn new_value(&mut self, data_type: DataType) -> SsaValue {
        let value = SsaValue(self.next_value_id);
        self.next_value_id += 1;
        self.value_types.insert(value, data_type);
        value
    }
    
    pub fn new_block(&mut self) -> BlockId {
        let block_id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        self.blocks.insert(block_id, SsaBlock {
            id: block_id,
            instructions: Vec::new(),
        });
        block_id
    }
    
    pub fn add_instruction(&mut self, block_id: BlockId, instruction: SsaInstruction) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.instructions.push(instruction);
        }
    }
}

/// Convert AST to SSA IR - now supports typed lambda expressions
pub fn ast_to_ssa(expr: &Expr) -> Result<SsaProgram, DioError> {
    match expr {
        // Handle typed lambda expressions
        Expr::Lambda { params, return_type, body } => {
            convert_typed_lambda(params, return_type, body)
        }
        // Legacy support for bare addition (for backwards compatibility)
        Expr::Add(operands) if operands.len() == 2 => {
            // Only support (+ a b) for now
            if let (Expr::Column(col_a), Expr::Column(col_b)) = (&operands[0], &operands[1]) {
                convert_simple_addition(col_a, col_b)
            } else {
                Err(DioError::Compilation("Only column addition supported in vertical slice".to_string()))
            }
        }
        _ => Err(DioError::Compilation("Only typed lambda expressions supported. Use: (lambda ([U64Array x] [U64Array y] U64Array) (+ x y))".to_string()))
    }
}

/// Convert (+ a b) to SSA IR
/// Generated function signature: fn(a_ptr: *const u64, b_ptr: *const u64, length: u64, output_ptr: *mut u64)
fn convert_simple_addition(_col_a: &str, _col_b: &str) -> Result<SsaProgram, DioError> {
    let mut program = SsaProgram::new();
    
    // Create blocks
    let entry_block = program.new_block();
    let loop_body = program.new_block();
    let exit_block = program.new_block();
    program.entry_block = entry_block;
    
    // Entry block: load parameters and setup loop
    // Function params: (a_ptr: *const u64, b_ptr: *const u64, length: u64, output_ptr: *mut u64)
    let a_array = program.new_value(DataType::ArrayU64);      // %0 = param 0 (a_ptr)
    let b_array = program.new_value(DataType::ArrayU64);      // %1 = param 1 (b_ptr) 
    let length = program.new_value(DataType::U64);            // %2 = param 2 (length)
    let output_array = program.new_value(DataType::ArrayU64); // %3 = param 3 (output_ptr)
    
    let loop_start = program.new_value(DataType::U64);        // %4 = 0
    let loop_var = program.new_value(DataType::U64);          // %5 = loop variable
    
    program.add_instruction(entry_block, SsaInstruction::LoadArrayParam {
        dest: a_array,
        param_index: 0,
        data_type: DataType::ArrayU64,
    });
    
    program.add_instruction(entry_block, SsaInstruction::LoadArrayParam {
        dest: b_array,
        param_index: 1,
        data_type: DataType::ArrayU64,
    });
    
    program.add_instruction(entry_block, SsaInstruction::LoadLengthParam {
        dest: length,
        param_index: 2,
    });
    
    program.add_instruction(entry_block, SsaInstruction::LoadArrayParam {
        dest: output_array,
        param_index: 3,
        data_type: DataType::ArrayU64,
    });
    
    program.add_instruction(entry_block, SsaInstruction::LoadScalar {
        dest: loop_start,
        value: 0,
    });
    
    program.add_instruction(entry_block, SsaInstruction::Loop {
        index_var: loop_var,
        start: loop_start,
        end: length,
        body: loop_body,
    });
    
    program.add_instruction(entry_block, SsaInstruction::Jump {
        target: exit_block,
    });
    
    // Loop body: a[i] + b[i] -> output[i]
    let a_element = program.new_value(DataType::U64);         // %6 = a[loop_var]
    let b_element = program.new_value(DataType::U64);         // %7 = b[loop_var]
    let sum = program.new_value(DataType::U64);               // %8 = %6 + %7
    
    program.add_instruction(loop_body, SsaInstruction::ArrayAccess {
        dest: a_element,
        array: a_array,
        index: loop_var,
    });
    
    program.add_instruction(loop_body, SsaInstruction::ArrayAccess {
        dest: b_element,
        array: b_array,
        index: loop_var,
    });
    
    program.add_instruction(loop_body, SsaInstruction::Add {
        dest: sum,
        lhs: a_element,
        rhs: b_element,
    });
    
    program.add_instruction(loop_body, SsaInstruction::StoreArrayElement {
        array: output_array,
        index: loop_var,
        value: sum,
    });
    
    program.add_instruction(loop_body, SsaInstruction::Jump {
        target: entry_block, // Continue loop
    });
    
    // Exit block: return
    program.add_instruction(exit_block, SsaInstruction::Return {
        value: None, // void return - output written to output_ptr
    });
    
    Ok(program)
}

/// Convert N-ary addition to SSA IR: (+ a b c d ...)
/// Generated function signature: fn(inputs: *const *const u8, input_count: u32, length: u64, output: *mut u8)
fn convert_nary_addition(_column_names: &[&str], operand_count: usize) -> Result<SsaProgram, DioError> {
    // For now, we'll create a simplified SSA program that mirrors the binary case
    // but with knowledge of the operand count for the Cranelift backend
    let mut program = SsaProgram::new();
    
    // Create blocks similar to binary case
    let entry_block = program.new_block();
    let loop_body = program.new_block();
    let exit_block = program.new_block();
    program.entry_block = entry_block;
    
    // For the initial implementation, we'll use the same structure as binary addition
    // The Cranelift backend will be responsible for handling the variable number of inputs
    
    // Entry block: load parameters and setup loop
    let inputs_ptr = program.new_value(DataType::ArrayU64);   // Array of input pointers
    let input_count_val = program.new_value(DataType::U64);   // Number of inputs
    let length = program.new_value(DataType::U64);            // Array length  
    let output_array = program.new_value(DataType::ArrayU64); // Output array
    let loop_start = program.new_value(DataType::U64);        // Loop start (0)
    let loop_var = program.new_value(DataType::U64);          // Loop variable
    
    // Load function parameters
    program.add_instruction(entry_block, SsaInstruction::LoadArrayParam {
        dest: inputs_ptr,
        param_index: 0,
        data_type: DataType::ArrayU64,
    });
    
    program.add_instruction(entry_block, SsaInstruction::LoadLengthParam {
        dest: input_count_val,
        param_index: 1,
    });
    
    program.add_instruction(entry_block, SsaInstruction::LoadLengthParam {
        dest: length,
        param_index: 2,
    });
    
    program.add_instruction(entry_block, SsaInstruction::LoadArrayParam {
        dest: output_array,
        param_index: 3,
        data_type: DataType::ArrayU64,
    });
    
    program.add_instruction(entry_block, SsaInstruction::LoadScalar {
        dest: loop_start,
        value: 0,
    });
    
    // Set up loop
    program.add_instruction(entry_block, SsaInstruction::Loop {
        index_var: loop_var,
        start: loop_start,
        end: length,
        body: loop_body,
    });
    
    program.add_instruction(entry_block, SsaInstruction::Jump {
        target: exit_block,
    });
    
    // Loop body: The Cranelift backend will handle N-ary addition using the operand_count
    // For now, we create a special SSA instruction that encodes the operand count
    let sum_result = program.new_value(DataType::U64);
    
    // This is a placeholder instruction that the Cranelift backend will recognize
    // and generate appropriate code for N-ary addition
    program.add_instruction(loop_body, SsaInstruction::LoadScalar {
        dest: sum_result,
        value: operand_count as u64, // Encode operand count for Cranelift
    });
    
    program.add_instruction(loop_body, SsaInstruction::StoreArrayElement {
        array: output_array,
        index: loop_var,
        value: sum_result,
    });
    
    program.add_instruction(loop_body, SsaInstruction::Jump {
        target: entry_block, // Continue loop
    });
    
    // Exit block: return
    program.add_instruction(exit_block, SsaInstruction::Return {
        value: None,
    });
    
    Ok(program)
}

/// Convert typed lambda to SSA IR
/// Supports integer array addition with automatic type coercion
fn convert_typed_lambda(params: &[TypedParam], return_type: &Type, body: &Expr) -> Result<SsaProgram, DioError> {
    match (params, return_type, body) {
        // Pattern: (lambda ([IntArray x] [IntArray y] [IntArray z]... IntArray) (+ x y z...))
        // where IntArray is U64Array or I64Array
        (params, return_type, Expr::Add(operands)) 
            if params.len() >= 2 
            && operands.len() == params.len() => {
            
            // Check that all parameters are integer array types
            if !params.iter().all(|p| p.type_.is_integer() && p.type_.is_array()) {
                return Err(DioError::Compilation("Only integer array types supported in vertical slice".to_string()));
            }
            
            // N-ary type coercion for the operation
            let param_types: Vec<Type> = params.iter().map(|p| p.type_.clone()).collect();
            let coerced_type = coerce_nary_op_types(&param_types)?;
            
            // Check return type matches coerced type
            if return_type != &coerced_type {
                return Err(DioError::TypeMismatch {
                    expected: coerced_type.to_string(),
                    found: return_type.to_string(),
                    context: "Return type must match the coerced type of the operation".to_string(),
                });
            }
            
            // Verify all operands are column references matching parameter names
            let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
            let mut column_names = Vec::new();
            
            for operand in operands {
                if let Expr::Column(col_name) = operand {
                    if param_names.contains(&col_name.as_str()) {
                        column_names.push(col_name.as_str());
                    } else {
                        return Err(DioError::Compilation("Column references in lambda body must match parameter names".to_string()));
                    }
                } else {
                    return Err(DioError::Compilation("Lambda body must contain only column references for vertical slice".to_string()));
                }
            }
            
            // For now, delegate to the appropriate conversion function based on operand count
            match params.len() {
                2 => convert_simple_addition(&column_names[0], &column_names[1]),
                n => convert_nary_addition(&column_names, n),
            }
        }
        
        // Pattern: (lambda ([IntArray x] Int) (sum x)) - Future extension
        (params, return_type, Expr::Sum(inner)) 
            if params.len() == 1 
            && params[0].type_.is_integer() && params[0].type_.is_array()
            && return_type.is_integer() && return_type.is_scalar() => {
            
            if let Expr::Column(col) = inner.as_ref() {
                if col == &params[0].name {
                    // For now, sum reductions are not implemented in the vertical slice
                    Err(DioError::Compilation("Sum reductions not implemented in vertical slice".to_string()))
                } else {
                    Err(DioError::Compilation("Column reference in lambda body must match parameter name".to_string()))
                }
            } else {
                Err(DioError::Compilation("Lambda body must be simple column reference for sum".to_string()))
            }
        }
        
        _ => Err(DioError::Compilation(
            "Only integer array addition supported: (lambda ([U64Array|I64Array x] [U64Array|I64Array y] I64Array) (+ x y))".to_string()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse_expr;

    #[test]
    fn test_typed_lambda_addition_ast_to_ssa() {
        let expr = parse_expr("(lambda ([U64Array x] [U64Array y] U64Array) (+ x y))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        
        // Should have 3 blocks: entry, loop_body, exit
        assert_eq!(ssa_program.blocks.len(), 3);
        
        // Should have entry block
        assert!(ssa_program.blocks.contains_key(&ssa_program.entry_block));
        
        // Entry block should have parameter loads and loop setup
        let entry_block = &ssa_program.blocks[&ssa_program.entry_block];
        assert!(!entry_block.instructions.is_empty());
        
        // Should have proper value types
        assert!(!ssa_program.value_types.is_empty());
    }

    #[test]
    fn test_typed_lambda_i64_arrays() {
        let expr = parse_expr("(lambda ([I64Array x] [I64Array y] I64Array) (+ x y))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        
        // Should compile successfully
        assert_eq!(ssa_program.blocks.len(), 3);
        assert!(ssa_program.blocks.contains_key(&ssa_program.entry_block));
    }

    #[test]
    fn test_typed_lambda_mixed_types() {
        // Mixed U64Array + I64Array should coerce to I64Array
        let expr = parse_expr("(lambda ([U64Array x] [I64Array y] I64Array) (+ x y))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        
        // Should compile successfully
        assert_eq!(ssa_program.blocks.len(), 3);
        assert!(ssa_program.blocks.contains_key(&ssa_program.entry_block));
    }

    #[test]
    fn test_simple_addition_ast_to_ssa() {
        let expr = parse_expr("(+ a b)").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        
        // Should have 3 blocks: entry, loop_body, exit
        assert_eq!(ssa_program.blocks.len(), 3);
        
        // Should have entry block
        assert!(ssa_program.blocks.contains_key(&ssa_program.entry_block));
        
        // Entry block should have parameter loads and loop setup
        let entry_block = &ssa_program.blocks[&ssa_program.entry_block];
        assert!(!entry_block.instructions.is_empty());
        
        // Should have proper value types
        assert!(!ssa_program.value_types.is_empty());
    }
    
    #[test]
    fn test_typed_lambda_ternary_addition() {
        let expr = parse_expr("(lambda ([U64Array x] [U64Array y] [U64Array z] U64Array) (+ x y z))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        
        // Should compile successfully with 3 operands
        assert_eq!(ssa_program.blocks.len(), 3);
        assert!(ssa_program.blocks.contains_key(&ssa_program.entry_block));
    }

    #[test]
    fn test_typed_lambda_quaternary_mixed_types() {
        // Mixed types: U64Array + I64Array + U64Array + I64Array should coerce to I64Array
        let expr = parse_expr("(lambda ([U64Array w] [I64Array x] [U64Array y] [I64Array z] I64Array) (+ w x y z))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        
        // Should compile successfully with mixed types
        assert_eq!(ssa_program.blocks.len(), 3);
        assert!(ssa_program.blocks.contains_key(&ssa_program.entry_block));
    }

    #[test]
    fn test_unsupported_expressions() {
        // Should reject non-lambda expressions (now required)
        let expr = parse_expr("(- a b)").unwrap();
        assert!(ast_to_ssa(&expr).is_err());
        
        // Should reject invalid lambda types
        let expr = parse_expr("(lambda ([F64Array x] [U64Array y] U64Array) (+ x y))").unwrap();
        assert!(ast_to_ssa(&expr).is_err());
        
        // Should reject lambda with wrong parameter count
        let expr = parse_expr("(lambda ([U64Array x] [U64Array y] [U64Array z] U64Array) (+ x y))").unwrap();
        assert!(ast_to_ssa(&expr).is_err());
        
        // Should reject literals in lambda body
        let expr = parse_expr("(lambda ([U64Array x] [U64Array y] U64Array) (+ x 42))").unwrap();
        assert!(ast_to_ssa(&expr).is_err());
        
        // Should reject mismatched parameter names
        let expr = parse_expr("(lambda ([U64Array x] [U64Array y] U64Array) (+ a b))").unwrap();
        assert!(ast_to_ssa(&expr).is_err());
        
        // Should reject sum (not implemented in vertical slice)
        let expr = parse_expr("(lambda ([U64Array x] U64) (sum x))").unwrap();
        assert!(ast_to_ssa(&expr).is_err());
        
        // Should reject wrong return type for mixed coercion
        let expr = parse_expr("(lambda ([U64Array x] [I64Array y] U64Array) (+ x y))").unwrap();
        assert!(ast_to_ssa(&expr).is_err()); // Should be I64Array, not U64Array
        
        // Should reject float types (not implemented)
        let expr = parse_expr("(lambda ([F64Array x] [F64Array y] F64Array) (+ x y))").unwrap();
        assert!(ast_to_ssa(&expr).is_err());
    }
}