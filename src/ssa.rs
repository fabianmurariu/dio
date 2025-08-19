use std::collections::HashMap;
use crate::ast::{Expr, Type, TypedParam};
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

/// Convert typed lambda to SSA IR
/// Supports (lambda ([U64Array x] [U64Array y] U64Array) (+ x y)) format
fn convert_typed_lambda(params: &[TypedParam], return_type: &Type, body: &Expr) -> Result<SsaProgram, DioError> {
    // For the vertical slice, only support specific patterns
    match (params, return_type, body) {
        // Pattern: (lambda ([U64Array x] [U64Array y] U64Array) (+ x y))
        (params, Type::U64Array, Expr::Add(operands)) 
            if params.len() == 2 
            && params.iter().all(|p| matches!(p.type_, Type::U64Array))
            && operands.len() == 2 => {
            
            // Verify the operands match the parameter names
            let param_names: Vec<&str> = params.iter().map(|p| p.name.as_str()).collect();
            if let (Expr::Column(col1), Expr::Column(col2)) = (&operands[0], &operands[1]) {
                if param_names.contains(&col1.as_str()) && param_names.contains(&col2.as_str()) {
                    convert_simple_addition(col1, col2)
                } else {
                    Err(DioError::Compilation("Column references in lambda body must match parameter names".to_string()))
                }
            } else {
                Err(DioError::Compilation("Lambda body must be column addition for vertical slice".to_string()))
            }
        }
        
        // Pattern: (lambda ([U64Array x] U64) (sum x))
        (params, Type::U64, Expr::Sum(inner)) 
            if params.len() == 1 
            && matches!(params[0].type_, Type::U64Array) => {
            
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
            "Only (lambda ([U64Array x] [U64Array y] U64Array) (+ x y)) supported in vertical slice".to_string()
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
    }
}