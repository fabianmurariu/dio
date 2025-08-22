use crate::ast::{Expr, Type, TypedParam};
use crate::casting::coerce_nary_op_types;
use crate::error::DioError;
use std::collections::HashMap;

/// SSA Value identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaValue(pub u32);

/// Basic Block identifier
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct BlockId(pub u32);

/// Simplified data types for our vertical slice
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    U64,
    ArrayU64,
    I64,
    ArrayI64,
    Bool,  // For comparison results and branch conditions
}

/// SSA Instructions
#[derive(Debug, Clone)]
pub enum SsaInstruction {
    // Data operations
    LoadArrayParam { dest: SsaValue, param_index: u32, data_type: DataType },
    LoadLengthParam { dest: SsaValue, param_index: u32 },
    LoadConstant { dest: SsaValue, value: i64 },
    ArrayAccess { dest: SsaValue, array: SsaValue, index: SsaValue },
    Add { dest: SsaValue, lhs: SsaValue, rhs: SsaValue },
    Sub { dest: SsaValue, lhs: SsaValue, rhs: SsaValue },
    StoreArrayElement { array: SsaValue, index: SsaValue, value: SsaValue },
    
    // Comparison operations
    LessThan { dest: SsaValue, lhs: SsaValue, rhs: SsaValue },
    
    // Control flow
    Jump { target: BlockId },
    Branch { condition: SsaValue, true_target: BlockId, false_target: BlockId },
    Return { value: Option<SsaValue> },
}

/// Basic block in SSA form
#[derive(Debug, Clone)]
pub struct SsaBlock {
    pub id: BlockId,
    pub instructions: Vec<SsaInstruction>,
}

/// Complete SSA program
#[derive(Debug, Clone, Default)]
pub struct SsaProgram {
    pub blocks: HashMap<BlockId, SsaBlock>,
    pub entry_block: BlockId,
    pub value_types: HashMap<SsaValue, DataType>,
    pub next_value_id: u32,
    pub next_block_id: u32,
}

impl SsaProgram {
    pub fn new() -> Self {
        Self::default()
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
        self.blocks.insert(block_id, SsaBlock { id: block_id, instructions: Vec::new() });
        block_id
    }

    pub fn add_instruction(&mut self, block_id: BlockId, instruction: SsaInstruction) {
        if let Some(block) = self.blocks.get_mut(&block_id) {
            block.instructions.push(instruction);
        }
    }
}

/// Convert AST to SSA IR
pub fn ast_to_ssa(expr: &Expr) -> Result<SsaProgram, DioError> {
    if let Expr::Lambda { params, return_type, body } = expr {
        convert_typed_lambda(params, return_type, body)
    } else {
        Err(DioError::Compilation("Only typed lambda expressions are supported".to_string()))
    }
}

fn convert_typed_lambda(
    params: &[TypedParam],
    return_type: &Type,
    body: &Expr,
) -> Result<SsaProgram, DioError> {
    let mut program = SsaProgram::new();
    let entry_block = program.new_block();
    program.entry_block = entry_block;

    let (op, operands) = match body {
        Expr::Add(ops) => ("add", ops),
        Expr::Sub(lhs, rhs) => ("sub", &vec![*lhs.clone(), *rhs.clone()]),
        _ => return Err(DioError::Compilation("Unsupported operation in lambda body".to_string())),
    };

    let param_types: Vec<Type> = params.iter().map(|p| p.type_.clone()).collect();
    let coerced_type = coerce_nary_op_types(&param_types)?;
    if return_type != &coerced_type {
        return Err(DioError::TypeMismatch {
            expected: coerced_type.to_string(),
            found: return_type.to_string(),
            context: "Return type mismatch".to_string(),
        });
    }

    let element_type = if coerced_type.is_i64() { DataType::I64 } else { DataType::U64 };
    
    // Load array parameters
    let mut param_values = Vec::new();
    for (i, param) in params.iter().enumerate() {
        let data_type = if param.type_.is_i64() { DataType::ArrayI64 } else { DataType::ArrayU64 };
        let param_val = program.new_value(data_type.clone());
        program.add_instruction(entry_block, SsaInstruction::LoadArrayParam {
            dest: param_val,
            param_index: i as u32,
            data_type,
        });
        param_values.push(param_val);
    }

    // Load array length (assume it's passed as additional parameter)
    let length_val = program.new_value(DataType::U64);
    program.add_instruction(entry_block, SsaInstruction::LoadLengthParam {
        dest: length_val,
        param_index: params.len() as u32,
    });

    // Load constants
    let zero_val = program.new_value(DataType::U64);
    program.add_instruction(entry_block, SsaInstruction::LoadConstant {
        dest: zero_val,
        value: 0,
    });

    // Create loop blocks
    let loop_header = program.new_block();
    let loop_body = program.new_block(); 
    let exit_block = program.new_block();

    // Jump from entry to loop header with i = 0
    // Note: We'll handle the initial jump value in the Cranelift backend
    program.add_instruction(entry_block, SsaInstruction::Jump { target: loop_header });

    // Loop header: check condition i < length
    // The loop index will be a block parameter, but we need a placeholder SSA value for it
    let loop_index = program.new_value(DataType::U64); // This represents the current loop index
    let condition_val = program.new_value(DataType::Bool);
    program.add_instruction(loop_header, SsaInstruction::LessThan {
        dest: condition_val,
        lhs: loop_index,
        rhs: length_val,
    });
    program.add_instruction(loop_header, SsaInstruction::Branch {
        condition: condition_val,
        true_target: loop_body,
        false_target: exit_block,
    });

    // Loop body: compute the operation for this element
    let mut operand_elements = Vec::new();
    for operand in operands {
        if let Expr::Column(name) = operand {
            let param_index = params.iter().position(|p| &p.name == name).unwrap();
            let array_val = param_values[param_index];
            let element_val = program.new_value(element_type.clone());
            program.add_instruction(loop_body, SsaInstruction::ArrayAccess {
                dest: element_val,
                array: array_val,
                index: loop_index, // Use actual loop index
            });
            operand_elements.push(element_val);
        }
    }

    // Perform the operation
    let mut acc = operand_elements[0];
    for &rhs in &operand_elements[1..] {
        let dest = program.new_value(element_type.clone());
        let instruction = match op {
            "add" => SsaInstruction::Add { dest, lhs: acc, rhs },
            "sub" => SsaInstruction::Sub { dest, lhs: acc, rhs },
            _ => unreachable!(),
        };
        program.add_instruction(loop_body, instruction);
        acc = dest;
    }

    // Store result to output array
    program.add_instruction(loop_body, SsaInstruction::StoreArrayElement {
        array: SsaValue(params.len() as u32), // Output array parameter
        index: loop_index,
        value: acc,
    });

    // Increment loop index and jump back to header
    let one_val = program.new_value(DataType::U64);
    program.add_instruction(loop_body, SsaInstruction::LoadConstant {
        dest: one_val,
        value: 1,
    });
    let next_index = program.new_value(DataType::U64);
    program.add_instruction(loop_body, SsaInstruction::Add {
        dest: next_index,
        lhs: loop_index,
        rhs: one_val,
    });
    program.add_instruction(loop_body, SsaInstruction::Jump { target: loop_header });

    // Exit block
    program.add_instruction(exit_block, SsaInstruction::Return { value: None });

    Ok(program)
}
