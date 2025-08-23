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
    
    // Reduction operations
    InitAccumulator { dest: SsaValue, initial_value: i64, scalar_type: DataType },
    UpdateAccumulator { dest: SsaValue, accumulator: SsaValue, value: SsaValue },
    
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
    match body {
        Expr::Add(ops) => convert_elementwise_lambda(params, return_type, "add", ops),
        Expr::Sub(lhs, rhs) => {
            let ops = vec![*lhs.clone(), *rhs.clone()];
            convert_elementwise_lambda(params, return_type, "sub", &ops)
        }
        Expr::Sum(inner_expr) => convert_reduction_lambda(params, return_type, inner_expr),
        _ => Err(DioError::Compilation("Unsupported operation in lambda body".to_string())),
    }
}

fn convert_elementwise_lambda(
    params: &[TypedParam],
    return_type: &Type,
    op: &str,
    operands: &[Expr],
) -> Result<SsaProgram, DioError> {
    let mut program = SsaProgram::new();
    let entry_block = program.new_block();
    program.entry_block = entry_block;

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

fn convert_reduction_lambda(
    params: &[TypedParam],
    return_type: &Type,
    inner_expr: &Expr,
) -> Result<SsaProgram, DioError> {
    let mut program = SsaProgram::new();
    let entry_block = program.new_block();
    program.entry_block = entry_block;

    // Validate return type is scalar
    if !return_type.is_scalar() {
        return Err(DioError::TypeMismatch {
            expected: "Scalar type (U64 or I64)".to_string(),
            found: return_type.to_string(),
            context: "Sum reduction must return scalar".to_string(),
        });
    }

    let scalar_type = if return_type.is_i64() { DataType::I64 } else { DataType::U64 };
    
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

    // Initialize accumulator
    let accumulator = program.new_value(scalar_type.clone());
    program.add_instruction(entry_block, SsaInstruction::InitAccumulator {
        dest: accumulator,
        initial_value: 0, // Sum starts at 0
        scalar_type: scalar_type.clone(),
    });

    // Create loop blocks
    let loop_header = program.new_block();
    let loop_body = program.new_block(); 
    let exit_block = program.new_block();

    // Jump from entry to loop header with i = 0 and initial accumulator
    program.add_instruction(entry_block, SsaInstruction::Jump { target: loop_header });

    // Loop header: check condition i < length
    // The loop header takes two parameters: loop_index and current_accumulator
    // These will be mapped to block parameters by the Cranelift backend
    let loop_index = program.new_value(DataType::U64);
    let current_accumulator = program.new_value(scalar_type.clone());
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

    // Loop body: evaluate inner expression and add to accumulator
    let element_value = convert_inner_expression(&mut program, loop_body, inner_expr, &param_values, params, loop_index, scalar_type.clone())?;
    
    // Update accumulator: new_acc = current_acc + element_value
    let new_accumulator = program.new_value(scalar_type.clone());
    program.add_instruction(loop_body, SsaInstruction::Add {
        dest: new_accumulator,
        lhs: current_accumulator,
        rhs: element_value,
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

    // Exit block: return final accumulator value
    // We use current_accumulator from the loop header block
    program.add_instruction(exit_block, SsaInstruction::Return { value: Some(current_accumulator) });

    if std::env::var("DIO_DEBUG_SSA").is_ok() {
        println!("--- Generated SSA Program for Reduction ---");
        for (block_id, block) in &program.blocks {
            println!("Block {:?}:", block_id);
            for instruction in &block.instructions {
                println!("  {:?}", instruction);
            }
        }
        println!("--- End SSA Program ---");
    }

    Ok(program)
}

fn convert_inner_expression(
    program: &mut SsaProgram,
    block_id: BlockId,
    expr: &Expr,
    param_values: &[SsaValue],
    params: &[TypedParam],
    loop_index: SsaValue,
    result_type: DataType,
) -> Result<SsaValue, DioError> {
    match expr {
        Expr::Column(name) => {
            let param_index = params.iter().position(|p| &p.name == name)
                .ok_or_else(|| DioError::Compilation(format!("Unknown parameter: {}", name)))?;
            let array_val = param_values[param_index];
            let element_val = program.new_value(result_type);
            program.add_instruction(block_id, SsaInstruction::ArrayAccess {
                dest: element_val,
                array: array_val,
                index: loop_index,
            });
            Ok(element_val)
        }
        Expr::Add(operands) => {
            let mut operand_values = Vec::new();
            for operand in operands {
                let val = convert_inner_expression(program, block_id, operand, param_values, params, loop_index, result_type.clone())?;
                operand_values.push(val);
            }
            
            let mut acc = operand_values[0];
            for &rhs in &operand_values[1..] {
                let dest = program.new_value(result_type.clone());
                program.add_instruction(block_id, SsaInstruction::Add { dest, lhs: acc, rhs });
                acc = dest;
            }
            Ok(acc)
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_val = convert_inner_expression(program, block_id, lhs, param_values, params, loop_index, result_type.clone())?;
            let rhs_val = convert_inner_expression(program, block_id, rhs, param_values, params, loop_index, result_type.clone())?;
            let dest = program.new_value(result_type);
            program.add_instruction(block_id, SsaInstruction::Sub { dest, lhs: lhs_val, rhs: rhs_val });
            Ok(dest)
        }
        _ => Err(DioError::Compilation("Unsupported expression in reduction body".to_string())),
    }
}
