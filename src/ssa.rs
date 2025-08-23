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

// ============================================================================
// SSA Version 2: New API that maps directly to Cranelift
// ============================================================================

/// Binary operation kinds that map directly to Cranelift operations
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
}

/// SSA Instructions V2 - designed to map directly to Cranelift
#[derive(Debug, Clone)]
pub enum SsaInstructionV2 {
    // Data flow
    Parameter { dest: SsaValue, param_index: u32, data_type: DataType },
    Constant { dest: SsaValue, value: i64, data_type: DataType },
    
    // Arithmetic (maps directly to Cranelift)
    BinaryOp { dest: SsaValue, op: BinaryOpKind, lhs: SsaValue, rhs: SsaValue },
    
    // Memory operations
    Load { dest: SsaValue, address: SsaValue, offset: i32, data_type: DataType },
    Store { address: SsaValue, offset: i32, value: SsaValue },
    
    // Control flow with explicit block parameters
    Branch { condition: SsaValue, true_block: BlockId, false_block: BlockId },
    Jump { target_block: BlockId },
    Return { value: Option<SsaValue> },
}

/// Basic block V2 with explicit parameters
#[derive(Debug, Clone)]
pub struct SsaBlockV2 {
    pub id: BlockId,
    pub parameters: Vec<(SsaValue, DataType)>, // Explicit block parameters
    pub instructions: Vec<SsaInstructionV2>,
}

/// Complete SSA program V2 with ordered blocks
#[derive(Debug, Clone)]
pub struct SsaProgramV2 {
    pub blocks: Vec<SsaBlockV2>, // Ordered list of blocks (not HashMap)
    pub block_map: HashMap<BlockId, usize>, // BlockId -> index in blocks vec
    pub entry_block: BlockId,
    pub value_types: HashMap<SsaValue, DataType>,
    pub next_value_id: u32,
    pub next_block_id: u32,
}

impl Default for SsaProgramV2 {
    fn default() -> Self {
        Self {
            blocks: Vec::new(),
            block_map: HashMap::new(),
            entry_block: BlockId(0),
            value_types: HashMap::new(),
            next_value_id: 0,
            next_block_id: 0,
        }
    }
}

impl SsaProgramV2 {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn new_value(&mut self, data_type: DataType) -> SsaValue {
        let value = SsaValue(self.next_value_id);
        self.next_value_id += 1;
        self.value_types.insert(value, data_type);
        value
    }

    pub fn new_block(&mut self, parameters: Vec<(SsaValue, DataType)>) -> BlockId {
        let block_id = BlockId(self.next_block_id);
        self.next_block_id += 1;
        
        let block = SsaBlockV2 {
            id: block_id,
            parameters,
            instructions: Vec::new(),
        };
        
        let index = self.blocks.len();
        self.blocks.push(block);
        self.block_map.insert(block_id, index);
        
        block_id
    }

    pub fn add_instruction(&mut self, block_id: BlockId, instruction: SsaInstructionV2) {
        if let Some(&index) = self.block_map.get(&block_id) {
            self.blocks[index].instructions.push(instruction);
        }
    }

    pub fn get_block(&self, block_id: BlockId) -> Option<&SsaBlockV2> {
        self.block_map.get(&block_id).map(|&index| &self.blocks[index])
    }

    pub fn get_block_mut(&mut self, block_id: BlockId) -> Option<&mut SsaBlockV2> {
        if let Some(&index) = self.block_map.get(&block_id) {
            Some(&mut self.blocks[index])
        } else {
            None
        }
    }
}

/// Braun algorithm state for SSA construction
#[derive(Debug)]
struct SsaBuilder {
    program: SsaProgramV2,
    // Variable definitions: Block -> Variable -> SSAValue
    defns: HashMap<BlockId, HashMap<String, SsaValue>>,
    // Incomplete phi functions: Block -> Variable -> SSAValue
    incomplete_phis: HashMap<BlockId, HashMap<String, SsaValue>>,
    // Block predecessors for phi completion
    preds: HashMap<BlockId, Vec<BlockId>>,
}

impl SsaBuilder {
    fn new() -> Self {
        Self {
            program: SsaProgramV2::new(),
            defns: HashMap::new(),
            incomplete_phis: HashMap::new(),
            preds: HashMap::new(),
        }
    }

    /// Read variable value in given block (Braun algorithm)
    fn read_variable(&mut self, var: &str, block: BlockId, data_type: DataType) -> SsaValue {
        // Check if variable is already defined in this block
        if let Some(&value) = self.defns.get(&block).and_then(|vars| vars.get(var)) {
            return value;
        }

        // Handle phi nodes for blocks with multiple predecessors
        self.read_variable_recursive(var, block, data_type)
    }

    fn read_variable_recursive(&mut self, var: &str, block: BlockId, data_type: DataType) -> SsaValue {
        let preds = self.preds.get(&block).cloned().unwrap_or_default();
        
        if preds.is_empty() {
            // No predecessors - this should be a parameter
            panic!("Variable {} read in block {:?} with no definition and no predecessors", var, block);
        } else if preds.len() == 1 {
            // Single predecessor - recursively read from it
            self.read_variable(var, preds[0], data_type)
        } else {
            // Multiple predecessors - need phi function
            let phi_value = self.program.new_value(data_type.clone());
            
            // Add as block parameter
            if let Some(block_ref) = self.program.get_block_mut(block) {
                block_ref.parameters.push((phi_value, data_type));
            }
            
            // Record the phi value as definition for this variable in this block
            self.defns.entry(block).or_default().insert(var.to_string(), phi_value);
            
            phi_value
        }
    }

    /// Write variable value in given block
    fn write_variable(&mut self, var: &str, block: BlockId, value: SsaValue) {
        self.defns.entry(block).or_default().insert(var.to_string(), value);
    }

    /// Add predecessor relationship
    fn add_predecessor(&mut self, block: BlockId, pred: BlockId) {
        self.preds.entry(block).or_default().push(pred);
    }

    /// Take ownership of the program
    fn into_program(self) -> SsaProgramV2 {
        self.program
    }
}

/// Convert AST to SSA IR V2 using Braun algorithm
pub fn ast_to_ssa_v2(expr: &Expr) -> Result<SsaProgramV2, DioError> {
    if let Expr::Lambda { params, return_type, body } = expr {
        convert_typed_lambda_v2(params, return_type, body)
    } else {
        Err(DioError::Compilation("Only typed lambda expressions are supported".to_string()))
    }
}

fn convert_typed_lambda_v2(
    params: &[TypedParam],
    return_type: &Type,
    body: &Expr,
) -> Result<SsaProgramV2, DioError> {
    let mut builder = SsaBuilder::new();
    
    match body {
        Expr::Add(ops) => convert_elementwise_lambda_v2(builder, params, return_type, BinaryOpKind::Add, ops),
        Expr::Sub(lhs, rhs) => {
            let ops = vec![*lhs.clone(), *rhs.clone()];
            convert_elementwise_lambda_v2(builder, params, return_type, BinaryOpKind::Sub, &ops)
        }
        Expr::Sum(inner_expr) => convert_reduction_lambda_v2(builder, params, return_type, inner_expr),
        _ => Err(DioError::Compilation("Unsupported operation in lambda body".to_string())),
    }
}

fn convert_elementwise_lambda_v2(
    mut builder: SsaBuilder,
    params: &[TypedParam],
    return_type: &Type,
    op: BinaryOpKind,
    operands: &[Expr],
) -> Result<SsaProgramV2, DioError> {
    // Validate types
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
    
    // Create entry block with parameters
    let mut entry_params = Vec::new();
    for (_i, param) in params.iter().enumerate() {
        let data_type = if param.type_.is_i64() { DataType::ArrayI64 } else { DataType::ArrayU64 };
        let param_val = builder.program.new_value(data_type.clone());
        entry_params.push((param_val, data_type));
        // Define parameter variables
        builder.write_variable(&param.name, BlockId(0), param_val);
    }
    
    // Add length parameter
    let length_val = builder.program.new_value(DataType::U64);
    entry_params.push((length_val, DataType::U64));
    builder.write_variable("__length", BlockId(0), length_val);
    
    // Add output array parameter
    let output_data_type = if coerced_type.is_i64() { DataType::ArrayI64 } else { DataType::ArrayU64 };
    let output_val = builder.program.new_value(output_data_type.clone());
    entry_params.push((output_val, output_data_type.clone()));
    builder.write_variable("__output", BlockId(0), output_val);

    let entry_block = builder.program.new_block(entry_params);
    builder.program.entry_block = entry_block;

    // Load constants in entry block
    let zero_val = builder.program.new_value(DataType::U64);
    builder.program.add_instruction(entry_block, SsaInstructionV2::Constant {
        dest: zero_val,
        value: 0,
        data_type: DataType::U64,
    });
    builder.write_variable("__zero", entry_block, zero_val);

    // Create loop blocks
    let loop_index_val = builder.program.new_value(DataType::U64);
    let loop_header = builder.program.new_block(vec![
        (loop_index_val, DataType::U64) // loop index
    ]);
    let loop_index_val2 = builder.program.new_value(DataType::U64);
    let loop_body = builder.program.new_block(vec![
        (loop_index_val2, DataType::U64) // loop index
    ]);
    let exit_block = builder.program.new_block(vec![]);

    // Set up predecessors
    builder.add_predecessor(loop_header, entry_block);
    builder.add_predecessor(loop_header, loop_body);
    builder.add_predecessor(loop_body, loop_header);
    builder.add_predecessor(exit_block, loop_header);

    // Entry block: jump to loop header with index = 0
    builder.program.add_instruction(entry_block, SsaInstructionV2::Jump { target_block: loop_header });

    // Loop header: check condition i < length
    let loop_index = builder.read_variable("__loop_index", loop_header, DataType::U64);
    let length_val_in_header = builder.read_variable("__length", loop_header, DataType::U64);
    let condition_val = builder.program.new_value(DataType::Bool);
    builder.program.add_instruction(loop_header, SsaInstructionV2::BinaryOp {
        dest: condition_val,
        op: BinaryOpKind::Lt,
        lhs: loop_index,
        rhs: length_val_in_header,
    });
    builder.program.add_instruction(loop_header, SsaInstructionV2::Branch {
        condition: condition_val,
        true_block: loop_body,
        false_block: exit_block,
    });

    // Loop body: perform elementwise operation
    let body_index = builder.read_variable("__loop_index", loop_body, DataType::U64);
    
    // Load array elements
    let mut operand_elements = Vec::new();
    for operand in operands {
        if let Expr::Column(name) = operand {
            let array_val = builder.read_variable(name, loop_body, if element_type == DataType::I64 { DataType::ArrayI64 } else { DataType::ArrayU64 });
            let element_val = builder.program.new_value(element_type.clone());
            builder.program.add_instruction(loop_body, SsaInstructionV2::Load {
                dest: element_val,
                address: array_val,
                offset: 0, // Will be computed using index in Cranelift backend
                data_type: element_type.clone(),
            });
            operand_elements.push(element_val);
        }
    }

    // Perform the operation
    let mut acc = operand_elements[0];
    for &rhs in &operand_elements[1..] {
        let dest = builder.program.new_value(element_type.clone());
        builder.program.add_instruction(loop_body, SsaInstructionV2::BinaryOp {
            dest,
            op,
            lhs: acc,
            rhs,
        });
        acc = dest;
    }

    // Store result
    let output_val_in_body = builder.read_variable("__output", loop_body, output_data_type);
    builder.program.add_instruction(loop_body, SsaInstructionV2::Store {
        address: output_val_in_body,
        offset: 0, // Will be computed using index in Cranelift backend
        value: acc,
    });

    // Increment loop index
    let one_val = builder.program.new_value(DataType::U64);
    builder.program.add_instruction(loop_body, SsaInstructionV2::Constant {
        dest: one_val,
        value: 1,
        data_type: DataType::U64,
    });
    let next_index = builder.program.new_value(DataType::U64);
    builder.program.add_instruction(loop_body, SsaInstructionV2::BinaryOp {
        dest: next_index,
        op: BinaryOpKind::Add,
        lhs: body_index,
        rhs: one_val,
    });
    builder.write_variable("__loop_index", loop_body, next_index);

    // Jump back to header
    builder.program.add_instruction(loop_body, SsaInstructionV2::Jump { target_block: loop_header });

    // Exit block
    builder.program.add_instruction(exit_block, SsaInstructionV2::Return { value: None });

    Ok(builder.into_program())
}

fn convert_reduction_lambda_v2(
    mut builder: SsaBuilder,
    params: &[TypedParam],
    return_type: &Type,
    inner_expr: &Expr,
) -> Result<SsaProgramV2, DioError> {
    // Validate return type is scalar
    if !return_type.is_scalar() {
        return Err(DioError::TypeMismatch {
            expected: "Scalar type (U64 or I64)".to_string(),
            found: return_type.to_string(),
            context: "Sum reduction must return scalar".to_string(),
        });
    }

    let scalar_type = if return_type.is_i64() { DataType::I64 } else { DataType::U64 };
    
    // Create entry block with parameters
    let mut entry_params = Vec::new();
    for (_i, param) in params.iter().enumerate() {
        let data_type = if param.type_.is_i64() { DataType::ArrayI64 } else { DataType::ArrayU64 };
        let param_val = builder.program.new_value(data_type.clone());
        entry_params.push((param_val, data_type));
        builder.write_variable(&param.name, BlockId(0), param_val);
    }
    
    // Add length parameter
    let length_val = builder.program.new_value(DataType::U64);
    entry_params.push((length_val, DataType::U64));
    builder.write_variable("__length", BlockId(0), length_val);

    let entry_block = builder.program.new_block(entry_params);
    builder.program.entry_block = entry_block;

    // Initialize accumulator
    let initial_acc = builder.program.new_value(scalar_type.clone());
    builder.program.add_instruction(entry_block, SsaInstructionV2::Constant {
        dest: initial_acc,
        value: 0,
        data_type: scalar_type.clone(),
    });
    builder.write_variable("__accumulator", entry_block, initial_acc);

    // Load constants
    let zero_val = builder.program.new_value(DataType::U64);
    builder.program.add_instruction(entry_block, SsaInstructionV2::Constant {
        dest: zero_val,
        value: 0,
        data_type: DataType::U64,
    });
    builder.write_variable("__loop_index", entry_block, zero_val);

    // Create loop blocks with parameters
    let header_index_val = builder.program.new_value(DataType::U64);
    let header_acc_val = builder.program.new_value(scalar_type.clone());
    let loop_header = builder.program.new_block(vec![
        (header_index_val, DataType::U64), // loop index
        (header_acc_val, scalar_type.clone()) // accumulator
    ]);
    let body_index_val = builder.program.new_value(DataType::U64);
    let body_acc_val = builder.program.new_value(scalar_type.clone());
    let loop_body = builder.program.new_block(vec![
        (body_index_val, DataType::U64), // loop index  
        (body_acc_val, scalar_type.clone()) // accumulator
    ]);
    let exit_acc_val = builder.program.new_value(scalar_type.clone());
    let exit_block = builder.program.new_block(vec![
        (exit_acc_val, scalar_type.clone()) // final accumulator
    ]);

    // Set up predecessors
    builder.add_predecessor(loop_header, entry_block);
    builder.add_predecessor(loop_header, loop_body);
    builder.add_predecessor(loop_body, loop_header);
    builder.add_predecessor(exit_block, loop_header);

    // Entry: jump to loop header
    builder.program.add_instruction(entry_block, SsaInstructionV2::Jump { target_block: loop_header });

    // Loop header: check condition
    let header_index = builder.read_variable("__loop_index", loop_header, DataType::U64);
    let _header_acc = builder.read_variable("__accumulator", loop_header, scalar_type.clone());
    let header_length = builder.read_variable("__length", loop_header, DataType::U64);
    let condition_val = builder.program.new_value(DataType::Bool);
    builder.program.add_instruction(loop_header, SsaInstructionV2::BinaryOp {
        dest: condition_val,
        op: BinaryOpKind::Lt,
        lhs: header_index,
        rhs: header_length,
    });
    builder.program.add_instruction(loop_header, SsaInstructionV2::Branch {
        condition: condition_val,
        true_block: loop_body,
        false_block: exit_block,
    });

    // Loop body: evaluate inner expression and update accumulator
    let body_index = builder.read_variable("__loop_index", loop_body, DataType::U64);
    let body_acc = builder.read_variable("__accumulator", loop_body, scalar_type.clone());
    
    let element_value = convert_inner_expression_v2(&mut builder, loop_body, inner_expr, params, body_index, scalar_type.clone())?;
    
    // Update accumulator
    let new_acc = builder.program.new_value(scalar_type.clone());
    builder.program.add_instruction(loop_body, SsaInstructionV2::BinaryOp {
        dest: new_acc,
        op: BinaryOpKind::Add,
        lhs: body_acc,
        rhs: element_value,
    });
    builder.write_variable("__accumulator", loop_body, new_acc);

    // Increment loop index
    let one_val = builder.program.new_value(DataType::U64);
    builder.program.add_instruction(loop_body, SsaInstructionV2::Constant {
        dest: one_val,
        value: 1,
        data_type: DataType::U64,
    });
    let next_index = builder.program.new_value(DataType::U64);
    builder.program.add_instruction(loop_body, SsaInstructionV2::BinaryOp {
        dest: next_index,
        op: BinaryOpKind::Add,
        lhs: body_index,
        rhs: one_val,
    });
    builder.write_variable("__loop_index", loop_body, next_index);

    // Jump back to header
    builder.program.add_instruction(loop_body, SsaInstructionV2::Jump { target_block: loop_header });

    // Exit block: return final accumulator
    let final_acc = builder.read_variable("__accumulator", exit_block, scalar_type.clone());
    builder.program.add_instruction(exit_block, SsaInstructionV2::Return { value: Some(final_acc) });

    Ok(builder.into_program())
}

fn convert_inner_expression_v2(
    builder: &mut SsaBuilder,
    block_id: BlockId,
    expr: &Expr,
    params: &[TypedParam],
    loop_index: SsaValue,
    result_type: DataType,
) -> Result<SsaValue, DioError> {
    match expr {
        Expr::Column(name) => {
            let array_data_type = if result_type == DataType::I64 { DataType::ArrayI64 } else { DataType::ArrayU64 };
            let array_val = builder.read_variable(name, block_id, array_data_type);
            let element_val = builder.program.new_value(result_type.clone());
            builder.program.add_instruction(block_id, SsaInstructionV2::Load {
                dest: element_val,
                address: array_val,
                offset: 0, // Will be computed using index in Cranelift backend
                data_type: result_type,
            });
            Ok(element_val)
        }
        Expr::Add(operands) => {
            let mut operand_values = Vec::new();
            for operand in operands {
                let val = convert_inner_expression_v2(builder, block_id, operand, params, loop_index, result_type.clone())?;
                operand_values.push(val);
            }
            
            let mut acc = operand_values[0];
            for &rhs in &operand_values[1..] {
                let dest = builder.program.new_value(result_type.clone());
                builder.program.add_instruction(block_id, SsaInstructionV2::BinaryOp {
                    dest,
                    op: BinaryOpKind::Add,
                    lhs: acc,
                    rhs,
                });
                acc = dest;
            }
            Ok(acc)
        }
        Expr::Sub(lhs, rhs) => {
            let lhs_val = convert_inner_expression_v2(builder, block_id, lhs, params, loop_index, result_type.clone())?;
            let rhs_val = convert_inner_expression_v2(builder, block_id, rhs, params, loop_index, result_type.clone())?;
            let dest = builder.program.new_value(result_type);
            builder.program.add_instruction(block_id, SsaInstructionV2::BinaryOp {
                dest,
                op: BinaryOpKind::Sub,
                lhs: lhs_val,
                rhs: rhs_val,
            });
            Ok(dest)
        }
        _ => Err(DioError::Compilation("Unsupported expression in reduction body".to_string())),
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ast::{Expr, Type, TypedParam};

    #[test]
    fn test_ssa_v2_basic_add() {
        let params = vec![
            TypedParam { name: "a".to_string(), type_: Type::U64Array },
            TypedParam { name: "b".to_string(), type_: Type::U64Array },
        ];
        let return_type = Type::U64Array;
        let body = Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Column("b".to_string()),
        ]);
        let expr = Expr::Lambda { params, return_type, body: Box::new(body) };

        let result = ast_to_ssa_v2(&expr);
        assert!(result.is_ok(), "SSA v2 generation failed: {:?}", result);

        let program = result.unwrap();
        assert_eq!(program.blocks.len(), 4, "Expected 4 blocks (entry, loop_header, loop_body, exit)");
        assert_eq!(program.entry_block, BlockId(0), "Entry block should be BlockId(0)");
        
        // Verify that blocks are ordered correctly
        assert_eq!(program.blocks[0].id, BlockId(0)); // entry
        assert_eq!(program.blocks[1].id, BlockId(1)); // loop_header  
        assert_eq!(program.blocks[2].id, BlockId(2)); // loop_body
        assert_eq!(program.blocks[3].id, BlockId(3)); // exit

        // Verify entry block has parameters
        assert_eq!(program.blocks[0].parameters.len(), 4); // a, b, length, output

        println!("SSA v2 Program Structure:");
        for (i, block) in program.blocks.iter().enumerate() {
            println!("Block {}: {:?} with {} parameters, {} instructions", 
                     i, block.id, block.parameters.len(), block.instructions.len());
        }
    }

    #[test]
    fn test_ssa_v2_reduction() {
        let params = vec![
            TypedParam { name: "a".to_string(), type_: Type::U64Array },
        ];
        let return_type = Type::U64;
        let body = Expr::Sum(Box::new(Expr::Column("a".to_string())));
        let expr = Expr::Lambda { params, return_type, body: Box::new(body) };

        let result = ast_to_ssa_v2(&expr);
        assert!(result.is_ok(), "SSA v2 reduction generation failed: {:?}", result);

        let program = result.unwrap();
        assert_eq!(program.blocks.len(), 4, "Expected 4 blocks for reduction");
        
        // Entry block should have parameters for array and length
        assert_eq!(program.blocks[0].parameters.len(), 2); // a, length

        println!("SSA v2 Reduction Program Structure:");
        for (i, block) in program.blocks.iter().enumerate() {
            println!("Block {}: {:?} with {} parameters, {} instructions", 
                     i, block.id, block.parameters.len(), block.instructions.len());
        }
    }
}
