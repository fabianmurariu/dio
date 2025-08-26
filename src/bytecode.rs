use crate::ast::*;
use crate::error::DioError;
use crate::ssa::{BlockId, DataType as SsaDataType, SsaInstructionV2, SsaProgramV2, SsaValue};
use std::fmt;

/// ByteCode intermediate representation - C-like imperative code
/// This bridges the gap between high-level vectorized Lisp and low-level SSA
#[derive(Debug, Clone)]
pub struct ByteCodeProgram {
    /// Input parameters with their types
    pub inputs: Vec<InputParam>,
    /// Return type
    pub return_type: DataType,
    /// Local variables/temporaries
    pub locals: Vec<LocalVar>,
    /// Sequence of statements
    pub statements: Vec<Statement>,
}

#[derive(Debug, Clone)]
pub struct InputParam {
    pub name: String,
    pub data_type: DataType,
}

#[derive(Debug, Clone)]
pub struct LocalVar {
    pub name: String,
    pub data_type: DataType,
}

/// Data types in ByteCode
#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    U64,
    I64,
    ArrayU64,
    ArrayI64,
}

impl DataType {
    pub fn is_array(&self) -> bool {
        matches!(self, DataType::ArrayU64 | DataType::ArrayI64)
    }

    pub fn is_scalar(&self) -> bool {
        !self.is_array()
    }
}

/// ByteCode statements - imperative operations
#[derive(Debug, Clone)]
pub enum Statement {
    /// Variable assignment: var = expr
    Assign { target: String, expr: Expression },

    /// Array element assignment: array[index] = expr
    ArrayAssign {
        array: String,
        index: Expression,
        value: Expression,
    },

    /// For loop: for (var = start; var < end; var += step) { body }
    ForLoop {
        index_var: String,
        start: Expression,
        end: Expression,
        step: Expression,
        body: Vec<Statement>,
    },

    /// Return statement
    Return { value: Option<Expression> },
}

/// ByteCode expressions - C-like expressions
#[derive(Debug, Clone)]
pub enum Expression {
    /// Variable reference
    Variable(String),

    /// Literal constant
    Literal(i64),

    /// Array element access: array[index]
    ArrayAccess {
        array: String,
        index: Box<Expression>,
    },

    /// Binary operation: left op right
    BinaryOp {
        op: BinaryOperator,
        left: Box<Expression>,
        right: Box<Expression>,
    },
}

#[derive(Debug, Clone)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    Div,
    Lt,
}

/// Convert high-level vectorized Lisp AST to imperative ByteCode
pub fn ast_to_bytecode(expr: &Expr) -> Result<ByteCodeProgram, DioError> {
    if let Expr::Lambda {
        params,
        return_type,
        body,
    } = expr
    {
        let mut compiler = ByteCodeCompiler::new();
        compiler.compile_lambda(params, return_type, body)
    } else {
        Err(DioError::Compilation(
            "Only typed lambda expressions are supported".to_string(),
        ))
    }
}

struct ByteCodeCompiler {
    inputs: Vec<InputParam>,
    locals: Vec<LocalVar>,
    statements: Vec<Statement>,
}

impl ByteCodeCompiler {
    fn new() -> Self {
        Self {
            inputs: Vec::new(),
            locals: Vec::new(),
            statements: Vec::new(),
        }
    }

    fn compile_lambda(
        &mut self,
        params: &[TypedParam],
        return_type: &Type,
        body: &Expr,
    ) -> Result<ByteCodeProgram, DioError> {
        // Convert input parameters
        for param in params {
            let data_type = match &param.type_ {
                Type::U64Array => DataType::ArrayU64,
                Type::I64Array => DataType::ArrayI64,
                Type::U64 => DataType::U64,
                Type::I64 => DataType::I64,
                Type::F64 => DataType::I64, // Treat F64 as I64 for now
                Type::F64Array => DataType::ArrayI64, // Treat F64Array as ArrayI64 for now
            };
            self.inputs.push(InputParam {
                name: param.name.clone(),
                data_type,
            });
        }

        // Convert return type
        let ret_type = match return_type {
            Type::U64Array => DataType::ArrayU64,
            Type::I64Array => DataType::ArrayI64,
            Type::U64 => DataType::U64,
            Type::I64 => DataType::I64,
            Type::F64 => DataType::I64,
            Type::F64Array => DataType::ArrayI64,
        };

        // Add length parameter (implicit in all functions)
        self.inputs.push(InputParam {
            name: "length".to_string(),
            data_type: DataType::U64,
        });

        // For array returns, add output parameter
        let output_param = if ret_type.is_array() {
            Some(InputParam {
                name: "output".to_string(),
                data_type: ret_type.clone(),
            })
        } else {
            None
        };

        // Compile the lambda body into ByteCode statements
        match body {
            Expr::Add(operands) => {
                // Elementwise addition: for(i = 0; i < length; i++) output[i] = a[i] + b[i] + ...
                if ret_type.is_array() {
                    self.compile_elementwise_operation(operands, BinaryOperator::Add)?;
                } else {
                    return Err(DioError::Compilation(
                        "Add with scalar return type not supported".to_string(),
                    ));
                }
            }
            Expr::Sub(lhs, rhs) => {
                // Elementwise subtraction: for(i = 0; i < length; i++) output[i] = lhs[i] - rhs[i]
                if ret_type.is_array() {
                    self.compile_elementwise_operation(
                        &[*lhs.clone(), *rhs.clone()],
                        BinaryOperator::Sub,
                    )?;
                } else {
                    return Err(DioError::Compilation(
                        "Sub with scalar return type not supported".to_string(),
                    ));
                }
            }
            Expr::Mul(operands) => {
                // Elementwise multiplication: for(i = 0; i < length; i++) output[i] = a[i] * b[i] * ...
                if ret_type.is_array() {
                    self.compile_elementwise_operation(operands, BinaryOperator::Mul)?;
                } else {
                    return Err(DioError::Compilation(
                        "Mul with scalar return type not supported".to_string(),
                    ));
                }
            }
            Expr::Div(lhs, rhs) => {
                // Elementwise division: for(i = 0; i < length; i++) output[i] = lhs[i] / rhs[i]
                if ret_type.is_array() {
                    self.compile_elementwise_operation(
                        &[*lhs.clone(), *rhs.clone()],
                        BinaryOperator::Div,
                    )?;
                } else {
                    return Err(DioError::Compilation(
                        "Div with scalar return type not supported".to_string(),
                    ));
                }
            }
            Expr::Sum(inner) => {
                // Reduction operation: sum over the inner expression
                self.compile_reduction_operation(inner, &ret_type)?;
            }
            _ => {
                return Err(DioError::Compilation(
                    "Unsupported operation in lambda body".to_string(),
                ));
            }
        }

        Ok(ByteCodeProgram {
            inputs: self.inputs.clone(),
            return_type: ret_type,
            locals: self.locals.clone(),
            statements: self.statements.clone(),
        })
    }

    /// Compile elementwise operations like (+ a b), (- a b), etc.
    fn compile_elementwise_operation(
        &mut self,
        operands: &[Expr],
        op: BinaryOperator,
    ) -> Result<(), DioError> {
        // Add loop index variable
        let index_var = "i".to_string();
        self.locals.push(LocalVar {
            name: index_var.clone(),
            data_type: DataType::U64,
        });

        // Create the for loop: for(i = 0; i < length; i++)
        let mut loop_body = Vec::new();

        // Build the expression for the operation
        let mut expr = self.compile_expression(&operands[0])?;
        for operand in &operands[1..] {
            let right_expr = self.compile_expression(operand)?;
            expr = Expression::BinaryOp {
                op: op.clone(),
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }

        // output[i] = expr
        loop_body.push(Statement::ArrayAssign {
            array: "output".to_string(),
            index: Expression::Variable(index_var.clone()),
            value: expr,
        });

        // Add the for loop
        self.statements.push(Statement::ForLoop {
            index_var,
            start: Expression::Literal(0),
            end: Expression::Variable("length".to_string()),
            step: Expression::Literal(1),
            body: loop_body,
        });

        // Return void for elementwise operations
        self.statements.push(Statement::Return { value: None });

        Ok(())
    }

    /// Compile reduction operations like (sum expr)
    fn compile_reduction_operation(
        &mut self,
        inner: &Expr,
        ret_type: &DataType,
    ) -> Result<(), DioError> {
        // Add accumulator variable
        let acc_var = "acc".to_string();
        self.locals.push(LocalVar {
            name: acc_var.clone(),
            data_type: ret_type.clone(),
        });

        // Add loop index variable
        let index_var = "i".to_string();
        self.locals.push(LocalVar {
            name: index_var.clone(),
            data_type: DataType::U64,
        });

        // Initialize accumulator: acc = 0
        self.statements.push(Statement::Assign {
            target: acc_var.clone(),
            expr: Expression::Literal(0),
        });

        // Create the for loop: for(i = 0; i < length; i++)
        let mut loop_body = Vec::new();

        // Compile the inner expression (e.g., a[i] + b[i] for sum(+ a b))
        let inner_expr = self.compile_expression(inner)?;

        // acc = acc + inner_expr
        loop_body.push(Statement::Assign {
            target: acc_var.clone(),
            expr: Expression::BinaryOp {
                op: BinaryOperator::Add,
                left: Box::new(Expression::Variable(acc_var.clone())),
                right: Box::new(inner_expr),
            },
        });

        // Add the for loop
        self.statements.push(Statement::ForLoop {
            index_var,
            start: Expression::Literal(0),
            end: Expression::Variable("length".to_string()),
            step: Expression::Literal(1),
            body: loop_body,
        });

        // Simplification 2: Treat reductions as length-1 vectors - store result in output[0]
        self.statements.push(Statement::ArrayAssign {
            array: "output".to_string(),
            index: Expression::Literal(0),
            value: Expression::Variable(acc_var),
        });
        // Return void (no explicit return value needed)
        self.statements.push(Statement::Return { value: None });

        Ok(())
    }

    /// Compile an expression to a ByteCode expression
    fn compile_expression(&mut self, expr: &Expr) -> Result<Expression, DioError> {
        match expr {
            Expr::Column(name) => {
                // For array access, return array[i] where i is the current loop index
                Ok(Expression::ArrayAccess {
                    array: name.clone(),
                    index: Box::new(Expression::Variable("i".to_string())),
                })
            }
            Expr::Literal(value) => match value {
                Value::Int64(i) => Ok(Expression::Literal(*i)),
                Value::Float64(_) => Err(DioError::Compilation(
                    "Float literals not supported yet".to_string(),
                )),
            },
            Expr::Add(operands) => {
                // Build nested binary operations
                let mut result = self.compile_expression(&operands[0])?;
                for operand in &operands[1..] {
                    let right = self.compile_expression(operand)?;
                    result = Expression::BinaryOp {
                        op: BinaryOperator::Add,
                        left: Box::new(result),
                        right: Box::new(right),
                    };
                }
                Ok(result)
            }
            Expr::Sub(lhs, rhs) => {
                let left = self.compile_expression(lhs)?;
                let right = self.compile_expression(rhs)?;
                Ok(Expression::BinaryOp {
                    op: BinaryOperator::Sub,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
            Expr::Mul(operands) => {
                // Build nested binary operations
                let mut result = self.compile_expression(&operands[0])?;
                for operand in &operands[1..] {
                    let right = self.compile_expression(operand)?;
                    result = Expression::BinaryOp {
                        op: BinaryOperator::Mul,
                        left: Box::new(result),
                        right: Box::new(right),
                    };
                }
                Ok(result)
            }
            Expr::Div(lhs, rhs) => {
                let left = self.compile_expression(lhs)?;
                let right = self.compile_expression(rhs)?;
                Ok(Expression::BinaryOp {
                    op: BinaryOperator::Div,
                    left: Box::new(left),
                    right: Box::new(right),
                })
            }
            _ => Err(DioError::Compilation(format!(
                "Unsupported expression: {:?}",
                expr
            ))),
        }
    }
}

/// Convert ByteCode to SSA v2
pub fn bytecode_to_ssa_v2(program: &ByteCodeProgram) -> Result<SsaProgramV2, DioError> {
    let mut ssa_program = SsaProgramV2::new();

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

    // 3. output*: *mut u64 (always needed since both elementwise and reductions write to output)  
    let output_ptr = ssa_program.new_value(SsaDataType::U64);
    entry_params.push((output_ptr, SsaDataType::U64));
    param_mapping.insert("output".to_string(), output_ptr);

    // 4. length: u64
    let length = ssa_program.new_value(SsaDataType::U64);
    entry_params.push((length, SsaDataType::U64));
    param_mapping.insert("length".to_string(), length);

    // Create entry block
    let entry_block_id = ssa_program.new_block(entry_params);
    ssa_program.entry_block = entry_block_id;

    // Convert ByteCode statements to SSA
    let mut converter = ByteCodeToSsaConverter {
        ssa_program: &mut ssa_program,
        current_block: entry_block_id,
        param_mapping,
        local_mapping: std::collections::HashMap::new(),
    };

    converter.convert_statements(&program.statements, &program.locals, &program.inputs)?;

    Ok(ssa_program)
}

struct ByteCodeToSsaConverter<'a> {
    ssa_program: &'a mut SsaProgramV2,
    current_block: BlockId,
    param_mapping: std::collections::HashMap<String, SsaValue>,
    local_mapping: std::collections::HashMap<String, SsaValue>,
}

impl<'a> ByteCodeToSsaConverter<'a> {
    fn convert_statements(
        &mut self,
        statements: &[Statement],
        locals: &[LocalVar],
        inputs: &[InputParam],
    ) -> Result<(), DioError> {
        // Initialize local variables
        for local in locals {
            let ssa_data_type = convert_bytecode_to_ssa_datatype(&local.data_type);
            let ssa_value = self.ssa_program.new_value(ssa_data_type);
            self.local_mapping.insert(local.name.clone(), ssa_value);
        }

        // Load array pointers from the inputs array
        let inputs_ptr = self.param_mapping["inputs"];
        for (i, input) in inputs.iter().enumerate() {
            if input.data_type.is_array() {
                let index_val = self
                    .ssa_program
                    .new_value_with_const(self.current_block, i as i64);
                let array_ptr_addr = self.ssa_program.new_value(SsaDataType::U64);
                self.ssa_program.add_instruction(
                    self.current_block,
                    SsaInstructionV2::GetElementPtr {
                        dest: array_ptr_addr,
                        address: inputs_ptr,
                        index: index_val,
                        element_size: 8,
                    },
                );

                let array_ptr = self.ssa_program.new_value(SsaDataType::U64);
                self.ssa_program.add_instruction(
                    self.current_block,
                    SsaInstructionV2::Load {
                        dest: array_ptr,
                        address: array_ptr_addr,
                        offset: 0,
                        data_type: SsaDataType::U64,
                    },
                );
                self.param_mapping.insert(input.name.clone(), array_ptr);
            }
        }

        // Convert each statement
        for statement in statements {
            self.convert_statement(statement)?;
        }

        Ok(())
    }

    fn convert_statement(&mut self, statement: &Statement) -> Result<(), DioError> {
        match statement {
            Statement::Assign { target, expr } => {
                let value = self.convert_expression(expr)?;
                // Update the mapping for this variable
                self.local_mapping.insert(target.clone(), value);
            }
            Statement::ArrayAssign {
                array,
                index,
                value,
            } => {
                let array_ssa = self.get_variable(array)?;
                let index_ssa = self.convert_expression(index)?;
                let value_ssa = self.convert_expression(value)?;

                // Calculate element address: GEP(array_ptr, index)
                let element_addr = self.ssa_program.new_value(SsaDataType::U64);
                self.ssa_program.add_instruction(
                    self.current_block,
                    SsaInstructionV2::GetElementPtr {
                        dest: element_addr,
                        address: array_ssa,
                        index: index_ssa,
                        element_size: 8, // Assuming 64-bit elements
                    },
                );

                // Store the value at the calculated address
                self.ssa_program.add_instruction(
                    self.current_block,
                    SsaInstructionV2::Store {
                        address: element_addr,
                        offset: 0,
                        value: value_ssa,
                    },
                );
            }
            Statement::ForLoop {
                index_var,
                start,
                end,
                step: _,
                body,
            } => {
                // Convert for loop to SSA loop structure
                self.convert_for_loop(index_var, start, end, body)?;
            }
            Statement::Return { value } => match value {
                Some(expr) => {
                    let return_value = self.convert_expression(expr)?;
                    self.ssa_program.add_instruction(
                        self.current_block,
                        SsaInstructionV2::Return {
                            value: Some(return_value),
                        },
                    );
                }
                None => {
                    self.ssa_program.add_instruction(
                        self.current_block,
                        SsaInstructionV2::Return { value: None },
                    );
                }
            },
        }
        Ok(())
    }

    fn convert_for_loop(
        &mut self,
        index_var: &str,
        start: &Expression,
        end: &Expression,
        body: &[Statement],
    ) -> Result<(), DioError> {
        let start_value = self.convert_expression(start)?;
        let end_value = self.convert_expression(end)?;

        // Create new SSA values for the loop parameters
        let loop_index_param = self.ssa_program.new_value(crate::ssa::DataType::U64);
        let loop_body_index_param = self.ssa_program.new_value(crate::ssa::DataType::U64);

        // Create loop header and body blocks
        let loop_header = self
            .ssa_program
            .new_block(vec![(loop_index_param, crate::ssa::DataType::U64)]);

        let loop_body = self
            .ssa_program
            .new_block(vec![(loop_body_index_param, crate::ssa::DataType::U64)]);

        let exit_block = self.ssa_program.new_block(vec![]);

        // Jump from entry to loop header
        self.ssa_program.add_instruction(
            self.current_block,
            SsaInstructionV2::Jump {
                target_block: loop_header,
                args: vec![start_value],
            },
        );

        // Loop header: check condition (use the block parameter)
        let condition = self.ssa_program.new_value(crate::ssa::DataType::Bool);
        self.ssa_program.add_instruction(
            loop_header,
            SsaInstructionV2::BinaryOp {
                dest: condition,
                op: crate::ssa::BinaryOpKind::Lt,
                lhs: loop_index_param, // Use the block parameter
                rhs: end_value,
            },
        );

        self.ssa_program.add_instruction(
            loop_header,
            SsaInstructionV2::Branch {
                condition,
                true_block: loop_body,
                false_block: exit_block,
                args: vec![loop_index_param], // Pass the block parameter
            },
        );

        // Convert the loop body statements
        let old_current_block = self.current_block;
        self.current_block = loop_body;

        // Update the loop variable mapping to point to the loop body parameter
        let old_loop_var = self
            .local_mapping
            .insert(index_var.to_string(), loop_body_index_param);

        // Convert each statement in the loop body
        for statement in body {
            self.convert_statement(statement)?;
        }

        // Restore the loop variable mapping
        if let Some(old_val) = old_loop_var {
            self.local_mapping.insert(index_var.to_string(), old_val);
        } else {
            self.local_mapping.remove(index_var);
        }

        // Loop body: increment and jump back (use the body block parameter)
        let one = self.ssa_program.new_value(crate::ssa::DataType::U64);
        self.ssa_program.add_instruction(
            loop_body,
            SsaInstructionV2::Constant {
                dest: one,
                value: 1,
                data_type: crate::ssa::DataType::U64,
            },
        );

        let next_index = self.ssa_program.new_value(crate::ssa::DataType::U64);
        self.ssa_program.add_instruction(
            loop_body,
            SsaInstructionV2::BinaryOp {
                dest: next_index,
                op: crate::ssa::BinaryOpKind::Add,
                lhs: loop_body_index_param, // Use the body block parameter
                rhs: one,
            },
        );

        self.ssa_program.add_instruction(
            loop_body,
            SsaInstructionV2::Jump {
                target_block: loop_header,
                args: vec![next_index],
            },
        );

        // Continue from exit block
        self.current_block = exit_block;

        Ok(())
    }

    fn convert_expression(&mut self, expr: &Expression) -> Result<SsaValue, DioError> {
        match expr {
            Expression::Variable(name) => self.get_variable(name),
            Expression::Literal(value) => {
                let ssa_value = self.ssa_program.new_value(crate::ssa::DataType::I64);
                self.ssa_program.add_instruction(
                    self.current_block,
                    SsaInstructionV2::Constant {
                        dest: ssa_value,
                        value: *value,
                        data_type: crate::ssa::DataType::I64,
                    },
                );
                Ok(ssa_value)
            }
            Expression::ArrayAccess { array, index } => {
                let array_ssa = self.get_variable(array)?;
                let index_ssa = self.convert_expression(index)?;

                // Calculate element address: GEP(array_ptr, index)
                let element_addr = self.ssa_program.new_value(SsaDataType::U64);
                self.ssa_program.add_instruction(
                    self.current_block,
                    SsaInstructionV2::GetElementPtr {
                        dest: element_addr,
                        address: array_ssa,
                        index: index_ssa,
                        element_size: 8, // Assuming 64-bit elements
                    },
                );

                // Load the value from the calculated address
                let element_value = self.ssa_program.new_value(SsaDataType::U64);
                self.ssa_program.add_instruction(
                    self.current_block,
                    SsaInstructionV2::Load {
                        dest: element_value,
                        address: element_addr,
                        offset: 0,
                        data_type: SsaDataType::U64,
                    },
                );

                Ok(element_value)
            }
            Expression::BinaryOp { op, left, right } => {
                let lhs = self.convert_expression(left)?;
                let rhs = self.convert_expression(right)?;
                let result = self.ssa_program.new_value(crate::ssa::DataType::U64);

                let ssa_op = match op {
                    BinaryOperator::Add => crate::ssa::BinaryOpKind::Add,
                    BinaryOperator::Sub => crate::ssa::BinaryOpKind::Sub,
                    BinaryOperator::Mul => crate::ssa::BinaryOpKind::Mul,
                    BinaryOperator::Div => crate::ssa::BinaryOpKind::Div,
                    BinaryOperator::Lt => crate::ssa::BinaryOpKind::Lt,
                };

                self.ssa_program.add_instruction(
                    self.current_block,
                    SsaInstructionV2::BinaryOp {
                        dest: result,
                        op: ssa_op,
                        lhs,
                        rhs,
                    },
                );

                Ok(result)
            }
        }
    }

    fn get_variable(&self, name: &str) -> Result<SsaValue, DioError> {
        if let Some(&value) = self.local_mapping.get(name) {
            Ok(value)
        } else if let Some(&value) = self.param_mapping.get(name) {
            Ok(value)
        } else {
            Err(DioError::Compilation(format!("Unknown variable: {}", name)))
        }
    }
}

fn convert_bytecode_to_ssa_datatype(data_type: &DataType) -> SsaDataType {
    match data_type {
        DataType::U64 => SsaDataType::U64,
        DataType::I64 => SsaDataType::I64,
        DataType::ArrayU64 => SsaDataType::ArrayU64,
        DataType::ArrayI64 => SsaDataType::ArrayI64,
    }
}

/// Display implementations for C-like syntax debugging
impl fmt::Display for DataType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DataType::U64 => write!(f, "u64"),
            DataType::I64 => write!(f, "i64"),
            DataType::ArrayU64 => write!(f, "u64[]"),
            DataType::ArrayI64 => write!(f, "i64[]"),
        }
    }
}

impl fmt::Display for InputParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.data_type, self.name)
    }
}

impl fmt::Display for LocalVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.data_type, self.name)
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::Literal(value) => write!(f, "{}", value),
            Expression::ArrayAccess { array, index } => write!(f, "{}[{}]", array, index),
            Expression::BinaryOp { op, left, right } => {
                write!(f, "({} {} {})", left, op, right)
            }
        }
    }
}

impl fmt::Display for BinaryOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinaryOperator::Add => write!(f, "+"),
            BinaryOperator::Sub => write!(f, "-"),
            BinaryOperator::Mul => write!(f, "*"),
            BinaryOperator::Div => write!(f, "/"),
            BinaryOperator::Lt => write!(f, "<"),
        }
    }
}

impl fmt::Display for Statement {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Statement::Assign { target, expr } => write!(f, "{} = {};", target, expr),
            Statement::ArrayAssign {
                array,
                index,
                value,
            } => {
                write!(f, "{}[{}] = {};", array, index, value)
            }
            Statement::ForLoop {
                index_var,
                start,
                end,
                step,
                body,
            } => {
                writeln!(
                    f,
                    "for ({} = {}; {} < {}; {} += {}) {{",
                    index_var, start, index_var, end, index_var, step
                )?;
                for stmt in body {
                    writeln!(f, "  {}", stmt)?;
                }
                write!(f, "}}")
            }
            Statement::Return { value } => match value {
                Some(v) => write!(f, "return {};", v),
                None => write!(f, "return;"),
            },
        }
    }
}

impl fmt::Display for ByteCodeProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Function signature
        write!(f, "function(")?;
        for (i, input) in self.inputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", input)?;
        }
        writeln!(f, ") -> {} {{", self.return_type)?;

        // Local variables
        if !self.locals.is_empty() {
            writeln!(f, "  // Local variables:")?;
            for local in &self.locals {
                writeln!(f, "  {};", local)?;
            }
            writeln!(f)?;
        }

        // Statements
        for stmt in &self.statements {
            writeln!(f, "  {}", stmt)?;
        }

        write!(f, "}}")
    }
}

/// Create a complete pipeline function: AST -> ByteCode -> SSA v2 with optional debug tracing
pub fn ast_to_ssa_v2_via_bytecode(expr: &Expr) -> Result<SsaProgramV2, DioError> {
    if std::env::var("DIO_DEBUG_PIPELINE").is_ok() {
        println!("=== PIPELINE DEBUG ===");
        println!("--- Input AST ---");
        println!("{}", expr);
        println!();
    }

    let bytecode = ast_to_bytecode(expr)?;

    if std::env::var("DIO_DEBUG_PIPELINE").is_ok() {
        println!("--- ByteCode (C-like) ---");
        println!("{}", bytecode);
        println!();
    }

    let ssa_program = bytecode_to_ssa_v2(&bytecode)?;

    if std::env::var("DIO_DEBUG_PIPELINE").is_ok() {
        println!("--- SSA v2 ---");
        println!("Entry block: {:?}", ssa_program.entry_block);
        for (i, block) in ssa_program.blocks.iter().enumerate() {
            let block_id = BlockId(i as u32); // Reconstruct BlockId from index
            println!("Block {:?}:", block_id);
            if !block.parameters.is_empty() {
                print!("  Parameters: ");
                for (j, (value, data_type)) in block.parameters.iter().enumerate() {
                    if j > 0 {
                        print!(", ");
                    }
                    print!("{:?}: {:?}", value, data_type);
                }
                println!();
            }
            for instruction in &block.instructions {
                println!("  {:?}", instruction);
            }
        }
        println!("======================");
    }

    Ok(ssa_program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_expr;

    #[test]
    fn test_bytecode_simple_add() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let bytecode = ast_to_bytecode(&expr).unwrap();

        // Should have: a, b, length parameters (output is now handled via direct assignments)
        assert_eq!(bytecode.inputs.len(), 3);
        assert_eq!(bytecode.return_type, DataType::ArrayU64);
    }

    #[test]
    fn test_bytecode_sum_reduction() {
        let expr = parse_expr("(lambda ([U64Array a] U64) (sum a))").unwrap();
        let bytecode = ast_to_bytecode(&expr).unwrap();

        // Should have: a, length parameters
        assert_eq!(bytecode.inputs.len(), 2);
        assert_eq!(bytecode.return_type, DataType::U64);
    }

    #[test]
    fn test_bytecode_to_ssa_simple_add() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let bytecode = ast_to_bytecode(&expr).unwrap();
        let ssa = bytecode_to_ssa_v2(&bytecode).unwrap();

        // Should create SSA program with blocks
        assert!(!ssa.blocks.is_empty());
    }

    #[test]
    fn test_full_pipeline_simple() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let ssa = ast_to_ssa_v2_via_bytecode(&expr).unwrap();

        // Test the full pipeline: Lisp -> ByteCode -> SSA v2
        assert!(!ssa.blocks.is_empty());
    }
}
