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
    current_loop_index: Option<String>, // Track the current loop index variable
}

impl ByteCodeCompiler {
    fn new() -> Self {
        Self {
            inputs: Vec::new(),
            locals: Vec::new(),
            statements: Vec::new(),
            current_loop_index: None,
        }
    }

    /// Analyze the type of an expression given the current variable context
    fn analyze_expression_type(&self, expr: &Expr) -> Result<DataType, DioError> {
        match expr {
            Expr::Literal(_) => Ok(DataType::I64), // Constants are scalars
            Expr::Column(name) => {
                // Check if it's an input parameter
                for input in &self.inputs {
                    if input.name == *name {
                        return Ok(input.data_type.clone());
                    }
                }
                // Check if it's a local variable (from let bindings)
                for local in &self.locals {
                    if local.name == *name {
                        return Ok(local.data_type.clone());
                    }
                }
                Err(DioError::Compilation(format!("Unknown variable: {}", name)))
            }
            Expr::Sum(_) | Expr::Count(_) => Ok(DataType::I64), // Reductions produce scalars
            Expr::Add(operands) | Expr::Mul(operands) => {
                // For n-ary operations, if any operand is an array, result is an array
                let mut has_array = false;
                for operand in operands {
                    let operand_type = self.analyze_expression_type(operand)?;
                    if operand_type.is_array() {
                        has_array = true;
                        break;
                    }
                }
                if has_array {
                    Ok(DataType::ArrayI64) // Broadcast to array
                } else {
                    Ok(DataType::I64) // All scalars
                }
            }
            Expr::Sub(lhs, rhs) | Expr::Div(lhs, rhs) => {
                // For binary operations, if either operand is an array, result is an array
                let lhs_type = self.analyze_expression_type(lhs)?;
                let rhs_type = self.analyze_expression_type(rhs)?;
                if lhs_type.is_array() || rhs_type.is_array() {
                    Ok(DataType::ArrayI64) // Broadcast to array
                } else {
                    Ok(DataType::I64) // Both scalars
                }
            }
            _ => Err(DioError::Compilation(format!("Cannot analyze type for expression: {:?}", expr)))
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
            Expr::Let { bindings, body } => {
                // Let bindings: compile each binding and then the body with all variables in scope
                self.compile_let_bindings(bindings, body, &ret_type)?;
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
        let index_var = "loop_idx".to_string();
        self.locals.push(LocalVar {
            name: index_var.clone(),
            data_type: DataType::U64,
        });

        // Create the for loop: for(i = 0; i < length; i++)
        let mut loop_body = Vec::new();

        // Set the current loop index context before compiling expressions
        let old_loop_index = self.current_loop_index.clone();
        self.current_loop_index = Some(index_var.clone());

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
        
        // Restore the previous loop index context
        self.current_loop_index = old_loop_index;

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

    /// Compile a scalar operation that needs to be broadcast to all array elements
    fn compile_scalar_broadcast_operation(&mut self, expr: &Expr) -> Result<(), DioError> {
        // Add loop index variable
        let index_var = "loop_idx".to_string();
        self.locals.push(LocalVar {
            name: index_var.clone(),
            data_type: DataType::U64,
        });

        // Create the for loop: for(i = 0; i < length; i++)
        let mut loop_body = Vec::new();

        // Compile the scalar expression once
        let scalar_expr = self.compile_expression(expr)?;
        
        // output[i] = scalar_expr (broadcast to all elements)
        loop_body.push(Statement::ArrayAssign {
            array: "output".to_string(),
            index: Expression::Variable(index_var.clone()),
            value: scalar_expr,
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

    /// Compile elementwise operations with support for scalar broadcasting
    fn compile_elementwise_operation_with_broadcast(
        &mut self,
        operands: &[Expr],
        op: BinaryOperator,
    ) -> Result<(), DioError> {
        // Add loop index variable
        let index_var = "loop_idx".to_string();
        self.locals.push(LocalVar {
            name: index_var.clone(),
            data_type: DataType::U64,
        });

        // Create the for loop: for(i = 0; i < length; i++)
        let mut loop_body = Vec::new();

        // Set the current loop index context before compiling expressions
        let old_loop_index = self.current_loop_index.clone();
        self.current_loop_index = Some(index_var.clone());

        // Build the expression for the operation with broadcasting support
        let mut expr = self.compile_expression_with_broadcast(&operands[0])?;
        for operand in &operands[1..] {
            let right_expr = self.compile_expression_with_broadcast(operand)?;
            expr = Expression::BinaryOp {
                op: op.clone(),
                left: Box::new(expr),
                right: Box::new(right_expr),
            };
        }
        
        // Restore the previous loop index context
        self.current_loop_index = old_loop_index;

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

    /// Compile expression with support for scalar broadcasting
    fn compile_expression_with_broadcast(&mut self, expr: &Expr) -> Result<Expression, DioError> {
        let expr_type = self.analyze_expression_type(expr)?;
        
        match expr {
            Expr::Column(name) => {
                if expr_type.is_array() {
                    // Array access: array[i]
                    Ok(Expression::ArrayAccess {
                        array: name.clone(),
                        index: Box::new(Expression::Variable(
                            self.current_loop_index.clone().unwrap_or("loop_idx".to_string())
                        )),
                    })
                } else {
                    // Scalar variable: just use the value directly (broadcasting)
                    Ok(Expression::Variable(name.clone()))
                }
            }
            Expr::Literal(value) => {
                // Literals are always scalars (broadcasting)
                let literal_val = match value {
                    crate::ast::Value::Int64(i) => *i,
                    crate::ast::Value::Float64(f) => f.0 as i64,
                };
                Ok(Expression::Literal(literal_val))
            }
            Expr::Mul(operands) => {
                // Handle multiplication of scalars within broadcasting context
                if expr_type.is_scalar() {
                    // Pure scalar multiplication - compile all operands as scalars
                    let mut result = self.compile_expression_with_broadcast(&operands[0])?;
                    for operand in &operands[1..] {
                        let right = self.compile_expression_with_broadcast(operand)?;
                        result = Expression::BinaryOp {
                            op: BinaryOperator::Mul,
                            left: Box::new(result),
                            right: Box::new(right),
                        };
                    }
                    Ok(result)
                } else {
                    // Mixed scalar-array multiplication - shouldn't reach here in current design
                    self.compile_expression(expr)
                }
            }
            _ => {
                // For now, delegate to the original compile_expression for complex cases
                self.compile_expression(expr)
            }
        }
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
        let index_var = "loop_idx".to_string();
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

    /// Compile let bindings as sequential variable declarations
    fn compile_let_bindings(
        &mut self,
        bindings: &[(Type, String, Expr)],
        body: &Expr,
        ret_type: &DataType,
    ) -> Result<(), DioError> {
        // For each binding, compile it and store the result in a local variable
        for (type_, var_name, binding_expr) in bindings {
            // Convert AST type to bytecode data type
            let data_type = match type_ {
                Type::U64 => DataType::U64,
                Type::I64 => DataType::I64,
                Type::U64Array => DataType::ArrayU64,
                Type::I64Array => DataType::ArrayI64,
                Type::F64 => DataType::I64, // Treat F64 as I64 for now
                Type::F64Array => DataType::ArrayI64, // Treat F64Array as ArrayI64 for now
            };
            
            // Validate that the binding expression makes sense with the declared type
            match binding_expr {
                Expr::Sum(_) | Expr::Count(_) => {
                    if !data_type.is_scalar() {
                        return Err(DioError::Compilation(
                            format!("Reductive function {} produces a scalar but variable {} is declared as array type {}", 
                                   if matches!(binding_expr, Expr::Sum(_)) { "sum" } else { "count" },
                                   var_name, type_)
                        ));
                    }
                }
                Expr::Literal(_) => {
                    if !data_type.is_scalar() {
                        return Err(DioError::Compilation(
                            format!("Literal constant produces a scalar but variable {} is declared as array type", var_name)
                        ));
                    }
                }
                _ => {
                    return Err(DioError::Compilation(
                        "Only reductive functions (sum, count) and literal constants are allowed in let bindings".to_string(),
                    ));
                }
            }
            
            // Create a local variable for this binding
            self.locals.push(LocalVar {
                name: var_name.clone(),
                data_type: data_type.clone(),
            });
            
            // Compile the binding expression and assign to the variable
            match binding_expr {
                Expr::Sum(inner) => {
                    self.compile_reduction_to_variable(inner, var_name, &data_type)?;
                }
                Expr::Count(inner) => {
                    self.compile_count_to_variable(inner, var_name)?;
                }
                Expr::Literal(value) => {
                    // Assign the literal value to the variable
                    let literal_val = match value {
                        crate::ast::Value::Int64(i) => *i,
                        crate::ast::Value::Float64(f) => f.0 as i64, // Convert to i64 for now
                    };
                    self.statements.push(Statement::Assign {
                        target: var_name.clone(),
                        expr: Expression::Literal(literal_val),
                    });
                }
                _ => unreachable!(), // Already checked above
            }
        }
        
        // Now compile the body with all bindings in scope
        self.compile_lambda_body_recursive(body, ret_type)?;
        
        Ok(())
    }
    
    /// Compile a sum reduction and store the result in a named variable
    fn compile_reduction_to_variable(
        &mut self,
        inner: &Expr,
        result_var: &str,
        ret_type: &DataType,
    ) -> Result<(), DioError> {
        // Add accumulator variable (use a unique name to avoid conflicts)
        let acc_var = format!("{}_acc", result_var);
        self.locals.push(LocalVar {
            name: acc_var.clone(),
            data_type: ret_type.clone(),
        });

        // Add loop index variable (use a unique name to avoid conflicts)
        let index_var = format!("{}_i", result_var);
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

        // Set the current loop index context before compiling the inner expression
        let old_loop_index = self.current_loop_index.clone();
        self.current_loop_index = Some(index_var.clone());
        
        // Compile the inner expression (e.g., a[i] + b[i] for sum(+ a b))
        let inner_expr = self.compile_expression(inner)?;
        
        // Restore the previous loop index context
        self.current_loop_index = old_loop_index;

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
            index_var: index_var.clone(),
            start: Expression::Literal(0),
            end: Expression::Variable("length".to_string()),
            step: Expression::Literal(1),
            body: loop_body,
        });

        // Store the final result in the result variable
        self.statements.push(Statement::Assign {
            target: result_var.to_string(),
            expr: Expression::Variable(acc_var),
        });

        Ok(())
    }
    
    /// Compile a count reduction and store the result in a named variable
    fn compile_count_to_variable(
        &mut self,
        inner: &Expr,
        result_var: &str,
    ) -> Result<(), DioError> {
        // Add accumulator variable (use a unique name to avoid conflicts)
        let acc_var = format!("{}_acc", result_var);
        self.locals.push(LocalVar {
            name: acc_var.clone(),
            data_type: DataType::U64,
        });

        // Add loop index variable (use a unique name to avoid conflicts)
        let index_var = format!("{}_i", result_var);
        self.locals.push(LocalVar {
            name: index_var.clone(),
            data_type: DataType::U64,
        });

        // Initialize counter: acc = 0
        self.statements.push(Statement::Assign {
            target: acc_var.clone(),
            expr: Expression::Literal(0),
        });

        // Create the for loop: for(i = 0; i < length; i++)
        let mut loop_body = Vec::new();
        
        // Set the current loop index context before compiling the inner expression
        let old_loop_index = self.current_loop_index.clone();
        self.current_loop_index = Some(index_var.clone());
        
        let _inner_expr = self.compile_expression(inner)?;
        
        // Restore the previous loop index context
        self.current_loop_index = old_loop_index;

        // For count, just increment the counter (assuming all values are non-null for now)
        loop_body.push(Statement::Assign {
            target: acc_var.clone(),
            expr: Expression::BinaryOp {
                op: BinaryOperator::Add,
                left: Box::new(Expression::Variable(acc_var.clone())),
                right: Box::new(Expression::Literal(1)),
            },
        });

        // Add the for loop
        self.statements.push(Statement::ForLoop {
            index_var: index_var.clone(),
            start: Expression::Literal(0),
            end: Expression::Variable("length".to_string()),
            step: Expression::Literal(1),
            body: loop_body,
        });

        // Store the final result in the result variable
        self.statements.push(Statement::Assign {
            target: result_var.to_string(),
            expr: Expression::Variable(acc_var),
        });

        Ok(())
    }
    
    /// Recursively compile lambda body expressions (for let binding support)
    fn compile_lambda_body_recursive(&mut self, body: &Expr, ret_type: &DataType) -> Result<(), DioError> {
        match body {
            Expr::Add(operands) => {
                // Analyze the actual type of this expression based on operands
                let expr_type = self.analyze_expression_type(body)?;
                
                // If the return type is an array, we need to broadcast even scalar operations
                if ret_type.is_array() {
                    if expr_type.is_array() {
                        // Mixed or array-array operations -> elementwise with broadcasting
                        self.compile_elementwise_operation_with_broadcast(operands, BinaryOperator::Add)?;
                    } else {
                        // Pure scalar operation that needs to be broadcast to array
                        self.compile_scalar_broadcast_operation(body)?;
                    }
                } else {
                    // Pure scalar operation for scalar return type
                    let result_expr = self.compile_expression(body)?;
                    self.statements.push(Statement::ArrayAssign {
                        array: "output".to_string(),
                        index: Expression::Literal(0), // Store scalar at index 0
                        value: result_expr,
                    });
                    self.statements.push(Statement::Return { value: None });
                }
            }
            Expr::Mul(operands) => {
                // Handle multiplication similar to addition
                let expr_type = self.analyze_expression_type(body)?;
                
                if ret_type.is_array() {
                    if expr_type.is_array() {
                        // Mixed or array-array operations -> elementwise with broadcasting
                        self.compile_elementwise_operation_with_broadcast(operands, BinaryOperator::Mul)?;
                    } else {
                        // Pure scalar operation that needs to be broadcast to array
                        self.compile_scalar_broadcast_operation(body)?;
                    }
                } else {
                    // Pure scalar operation for scalar return type
                    let result_expr = self.compile_expression(body)?;
                    self.statements.push(Statement::ArrayAssign {
                        array: "output".to_string(),
                        index: Expression::Literal(0), // Store scalar at index 0
                        value: result_expr,
                    });
                    self.statements.push(Statement::Return { value: None });
                }
            }
            Expr::Sum(inner) => {
                self.compile_reduction_operation(inner, ret_type)?;
            }
            Expr::Let { bindings, body } => {
                self.compile_let_bindings(bindings, body, ret_type)?;
            }
            Expr::Column(name) => {
                // Direct variable reference (e.g., just returning a let-bound variable)
                let var_expr = self.compile_expression(body)?;
                self.statements.push(Statement::ArrayAssign {
                    array: "output".to_string(),
                    index: Expression::Literal(0), // Store scalar at index 0
                    value: var_expr,
                });
                self.statements.push(Statement::Return { value: None });
            }
            _ => {
                return Err(DioError::Compilation(
                    format!("Unsupported operation in let body: {:?}", body)
                ));
            }
        }
        Ok(())
    }
    

    /// Compile an expression to a ByteCode expression
    fn compile_expression(&mut self, expr: &Expr) -> Result<Expression, DioError> {
        match expr {
            Expr::Column(name) => {
                // Check if this is a reference to a let-bound variable (locals have precedence)
                if self.locals.iter().any(|local| local.name == *name && local.data_type.is_scalar()) {
                    // This is a scalar variable reference (from let binding)
                    Ok(Expression::Variable(name.clone()))
                } else {
                    // For array access, return array[i] where i is the current loop index
                    let index_var = self.current_loop_index.as_ref()
                        .cloned()
                        .unwrap_or_else(|| "loop_idx".to_string()); // Fallback to "loop_idx" for backward compatibility
                    Ok(Expression::ArrayAccess {
                        array: name.clone(),
                        index: Box::new(Expression::Variable(index_var)),
                    })
                }
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
            Expr::Let { .. } => {
                Err(DioError::Compilation(
                    "Let expressions not supported in arithmetic contexts - use at lambda body level".to_string()
                ))
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

        // Detect accumulator variables that are updated inside the loop
        let mut loop_variables = std::collections::HashMap::new();
        let mut initial_values = std::collections::HashMap::new();
        
        for statement in body {
            if let Statement::Assign { target, expr } = statement {
                // Check if this is an accumulator pattern: acc = acc + something
                if let Expression::BinaryOp { op: BinaryOperator::Add, left, right: _ } = expr {
                    if let Expression::Variable(var_name) = left.as_ref() {
                        if var_name == target {
                            // This is an accumulator update: var = var + something
                            // Check if we have an initial value for this variable
                            if let Some(&initial_val) = self.local_mapping.get(target) {
                                loop_variables.insert(target.clone(), crate::ssa::DataType::U64);
                                initial_values.insert(target.clone(), initial_val);
                            }
                        }
                    }
                }
            }
        }

        // Create loop parameters: index + any accumulator variables
        let loop_index_param = self.ssa_program.new_value(crate::ssa::DataType::U64);
        let loop_body_index_param = self.ssa_program.new_value(crate::ssa::DataType::U64);
        
        let mut loop_header_params = vec![(loop_index_param, crate::ssa::DataType::U64)];
        let mut loop_body_params = vec![(loop_body_index_param, crate::ssa::DataType::U64)];
        let mut loop_var_params = std::collections::HashMap::new();
        let mut loop_body_var_params = std::collections::HashMap::new();

        for (var_name, data_type) in &loop_variables {
            let header_param = self.ssa_program.new_value(data_type.clone());
            let body_param = self.ssa_program.new_value(data_type.clone());
            loop_header_params.push((header_param, data_type.clone()));
            loop_body_params.push((body_param, data_type.clone()));
            loop_var_params.insert(var_name.clone(), header_param);
            loop_body_var_params.insert(var_name.clone(), body_param);
        }

        // Create loop header and body blocks
        let loop_header = self.ssa_program.new_block(loop_header_params);
        let loop_body = self.ssa_program.new_block(loop_body_params);
        let exit_block = self.ssa_program.new_block(vec![]);

        // Jump from entry to loop header (pass index + initial accumulator values)
        let mut initial_args = vec![start_value];
        for (var_name, _) in &loop_variables {
            if let Some(&initial_val) = initial_values.get(var_name) {
                initial_args.push(initial_val);
            }
        }
        
        self.ssa_program.add_instruction(
            self.current_block,
            SsaInstructionV2::Jump {
                target_block: loop_header,
                args: initial_args,
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

        // Branch: pass index + current accumulator values to loop body
        let mut branch_args = vec![loop_index_param];
        for (var_name, _) in &loop_variables {
            if let Some(&header_param) = loop_var_params.get(var_name) {
                branch_args.push(header_param);
            }
        }
        
        self.ssa_program.add_instruction(
            loop_header,
            SsaInstructionV2::Branch {
                condition,
                true_block: loop_body,
                false_block: exit_block,
                args: branch_args,
            },
        );

        // Convert the loop body statements
        let old_current_block = self.current_block;
        self.current_block = loop_body;

        // Update variable mappings to point to loop body parameters
        let old_loop_var = self
            .local_mapping
            .insert(index_var.to_string(), loop_body_index_param);

        let mut old_accumulator_vars = std::collections::HashMap::new();
        for (var_name, _) in &loop_variables {
            if let Some(&body_param) = loop_body_var_params.get(var_name) {
                let old_val = self.local_mapping.insert(var_name.clone(), body_param);
                old_accumulator_vars.insert(var_name.clone(), old_val);
            }
        }

        // Convert each statement in the loop body
        for statement in body {
            self.convert_statement(statement)?;
        }

        // Collect updated accumulator values after processing the loop body
        let mut updated_accumulator_values = std::collections::HashMap::new();
        for (var_name, _) in &loop_variables {
            if let Some(&updated_val) = self.local_mapping.get(var_name) {
                updated_accumulator_values.insert(var_name.clone(), updated_val);
            }
        }

        // Restore variable mappings
        if let Some(old_val) = old_loop_var {
            self.local_mapping.insert(index_var.to_string(), old_val);
        } else {
            self.local_mapping.remove(index_var);
        }

        for (var_name, old_val) in old_accumulator_vars {
            if let Some(old_val) = old_val {
                self.local_mapping.insert(var_name, old_val);
            } else {
                self.local_mapping.remove(&var_name);
            }
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

        // Jump back to loop header with updated index and accumulator values
        let mut jump_back_args = vec![next_index];
        for (var_name, _) in &loop_variables {
            if let Some(&updated_val) = updated_accumulator_values.get(var_name) {
                jump_back_args.push(updated_val);
            }
        }
        
        self.ssa_program.add_instruction(
            loop_body,
            SsaInstructionV2::Jump {
                target_block: loop_header,
                args: jump_back_args,
            },
        );

        // Continue from exit block
        self.current_block = exit_block;
        
        // Update variable mappings to point to the loop header parameters (final accumulator values)
        for (var_name, _) in &loop_variables {
            if let Some(&header_param) = loop_var_params.get(var_name) {
                self.local_mapping.insert(var_name.clone(), header_param);
            }
        }

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

    #[test]
    fn test_let_binding_single_bytecode() {
        // Test let binding with single typed reductive function
        let expr = parse_expr("(lambda ([U64Array b] U64) (let [U64 s (sum b)] s))").unwrap();
        let bytecode = ast_to_bytecode(&expr).unwrap();

        // Should successfully compile let bindings with reductive functions
        assert_eq!(bytecode.return_type, DataType::U64);
        assert!(!bytecode.locals.is_empty()); // Should have local variable for let binding
    }

    #[test]
    fn test_let_binding_multiple_bytecode() {
        // Test let binding with multiple typed reductive functions
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64) (let [U64 s (sum a) U64 c (count b)] (+ s c)))").unwrap();
        let bytecode = ast_to_bytecode(&expr).unwrap();

        // Should successfully compile multiple let bindings
        assert_eq!(bytecode.return_type, DataType::U64);
        // Should have local variables for both bindings plus internal temporaries
        assert!(bytecode.locals.len() >= 2);
    }

    #[test]
    fn test_let_binding_rejects_elementwise() {
        // Test that let bindings reject elementwise operations
        let expr_result = parse_expr("(lambda ([U64Array b] [U64Array c] U64) (let [U64 a (+ b c)] (sum a)))");
        // This should fail at the parser level now
        assert!(expr_result.is_err(), "Expected parser to reject elementwise let binding");
    }

    #[test]
    fn test_let_binding_with_constants() {
        // Test constants in let bindings
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (let [I64 x -1234 U64 y 567] (+ x y)))").unwrap();
        let bytecode = ast_to_bytecode(&expr).unwrap();
        
        // Should have a, b, length as inputs
        assert_eq!(bytecode.inputs.len(), 3);
        
        // Should have local variables for the constants plus loop index
        // x: I64, y: U64, i: U64 for the loop (if it's pure scalar it may not have a loop)
        assert!(bytecode.locals.len() >= 2);
        
        // Check that constants are properly assigned
        let x_local = bytecode.locals.iter().find(|l| l.name == "x");
        let y_local = bytecode.locals.iter().find(|l| l.name == "y");
        assert!(x_local.is_some());
        assert!(y_local.is_some());
        assert_eq!(x_local.unwrap().data_type, DataType::I64);
        assert_eq!(y_local.unwrap().data_type, DataType::U64);
    }

    #[test]
    fn test_scalar_array_broadcasting() {
        // Test scalar-array broadcasting: (+ s b) where s is scalar, b is array
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (let [U64 s (sum a)] (+ s b)))").unwrap();
        let bytecode = ast_to_bytecode(&expr).unwrap();
        
        // Should have a, b, length as inputs
        assert_eq!(bytecode.inputs.len(), 3);
        
        // Should have local variables: s (from sum), loop variables for reduction and main loop
        assert!(bytecode.locals.len() >= 2);
        
        // Should have a scalar variable s
        let s_local = bytecode.locals.iter().find(|l| l.name == "s");
        assert!(s_local.is_some());
        assert_eq!(s_local.unwrap().data_type, DataType::U64);
    }

    #[test] 
    fn test_all_requested_functionality() {
        // Test all requested functionality in one comprehensive test
        
        // Example 1: (lambda ([U64Array a] [U64Array b] U64Array) (let [U64 s (sum a)] (+ s b)))
        let expr1 = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (let [U64 s (sum a)] (+ s b)))").unwrap();
        let bytecode1 = ast_to_bytecode(&expr1).unwrap();
        assert!(bytecode1.inputs.len() == 3); // a, b, length
        assert!(bytecode1.locals.iter().any(|l| l.name == "s" && l.data_type == DataType::U64));
        
        // Example 2: (let [I64 x -1234 U64 y 567] (+ x y))
        let expr2 = parse_expr("(lambda ([U64Array a] U64) (let [I64 x -1234 U64 y 567] (+ x y)))").unwrap();
        let bytecode2 = ast_to_bytecode(&expr2).unwrap();
        assert!(bytecode2.locals.iter().any(|l| l.name == "x" && l.data_type == DataType::I64));
        assert!(bytecode2.locals.iter().any(|l| l.name == "y" && l.data_type == DataType::U64));
        
        // Example 3: (lambda ([U64Array a] [U64Array b] U64Array) (let [U64 s (sum a) I64 i -1] (+ (* s i) b)))
        let expr3 = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (let [U64 s (sum a) I64 i -1] (+ (* s i) b)))").unwrap();
        let bytecode3 = ast_to_bytecode(&expr3).unwrap();
        assert!(bytecode3.inputs.len() == 3); // a, b, length
        assert!(bytecode3.locals.iter().any(|l| l.name == "s" && l.data_type == DataType::U64));
        assert!(bytecode3.locals.iter().any(|l| l.name == "i" && l.data_type == DataType::I64));
    }

    #[test]
    fn test_let_binding_full_pipeline() {
        // Test the full pipeline with let bindings using typed reductive functions
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64) (let [U64 s (sum a) U64 c (count b)] (+ s c)))").unwrap();
        let ssa = ast_to_ssa_v2_via_bytecode(&expr).unwrap();

        // Should create SSA program with blocks
        assert!(!ssa.blocks.is_empty());
    }
}
