use crate::ast::*;
use crate::error::DioError;
use crate::ssa::{SsaProgramV2, SsaInstructionV2, BlockId, DataType as SsaDataType};
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
    Assign {
        target: String,
        expr: Expression,
    },
    
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
    Return {
        value: Option<Expression>,
    },
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
    if let Expr::Lambda { params, return_type, body } = expr {
        let mut compiler = ByteCodeCompiler::new();
        compiler.compile_lambda(params, return_type, body)
    } else {
        Err(DioError::Compilation("Only typed lambda expressions are supported".to_string()))
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
        if ret_type.is_array() {
            self.inputs.push(InputParam {
                name: "output".to_string(),
                data_type: ret_type.clone(),
            });
        }
        
        // For now, just create a simple return statement
        // TODO: Implement full expression compilation
        match body {
            Expr::Add(_) | Expr::Sub(_, _) | Expr::Mul(_) | Expr::Div(_, _) => {
                // Elementwise operation - return void
                self.statements.push(Statement::Return { value: None });
            }
            Expr::Sum(_) => {
                // Reduction - return a value
                self.statements.push(Statement::Return { 
                    value: Some(Expression::Variable("acc".to_string())) 
                });
                
                // Add accumulator local variable
                self.locals.push(LocalVar {
                    name: "acc".to_string(),
                    data_type: ret_type.clone(),
                });
            }
            _ => {
                return Err(DioError::Compilation("Unsupported operation in lambda body".to_string()));
            }
        }
        
        Ok(ByteCodeProgram {
            inputs: self.inputs.clone(),
            return_type: ret_type,
            locals: self.locals.clone(),
            statements: self.statements.clone(),
        })
    }
}

/// Convert ByteCode to SSA v2 (simplified version)
pub fn bytecode_to_ssa_v2(program: &ByteCodeProgram) -> Result<SsaProgramV2, DioError> {
    let mut ssa_program = SsaProgramV2::new();
    
    // Create entry block parameters
    let mut entry_params = Vec::new();
    for input in &program.inputs {
        let ssa_data_type = convert_bytecode_to_ssa_datatype(&input.data_type);
        let ssa_value = ssa_program.new_value(ssa_data_type.clone());
        entry_params.push((ssa_value, ssa_data_type));
    }
    
    // Create entry block
    let entry_block_id = ssa_program.new_block(entry_params);
    ssa_program.entry_block = entry_block_id;
    
    // Add a simple return instruction
    if program.return_type.is_scalar() {
        // For scalar returns (reductions), return a constant value
        let return_value = ssa_program.new_value(convert_bytecode_to_ssa_datatype(&program.return_type));
        ssa_program.add_instruction(entry_block_id, SsaInstructionV2::Constant {
            dest: return_value,
            value: 42, // Placeholder value
            data_type: convert_bytecode_to_ssa_datatype(&program.return_type),
        });
        ssa_program.add_instruction(entry_block_id, SsaInstructionV2::Return {
            value: Some(return_value),
        });
    } else {
        // For array returns, just return void
        ssa_program.add_instruction(entry_block_id, SsaInstructionV2::Return {
            value: None,
        });
    }
    
    Ok(ssa_program)
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
            Statement::ArrayAssign { array, index, value } => {
                write!(f, "{}[{}] = {};", array, index, value)
            }
            Statement::ForLoop { index_var, start, end, step, body } => {
                writeln!(f, "for ({} = {}; {} < {}; {} += {}) {{", 
                         index_var, start, index_var, end, index_var, step)?;
                for stmt in body {
                    writeln!(f, "  {}", stmt)?;
                }
                write!(f, "}}")
            }
            Statement::Return { value } => {
                match value {
                    Some(v) => write!(f, "return {};", v),
                    None => write!(f, "return;"),
                }
            }
        }
    }
}

impl fmt::Display for ByteCodeProgram {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Function signature
        write!(f, "function(")?;
        for (i, input) in self.inputs.iter().enumerate() {
            if i > 0 { write!(f, ", ")?; }
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
                    if j > 0 { print!(", "); }
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
        
        // Should have: a, b, length, output parameters
        assert_eq!(bytecode.inputs.len(), 4);
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