//! Chapter 2's syntax-independent representation of a Kaleidoscope program.

use crate::lexer::Span;

/// A complete source file or REPL submission.
#[derive(Clone, Debug, PartialEq)]
pub struct Program {
    pub items: Vec<Item>,
    pub span: Span,
}

/// A construct accepted at the top level.
#[derive(Clone, Debug, PartialEq)]
pub enum Item {
    Definition(Function),
    Extern(Prototype),
    Expression(Expr),
}

/// A named function definition.
#[derive(Clone, Debug, PartialEq)]
pub struct Function {
    pub prototype: Prototype,
    pub body: Expr,
    pub span: Span,
}

/// A function's name, parameter names, and optional operator metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct Prototype {
    pub name: String,
    pub parameters: Vec<String>,
    pub kind: PrototypeKind,
    pub span: Span,
}

/// The source form which introduced a function prototype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PrototypeKind {
    Function,
    Unary { operator: char },
    Binary { operator: char, precedence: u8 },
}

/// An expression and the source range which produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

/// The expression forms introduced through Chapter 6.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Number(f64),
    Variable(String),
    Unary {
        operator: char,
        operand: Box<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Call {
        callee: String,
        arguments: Vec<Expr>,
    },
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    For {
        variable: String,
        start: Box<Expr>,
        end: Box<Expr>,
        step: Option<Box<Expr>>,
        body: Box<Expr>,
    },
}

/// A built-in or user-defined binary operator.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    LessThan,
    Add,
    Subtract,
    Multiply,
    UserDefined(char),
}

impl BinaryOp {
    pub fn symbol(self) -> char {
        match self {
            Self::LessThan => '<',
            Self::Add => '+',
            Self::Subtract => '-',
            Self::Multiply => '*',
            Self::UserDefined(operator) => operator,
        }
    }
}
