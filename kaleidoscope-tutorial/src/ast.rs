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

/// A function's name and parameter names.
#[derive(Clone, Debug, PartialEq)]
pub struct Prototype {
    pub name: String,
    pub parameters: Vec<String>,
    pub span: Span,
}

/// An expression and the source range which produced it.
#[derive(Clone, Debug, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: Span,
}

/// The expression forms introduced through Chapter 5.
#[derive(Clone, Debug, PartialEq)]
pub enum ExprKind {
    Number(f64),
    Variable(String),
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

/// Chapter 2's four built-in binary operators.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    LessThan,
    Add,
    Subtract,
    Multiply,
}

impl BinaryOp {
    pub fn symbol(self) -> char {
        match self {
            Self::LessThan => '<',
            Self::Add => '+',
            Self::Subtract => '-',
            Self::Multiply => '*',
        }
    }
}
