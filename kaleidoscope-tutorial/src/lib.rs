//! A chapter-by-chapter Kaleidoscope implementation built for `rust-lms`.
//!
//! Chapter 1 introduces the language and turns source text into a token stream.
//! Chapter 2 parses those constructs into an owned AST. Later chapters will
//! lower that AST to staged `rust-lms` computations and JIT-compile those
//! computations to native code.

pub mod ast;
pub mod lexer;
pub mod parser;
mod syntax;

pub use ast::{BinaryOp, Expr, ExprKind, Function, Item, Program, Prototype};
pub use lexer::{LexError, Position, Span, Token, TokenKind, lex};
pub use parser::{ParseError, parse_program};
