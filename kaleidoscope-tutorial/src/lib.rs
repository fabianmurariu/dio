//! A chapter-by-chapter Kaleidoscope implementation built for `rust-lms`.
//!
//! Chapter 1 introduces the language and turns source text into a token stream.
//! Chapter 2 parses those constructs into an owned AST. Chapter 3 lowers that
//! AST to staged `rust-lms` computations and JIT-compiles them to native code.

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
mod syntax;

pub use ast::{BinaryOp, Expr, ExprKind, Function, Item, Program, Prototype};
pub use codegen::{CodegenError, NativeNullary, compile_top_level, evaluate};
pub use lexer::{LexError, Position, Span, Token, TokenKind, lex};
pub use parser::{ParseError, parse_program};
