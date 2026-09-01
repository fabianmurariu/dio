//! A chapter-by-chapter Kaleidoscope implementation built for `rust-lms`.
//!
//! Chapter 1 introduces the language and turns source text into a token stream.
//! Chapter 2 parses those constructs into an owned AST. Chapter 3 lowers that
//! AST to staged `rust-lms` computations and JIT-compiles them to native code.
//! Chapter 4 adds typed host externs and an interactive session while leaving
//! low-level optimization to the rust-lms backends.
//! Chapter 5 adds value-producing conditionals and `for` loops. Chapter 6
//! makes unary and binary operators user-definable with dynamic precedence.

pub mod ast;
pub mod codegen;
pub mod lexer;
pub mod parser;
pub mod runtime;
mod syntax;

pub use ast::{BinaryOp, Expr, ExprKind, Function, Item, Program, Prototype, PrototypeKind};
pub use codegen::{CodegenError, NativeNullary, compile_top_level, evaluate, validate};
pub use lexer::{LexError, Position, Span, Token, TokenKind, lex};
pub use parser::{ParseError, ParserState, parse_program};
pub use runtime::{STANDARD_EXTERN_NAMES, Session, SubmissionError};
