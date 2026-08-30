//! A chapter-by-chapter Kaleidoscope implementation built for `rust-lms`.
//!
//! Chapter 1 introduces the language and turns source text into a token stream.
//! Later chapters will add the AST, lower it to staged `rust-lms` computations,
//! and JIT-compile those computations to native code.

pub mod lexer;

pub use lexer::{LexError, Position, Span, Token, TokenKind, lex};
