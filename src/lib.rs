pub mod ast;
pub mod parser;
pub mod error;
pub mod ssa;
pub mod cranelift_backend;
pub mod execution;

pub use ast::{Expr, Value};
pub use parser::parse_expr;
pub use error::{DioError, ParseError};
pub use ssa::{SsaProgram, SsaInstruction, SsaValue, BlockId};
pub use execution::{CompiledFunction, execute_add_u64};

#[cfg(test)]
mod tests;