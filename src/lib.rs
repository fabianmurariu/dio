pub mod array_support;
pub mod ast;
pub mod casting;
pub mod cranelift_backend;
pub mod error;
pub mod execution;
pub mod parser;
pub mod ssa;

pub use ast::{Expr, Type, TypedParam, Value};
pub use error::{DioError, ParseError};
pub use execution::{
    clear_function_cache, execute_generic, execute_generic_cached, CompiledFunction,
};
pub use parser::parse_expr;
pub use ssa::{BlockId, SsaInstruction, SsaProgram, SsaValue};

#[cfg(test)]
mod tests;
