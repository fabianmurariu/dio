pub mod array_support;
pub mod ast;
pub mod bytecode;
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
pub use ssa::{BlockId, SsaInstruction, SsaProgram, SsaValue, 
              BinaryOpKind, SsaInstructionV2, SsaBlockV2, SsaProgramV2, ast_to_ssa_v2};
pub use bytecode::{ByteCodeProgram, ast_to_bytecode, bytecode_to_ssa_v2};

#[cfg(test)]
mod tests;
