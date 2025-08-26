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
    // Deprecated old pipeline functions
    clear_function_cache,
    execute_generic,
    // New ByteCode pipeline (recommended)
    execute_generic_bytecode,
    execute_generic_cached,
    // Still needed for internal use
    CompiledFunction,
};
pub use parser::parse_expr;
// New ByteCode pipeline exports (recommended)
pub use bytecode::{
    ast_to_bytecode, ast_to_ssa_v2_via_bytecode, bytecode_to_ssa_v2, ByteCodeProgram,
};
// New SSA v2 types
pub use ssa::{ast_to_ssa_v2, BinaryOpKind, SsaBlockV2, SsaInstructionV2, SsaProgramV2};
// Deprecated SSA v1 types (for backward compatibility)
pub use ssa::{BlockId, SsaInstruction, SsaProgram, SsaValue};

#[cfg(test)]
mod tests;
