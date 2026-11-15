//! Shared test helpers.

use rust_lms::prelude::*;

/// Runs `body` with a compiler for each backend: Cranelift, and LLVM with the `llvm`
/// feature.
pub fn for_each_backend(mut body: impl FnMut(Compiler)) {
    body(Compiler::new());
    #[cfg(feature = "llvm")]
    body(Compiler::new().with_backend(JitBackend::Llvm));
}
