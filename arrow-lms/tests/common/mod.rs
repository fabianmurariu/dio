//! Shared test helpers.

use rust_lms::prelude::*;

/// Runs `body` with a compiler for each backend: Cranelift, and LLVM with the `llvm`
/// feature.
#[allow(dead_code, reason = "not every test file uses it")]
pub fn for_each_backend(mut body: impl FnMut(Compiler)) {
    body(Compiler::new());
    #[cfg(feature = "llvm")]
    body(Compiler::new().with_backend(JitBackend::Llvm));
}

/// The backends [`for_each_backend`] runs, for tests that build several compilers.
#[allow(dead_code, reason = "not every test file builds several compilers")]
pub fn backends() -> Vec<JitBackend> {
    let mut backends = vec![JitBackend::Cranelift];
    #[cfg(feature = "llvm")]
    backends.push(JitBackend::Llvm);
    backends
}
