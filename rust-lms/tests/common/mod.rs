//! Shared test harness for running a test against every available backend.
//!
//! The neutral AST (docs/llvm.md) is meant to produce identical native code on
//! both Cranelift and MLIR/LLVM. These helpers let a single test body assert
//! that: the closure is run once per backend, so any divergence fails the test.
//!
//! Cranelift always runs. The LLVM arm is compiled in only with `--features
//! llvm` (the MLIR toolchain is optional), so a default `cargo test` behaves
//! exactly as before — one Cranelift pass.
//!
//! ## Usage
//!
//! The body must **build and exercise** the function fresh on each call — it
//! receives a fresh [`Compiler`] and owns all mutable test state inside, so the
//! two backend passes never share state:
//!
//! ```ignore
//! for_each_backend(|mut compiler| {
//!     let f = compiler.fun1("inc", |_ctx, x: Var<i64>| x + 1i64);
//!     let compiled = compiler.compile(f).unwrap();
//!     assert_eq!(compiled.as_fn().call(41), 42);
//! });
//! ```

// Not every test file uses every helper; silence per-file dead-code warnings.
#![allow(dead_code)]

use rust_lms::prelude::*;

/// Run `body` once for each backend available in this build: always Cranelift,
/// and additionally LLVM/MLIR when the `llvm` feature is enabled.
///
/// `body` gets a fresh [`Compiler`] already pointed at the backend under test.
/// Declare all mutable test state (out-params, accumulators) *inside* `body` so
/// the Cranelift and LLVM passes cannot contaminate each other.
pub fn for_each_backend(mut body: impl FnMut(Compiler)) {
    body(Compiler::new());
    #[cfg(feature = "llvm")]
    body(Compiler::new().with_backend(JitBackend::Llvm));
}

/// Like [`for_each_backend`], but for tests that need *more than one* `Compiler`
/// in a single pass (each `Compiler::compile` consumes `self`, so a test that
/// JITs two kernels needs two compilers). `body` receives a factory `make` that
/// mints a fresh `Compiler` already pointed at the backend under test, so every
/// kernel in the test runs on the *same* backend for that pass.
///
/// ```ignore
/// with_backends(|make| {
///     let fill = make().fun1(..);   // both kernels run on the current
///     let sum  = make().fun1(..);   // backend for this pass
///     ...
/// });
/// ```
pub fn with_backends(mut body: impl FnMut(&dyn Fn() -> Compiler)) {
    body(&|| Compiler::new());
    #[cfg(feature = "llvm")]
    body(&|| Compiler::new().with_backend(JitBackend::Llvm));
}
