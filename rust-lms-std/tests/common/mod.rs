//! Run a test body against every backend available in this build.
//!
//! Mirrors `rust-lms/tests/common`: Cranelift always runs, and the LLVM/MLIR
//! arm is compiled in only with `--features llvm`. The body must build and
//! exercise its kernel fresh on each call, so the two passes never share state.

#![allow(dead_code)]

use rust_lms::prelude::*;

pub fn for_each_backend(mut body: impl FnMut(Compiler)) {
    body(Compiler::new());
    #[cfg(feature = "llvm")]
    body(Compiler::new().with_backend(JitBackend::Llvm));
}
