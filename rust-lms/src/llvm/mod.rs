//! LLVM/MLIR backend (docs/llvm.md, Phase 1+).
//!
//! **Feature-gated behind `--features llvm`.** This module is the second code
//! generator alongside Cranelift. It needs a system LLVM/MLIR 22 install reachable
//! through melior / mlir-sys (set `MLIR_SYS_220_PREFIX`), so it is compiled *only*
//! when the `llvm` feature is on — the default pure-Rust Cranelift build never pulls
//! melior in.
//!
//! ## Where Phase 1 is
//!
//! Phase -1 (the `mlir-spike/` crate) proved the JIT/ABI/lowering pipeline against
//! *textual* MLIR. Phase 1 proves the same pipeline built **programmatically** through
//! melior's op builders — the alpha-API risk the spike deliberately skipped — and lays
//! the groundwork for the `Backend`-shaped value-op layer:
//!
//! - [`make_context`] — the known-good context/dialect/translation setup (from the spike).
//! - [`jit_return_i64_const`] — end-to-end proof: build a nullary `() -> i64` `func.func`
//!   returning a constant with `arith.constant` + `func.return`, verify, lower to the LLVM
//!   dialect, JIT with `ExecutionEngine`, look up the **native** function pointer, and run
//!   it. This is the Phase 1 milestone ("a nullary `fun0` returning a constant JITs and
//!   runs via `ExecutionEngine`").
//!
//! ## The value arena (Phase 2 groundwork — findings)
//!
//! The opaque [`crate::staged::ValueId`] maps to an MLIR value the same stateless way it
//! maps to a Cranelift `Value`: melior's `Value<'c, 'a>` is `#[repr(transparent)]` over a
//! lifetime-free `mlir_sys::MlirValue`, reachable via the public `ValueLike::to_raw` and
//! reconstructable via `Value::from_raw`. So a `Vec<MlirValue>` indexed by `u32` is the
//! MLIR analogue of Cranelift's entity arena — no self-referential-struct problem, exactly
//! as docs/llvm.md §9 predicted. That arena lands with the multi-value op layer (arithmetic,
//! compares, memory) in the next increment; Phase 1 keeps values live inline.

use melior::dialect::{arith, func, DialectRegistry};
use melior::ir::attribute::{IntegerAttribute, StringAttribute, TypeAttribute};
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{Block, Location, Module, Region, RegionLike, Type};
use melior::pass::{self, PassManager};
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use melior::{Context, ExecutionEngine};

/// The optimization level passed to `ExecutionEngine`. Must be ≥ 2 so LLVM's own
/// `mem2reg` promotes our entry-block allocas to SSA (the Phase -1 spike established
/// that mlir-sys 220 exposes no MLIR-level mem2reg constructor — see docs/llvm.md §5).
const JIT_OPT_LEVEL: usize = 2;

/// A context with all dialects and LLVM translations registered — the known-good
/// setup proven by the Phase -1 spike.
pub(crate) fn make_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);
    context
}

/// Build a nullary `() -> i64` kernel that returns `value`, JIT it, and run it.
///
/// The Phase 1 end-to-end proof, built entirely through melior's op builders:
/// `arith.constant` + `func.return` inside a `func.func`, verified, lowered to the LLVM
/// dialect, and executed via the **native** function pointer from
/// `ExecutionEngine::lookup` (the Phase-3 ABI, not `invoke_packed`).
pub fn jit_return_i64_const(value: i64) -> i64 {
    let context = make_context();
    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    let i64_ty: Type = IntegerType::new(&context, 64).into();
    let fn_ty = FunctionType::new(&context, &[], &[i64_ty]);

    let function = func::func(
        &context,
        StringAttribute::new(&context, "kernel"),
        TypeAttribute::new(fn_ty.into()),
        {
            let block = Block::new(&[]);
            let constant = block
                .append_operation(arith::constant(
                    &context,
                    IntegerAttribute::new(i64_ty, value).into(),
                    location,
                ))
                .result(0)
                .expect("arith.constant has one result")
                .into();
            block.append_operation(func::r#return(&[constant], location));

            let region = Region::new();
            region.append_block(block);
            region
        },
        &[],
        location,
    );
    module.body().append_operation(function);

    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification before lowering"
    );

    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager
        .run(&mut module)
        .expect("lower to the LLVM dialect");
    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification after lowering"
    );

    let engine = ExecutionEngine::new(&module, JIT_OPT_LEVEL, &[], false, false);
    let pointer = engine.lookup("kernel");
    assert!(
        !pointer.is_null(),
        "ExecutionEngine::lookup(\"kernel\") is null"
    );

    // SAFETY: `kernel` was emitted with the nullary `() -> i64` signature and the
    // engine (which owns the executable memory) outlives this call.
    let kernel: extern "C" fn() -> i64 = unsafe { std::mem::transmute(pointer) };
    kernel()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nullary_i64_constant_jits_and_runs() {
        assert_eq!(jit_return_i64_const(42), 42);
        assert_eq!(jit_return_i64_const(-7), -7);
        assert_eq!(jit_return_i64_const(0), 0);
        assert_eq!(jit_return_i64_const(i64::MAX), i64::MAX);
    }
}
