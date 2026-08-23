//! LLVM/MLIR backend (docs/llvm.md, Phase 1+).
//!
//! **Feature-gated behind `--features llvm`.** This module is the second code
//! generator alongside Cranelift. It needs a system LLVM/MLIR 22 install reachable
//! through melior / mlir-sys (set `MLIR_SYS_220_PREFIX`), so it is compiled *only*
//! when the `llvm` feature is on — the default pure-Rust Cranelift build never pulls
//! melior in.
//!
//! ## Where this is (Phases 1–3)
//!
//! Phase -1 (the `mlir-spike/` crate) proved the JIT/ABI/lowering pipeline against
//! *textual* MLIR. Phases 1–3 prove the same pipeline built **programmatically** through
//! melior's op builders — the alpha-API risk the spike deliberately skipped:
//!
//! - [`make_context`] — the known-good context/dialect/translation setup (from the spike).
//! - [`jit_return_i64_const`] — the nullary-constant milestone: a `() -> i64` `func.func`
//!   returning `arith.constant`, verified, lowered, JIT-run via the **native**
//!   `ExecutionEngine::lookup` pointer.
//! - [`MlirBackend`] — the MLIR analogue of `CraneliftBackend`, addressed by opaque
//!   [`crate::staged::ValueId`]/[`crate::staged::BlockHandle`]/[`crate::staged::VarHandle`]s:
//!   - *values (Phase 1):* constants, integer/float arithmetic, bitwise, shifts,
//!     `IntCmp`/`FloatCmp` compares, branchless `select`, sign/zero casts, `alloca`/`load`/`store`;
//!   - *variables & control flow (Phase 2, §5):* `declare_var`/`def_var`/`use_var` as
//!     entry-block `alloca` + `store`/`load`, `create_block`/`append_block_param`/
//!     `switch_to_block`/`jump`/`brif` over `cf`, and a no-op `seal_block`;
//!   - *calls & externs (Phase 3):* `declare_extern` emits a module-level `func.func
//!     private` declaration, `call` a `func.call` (value or void); the driver binds each
//!     extern to its host address with `ExecutionEngine::register_symbol`.
//!   Still an inherent API (not yet the shared [`crate::staged::Backend`] trait — that
//!   waits on the calls/signatures neutralization, next).
//! - [`jit_run_i64_unary`] — drives `MlirBackend` end to end: build a `(i64) -> i64` body
//!   (straight-line *or* with loops/branches), lower, JIT, run.
//!
//! ## The value arena (docs/llvm.md §9 — confirmed)
//!
//! The opaque [`crate::staged::ValueId`] maps to an MLIR value the same stateless way it
//! maps to a Cranelift `Value`: melior's `Value<'c, 'a>` is `#[repr(transparent)]` over a
//! lifetime-free `mlir_sys::MlirValue`, reachable via the public `ValueLike::to_raw` and
//! reconstructable via `Value::from_raw`. So [`MlirBackend`]'s `Vec<MlirValue>` indexed by
//! `ValueId` is the MLIR analogue of Cranelift's entity arena — an ordinary safe `Vec`, no
//! self-referential-struct problem: every op appends to the entry block and immediately
//! stashes the result's raw value, so the block borrow never outlives one method call.

mod backend;
use backend::scalar_to_mlir;
pub use backend::MlirBackend;

use melior::dialect::{arith, func, DialectRegistry};
use melior::ir::attribute::{IntegerAttribute, StringAttribute, TypeAttribute};
use melior::ir::block::BlockLike;
use melior::ir::operation::OperationLike;
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{Block, Location, Module, Region, RegionLike, Type};
use melior::pass::{self, PassManager};
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use melior::{Context, ExecutionEngine};

use crate::staged::ValueId;
use crate::types::ScalarType;

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

/// Lower, verify, JIT, register host `symbols`, and look up `module`'s function `name`
/// as a native pointer.
///
/// Shared tail of the JIT drivers: verify → `create_to_llvm` → verify → `ExecutionEngine`
/// → `register_symbol` (bind each extern to its host address) → `lookup`. Returns the raw
/// pointer; the caller transmutes to the concrete ABI.
fn jit_lookup(
    context: &Context,
    mut module: Module,
    name: &str,
    symbols: &[(&str, *const u8)],
) -> (ExecutionEngine, *mut ()) {
    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification before lowering"
    );
    let pass_manager = PassManager::new(context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager
        .run(&mut module)
        .expect("lower to the LLVM dialect");
    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification after lowering"
    );

    let engine = ExecutionEngine::new(&module, JIT_OPT_LEVEL, &[], false, false);
    for (symbol, address) in symbols {
        // SAFETY: each `address` is a real `extern "C"` fn pointer supplied by the caller,
        // matching the declared signature of the extern named `symbol`.
        unsafe {
            engine.register_symbol(symbol, *address as *mut ());
        }
    }
    let pointer = engine.lookup(name);
    assert!(
        !pointer.is_null(),
        "ExecutionEngine::lookup found no symbol"
    );
    (engine, pointer)
}

/// Build a `(i64) -> i64` kernel whose body is `build`, JIT it, and run it on `arg`.
///
/// The Phase 1 value-op proof: `build` emits the body through [`MlirBackend`]'s neutral
/// value ops (arithmetic, compares, casts, memory), driven by opaque [`ValueId`]s.
pub fn jit_run_i64_unary(
    build: impl for<'c> FnOnce(&mut MlirBackend<'c>, ValueId) -> ValueId,
    arg: i64,
) -> i64 {
    let context = make_context();
    let i64_ty = scalar_to_mlir(&context, ScalarType::I64);

    let mut backend = MlirBackend::new(&context, vec![i64_ty]);
    let x = backend.param(0);
    let result = build(&mut backend, x);
    backend.ret(Some(result));
    let module = backend.into_module("kernel", &[ScalarType::I64]);

    let (engine, pointer) = jit_lookup(&context, module, "kernel", &[]);
    // SAFETY: emitted with the `(i64) -> i64` signature; `engine` owns the executable
    // memory and outlives the call below.
    let kernel: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(pointer) };
    let output = kernel(arg);
    drop(engine);
    output
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::staged::Backend;
    use crate::types::IntCmp;

    #[test]
    fn nullary_i64_constant_jits_and_runs() {
        assert_eq!(jit_return_i64_const(42), 42);
        assert_eq!(jit_return_i64_const(-7), -7);
        assert_eq!(jit_return_i64_const(0), 0);
        assert_eq!(jit_return_i64_const(i64::MAX), i64::MAX);
    }

    #[test]
    fn arithmetic_on_param_and_constants() {
        // f(x) = x * 3 + 7
        let f = |b: &mut MlirBackend, x: ValueId| {
            let three = b.iconst(ScalarType::I64, 3);
            let seven = b.iconst(ScalarType::I64, 7);
            let scaled = b.imul(x, three);
            b.iadd(scaled, seven)
        };
        for x in [-4i64, 0, 5, 1000] {
            assert_eq!(jit_run_i64_unary(f, x), x * 3 + 7);
        }
    }

    #[test]
    fn compare_and_branchless_select() {
        // f(x) = if x < 0 { -x } else { x }  — abs via icmp + select, no branches.
        let f = |b: &mut MlirBackend, x: ValueId| {
            let zero = b.iconst(ScalarType::I64, 0);
            let neg = b.isub(zero, x);
            let is_neg = b.icmp(IntCmp::Slt, x, zero);
            b.select(is_neg, neg, x)
        };
        for x in [-9i64, -1, 0, 3, 42] {
            assert_eq!(jit_run_i64_unary(f, x), x.abs());
        }
    }

    #[test]
    fn memory_roundtrip_through_stack_slot() {
        // f(x) = { s = alloc 16B; store x@0; store x+1 @8; load @8 } — exercises the
        // StackSlotId model plus a non-zero byte offset (GEP-based load/store).
        let f = |b: &mut MlirBackend, x: ValueId| {
            let one = b.iconst(ScalarType::I64, 1);
            let incremented = b.iadd(x, one);
            let slot = b.alloc_stack_slot(16, 3);
            let base = b.stack_addr(slot, 0);
            b.store(x, base, 0);
            b.store(incremented, base, 8);
            b.load(ScalarType::I64, base, 8)
        };
        for x in [-1i64, 0, 41, 99] {
            assert_eq!(jit_run_i64_unary(f, x), x + 1);
        }
    }

    #[test]
    fn float_constants_arithmetic_and_bitcast() {
        // f(_) = bitcast_i64(f64const(2.5) + f64const(1.5))  ==  (4.0f64).to_bits()
        let f = |b: &mut MlirBackend, _x: ValueId| {
            let a = b.f64const(2.5);
            let c = b.f64const(1.5);
            let sum = b.fadd(a, c);
            b.bitcast(ScalarType::I64, sum)
        };
        let expected = 4.0f64.to_bits() as i64;
        assert_eq!(jit_run_i64_unary(f, 0), expected);
    }

    #[test]
    fn copy_nonoverlapping_between_slots() {
        // f(x) = { src=16B; dst=16B; store x@src+8; memcpy(dst,src,16); load dst+8 } == x
        let f = |b: &mut MlirBackend, x: ValueId| {
            let src = b.alloc_stack_slot(16, 3);
            let dst = b.alloc_stack_slot(16, 3);
            let src_ptr = b.stack_addr(src, 0);
            let dst_ptr = b.stack_addr(dst, 0);
            let zero = b.iconst(ScalarType::I64, 0);
            b.store(zero, src_ptr, 0);
            b.store(x, src_ptr, 8);
            b.copy_nonoverlapping(dst_ptr, src_ptr, 16, 8);
            b.load(ScalarType::I64, dst_ptr, 8)
        };
        for x in [-3i64, 0, 7, 12345] {
            assert_eq!(jit_run_i64_unary(f, x), x);
        }
    }

    #[test]
    fn narrowing_then_signed_widening_cast() {
        // f(x) = sextend_i64(ireduce_i32(x)) — truncates to 32 bits, sign-extends back.
        let f = |b: &mut MlirBackend, x: ValueId| {
            let narrow = b.ireduce(ScalarType::I32, x);
            b.sextend(ScalarType::I64, narrow)
        };
        for x in [0i64, 5, -5, i64::from(i32::MAX), i64::from(i32::MIN)] {
            assert_eq!(jit_run_i64_unary(f, x), i64::from(x as i32));
        }
        // A value with nonzero high bits truncates to its low 32 bits (sign-extended).
        let x = 0x1_2345_6789i64;
        assert_eq!(jit_run_i64_unary(f, x), i64::from(x as i32));
    }

    #[test]
    fn mutable_loop_with_alloca_vars() {
        // sum_to(n) = 0 + 1 + ... + (n-1), via two entry-block alloca variables mutated
        // across a `cf` back-edge (promoted to SSA by mem2reg). Exercises declare/def/use_var,
        // create_block, switch_to_block, jump, and brif.
        let sum_to = |b: &mut MlirBackend, n: ValueId| {
            let acc = b.declare_var(ScalarType::I64);
            let iv = b.declare_var(ScalarType::I64);
            let zero = b.iconst(ScalarType::I64, 0);
            b.def_var(acc, zero);
            b.def_var(iv, zero);

            let header = b.create_block();
            let body = b.create_block();
            let exit = b.create_block();
            b.jump(header, &[]);

            b.switch_to_block(header);
            let i = b.use_var(iv);
            let cond = b.icmp(IntCmp::Slt, i, n);
            b.brif(cond, body, &[], exit, &[]);

            b.switch_to_block(body);
            let a = b.use_var(acc);
            let i2 = b.use_var(iv);
            let a2 = b.iadd(a, i2);
            b.def_var(acc, a2);
            let one = b.iconst(ScalarType::I64, 1);
            let i3 = b.iadd(i2, one);
            b.def_var(iv, i3);
            b.jump(header, &[]);

            b.switch_to_block(exit);
            b.use_var(acc)
        };
        for n in [0i64, 1, 5, 10, 100] {
            let expected = (0..n).sum::<i64>();
            assert_eq!(jit_run_i64_unary(sum_to, n), expected);
        }
    }

    #[test]
    fn if_then_else_with_block_argument_phi() {
        // f(x) = if x < 10 { x * 2 } else { x + 100 }, merged through a real `cf` block
        // argument (the phi) rather than a variable — the analogue of Cranelift's merge
        // block param.
        let f = |b: &mut MlirBackend, x: ValueId| {
            let ten = b.iconst(ScalarType::I64, 10);
            let cond = b.icmp(IntCmp::Slt, x, ten);

            let then_block = b.create_block();
            let else_block = b.create_block();
            let merge = b.create_block();
            let phi = b.append_block_param(merge, ScalarType::I64);
            b.brif(cond, then_block, &[], else_block, &[]);

            b.switch_to_block(then_block);
            let two = b.iconst(ScalarType::I64, 2);
            let doubled = b.imul(x, two);
            b.jump(merge, &[doubled]);

            b.switch_to_block(else_block);
            let hundred = b.iconst(ScalarType::I64, 100);
            let bumped = b.iadd(x, hundred);
            b.jump(merge, &[bumped]);

            b.switch_to_block(merge);
            phi
        };
        for x in [-5i64, 0, 9, 10, 11, 50] {
            let expected = if x < 10 { x * 2 } else { x + 100 };
            assert_eq!(jit_run_i64_unary(f, x), expected);
        }
    }

    // Host externs for the call tests, both shaped like the Phase-3 storage-pointer ABI
    // (values reached through pointers).
    extern "C" fn host_read_i64(p: *const i64) -> i64 {
        // SAFETY: the kernel passes a valid `*const i64` argument through.
        unsafe { *p }
    }
    extern "C" fn host_write_i64(p: *mut i64, value: i64) {
        // SAFETY: the kernel passes a valid `*mut i64` argument through.
        unsafe { *p = value }
    }

    #[test]
    fn calls_registered_extern_returning_i64() {
        // kernel(p: ptr) -> i64 { func.call @host_read_i64(p) }
        let context = make_context();
        let mut backend =
            MlirBackend::new(&context, vec![scalar_to_mlir(&context, ScalarType::Ptr)]);
        let extern_id =
            backend.declare_extern("host_read_i64", &[ScalarType::Ptr], Some(ScalarType::I64));
        let func = backend.declare_extern_func(extern_id);
        let p = backend.param(0);
        let result = backend.call(func, &[p]).expect("i64 result");
        backend.ret(Some(result));
        let module = backend.into_module("kernel", &[ScalarType::I64]);

        let (engine, pointer) = jit_lookup(
            &context,
            module,
            "kernel",
            &[("host_read_i64", host_read_i64 as *const u8)],
        );
        let kernel: extern "C" fn(*const i64) -> i64 = unsafe { std::mem::transmute(pointer) };
        let value: i64 = 0x0BAD_F00D;
        assert_eq!(kernel(&value as *const i64), 0x0BAD_F00D);
        drop(engine);
    }

    #[test]
    fn calls_registered_void_extern_with_two_args() {
        // kernel(out: ptr, x: i64) { func.call @host_write_i64(out, x); return }
        let context = make_context();
        let mut backend = MlirBackend::new(
            &context,
            vec![
                scalar_to_mlir(&context, ScalarType::Ptr),
                scalar_to_mlir(&context, ScalarType::I64),
            ],
        );
        let extern_id =
            backend.declare_extern("host_write_i64", &[ScalarType::Ptr, ScalarType::I64], None);
        let func = backend.declare_extern_func(extern_id);
        let out = backend.param(0);
        let x = backend.param(1);
        assert!(backend.call(func, &[out, x]).is_none());
        backend.ret(None);
        let module = backend.into_module("kernel", &[]);

        let (engine, pointer) = jit_lookup(
            &context,
            module,
            "kernel",
            &[("host_write_i64", host_write_i64 as *const u8)],
        );
        let kernel: extern "C" fn(*mut i64, i64) = unsafe { std::mem::transmute(pointer) };
        let mut slot: i64 = 0;
        kernel(&mut slot as *mut i64, 987_654);
        assert_eq!(slot, 987_654);
        drop(engine);
    }
}
