//! LLVM/MLIR backend (docs/llvm.md, Phase 1+).
//!
//! **Feature-gated behind `--features llvm`.** This module is the second code
//! generator alongside Cranelift. It needs a system LLVM/MLIR 22 install reachable
//! through melior / mlir-sys (set `MLIR_SYS_220_PREFIX`), so it is compiled *only*
//! when the `llvm` feature is on — the default pure-Rust Cranelift build never pulls
//! melior in.
//!
//! ## Where this is
//!
//! Phase -1 (the `mlir-spike/` crate) proved the JIT/ABI/lowering pipeline against
//! *textual* MLIR. This module proves the same pipeline built **programmatically** through
//! melior's op builders — the alpha-API risk the spike deliberately skipped:
//!
//! - [`make_context`] — the known-good context/dialect/translation setup (from the spike).
//! - [`jit_return_i64_const`] — the nullary-constant milestone: a `() -> i64` `func.func`
//!   returning `arith.constant`, verified, lowered, JIT-run via the **native**
//!   `ExecutionEngine::lookup` pointer.
//! - [`MlirBackend`] (in [`backend`]) — the MLIR implementation of the shared
//!   [`crate::staged::Backend`] trait: the full op surface (constants, arithmetic, compares,
//!   `select`, casts, memory with a `StackSlotId` model, pointer ops, variables + `cf`
//!   control flow, and id-based calls/externs). The MLIR analogue of `CraneliftBackend`.
//! - [`jit_run_i64_unary`] — drives `MlirBackend` end to end over a `(i64) -> i64` body
//!   (straight-line *or* with loops/branches).
//! - [`assemble`] — the MLIR half of `func::compile`: builds `__main__` + the helper functions
//!   (each via the *shared neutral* `emit_function_body` under the storage-pointer ABI),
//!   assembles them into one module, JITs, and returns an [`MlirExecutable`] + the `__main__`
//!   pointer. This is what `Compiler::with_backend(JitBackend::Llvm).compile(..)` routes to —
//!   the public API running the neutral AST through MLIR, differential-tested against Cranelift.
//! - [`jit_eval_nullary_i64`] / [`jit_eval_ctx_i64`] / [`jit_eval_ctx_unary_i64`] — earlier
//!   lower-level drivers that run `Staged::codegen` over an `MlirBackend`-backed context; kept
//!   as focused differential tests for expressions, the §5 loop/mem2reg path, and `fun1` bodies.
//!
//! ## The value arena (docs/llvm.md §9 — confirmed)
//!
//! The index in each typed [`crate::staged::ValueId`] maps to an MLIR value arena slot.
//! melior's `Value<'c, 'a>` is `#[repr(transparent)]` over a lifetime-free
//! `mlir_sys::MlirValue`, reachable via `ValueLike::to_raw` and reconstructable via
//! `Value::from_raw`. The arena is therefore an ordinary safe `Vec`, with no
//! self-referential-struct problem: every op appends to a block and immediately stashes its
//! raw result. The neutral `ScalarType` stays on `ValueId` and is checked against MLIR when
//! the value is retrieved; it is never reconstructed from the MLIR type.

mod backend;
pub use backend::MlirBackend;

use melior::dialect::{arith, func, DialectRegistry};
use melior::ir::attribute::{IntegerAttribute, StringAttribute, TypeAttribute};
use melior::ir::block::BlockLike;
use melior::ir::operation::{Operation, OperationLike};
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{Block, Location, Module, Region, RegionLike, Type};
use melior::pass::{self, PassManager};
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use melior::{Context, ExecutionEngine};

use std::collections::HashMap;

use crate::func::{CompileError, Ctx, ExternFnDef, FunDef, TypeInfo};
use crate::staged::{Backend, CompilationContext, Staged, Value, ValueId, Var};
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
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
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
    if std::env::var("RUST_LMS_DEBUG_IR").is_ok() {
        eprintln!("=== MLIR before lowering ===\n{}", module.as_operation());
    }
    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification before lowering"
    );
    let pass_manager = PassManager::new(context);
    // Iterator terminals with early exits (`take_while`/`all`) leave *unreachable* blocks
    // (no predecessors) whose `cf.br` the dialect-conversion framework skips — they survive
    // to LLVM translation and fail with "missing LLVMTranslationDialectInterface for cf.br".
    // Canonicalization does region simplification (prunes unreachable blocks) so only live
    // branches remain to lower. (Cranelift prunes these itself; MLIR needs the pass.)
    pass_manager.add_pass(pass::transform::create_canonicalizer());
    // `cf` (`cf.br`/`cf.cond_br`, emitted by loops and the surviving branches) is not reliably
    // picked up by the interface-based generic `convert-to-llvm` pass under these melior
    // bindings, so lower `cf` explicitly first — every branch becomes `llvm.br`/`llvm.cond_br`.
    pass_manager.add_pass(pass::conversion::create_control_flow_to_llvm());
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    // Resolve the `builtin.unrealized_conversion_cast`s that `func_addr` inserts (function
    // value → `llvm.ptr`) once `func.constant` has become `llvm.mlir.addressof` (docs/llvm.md §7).
    pass_manager.add_pass(pass::conversion::create_reconcile_unrealized_casts());
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
    let mut backend = MlirBackend::new(&context, vec![ScalarType::I64]);
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

/// Compile and run a **nullary** `Staged` expression (`Out = i64`) through the MLIR
/// backend, by driving the *neutral AST* `Staged::codegen` over a [`CompilationContext`]
/// backed by [`MlirBackend`] — the same code path the Cranelift backend runs, just with a
/// different `dyn Backend`. This is the first end-to-end proof that a real AST graph lowers
/// through MLIR; it powers the differential tests against Cranelift.
///
/// Nullary (no parameters) sidesteps the storage-pointer parameter ABI, which still lives in
/// the Cranelift-specific `compile()` driver (its abstraction is the next step).
pub fn jit_eval_nullary_i64(expr: impl Staged<Out = i64>) -> i64 {
    run_kernel_over_mlir(|ctx| expr.codegen(ctx))
}

/// Compile and run a **nullary imperative `Ctx` body** (`Out = i64`) through the MLIR
/// backend — the [`Ctx`] form used by `fun0`, exercising `ctx.var`/`store`/`while_loop`/
/// `if_then` (i.e. the §5 variable + `cf`-loop machinery) *through the real AST* rather than
/// hand-built `MlirBackend` calls. Var ids start at 0, matching a fresh `Compiler`.
pub fn jit_eval_ctx_i64<R, F>(build: F) -> i64
where
    F: FnOnce(&mut Ctx) -> R,
    R: Staged<Out = i64> + 'static,
{
    let mut builder = Ctx::new(0);
    let ret = build(&mut builder);
    let body = builder.into_body(ret);
    run_kernel_over_mlir(|ctx| body(ctx))
}

/// Compile and run a **unary** `fun1`-style imperative body (`(i64) -> i64`) through the MLIR
/// backend, and call it with `arg`. Mirrors `make_fun1`'s var-id assignment (param = id 0,
/// body locals start at id 1) so the body's `Var`s resolve identically to the Cranelift path.
///
/// A direct scalar ABI (param passed by value, stored into a variable) — this sidesteps the
/// full storage-pointer parameter ABI in `compile()` while still exercising the *parameterized*
/// AST end to end on MLIR. Powers the parameterized differential tests.
pub fn jit_eval_ctx_unary_i64<R, F>(build: F, arg: i64) -> i64
where
    F: FnOnce(&mut Ctx, Var<i64>) -> R,
    R: Staged<Out = i64> + 'static,
{
    // Mirror `make_fun1`: the parameter takes var id 0; body locals start at id 1.
    let param = Var::<i64>::new(0);
    let param_id = param.id;
    let mut builder = Ctx::new(1);
    let ret = build(&mut builder, param);
    let body = builder.into_body(ret);

    let context = make_context();
    let mut backend = MlirBackend::new(&context, vec![ScalarType::I64]);

    // Store the incoming argument into the parameter's variable slot, then map its var id.
    let incoming = backend.param(0);
    let param_var = backend.declare_var(ScalarType::I64);
    backend.def_var(param_var, incoming);

    let mut variables = HashMap::new();
    variables.insert(param_id, crate::staged::VarValue::Scalar(param_var));
    let result = {
        let mut ctx = CompilationContext {
            backend: &mut backend,
            variables: &mut variables,
            unit_value: None,
            block_params: HashMap::new(),
            loop_exit_stack: Vec::new(),
        };
        body(&mut ctx)
    };
    backend.ret(Some(result.leaf()));
    let module = backend.into_module("kernel", &[ScalarType::I64]);

    let (engine, pointer) = jit_lookup(&context, module, "kernel", &[]);
    // SAFETY: emitted with the `(i64) -> i64` signature; `engine` owns the executable memory.
    let kernel: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(pointer) };
    let output = kernel(arg);
    drop(engine);
    output
}

/// Shared tail: build a nullary `() -> i64` kernel whose body is `emit_body` (run against a
/// [`CompilationContext`] backed by [`MlirBackend`]), JIT it, and run it.
fn run_kernel_over_mlir(emit_body: impl FnOnce(&mut CompilationContext) -> Value) -> i64 {
    let context = make_context();
    let mut backend = MlirBackend::new(&context, Vec::new());

    let mut variables = HashMap::new();
    let result = {
        let mut ctx = CompilationContext {
            backend: &mut backend,
            variables: &mut variables,
            unit_value: None,
            block_params: HashMap::new(),
            loop_exit_stack: Vec::new(),
        };
        emit_body(&mut ctx)
    };
    backend.ret(Some(result.leaf()));
    let module = backend.into_module("kernel", &[ScalarType::I64]);

    let (engine, pointer) = jit_lookup(&context, module, "kernel", &[]);
    // SAFETY: emitted with the `() -> i64` signature; `engine` owns the executable memory
    // and outlives the call below.
    let kernel: extern "C" fn() -> i64 = unsafe { std::mem::transmute(pointer) };
    let output = kernel();
    drop(engine);
    output
}

/// A JIT-compiled MLIR module and the resources that keep its native code alive: the
/// `ExecutionEngine` (owns the executable memory) and the `Context` it was built in.
/// Dropped in declaration order — engine first, then context — when the owning `Compiled`
/// is dropped. **Thread-affine:** melior's `ExecutionEngine` is `!Send + !Sync` (docs/llvm.md
/// §9), so an LLVM-compiled `Compiled` is too.
pub(crate) struct MlirExecutable {
    _engine: ExecutionEngine,
    _context: Context,
}

/// Assemble a whole compilation — the `functions` (helpers) plus `__main__` — into one MLIR
/// module and JIT it, returning the executable and the native `__main__` pointer.
///
/// This is the MLIR counterpart of the Cranelift shell in `func::compile`: each function is
/// built with the uniform storage-pointer ABI (`N+1` `llvm.ptr` params, `void` return) via
/// the *shared, neutral* [`crate::func::emit_function_body`], so the AST lowers identically to
/// the Cranelift path. Internal/extern references resolve through id-aligned symbol tables
/// pre-registered on each backend; extern host addresses are bound by symbol at JIT time.
pub(crate) fn assemble(
    functions: Vec<Option<FunDef>>,
    externs: &[ExternFnDef],
    main_return_info: &TypeInfo,
    main_body: impl FnOnce(&mut CompilationContext) -> Value,
) -> Result<(MlirExecutable, *const u8), CompileError> {
    let context = make_context();

    // Id-aligned metadata so a body's `declare_func(id)`/`declare_extern_func(id)` resolves to
    // the right symbol (the storage-pointer ABI: every callee is `(ptr, …) -> void`).
    let internal_meta: Vec<Option<(String, usize)>> = functions
        .iter()
        .map(|f| f.as_ref().map(|d| (d.name.clone(), d.param_infos.len())))
        .collect();
    let extern_meta: Vec<(String, usize)> = externs
        .iter()
        .map(|e| (e.name.clone(), e.num_params))
        .collect();

    let mut ops = Vec::new();
    for def in functions.into_iter().flatten() {
        let (op, _) = build_function(
            &context,
            &def.name,
            &def.param_infos,
            &def.param_var_ids,
            def.body,
            &def.return_info,
            &internal_meta,
            &extern_meta,
        );
        ops.push(op);
    }
    // `__main__` is a zero-argument function under the same ABI (one output pointer). Every
    // backend registers all externs, so its declaration set is the full one the module needs.
    let (main_op, extern_decls) = build_function(
        &context,
        "__main__",
        &[],
        &[],
        main_body,
        main_return_info,
        &internal_meta,
        &extern_meta,
    );
    ops.push(main_op);

    let module = backend::assemble_module(&context, ops, &extern_decls);
    let symbols: Vec<(&str, *const u8)> = externs
        .iter()
        .map(|e| (e.name.as_str(), e.fn_ptr))
        .collect();
    let (engine, pointer) = jit_lookup(&context, module, "__main__", &symbols);
    Ok((
        MlirExecutable {
            _engine: engine,
            _context: context,
        },
        pointer as *const u8,
    ))
}

/// Build one function body (`name`) into a `func.func` op under the storage-pointer ABI:
/// `N+1` `llvm.ptr` params, `void` return, body emitted by the shared neutral
/// [`crate::func::emit_function_body`]. `internal_meta`/`extern_meta` are pre-registered
/// id-aligned so the body's calls resolve to symbols.
#[allow(clippy::too_many_arguments)]
fn build_function<'c>(
    context: &'c Context,
    name: &str,
    param_infos: &[TypeInfo],
    param_var_ids: &[usize],
    body: impl FnOnce(&mut CompilationContext) -> Value,
    return_info: &TypeInfo,
    internal_meta: &[Option<(String, usize)>],
    extern_meta: &[(String, usize)],
) -> (Operation<'c>, Vec<backend::FuncDecl>) {
    let num_params = param_infos.len();
    let mut mlir = MlirBackend::new(context, vec![ScalarType::Ptr; num_params + 1]);

    // Register internal functions id-aligned (placeholder for undefined slots) and externs.
    for meta in internal_meta {
        match meta {
            Some((callee_name, callee_params)) => {
                let sig = vec![ScalarType::Ptr; callee_params + 1];
                mlir.declare_internal_func(callee_name, &sig, None);
            }
            None => {
                mlir.declare_internal_func("__undefined__", &[], None);
            }
        }
    }
    for (extern_name, extern_params) in extern_meta {
        let sig = vec![ScalarType::Ptr; extern_params + 1];
        mlir.declare_extern(extern_name, &sig, None);
    }

    let params: Vec<ValueId> = (0..=num_params).map(|i| mlir.param(i)).collect();
    {
        let mut variables = HashMap::new();
        let mut ctx = CompilationContext {
            backend: &mut mlir,
            variables: &mut variables,
            unit_value: None,
            block_params: HashMap::new(),
            loop_exit_stack: Vec::new(),
        };
        crate::func::emit_function_body(
            &mut ctx,
            &params,
            param_infos,
            param_var_ids,
            body,
            return_info,
        );
    }
    mlir.ret(None);
    mlir.into_function_op(name, &[])
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
        let mut backend = MlirBackend::new(&context, vec![ScalarType::Ptr]);
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
        let mut backend = MlirBackend::new(&context, vec![ScalarType::Ptr, ScalarType::I64]);
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

    #[test]
    fn differential_nullary_matches_cranelift() {
        // The payoff: run the *same neutral `Staged` graph* through Cranelift's public
        // `compile()` and through the MLIR backend, and assert identical results — proving
        // the AST lowers the same on both. Nullary (no params) for now; the parameter ABI
        // still lives in the Cranelift-specific `compile()` driver.
        use crate::control::if_then_else;
        use crate::func::Compiler;
        use crate::num::{
            add, bitand, bitor, bitxor, div, eq, gt, int_cast, lt, max, min, mul, rem, select, shl,
            shr, sub,
        };
        use crate::staged::Const;

        fn both<S: Staged<Out = i64> + 'static>(make: impl Fn() -> S) {
            let cranelift = Compiler::new()
                .compile(make())
                .expect("cranelift compile")
                .run();
            let mlir = jit_eval_nullary_i64(make());
            assert_eq!(cranelift, mlir, "Cranelift/MLIR backend divergence");
        }

        both(|| add(mul(6i64, 7i64), 1i64)); // 43
        both(|| sub(mul(9i64, 9i64), 1i64)); // 80
        both(|| select(lt(3i64, 5i64), 100i64, 200i64)); // 100 (branchless)
        both(|| select(lt(5i64, 3i64), 100i64, 200i64)); // 200
                                                         // Control flow: `cf` blocks + block-argument phi via the AST's `IfThenElse`.
        both(|| if_then_else(lt(2i64, 9i64), Const::<i64>::new(1), Const::<i64>::new(2))); // 1
        both(|| if_then_else(lt(9i64, 2i64), Const::<i64>::new(1), Const::<i64>::new(2))); // 2

        // Signed division/remainder, incl. negative operands (truncated toward zero).
        both(|| div(20i64, 6i64)); // 3
        both(|| div(-20i64, 6i64)); // -3
        both(|| rem(20i64, 6i64)); // 2
        both(|| rem(-20i64, 6i64)); // -2

        // Bitwise + shifts (arithmetic right shift on a negative operand).
        both(|| bitand(0b1100i64, 0b1010i64)); // 8
        both(|| bitor(0b1100i64, 0b1010i64)); // 14
        both(|| bitxor(0b1100i64, 0b1010i64)); // 6
        both(|| shl(1i64, 40i64)); // 1 << 40
        both(|| shr(256i64, 2i64)); // 64
        both(|| shr(-256i64, 2i64)); // -64 (arithmetic)

        // Branchless min/max, and comparisons surfaced through `select`.
        both(|| min(3i64, 8i64)); // 3
        both(|| max(3i64, 8i64)); // 8
        both(|| select(eq(5i64, 5i64), 1i64, 0i64)); // 1
        both(|| select(gt(5i64, 3i64), 1i64, 0i64)); // 1
                                                     // Unsigned comparison: u64::MAX > 1 is true unsigned; it would be false if either
                                                     // backend wrongly used a *signed* predicate (MAX as i64 == -1), so this discriminates.
        both(|| select(gt(u64::MAX, 1u64), 1i64, 0i64)); // 1

        // Integer casts: truncate i64->i32 then sign-extend back — positive and negative.
        both(|| int_cast::<i64, i32, _>(int_cast::<i32, i64, _>(0x1_0000_0007i64))); // 7
        both(|| int_cast::<i64, i32, _>(int_cast::<i32, i64, _>(0x1_FFFF_FFFFi64)));
        // -1
    }

    #[test]
    fn differential_imperative_ctx_matches_cranelift() {
        // The §5 payoff: imperative `Ctx` bodies — mutable locals + `while` loops — lowered
        // *through the real AST* to entry-block alloca / `cf` loops / mem2reg on MLIR, and
        // asserted identical to Cranelift. `fn` bodies (Copy) run on both backends.
        use crate::func::{call0, Compiler};
        use crate::num::{add, lt, mul};
        use crate::staged::Var;

        fn both<R: Staged<Out = i64> + 'static>(build: fn(&mut Ctx) -> R) {
            let cranelift = {
                let mut c = Compiler::new();
                let f = c.fun0("k", build);
                c.compile(call0(f)).expect("cranelift compile").run()
            };
            let mlir = jit_eval_ctx_i64(build);
            assert_eq!(
                cranelift, mlir,
                "Cranelift/MLIR divergence (imperative Ctx)"
            );
        }

        // sum 0..10 == 45, via a mutable accumulator + induction var over a `while` loop.
        fn sum_to_ten(ctx: &mut Ctx) -> Var<i64> {
            let acc = ctx.var(0i64);
            let i = ctx.var(0i64);
            ctx.while_loop(lt(i, 10i64), move |ctx| {
                ctx.store(acc, add(acc, i));
                ctx.store(i, add(i, 1i64));
            });
            acc
        }
        both(sum_to_ten);

        // 6! == 720, a multiply-accumulate loop.
        fn factorial_six(ctx: &mut Ctx) -> Var<i64> {
            let acc = ctx.var(1i64);
            let i = ctx.var(1i64);
            ctx.while_loop(lt(i, 7i64), move |ctx| {
                ctx.store(acc, mul(acc, i));
                ctx.store(i, add(i, 1i64));
            });
            acc
        }
        both(factorial_six);
    }

    #[test]
    fn differential_unary_param_matches_cranelift() {
        // Parameterized functions: the same `fun1` body compiled through Cranelift
        // (`compile(f).as_fn().call(x)`) and MLIR (`jit_eval_ctx_unary_i64`), asserted equal
        // across arguments. Covers a pure expression, a polynomial, and a `while` loop whose
        // bound is the *parameter* (data-dependent iteration count).
        use crate::func::Compiler;
        use crate::num::{add, lt, mul};
        use crate::staged::Var;

        fn both<R: Staged<Out = i64> + 'static>(build: fn(&mut Ctx, Var<i64>) -> R, args: &[i64]) {
            for &x in args {
                let cranelift = {
                    let mut c = Compiler::new();
                    let f = c.fun1("f", build);
                    c.compile(f).expect("cranelift compile").as_fn().call(x)
                };
                let mlir = jit_eval_ctx_unary_i64(build, x);
                assert_eq!(cranelift, mlir, "Cranelift/MLIR divergence at x={x}");
            }
        }

        fn square(_ctx: &mut Ctx, x: Var<i64>) -> impl Staged<Out = i64> {
            mul(x, x)
        }
        both(square, &[-3, 0, 5, 1000]);

        fn poly(_ctx: &mut Ctx, x: Var<i64>) -> impl Staged<Out = i64> {
            add(mul(x, 3i64), 7i64)
        }
        both(poly, &[-4, 0, 5, 100]);

        // sum 0..x — the loop trip count is the parameter (data-dependent control flow).
        fn sum_to_x(ctx: &mut Ctx, x: Var<i64>) -> Var<i64> {
            let acc = ctx.var(0i64);
            let i = ctx.var(0i64);
            ctx.while_loop(lt(i, x), move |ctx| {
                ctx.store(acc, add(acc, i));
                ctx.store(i, add(i, 1i64));
            });
            acc
        }
        both(sum_to_x, &[0, 1, 5, 10, 50]);
    }

    #[test]
    fn multi_function_module_with_internal_call() {
        // Two functions in one MLIR module: `double(x) = 2x` and `quad(x) = double(double(x))`,
        // the latter resolving its callee via `declare_func`/`call`. This is the multi-function
        // capability `compile()` needs (helper functions alongside `__main__`).
        let context = make_context();

        let double_op = {
            let mut b = MlirBackend::new(&context, vec![ScalarType::I64]);
            let x = b.param(0);
            let two = b.iconst(ScalarType::I64, 2);
            let doubled = b.imul(x, two);
            b.ret(Some(doubled));
            b.into_function_op("double", &[ScalarType::I64]).0
        };

        let quad_op = {
            let mut b = MlirBackend::new(&context, vec![ScalarType::I64]);
            let double_id =
                b.declare_internal_func("double", &[ScalarType::I64], Some(ScalarType::I64));
            let callee = b.declare_func(double_id);
            let x = b.param(0);
            let once = b.call(callee, &[x]).expect("i64 result");
            let twice = b.call(callee, &[once]).expect("i64 result");
            b.ret(Some(twice));
            b.into_function_op("quad", &[ScalarType::I64]).0
        };

        let module = super::backend::assemble_module(&context, vec![double_op, quad_op], &[]);
        let (engine, pointer) = jit_lookup(&context, module, "quad", &[]);
        let quad: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(pointer) };
        assert_eq!(quad(5), 20);
        assert_eq!(quad(-3), -12);
        drop(engine);
    }

    #[test]
    fn compile_with_llvm_matches_cranelift() {
        // The capstone: the *real* public API — `Compiler::with_backend(Llvm).compile(expr)
        // .run()` — routed through MLIR, differential-tested against Cranelift. `setup`
        // defines any helper functions on the compiler and returns the top-level expression;
        // it runs once per backend.
        use crate::func::{call0, call1, Compiler, JitBackend};
        use crate::num::{add, lt, mul};
        use crate::staged::{Const, Var};

        fn both<S: Staged<Out = i64> + 'static>(setup: impl Fn(&mut Compiler) -> S) {
            let cranelift = {
                let mut c = Compiler::new();
                let expr = setup(&mut c);
                c.compile(expr).expect("cranelift compile").run()
            };
            let llvm = {
                let mut c = Compiler::new().with_backend(JitBackend::Llvm);
                let expr = setup(&mut c);
                c.compile(expr).expect("llvm compile").run()
            };
            assert_eq!(
                cranelift, llvm,
                "Cranelift/MLIR divergence via compile().run()"
            );
        }

        // A plain top-level expression (`__main__` only).
        both(|_c| add(mul(6i64, 7i64), 1i64)); // 43

        // A nullary helper called from `__main__` (internal call).
        both(|c| {
            let f = c.fun0("answer", |_ctx| Const::<i64>::new(42));
            call0(f)
        });

        // A unary helper (storage-pointer arg + output).
        both(|c| {
            let sq = c.fun1("sq", |_ctx, x: Var<i64>| mul(x, x));
            call1(sq, 7i64)
        }); // 49

        // A helper containing a data-dependent `while` loop.
        both(|c| {
            let sum = c.fun1("sum_to", |ctx, n: Var<i64>| {
                let acc = ctx.var(0i64);
                let i = ctx.var(0i64);
                ctx.while_loop(lt(i, n), move |ctx| {
                    ctx.store(acc, add(acc, i));
                    ctx.store(i, add(i, 1i64));
                });
                acc
            });
            call1(sum, 10i64)
        }); // 45
    }

    #[test]
    fn compile_with_llvm_as_fn_entry() {
        // `as_fn().call(x)` goes through the `func_addr`/`__main__` trampoline: `__main__`
        // returns the helper's address, which `as_fn` then calls with the storage-pointer ABI.
        // This checks that `func.constant`/addressof yields a callable native address on MLIR.
        use crate::func::{Compiler, JitBackend};
        use crate::num::{add, mul};
        use crate::staged::Var;

        let mut c = Compiler::new().with_backend(JitBackend::Llvm);
        let sq = c.fun1("sq", |_ctx, x: Var<i64>| mul(x, x));
        let compiled = c.compile(sq).expect("llvm compile");
        let f = compiled.as_fn();
        assert_eq!(f.call(7), 49);
        assert_eq!(f.call(-4), 16);

        let mut c = Compiler::new().with_backend(JitBackend::Llvm);
        let axpy = c.fun2("axpy", |_ctx, a: Var<i64>, b: Var<i64>| add(mul(a, b), b));
        let compiled = c.compile(axpy).expect("llvm compile");
        let g = compiled.as_fn();
        assert_eq!(g.call(3, 5), 20); // 3*5 + 5
    }

    #[test]
    fn differential_fun1_via_as_fn() {
        // The high-level API on both backends: define a `fun1`, `compile().as_fn().call(x)`,
        // and assert Cranelift == MLIR across arguments. `define` (a `fn`, Copy) runs on each.
        use crate::control::if_then_else;
        use crate::func::{call1, Compiler, FunRef1, JitBackend};
        use crate::num::{lt, mul, sub};
        use crate::staged::{Const, Var};

        fn both1(define: fn(&mut Compiler) -> FunRef1<i64, i64>, args: &[i64]) {
            for &x in args {
                let cranelift = {
                    let mut c = Compiler::new();
                    let f = define(&mut c);
                    c.compile(f).expect("cranelift").as_fn().call(x)
                };
                let llvm = {
                    let mut c = Compiler::new().with_backend(JitBackend::Llvm);
                    let f = define(&mut c);
                    c.compile(f).expect("llvm").as_fn().call(x)
                };
                assert_eq!(cranelift, llvm, "divergence at x={x}");
            }
        }

        both1(
            |c| c.fun1("sq", |_ctx, x: Var<i64>| mul(x, x)),
            &[-4, 0, 5, 1000],
        );

        // Recursion: factorial via `fun1_rec` (self-call through `call1`) + `if_then_else`
        // — exercises internal calls and control flow together on both backends.
        both1(
            |c| {
                c.fun1_rec("fact", |f, _ctx, n: Var<i64>| {
                    if_then_else(
                        lt(n, 2),
                        Const::<i64>::new(1),
                        mul(n, call1(f, sub(n, 1i64))),
                    )
                })
            },
            &[0, 1, 2, 5, 10],
        );
    }

    #[test]
    fn differential_slice_sum() {
        // A slice *parameter* (fat pointer) summed in a loop — the sql-gen columnar shape.
        // Exercises the storage-pointer ABI unpacking a `(ptr, len)` and `getelementptr`-based
        // element access on MLIR. Same kernel, both backends, identical result.
        use crate::func::{Compiler, JitBackend};
        use crate::num::{add, lt};
        use crate::refer::SRef;
        use crate::slice::{Slice, SliceRefOps};
        use crate::staged::Var;

        let data = [10i64, 20, 30, 40, 50, -5, 7];

        let cranelift = {
            let mut c = Compiler::new();
            let sum = c.fun1("sum", |ctx, arr: Var<SRef<Slice<i64>>>| {
                let i = ctx.var(0u64);
                let total = ctx.var(0i64);
                ctx.while_loop(lt(i, arr.count()), move |ctx| {
                    ctx.store(total, add(total, unsafe { arr.get_unchecked(i) }));
                    ctx.store(i, add(i, 1u64));
                });
                total
            });
            c.compile(sum).expect("cranelift").as_fn().call(&data[..])
        };
        let llvm = {
            let mut c = Compiler::new().with_backend(JitBackend::Llvm);
            let sum = c.fun1("sum", |ctx, arr: Var<SRef<Slice<i64>>>| {
                let i = ctx.var(0u64);
                let total = ctx.var(0i64);
                ctx.while_loop(lt(i, arr.count()), move |ctx| {
                    ctx.store(total, add(total, unsafe { arr.get_unchecked(i) }));
                    ctx.store(i, add(i, 1u64));
                });
                total
            });
            c.compile(sum).expect("llvm").as_fn().call(&data[..])
        };
        assert_eq!(cranelift, 152);
        assert_eq!(llvm, cranelift, "slice sum: Cranelift/MLIR divergence");
    }
}
