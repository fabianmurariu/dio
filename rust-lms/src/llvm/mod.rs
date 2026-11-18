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
//! - [`MlirBackend`] (in [`backend`]) — the MLIR implementation of the shared
//!   [`crate::staged::Backend`] trait: the full op surface (constants, arithmetic, compares,
//!   `select`, casts, memory with a `StackSlotId` model, pointer ops, variables + `cf`
//!   control flow, and id-based calls/externs). The MLIR analogue of `CraneliftBackend`.
//! - [`assemble`] — the MLIR half of `func::compile`: builds `__main__` + the helper functions
//!   (each via the *shared neutral* `emit_function_body` under the storage-pointer ABI),
//!   assembles them into one module, JITs, and returns an [`MlirExecutable`] + the `__main__`
//!   pointer. This is what `Compiler::with_backend(JitBackend::Llvm).compile(..)` routes to —
//!   the public API running the neutral AST through MLIR, differential-tested against Cranelift.
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

use melior::{
    Context, ExecutionEngine,
    dialect::DialectRegistry,
    ir::{
        Module,
        operation::{Operation, OperationLike},
    },
    pass::{self, PassManager},
    utility::{register_all_dialects, register_all_llvm_translations},
};

use std::collections::HashMap;

use crate::{
    compiler::CompileError,
    func::{ExternFnDef, FunDef, TypeInfo},
    staged::{CompilationContext, LoadKind, Value, ValueId},
    types::ScalarType,
};

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
    if std::env::var_os(crate::compiler::DEBUG_IR).is_some() {
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
    // The LLVM dialect, before LLVM's own passes: `ExecutionEngine` optimises it
    // internally and does not hand back the optimised IR.
    if std::env::var_os(crate::compiler::DEBUG_IR_OPT).is_some() {
        eprintln!("=== MLIR after lowering ===\n{}", module.as_operation());
    }
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

/// A JIT-compiled MLIR module whose `ExecutionEngine` owns its native code.
///
/// The MLIR C API permits the source module to be destroyed once engine creation returns;
/// the same applies to its source `Context`, so neither is retained here. `func` wraps this
/// resource in its private post-finalization `FrozenExecutable`, which supplies the audited
/// shared-ownership/threading boundary without exposing any engine methods.
pub(crate) struct MlirExecutable {
    _engine: ExecutionEngine,
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
    Ok((MlirExecutable { _engine: engine }, pointer as *const u8))
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
            labels: HashMap::new(),
            load_kind: LoadKind::Plain,
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
