//! First-class function support for staged computations.
//!
//! This module provides:
//! - `Compiler`: The central coordinator that owns function and variable definitions
//! - `FunRefN<T0, ..., OUT>`: Type-safe handles to function definitions (N = 0..8)
//! - `CallN`: Function call expressions
//! - `Compiled<T>`: The result of compilation, owns the JIT module
//!
//! # Multi-Parameter Functions
//!
//! Functions with 0-8 parameters are supported via `fun0`, `fun1`, ..., `fun8`.
//! Each returns a type-safe `FunRefN` that encodes the parameter and return types.
//!
//! # Aggregate Values
//!
//! Generated functions use a private, platform-independent storage-pointer ABI.
//! Rust-facing wrappers preserve value semantics without exposing platform
//! aggregate classification to Cranelift.

use crate::cranelift::CraneliftBackend;
use crate::staged::{CompilationContext, SigSpec, Staged, Value, ValueId, Var, VarValue, assign};
use crate::types::{RuntimeParam, RuntimeResult, ScalarType, StagedType};
use cranelift_codegen::ir::{AbiParam, InstBuilder, types};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module, default_libcall_names};
use std::collections::HashMap;
use std::marker::PhantomData;
use std::mem::MaybeUninit;

pub use crate::func_impl::*;

// =============================================================================
// Internal: FunDef - Stored function definition
// =============================================================================

/// Internal storage for a function definition (type-erased body)
pub(crate) struct FunDef {
    pub name: String,
    /// The body expression, type-erased but we know its signature
    pub body: Box<dyn FnOnce(&mut CompilationContext) -> Value>,
    /// Type info for each parameter (supports 0..N parameters)
    pub param_infos: Vec<TypeInfo>,
    /// Return type info
    pub return_info: TypeInfo,
    /// Variable IDs for each parameter (one per logical parameter)
    pub param_var_ids: Vec<usize>,
}

type CodegenAction = Box<dyn FnOnce(&mut CompilationContext) + 'static>;

// =============================================================================
// Ctx: Imperative context for building staged function bodies
// =============================================================================

/// Imperative context for building staged function bodies.
///
/// Passed to closures in `fun1`, `fun2`, etc. Call methods to emit code in
/// declaration order — no expression tree, no `Clone` constraints, no tuple
/// sequencing boilerplate.
///
/// # Example
/// ```ignore
/// compiler.fun1("sum", |ctx, arr: Var<SRef<Slice<f64>>>| {
///     let acc = ctx.var(0.0f64);
///     arr.staged_iter().for_each(ctx, |ctx, elem| {
///         ctx.assign(acc, add(acc, elem));
///     });
///     acc
/// });
/// ```
pub struct Ctx {
    pub(crate) next_var_id: usize,
    actions: Vec<CodegenAction>,
}

impl Ctx {
    pub(crate) fn new(start: usize) -> Self {
        Ctx {
            next_var_id: start,
            actions: Vec::new(),
        }
    }

    pub(crate) fn final_id(&self) -> usize {
        self.next_var_id
    }

    /// Consume this context, producing a `FunDef.body` closure that replays
    /// all accumulated actions then evaluates and returns `ret`.
    pub(crate) fn into_body<Ret>(
        self,
        ret: Ret,
    ) -> Box<dyn FnOnce(&mut CompilationContext) -> Value>
    where
        Ret: Staged + 'static,
    {
        let actions = self.actions;
        Box::new(move |ctx| {
            for action in actions {
                action(ctx);
            }
            ret.codegen(ctx)
        })
    }

    fn alloc<T: StagedType + 'static>(&mut self) -> Var<T> {
        let id = self.next_var_id;
        self.next_var_id += 1;
        Var::new(id)
    }

    /// Allocate a variable ID without registering an initialization action.
    ///
    /// # Safety
    /// Caller must ensure the variable is assigned before it is used in codegen.
    pub(crate) unsafe fn var_unchecked<T: StagedType>(&mut self) -> Var<T> {
        let id = self.next_var_id;
        self.next_var_id += 1;
        Var::new(id)
    }

    /// Declare a variable with an initial value.
    ///
    /// Returns a `LetVar<T, E::Staged>` for backward compatibility with old
    /// tuple-sequencing code. The initialization is automatically registered as
    /// an action in the `Ctx`; including the returned `LetVar` in a tuple
    /// sequence double-inits (harmlessly). Prefer `Ctx::var()` for new code.
    pub fn let_var<T, E>(&mut self, init: E) -> crate::staged::LetVar<T, E::Staged>
    where
        T: StagedType + 'static,
        E: crate::staged::IntoStaged<T>,
        E::Staged: Clone + 'static,
    {
        let init_staged = init.into_staged();
        let v = self.alloc::<T>();
        let id = v.id;
        let init_for_action = init_staged.clone();
        self.actions.push(Box::new(move |ctx| {
            let value = init_for_action.codegen(ctx);
            ctx.assign_var::<T>(id, value, false);
        }));
        crate::staged::LetVar::new(v, init_staged)
    }

    /// Declare a new variable initialized to `init` at this point in the body.
    ///
    /// Returns a `Var<T>` that can be used in expressions and passed to `assign`.
    pub fn var<T, E>(&mut self, init: E) -> Var<T>
    where
        T: StagedType + 'static,
        E: crate::staged::IntoStaged<T>,
        E::Staged: 'static,
    {
        let v = self.alloc::<T>();
        let init_staged = init.into_staged();
        let id = v.id;
        self.actions.push(Box::new(move |ctx| {
            let value = init_staged.codegen(ctx);
            ctx.assign_var::<T>(id, value, false);
        }));
        v
    }

    /// Evaluate a complex staged expression once, binding the result to a new
    /// variable. Avoids recomputing the expression if used in multiple places.
    pub fn bind<T, E>(&mut self, expr: E) -> Var<T>
    where
        T: StagedType + 'static,
        E: Staged<Out = T> + 'static,
    {
        let v = self.alloc::<T>();
        let id = v.id;
        self.actions.push(Box::new(move |ctx| {
            let value = expr.codegen(ctx);
            ctx.assign_var::<T>(id, value, true);
        }));
        v
    }

    /// Emit an assignment: `var = expr`.
    ///
    /// Accepts any value that implements `IntoStaged<T>` — primitives like
    /// `42i64` work directly, as do staged expressions.
    pub fn store<T, E>(&mut self, var: Var<T>, expr: E)
    where
        T: StagedType + 'static,
        E: crate::staged::IntoStaged<T>,
        E::Staged: 'static,
    {
        let staged_expr = expr.into_staged();
        self.actions.push(Box::new(move |ctx| {
            assign(var, staged_expr).codegen(ctx);
        }));
    }

    /// Emit any unit-typed staged expression (e.g. a store, an extern call).
    pub fn emit<S: Staged<Out = ()> + 'static>(&mut self, stmt: S) {
        self.actions.push(Box::new(move |ctx| {
            stmt.codegen(ctx);
        }));
    }

    /// Emit a while loop: `while cond { body }`.
    ///
    /// `body` is called once at staging time; the closure emits the per-iteration
    /// side effects into a child `Ctx`. `cond` accepts `IntoStaged<bool>`
    /// so `false` and `true` work directly.
    pub fn while_loop<C, F>(&mut self, cond: C, body: F)
    where
        C: crate::staged::IntoStaged<bool>,
        C::Staged: 'static,
        F: FnOnce(&mut Ctx),
    {
        let cond = cond.into_staged();
        let mut child = Ctx::new(self.next_var_id);
        body(&mut child);
        self.next_var_id = child.next_var_id;
        let body_actions = child.actions;

        self.actions.push(Box::new(move |ctx| {
            let loop_header = ctx.create_block();
            let loop_body = ctx.create_block();
            let loop_exit = ctx.create_block();

            ctx.jump(loop_header, &[]);

            ctx.switch_to_block(loop_header);
            let cond_val = cond.codegen(ctx);
            ctx.brif(cond_val.leaf(), loop_body, &[], loop_exit, &[]);

            ctx.switch_to_block(loop_body);
            ctx.seal_block(loop_body);
            // Expose this loop's exit block so `break_loop` inside the body can
            // jump to it; pop once the body is fully emitted.
            ctx.loop_exit_stack.push(loop_exit);
            for action in body_actions {
                action(ctx);
            }
            ctx.loop_exit_stack.pop();
            ctx.jump(loop_header, &[]);
            ctx.seal_block(loop_header);

            ctx.switch_to_block(loop_exit);
            ctx.seal_block(loop_exit);
        }));
    }

    /// Break out of the innermost enclosing loop.
    ///
    /// Emits a jump to the current loop's exit block. Typically used inside an
    /// `if_then` to exit early. Panics at codegen time if called outside a loop.
    /// Drive a push loop over an opaque external iterator via `next`/`drop`.
    ///
    /// Emits the storage-pointer loop (see `iter::opaque`):
    /// ```text
    /// header: next(&it, &option)        ; Rust thunk writes COption<Item>
    ///         brif tag, body, exit      ; tag: 1 = Some, 0 = None
    /// body:   elem = val ; <consumer> ; jump header
    /// exit:   drop(it)
    /// ```
    /// `drop` sits at the top of `exit`, which is reached both by the `None`
    /// branch and by any `break_loop` from the body, so the handle is always
    /// freed. The element is loaded from the exact `COption<Item>` payload.
    ///
    /// `Item` must be an integer no wider than 64 bits (the `COption` FFI ABI
    /// returns the payload in an integer register). Float/compound items go via
    /// the ExactSize path instead.
    pub fn opaque_for_each<Item, F>(
        &mut self,
        handle: Var<crate::refer::SMutPtr<()>>,
        next_id: usize,
        drop_id: usize,
        consumer: F,
    ) where
        Item: StagedType + 'static,
        F: FnOnce(&mut Ctx, Var<Item>) + 'static,
    {
        // Element var: defined inside the body from `next`'s value register.
        let elem: Var<Item> = unsafe { self.var_unchecked() };
        let elem_id = elem.id;
        let handle_id = handle.id;

        // Build the body into a child Ctx (same shape as `while_loop`).
        let mut child = Ctx::new(self.next_var_id);
        consumer(&mut child, elem);
        self.next_var_id = child.next_var_id;
        let body_actions = child.actions;

        self.actions.push(Box::new(move |ctx| {
            let header = ctx.create_block();
            let body = ctx.create_block();
            let exit = ctx.create_block();

            ctx.jump(header, &[]);

            // header: call the canonical thunk and branch on the stored tag.
            ctx.switch_to_block(header);
            let it_val = ctx
                .resolve_var::<crate::refer::SMutPtr<()>>(handle_id)
                .leaf();
            let next_ref = ctx.declare_extern_func(next_id);
            let mut args = Vec::with_capacity(1);
            crate::ffi::push_extern_value::<crate::refer::SMutPtr<()>>(ctx, &mut args, it_val);
            let option_ptr = crate::ffi::emit_extern_call::<crate::option::COptionType<Item>>(
                ctx, next_ref, args,
            )
            .leaf();
            let tag = ctx.load(ScalarType::I64, option_ptr, 0);
            // Single source of truth for the COption payload offset (see
            // COptionType::payload_offset); do not re-derive align_up(8, align) here.
            let payload_offset = crate::option::COptionType::<Item>::payload_offset() as i64;
            let payload_ptr = ctx.ptr_offset_const(option_ptr, payload_offset);
            let val = ctx.load_value::<Item>(payload_ptr);
            ctx.brif(tag, body, &[], exit, &[]);

            // body: bind elem = value register (already the element's ABI type,
            // since COption<Item> returns [tag, ...Item's abi...]), replay
            // consumer, loop.
            ctx.switch_to_block(body);
            ctx.seal_block(body);
            ctx.assign_var::<Item>(elem_id, val, false);
            ctx.loop_exit_stack.push(exit);
            for action in body_actions {
                action(ctx);
            }
            ctx.loop_exit_stack.pop();
            ctx.jump(header, &[]);
            ctx.seal_block(header);

            // exit: free the iterator (reached by None and by break_loop).
            ctx.switch_to_block(exit);
            ctx.seal_block(exit);
            let it_val2 = ctx
                .resolve_var::<crate::refer::SMutPtr<()>>(handle_id)
                .leaf();
            let drop_ref = ctx.declare_extern_func(drop_id);
            let mut args = Vec::with_capacity(1);
            crate::ffi::push_extern_value::<crate::refer::SMutPtr<()>>(ctx, &mut args, it_val2);
            crate::ffi::emit_extern_call::<()>(ctx, drop_ref, args);
        }));
    }

    /// Drive a push loop over a *reused-storage* opaque iterator (see
    /// `iter::opaque`): reserve one per-level slot in the JIT frame, let the
    /// producer build the iterator into it (`init_call`), then drive it through
    /// the slot's hand-rolled mini-vtable with indirect calls:
    /// ```text
    /// reserve slot[size]            ; once, reused every outer iteration
    /// init(args.., &slot)           ; producer fills next/drop ptrs + storage
    /// header: data = slot.data
    ///         (tag, val) = call_indirect slot.next (data)   ; COption in regs
    ///         brif tag, body, exit
    /// body:   elem = val ; <consumer> ; jump header
    /// exit:   call_indirect slot.drop (slot.data)
    /// ```
    /// The slot is a function-level stack slot, so nesting just reserves one per
    /// level; `drop` runs on every exit (None and `break_loop`).
    #[allow(clippy::too_many_arguments)]
    pub fn reused_opaque_for_each<Item, InitFn, F>(
        &mut self,
        slot_size: u32,
        slot_align_shift: u8,
        next_off: i32,
        drop_off: i32,
        data_off: i32,
        init_call: InitFn,
        consumer: F,
    ) where
        Item: StagedType + 'static,
        InitFn: FnOnce(&mut CompilationContext, ValueId) + 'static,
        F: FnOnce(&mut Ctx, Var<Item>) + 'static,
    {
        let elem: Var<Item> = unsafe { self.var_unchecked() };
        let elem_id = elem.id;

        let mut child = Ctx::new(self.next_var_id);
        consumer(&mut child, elem);
        self.next_var_id = child.next_var_id;
        let body_actions = child.actions;

        self.actions.push(Box::new(move |ctx| {
            // One per-level slot, reserved once in the frame and reused.
            let slot = ctx.alloc_stack_slot(slot_size, slot_align_shift);
            let slot_ptr = ctx.stack_addr(slot, 0);

            // Producer builds the iterator into the slot (fills the mini-vtable).
            init_call(ctx, slot_ptr);

            // Canonical storage-pointer signatures for indirect next/drop:
            // (data slot ptr, output slot ptr) -> void.
            let storage_ptr_sig = SigSpec {
                params: vec![ScalarType::Ptr, ScalarType::Ptr],
                ret: None,
            };
            let next_sigref = ctx.import_signature(&storage_ptr_sig);
            let drop_sigref = ctx.import_signature(&storage_ptr_sig);

            let data_slot = ctx.alloc_stack_slot(8, 3);
            let data_ptr = ctx.stack_addr(data_slot, 0);
            let option_slot = ctx.alloc_stack_slot(
                crate::option::COptionType::<Item>::size_of() as u32,
                crate::option::COptionType::<Item>::align_of().trailing_zeros() as u8,
            );
            let option_ptr = ctx.stack_addr(option_slot, 0);

            let header = ctx.create_block();
            let body = ctx.create_block();
            let exit = ctx.create_block();
            ctx.jump(header, &[]);

            // header: load data + next ptr, call it, branch on the tag register.
            ctx.switch_to_block(header);
            let data = ctx.load(ScalarType::Ptr, slot_ptr, data_off);
            let next_fn = ctx.load(ScalarType::Ptr, slot_ptr, next_off);
            ctx.store(data, data_ptr, 0);
            ctx.call_indirect(next_sigref, next_fn, &[data_ptr, option_ptr]);
            let tag = ctx.load(ScalarType::I64, option_ptr, 0);
            // Single source of truth for the COption payload offset (see
            // COptionType::payload_offset); do not re-derive align_up(8, align) here.
            let payload_offset = crate::option::COptionType::<Item>::payload_offset() as i64;
            let payload_ptr = ctx.ptr_offset_const(option_ptr, payload_offset);
            let val = ctx.load_value::<Item>(payload_ptr);
            ctx.brif(tag, body, &[], exit, &[]);

            // body: bind elem = value register, replay consumer, loop.
            ctx.switch_to_block(body);
            ctx.seal_block(body);
            ctx.assign_var::<Item>(elem_id, val, false);
            ctx.loop_exit_stack.push(exit);
            for action in body_actions {
                action(ctx);
            }
            ctx.loop_exit_stack.pop();
            ctx.jump(header, &[]);
            ctx.seal_block(header);

            // exit: drop the iterator (frees only if it was heap-boxed).
            ctx.switch_to_block(exit);
            ctx.seal_block(exit);
            let data2 = ctx.load(ScalarType::Ptr, slot_ptr, data_off);
            let drop_fn = ctx.load(ScalarType::Ptr, slot_ptr, drop_off);
            ctx.store(data2, data_ptr, 0);
            ctx.call_indirect(drop_sigref, drop_fn, &[data_ptr, option_ptr]);
        }));
    }

    pub fn break_loop(&mut self) {
        self.actions.push(Box::new(move |ctx| {
            let exit = *ctx
                .loop_exit_stack
                .last()
                .expect("break_loop called outside of a loop");
            ctx.jump(exit, &[]);
            // The current block is now terminated; switch to a fresh (dead)
            // block so any following emitted instructions remain well-formed.
            let dead = ctx.create_block();
            ctx.switch_to_block(dead);
            ctx.seal_block(dead);
        }));
    }

    /// Emit a one-sided conditional: `if cond { then }`.
    pub fn if_then<C, F>(&mut self, cond: C, then: F)
    where
        C: Staged<Out = bool> + 'static,
        F: FnOnce(&mut Ctx),
    {
        let mut child = Ctx::new(self.next_var_id);
        then(&mut child);
        self.next_var_id = child.next_var_id;
        let then_actions = child.actions;

        self.actions.push(Box::new(move |ctx| {
            let then_block = ctx.create_block();
            let merge_block = ctx.create_block();

            let cond_val = cond.codegen(ctx);
            ctx.brif(cond_val.leaf(), then_block, &[], merge_block, &[]);

            ctx.switch_to_block(then_block);
            ctx.seal_block(then_block);
            for action in then_actions {
                action(ctx);
            }
            ctx.jump(merge_block, &[]);

            ctx.switch_to_block(merge_block);
            ctx.seal_block(merge_block);
        }));
    }

    /// Emit a two-sided conditional: `if cond { then } else { els }`.
    ///
    /// Both branches are side-effecting (they emit into the `Ctx` they receive)
    /// and the construct yields no value — sequence with `ctx.var`/`ctx.store`
    /// if you need a result out.
    pub fn if_then_else<C, T, E>(&mut self, cond: C, then: T, els: E)
    where
        C: Staged<Out = bool> + 'static,
        T: FnOnce(&mut Ctx),
        E: FnOnce(&mut Ctx),
    {
        // Stage both branches into child contexts, keeping var ids disjoint.
        let mut then_child = Ctx::new(self.next_var_id);
        then(&mut then_child);
        let mut else_child = Ctx::new(then_child.next_var_id);
        els(&mut else_child);
        self.next_var_id = else_child.next_var_id;
        let then_actions = then_child.actions;
        let else_actions = else_child.actions;

        self.actions.push(Box::new(move |ctx| {
            let then_block = ctx.create_block();
            let else_block = ctx.create_block();
            let merge_block = ctx.create_block();

            let cond_val = cond.codegen(ctx);
            ctx.brif(cond_val.leaf(), then_block, &[], else_block, &[]);

            ctx.switch_to_block(then_block);
            ctx.seal_block(then_block);
            for action in then_actions {
                action(ctx);
            }
            ctx.jump(merge_block, &[]);

            ctx.switch_to_block(else_block);
            ctx.seal_block(else_block);
            for action in else_actions {
                action(ctx);
            }
            ctx.jump(merge_block, &[]);

            ctx.switch_to_block(merge_block);
            ctx.seal_block(merge_block);
        }));
    }
}

/// Backward-compatible alias. Prefer `Ctx`.
pub type VarBuilder = Ctx;

// =============================================================================
// Compiler: Owns everything, coordinates compilation
// =============================================================================

/// Stored metadata for an external function
pub(crate) struct ExternFnDef {
    /// Unique link name (`NAME` + the fn pointer), so distinct monomorphizations
    /// of a generic extern fn don't collide in cranelift's name-keyed symbol map.
    pub name: String,
    pub num_params: usize,
    pub fn_ptr: *const u8,
}

/// Emit one function body under the private storage-pointer ABI, **backend-neutrally**.
///
/// The ABI is uniform: `params` is one storage pointer per logical argument followed by a
/// single output pointer, and the function returns `void`. This unpacks each argument from
/// its storage pointer into a shape-aware variable binding,
/// runs `body` to produce the result, and writes the result back through the output pointer.
/// Every step goes through the neutral [`Backend`](crate::staged::Backend) ops on `ctx`, so
/// the same code drives Cranelift and MLIR; the caller supplies the (backend-specific)
/// function entry and terminating `return`.
pub(crate) fn emit_function_body(
    ctx: &mut CompilationContext,
    params: &[ValueId],
    param_infos: &[TypeInfo],
    param_var_ids: &[usize],
    body: impl FnOnce(&mut CompilationContext) -> Value,
    return_info: &TypeInfo,
) {
    for (index, info) in param_infos.iter().enumerate() {
        let var_id = param_var_ids[index];
        let storage_ptr = params[index];

        if info.is_fat_pointer {
            // A slice arrives as a `{ptr, len}` pair in memory; load it once into a register
            // pair — that *is* the slice's fat value (read back by `resolve_var`). The data
            // pointer is `Ptr` (an `llvm.ptr` on MLIR); `len` is `I64`. This memory→fat load
            // is the one boundary where a slice touches memory on the way in.
            let ptr_value = ctx.load(ScalarType::Ptr, storage_ptr, 0);
            let len_value = ctx.load(ScalarType::I64, storage_ptr, 8);
            let ptr_var = ctx.declare_var(ScalarType::Ptr);
            let len_var = ctx.declare_var(ScalarType::I64);
            ctx.def_var(ptr_var, ptr_value);
            ctx.def_var(len_var, len_value);
            ctx.variables.insert(
                var_id,
                VarValue::Fat {
                    ptr: ptr_var,
                    len: len_var,
                },
            );
        } else if info.is_aggregate {
            // A non-slice aggregate is represented by a pointer to its storage.
            let param_var = ctx.declare_var(ScalarType::Ptr);
            ctx.def_var(param_var, storage_ptr);
            ctx.variables.insert(var_id, VarValue::Scalar(param_var));
        } else {
            let param_value = if info.size == 0 {
                ctx.iconst(ScalarType::I8, 0)
            } else {
                ctx.load(info.repr, storage_ptr, 0)
            };
            let param_var = ctx.declare_var(info.repr);
            ctx.def_var(param_var, param_value);
            ctx.variables.insert(var_id, VarValue::Scalar(param_var));
        }
    }

    let result = body(ctx);

    let output_ptr = params[param_infos.len()];
    if return_info.is_fat_pointer {
        // A returned slice is a fat value; write its (ptr, len) into the output `{ptr,len}`.
        let (ptr, len) = result.parts();
        ctx.store(ptr, output_ptr, 0);
        ctx.store(len, output_ptr, 8);
    } else if return_info.is_aggregate {
        ctx.copy_nonoverlapping(
            output_ptr,
            result.leaf(),
            return_info.size as usize,
            return_info.alignment as usize,
        );
    } else if return_info.size != 0 {
        let result = result.leaf();
        assert_eq!(
            result.scalar_type(),
            return_info.repr,
            "function result: expected {:?}, found {:?}",
            return_info.repr,
            result.scalar_type()
        );
        ctx.store(result, output_ptr, 0);
    }
}

/// The central coordinator for staged computations.
///
/// `Compiler` owns all function definitions and variable IDs. It provides
/// methods to create functions and variables, and to compile expressions
/// to native code.
/// Which code-generation backend [`Compiler::compile`] targets.
///
/// Cranelift is the default (pure-Rust, fast compile). `Llvm` (behind `--features llvm`)
/// JITs through MLIR — same neutral AST, harder optimization.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum JitBackend {
    Cranelift,
    #[cfg(feature = "llvm")]
    Llvm,
}

pub struct Compiler<'a> {
    /// Function definitions indexed by ID
    functions: Vec<Option<FunDef>>,
    /// External function definitions indexed by ID
    extern_functions: Vec<ExternFnDef>,
    /// Next variable ID to assign
    next_var_id: usize,
    /// Code-generation backend `compile` targets.
    backend: JitBackend,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Default for Compiler<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Compiler<'a> {
    /// Create a new compiler (targeting the default Cranelift backend).
    pub fn new() -> Self {
        Compiler {
            functions: Vec::new(),
            extern_functions: Vec::new(),
            next_var_id: 0,
            backend: JitBackend::Cranelift,
            _marker: PhantomData,
        }
    }

    /// Select the code-generation backend (builder style). See [`JitBackend`].
    pub fn with_backend(mut self, backend: JitBackend) -> Self {
        self.backend = backend;
        self
    }

    /// Register an external function and get a handle to call it.
    ///
    /// The type parameter `S` must be a type generated by the `#[extern_fn]` macro,
    /// which implements the `ExternFn` trait with all necessary metadata.
    ///
    /// # Example
    ///
    /// ```ignore
    /// use rust_lms_derive::extern_fn;
    ///
    /// #[extern_fn]
    /// #[no_mangle]
    /// pub extern "C" fn my_add(x: i64, y: i64) -> i64 {
    ///     x + y
    /// }
    ///
    /// let mut compiler = Compiler::new();
    /// let my_add = compiler.extern_fn::<MyAddExtern>();
    /// let result = call_extern2(my_add, x, y);
    /// ```
    pub fn extern_fn<S: crate::ffi::ExternFn>(&mut self) -> crate::ffi::ExternRef<S> {
        let extern_id = self.extern_functions.len();

        self.extern_functions.push(ExternFnDef {
            // Disambiguate generic instantiations (same NAME, different fn ptr).
            name: format!("{}_{:x}", S::NAME, S::FN_PTR as usize),
            num_params: S::NUM_PARAMS,
            fn_ptr: S::FN_PTR,
        });

        crate::ffi::ExternRef::new(extern_id)
    }

    /// Define a unary function.
    ///
    /// The body function is called immediately to build the expression tree.
    /// No Cranelift calls happen until `compile()` is called.
    ///
    /// The body function receives a `VarBuilder` context that allows creating
    /// local variables within the function.
    ///
    /// # Struct Pass-by-Value
    ///
    /// If `A` is a `#[repr(C)] Copy` struct, the function will accept the struct
    /// by value at the Rust level (`fn(Point)`), but internally store it in a
    /// stack slot for field access.
    pub fn fun1<A, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef1<A, OUT>
    where
        A: StagedType,
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder, Var<A>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun1(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive unary function.
    ///
    /// Similar to `fun1`, but the body function receives a reference to itself,
    /// allowing for recursive calls. The function reference is passed as the first
    /// argument to the body closure, followed by the VarBuilder and parameter.
    ///
    /// # Example
    /// ```ignore
    /// let factorial = compiler.fun1_rec("factorial", |f, ctx, x: Var<i64>| {
    ///     // Can create local variables
    ///     let temp = ctx.let_var(0i64);
    ///     // Recursive call: f(x - 1)
    ///     call1(f, sub(x, Const::<i64>::new(1)))
    /// });
    /// ```
    pub fn fun1_rec<A, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef1<A, OUT>
    where
        A: StagedType,
        OUT: StagedType,
        F: FnOnce(FunRef1<A, OUT>, &mut VarBuilder, Var<A>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun1_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a zero-argument function.
    pub fn fun0<OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef0<OUT>
    where
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun0(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive zero-argument function.
    pub fn fun0_rec<OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef0<OUT>
    where
        OUT: StagedType,
        F: FnOnce(FunRef0<OUT>, &mut VarBuilder) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun0_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a binary function.
    pub fn fun2<A, B, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef2<A, B, OUT>
    where
        A: StagedType,
        B: StagedType,
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder, Var<A>, Var<B>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun2(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive binary function.
    pub fn fun2_rec<A, B, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef2<A, B, OUT>
    where
        A: StagedType,
        B: StagedType,
        OUT: StagedType,
        F: FnOnce(FunRef2<A, B, OUT>, &mut VarBuilder, Var<A>, Var<B>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun2_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a ternary function.
    pub fn fun3<A, B, C, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef3<A, B, C, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder, Var<A>, Var<B>, Var<C>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun3(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive ternary function.
    pub fn fun3_rec<A, B, C, OUT, F, BODY>(
        &mut self,
        name: &str,
        body_fn: F,
    ) -> FunRef3<A, B, C, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        OUT: StagedType,
        F: FnOnce(FunRef3<A, B, C, OUT>, &mut VarBuilder, Var<A>, Var<B>, Var<C>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun3_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a 4-parameter function.
    pub fn fun4<A, B, C, D, OUT, F, BODY>(
        &mut self,
        name: &str,
        body_fn: F,
    ) -> FunRef4<A, B, C, D, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder, Var<A>, Var<B>, Var<C>, Var<D>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun4(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive 4-parameter function.
    pub fn fun4_rec<A, B, C, D, OUT, F, BODY>(
        &mut self,
        name: &str,
        body_fn: F,
    ) -> FunRef4<A, B, C, D, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        OUT: StagedType,
        F: FnOnce(
            FunRef4<A, B, C, D, OUT>,
            &mut VarBuilder,
            Var<A>,
            Var<B>,
            Var<C>,
            Var<D>,
        ) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun4_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a 5-parameter function.
    pub fn fun5<A, B, C, D, E, OUT, F, BODY>(
        &mut self,
        name: &str,
        body_fn: F,
    ) -> FunRef5<A, B, C, D, E, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        E: StagedType,
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder, Var<A>, Var<B>, Var<C>, Var<D>, Var<E>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun5(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive 5-parameter function.
    pub fn fun5_rec<A, B, C, D, E, OUT, F, BODY>(
        &mut self,
        name: &str,
        body_fn: F,
    ) -> FunRef5<A, B, C, D, E, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        E: StagedType,
        OUT: StagedType,
        F: FnOnce(
            FunRef5<A, B, C, D, E, OUT>,
            &mut VarBuilder,
            Var<A>,
            Var<B>,
            Var<C>,
            Var<D>,
            Var<E>,
        ) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun5_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a 6-parameter function.
    pub fn fun6<A, B, C, D, E, FF, OUT, FN, BODY>(
        &mut self,
        name: &str,
        body_fn: FN,
    ) -> FunRef6<A, B, C, D, E, FF, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        E: StagedType,
        FF: StagedType,
        OUT: StagedType,
        FN: FnOnce(&mut VarBuilder, Var<A>, Var<B>, Var<C>, Var<D>, Var<E>, Var<FF>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun6(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive 6-parameter function.
    pub fn fun6_rec<A, B, C, D, E, FF, OUT, FN, BODY>(
        &mut self,
        name: &str,
        body_fn: FN,
    ) -> FunRef6<A, B, C, D, E, FF, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        E: StagedType,
        FF: StagedType,
        OUT: StagedType,
        FN: FnOnce(
            FunRef6<A, B, C, D, E, FF, OUT>,
            &mut VarBuilder,
            Var<A>,
            Var<B>,
            Var<C>,
            Var<D>,
            Var<E>,
            Var<FF>,
        ) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun6_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a 7-parameter function.
    pub fn fun7<A, B, C, D, E, FF, G, OUT, FN, BODY>(
        &mut self,
        name: &str,
        body_fn: FN,
    ) -> FunRef7<A, B, C, D, E, FF, G, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        E: StagedType,
        FF: StagedType,
        G: StagedType,
        OUT: StagedType,
        FN: FnOnce(
            &mut VarBuilder,
            Var<A>,
            Var<B>,
            Var<C>,
            Var<D>,
            Var<E>,
            Var<FF>,
            Var<G>,
        ) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun7(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive 7-parameter function.
    pub fn fun7_rec<A, B, C, D, E, FF, G, OUT, FN, BODY>(
        &mut self,
        name: &str,
        body_fn: FN,
    ) -> FunRef7<A, B, C, D, E, FF, G, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        E: StagedType,
        FF: StagedType,
        G: StagedType,
        OUT: StagedType,
        FN: FnOnce(
            FunRef7<A, B, C, D, E, FF, G, OUT>,
            &mut VarBuilder,
            Var<A>,
            Var<B>,
            Var<C>,
            Var<D>,
            Var<E>,
            Var<FF>,
            Var<G>,
        ) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun7_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define an 8-parameter function.
    pub fn fun8<A, B, C, D, E, FF, G, H, OUT, FN, BODY>(
        &mut self,
        name: &str,
        body_fn: FN,
    ) -> FunRef8<A, B, C, D, E, FF, G, H, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        E: StagedType,
        FF: StagedType,
        G: StagedType,
        H: StagedType,
        OUT: StagedType,
        FN: FnOnce(
            &mut VarBuilder,
            Var<A>,
            Var<B>,
            Var<C>,
            Var<D>,
            Var<E>,
            Var<FF>,
            Var<G>,
            Var<H>,
        ) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun8(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Define a recursive 8-parameter function.
    pub fn fun8_rec<A, B, C, D, E, FF, G, H, OUT, FN, BODY>(
        &mut self,
        name: &str,
        body_fn: FN,
    ) -> FunRef8<A, B, C, D, E, FF, G, H, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        D: StagedType,
        E: StagedType,
        FF: StagedType,
        G: StagedType,
        H: StagedType,
        OUT: StagedType,
        FN: FnOnce(
            FunRef8<A, B, C, D, E, FF, G, H, OUT>,
            &mut VarBuilder,
            Var<A>,
            Var<B>,
            Var<C>,
            Var<D>,
            Var<E>,
            Var<FF>,
            Var<G>,
            Var<H>,
        ) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        FunDef::make_fun8_rec(&mut self.next_var_id, &mut self.functions, name, body_fn)
    }

    /// Compile an expression to native code.
    ///
    /// This compiles all referenced functions and the main expression,
    /// returning a `Compiled<T>` that owns the JIT module and can extract
    /// the computed value.
    ///
    /// Generated functions use one storage pointer per logical parameter and
    /// one caller-owned output pointer. This keeps aggregate classification out
    /// of the private JIT ABI.
    /// Compile the top-level expression to native code via the selected [`JitBackend`].
    pub fn compile<S: Staged>(self, expr: S) -> Result<Compiled<'a, S::Out>, CompileError> {
        match self.backend {
            JitBackend::Cranelift => self.compile_cranelift(expr),
            #[cfg(feature = "llvm")]
            JitBackend::Llvm => self.compile_llvm(expr),
        }
    }

    /// Compile through the MLIR backend: build `__main__` (from `expr`) alongside the
    /// helper functions and JIT the module. The result reuses the same [`Compiled`]/`run`/
    /// `as_fn` machinery as Cranelift — only the `Executable` resource differs.
    #[cfg(feature = "llvm")]
    fn compile_llvm<S: Staged>(self, expr: S) -> Result<Compiled<'a, S::Out>, CompileError> {
        let return_info = TypeInfo::from_staged_type::<S::Out>();
        let (executable, main_ptr) = crate::llvm::assemble(
            self.functions,
            &self.extern_functions,
            &return_info,
            move |ctx| expr.codegen(ctx),
        )?;
        Ok(Compiled {
            executable: Some(Executable::Mlir(executable)),
            main_ptr,
            _phantom: PhantomData,
        })
    }

    fn compile_cranelift<S: Staged>(self, expr: S) -> Result<Compiled<'a, S::Out>, CompileError> {
        // Create ISA with optimization level "speed" and other performance settings
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| CompileError::JitError(e.to_string()))?;
        flag_builder
            .set("use_colocated_libcalls", "true")
            .map_err(|e| CompileError::JitError(e.to_string()))?;
        let isa_builder =
            cranelift_native::builder().map_err(|e| CompileError::JitError(e.to_string()))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| CompileError::JitError(e.to_string()))?;

        // Create the JIT module with optimized ISA
        let mut builder = JITBuilder::with_isa(isa, default_libcall_names());

        // Register external function symbols
        for extern_def in &self.extern_functions {
            builder.symbol(extern_def.name.clone(), extern_def.fn_ptr);
        }

        let mut module = JITModule::new(builder);

        // Use the convention selected from the complete target triple. In
        // particular, Apple AArch64 is distinct from generic System V.
        let call_conv = module.isa().default_call_conv();

        // First pass: declare all internal functions
        let mut func_map: HashMap<usize, FuncId> = HashMap::new();

        for (id, func_opt) in self.functions.iter().enumerate() {
            if let Some(func_def) = func_opt {
                let mut sig = module.make_signature();
                sig.call_conv = call_conv;

                // Private JIT ABI: one storage pointer per logical parameter,
                // followed by one output storage pointer.
                for _ in &func_def.param_infos {
                    sig.params.push(AbiParam::new(types::I64));
                }
                sig.params.push(AbiParam::new(types::I64));

                let func_id = module
                    .declare_function(&func_def.name, Linkage::Local, &sig)
                    .map_err(|e| CompileError::ModuleError(e.to_string()))?;

                func_map.insert(id, func_id);
            }
        }

        // Declare all external functions
        let mut extern_func_ids: HashMap<usize, FuncId> = HashMap::new();

        for (id, extern_def) in self.extern_functions.iter().enumerate() {
            let mut sig = module.make_signature();
            sig.call_conv = call_conv;

            for _ in 0..=extern_def.num_params {
                sig.params.push(AbiParam::new(types::I64));
            }

            // Declare the function (will be linked to the actual function pointer)
            let func_id = module
                .declare_function(&extern_def.name, Linkage::Import, &sig)
                .map_err(|e| CompileError::ModuleError(e.to_string()))?;

            extern_func_ids.insert(id, func_id);
        }

        // Declare the main function
        let mut main_sig = module.make_signature();
        main_sig.call_conv = call_conv;

        // `__main__` writes its result into caller-owned storage.
        main_sig.params.push(AbiParam::new(types::I64));

        let main_func_id = module
            .declare_function("__main__", Linkage::Local, &main_sig)
            .map_err(|e| CompileError::ModuleError(e.to_string()))?;

        // Second pass: define all functions
        // We need to consume self.functions since FunDef contains FnOnce
        let mut functions = self.functions;

        for (id, func_opt) in functions.iter_mut().enumerate() {
            if let Some(func_def) = func_opt.take() {
                let func_id = func_map[&id];

                let mut sig = module.make_signature();
                sig.call_conv = call_conv;

                for _ in &func_def.param_infos {
                    sig.params.push(AbiParam::new(types::I64));
                }
                sig.params.push(AbiParam::new(types::I64));

                let mut func_ctx = module.make_context();
                func_ctx.func.signature = sig;

                {
                    let mut builder_context = FunctionBuilderContext::new();
                    let mut builder =
                        FunctionBuilder::new(&mut func_ctx.func, &mut builder_context);
                    let entry_block = builder.create_block();
                    builder.append_block_params_for_function_params(entry_block);
                    builder.switch_to_block(entry_block);
                    builder.seal_block(entry_block);

                    let mut variables = HashMap::new();

                    // Storage pointers (N args + output) as neutral handles.
                    let params: Vec<ValueId> = builder
                        .block_params(entry_block)
                        .iter()
                        .map(|value| ValueId::from_cranelift(*value, ScalarType::Ptr))
                        .collect();

                    {
                        let mut backend = CraneliftBackend {
                            builder: &mut builder,
                            module: &mut module,
                            func_ids: &func_map,
                            extern_func_ids: &extern_func_ids,
                            func_ref_cache: HashMap::new(),
                            extern_ref_cache: HashMap::new(),
                            sig_specs: HashMap::new(),
                        };
                        let mut ctx = CompilationContext {
                            backend: &mut backend,
                            variables: &mut variables,
                            unit_value: None,
                            block_params: HashMap::new(),
                            loop_exit_stack: Vec::new(),
                        };
                        emit_function_body(
                            &mut ctx,
                            &params,
                            &func_def.param_infos,
                            &func_def.param_var_ids,
                            func_def.body,
                            &func_def.return_info,
                        );
                    }

                    builder.ins().return_(&[]);
                    builder.finalize(module.isa().frontend_config());
                }

                // Debug output for Cranelift IR
                if std::env::var("RUST_LMS_DEBUG_IR").is_ok() {
                    eprintln!("=== Function: {} ===", func_def.name);
                    eprintln!("{}", func_ctx.func);
                    eprintln!();
                }

                module
                    .define_function(func_id, &mut func_ctx)
                    .map_err(|e| CompileError::ModuleError(e.to_string()))?;
                module.clear_context(&mut func_ctx);
            }
        }

        // Define the main function
        {
            let mut func_ctx = module.make_context();
            func_ctx.func.signature = main_sig;

            {
                let mut builder_context = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut func_ctx.func, &mut builder_context);
                let entry_block = builder.create_block();
                builder.append_block_params_for_function_params(entry_block);
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut variables = HashMap::new();

                // `__main__` is a zero-argument function under the storage-pointer ABI: its
                // one parameter is the output pointer.
                let params: Vec<ValueId> = builder
                    .block_params(entry_block)
                    .iter()
                    .map(|value| ValueId::from_cranelift(*value, ScalarType::Ptr))
                    .collect();
                let return_info = TypeInfo::from_staged_type::<S::Out>();

                {
                    let mut backend = CraneliftBackend {
                        builder: &mut builder,
                        module: &mut module,
                        func_ids: &func_map,
                        extern_func_ids: &extern_func_ids,
                        func_ref_cache: HashMap::new(),
                        extern_ref_cache: HashMap::new(),
                        sig_specs: HashMap::new(),
                    };
                    let mut ctx = CompilationContext {
                        backend: &mut backend,
                        variables: &mut variables,
                        unit_value: None,
                        block_params: HashMap::new(),
                        loop_exit_stack: Vec::new(),
                    };
                    emit_function_body(
                        &mut ctx,
                        &params,
                        &[],
                        &[],
                        |ctx| expr.codegen(ctx),
                        &return_info,
                    );
                }

                builder.ins().return_(&[]);
                builder.finalize(module.isa().frontend_config());
            }

            // Debug output for main function IR
            if std::env::var("RUST_LMS_DEBUG_IR").is_ok() {
                eprintln!("=== Function: __main__ ===");
                eprintln!("{}", func_ctx.func);
                eprintln!();
            }

            module
                .define_function(main_func_id, &mut func_ctx)
                .map_err(|e| CompileError::ModuleError(e.to_string()))?;
            module.clear_context(&mut func_ctx);
        }

        // Finalize the module (extern functions are already registered via JITBuilder::symbol)
        module
            .finalize_definitions()
            .map_err(|e| CompileError::ModuleError(e.to_string()))?;

        // Get the main function pointer
        let main_ptr = module.get_finalized_function(main_func_id);

        Ok(Compiled {
            executable: Some(Executable::Cranelift(Box::new(module))),
            main_ptr,
            _phantom: PhantomData,
        })
    }
}

// =============================================================================
// CompileError
// =============================================================================

/// Errors that can occur during compilation
#[derive(Debug)]
pub enum CompileError {
    JitError(String),
    ModuleError(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::JitError(msg) => write!(f, "JIT error: {}", msg),
            CompileError::ModuleError(msg) => write!(f, "Module error: {}", msg),
        }
    }
}

impl std::error::Error for CompileError {}

// =============================================================================
// Compiled<T>: The result of compilation
// =============================================================================

/// A compiled expression that owns its JIT module and executable memory.
///
/// Use [`run`](Self::run) for a plain expression or the arity-specific `call`
/// methods for a compiled staged function. Executable memory is reclaimed when
/// this value is dropped.
pub struct Compiled<'a, T: StagedType> {
    executable: Option<Executable>,
    main_ptr: *const u8,
    _phantom: PhantomData<&'a T>,
}

/// The backend-specific JIT resource a [`Compiled`] owns and frees on drop. `main_ptr`
/// points into whichever variant is live; keeping the resource here keeps it valid.
enum Executable {
    Cranelift(Box<JITModule>),
    #[cfg(feature = "llvm")]
    Mlir(crate::llvm::MlirExecutable),
}

impl<'a, T: StagedType> Drop for Compiled<'a, T> {
    fn drop(&mut self) {
        // SAFETY: safe entry points borrow this Compiled value, so no safe callable can
        // remain when Drop obtains exclusive access. Escaped pointers are governed by
        // `as_fn_unchecked`'s safety contract.
        match self.executable.take() {
            Some(Executable::Cranelift(module)) => unsafe { (*module).free_memory() },
            #[cfg(feature = "llvm")]
            Some(Executable::Mlir(executable)) => drop(executable),
            None => {}
        }
    }
}

impl<'a, T: StagedType> Compiled<'a, T> {
    /// Execute the compiled code and return the result.
    pub fn run(&self) -> T::RuntimeValue {
        let mut output = MaybeUninit::<T::RuntimeValue>::uninit();
        // SAFETY: `compile` creates `__main__` with the canonical one-output-
        // pointer signature and writes a valid `T::RuntimeValue` before return.
        let function: unsafe extern "C" fn(*mut u8) = unsafe { std::mem::transmute(self.main_ptr) };
        unsafe {
            function(output.as_mut_ptr().cast());
            output.assume_init()
        }
    }

    /// Read the generated function address returned by a function-valued
    /// `__main__` trampoline.
    unsafe fn function_entry_ptr(&self) -> *const u8 {
        let mut output = MaybeUninit::<*const u8>::uninit();
        let get_ptr: unsafe extern "C" fn(*mut u8) = unsafe { std::mem::transmute(self.main_ptr) };
        unsafe {
            get_ptr(output.as_mut_ptr().cast());
            output.assume_init()
        }
    }
}

/// A callable entry point tied to the lifetime of its owning [`Compiled`]
/// module.
///
/// The function pointer is deliberately private and this type does not
/// implement `Deref`: exposing the pointer would allow safe code to copy it and
/// outlive the executable memory. Use [`call`](CompiledFn::call), or use
/// `Compiled::as_fn_unchecked` when a foreign API genuinely requires a bare
/// function pointer.
///
/// Reference results are bounded by both the invocation arguments and the
/// compiled module. They cannot escape a shorter-lived argument:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// let mut compiler = Compiler::new();
/// let identity = compiler.fun1("identity", |_ctx, value: Var<SRef<i64>>| value);
/// let compiled = compiler.compile(identity).unwrap();
/// let escaped = {
///     let value = 42i64;
///     compiled.call(&value)
/// };
/// assert_eq!(*escaped, 42);
/// ```
#[must_use = "a compiled entry point does nothing until it is called"]
pub struct CompiledFn<'compiled, F> {
    function: *const u8,
    _signature: PhantomData<fn() -> F>,
    _owner: PhantomData<&'compiled JITModule>,
}

impl<F> Clone for CompiledFn<'_, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<F> Copy for CompiledFn<'_, F> {}

macro_rules! raw_argument_pointer {
    ($type:ident) => {
        *const u8
    };
}

// Generate borrowed entry points and direct calls for every function arity.
macro_rules! impl_compiled_fn {
    // Base case: zero parameters
    (0, $FunType:ident) => {
        impl<'compiled, OUT: RuntimeResult> CompiledFn<'compiled, $FunType<OUT>> {
            /// Invoke the entry point with a result lifetime bounded by its
            /// owning module.
            pub fn call<'call>(&self) -> OUT::Output<'call>
            where
                'compiled: 'call,
            {
                let mut output = MaybeUninit::<OUT::Output<'call>>::uninit();
                // SAFETY: RuntimeResult witnesses the output storage layout,
                // and the wrapper keeps the generated code live for 'call.
                let function: unsafe extern "C" fn(*mut u8) =
                    unsafe { std::mem::transmute(self.function) };
                unsafe {
                    function(output.as_mut_ptr().cast());
                    output.assume_init()
                }
            }
        }

        impl<'a, OUT: RuntimeResult> Compiled<'a, $FunType<OUT>> {
            /// Borrow the compiled function as a safe callable entry point.
            ///
            /// The borrow prevents this pattern from compiling:
            ///
            /// ```compile_fail
            /// use rust_lms::prelude::*;
            ///
            /// let mut compiler = Compiler::new();
            /// let function = compiler.fun0("one", |_ctx| Const::new(1i64));
            /// let entry = compiler.compile(function).unwrap().as_fn();
            /// assert_eq!(entry.call(), 1);
            /// ```
            pub fn as_fn(&self) -> CompiledFn<'_, $FunType<OUT>> {
                CompiledFn {
                    // SAFETY: the returned wrapper borrows self and never
                    // exposes the bare pointer.
                    function: unsafe { self.function_entry_ptr() },
                    _signature: PhantomData,
                    _owner: PhantomData,
                }
            }

            /// Invoke the compiled function while borrowing its owner.
            pub fn call<'call>(&'call self) -> OUT::Output<'call> {
                self.as_fn().call()
            }

            /// Extract the compiled function as an untracked function pointer.
            ///
            /// # Safety
            ///
            /// The returned pointer must never be invoked after `self` is
            /// dropped. Its argument must point to writable, properly aligned
            /// storage for `OUT::RuntimeValue`.
            pub unsafe fn as_fn_unchecked(
                &self,
            ) -> unsafe extern "C" fn(*mut u8) {
                unsafe { std::mem::transmute(self.function_entry_ptr()) }
            }
        }
    };
    // N parameters (N >= 1)
    ($n:tt, $FunType:ident, [$($T:ident : $arg:ident),+]) => {
        impl<'compiled, $($T: RuntimeParam,)+ OUT: RuntimeResult>
            CompiledFn<'compiled, $FunType<$($T,)+ OUT>>
        {
            /// Invoke the entry point with one fresh lifetime shared by its
            /// reference arguments and result.
            #[allow(clippy::too_many_arguments)]
            pub fn call<'call>(&self, $($arg: $T::Arg<'call>),+) -> OUT::Output<'call>
            where
                'compiled: 'call,
            {
                let mut output = MaybeUninit::<OUT::Output<'call>>::uninit();
                // SAFETY: RuntimeParam and RuntimeResult witness the storage
                // layouts. The shared 'call lifetime enforces the staged
                // reference contract at each invocation.
                let function: unsafe extern "C" fn(
                    $(raw_argument_pointer!($T)),+,
                    *mut u8,
                ) =
                    unsafe { std::mem::transmute(self.function) };
                unsafe {
                    function(
                        $(std::ptr::from_ref(&$arg).cast::<u8>()),+,
                        output.as_mut_ptr().cast(),
                    );
                    output.assume_init()
                }
            }
        }

        impl<'a, $($T: RuntimeParam,)+ OUT: RuntimeResult>
            Compiled<'a, $FunType<$($T,)+ OUT>>
        {
            /// Borrow the compiled function as a safe callable entry point.
            pub fn as_fn(&self) -> CompiledFn<'_, $FunType<$($T,)+ OUT>> {
                CompiledFn {
                    // SAFETY: the returned wrapper borrows self and never
                    // exposes the bare pointer.
                    function: unsafe { self.function_entry_ptr() },
                    _signature: PhantomData,
                    _owner: PhantomData,
                }
            }

            /// Invoke the compiled function while borrowing its owner.
            #[allow(clippy::too_many_arguments)]
            pub fn call<'call>(&'call self, $($arg: $T::Arg<'call>),+) -> OUT::Output<'call> {
                self.as_fn().call($($arg),+)
            }

            /// Extract the compiled function as an untracked function pointer.
            ///
            /// # Safety
            ///
            /// The returned pointer must never be invoked after `self` is
            /// dropped. Each input must point to the exact runtime
            /// representation of its corresponding staged parameter, and the
            /// final pointer must reference writable, properly aligned output
            /// storage.
            pub unsafe fn as_fn_unchecked(
                &self,
            ) -> unsafe extern "C" fn($(raw_argument_pointer!($T)),+, *mut u8) {
                unsafe { std::mem::transmute(self.function_entry_ptr()) }
            }
        }
    };
}

impl_compiled_fn!(0, FunType0);
impl_compiled_fn!(1, FunType1, [A: a]);
impl_compiled_fn!(2, FunType2, [A: a, B: b]);
impl_compiled_fn!(3, FunType3, [A: a, B: b, C: c]);
impl_compiled_fn!(4, FunType4, [A: a, B: b, C: c, D: d]);
impl_compiled_fn!(5, FunType5, [A: a, B: b, C: c, D: d, E: e]);
impl_compiled_fn!(6, FunType6, [A: a, B: b, C: c, D: d, E: e, F: f]);
impl_compiled_fn!(7, FunType7, [A: a, B: b, C: c, D: d, E: e, F: f, G: g]);
impl_compiled_fn!(8, FunType8, [A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h]);
