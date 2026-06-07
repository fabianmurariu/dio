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
//! # Struct Pass-by-Value
//!
//! For `#[repr(C)] Copy` structs, this module implements pass-by-value semantics:
//! - At the Rust ABI level: `fn(Point) -> Point` (value semantics)
//! - At the Cranelift level: multiple i64 parameters, stored to stack slot
//! - Internally: pointer to stack slot for field access

use crate::staged::{assign, CompilationContext, Staged, Var};
use crate::types::StagedType;
use cranelift_codegen::ir::{
    types, AbiParam, InstBuilder, MemFlags, Signature, StackSlotData, StackSlotKind, Value,
};
use cranelift_codegen::settings::{self, Configurable};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use std::collections::HashMap;
use std::marker::PhantomData;

pub use crate::func_impl::*;

// =============================================================================
// StructInfo: Information about struct types for ABI handling
// =============================================================================

/// Information about a struct type needed for ABI handling.
///
/// This captures the struct's layout so we can:
/// 1. Pass it as multiple i64 values at the ABI boundary
/// 2. Create a properly-sized stack slot
/// 3. Store/load the values to/from the stack slot
#[derive(Clone, Debug)]
pub struct StructInfo {
    /// Size in bytes
    pub size: u32,
    /// Alignment in bytes
    pub alignment: u32,
    /// Number of i64 values needed at ABI boundary
    pub num_abi_values: usize,
}

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
    actions: Vec<Box<dyn FnOnce(&mut CompilationContext) + 'static>>,
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
        let ctype = T::cranelift_type();
        let id = v.id;
        let init_for_action = init_staged.clone();
        self.actions.push(Box::new(move |ctx| {
            let value = init_for_action.codegen(ctx);
            let cv = ctx.builder.declare_var(ctype);
            ctx.var_map.insert(id, cv);
            ctx.builder.def_var(cv, value);
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
        let ctype = T::cranelift_type();
        let id = v.id;
        self.actions.push(Box::new(move |ctx| {
            let value = init_staged.codegen(ctx);
            let cv = ctx.builder.declare_var(ctype);
            ctx.var_map.insert(id, cv);
            ctx.builder.def_var(cv, value);
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
        let ctype = T::cranelift_type();
        let id = v.id;
        self.actions.push(Box::new(move |ctx| {
            let value = expr.codegen(ctx);
            let cv = if let Some(&existing) = ctx.var_map.get(&id) {
                existing
            } else {
                let cv = ctx.builder.declare_var(ctype);
                ctx.var_map.insert(id, cv);
                cv
            };
            ctx.builder.def_var(cv, value);
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
        F: FnOnce(&mut Ctx) + 'static,
    {
        let cond = cond.into_staged();
        let mut child = Ctx::new(self.next_var_id);
        body(&mut child);
        self.next_var_id = child.next_var_id;
        let body_actions = child.actions;

        self.actions.push(Box::new(move |ctx| {
            let loop_header = ctx.builder.create_block();
            let loop_body = ctx.builder.create_block();
            let loop_exit = ctx.builder.create_block();

            ctx.builder.ins().jump(loop_header, &[]);

            ctx.builder.switch_to_block(loop_header);
            let cond_val = cond.codegen(ctx);
            ctx.builder
                .ins()
                .brif(cond_val, loop_body, &[], loop_exit, &[]);

            ctx.builder.switch_to_block(loop_body);
            ctx.builder.seal_block(loop_body);
            // Expose this loop's exit block so `break_loop` inside the body can
            // jump to it; pop once the body is fully emitted.
            ctx.loop_exit_stack.push(loop_exit);
            for action in body_actions {
                action(ctx);
            }
            ctx.loop_exit_stack.pop();
            ctx.builder.ins().jump(loop_header, &[]);
            ctx.builder.seal_block(loop_header);

            ctx.builder.switch_to_block(loop_exit);
            ctx.builder.seal_block(loop_exit);
        }));
    }

    /// Break out of the innermost enclosing loop.
    ///
    /// Emits a jump to the current loop's exit block. Typically used inside an
    /// `if_then` to exit early. Panics at codegen time if called outside a loop.
    /// Drive a push loop over an opaque external iterator via `next`/`drop`.
    ///
    /// Emits the register-consume loop (see `iter::opaque`):
    /// ```text
    /// header: (tag, val) = next(it)     ; COption<Item> in two return regs
    ///         brif tag, body, exit      ; tag: 1 = Some, 0 = None
    /// body:   elem = val ; <consumer> ; jump header
    /// exit:   drop(it)
    /// ```
    /// `drop` sits at the top of `exit`, which is reached both by the `None`
    /// branch and by any `break_loop` from the body, so the handle is always
    /// freed. The value register feeds the element `Var` directly — no stack
    /// slot, no `COption` materialization.
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
        let item_cty = Item::cranelift_type();

        // Build the body into a child Ctx (same shape as `while_loop`).
        let mut child = Ctx::new(self.next_var_id);
        consumer(&mut child, elem);
        self.next_var_id = child.next_var_id;
        let body_actions = child.actions;

        self.actions.push(Box::new(move |ctx| {
            let header = ctx.builder.create_block();
            let body = ctx.builder.create_block();
            let exit = ctx.builder.create_block();

            ctx.builder.ins().jump(header, &[]);

            // header: call next(it); branch on the discriminant register.
            ctx.builder.switch_to_block(header);
            let it = ctx.var_map[&handle_id];
            let it_val = ctx.builder.use_var(it);
            let next_ref = ctx.get_extern_func_ref(next_id);
            let call = ctx.builder.ins().call(next_ref, &[it_val]);
            let (tag, val) = {
                let r = ctx.builder.inst_results(call);
                (r[0], r[1])
            };
            ctx.builder.ins().brif(tag, body, &[], exit, &[]);

            // body: bind elem = value register (already the element's ABI type,
            // since COption<Item> returns [tag, ...Item's abi...]), replay
            // consumer, loop.
            ctx.builder.switch_to_block(body);
            ctx.builder.seal_block(body);
            let elem_cv = ctx.builder.declare_var(item_cty);
            ctx.var_map.insert(elem_id, elem_cv);
            ctx.builder.def_var(elem_cv, val);
            ctx.loop_exit_stack.push(exit);
            for action in body_actions {
                action(ctx);
            }
            ctx.loop_exit_stack.pop();
            ctx.builder.ins().jump(header, &[]);
            ctx.builder.seal_block(header);

            // exit: free the iterator (reached by None and by break_loop).
            ctx.builder.switch_to_block(exit);
            ctx.builder.seal_block(exit);
            let it_val2 = ctx.builder.use_var(it);
            let drop_ref = ctx.get_extern_func_ref(drop_id);
            ctx.builder.ins().call(drop_ref, &[it_val2]);
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
        InitFn: FnOnce(&mut CompilationContext, Value) + 'static,
        F: FnOnce(&mut Ctx, Var<Item>) + 'static,
    {
        let elem: Var<Item> = unsafe { self.var_unchecked() };
        let elem_id = elem.id;
        let item_cty = Item::cranelift_type();

        let mut child = Ctx::new(self.next_var_id);
        consumer(&mut child, elem);
        self.next_var_id = child.next_var_id;
        let body_actions = child.actions;

        self.actions.push(Box::new(move |ctx| {
            let call_conv = if cfg!(target_os = "windows") {
                cranelift_codegen::isa::CallConv::WindowsFastcall
            } else {
                cranelift_codegen::isa::CallConv::SystemV
            };

            // One per-level slot, reserved once in the frame and reused.
            let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                slot_size,
                slot_align_shift,
            ));
            let slot_ptr = ctx.builder.ins().stack_addr(types::I64, slot, 0);

            // Producer builds the iterator into the slot (fills the mini-vtable).
            init_call(ctx, slot_ptr);

            // `extern "C"` signatures for the indirect next/drop calls.
            let mut next_sig = Signature::new(call_conv);
            next_sig.params.push(AbiParam::new(types::I64)); // data ptr
            next_sig.returns.push(AbiParam::new(types::I64)); // COption tag
            next_sig.returns.push(AbiParam::new(item_cty)); // COption value
            let next_sigref = ctx.builder.import_signature(next_sig);

            let mut drop_sig = Signature::new(call_conv);
            drop_sig.params.push(AbiParam::new(types::I64));
            let drop_sigref = ctx.builder.import_signature(drop_sig);

            let header = ctx.builder.create_block();
            let body = ctx.builder.create_block();
            let exit = ctx.builder.create_block();
            ctx.builder.ins().jump(header, &[]);

            // header: load data + next ptr, call it, branch on the tag register.
            ctx.builder.switch_to_block(header);
            let data = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), slot_ptr, data_off);
            let next_fn =
                ctx.builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), slot_ptr, next_off);
            let call = ctx
                .builder
                .ins()
                .call_indirect(next_sigref, next_fn, &[data]);
            let (tag, val) = {
                let r = ctx.builder.inst_results(call);
                (r[0], r[1])
            };
            ctx.builder.ins().brif(tag, body, &[], exit, &[]);

            // body: bind elem = value register, replay consumer, loop.
            ctx.builder.switch_to_block(body);
            ctx.builder.seal_block(body);
            let elem_cv = ctx.builder.declare_var(item_cty);
            ctx.var_map.insert(elem_id, elem_cv);
            ctx.builder.def_var(elem_cv, val);
            ctx.loop_exit_stack.push(exit);
            for action in body_actions {
                action(ctx);
            }
            ctx.loop_exit_stack.pop();
            ctx.builder.ins().jump(header, &[]);
            ctx.builder.seal_block(header);

            // exit: drop the iterator (frees only if it was heap-boxed).
            ctx.builder.switch_to_block(exit);
            ctx.builder.seal_block(exit);
            let data2 = ctx
                .builder
                .ins()
                .load(types::I64, MemFlags::trusted(), slot_ptr, data_off);
            let drop_fn =
                ctx.builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), slot_ptr, drop_off);
            ctx.builder
                .ins()
                .call_indirect(drop_sigref, drop_fn, &[data2]);
        }));
    }

    pub fn break_loop(&mut self) {
        self.actions.push(Box::new(move |ctx| {
            let exit = *ctx
                .loop_exit_stack
                .last()
                .expect("break_loop called outside of a loop");
            ctx.builder.ins().jump(exit, &[]);
            // The current block is now terminated; switch to a fresh (dead)
            // block so any following emitted instructions remain well-formed.
            let dead = ctx.builder.create_block();
            ctx.builder.switch_to_block(dead);
            ctx.builder.seal_block(dead);
        }));
    }

    /// Emit a one-sided conditional: `if cond { then }`.
    pub fn if_then<C, F>(&mut self, cond: C, then: F)
    where
        C: Staged<Out = bool> + 'static,
        F: FnOnce(&mut Ctx) + 'static,
    {
        let mut child = Ctx::new(self.next_var_id);
        then(&mut child);
        self.next_var_id = child.next_var_id;
        let then_actions = child.actions;

        self.actions.push(Box::new(move |ctx| {
            let then_block = ctx.builder.create_block();
            let merge_block = ctx.builder.create_block();

            let cond_val = cond.codegen(ctx);
            ctx.builder
                .ins()
                .brif(cond_val, then_block, &[], merge_block, &[]);

            ctx.builder.switch_to_block(then_block);
            ctx.builder.seal_block(then_block);
            for action in then_actions {
                action(ctx);
            }
            ctx.builder.ins().jump(merge_block, &[]);

            ctx.builder.switch_to_block(merge_block);
            ctx.builder.seal_block(merge_block);
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
        T: FnOnce(&mut Ctx) + 'static,
        E: FnOnce(&mut Ctx) + 'static,
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
            let then_block = ctx.builder.create_block();
            let else_block = ctx.builder.create_block();
            let merge_block = ctx.builder.create_block();

            let cond_val = cond.codegen(ctx);
            ctx.builder
                .ins()
                .brif(cond_val, then_block, &[], else_block, &[]);

            ctx.builder.switch_to_block(then_block);
            ctx.builder.seal_block(then_block);
            for action in then_actions {
                action(ctx);
            }
            ctx.builder.ins().jump(merge_block, &[]);

            ctx.builder.switch_to_block(else_block);
            ctx.builder.seal_block(else_block);
            for action in else_actions {
                action(ctx);
            }
            ctx.builder.ins().jump(merge_block, &[]);

            ctx.builder.switch_to_block(merge_block);
            ctx.builder.seal_block(merge_block);
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
    pub param_abi_types: Vec<Vec<cranelift_codegen::ir::Type>>,
    pub return_abi_types: Vec<cranelift_codegen::ir::Type>,
    pub fn_ptr: *const u8,
}

/// The central coordinator for staged computations.
///
/// `Compiler` owns all function definitions and variable IDs. It provides
/// methods to create functions and variables, and to compile expressions
/// to native code.
pub struct Compiler<'a> {
    /// Function definitions indexed by ID
    functions: Vec<Option<FunDef>>,
    /// External function definitions indexed by ID
    extern_functions: Vec<ExternFnDef>,
    /// Next variable ID to assign
    next_var_id: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Default for Compiler<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Compiler<'a> {
    /// Create a new compiler
    pub fn new() -> Self {
        Compiler {
            functions: Vec::new(),
            extern_functions: Vec::new(),
            next_var_id: 0,
            _marker: PhantomData,
        }
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
    /// let result = call_extern2::<_, _, _, i64, i64, i64>(my_add, x, y);
    /// ```
    pub fn extern_fn<S: crate::ffi::ExternFn>(&mut self) -> crate::ffi::ExternRef<S> {
        let extern_id = self.extern_functions.len();

        self.extern_functions.push(ExternFnDef {
            // Disambiguate generic instantiations (same NAME, different fn ptr).
            name: format!("{}_{:x}", S::NAME, S::FN_PTR as usize),
            param_abi_types: S::param_abi_types(),
            return_abi_types: S::return_abi_types(),
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
    /// # Struct Handling
    ///
    /// For functions with struct parameters or returns:
    /// - Parameters: Multiple i64 values are received, stored to a stack slot,
    ///   and the variable holds the stack slot pointer
    /// - Returns: The result pointer is used to load multiple i64 values for return
    pub fn compile<S: Staged>(self, expr: S) -> Result<Compiled<'a, S::Out>, CompileError> {
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

        // Get the default calling convention for this platform
        // This ensures proper ABI compatibility, especially important for external function calls
        // Using SystemV on Unix-like systems, WindowsFastcall on Windows
        let call_conv = if cfg!(target_os = "windows") {
            cranelift_codegen::isa::CallConv::WindowsFastcall
        } else {
            cranelift_codegen::isa::CallConv::SystemV
        };

        // First pass: declare all internal functions
        let mut func_map: HashMap<usize, FuncId> = HashMap::new();

        for (id, func_opt) in self.functions.iter().enumerate() {
            if let Some(func_def) = func_opt {
                let mut sig = module.make_signature();
                sig.call_conv = call_conv;

                // Add all ABI param types (multiple per logical param for structs)
                for param_info in &func_def.param_infos {
                    for abi_type in &param_info.abi_types {
                        sig.params.push(AbiParam::new(*abi_type));
                    }
                }

                // Add all ABI return types (multiple for structs)
                for abi_type in &func_def.return_info.abi_types {
                    sig.returns.push(AbiParam::new(*abi_type));
                }

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

            // Add all ABI param types
            for param_types in &extern_def.param_abi_types {
                for abi_type in param_types {
                    sig.params.push(AbiParam::new(*abi_type));
                }
            }

            // Add return types
            for abi_type in &extern_def.return_abi_types {
                sig.returns.push(AbiParam::new(*abi_type));
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

        // For main function, use the expression's return type ABI types
        for abi_type in S::Out::abi_types() {
            main_sig.returns.push(AbiParam::new(abi_type));
        }

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

                for param_info in &func_def.param_infos {
                    for abi_type in &param_info.abi_types {
                        sig.params.push(AbiParam::new(*abi_type));
                    }
                }
                for abi_type in &func_def.return_info.abi_types {
                    sig.returns.push(AbiParam::new(*abi_type));
                }

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

                    // Create var_map for this function
                    let mut var_map: HashMap<usize, Variable> = HashMap::new();
                    // Optimized slice storage: var_id -> (ptr_var, len_var)
                    let mut slice_vars = HashMap::new();

                    // Handle all parameters
                    let block_params = builder.block_params(entry_block).to_vec();
                    let mut abi_idx = 0;

                    for (param_idx, param_info) in func_def.param_infos.iter().enumerate() {
                        let var_id = func_def.param_var_ids[param_idx];

                        if let Some(ref struct_info) = param_info.struct_info {
                            if param_info.pass_by_pointer {
                                // LARGE STRUCT (>16 bytes): Passed by pointer
                                // The caller already has the struct in memory and passes a pointer.
                                // We use that pointer directly without copying.
                                let ptr_value = block_params[abi_idx];
                                let param_var = builder.declare_var(types::I64);
                                builder.def_var(param_var, ptr_value);
                                var_map.insert(var_id, param_var);
                                abi_idx += 1;
                            } else if param_info.is_fat_pointer {
                                // FAT POINTER (slice): Store ptr and len in separate variables
                                // This avoids stack slot loads in tight loops
                                let ptr_value = block_params[abi_idx];
                                let len_value = block_params[abi_idx + 1];

                                // Create separate Cranelift variables for ptr and len
                                let ptr_var = builder.declare_var(types::I64);
                                let len_var = builder.declare_var(types::I64);
                                builder.def_var(ptr_var, ptr_value);
                                builder.def_var(len_var, len_value);

                                // Store in slice_vars for optimized access
                                slice_vars
                                    .insert(var_id, crate::staged::SliceVars { ptr_var, len_var });

                                // Store ptr_var in var_map - slice operations should use slice_vars
                                // for the optimized path. The fallback path (loading from stack)
                                // is not supported for parameter slices.
                                var_map.insert(var_id, ptr_var);

                                abi_idx += 2;
                            } else {
                                // SMALL STRUCT (≤16 bytes): Passed by value in registers
                                // Create stack slot, store values, use pointer
                                let align_shift = struct_info.alignment.trailing_zeros() as u8;
                                let stack_slot =
                                    builder.create_sized_stack_slot(StackSlotData::new(
                                        StackSlotKind::ExplicitSlot,
                                        struct_info.size,
                                        align_shift,
                                    ));

                                let slot_ptr = builder.ins().stack_addr(types::I64, stack_slot, 0);

                                // Store each i64 value to the stack slot
                                for i in 0..struct_info.num_abi_values {
                                    let offset = (i * 8) as i32;
                                    builder.ins().store(
                                        MemFlags::trusted(),
                                        block_params[abi_idx],
                                        slot_ptr,
                                        offset,
                                    );
                                    abi_idx += 1;
                                }

                                // Declare variable to hold the stack pointer
                                let param_var = builder.declare_var(types::I64);
                                builder.def_var(param_var, slot_ptr);
                                var_map.insert(var_id, param_var);
                            }
                        } else {
                            // PRIMITIVE PARAMETER: Direct binding
                            let param_value = block_params[abi_idx];
                            let param_type = param_info.abi_types[0];
                            let param_var = builder.declare_var(param_type);
                            builder.def_var(param_var, param_value);
                            var_map.insert(var_id, param_var);
                            abi_idx += 1;
                        }
                    }

                    // Generate the body code
                    let result = {
                        let mut extern_func_refs = HashMap::new();
                        let mut ctx = CompilationContext {
                            builder: &mut builder,
                            module: &mut module,
                            var_map: &mut var_map,
                            func_map: &func_map,
                            extern_func_refs: &mut extern_func_refs,
                            extern_func_ids: &extern_func_ids,
                            slice_vars: &mut slice_vars,
                            unit_value: None,
                            loop_exit_stack: Vec::new(),
                        };
                        (func_def.body)(&mut ctx)
                    };

                    // Handle return - either primitive or struct
                    if let Some(ref struct_info) = func_def.return_info.struct_info {
                        if func_def.return_info.pass_by_pointer {
                            // LARGE STRUCT RETURN (>16 bytes): Return pointer directly
                            builder.ins().return_(&[result]);
                        } else {
                            // SMALL STRUCT RETURN (≤16 bytes): Load multiple values from the result pointer
                            let mut return_values = Vec::with_capacity(struct_info.num_abi_values);
                            for i in 0..struct_info.num_abi_values {
                                let offset = (i * 8) as i32;
                                let val = builder.ins().load(
                                    types::I64,
                                    MemFlags::trusted(),
                                    result,
                                    offset,
                                );
                                return_values.push(val);
                            }
                            builder.ins().return_(&return_values);
                        }
                    } else {
                        // PRIMITIVE RETURN: Direct return
                        builder.ins().return_(&[result]);
                    }

                    builder.finalize();
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
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut var_map: HashMap<usize, Variable> = HashMap::new();

                let result = {
                    let mut extern_func_refs = HashMap::new();
                    let mut slice_vars = HashMap::new();
                    let mut ctx = CompilationContext {
                        builder: &mut builder,
                        module: &mut module,
                        var_map: &mut var_map,
                        func_map: &func_map,
                        extern_func_refs: &mut extern_func_refs,
                        extern_func_ids: &extern_func_ids,
                        slice_vars: &mut slice_vars,
                        unit_value: None,
                        loop_exit_stack: Vec::new(),
                    };
                    expr.codegen(&mut ctx)
                };

                // Handle return for main function
                if S::Out::is_copy_struct() {
                    // STRUCT RETURN: Load multiple values from the result pointer
                    let num_values = S::Out::num_abi_values();
                    let mut return_values = Vec::with_capacity(num_values);
                    for i in 0..num_values {
                        let offset = (i * 8) as i32;
                        let val =
                            builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), result, offset);
                        return_values.push(val);
                    }
                    builder.ins().return_(&return_values);
                } else {
                    // PRIMITIVE RETURN
                    builder.ins().return_(&[result]);
                }

                builder.finalize();
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
            module,
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

/// A compiled expression that owns its JIT module.
///
/// Use `.run()` to execute the compiled code and get the result.
pub struct Compiled<'a, T: StagedType> {
    #[allow(dead_code)]
    module: JITModule,
    main_ptr: *const u8,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: StagedType> Compiled<'a, T> {
    /// Execute the compiled code and return the result.
    ///
    /// # Safety
    /// This is safe as long as the compilation was done correctly.
    pub fn run(&self) -> T::RuntimeValue
    where
        T::RuntimeValue: Copy,
    {
        let func: fn() -> T::RuntimeValue = unsafe { std::mem::transmute(self.main_ptr) };
        func()
    }
}

// Macro to generate as_fn implementations for all function arities
macro_rules! impl_compiled_as_fn {
    // Base case: zero parameters
    (0, $FunType:ident) => {
        impl<'a, OUT: StagedType> Compiled<'a, $FunType<OUT>> {
            /// Get the compiled function as a callable function pointer.
            pub fn as_fn(&self) -> extern "C" fn() -> OUT::RuntimeValue {
                let get_ptr: fn() -> i64 = unsafe { std::mem::transmute(self.main_ptr) };
                let fn_ptr = get_ptr();
                unsafe { std::mem::transmute(fn_ptr) }
            }
        }
    };
    // N parameters (N >= 1)
    ($n:tt, $FunType:ident, [$($T:ident),+]) => {
        impl<'a, $($T: StagedType,)+ OUT: StagedType> Compiled<'a, $FunType<$($T,)+ OUT>> {
            /// Get the compiled function as a callable function pointer.
            ///
            /// Returns an `extern "C"` function pointer to match the System V calling
            /// convention used by Cranelift. This is important for structs passed by value,
            /// as Rust's default calling convention may differ from the C ABI.
            pub fn as_fn(&self) -> extern "C" fn($($T::RuntimeValue),+) -> OUT::RuntimeValue {
                let get_ptr: fn() -> i64 = unsafe { std::mem::transmute(self.main_ptr) };
                let fn_ptr = get_ptr();
                unsafe { std::mem::transmute(fn_ptr) }
            }
        }
    };
}

impl_compiled_as_fn!(0, FunType0);
impl_compiled_as_fn!(1, FunType1, [A]);
impl_compiled_as_fn!(2, FunType2, [A, B]);
impl_compiled_as_fn!(3, FunType3, [A, B, C]);
impl_compiled_as_fn!(4, FunType4, [A, B, C, D]);
impl_compiled_as_fn!(5, FunType5, [A, B, C, D, E]);
impl_compiled_as_fn!(6, FunType6, [A, B, C, D, E, F]);
impl_compiled_as_fn!(7, FunType7, [A, B, C, D, E, F, G]);
impl_compiled_as_fn!(8, FunType8, [A, B, C, D, E, F, G, H]);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_simple_constant() {
        let compiler = Compiler::new();
        let five = Const::<i64>::new(5);

        let compiled = compiler.compile(five).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 5);
    }

    #[test]
    fn test_simple_addition() {
        let compiler = Compiler::new();
        let expr = add::<i64, _, _>(3i64, 4i64);

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 7);
    }

    #[test]
    fn test_nested_arithmetic() {
        let compiler = Compiler::new();
        // (3 + 4) * 2 = 14
        let expr = mul(
            add(Const::<i64>::new(3), Const::<i64>::new(4)),
            Const::<i64>::new(2),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 14);
    }

    #[test]
    fn test_fun1_and_call() {
        let mut compiler = Compiler::new();

        // Define: square(x) = x * x
        let square = compiler.fun1("square", |_ctx, x: Var<i64>| mul(x, x));

        // Call: square(5) = 25
        let expr = call1(square, Const::<i64>::new(5));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 25);
    }

    #[test]
    fn test_var_before_fun1() {
        let mut compiler = Compiler::new();

        // Ensure that functions defined after internal var allocation don't get wrong param IDs.
        // Allocate a var inside a fun0 first to advance the counter.
        let _prime = compiler.fun0("prime_counter", |ctx| {
            let _x = ctx.var(0i64);
            Const::<i64>::new(0)
        });

        // Define: double(x) = x + x
        let double = compiler.fun1("double", |_ctx, x: Var<i64>| add(x, x));

        // Call: double(7) = 14
        let expr = call1(double, Const::<i64>::new(7));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 14);
    }

    #[test]
    fn test_return_function_pointer() {
        let mut compiler = Compiler::new();

        // Define: cube(x) = x * x * x
        let cube = compiler.fun1("cube", |_ctx, x: Var<i64>| mul(mul(x, x), x));

        // Compile the function reference itself (not a call)
        let compiled = compiler.compile(cube).expect("compilation failed");

        // Extract the function pointer
        let cube_fn = compiled.as_fn();

        // Test the function with various inputs
        assert_eq!(cube_fn(2), 8);
        assert_eq!(cube_fn(3), 27);
        assert_eq!(cube_fn(5), 125);
        assert_eq!(cube_fn(-2), -8);
    }

    #[test]
    fn test_recursive_function_compiles() {
        let mut compiler = Compiler::new();

        // Define a recursive function: rec(x) = x + rec(x - 1)
        // Note: This will infinite loop if called, but we're just testing
        // that it compiles and the function can reference itself
        let _rec = compiler.fun1_rec("recursive", |f, _ctx, x: Var<i64>| {
            // Body references itself: call f recursively
            add(x, call1(f, sub(x, Const::<i64>::new(1))))
        });

        // Just test that compilation succeeds
        // We don't call it since it would infinite loop without conditionals
        let expr = Const::<i64>::new(42);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    // =========================================================================
    // Control Flow Tests
    // =========================================================================

    #[test]
    fn test_if_then_else_basic() {
        // Test true branch
        let compiler = Compiler::new();
        let expr_true = if_then_else(true, Const::<i64>::new(10), Const::<i64>::new(20));
        assert_eq!(compiler.compile(expr_true).unwrap().run(), 10);

        // Test false branch
        let compiler = Compiler::new();
        let expr_false = if_then_else(false, Const::<i64>::new(10), Const::<i64>::new(20));
        assert_eq!(compiler.compile(expr_false).unwrap().run(), 20);
    }

    #[test]
    fn test_if_then_else_clamp() {
        let mut compiler = Compiler::new();

        // clamp(x) = if x < 0 then 0 else (if x > 10 then 10 else x)
        let clamp = compiler.fun1("clamp", |_ctx, x: Var<i64>| {
            if_then_else(
                lt(x, 0),
                Const::<i64>::new(0),
                if_then_else(lt(10, x), Const::<i64>::new(10), x),
            )
        });

        let compiled = compiler.compile(clamp).expect("compilation failed");
        let clamp_fn = compiled.as_fn();

        assert_eq!(clamp_fn(-5), 0); // Clamped at min
        assert_eq!(clamp_fn(5), 5); // In range
        assert_eq!(clamp_fn(15), 10); // Clamped at max
    }

    #[test]
    fn test_seq_basic() {
        let compiler = Compiler::new();

        // (5, 10) => 10 (first value ignored, second returned)
        let expr = (Const::<i64>::new(5), Const::<i64>::new(10));

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 10);
    }

    #[test]
    fn test_let_var() {
        let mut compiler = Compiler::new();

        let f = compiler.fun0("let_var_test", |ctx| {
            let x = ctx.var(42i64);
            let y = ctx.var(8i64);
            add::<i64, _, _>(x, y)
        });

        let compiled = compiler.compile(call0(f)).expect("compilation failed");
        assert_eq!(compiled.run(), 50);
    }

    #[test]
    fn test_ergonomic_assign() {
        let mut compiler = Compiler::new();

        let f = compiler.fun0("ergonomic_assign", |ctx| {
            let x = ctx.var(0i64);
            let y = ctx.var(0i64);
            ctx.store(x, 10i64);
            ctx.store(y, 32i64);
            add(x, y)
        });

        let compiled = compiler.compile(call0(f)).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_recursive_factorial() {
        let mut compiler = Compiler::new();

        // factorial(n) = if n <= 1 then 1 else n * factorial(n - 1)
        let factorial = compiler.fun1_rec("factorial", |f, _ctx, n: Var<i64>| {
            if_then_else(lt(n, 2), Const::new(1), mul(n, call1(f, sub(n, 1))))
        });

        let compiled = compiler.compile(factorial).expect("compilation failed");
        let factorial_fn = compiled.as_fn();

        assert_eq!(factorial_fn(0), 1);
        assert_eq!(factorial_fn(1), 1);
        assert_eq!(factorial_fn(5), 120);
        assert_eq!(factorial_fn(10), 3628800);
    }

    #[test]
    fn test_fibonacci() {
        let mut compiler = Compiler::new();

        // fib(n) = if n < 2 then n else fib(n-1) + fib(n-2)
        let fib = compiler.fun1_rec("fib", |f, _ctx, n: Var<i64>| {
            if_then_else(
                lt(n, Const::<i64>::new(2)),
                n, // fib(0) = 0, fib(1) = 1
                add(
                    call1(f, sub(n, Const::<i64>::new(1))),
                    call1(f, sub(n, Const::<i64>::new(2))),
                ),
            )
        });

        // Compile and get function pointer
        let compiled = compiler.compile(fib).expect("compilation failed");
        let fib_fn = compiled.as_fn();

        // Test Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
        assert_eq!(fib_fn(0), 0);
        assert_eq!(fib_fn(1), 1);
        assert_eq!(fib_fn(2), 1);
        assert_eq!(fib_fn(3), 2);
        assert_eq!(fib_fn(4), 3);
        assert_eq!(fib_fn(5), 5);
        assert_eq!(fib_fn(6), 8);
        assert_eq!(fib_fn(7), 13);
        assert_eq!(fib_fn(10), 55);
    }

    // =========================================================================
    // While Loop Tests
    // =========================================================================

    #[test]
    fn test_while_loop_zero_iterations() {
        let mut compiler = Compiler::new();

        let f = compiler.fun0("while_zero", |ctx| {
            let result = ctx.var(0i64);
            ctx.while_loop(false, move |ctx| {
                ctx.store(result, 999i64);
            });
            ctx.store(result, 42i64);
            result
        });

        let compiled = compiler.compile(call0(f)).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_while_loop_factorial() {
        let mut compiler = Compiler::new();

        let factorial_iter = compiler.fun1("factorial_iter", |ctx, n: Var<i64>| {
            let i = ctx.var(1i64);
            let result = ctx.var(1i64);
            ctx.while_loop(lt(i, add(n, 1i64)), move |ctx| {
                ctx.store(result, mul(result, i));
                ctx.store(i, add(i, 1i64));
            });
            result
        });

        let compiled = compiler
            .compile(factorial_iter)
            .expect("compilation failed");
        let factorial_fn = compiled.as_fn();

        assert_eq!(factorial_fn(0), 1);
        assert_eq!(factorial_fn(1), 1);
        assert_eq!(factorial_fn(2), 2);
        assert_eq!(factorial_fn(3), 6);
        assert_eq!(factorial_fn(5), 120);
        assert_eq!(factorial_fn(10), 3628800);
    }

    #[test]
    fn test_while_loop_fibonacci_iterative() {
        let mut compiler = Compiler::new();

        let fib_iter = compiler.fun1("fib_iter", |ctx, n: Var<i64>| {
            let i = ctx.var(2i64);
            let a = ctx.var(0i64);
            let b = ctx.var(1i64);
            let temp = ctx.var(0i64);
            if_then_else(lt(n, 2), n, {
                ctx.while_loop(lt(i, add(n, 1i64)), move |ctx| {
                    ctx.store(temp, add(a, b));
                    ctx.store(a, b);
                    ctx.store(b, temp);
                    ctx.store(i, add(i, 1i64));
                });
                b
            })
        });

        let compiled = compiler.compile(fib_iter).expect("compilation failed");
        let fib_fn = compiled.as_fn();

        assert_eq!(fib_fn(0), 0);
        assert_eq!(fib_fn(1), 1);
        assert_eq!(fib_fn(2), 1);
        assert_eq!(fib_fn(3), 2);
        assert_eq!(fib_fn(4), 3);
        assert_eq!(fib_fn(5), 5);
        assert_eq!(fib_fn(10), 55);
        assert_eq!(fib_fn(20), 6765);
        assert_eq!(fib_fn(30), 832040);
    }

    #[test]
    fn test_local_variables_in_fun1() {
        use crate::refer::SRef;
        use crate::slice::Slice;

        let mut compiler = Compiler::new();

        // Function that sums elements > 5 using local variables
        // fn sum_gt_5(arr: &[i64]) -> i64
        let sum_gt_5 = compiler.fun1("sum_gt_5", |ctx, arr: Var<SRef<Slice<i64>>>| {
            // Create local variables inside the function using ctx
            let i = ctx.let_var(0u64);
            let sum = ctx.let_var(0i64);
            let v = ctx.let_var(0i64);

            (
                (i, sum, v),
                while_loop(
                    lt(*i, arr.len()),
                    (
                        // v = arr.get_unchecked(i)
                        assign(*v, arr.get_unchecked(*i)),
                        // sum = if v > 5 then sum + v else sum
                        assign(
                            *sum,
                            if_then_else(
                                lt(5, *v), // v > 5
                                add(*sum, *v),
                                *sum,
                            ),
                        ),
                        assign(*i, add(*i, 1u64)),
                    ),
                ),
                *sum,
            )
        });

        let compiled = compiler.compile(sum_gt_5).expect("compilation failed");
        let f = compiled.as_fn();

        // Test with array [0, 3, 5, 7, 2, 8, 1, 9, 4, 6]
        // Elements > 5: 7, 8, 9, 6 => sum = 30
        let data: [i64; 10] = [0, 3, 5, 7, 2, 8, 1, 9, 4, 6];
        let slice: &[i64] = &data;

        let result = f(slice);
        assert_eq!(result, 30); // 7 + 8 + 9 + 6 = 30
    }

    // =========================================================================
    // Multi-Parameter Function Tests
    // =========================================================================

    #[test]
    fn test_fun0_constant() {
        use crate::func_impl::call0;

        let mut compiler = Compiler::new();

        // Define: get_answer() = 42
        let get_answer = compiler.fun0("get_answer", |_ctx| Const::<i64>::new(42));

        let expr = call0(get_answer);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_fun2_add() {
        let mut compiler = Compiler::new();

        let add_fn = compiler.fun2("add", |_ctx, a: Var<i64>, b: Var<i64>| add(a, b));

        let compiled = compiler.compile(add_fn).expect("compilation failed");
        let add_ptr = compiled.as_fn();

        assert_eq!(add_ptr(10, 32), 42);
        assert_eq!(add_ptr(-5, 5), 0);
    }

    #[test]
    fn test_fun3_clamp() {
        let mut compiler = Compiler::new();

        // clamp(x, min, max) = if x < min then min else (if x > max then max else x)
        let clamp_fn = compiler.fun3(
            "clamp",
            |_ctx, x: Var<i64>, min: Var<i64>, max: Var<i64>| {
                if_then_else(lt(x, min), min, if_then_else(lt(max, x), max, x))
            },
        );

        let compiled = compiler.compile(clamp_fn).expect("compilation failed");
        let clamp = compiled.as_fn();

        assert_eq!(clamp(-5, 0, 10), 0); // Clamped at min
        assert_eq!(clamp(5, 0, 10), 5); // In range
        assert_eq!(clamp(15, 0, 10), 10); // Clamped at max
    }

    #[test]
    fn test_fun2_rec_gcd() {
        use crate::func_impl::call2;

        let mut compiler = Compiler::new();

        // Define: gcd(a, b) = if b == 0 then a else gcd(b, a % b)
        // Note: We'll use a different implementation since we don't have modulo
        // gcd(a, b) = if b == 0 then a else gcd(b, a - b * (a / b))
        let gcd = compiler.fun2_rec("gcd", |f, _ctx, a: Var<i64>, b: Var<i64>| {
            if_then_else(
                eq(b, 0i64),
                a,
                call2(f, b, sub(a, mul(b, div(a, b)))), // a % b = a - b * (a / b)
            )
        });

        // gcd(48, 18) = 6
        let expr = call2(gcd, 48i64, 18i64);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 6);
    }
}
