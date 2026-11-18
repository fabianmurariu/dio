use crate::{
    label::{Again, Label},
    prelude::{
        CompilationContext, LifetimeErased, Shape, Staged, StagedType, Value, ValueId, Var, assign,
    },
    staged::LoadKind,
};
use std::{
    cell::{RefCell, RefMut},
    rc::Rc,
};

/// Imperative context for building staged function bodies.
///
/// Passed to closures in `fun1`, `fun2`, etc. Call methods to emit code in
/// declaration order.
///
/// A `Ctx` is a cheap handle (`Rc`) to one function body's builder. Clone it
/// freely, including into closures: whatever the clone emits lands in the
/// innermost scope (loop body, branch, block) open at the time of the call.
/// The builder is only borrowed while a statement is appended, so nested and
/// captured uses never conflict; a `Ctx` used after its function body was
/// finished panics.
///
/// # Example
/// ```ignore
/// compiler.fun1("sum", |ctx, arr: Var<SRef<Slice<f64>>>| {
///     let acc = ctx.var(0.0f64);
///     arr.staged_iter().for_each(ctx, |ctx, elem| {
///         ctx.store(acc, add(acc, elem));
///     });
///     acc
/// });
/// ```
#[derive(Clone)]
pub struct Ctx(Rc<RefCell<CtxInner>>);

pub(crate) type CodegenAction = Box<dyn FnOnce(&mut CompilationContext) + 'static>;

// =============================================================================
// Ctx: Imperative context for building staged function bodies
// =============================================================================

/// The state behind a [`Ctx`].
struct CtxInner {
    next_var_id: usize,
    /// One action list per open scope; statements are appended to the last.
    frames: Vec<Vec<CodegenAction>>,
}

impl Ctx {
    pub(crate) fn new(start: usize) -> Self {
        Ctx(Rc::new(RefCell::new(CtxInner {
            next_var_id: start,
            frames: vec![Vec::new()],
        })))
    }

    fn inner(&self) -> RefMut<'_, CtxInner> {
        self.0
            .try_borrow_mut()
            .expect("rust-lms bug: Ctx re-entered while appending a statement")
    }

    pub(crate) fn final_id(&self) -> usize {
        self.inner().next_var_id
    }

    /// Append a statement to the innermost open scope.
    fn push(&self, action: CodegenAction) {
        self.inner()
            .frames
            .last_mut()
            .expect("Ctx used after its function body was finished")
            .push(action);
    }

    fn next_id(&self) -> usize {
        let mut inner = self.inner();
        let id = inner.next_var_id;
        inner.next_var_id += 1;
        id
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
        let actions = {
            let mut inner = self.inner();
            assert_eq!(inner.frames.len(), 1, "rust-lms bug: a scope was left open");
            // Leave no frame behind, so a leaked clone panics instead of emitting.
            inner.frames.pop().expect("the body's own frame")
        };
        Box::new(move |ctx| {
            for action in actions {
                action(ctx);
            }
            ret.codegen(ctx)
        })
    }

    fn alloc<T: StagedType>(&self) -> Var<T> {
        Var::new(self.next_id())
    }

    /// Allocate a variable ID without registering an initialization action.
    ///
    /// # Safety
    /// Caller must ensure the variable is assigned before it is used in codegen.
    pub(crate) unsafe fn var_unchecked<T: StagedType>(&self) -> Var<T> {
        Var::new(self.next_id())
    }

    /// Declare a new variable initialized to `init` at this point in the body.
    ///
    /// Returns a `Var<T>` that can be used in expressions and passed to `assign`.
    pub fn var<T, E>(&self, init: E) -> Var<T>
    where
        T: StagedType + 'static,
        E: crate::staged::IntoStaged<T>,
        E::Staged: 'static,
    {
        let v = self.alloc::<T>();
        let init_staged = init.into_staged();
        let id = v.id;
        self.push(Box::new(move |ctx| {
            let value = init_staged.codegen(ctx);
            ctx.assign_var::<T>(id, value, false);
        }));
        v
    }

    /// Evaluate a complex staged expression once, binding the result to a new
    /// variable. Avoids recomputing the expression if used in multiple places.
    pub fn bind<T, E>(&self, expr: E) -> Var<T>
    where
        T: StagedType + 'static,
        E: Staged<Out = T> + 'static,
    {
        let v = self.alloc::<T>();
        let id = v.id;
        self.push(Box::new(move |ctx| {
            let value = expr.codegen(ctx);
            ctx.assign_var::<T>(id, value, true);
        }));
        v
    }

    /// Bind an expression that **carries a Rust borrow**.
    ///
    /// [`bind`](Self::bind) cannot be used for these. Its expression is stored
    /// in the `'static` action queue, so it requires `E: 'static`, which a
    /// borrowing expression is not. Erasing the lifetime to get it into the
    /// queue would also erase it from the result, and the borrow would stop
    /// protecting anything.
    ///
    /// So the lifetime is split across two types: the *erased* expression goes
    /// into the queue, while the returned `Var` is typed with the **un-erased**
    /// `Out`. The two have identical layout and differ only in phantom
    /// lifetimes, so the generated code is the same — but the `Var` keeps its
    /// source borrowed, and the borrow checker rejects a mutation of that
    /// source while the `Var` is live.
    ///
    /// That is what makes a captured `(ptr, len)` safe: growing an `SVec` may
    /// move its buffer, so a snapshot of its descriptor must not outlive a
    /// `push`. Binding the length of a slice yields `Var<u64>`, which carries
    /// no borrow and therefore does not restrict later growth; binding the
    /// slice itself yields `Var<BorrowedSlice<'a, _>>`, which does.
    pub fn bind_lt<E>(&self, expr: E) -> Var<E::Out>
    where
        E: LifetimeErased,
    {
        let v = self.alloc::<E::Out>();
        let id = v.id;
        // Layout comes from the erased twin — identical to `E::Out`'s, and read
        // here so `E::Out` never enters the `'static` closure.
        let (shape, name) = (
            Shape::of::<E::ErasedOut>(),
            std::any::type_name::<E::ErasedOut>(),
        );
        let expr = expr.erase_lifetime();
        self.push(Box::new(move |ctx| {
            let value = expr.codegen(ctx);
            ctx.assign_var_layout(id, value, shape, name, true);
        }));
        v
    }

    /// Emit an assignment: `var = expr`.
    ///
    /// Accepts any value that implements `IntoStaged<T>` — primitives like
    /// `42i64` work directly, as do staged expressions.
    pub fn store<T, E>(&self, var: Var<T>, expr: E)
    where
        T: StagedType + 'static,
        E: crate::staged::IntoStaged<T>,
        E::Staged: 'static,
    {
        let staged_expr = expr.into_staged();
        self.push(Box::new(move |ctx| {
            assign(var, staged_expr).codegen(ctx);
        }));
    }

    /// [`store`](Self::store) for an expression that borrows. `T` is the
    /// variable's own type, which carries no borrow, so nothing survives the
    /// assignment holding one.
    pub fn store_lt<T, E>(&self, var: Var<T>, expr: E)
    where
        T: StagedType + 'static,
        E: LifetimeErased<Out = T, ErasedOut = T>,
    {
        let expr = expr.erase_lifetime();
        self.push(Box::new(move |ctx| {
            assign(var, expr).codegen(ctx);
        }));
    }

    /// [`emit`](Self::emit) for a statement that borrows — a write through a
    /// borrowed slice, say. Same split as [`bind_lt`](Self::bind_lt): the
    /// statement is erased to enter the `'static` queue, and because its `Out` is
    /// `()` there is nothing left holding the borrow afterwards.
    pub fn emit_lt<S>(&self, stmt: S)
    where
        S: LifetimeErased<Out = (), ErasedOut = ()>,
    {
        let stmt = stmt.erase_lifetime();
        self.push(Box::new(move |ctx| {
            stmt.codegen(ctx);
        }));
    }

    /// Emit any unit-typed staged expression (e.g. a store, an extern call).
    pub fn emit<S: Staged<Out = ()> + 'static>(&self, stmt: S) {
        self.push(Box::new(move |ctx| {
            stmt.codegen(ctx);
        }));
    }

    /// Crate-internal: emit backend code directly, for sources whose step has
    /// no staged-expression form (an indirect call through a slot's vtable).
    pub(crate) fn emit_raw(&self, f: impl FnOnce(&mut CompilationContext) + 'static) {
        self.push(Box::new(f));
    }

    /// Crate-internal: bind the value produced by raw backend code.
    pub(crate) fn bind_raw<T: StagedType + 'static>(
        &self,
        f: impl FnOnce(&mut CompilationContext) -> Value + 'static,
    ) -> Var<T> {
        let v = self.alloc::<T>();
        let id = v.id;
        self.push(Box::new(move |ctx| {
            let value = f(ctx);
            ctx.assign_var::<T>(id, value, true);
        }));
        v
    }

    /// Emit a while loop: `while cond { body }`.
    ///
    /// `body` is called once at staging time; the closure emits the per-iteration
    /// side effects into a child `Ctx`. `cond` accepts `IntoStaged<bool>`
    /// so `false` and `true` work directly.
    pub fn while_loop<C, F>(&self, cond: C, body: F)
    where
        C: crate::staged::IntoStaged<bool>,
        C::Staged: 'static,
        F: FnOnce(&Ctx),
    {
        let cond = cond.into_staged();
        let (body_actions, ()) = self.child(body);
        self.push(Box::new(move |ctx| {
            emit_loop(ctx, Some(cond), None, body_actions);
        }));
    }

    /// Stage `body` into a new innermost scope and return its actions.
    fn child<R>(&self, body: impl FnOnce(&Ctx) -> R) -> (Vec<CodegenAction>, R) {
        let depth = {
            let mut inner = self.inner();
            inner.frames.push(Vec::new());
            inner.frames.len()
        };
        let result = body(self);
        let mut inner = self.inner();
        assert_eq!(
            inner.frames.len(),
            depth,
            "rust-lms bug: a scope was left open"
        );
        let actions = inner.frames.pop().expect("the scope opened above");
        (actions, result)
    }

    /// A fresh label id. Labels share the variable id counter, which is unique
    /// per function body, so ids never collide across nested scopes.
    fn label_id(&self) -> usize {
        self.next_id()
    }

    // =========================================================================
    // Scoped labels (see `crate::label`)
    // =========================================================================

    /// The iteration loop: `'done: loop { body }`.
    ///
    /// `body` receives the loop's exit as `done`; [`exit`](Self::exit) to it (or
    /// [`break_loop`](Self::break_loop), which targets the same block) to leave.
    /// Falling off the end of `body` runs it again.
    ///
    /// ```ignore
    /// let i = ctx.var(0u64);
    /// ctx.iterate(|ctx, done| {
    ///     ctx.exit_if(ge(i, n), done);
    ///     ctx.store(i, i + 1u64);
    /// });
    /// ```
    pub fn iterate<F>(&self, body: F)
    where
        F: for<'s> FnOnce(&Ctx, Label<'s>),
    {
        let id = self.label_id();
        let (body_actions, ()) = self.child(|ctx| body(ctx, Label::new(id)));
        self.push(Box::new(move |ctx| {
            emit_loop::<crate::staged::Const<bool>>(ctx, None, Some(id), body_actions);
        }));
    }

    /// Stages `body` with every load it emits marked [`LoadKind::Invariant`]: removable
    /// when unused, merged when repeated, and movable, out of loops and above branches.
    /// For reads of descriptors a kernel resolves (a table row at a clamped index, a
    /// field of one), so the ones nothing uses cost nothing and the ones a loop repeats
    /// are hoisted out of it.
    ///
    /// # Safety
    ///
    /// Every load `body` stages, through any pointer and in any nested block, reads
    /// memory that nothing writes while the function runs, and that is readable
    /// wherever the load's operands are available: not only where `body` puts it, but
    /// also before any branch or loop test that guards it. Element reads behind a bounds
    /// check, and anything that reads the stack, stay outside. Calls may be staged
    /// inside: their results are read back with plain loads.
    pub unsafe fn invariant<T>(&self, body: impl FnOnce(&Ctx) -> T) -> T {
        let (actions, out) = self.child(body);
        self.push(Box::new(move |ctx| {
            let outer = std::mem::replace(&mut ctx.load_kind, LoadKind::Invariant);
            for action in actions {
                action(ctx);
            }
            ctx.load_kind = outer;
        }));
        out
    }

    /// A forward label with no value: `'out: { body }`.
    ///
    /// [`exit`](Self::exit)`(out)` anywhere inside `body` — however deeply
    /// nested — continues after the block. Not a loop scope: `break_loop`
    /// passes through it.
    pub fn block<F>(&self, body: F)
    where
        F: for<'s> FnOnce(&Ctx, Label<'s>),
    {
        let id = self.label_id();
        let (body_actions, ()) = self.child(|ctx| body(ctx, Label::new(id)));
        self.push(Box::new(move |ctx| {
            let out = ctx.create_block();
            ctx.labels.insert(id, out);
            for action in body_actions {
                action(ctx);
            }
            ctx.labels.remove(&id);
            ctx.jump(out, &[]);
            ctx.switch_to_block(out);
            // Every predecessor — each `exit` and the fall-through — is emitted.
            ctx.seal_block(out);
        }));
    }

    /// A forward label carrying a value: `'out: { body }` where `body` either
    /// [`goto`](Self::goto)s `out` with a `T` or falls through with its result.
    /// Returns the merged value (the merge block's parameters — a phi).
    ///
    /// ```ignore
    /// let sign = ctx.join(|ctx, out| {
    ///     ctx.if_then(lt(x, 0i64), |ctx| ctx.goto(out, -1i64));
    ///     ctx.if_then(gt(x, 0i64), |ctx| ctx.goto(out, 1i64));
    ///     0i64
    /// });
    /// ```
    pub fn join<T, E, F>(&self, body: F) -> Var<T>
    where
        T: StagedType + 'static,
        E: crate::staged::IntoStaged<T>,
        E::Staged: 'static,
        F: for<'s> FnOnce(&Ctx, Label<'s, T>) -> E,
    {
        let id = self.label_id();
        let (body_actions, fall_through) = self.child(|ctx| body(ctx, Label::new(id)));
        let fall_through = fall_through.into_staged();
        let result = self.alloc::<T>();
        let result_id = result.id;
        self.push(Box::new(move |ctx| {
            let out = ctx.create_block();
            ctx.append_value_block_params::<T>(out);
            ctx.labels.insert(id, out);
            for action in body_actions {
                action(ctx);
            }
            let value = fall_through.codegen(ctx);
            ctx.labels.remove(&id);
            ctx.jump_value(out, value);
            ctx.switch_to_block(out);
            ctx.seal_block(out);
            let merged = ctx.block_value::<T>(out);
            ctx.assign_var::<T>(result_id, merged, true);
        }));
        result
    }

    /// A retry point: `'again: loop { body; break }`.
    ///
    /// [`again`](Self::again) re-runs `body` from the top; falling off the end
    /// leaves. Not a loop scope — `break_loop` inside passes through to the
    /// enclosing `iterate`/`while_loop` — which is what separates it from
    /// `while_loop`. Returns `body`'s stage-0 result.
    ///
    /// ```ignore
    /// let x = ctx.repeat(|ctx, again| {
    ///     let x = ctx.bind(pull(h));
    ///     ctx.again_if(is_tombstone(x), again);
    ///     x
    /// });
    /// ```
    pub fn repeat<R, F>(&self, body: F) -> R
    where
        F: for<'s> FnOnce(&Ctx, Again<'s>) -> R,
    {
        let id = self.label_id();
        let (body_actions, result) = self.child(|ctx| body(ctx, Again::new(id)));
        self.push(Box::new(move |ctx| {
            let head = ctx.create_block();
            ctx.jump(head, &[]);
            ctx.switch_to_block(head);
            ctx.labels.insert(id, head);
            for action in body_actions {
                action(ctx);
            }
            ctx.labels.remove(&id);
            // Every back-edge (`again`) is inside the body, so all are emitted.
            ctx.seal_block(head);
        }));
        result
    }

    /// Jump to `label` carrying `value` (`break 'label value`). Code staged
    /// after it in the same block is unreachable.
    pub fn goto<T, E>(&self, label: Label<'_, T>, value: E)
    where
        T: StagedType + 'static,
        E: crate::staged::IntoStaged<T>,
        E::Staged: 'static,
    {
        let id = label.id;
        let value = value.into_staged();
        self.push(Box::new(move |ctx| {
            let value = value.codegen(ctx);
            let target = label_block(ctx, id);
            ctx.jump_value(target, value);
            switch_to_dead_block(ctx);
        }));
    }

    /// Jump to a value-less `label` (`break 'label`).
    pub fn exit(&self, label: Label<'_>) {
        let id = label.id;
        self.push(Box::new(move |ctx| {
            let target = label_block(ctx, id);
            ctx.jump(target, &[]);
            switch_to_dead_block(ctx);
        }));
    }

    /// `if cond { break 'label }` as a single conditional branch.
    pub fn exit_if<C>(&self, cond: C, label: Label<'_>)
    where
        C: Staged<Out = bool> + 'static,
    {
        self.branch_to(cond, label.id, true);
    }

    /// `if !cond { break 'label }` — branches on `cond` itself, no negation.
    pub fn exit_unless<C>(&self, cond: C, label: Label<'_>)
    where
        C: Staged<Out = bool> + 'static,
    {
        self.branch_to(cond, label.id, false);
    }

    /// Jump to label `id` when `cond == when`, else continue.
    fn branch_to<C>(&self, cond: C, id: usize, when: bool)
    where
        C: Staged<Out = bool> + 'static,
    {
        self.push(Box::new(move |ctx| {
            let cond = cond.codegen(ctx);
            let target = label_block(ctx, id);
            branch_or_continue(ctx, cond.leaf(), target, when);
        }));
    }

    /// Re-run the enclosing [`repeat`](Self::repeat)'s body (`continue 'again`).
    pub fn again(&self, again: Again<'_>) {
        let id = again.id;
        self.push(Box::new(move |ctx| {
            let head = label_block(ctx, id);
            ctx.jump(head, &[]);
            switch_to_dead_block(ctx);
        }));
    }

    /// `if cond { continue 'again }` as a single conditional branch.
    pub fn again_if<C>(&self, cond: C, again: Again<'_>)
    where
        C: Staged<Out = bool> + 'static,
    {
        self.branch_to(cond, again.id, true);
    }

    /// `if !cond { continue 'again }` — branches on `cond` itself, no negation.
    pub fn again_unless<C>(&self, cond: C, again: Again<'_>)
    where
        C: Staged<Out = bool> + 'static,
    {
        self.branch_to(cond, again.id, false);
    }

    /// Break out of the innermost enclosing loop — an [`iterate`](Self::iterate)
    /// or [`while_loop`](Self::while_loop). [`block`](Self::block),
    /// [`join`](Self::join) and [`repeat`](Self::repeat) are not loops and are
    /// passed through.
    ///
    /// Emits a jump to the loop's exit block. Typically used inside an
    /// `if_then` to exit early. Panics at codegen time if called outside a loop.
    pub fn break_loop(&self) {
        self.push(Box::new(move |ctx| {
            let exit = *ctx
                .loop_exit_stack
                .last()
                .expect("break_loop called outside of a loop");
            ctx.jump(exit, &[]);
            switch_to_dead_block(ctx);
        }));
    }

    /// Emit a one-sided conditional: `if cond { then }`.
    pub fn if_then<C, F>(&self, cond: C, then: F)
    where
        C: Staged<Out = bool> + 'static,
        F: FnOnce(&Ctx),
    {
        let (then_actions, ()) = self.child(then);

        self.push(Box::new(move |ctx| {
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

    /// Run `arm(j)` for `index == j`, `j` in `0..n`: one indirect jump through a table,
    /// with each arm staged once. `index` must be below `n`.
    pub fn switch<F>(&self, index: Var<u32>, n: u32, mut arm: F)
    where
        F: FnMut(&Ctx, u32),
    {
        let arms: Vec<Vec<CodegenAction>> =
            (0..n).map(|j| self.child(|ctx| arm(ctx, j)).0).collect();

        self.push(Box::new(move |ctx| {
            let blocks: Vec<_> = (0..n).map(|_| ctx.create_block()).collect();
            let merge_block = ctx.create_block();

            let index = index.codegen(ctx);
            ctx.br_table(index.leaf(), merge_block, &blocks);

            for (block, actions) in blocks.into_iter().zip(arms) {
                ctx.switch_to_block(block);
                ctx.seal_block(block);
                for action in actions {
                    action(ctx);
                }
                ctx.jump(merge_block, &[]);
            }

            ctx.switch_to_block(merge_block);
            ctx.seal_block(merge_block);
        }));
    }

    /// Emit a two-sided conditional: `if cond { then } else { els }`.
    ///
    /// Both branches are side-effecting (they emit into the `Ctx` they receive)
    /// and the construct yields no value — sequence with `ctx.var`/`ctx.store`
    /// if you need a result out.
    pub fn if_then_else<C, T, E>(&self, cond: C, then: T, els: E)
    where
        C: Staged<Out = bool> + 'static,
        T: FnOnce(&Ctx),
        E: FnOnce(&Ctx),
    {
        let (then_actions, ()) = self.child(then);
        let (else_actions, ()) = self.child(els);

        self.push(Box::new(move |ctx| {
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

// =============================================================================
// Shared lowering for loops and labels
// =============================================================================

/// The one loop lowering, shared by `while_loop` (a header test) and `iterate`
/// (no test; its exit doubles as the `done` label).
///
/// ```text
/// jump header
/// header: [brif cond, body, exit]   ; while_loop only
/// body:   <actions> ; jump header    ; break_loop / exit(done) → exit
/// exit:
/// ```
fn emit_loop<C: Staged<Out = bool>>(
    ctx: &mut CompilationContext,
    cond: Option<C>,
    label: Option<usize>,
    body_actions: Vec<CodegenAction>,
) {
    let header = ctx.create_block();
    let exit = ctx.create_block();
    ctx.jump(header, &[]);
    ctx.switch_to_block(header);
    if let Some(cond) = cond {
        let body = ctx.create_block();
        let cond_val = cond.codegen(ctx);
        ctx.brif(cond_val.leaf(), body, &[], exit, &[]);
        ctx.switch_to_block(body);
        ctx.seal_block(body);
    }
    // Expose the exit to `break_loop` (and, for `iterate`, to `exit(done)`)
    // while the body is emitted.
    ctx.loop_exit_stack.push(exit);
    if let Some(id) = label {
        ctx.labels.insert(id, exit);
    }
    for action in body_actions {
        action(ctx);
    }
    if let Some(id) = label {
        ctx.labels.remove(&id);
    }
    ctx.loop_exit_stack.pop();
    ctx.jump(header, &[]);
    // All back-edges and exits are emitted now.
    ctx.seal_block(header);
    ctx.switch_to_block(exit);
    ctx.seal_block(exit);
}

/// The block a label id names. Its scope is being emitted — the stage-0 brand
/// guarantees that — so a miss is a library bug, not a user error.
fn label_block(ctx: &CompilationContext, id: usize) -> crate::staged::BlockHandle {
    *ctx.labels
        .get(&id)
        .expect("label used outside its scope (the lifetime brand should prevent this)")
}

/// After an unconditional jump the current block is terminated; continue in a
/// fresh, unreachable block so any code staged afterwards stays well-formed.
fn switch_to_dead_block(ctx: &mut CompilationContext) {
    let dead = ctx.create_block();
    ctx.switch_to_block(dead);
    ctx.seal_block(dead);
}

/// Branch to `target` when `cond == when`, else continue in a fresh block —
/// one `brif`, no merge block.
fn branch_or_continue(
    ctx: &mut CompilationContext,
    cond: ValueId,
    target: crate::staged::BlockHandle,
    when: bool,
) {
    let cont = ctx.create_block();
    if when {
        ctx.brif(cond, target, &[], cont, &[]);
    } else {
        ctx.brif(cond, cont, &[], target, &[]);
    }
    ctx.switch_to_block(cont);
    ctx.seal_block(cont);
}
