//! Staging-time optionals for control flow.
//!
//! A [`StagedOpt`] is **not** a [`StagedType`] and never materializes: there is
//! no discriminant in memory, no stack slot. You *consume* it with
//! [`StagedOpt::eliminate`], handing it a "Some" continuation and a "None"
//! continuation; it emits a branch (or, for the static `s_some`/`s_none`
//! constructors, no branch at all) and the payload only ever lives in a
//! register inside the Some arm.
//!
//! This is the right representation for fused iterator combinators
//! (`filter_map`/`find_map`): the `Some`/`None` becomes control flow rather
//! than a value. For an option that must be *stored*, returned across a
//! function boundary, or passed via FFI, use [`crate::option::COption`]
//! instead — you can bridge between them:
//!
//! - materialize: `opt.eliminate(ctx, c_some-ish, c_none-ish)`
//! - destructure a `COption`: `match_opt(...)`

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::staged::{IntoStaged, Staged, Var};
use crate::types::{BoolType, StagedType};

/// A staging-time optional consumed via [`eliminate`](StagedOpt::eliminate).
pub trait StagedOpt {
    /// The payload type produced on the `Some` path.
    type Item: StagedType;

    /// Emit code that runs `on_some(ctx, value)` when this is `Some`, or
    /// `on_none(ctx)` when it is `None`. The payload is bound (in a register)
    /// only inside the `Some` arm.
    fn eliminate<S, N>(self, ctx: &mut Ctx, on_some: S, on_none: N)
    where
        Self: Sized,
        S: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
        N: FnOnce(&mut Ctx) + 'static;
}

// =============================================================================
// When: runtime-conditional Some/None — the workhorse (`cond.then_some(value)`)
// =============================================================================

/// `if cond { Some(value) } else { None }`, evaluated lazily: `value` is only
/// codegen'd on the `Some` path. Built by [`ThenSome::then_some`].
pub struct When<C, V, T> {
    cond: C,
    value: V,
    _phantom: PhantomData<T>,
}

impl<C, V, T> StagedOpt for When<C, V, T>
where
    C: Staged<Out = BoolType> + 'static,
    V: Staged<Out = T> + 'static,
    T: StagedType + 'static,
{
    type Item = T;

    fn eliminate<S, N>(self, ctx: &mut Ctx, on_some: S, on_none: N)
    where
        S: FnOnce(&mut Ctx, Var<T>) + 'static,
        N: FnOnce(&mut Ctx) + 'static,
    {
        let value = self.value;
        ctx.if_then_else(
            self.cond,
            move |ctx| {
                // `value` is codegen'd only here, in the taken branch.
                let v = ctx.bind(value);
                on_some(ctx, v);
            },
            move |ctx| on_none(ctx),
        );
    }
}

/// Extension giving every boolean staged expression a `then_some`.
pub trait ThenSome: Staged<Out = BoolType> + Sized {
    /// `self.then_some(value)` == `if self { Some(value) } else { None }` as a
    /// [`StagedOpt`]. `value` is lazy — only evaluated when `self` is true.
    fn then_some<T, E>(self, value: E) -> When<Self, E::Staged, T>
    where
        T: StagedType,
        E: IntoStaged<T>,
    {
        When {
            cond: self,
            value: value.into_staged(),
            _phantom: PhantomData,
        }
    }
}

impl<S: Staged<Out = BoolType>> ThenSome for S {}

// =============================================================================
// SSome / SNone: static (unconditional) constructors — no branch
// =============================================================================

/// Always-`Some(value)`. `eliminate` emits no branch.
pub struct SSome<V, T> {
    value: V,
    _phantom: PhantomData<T>,
}

impl<V, T> StagedOpt for SSome<V, T>
where
    V: Staged<Out = T> + 'static,
    T: StagedType + 'static,
{
    type Item = T;

    fn eliminate<S, N>(self, ctx: &mut Ctx, on_some: S, _on_none: N)
    where
        S: FnOnce(&mut Ctx, Var<T>) + 'static,
        N: FnOnce(&mut Ctx) + 'static,
    {
        let v = ctx.bind(self.value);
        on_some(ctx, v);
    }
}

/// Build an always-`Some` staged optional.
pub fn s_some<T, E>(value: E) -> SSome<E::Staged, T>
where
    T: StagedType,
    E: IntoStaged<T>,
{
    SSome {
        value: value.into_staged(),
        _phantom: PhantomData,
    }
}

/// Always-`None`. `eliminate` emits no branch (just the None continuation).
pub struct SNone<T> {
    _phantom: PhantomData<T>,
}

impl<T> StagedOpt for SNone<T>
where
    T: StagedType + 'static,
{
    type Item = T;

    fn eliminate<S, N>(self, ctx: &mut Ctx, _on_some: S, on_none: N)
    where
        S: FnOnce(&mut Ctx, Var<T>) + 'static,
        N: FnOnce(&mut Ctx) + 'static,
    {
        on_none(ctx);
    }
}

/// Build an always-`None` staged optional.
pub fn s_none<T: StagedType>() -> SNone<T> {
    SNone {
        _phantom: PhantomData,
    }
}
