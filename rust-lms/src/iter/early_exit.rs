//! Early-exit terminal operations for indexed sources.
//!
//! These drive their own indexed loop (rather than the push-based `for_each`)
//! so they can `break_loop` as soon as the answer is known. They are available
//! on anything implementing [`IndexedSource`] — slice iterators, bare slice
//! variables, and `u64` ranges.
//!
//! Note: the non-early-exit terminals (`sum`, `fold`, `min`, …) deliberately
//! stay on the push-based path so they keep emitting the optimal loop body with
//! no per-iteration flag check.

use crate::control::not;
use crate::func::Ctx;
use crate::num::{add, lt};
use crate::staged::{Staged, Var};
use crate::types::{BoolType, U64Type};

use super::traits::IndexedSource;

/// Early-exit terminals for indexed (random-access) sources.
pub trait IndexedEarlyExit: IndexedSource {
    /// Return `true` as soon as any element satisfies `pred` (short-circuits).
    fn any<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<bool>
    where
        Self: Sized,
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = BoolType> + 'static,
    {
        let i = ctx.var(0u64);
        let found = ctx.var(false);
        let len = self.len();
        let src = self;
        ctx.while_loop(lt(i, len), move |ctx| {
            let elem = ctx.bind(IndexedSource::get_at(src.clone(), i));
            ctx.if_then(pred(elem), move |ctx| {
                ctx.store(found, true);
                ctx.break_loop();
            });
            ctx.store(i, add(i, 1u64));
        });
        found
    }

    /// Return `true` only if *every* element satisfies `pred` (short-circuits
    /// on the first failure).
    fn all<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<bool>
    where
        Self: Sized,
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = BoolType> + 'static,
    {
        let i = ctx.var(0u64);
        let result = ctx.var(true);
        let len = self.len();
        let src = self;
        ctx.while_loop(lt(i, len), move |ctx| {
            let elem = ctx.bind(IndexedSource::get_at(src.clone(), i));
            ctx.if_then(not(pred(elem)), move |ctx| {
                ctx.store(result, false);
                ctx.break_loop();
            });
            ctx.store(i, add(i, 1u64));
        });
        result
    }

    /// Index of the first element satisfying `pred`, or the source length if
    /// none match (short-circuits).
    fn position<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<U64Type>
    where
        Self: Sized,
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = BoolType> + 'static,
    {
        let i = ctx.var(0u64);
        let len = ctx.bind(self.len());
        let pos = ctx.var(len); // sentinel: len means "not found"
        let src = self;
        ctx.while_loop(lt(i, len), move |ctx| {
            let elem = ctx.bind(IndexedSource::get_at(src.clone(), i));
            ctx.if_then(pred(elem), move |ctx| {
                ctx.store(pos, i);
                ctx.break_loop();
            });
            ctx.store(i, add(i, 1u64));
        });
        pos
    }
}

impl<S: IndexedSource> IndexedEarlyExit for S {}
