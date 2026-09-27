//! FilterMap combinator — map-and-filter fused via a staging-time optional.

use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::staged::Var;
use crate::staged_opt::StagedOpt;

use super::traits::{Cursor, StagedIterator};

/// Iterator adapter applying `f: Item -> impl StagedOpt`, keeping the `Some`
/// payloads. No `Option` is materialized: each element emits one branch and the
/// kept value stays in a register (see [`crate::staged_opt`]).
pub struct FilterMap<I, F> {
    inner: I,
    f: F,
}

impl<I, F> FilterMap<I, F> {
    pub(crate) fn new(inner: I, f: F) -> Self {
        FilterMap { inner, f }
    }
}

impl<I, F, O> StagedIterator for FilterMap<I, F>
where
    I: StagedIterator,
    F: Fn(Var<I::Item>) -> O + 'static,
    O: StagedOpt + 'static,
    O::Item: 'static,
{
    type Item = O::Item;
    type Cursor = FilterMapCursor<I::Cursor, F>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        FilterMapCursor {
            inner: self.inner.open(ctx),
            f: self.f,
        }
    }
}

/// The cursor of a [`FilterMap`]: pull until `f` yields `Some`.
pub struct FilterMapCursor<C, F> {
    inner: C,
    f: F,
}

impl<C, F, O> Cursor for FilterMapCursor<C, F>
where
    C: Cursor,
    F: Fn(Var<C::Item>) -> O + 'static,
    O: StagedOpt + 'static,
    O::Item: 'static,
{
    type Item = O::Item;
    type Close = C::Close;

    /// ```text
    /// got: {
    ///   again: elem = inner.next()      ; exhausted → done
    ///          match f(elem) { Some(v) → got(v), None → again }
    /// }
    /// ```
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<O::Item>, C::Close) {
        let (inner, f) = (self.inner, self.f);
        let mut close = None;
        let value = ctx.join(|ctx, got| {
            ctx.repeat(|ctx, again| {
                let (elem, c) = inner.next(ctx, done);
                close = Some(c);
                f(elem).eliminate(ctx, |ctx, v| ctx.goto(got, v), |ctx| ctx.again(again));
            });
            dead::<O::Item>()
        });
        (value, close.expect("the inner step is staged exactly once"))
    }
}
