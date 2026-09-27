//! SkipWhile combinator — drops a leading run, yields the rest.

use crate::func::Ctx;
use crate::staged::{Staged, Var};
use crate::types::CopyType;

use crate::label::Label;

use super::traits::{Cursor, StagedIterator};

/// Iterator adapter that skips leading elements while a predicate holds, then
/// yields every element from the first failure onward.
pub struct SkipWhile<I, P> {
    inner: I,
    pred: P,
}

impl<I, P> SkipWhile<I, P> {
    pub(crate) fn new(inner: I, pred: P) -> Self {
        SkipWhile { inner, pred }
    }
}

impl<I, P, Cond> StagedIterator for SkipWhile<I, P>
where
    I: StagedIterator,
    I::Item: CopyType + 'static,
    P: Fn(Var<I::Item>) -> Cond + 'static,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = I::Item;
    type Cursor = SkipWhileCursor<I::Cursor, P>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        // `skipping` starts true and latches to false at the first element where
        // the predicate fails; from then on every element is yielded.
        let skipping = ctx.var(true);
        SkipWhileCursor {
            inner: self.inner.open(ctx),
            pred: self.pred,
            skipping,
        }
    }
}

/// The cursor of a [`SkipWhile`]: retry while the leading run lasts.
pub struct SkipWhileCursor<C, P> {
    inner: C,
    pred: P,
    skipping: Var<bool>,
}

impl<C, P, Cond> Cursor for SkipWhileCursor<C, P>
where
    C: Cursor,
    C::Item: CopyType + 'static,
    P: Fn(Var<C::Item>) -> Cond + 'static,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = C::Item;
    type Close = C::Close;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<C::Item>, C::Close) {
        let (inner, pred, skipping) = (self.inner, self.pred, self.skipping);
        ctx.repeat(|ctx, again| {
            let (elem, close) = inner.next(ctx, done);
            ctx.if_then(skipping, move |ctx| {
                ctx.again_if(pred(elem), again);
                ctx.store(skipping, false);
            });
            (elem, close)
        })
    }
}
