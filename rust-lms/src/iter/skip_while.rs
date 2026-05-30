//! SkipWhile combinator — drops a leading run, yields the rest.

use crate::control::not;
use crate::func::Ctx;
use crate::staged::{Staged, Var};
use crate::types::{BoolType, CopyType};

use super::traits::StagedIterator;

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
    Cond: Staged<Out = BoolType> + 'static,
{
    type Item = I::Item;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
    {
        let pred = self.pred;
        // `skipping` starts true and latches to false at the first element where
        // the predicate fails; from then on every element is emitted.
        let skipping = ctx.var(true);
        self.inner.for_each(ctx, move |ctx, elem| {
            ctx.if_then(skipping, move |ctx| {
                ctx.if_then(not(pred(elem)), move |ctx| {
                    ctx.store(skipping, false);
                });
            });
            ctx.if_then(not(skipping), move |ctx| {
                consumer(ctx, elem);
            });
        });
    }
}
