//! TakeWhile combinator — yields a prefix, stops at the first failing element.

use crate::control::not;
use crate::func::Ctx;
use crate::staged::{Staged, Var};
use crate::types::CopyType;

use super::traits::StagedIterator;

/// Iterator adapter that yields elements while a predicate holds, then stops
/// the entire iteration (short-circuits via `break_loop`).
pub struct TakeWhile<I, P> {
    inner: I,
    pred: P,
}

impl<I, P> TakeWhile<I, P> {
    pub(crate) fn new(inner: I, pred: P) -> Self {
        TakeWhile { inner, pred }
    }
}

impl<I, P, Cond> StagedIterator for TakeWhile<I, P>
where
    I: StagedIterator,
    I::Item: CopyType + 'static,
    P: Fn(Var<I::Item>) -> Cond + 'static,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = I::Item;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
    {
        let pred = self.pred;
        self.inner.for_each(ctx, move |ctx, elem| {
            // Break out of the source loop at the first failing element. On the
            // passing path control falls through to `consumer`; the merge block
            // after the `if_then` is only reachable when `pred` held.
            ctx.if_then(not(pred(elem)), move |ctx| {
                ctx.break_loop();
            });
            consumer(ctx, elem);
        });
    }
}
