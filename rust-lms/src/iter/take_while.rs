//! TakeWhile combinator — yields a prefix, stops at the first failing element.

use crate::func::Ctx;
use crate::staged::{Staged, Var};
use crate::types::CopyType;

use crate::label::Label;

use super::traits::{Cursor, StagedIterator};

/// Iterator adapter that yields elements while a predicate holds, and ends at
/// the first element where it fails.
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
    type Cursor = TakeWhileCursor<I::Cursor, P>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        TakeWhileCursor {
            inner: self.inner.open(ctx),
            pred: self.pred,
        }
    }
}

/// The cursor of a [`TakeWhile`]: the first failing element *is* exhaustion.
///
/// No `break_loop`: jumping to `done` ends exactly this stream, so
/// `a.take_while(p).chain(b)` moves on to `b` and a `zip` stops cleanly.
pub struct TakeWhileCursor<C, P> {
    inner: C,
    pred: P,
}

impl<C, P, Cond> Cursor for TakeWhileCursor<C, P>
where
    C: Cursor,
    C::Item: CopyType + 'static,
    P: Fn(Var<C::Item>) -> Cond + 'static,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = C::Item;
    type Close = C::Close;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<C::Item>, C::Close) {
        let (elem, close) = self.inner.next(ctx, done);
        ctx.exit_unless((self.pred)(elem), done);
        (elem, close)
    }
}
