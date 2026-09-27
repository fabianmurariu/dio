//! Filter combinator — keeps only elements matching a predicate.

use crate::func::Ctx;
use crate::label::Label;
use crate::staged::{Staged, Var};

use super::traits::{Cursor, StagedIterator};

/// Iterator adapter that keeps only elements satisfying a predicate.
///
/// Does NOT implement `IndexedStagedIterator` — filter breaks index correspondence.
pub struct Filter<I, P> {
    pub(crate) inner: I,
    pub(crate) predicate: P,
}

impl<I, P> Filter<I, P> {
    pub(crate) fn new(inner: I, predicate: P) -> Self {
        Filter { inner, predicate }
    }
}

impl<I, P, Cond> StagedIterator for Filter<I, P>
where
    I: StagedIterator,
    I::Item: crate::types::CopyType + 'static,
    P: Fn(Var<I::Item>) -> Cond + 'static,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = I::Item;
    type Cursor = FilterCursor<I::Cursor, P>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        FilterCursor {
            inner: self.inner.open(ctx),
            predicate: self.predicate,
        }
    }
}

/// The cursor of a [`Filter`]: pull until the predicate holds.
pub struct FilterCursor<C, P> {
    inner: C,
    predicate: P,
}

impl<C, P, Cond> Cursor for FilterCursor<C, P>
where
    C: Cursor,
    C::Item: crate::types::CopyType + 'static,
    P: Fn(Var<C::Item>) -> Cond + 'static,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = C::Item;
    type Close = C::Close;

    /// ```text
    /// again: elem = inner.next()        ; exhausted → done
    ///        if !p(elem) → again
    /// ```
    /// The retry is a `repeat`, not a loop scope, so a consumer's `break_loop`
    /// still leaves the whole iteration.
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<C::Item>, C::Close) {
        let (inner, predicate) = (self.inner, self.predicate);
        ctx.repeat(|ctx, again| {
            let (elem, close) = inner.next(ctx, done);
            ctx.again_unless(predicate(elem), again);
            (elem, close)
        })
    }
}
