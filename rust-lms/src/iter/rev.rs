//! Back-to-front iteration: [`Rev`].

use crate::func::Ctx;
use crate::num::{gt, sub};
use crate::staged::Var;

use crate::label::Label;

use super::traits::{Cursor, IndexedSource, StagedIterator};

/// Yields an indexed source's elements from last to first.
///
/// Built by [`IndexedStagedIterator::rev`](super::IndexedStagedIterator::rev).
/// Reversal needs random access — a stream can only be pulled forwards —
/// which is why this is keyed on [`IndexedSource`] rather than on
/// [`StagedIterator`].
pub struct Rev<I> {
    iter: I,
}

impl<I: Clone> Clone for Rev<I> {
    fn clone(&self) -> Self {
        Rev {
            iter: self.iter.clone(),
        }
    }
}

impl<I> Rev<I> {
    pub(super) fn new(iter: I) -> Self {
        Rev { iter }
    }
}

impl<I> StagedIterator for Rev<I>
where
    I: IndexedSource,
{
    type Item = I::Item;
    type Cursor = RevCursor<I>;

    fn open(self, ctx: &mut Ctx) -> RevCursor<I> {
        let len = ctx.bind(self.iter.count());
        let remaining = ctx.var(len);
        RevCursor {
            src: self.iter,
            len,
            remaining,
        }
    }
}

/// The cursor of a [`Rev`]: counts down over the source. Random access
/// (element `k` of the reversal is `src[len - 1 - k]`).
pub struct RevCursor<I> {
    src: I,
    len: Var<u64>,
    remaining: Var<u64>,
}

impl<I: IndexedSource> Cursor for RevCursor<I> {
    type Item = I::Item;
    type Close = ();

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<I::Item>, ()) {
        let i = self.remaining;
        // Counting down to an exclusive `0` keeps the counter unsigned.
        ctx.exit_unless(gt(i, 0u64), done);
        ctx.store(i, sub(i, 1u64));
        // SAFETY: `i > 0` held before the decrement, so `i < count` here.
        let item = ctx.bind(unsafe { IndexedSource::get_at(self.src, i) });
        (item, ())
    }

    fn indexed_len(&mut self, _ctx: &mut Ctx) -> Option<Var<u64>> {
        Some(self.len)
    }

    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<I::Item>, ()) {
        let at = ctx.bind(sub(sub(self.len, 1u64), index));
        // SAFETY: `index < len`, so `len - 1 - index < len`.
        (ctx.bind(unsafe { IndexedSource::get_at(self.src, at) }), ())
    }
}
