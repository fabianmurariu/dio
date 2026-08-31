//! Back-to-front iteration: [`Rev`].

use crate::func::Ctx;
use crate::num::{gt, sub};
use crate::staged::Var;

use super::traits::{IndexedSource, StagedIterator};

/// Yields an indexed source's elements from last to first.
///
/// Built by [`IndexedStagedIterator::rev`](super::IndexedStagedIterator::rev).
/// Reversal needs random access — a push-based iterator has no way to run
/// backwards — which is why this is keyed on [`IndexedSource`] rather than on
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

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>),
    {
        let n = ctx.bind(self.iter.count());
        let i = ctx.var(n);
        let src = self.iter;

        ctx.while_loop(gt(i, 0u64), move |ctx| {
            // Decrement first, so `i` walks `count - 1` down to `0`. Counting
            // down to an exclusive `0` keeps the counter unsigned throughout.
            ctx.store(i, sub(i, 1u64));
            // SAFETY: the loop condition proved `i > 0` before the decrement,
            // so `i < count` here.
            let item = ctx.bind(unsafe { IndexedSource::get_at(src.clone(), i) });
            consumer(ctx, item);
        });
    }
}
