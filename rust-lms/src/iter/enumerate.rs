//! Index-carrying iteration: [`Enumerate`].

use crate::func::Ctx;
use crate::num::add;
use crate::staged::Var;
use crate::types::CopyType;

use crate::label::Label;

use super::traits::{Cursor, StagedIterator, drive};
use super::zip::{Pair, ZipItem};

/// Yields each element together with its zero-based position.
///
/// Like [`Zip`](super::Zip) it has two paths. The inherent
/// [`for_each`](Self::for_each) hands the index and the element over as separate
/// vars and materialises no pair at all; the [`StagedIterator`] impl reports
/// `Item = ZipItem<u64, _>` so `enumerate` still composes with `map`, `filter`
/// and the terminals.
pub struct Enumerate<I> {
    iter: I,
}

impl<I: Clone> Clone for Enumerate<I> {
    fn clone(&self) -> Self {
        Enumerate {
            iter: self.iter.clone(),
        }
    }
}

impl<I: Copy> Copy for Enumerate<I> {}

impl<I> Enumerate<I> {
    pub(super) fn new(iter: I) -> Self {
        Enumerate { iter }
    }
}

impl<I> Enumerate<I>
where
    I: StagedIterator,
{
    /// Visit every `(index, element)` pair.
    ///
    /// Shadows [`StagedIterator::for_each`] with a three-argument consumer, so
    /// the index and the element arrive as separate vars rather than through a
    /// pair that would round-trip a stack slot.
    pub fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<u64>, Var<I::Item>),
    {
        let index = ctx.var(0u64);
        let cursor = self.iter.open(ctx);
        drive(cursor, ctx, move |ctx, elem| {
            consumer(ctx, index, elem);
            ctx.store(index, add(index, 1u64));
        });
    }
}

impl<I> StagedIterator for Enumerate<I>
where
    I: StagedIterator,
    I::Item: CopyType + 'static,
{
    type Item = ZipItem<u64, I::Item>;
    type Cursor = EnumerateCursor<I::Cursor>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        let index = ctx.var(0u64);
        EnumerateCursor {
            inner: self.iter.open(ctx),
            index,
        }
    }
}

/// The cursor of an [`Enumerate`]: the inner step paired with a counter.
/// Keeps random access (element `i` is `(i, inner[i])`).
pub struct EnumerateCursor<C> {
    inner: C,
    index: Var<u64>,
}

impl<C> Cursor for EnumerateCursor<C>
where
    C: Cursor,
    C::Item: CopyType + 'static,
{
    type Item = ZipItem<u64, C::Item>;
    type Close = C::Close;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<Self::Item>, C::Close) {
        let (elem, close) = self.inner.next(ctx, done);
        let index = self.index;
        // The pair captures the index's current value before the increment.
        let pair = ctx.bind(Pair::new(index, elem));
        ctx.store(index, add(index, 1u64));
        (pair, close)
    }

    fn indexed_len(&mut self, ctx: &mut Ctx) -> Option<Var<u64>> {
        self.inner.indexed_len(ctx)
    }

    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<Self::Item>, C::Close) {
        // SAFETY: forwarded from the caller.
        let (elem, close) = unsafe { self.inner.next_at(ctx, index) };
        (ctx.bind(Pair::new(index, elem)), close)
    }
}
