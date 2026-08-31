//! Index-carrying iteration: [`Enumerate`].

use crate::func::Ctx;
use crate::num::add;
use crate::staged::Var;
use crate::types::CopyType;

use super::traits::StagedIterator;
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
        self.iter.for_each(ctx, move |ctx, elem| {
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

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>),
    {
        let index = ctx.var(0u64);
        self.iter.for_each(ctx, move |ctx, elem| {
            let pair = ctx.bind(Pair::new(index, elem));
            consumer(ctx, pair);
            ctx.store(index, add(index, 1u64));
        });
    }
}
