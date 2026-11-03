//! Boxed iterators — one Rust type per item type, whatever the pipeline behind it.
//!
//! A compiler choosing its iterators while it walks a plan (a list expression, a
//! layer's runs) cannot name their types. [`StagedIterator::boxed`] erases the type at
//! stage 0 only: the box and its dynamic calls exist while staging, and the emitted code
//! is the same as the unboxed iterator's. `for_each` and `count` reach the boxed
//! iterator's own, so a `flat_map` keeps its nested loops and a slice counts as its
//! length.
//!
//! [`StagedIterator`], [`Cursor`] and [`Close`] are not dyn-compatible (by-value `self`,
//! generic methods, `Clone`), so each has an erased twin here, implemented for every
//! type that implements the original. The boxes of those twins implement the originals
//! again. The forwarding calls name the `dyn` impl: the blanket impls also cover the
//! boxes, and a method call could pick the box's own impl and recurse.

use crate::func::Ctx;
use crate::label::Label;
use crate::staged::Var;
use crate::types::StagedType;

use super::traits::{Close, Cursor, IntoStagedIterator, StagedIterator};

/// A [`StagedIterator`] of `T` whose concrete type is erased.
pub type BoxedStagedIter<'a, T> = Box<dyn ErasedIter<'a, T> + 'a>;
/// The cursor of a [`BoxedStagedIter`].
pub type BoxedCursor<'a, T> = Box<dyn ErasedCursor<'a, T> + 'a>;
/// The close of a [`BoxedCursor`].
pub type BoxedClose<'a> = Box<dyn ErasedClose<'a> + 'a>;

/// [`StagedIterator`], dyn-compatible.
pub trait ErasedIter<'a, T: StagedType> {
    fn open_erased(self: Box<Self>, ctx: &Ctx) -> BoxedCursor<'a, T>;
    fn for_each_erased<'f>(
        self: Box<Self>,
        ctx: &Ctx,
        consumer: Box<dyn FnOnce(&Ctx, Var<T>) + 'f>,
    );
    fn count_erased(self: Box<Self>, ctx: &Ctx) -> Var<u64>;
}

impl<'a, I> ErasedIter<'a, I::Item> for I
where
    I: StagedIterator + 'a,
    I::Cursor: 'a,
    I::Item: 'static,
{
    fn open_erased(self: Box<Self>, ctx: &Ctx) -> BoxedCursor<'a, I::Item> {
        Box::new((*self).open(ctx))
    }

    fn for_each_erased<'f>(
        self: Box<Self>,
        ctx: &Ctx,
        consumer: Box<dyn FnOnce(&Ctx, Var<I::Item>) + 'f>,
    ) {
        (*self).for_each(ctx, consumer)
    }

    fn count_erased(self: Box<Self>, ctx: &Ctx) -> Var<u64> {
        (*self).count(ctx)
    }
}

impl<'a, T: StagedType + 'static> StagedIterator for BoxedStagedIter<'a, T> {
    type Item = T;
    type Cursor = BoxedCursor<'a, T>;

    fn open(self, ctx: &Ctx) -> Self::Cursor {
        <dyn ErasedIter<'a, T>>::open_erased(self, ctx)
    }

    fn for_each<F>(self, ctx: &Ctx, consumer: F)
    where
        F: FnOnce(&Ctx, Var<T>),
    {
        <dyn ErasedIter<'a, T>>::for_each_erased(self, ctx, Box::new(consumer))
    }

    fn count(self, ctx: &Ctx) -> Var<u64> {
        <dyn ErasedIter<'a, T>>::count_erased(self, ctx)
    }
}

impl<'a, T: StagedType + 'static> IntoStagedIterator for BoxedStagedIter<'a, T> {
    type Iter = Self;

    fn staged_iter(self) -> Self {
        self
    }
}

/// [`Cursor`], dyn-compatible.
pub trait ErasedCursor<'a, T: StagedType> {
    fn next_erased(self: Box<Self>, ctx: &Ctx, done: Label<'_>) -> (Var<T>, BoxedClose<'a>);
    fn indexed_len_erased(&mut self, ctx: &Ctx) -> Option<Var<u64>>;
    /// # Safety
    ///
    /// As [`Cursor::next_at`].
    unsafe fn next_at_erased(
        self: Box<Self>,
        ctx: &Ctx,
        index: Var<u64>,
    ) -> (Var<T>, BoxedClose<'a>);
}

impl<'a, C> ErasedCursor<'a, C::Item> for C
where
    C: Cursor + 'a,
    C::Close: 'a,
{
    fn next_erased(self: Box<Self>, ctx: &Ctx, done: Label<'_>) -> (Var<C::Item>, BoxedClose<'a>) {
        let (item, close) = (*self).next(ctx, done);
        (item, Box::new(close))
    }

    fn indexed_len_erased(&mut self, ctx: &Ctx) -> Option<Var<u64>> {
        self.indexed_len(ctx)
    }

    unsafe fn next_at_erased(
        self: Box<Self>,
        ctx: &Ctx,
        index: Var<u64>,
    ) -> (Var<C::Item>, BoxedClose<'a>) {
        // SAFETY: forwarded from the caller.
        let (item, close) = unsafe { (*self).next_at(ctx, index) };
        (item, Box::new(close))
    }
}

impl<'a, T: StagedType> Cursor for BoxedCursor<'a, T> {
    type Item = T;
    type Close = BoxedClose<'a>;

    fn next(self, ctx: &Ctx, done: Label<'_>) -> (Var<T>, BoxedClose<'a>) {
        <dyn ErasedCursor<'a, T>>::next_erased(self, ctx, done)
    }

    fn indexed_len(&mut self, ctx: &Ctx) -> Option<Var<u64>> {
        <dyn ErasedCursor<'a, T>>::indexed_len_erased(&mut **self, ctx)
    }

    unsafe fn next_at(self, ctx: &Ctx, index: Var<u64>) -> (Var<T>, BoxedClose<'a>) {
        // SAFETY: forwarded from the caller.
        unsafe { <dyn ErasedCursor<'a, T>>::next_at_erased(self, ctx, index) }
    }
}

/// [`Close`], dyn-compatible.
pub trait ErasedClose<'a> {
    fn close_erased(self: Box<Self>, ctx: &Ctx);
    fn clone_erased(&self) -> BoxedClose<'a>;
}

impl<'a, C: Close + 'a> ErasedClose<'a> for C {
    fn close_erased(self: Box<Self>, ctx: &Ctx) {
        (*self).close(ctx)
    }

    fn clone_erased(&self) -> BoxedClose<'a> {
        Box::new(self.clone())
    }
}

impl<'a> Clone for BoxedClose<'a> {
    fn clone(&self) -> Self {
        <dyn ErasedClose<'a>>::clone_erased(&**self)
    }
}

impl<'a> Close for BoxedClose<'a> {
    fn close(self, ctx: &Ctx) {
        <dyn ErasedClose<'a>>::close_erased(self, ctx)
    }
}
