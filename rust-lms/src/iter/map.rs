//! Map combinator — transforms each element.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::staged::{Staged, Var};
use crate::types::{ConstantType, CopyType, StagedType};

use crate::label::Label;

use super::traits::{Cursor, IndexedStagedIterator, StagedIterator};

/// Iterator adapter that transforms each element.
///
/// Preserves `IndexedStagedIterator` if the inner iterator has it.
pub struct Map<I, F, U> {
    pub(crate) inner: I,
    pub(crate) map_fn: F,
    _phantom: PhantomData<U>,
}

impl<I, F, U> Map<I, F, U> {
    pub(crate) fn new(inner: I, map_fn: F) -> Self {
        Map {
            inner,
            map_fn,
            _phantom: PhantomData,
        }
    }
}

impl<I, F, U, MapOut> StagedIterator for Map<I, F, U>
where
    I: StagedIterator,
    F: Fn(Var<I::Item>) -> MapOut + 'static,
    MapOut: Staged<Out = U> + 'static,
    U: StagedType + ConstantType + CopyType + 'static,
    U::RuntimeValue: Default,
{
    type Item = U;
    type Cursor = MapCursor<I::Cursor, F, U>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        MapCursor {
            inner: self.inner.open(ctx),
            map_fn: self.map_fn,
            _phantom: PhantomData,
        }
    }
}

/// The cursor of a [`Map`]: the inner step, then `f`. Keeps random access.
pub struct MapCursor<C, F, U> {
    inner: C,
    map_fn: F,
    _phantom: PhantomData<U>,
}

impl<C, F, U, MapOut> Cursor for MapCursor<C, F, U>
where
    C: Cursor,
    F: Fn(Var<C::Item>) -> MapOut + 'static,
    MapOut: Staged<Out = U> + 'static,
    U: StagedType + 'static,
{
    type Item = U;
    type Close = C::Close;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<U>, C::Close) {
        let (elem, close) = self.inner.next(ctx, done);
        (ctx.bind((self.map_fn)(elem)), close)
    }

    fn indexed_len(&mut self, ctx: &mut Ctx) -> Option<Var<u64>> {
        self.inner.indexed_len(ctx)
    }

    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<U>, C::Close) {
        // SAFETY: forwarded from the caller.
        let (elem, close) = unsafe { self.inner.next_at(ctx, index) };
        (ctx.bind((self.map_fn)(elem)), close)
    }
}

impl<I, F, U, MapOut> IndexedStagedIterator for Map<I, F, U>
where
    I: IndexedStagedIterator,
    F: Fn(Var<I::Item>) -> MapOut + 'static,
    MapOut: Staged<Out = U> + 'static,
    U: StagedType + ConstantType + CopyType + 'static,
    U::RuntimeValue: Default,
{
    type LenExpr = I::LenExpr;

    fn len(&self) -> Self::LenExpr {
        self.inner.len()
    }
}
