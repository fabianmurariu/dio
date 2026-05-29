//! Map combinator — transforms each element.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::staged::{Staged, Var};
use crate::types::{ConstantType, CopyType, StagedType, U64Type};

use super::traits::{IndexedStagedIterator, StagedIterator};

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
        Map { inner, map_fn, _phantom: PhantomData }
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

    fn for_each<G>(self, ctx: &mut Ctx, consumer: G)
    where
        G: FnOnce(&mut Ctx, Var<U>) + 'static,
    {
        let map_fn = self.map_fn;
        self.inner.for_each(ctx, move |ctx, inner_elem| {
            let mapped = ctx.bind(map_fn(inner_elem));
            consumer(ctx, mapped);
        });
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
