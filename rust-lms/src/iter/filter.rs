//! Filter combinator — keeps only elements matching a predicate.

use crate::func::Ctx;
use crate::staged::{Staged, Var};
use crate::types::BoolType;

use super::traits::StagedIterator;

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
    Cond: Staged<Out = BoolType> + 'static,
{
    type Item = I::Item;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
    {
        let predicate = self.predicate;
        self.inner.for_each(ctx, move |ctx, elem| {
            let cond = predicate(elem);
            ctx.if_then(cond, move |ctx| {
                consumer(ctx, elem);
            });
        });
    }
}
