//! FilterMap combinator — map-and-filter fused via a staging-time optional.

use crate::func::Ctx;
use crate::staged::Var;
use crate::staged_opt::StagedOpt;

use super::traits::StagedIterator;

/// Iterator adapter applying `f: Item -> impl StagedOpt`, keeping the `Some`
/// payloads. No `Option` is materialized: each element emits one branch and the
/// kept value stays in a register (see [`crate::staged_opt`]).
pub struct FilterMap<I, F> {
    inner: I,
    f: F,
}

impl<I, F> FilterMap<I, F> {
    pub(crate) fn new(inner: I, f: F) -> Self {
        FilterMap { inner, f }
    }
}

impl<I, F, O> StagedIterator for FilterMap<I, F>
where
    I: StagedIterator,
    F: Fn(Var<I::Item>) -> O + 'static,
    O: StagedOpt + 'static,
    O::Item: 'static,
{
    type Item = O::Item;

    fn for_each<G>(self, ctx: &mut Ctx, consumer: G)
    where
        G: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
    {
        let f = self.f;
        self.inner.for_each(ctx, move |ctx, elem| {
            // None => drop (empty continuation); Some(v) => push downstream.
            f(elem).eliminate(ctx, move |ctx, v| consumer(ctx, v), |_| {});
        });
    }
}
