//! Scan combinator — a stateful map threading a user-managed `Var` state.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::staged::{IntoStaged, Var};
use crate::types::{CopyType, StagedType};

use super::traits::StagedIterator;

/// Iterator adapter that threads a mutable `Var<St>` state through the
/// iteration, emitting the post-update state as each element.
///
/// Unlike Rust's `scan`, this does not short-circuit (no `Option` return); the
/// state update `f(ctx, state, elem)` runs for every element.
pub struct Scan<I, St, Init, F> {
    inner: I,
    init: Init,
    f: F,
    _phantom: PhantomData<St>,
}

impl<I, St, Init, F> Scan<I, St, Init, F> {
    pub(crate) fn new(inner: I, init: Init, f: F) -> Self {
        Scan {
            inner,
            init,
            f,
            _phantom: PhantomData,
        }
    }
}

impl<I, St, Init, F> StagedIterator for Scan<I, St, Init, F>
where
    I: StagedIterator,
    St: StagedType + CopyType + 'static,
    Init: IntoStaged<St>,
    Init::Staged: 'static,
    F: Fn(&mut Ctx, Var<St>, Var<I::Item>) + 'static,
{
    type Item = St;

    fn for_each<G>(self, ctx: &mut Ctx, consumer: G)
    where
        G: FnOnce(&mut Ctx, Var<St>) + 'static,
    {
        // Allocate the state once, before the loop; update it each iteration.
        let state = ctx.var(self.init);
        let f = self.f;
        self.inner.for_each(ctx, move |ctx, elem| {
            f(ctx, state, elem);
            consumer(ctx, state);
        });
    }
}
