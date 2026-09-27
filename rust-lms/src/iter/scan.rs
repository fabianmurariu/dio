//! Scan combinator — a stateful map threading a user-managed `Var` state.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::staged::{IntoStaged, Var};
use crate::types::{CopyType, StagedType};

use crate::label::Label;

use super::traits::{Cursor, StagedIterator};

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
    type Cursor = ScanCursor<I::Cursor, St, F>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        // Allocate the state once, before the loop; update it each step.
        let state = ctx.var(self.init);
        ScanCursor {
            inner: self.inner.open(ctx),
            state,
            f: self.f,
        }
    }
}

/// The cursor of a [`Scan`]: the inner step, then the state update.
pub struct ScanCursor<C, St: StagedType, F> {
    inner: C,
    state: Var<St>,
    f: F,
}

impl<C, St, F> Cursor for ScanCursor<C, St, F>
where
    C: Cursor,
    St: StagedType + CopyType + 'static,
    F: Fn(&mut Ctx, Var<St>, Var<C::Item>) + 'static,
{
    type Item = St;
    type Close = C::Close;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<St>, C::Close) {
        let (elem, close) = self.inner.next(ctx, done);
        (self.f)(ctx, self.state, elem);
        (self.state, close)
    }
}
