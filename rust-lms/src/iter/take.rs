//! Counted prefixes and suffixes: [`Take`] and [`Skip`].

use crate::func::Ctx;
use crate::label::Label;
use crate::num::{add, ge, lt, select, sub};
use crate::staged::{Staged, Var};

use super::traits::{Cursor, StagedIterator};

// =============================================================================
// Take
// =============================================================================

/// At most the first `n` elements. Built by [`StagedIterator::take`].
///
/// The bound is checked *before* pulling, so the element after the `n`-th is
/// never produced (an extern source is not advanced past it).
pub struct Take<I, N> {
    inner: I,
    n: N,
}

impl<I, N> Take<I, N> {
    pub(crate) fn new(inner: I, n: N) -> Self {
        Take { inner, n }
    }
}

impl<I, N> StagedIterator for Take<I, N>
where
    I: StagedIterator,
    N: Staged<Out = u64> + 'static,
{
    type Item = I::Item;
    type Cursor = TakeCursor<I::Cursor>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        let n = ctx.bind(self.n);
        TakeCursor {
            inner: self.inner.open(ctx),
            n,
            taken: ctx.var(0u64),
        }
    }
}

/// The cursor of a [`Take`]. Keeps random access (length `min(len, n)`).
pub struct TakeCursor<C> {
    inner: C,
    n: Var<u64>,
    taken: Var<u64>,
}

impl<C: Cursor> Cursor for TakeCursor<C> {
    type Item = C::Item;
    type Close = C::Close;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<C::Item>, C::Close) {
        let taken = self.taken;
        ctx.exit_if(ge(taken, self.n), done);
        let (elem, close) = self.inner.next(ctx, done);
        ctx.store(taken, add(taken, 1u64));
        (elem, close)
    }

    fn indexed_len(&mut self, ctx: &mut Ctx) -> Option<Var<u64>> {
        let len = self.inner.indexed_len(ctx)?;
        Some(ctx.bind(select(lt(len, self.n), len, self.n)))
    }

    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<C::Item>, C::Close) {
        // SAFETY: `index < min(len, n) <= len`.
        unsafe { self.inner.next_at(ctx, index) }
    }
}

// =============================================================================
// Skip
// =============================================================================

/// All but the first `n` elements. Built by [`StagedIterator::skip`].
pub struct Skip<I, N> {
    inner: I,
    n: N,
}

impl<I, N> Skip<I, N> {
    pub(crate) fn new(inner: I, n: N) -> Self {
        Skip { inner, n }
    }
}

impl<I, N> StagedIterator for Skip<I, N>
where
    I: StagedIterator,
    N: Staged<Out = u64> + 'static,
{
    type Item = I::Item;
    type Cursor = SkipCursor<I::Cursor>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        let n = ctx.bind(self.n);
        SkipCursor {
            inner: self.inner.open(ctx),
            n,
            skipped: ctx.var(0u64),
        }
    }
}

/// The cursor of a [`Skip`]. Keeps random access (element `i` is `inner[i + n]`).
pub struct SkipCursor<C> {
    inner: C,
    n: Var<u64>,
    skipped: Var<u64>,
}

impl<C: Cursor> Cursor for SkipCursor<C> {
    type Item = C::Item;
    type Close = C::Close;

    /// Pull and discard until `n` elements are gone; one predictable branch per
    /// element afterwards. (Driven by index, a skip is just an offset.)
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<C::Item>, C::Close) {
        let (inner, n, skipped) = (self.inner, self.n, self.skipped);
        ctx.repeat(|ctx, again| {
            let (elem, close) = inner.next(ctx, done);
            ctx.if_then(lt(skipped, n), move |ctx| {
                ctx.store(skipped, add(skipped, 1u64));
                ctx.again(again);
            });
            (elem, close)
        })
    }

    fn indexed_len(&mut self, ctx: &mut Ctx) -> Option<Var<u64>> {
        let len = self.inner.indexed_len(ctx)?;
        Some(ctx.bind(select(lt(self.n, len), sub(len, self.n), 0u64)))
    }

    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<C::Item>, C::Close) {
        let at = ctx.bind(add(index, self.n));
        // SAFETY: `index < len - n`, so `index + n < len`.
        unsafe { self.inner.next_at(ctx, at) }
    }
}
