//! FlatMap combinator — an inner iterator per outer element, flattened.
//!
//! Two lowerings, picked by how the stream is used (docs/pull_iter.md §5.7):
//!
//! - **Consumed** (`for_each` and every terminal): natural nested loops — the
//!   inner loop stays a tight loop of its own.
//! - **Pulled** (inside `zip`, `chain`, `merge_by`, `take`, …): a cursor whose
//!   step is a small state machine — an `active` flag says whether an inner
//!   iterator is open, and exhausting it retries with the next outer element.

use crate::control::not;
use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::staged::Var;

use super::traits::{Close, Cursor, IntoStagedIterator, StagedIterator};

/// Maps each element to an iterator and yields the inner iterators' elements
/// in order. Built by [`StagedIterator::flat_map`].
pub struct FlatMap<I, F> {
    outer: I,
    f: F,
}

impl<I, F> FlatMap<I, F> {
    pub(crate) fn new(outer: I, f: F) -> Self {
        FlatMap { outer, f }
    }
}

/// The inner iterator type `f` produces for an outer element.
type Inner<J> = <J as IntoStagedIterator>::Iter;
/// That inner iterator's cursor.
type InnerCursor<J> = <Inner<J> as StagedIterator>::Cursor;

impl<I, F, J> StagedIterator for FlatMap<I, F>
where
    I: StagedIterator,
    F: Fn(Var<I::Item>) -> J,
    J: IntoStagedIterator,
    Inner<J>: StagedIterator,
    <Inner<J> as StagedIterator>::Item: 'static,
{
    type Item = <Inner<J> as StagedIterator>::Item;
    type Cursor = FlatMapCursor<I::Cursor, F, J>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        let active = ctx.var(false);
        FlatMapCursor {
            outer: self.outer.open(ctx),
            f: self.f,
            active,
            _inner: std::marker::PhantomData,
        }
    }

    /// Nested loops: `for o in outer { for x in f(o) { consumer(x) } }`.
    ///
    /// A consumer's `break_loop` (`any`, `find_map`, `position`, a downstream
    /// early exit) must stop the *whole* traversal, not just the current inner
    /// iterator — and still release that inner iterator first. So the inner
    /// loop's two exits are told apart with a label:
    ///
    /// ```text
    /// for o in outer {                       ; the outer driver loop
    ///   cursor = f(o).open()
    ///   exhausted: {
    ///     loop {                             ; inner loop; break_loop → exit
    ///       x = cursor.next()                ; exhausted → `exhausted`
    ///       consumer(x)
    ///     }
    ///     close(cursor); break_loop          ; reached only by a consumer break:
    ///   }                                    ;   re-raise it on the outer loop
    ///   close(cursor)                        ; normal end of this inner list
    /// }
    /// ```
    fn for_each<G>(self, ctx: &mut Ctx, consumer: G)
    where
        G: FnOnce(&mut Ctx, Var<Self::Item>),
    {
        let f = self.f;
        self.outer.for_each(ctx, move |ctx, o| {
            let cursor = f(o).staged_iter().open(ctx);
            let mut close = None;
            ctx.block(|ctx, exhausted| {
                ctx.iterate(|ctx, _| {
                    let (x, c) = cursor.next(ctx, exhausted);
                    close = Some(c);
                    consumer(ctx, x);
                });
                let close = close.expect("the inner step is staged exactly once");
                close.close(ctx);
                ctx.break_loop();
            });
            close
                .expect("the inner step is staged exactly once")
                .close(ctx);
        });
    }
}

/// The cursor of a [`FlatMap`] (the pulled lowering).
pub struct FlatMapCursor<C, F, J> {
    outer: C,
    f: F,
    /// Whether an inner iterator is open (and not yet exhausted).
    active: Var<bool>,
    _inner: std::marker::PhantomData<fn() -> J>,
}

impl<C, F, J> Cursor for FlatMapCursor<C, F, J>
where
    C: Cursor,
    F: Fn(Var<C::Item>) -> J,
    J: IntoStagedIterator,
    Inner<J>: StagedIterator,
    <Inner<J> as StagedIterator>::Item: 'static,
{
    type Item = <Inner<J> as StagedIterator>::Item;
    type Close = (C::Close, CloseIfActive<<InnerCursor<J> as Cursor>::Close>);

    /// ```text
    /// got: {
    ///   again:
    ///     if !active {
    ///       o = outer.next()              ; exhausted → done (the whole stream)
    ///       inner = f(o).open(); active = true
    ///     }
    ///     inner_done: { x = inner.next() (exhausted → inner_done); got(x) }
    ///     close(inner); active = false
    ///     → again
    /// }
    /// ```
    /// Each step — outer, inner, and the inner open — is staged once; the inner
    /// cursor's state is defined in the `!active` branch and loop-carried after.
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<Self::Item>, Self::Close) {
        let (outer, f, active) = (self.outer, self.f, self.active);
        let (mut outer_close, mut inner_close) = (None, None);
        let item = ctx.join(|ctx, got| {
            ctx.repeat(|ctx, again| {
                let mut inner = None;
                ctx.if_then(not(active), |ctx| {
                    let (o, c) = outer.next(ctx, done);
                    outer_close = Some(c);
                    inner = Some(f(o).staged_iter().open(ctx));
                    ctx.store(active, true);
                });
                let inner = inner.expect("the inner open is staged exactly once");
                ctx.block(|ctx, inner_done| {
                    let (x, c) = inner.next(ctx, inner_done);
                    inner_close = Some(c);
                    ctx.goto(got, x);
                });
                let close = inner_close.expect("the inner step is staged exactly once");
                close.close(ctx);
                ctx.store(active, false);
                ctx.again(again);
            });
            dead::<Self::Item>()
        });
        let staged_once = "each step is staged exactly once";
        (
            item,
            (
                outer_close.expect(staged_once),
                CloseIfActive {
                    active,
                    inner: inner_close.expect(staged_once),
                },
            ),
        )
    }
}

/// Releases the current inner iterator if the traversal stopped while one was
/// open (an early exit); an exhausted inner one was already closed in the step.
pub struct CloseIfActive<C> {
    active: Var<bool>,
    inner: C,
}

impl<C: Copy> Clone for CloseIfActive<C> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<C: Copy> Copy for CloseIfActive<C> {}

impl<C: Close> Close for CloseIfActive<C> {
    fn close(self, ctx: &mut Ctx) {
        let inner = self.inner;
        ctx.if_then(self.active, move |ctx| inner.close(ctx));
    }
}
