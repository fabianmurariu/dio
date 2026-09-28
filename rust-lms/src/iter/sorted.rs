//! Operators on sorted streams: [`MergeBy`] (sorted union) and [`IntersectBy`]
//! (sorted intersection, i.e. a merge join).
//!
//! Both keep one element of lookahead per input — a [`Lookahead`]: the current
//! head, and a `need` flag saying it was consumed. The pull that replaces a
//! consumed head is deferred to the top of the next step, which is what lets
//! each input's step be staged exactly once (a priming pull in `open` would
//! stage it twice).
//!
//! There is deliberately no public `peekable()`: a consumer closure only ever
//! receives items, never the cursor, so a peek is only usable *inside* an
//! operator — and these are those operators.

use crate::control::not;
use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::num::select;
use crate::staged::{Staged, Var};
use crate::types::CopyType;

use super::traits::{Cursor, StagedIterator};

// =============================================================================
// Lookahead
// =============================================================================

/// One input of a sorted-stream operator: its cursor and one element of
/// lookahead.
pub struct Lookahead<C: Cursor> {
    cursor: C,
    /// The current head; valid once pulled and until consumed.
    head: Var<C::Item>,
    /// The head was consumed (or never pulled): pull at the next refill.
    need: Var<bool>,
}

impl<C> Lookahead<C>
where
    C: Cursor,
    C::Item: CopyType + 'static,
{
    /// Declare the lookahead state (before the driving loop).
    fn declare(ctx: &mut Ctx, cursor: C) -> Self {
        Lookahead {
            cursor,
            // Never read before the first pull writes it (`need` starts true).
            head: ctx.var(dead::<C::Item>()),
            need: ctx.var(true),
        }
    }

    /// If the head was consumed, pull the next one; jump to `exhausted` when
    /// there is none. Stages the input's step (once). Returns the head, the
    /// flag that marks it consumed, and the input's close.
    fn refill(self, ctx: &mut Ctx, exhausted: Label<'_>) -> (Var<C::Item>, Var<bool>, C::Close) {
        let Lookahead { cursor, head, need } = self;
        let mut close = None;
        ctx.if_then(need, |ctx| {
            let (x, c) = cursor.next(ctx, exhausted);
            close = Some(c);
            ctx.store(head, x);
            ctx.store(need, false);
        });
        (
            head,
            need,
            close.expect("each input's step is staged exactly once"),
        )
    }
}

// =============================================================================
// MergeBy
// =============================================================================

/// The elements of two streams, each sorted by `le`, merged into one sorted
/// stream. Stable: on ties the left stream's element comes first. Built by
/// [`StagedIterator::merge_by`] / [`merge`](StagedIterator::merge).
///
/// The query-engine use: one time-ordered stream from an in-memory structure
/// (an extern iterator) and one from an Arrow column, merged by timestamp.
pub struct MergeBy<A, B, P> {
    a: A,
    b: B,
    le: P,
}

impl<A, B, P> MergeBy<A, B, P> {
    pub(crate) fn new(a: A, b: B, le: P) -> Self {
        MergeBy { a, b, le }
    }
}

impl<A, B, P, Cond> StagedIterator for MergeBy<A, B, P>
where
    A: StagedIterator,
    B: StagedIterator<Item = A::Item>,
    A::Item: CopyType + 'static,
    P: Fn(Var<A::Item>, Var<A::Item>) -> Cond,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = A::Item;
    type Cursor = MergeCursor<A::Cursor, B::Cursor, P>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        let a = self.a.open(ctx);
        let b = self.b.open(ctx);
        MergeCursor {
            a: Lookahead::declare(ctx, a),
            b: Lookahead::declare(ctx, b),
            live_a: ctx.var(true),
            live_b: ctx.var(true),
            le: self.le,
        }
    }
}

/// The cursor of a [`MergeBy`]: two lookahead heads, the smaller one yielded.
pub struct MergeCursor<A: Cursor, B: Cursor, P> {
    a: Lookahead<A>,
    b: Lookahead<B>,
    /// Not yet exhausted. A dead side's `need` stays true, but its refill is
    /// guarded by `live`, so it is never pulled again (the fused contract).
    live_a: Var<bool>,
    live_b: Var<bool>,
    le: P,
}

/// Refill one side of a merge; exhaustion marks it dead instead of ending the
/// stream (the other side may still have elements).
fn refill_or_die<C>(
    ctx: &mut Ctx,
    side: Lookahead<C>,
    live: Var<bool>,
) -> (Var<C::Item>, Var<bool>, C::Close)
where
    C: Cursor,
    C::Item: CopyType + 'static,
{
    let mut out = None;
    ctx.if_then(live, |ctx| {
        ctx.block(|ctx, pulled| {
            ctx.block(|ctx, exhausted| {
                out = Some(side.refill(ctx, exhausted));
                ctx.exit(pulled);
            });
            ctx.store(live, false);
        });
    });
    out.expect("each input's step is staged exactly once")
}

impl<A, B, P, Cond> Cursor for MergeCursor<A, B, P>
where
    A: Cursor,
    B: Cursor<Item = A::Item>,
    A::Item: CopyType + 'static,
    P: Fn(Var<A::Item>, Var<A::Item>) -> Cond,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = A::Item;
    type Close = (A::Close, B::Close);

    /// ```text
    /// if a.live { refill a (exhausted → a.live = false) }; same for b
    /// got: {
    ///   if a.live {
    ///     take_a: { if !b.live → take_a; if le(a.head, b.head) → take_a
    ///               b.need = true; got(b.head) }
    ///     a.need = true; got(a.head)
    ///   }
    ///   if !b.live → done
    ///   b.need = true; b.head
    /// }
    /// ```
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<A::Item>, Self::Close) {
        let (live_a, live_b) = (self.live_a, self.live_b);
        let (ha, need_a, close_a) = refill_or_die(ctx, self.a, live_a);
        let (hb, need_b, close_b) = refill_or_die(ctx, self.b, live_b);
        let le = self.le;
        let item = ctx.join(|ctx, got| {
            ctx.if_then(live_a, |ctx| {
                ctx.block(|ctx, take_a| {
                    ctx.exit_unless(live_b, take_a);
                    ctx.exit_if(le(ha, hb), take_a);
                    ctx.store(need_b, true);
                    ctx.goto(got, hb);
                });
                ctx.store(need_a, true);
                ctx.goto(got, ha);
            });
            ctx.exit_unless(live_b, done);
            ctx.store(need_b, true);
            hb
        });
        (item, (close_a, close_b))
    }
}

// =============================================================================
// IntersectBy
// =============================================================================

/// The elements two sorted streams have in common — a merge join. Built by
/// [`StagedIterator::intersect_by`] / [`intersect`](StagedIterator::intersect).
///
/// `lt` is the strict order both streams are sorted by. Matching is one-to-one
/// in order (multiset intersection): `[1, 1, 2] ∩ [1, 2, 2]` is `[1, 2]`. The
/// left stream's element is yielded. Ends as soon as either side is exhausted.
///
/// The graph use: common neighbours of two nodes (and so triangle counting)
/// from two sorted adjacency lists.
pub struct IntersectBy<A, B, P> {
    a: A,
    b: B,
    lt: P,
}

impl<A, B, P> IntersectBy<A, B, P> {
    pub(crate) fn new(a: A, b: B, lt: P) -> Self {
        IntersectBy { a, b, lt }
    }
}

impl<A, B, P, Cond> StagedIterator for IntersectBy<A, B, P>
where
    A: StagedIterator,
    B: StagedIterator<Item = A::Item>,
    A::Item: CopyType + 'static,
    P: Fn(Var<A::Item>, Var<A::Item>) -> Cond,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = A::Item;
    type Cursor = IntersectCursor<A::Cursor, B::Cursor, P>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        let a = self.a.open(ctx);
        let b = self.b.open(ctx);
        IntersectCursor {
            a: Lookahead::declare(ctx, a),
            b: Lookahead::declare(ctx, b),
            lt: self.lt,
        }
    }
}

/// The cursor of an [`IntersectBy`].
pub struct IntersectCursor<A: Cursor, B: Cursor, P> {
    a: Lookahead<A>,
    b: Lookahead<B>,
    lt: P,
}

impl<A, B, P, Cond> Cursor for IntersectCursor<A, B, P>
where
    A: Cursor,
    B: Cursor<Item = A::Item>,
    A::Item: CopyType + 'static,
    P: Fn(Var<A::Item>, Var<A::Item>) -> Cond,
    Cond: Staged<Out = bool> + 'static,
{
    type Item = A::Item;
    type Close = (A::Close, B::Close);

    /// ```text
    /// again:
    ///   refill a; refill b          ; either exhausted → done
    ///   a_lt = lt(a.head, b.head); b_lt = lt(b.head, a.head)
    ///   a.need = !b_lt; b.need = !a_lt     ; smaller side advances; equal: both
    ///   if a_lt || b_lt → again
    ///   a.head
    /// ```
    /// The bookkeeping is branchless — only the retry branches. Exhaustion of
    /// either side ends the stream, so unlike a merge there is no "live" state:
    /// both refills jump straight to `done`.
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<A::Item>, Self::Close) {
        let (a, b, lt) = (self.a, self.b, self.lt);
        ctx.repeat(|ctx, again| {
            let (ha, need_a, close_a) = a.refill(ctx, done);
            let (hb, need_b, close_b) = b.refill(ctx, done);
            let a_lt = ctx.bind(lt(ha, hb));
            let b_lt = ctx.bind(lt(hb, ha));
            ctx.store(need_a, not(b_lt));
            ctx.store(need_b, not(a_lt));
            ctx.again_if(select(a_lt, true, b_lt), again);
            // Hand out a copy: a consumer that stores to its item must not be
            // able to change the lookahead.
            (ctx.bind(ha), (close_a, close_b))
        })
    }
}
