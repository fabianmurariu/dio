//! MergeBy combinator — a stable merge of two sorted streams.

use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::staged::{Staged, Var};
use crate::types::CopyType;

use super::traits::{Cursor, StagedIterator};

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
            a: Side::declare(ctx, a),
            b: Side::declare(ctx, b),
            le: self.le,
        }
    }
}

/// One input of a merge: its cursor and its lookahead.
struct Side<C: Cursor> {
    cursor: C,
    /// The current head, valid while `live` and not `need`.
    head: Var<C::Item>,
    /// The head was consumed; pull a new one at the start of the next step.
    need: Var<bool>,
    /// Not yet exhausted.
    live: Var<bool>,
}

impl<C> Side<C>
where
    C: Cursor,
    C::Item: CopyType + 'static,
{
    fn declare(ctx: &mut Ctx, cursor: C) -> Self {
        Side {
            cursor,
            // Never read before the first pull writes it (`need` starts true).
            head: ctx.var(dead::<C::Item>()),
            need: ctx.var(true),
            live: ctx.var(true),
        }
    }

    /// If the head was consumed, pull the next one; on exhaustion mark the side
    /// dead. `need` is only ever true while `live`, so a dead side is never
    /// pulled again (the fused contract).
    fn refill(self, ctx: &mut Ctx) -> (Var<C::Item>, Var<bool>, Var<bool>, C::Close) {
        let Side {
            cursor,
            head,
            need,
            live,
        } = self;
        let mut close = None;
        ctx.if_then(need, |ctx| {
            ctx.block(|ctx, pulled| {
                ctx.block(|ctx, exhausted| {
                    let (x, c) = cursor.next(ctx, exhausted);
                    close = Some(c);
                    ctx.store(head, x);
                    ctx.exit(pulled);
                });
                ctx.store(live, false);
            });
            ctx.store(need, false);
        });
        (
            head,
            need,
            live,
            close.expect("each side's step is staged exactly once"),
        )
    }
}

/// The cursor of a [`MergeBy`]: two lookahead heads, the smaller one yielded.
pub struct MergeCursor<A: Cursor, B: Cursor, P> {
    a: Side<A>,
    b: Side<B>,
    le: P,
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
    /// refill a; refill b            ; pull a side only if its head was consumed
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
    /// Each side's step is staged once (in its refill) — deferring the pull to
    /// the next step is what avoids a priming pull in `open`, which would stage
    /// the step twice.
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<A::Item>, Self::Close) {
        let (ha, need_a, live_a, close_a) = self.a.refill(ctx);
        let (hb, need_b, live_b, close_b) = self.b.refill(ctx);
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
