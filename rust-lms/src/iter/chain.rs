//! Chain combinator — one iterator's elements, then another's.

use crate::control::not;
use crate::func::Ctx;
use crate::label::Label;
use crate::staged::Var;

use super::traits::{Cursor, StagedIterator};

/// All elements of the first iterator, then all elements of the second.
///
/// Built by [`StagedIterator::chain`]. Both sides are opened up front (like
/// `std`'s `Chain`, which holds both iterators); a phase flag picks the side at
/// runtime, and the consumer is emitted once.
pub struct Chain<A, B> {
    a: A,
    b: B,
}

impl<A, B> Chain<A, B> {
    pub(crate) fn new(a: A, b: B) -> Self {
        Chain { a, b }
    }
}

impl<A, B> StagedIterator for Chain<A, B>
where
    A: StagedIterator,
    B: StagedIterator<Item = A::Item>,
    A::Item: 'static,
{
    type Item = A::Item;
    type Cursor = ChainCursor<A::Cursor, B::Cursor>;

    fn open(self, ctx: &mut Ctx) -> Self::Cursor {
        let a = self.a.open(ctx);
        let b = self.b.open(ctx);
        ChainCursor {
            a,
            b,
            in_b: ctx.var(false),
        }
    }
}

/// The cursor of a [`Chain`].
pub struct ChainCursor<A, B> {
    a: A,
    b: B,
    /// Latches to `true` once `a` is exhausted — `a`'s step never runs again,
    /// which is the fused contract `Cursor::next` asks of its caller.
    in_b: Var<bool>,
}

impl<A, B> Cursor for ChainCursor<A, B>
where
    A: Cursor,
    B: Cursor<Item = A::Item>,
    A::Item: 'static,
{
    type Item = A::Item;
    type Close = (A::Close, B::Close);

    /// ```text
    /// got: {
    ///   if !in_b {
    ///     a_done: { x = a.next() (exhausted → a_done); got(x) }
    ///     in_b = true
    ///   }
    ///   b.next()                  ; exhausted → done; falls through to got
    /// }
    /// ```
    /// `a` gets a private exhaustion label meaning "switch to `b`"; only `b`
    /// sees the real `done`. Neither side knows it is chained.
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<A::Item>, Self::Close) {
        let (a, b, in_b) = (self.a, self.b, self.in_b);
        let (mut close_a, mut close_b) = (None, None);
        let item = ctx.join(|ctx, got| {
            ctx.if_then(not(in_b), |ctx| {
                ctx.block(|ctx, a_done| {
                    let (x, c) = a.next(ctx, a_done);
                    close_a = Some(c);
                    ctx.goto(got, x);
                });
                ctx.store(in_b, true);
            });
            let (y, c) = b.next(ctx, done);
            close_b = Some(c);
            y
        });
        let staged_once = "each side's step is staged exactly once";
        (
            item,
            (close_a.expect(staged_once), close_b.expect(staged_once)),
        )
    }
}
