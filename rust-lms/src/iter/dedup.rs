//! Drop repeats of a sorted stream: [`DedupByKey`].

use crate::func::Ctx;
use crate::label::{Label, zeroed};
use crate::num::{Ordered, eq};
use crate::staged::{Staged, Var};
use crate::types::{CopyType, StagedType};

use super::traits::{Cursor, StagedIterator};

/// Iterator adapter that drops each element whose key equals the previous element's.
/// On a stream sorted by that key, every key comes once. Built by
/// [`StagedIterator::dedup_by_key`].
pub struct DedupByKey<I, F> {
    inner: I,
    key: F,
}

impl<I, F> DedupByKey<I, F> {
    pub(crate) fn new(inner: I, key: F) -> Self {
        DedupByKey { inner, key }
    }
}

impl<I, F, KE, K> StagedIterator for DedupByKey<I, F>
where
    I: StagedIterator,
    I::Item: CopyType + 'static,
    F: Fn(&Ctx, Var<I::Item>) -> KE,
    KE: Staged<Out = K> + 'static,
    K: Ordered,
{
    type Item = I::Item;
    type Cursor = DedupCursor<I::Cursor, F, K>;

    fn open(self, ctx: &Ctx) -> Self::Cursor {
        DedupCursor {
            inner: self.inner.open(ctx),
            key: self.key,
            last: ctx.var(zeroed::<K>()),
            seen: ctx.var(false),
        }
    }
}

/// The cursor of a [`DedupByKey`]: the last key passed on, once there is one.
pub struct DedupCursor<C, F, K: StagedType> {
    inner: C,
    key: F,
    last: Var<K>,
    seen: Var<bool>,
}

impl<C, F, KE, K> Cursor for DedupCursor<C, F, K>
where
    C: Cursor,
    C::Item: CopyType + 'static,
    F: Fn(&Ctx, Var<C::Item>) -> KE,
    KE: Staged<Out = K> + 'static,
    K: Ordered,
{
    type Item = C::Item;
    type Close = C::Close;

    fn next(self, ctx: &Ctx, done: Label<'_>) -> (Var<C::Item>, C::Close) {
        let (inner, key, last, seen) = (self.inner, self.key, self.last, self.seen);
        ctx.repeat(|ctx, again| {
            let (x, close) = inner.next(ctx, done);
            let k = ctx.bind(key(ctx, x));
            ctx.if_then(seen, |ctx| ctx.again_if(eq(k, last), again));
            ctx.store(last, k);
            ctx.store(seen, true);
            (x, close)
        })
    }
}
