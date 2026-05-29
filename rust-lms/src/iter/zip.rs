//! Zip combinator — pairs elements from two sources at the same index.

use crate::func::Ctx;
use crate::num::{add, lt};
use crate::staged::{Const, Var};
use crate::types::{ConstantType, CopyType, StagedType};

use super::traits::{IndexedSource, IndexedStagedIterator};

/// Combinator that pairs elements from two sources at the same 0-based index.
///
/// Created by `indexed_iter.zip(secondary)`. Use `for_each` to drive the loop.
pub struct Zip<I, S> {
    pub(crate) iter: I,
    pub(crate) other: S,
}

impl<I, S> Zip<I, S> {
    pub(crate) fn new(iter: I, other: S) -> Self {
        Zip { iter, other }
    }
}

impl<I, S> Zip<I, S>
where
    I: IndexedStagedIterator + IndexedSource + 'static,
    <I as IndexedSource>::Item: StagedType + CopyType + ConstantType + 'static,
    <<I as IndexedSource>::Item as StagedType>::RuntimeValue: Default,
    S: IndexedSource + 'static,
    <S as IndexedSource>::Item: StagedType + CopyType + ConstantType + 'static,
    <<S as IndexedSource>::Item as StagedType>::RuntimeValue: Default,
    <I as IndexedSource>::GetExpr: 'static,
    <S as IndexedSource>::GetExpr: 'static,
{
    /// Drive a loop over `(primary_elem, secondary_elem)` pairs.
    ///
    /// The primary source's length drives the loop; the secondary is accessed
    /// at the same 0-based index. Caller must ensure equal-length sources.
    pub fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<<I as IndexedSource>::Item>, Var<<S as IndexedSource>::Item>)
            + 'static,
    {
        let i = ctx.var(0u64);
        let elem1 = ctx.var(Const::<<I as IndexedSource>::Item>::new(Default::default()));
        let elem2 = ctx.var(Const::<<S as IndexedSource>::Item>::new(Default::default()));
        let prim_len = IndexedSource::len(&self.iter);
        let prim = self.iter;
        let sec = self.other;

        ctx.while_loop(lt(i, prim_len), move |ctx| {
            ctx.store(elem1, IndexedSource::get_at(prim.clone(), i));
            ctx.store(elem2, IndexedSource::get_at(sec.clone(), i));
            consumer(ctx, elem1, elem2);
            ctx.store(i, add(i, 1u64));
        });
    }
}
