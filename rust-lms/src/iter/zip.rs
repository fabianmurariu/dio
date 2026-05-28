//! Zip combinator — pairs elements from two sources at the same index.

use crate::func::VarBuilder;
use crate::refer::SRef;
use crate::slice::{Slice, SliceGetUnchecked, SliceLen, SliceRefOps};
use crate::staged::{assign, Const, LetVar, Staged, Var};
use crate::types::{ConstantType, CopyType, StagedType, U64Type, UnitType};

use super::traits::{IndexedSource, IndexedStagedIterator};

// =============================================================================
// Zip combinator
// =============================================================================

/// Combinator that pairs elements from two sources at the same index.
///
/// Created by `indexed_iter.zip(secondary_source)`.
/// Call `for_each` to consume the paired loop.
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
    I: IndexedStagedIterator,
    I::Item: StagedType + ConstantType + CopyType,
    <I::Item as StagedType>::RuntimeValue: Default,
    S: IndexedSource,
    S::Item: StagedType + ConstantType + CopyType,
    <S::Item as StagedType>::RuntimeValue: Default,
{
    /// Drive a loop over `(primary_elem, secondary_elem)` pairs.
    ///
    /// The primary iterator drives loop length; the secondary source is accessed
    /// at the same index. Caller must ensure both sources have equal length.
    ///
    /// # How it works
    ///
    /// The returned staged expression, when codegenned, first initializes a
    /// variable for the secondary element (to its default value), then runs
    /// the primary iterator's loop. Each iteration re-assigns the secondary
    /// element from `secondary_source[i]` before calling the consumer body.
    pub fn for_each<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType> + use<I, S, F, Body>
    where
        F: FnOnce(Var<I::Item>, Var<S::Item>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
        S::GetExpr: Clone,
    {
        // Allocate the secondary element variable. The LetVar is placed BEFORE
        // the loop body (as the first element of the returned tuple) so that
        // Cranelift sees a definition of the variable in the entry block, which
        // satisfies SSA requirements for the loop header phi node.
        let elem2_lv = builder.let_var(Const::<S::Item>::new(Default::default()));
        let elem2_var = elem2_lv.var();

        let sec = self.other;

        // Build the inner loop using the primary iterator's consume_indexed.
        // Each iteration: (1) assign elem2 from secondary, (2) run consumer body.
        let inner = self.iter.consume_indexed(builder, move |idx, elem1| {
            let body = consumer(elem1, elem2_var);
            (assign(elem2_var, sec.clone().get_at(idx)), body)
        });

        // Return (init_elem2, inner_loop) as a tuple.
        // Tuple impls in tuple.rs ensure (A: Staged<Out=Unit>, B: Staged<Out=Unit>)
        // implements Staged<Out=Unit>. Codegen runs A first (declares & sets elem2
        // to default before the loop), then B (the while loop itself).
        (elem2_lv, inner)
    }
}

// =============================================================================
// IndexedSource for Slice Variables
// =============================================================================

impl<'a, T> IndexedSource for Var<SRef<'a, Slice<T>>>
where
    T: StagedType + CopyType + ConstantType,
{
    type Item = T;
    type LenExpr = SliceLen<Self>;
    type GetExpr = SliceGetUnchecked<Self, Var<U64Type>>;

    fn len(&self, builder: &mut VarBuilder) -> LetVar<U64Type, Self::LenExpr> {
        builder.let_var(SliceRefOps::len(*self))
    }

    fn get_at(self, index: Var<U64Type>) -> Self::GetExpr {
        SliceRefOps::get_unchecked(self, index)
    }
}
