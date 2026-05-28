//! Zip combinator - combines two sources element-wise.
//!
//! TODO: Zip implementation is temporarily disabled due to lifetime capture
//! issues with opaque return types. The API is designed but needs a concrete
//! struct-based approach similar to SliceIterLoop to avoid the issue.

use crate::func::VarBuilder;
use crate::refer::SRef;
use crate::slice::{Slice, SliceGetUnchecked, SliceLen, SliceRefOps};
use crate::staged::{LetVar, Staged, Var};
use crate::types::{ConstantType, CopyType, StagedType, U64Type};

use super::traits::IndexedSource;

/// Combinator that pairs elements from two sources using the same index.
///
/// Unlike regular iterators, Zip produces pairs and has its own consume methods
/// that take two-argument closures.
pub struct Zip<I, S> {
    iter: I,   // Primary iterator (controls the loop)
    other: S,  // Secondary source (accessed by index)
}

impl<I, S> Zip<I, S> {
    pub(crate) fn new(iter: I, other: S) -> Self {
        Zip { iter, other }
    }
}

// TODO: Zip methods disabled pending lifetime capture fix
// The issue is that `consume_indexed` returns `impl Staged` which captures
// lifetimes that don't appear in our method signatures.

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
        // Use SliceRefOps::len which is implemented for Var<SRef<Slice<T>>>
        builder.let_var(SliceRefOps::len(*self))
    }

    fn get_unchecked(self, index: Var<U64Type>) -> Self::GetExpr {
        // Use SliceRefOps::get_unchecked which is implemented for Var<SRef<Slice<T>>>
        SliceRefOps::get_unchecked(self, index)
    }
}
