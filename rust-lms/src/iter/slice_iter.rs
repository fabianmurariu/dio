//! Iterator over slice elements.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::num::{add, lt};
use crate::refer::SRef;
use crate::slice::{Slice, SliceGetUnchecked, SliceLen, SliceRefOps};
use crate::staged::{Const, Var};
use crate::types::{ConstantType, CopyType, StagedType, U64Type};

use super::traits::{IndexedSource, IndexedStagedIterator, IntoStagedIterator, StagedIterator};

/// Iterator over elements of a staged slice reference.
#[derive(Clone)]
pub struct SliceIter<'a, T, S>
where
    T: StagedType,
    S: crate::staged::Staged<Out = SRef<'a, Slice<T>>>,
{
    pub(crate) slice: S,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, S> SliceIter<'a, T, S>
where
    T: StagedType + CopyType + ConstantType + 'static,
    S: crate::staged::Staged<Out = SRef<'a, Slice<T>>> + Clone + 'static,
    'a: 'static,
{
    pub fn new(slice: S) -> Self {
        SliceIter { slice, _phantom: PhantomData }
    }
}

impl<'a, T, S> StagedIterator for SliceIter<'a, T, S>
where
    'a: 'static,
    T: StagedType + CopyType + ConstantType + 'static,
    S: crate::staged::Staged<Out = SRef<'a, Slice<T>>> + Clone + 'static,
    T::RuntimeValue: Default,
{
    type Item = T;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<T>) + 'static,
    {
        let i    = ctx.var(0u64);
        let elem = ctx.var(Const::<T>::new(Default::default()));
        let slice = self.slice;

        ctx.while_loop(lt(i, slice.clone().len()), move |ctx| {
            ctx.assign(elem, SliceRefOps::get_unchecked(slice.clone(), i));
            consumer(ctx, elem);
            ctx.assign(i, add(i, 1u64));
        });
    }
}

impl<'a, T, S> IndexedStagedIterator for SliceIter<'a, T, S>
where
    'a: 'static,
    T: StagedType + CopyType + ConstantType + 'static,
    S: crate::staged::Staged<Out = SRef<'a, Slice<T>>> + Clone + 'static,
    T::RuntimeValue: Default,
{
    type LenExpr = SliceLen<S>;

    fn len(&self) -> Self::LenExpr {
        self.slice.clone().len()
    }
}

impl<'a, T, S> IndexedSource for SliceIter<'a, T, S>
where
    'a: 'static,
    T: StagedType + CopyType + ConstantType + 'static,
    S: crate::staged::Staged<Out = SRef<'a, Slice<T>>> + Clone + 'static,
    T::RuntimeValue: Default,
{
    type Item = T;
    type LenExpr = SliceLen<S>;
    type GetExpr = SliceGetUnchecked<S, Var<U64Type>>;

    fn len(&self) -> Self::LenExpr {
        self.slice.clone().len()
    }

    fn get_at(self, index: Var<U64Type>) -> Self::GetExpr {
        SliceRefOps::get_unchecked(self.slice, index)
    }
}

// =============================================================================
// IndexedSource for Slice Variable references (secondary source in zip)
// =============================================================================

impl<'a, T> IndexedSource for Var<SRef<'a, Slice<T>>>
where
    'a: 'static,
    T: StagedType + CopyType + ConstantType + 'static,
{
    type Item = T;
    type LenExpr = SliceLen<Self>;
    type GetExpr = SliceGetUnchecked<Self, Var<U64Type>>;

    fn len(&self) -> Self::LenExpr {
        SliceRefOps::len(*self)
    }

    fn get_at(self, index: Var<U64Type>) -> Self::GetExpr {
        SliceRefOps::get_unchecked(self, index)
    }
}

// =============================================================================
// IntoStagedIterator
// =============================================================================

impl<'a, T, S> IntoStagedIterator for S
where
    'a: 'static,
    T: StagedType + CopyType + ConstantType + 'static,
    S: crate::staged::Staged<Out = SRef<'a, Slice<T>>> + Clone + 'static,
    T::RuntimeValue: Default,
{
    type Iter = SliceIter<'a, T, S>;

    fn staged_iter(self) -> Self::Iter {
        SliceIter::new(self)
    }
}
