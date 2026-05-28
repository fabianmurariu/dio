//! Iterator over slice elements.

use std::marker::PhantomData;

use crate::control::while_loop;
use crate::func::VarBuilder;
use crate::num::{add, lt};
use crate::refer::SRef;
use crate::slice::{Slice, SliceRefOps};
use crate::slice::SliceLen;
use crate::staged::{assign, CompilationContext, Const, LetVar, Staged, Var};
use crate::types::{ConstantType, CopyType, StagedType, U64Type, UnitType};
use cranelift_codegen::ir::Value;

use super::traits::{IndexedStagedIterator, IntoStagedIterator, StagedIterator};

/// Iterator over elements of a staged slice.
pub struct SliceIter<'a, T, S>
where
    T: StagedType,
    S: Staged<Out = SRef<'a, Slice<T>>>,
{
    slice: S,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, S> SliceIter<'a, T, S>
where
    T: StagedType + CopyType + ConstantType,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
{
    pub fn new(slice: S) -> Self {
        SliceIter {
            slice,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, S> StagedIterator for SliceIter<'a, T, S>
where
    T: StagedType + CopyType + ConstantType,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
    T::RuntimeValue: Default,
{
    type Item = T;

    fn consume<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<T>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        let i = builder.let_var(0u64);
        let elem = builder.let_var(Const::<T>::new(<T as StagedType>::RuntimeValue::default()));
        let body = consumer(*elem);

        SliceIterLoop {
            index: i,
            elem_var: elem,
            slice: self.slice,
            body,
        }
    }
}

impl<'a, T, S> IndexedStagedIterator for SliceIter<'a, T, S>
where
    T: StagedType + CopyType + ConstantType + 'a,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
    T::RuntimeValue: Default,
{
    type LenExpr = SliceLen<S>;

    fn len(&self, builder: &mut VarBuilder) -> LetVar<U64Type, Self::LenExpr> {
        builder.let_var(self.slice.clone().len())
    }

    fn consume_indexed<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>, Var<T>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        let i = builder.let_var(0u64);
        let elem = builder.let_var(Const::<T>::new(<T as StagedType>::RuntimeValue::default()));
        let body = consumer(*i, *elem);

        SliceIterLoop {
            index: i,
            elem_var: elem,
            slice: self.slice,
            body,
        }
    }
}

/// Internal: The actual loop structure for slice iteration.
///
/// Stores the InitVar wrappers which contain both the Var and initialization expr.
struct SliceIterLoop<I, E, S, Body> {
    index: I,
    elem_var: E,
    slice: S,
    body: Body,
}

impl<'a, I, E, S, Body, T> Staged for SliceIterLoop<I, E, S, Body>
where
    I: Staged<Out = UnitType> + std::ops::Deref<Target = Var<U64Type>>,
    E: Staged<Out = UnitType> + std::ops::Deref<Target = Var<T>>,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
    Body: Staged<Out = UnitType> + Clone,
    T: StagedType + CopyType + 'a,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Initialize index and element variables
        self.index.codegen(ctx);
        self.elem_var.codegen(ctx);

        // Extract Var references from InitVar via Deref
        let i_var = *self.index.deref();
        let elem_var = *self.elem_var.deref();

        // Generate: while (i < len) { elem = slice[i]; body; i++; }
        while_loop(
            lt(i_var, self.slice.clone().len()),
            (
                assign(elem_var, self.slice.clone().get_unchecked(i_var)),
                self.body.clone(),
                assign(i_var, add(i_var, 1u64)),
            ),
        )
        .codegen(ctx)
    }
}

// =============================================================================
// IntoStagedIterator for Slices
// =============================================================================

impl<'a, T, S> IntoStagedIterator for S
where
    T: StagedType + CopyType + ConstantType + 'a,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
    T::RuntimeValue: Default,
{
    type Iter = SliceIter<'a, T, S>;

    fn staged_iter(self) -> Self::Iter {
        SliceIter::new(self)
    }
}
