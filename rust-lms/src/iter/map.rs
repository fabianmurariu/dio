//! Map combinator - transforms each element.

use std::marker::PhantomData;

use crate::func::VarBuilder;
use crate::staged::{assign, Const, LetVar, Staged, Var};
use crate::types::{ConstantType, CopyType, StagedType, U64Type, UnitType};

use super::traits::{IndexedStagedIterator, StagedIterator};

/// Iterator adapter that transforms each element.
///
/// Preserves `IndexedStagedIterator` if inner iterator has it.
pub struct Map<I, F, U> {
    inner: I,
    map_fn: F,
    _phantom: PhantomData<U>,
}

impl<I, F, U> Map<I, F, U> {
    pub(crate) fn new(inner: I, map_fn: F) -> Self {
        Map {
            inner,
            map_fn,
            _phantom: PhantomData,
        }
    }
}

impl<I, F, U, MapOut> StagedIterator for Map<I, F, U>
where
    I: StagedIterator,
    I::Item: StagedType,
    F: Fn(Var<I::Item>) -> MapOut,
    MapOut: Staged<Out = U> + Clone,
    U: StagedType + ConstantType + CopyType,
    <U as StagedType>::RuntimeValue: Default + Clone,
{
    type Item = U;

    fn consume<G, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: G,
    ) -> impl Staged<Out = UnitType>
    where
        G: FnOnce(Var<U>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        // Create variable for mapped value
        let mapped_var = builder.let_var(Const::<U>::new(<U as StagedType>::RuntimeValue::default()));

        // Build outer body using mapped variable
        let outer_body = consumer(*mapped_var);

        // Delegate to inner iterator
        self.inner.consume(builder, move |inner_elem| {
            let mapped_expr = (self.map_fn)(inner_elem);
            let var = *mapped_var; // Extract Var before moving mapped_var
            (
                mapped_var,                // Initialize variable
                assign(var, mapped_expr),  // Compute mapped value
                outer_body,                // Run consumer
            )
        })
    }
}

// Map preserves IndexedStagedIterator
impl<I, F, U, MapOut> IndexedStagedIterator for Map<I, F, U>
where
    I: IndexedStagedIterator,
    I::Item: StagedType,
    F: Fn(Var<I::Item>) -> MapOut,
    MapOut: Staged<Out = U> + Clone,
    U: StagedType + ConstantType + CopyType,
    <U as StagedType>::RuntimeValue: Default + Clone,
{
    type LenExpr = I::LenExpr;

    fn len(&self, builder: &mut VarBuilder) -> LetVar<U64Type, Self::LenExpr> {
        self.inner.len(builder)
    }

    fn consume_indexed<G, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: G,
    ) -> impl Staged<Out = UnitType>
    where
        G: FnOnce(Var<U64Type>, Var<U>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        let mapped_var = builder.let_var(Const::<U>::new(<U as StagedType>::RuntimeValue::default()));

        self.inner.consume_indexed(builder, move |idx, inner_elem| {
            let mapped_expr = (self.map_fn)(inner_elem);
            let var = *mapped_var; // Extract Var before moving mapped_var
            let outer_body = consumer(idx, var);
            (mapped_var, assign(var, mapped_expr), outer_body)
        })
    }
}
