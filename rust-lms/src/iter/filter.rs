//! Filter combinator - keeps only elements matching a predicate.

use crate::control::if_then;
use crate::func::VarBuilder;
use crate::staged::{Staged, Var};
use crate::types::{BoolType, CopyType, UnitType};

use super::traits::StagedIterator;

/// Iterator adapter that keeps only elements matching a predicate.
///
/// **Does NOT implement `IndexedStagedIterator`** - filter breaks index correspondence.
pub struct Filter<I, P> {
    inner: I,
    predicate: P,
}

impl<I, P> Filter<I, P> {
    pub(crate) fn new(inner: I, predicate: P) -> Self {
        Filter { inner, predicate }
    }
}

impl<I, P, Cond> StagedIterator for Filter<I, P>
where
    I: StagedIterator,
    I::Item: CopyType,
    P: Fn(Var<I::Item>) -> Cond,
    Cond: Staged<Out = BoolType> + Clone,
{
    type Item = I::Item;

    fn consume<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType> + use<I, P, Cond, F, Body>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        self.inner.consume(builder, move |elem| {
            let condition = (self.predicate)(elem);
            let body = consumer(elem);
            if_then(condition, body) // Only execute if predicate passes
        })
    }
}

// Filter does NOT implement IndexedStagedIterator - this is intentional!
