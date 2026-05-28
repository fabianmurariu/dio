//! Iterator over integer ranges.

use crate::control::while_loop;
use crate::func::VarBuilder;
use crate::num::{add, lt, sub, Sub};
use crate::staged::{assign, CompilationContext, IntoStaged, LetVar, Staged, Var};
use crate::types::{U64Type, UnitType};
use cranelift_codegen::ir::Value;

use super::traits::{IndexedStagedIterator, StagedIterator};

/// Iterator over a range of integers [start, end).
pub struct RangeIter<Start, End> {
    start: Start,
    end: End,
}

/// Create a range iterator.
pub fn range<S, E>(start: S, end: E) -> RangeIter<S::Staged, E::Staged>
where
    S: IntoStaged<U64Type>,
    E: IntoStaged<U64Type>,
{
    RangeIter {
        start: start.into_staged(),
        end: end.into_staged(),
    }
}

impl<Start, End> StagedIterator for RangeIter<Start, End>
where
    Start: Staged<Out = U64Type> + Clone,
    End: Staged<Out = U64Type> + Clone,
{
    type Item = U64Type;

    fn consume<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        // Initialize index to start value
        let i = builder.let_var(0u64); // Will assign to start in loop struct
        let body = consumer(*i);

        RangeIterLoop {
            index: i,
            start: self.start,
            end: self.end,
            body,
        }
    }
}

impl<Start, End> IndexedStagedIterator for RangeIter<Start, End>
where
    Start: Staged<Out = U64Type> + Clone,
    End: Staged<Out = U64Type> + Clone,
{
    type LenExpr = Sub<End, Start>;

    fn len(&self, builder: &mut VarBuilder) -> LetVar<U64Type, Self::LenExpr> {
        builder.let_var(sub(self.end.clone(), self.start.clone()))
    }

    fn consume_indexed<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>, Var<U64Type>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        let i = builder.let_var(0u64);
        // For range, index and value are the same
        let body = consumer(*i, *i);

        RangeIterLoop {
            index: i,
            start: self.start,
            end: self.end,
            body,
        }
    }
}

struct RangeIterLoop<I, Start, End, Body> {
    index: I,
    start: Start,
    end: End,
    body: Body,
}

impl<I, Start, End, Body> Staged for RangeIterLoop<I, Start, End, Body>
where
    I: Staged<Out = UnitType> + std::ops::Deref<Target = Var<U64Type>>,
    Start: Staged<Out = U64Type> + Clone,
    End: Staged<Out = U64Type> + Clone,
    Body: Staged<Out = UnitType> + Clone,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Initialize index
        self.index.codegen(ctx);
        let i_var = *self.index.deref();

        // Assign start value to index
        assign(i_var, self.start.clone()).codegen(ctx);

        // Generate: while (i < end) { body; i++; }
        while_loop(
            lt(i_var, self.end.clone()),
            (self.body.clone(), assign(i_var, add(i_var, 1u64))),
        )
        .codegen(ctx)
    }
}
