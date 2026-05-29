//! Iterator over integer ranges [start, end).

use crate::func::Ctx;
use crate::num::{add, lt, sub, Sub};
use crate::staged::{IntoStaged, Var};
use crate::types::U64Type;

use super::traits::{IndexedStagedIterator, StagedIterator};

/// Iterator over a range of u64 values [start, end).
pub struct RangeIter<Start, End> {
    start: Start,
    end: End,
}

/// Create a range iterator over [start, end).
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
    Start: crate::staged::Staged<Out = U64Type> + Clone + 'static,
    End: crate::staged::Staged<Out = U64Type> + Clone + 'static,
{
    type Item = U64Type;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<U64Type>) + 'static,
    {
        // i starts at `start`, increments to `end`
        let i = ctx.var(self.start.clone());
        let end = self.end;

        ctx.while_loop(lt(i, end), move |ctx| {
            consumer(ctx, i);
            ctx.store(i, add(i, 1u64));
        });
    }
}

impl<Start, End> IndexedStagedIterator for RangeIter<Start, End>
where
    Start: crate::staged::Staged<Out = U64Type> + Clone + 'static,
    End: crate::staged::Staged<Out = U64Type> + Clone + 'static,
{
    type LenExpr = Sub<End, Start>;

    fn len(&self) -> Self::LenExpr {
        sub(self.end.clone(), self.start.clone())
    }
}
