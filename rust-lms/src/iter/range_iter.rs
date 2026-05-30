//! Iterator over numeric ranges `[start, end)` with an optional step.
//!
//! Generic over the two integer element types we support as counters, `u64`
//! and `i64` (see [`RangeStep`]). The step is always forward (`>= 1`); the loop
//! runs while `i < end`, incrementing `i` by `step` each iteration.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::num::{lt, sub, Add, Div, Num, Sub};
use crate::staged::{Const, IntoStaged, Staged, Var};

use super::traits::{IndexedStagedIterator, IntoStagedIterator, StagedIterator};

/// Numeric types usable as a range element/step: `u64` and `i64`.
///
/// Provides the `1` literal needed for the default unit step and for the
/// element-count (`len`) computation. Restricting the range to these two types
/// is intentional — see the design discussion.
pub trait RangeStep: Num {
    fn one() -> Self::RuntimeValue;
}

impl RangeStep for u64 {
    fn one() -> u64 {
        1
    }
}

impl RangeStep for i64 {
    fn one() -> i64 {
        1
    }
}

/// Iterator over `[start, end)` stepping by `step` (defaults to `1`).
///
/// Construct with [`range`] (unit step) or [`range_step`] (explicit step).
pub struct RangeIter<T, Start, End, Step> {
    start: Start,
    end: End,
    step: Step,
    _phantom: PhantomData<T>,
}

/// Create a range iterator over `[start, end)` with unit step.
pub fn range<T, S, E>(start: S, end: E) -> RangeIter<T, S::Staged, E::Staged, Const<T>>
where
    T: RangeStep,
    S: IntoStaged<T>,
    E: IntoStaged<T>,
{
    RangeIter {
        start: start.into_staged(),
        end: end.into_staged(),
        step: Const::<T>::new(T::one()),
        _phantom: PhantomData,
    }
}

/// Create a range iterator over `[start, end)` stepping by `step` (`>= 1`).
pub fn range_step<T, S, E, P>(
    start: S,
    end: E,
    step: P,
) -> RangeIter<T, S::Staged, E::Staged, P::Staged>
where
    T: RangeStep,
    S: IntoStaged<T>,
    E: IntoStaged<T>,
    P: IntoStaged<T>,
{
    RangeIter {
        start: start.into_staged(),
        end: end.into_staged(),
        step: step.into_staged(),
        _phantom: PhantomData,
    }
}

impl<T, Start, End, Step> StagedIterator for RangeIter<T, Start, End, Step>
where
    T: RangeStep,
    Start: Staged<Out = T> + Clone + 'static,
    End: Staged<Out = T> + Clone + 'static,
    Step: Staged<Out = T> + Clone + 'static,
{
    type Item = T;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<T>) + 'static,
    {
        // i starts at `start`, advances by `step` until it reaches `end`.
        let i = ctx.var(self.start.clone());
        let end = self.end;
        let step = self.step;

        ctx.while_loop(lt(i, end), move |ctx| {
            consumer(ctx, i);
            ctx.store(i, i + step);
        });
    }
}

// `IndexedStagedIterator` (which powers `zip`) requires the length to be a
// `u64`. We therefore only provide it for `u64` ranges — the element count of
// an `i64` range would itself be `i64` and there is no integer cast op yet.
// `i64` ranges remain full `StagedIterator`s (sum/fold/map/filter/…).
impl<Start, End, Step> IndexedStagedIterator for RangeIter<u64, Start, End, Step>
where
    Start: Staged<Out = u64> + Clone + 'static,
    End: Staged<Out = u64> + Clone + 'static,
    Step: Staged<Out = u64> + Clone + 'static,
{
    // Number of elements with a forward step: ceil((end - start) / step),
    // computed as (end - start + step - 1) / step. Reduces to `end - start`
    // when `step == 1` (Cranelift folds the constants).
    type LenExpr = Div<Add<Sub<End, Start>, Sub<Step, Const<u64>>>, Step>;

    fn len(&self) -> Self::LenExpr {
        let span = sub(self.end.clone(), self.start.clone());
        let step_minus_1 = sub(self.step.clone(), Const::<u64>::new(1));
        (span + step_minus_1) / self.step.clone()
    }
}

impl<T, Start, End, Step> IntoStagedIterator for RangeIter<T, Start, End, Step>
where
    T: RangeStep,
    Start: Staged<Out = T> + Clone + 'static,
    End: Staged<Out = T> + Clone + 'static,
    Step: Staged<Out = T> + Clone + 'static,
{
    type Iter = Self;

    fn staged_iter(self) -> Self::Iter {
        self
    }
}
