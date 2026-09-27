//! Iterator over numeric ranges `[start, end)` with an optional step.
//!
//! Generic over the two integer element types we support as counters, `u64`
//! and `i64` (see [`RangeStep`]). The step is always forward (`>= 1`); the loop
//! runs while `i < end`, incrementing `i` by `step` each iteration.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::label::Label;
use crate::num::{
    Add, Div, Lt, Mul, Num, Select, Sub, add, div, ge, int_cast, lt, mul, select, sub,
};
use crate::staged::{Const, IntoStaged, Staged, Var};

use super::traits::{
    Cursor, IndexedSource, IndexedStagedIterator, IntoStagedIterator, StagedIterator,
};

/// Numeric types usable as a range element/step: `u64` and `i64`.
///
/// Provides the `1` literal for the default unit step, and the two stage-0
/// hooks that give a range random access: its element count and its `index`-th
/// element. Restricting the range to these two types is intentional.
pub trait RangeStep: Num {
    fn one() -> Self::RuntimeValue;

    /// Element count of `[start, end)` stepping by `step`: 0 when the range is
    /// empty (`start >= end`), otherwise `ceil((end - start) / step)`.
    fn count(ctx: &mut Ctx, start: Var<Self>, end: Var<Self>, step: Var<Self>) -> Var<u64>;

    /// The `index`-th element: `start + index * step`.
    fn nth(ctx: &mut Ctx, start: Var<Self>, step: Var<Self>, index: Var<u64>) -> Var<Self>;
}

impl RangeStep for u64 {
    fn one() -> u64 {
        1
    }

    fn count(ctx: &mut Ctx, start: Var<u64>, end: Var<u64>, step: Var<u64>) -> Var<u64> {
        // Guarded: `end - start` would wrap for an empty range.
        let span = add(sub(end, start), sub(step, 1u64));
        ctx.bind(select(lt(start, end), div(span, step), 0u64))
    }

    fn nth(ctx: &mut Ctx, start: Var<u64>, step: Var<u64>, index: Var<u64>) -> Var<u64> {
        ctx.bind(add(start, mul(index, step)))
    }
}

impl RangeStep for i64 {
    fn one() -> i64 {
        1
    }

    fn count(ctx: &mut Ctx, start: Var<i64>, end: Var<i64>, step: Var<i64>) -> Var<u64> {
        // Non-negative whenever `start < end`, so the cast is exact there.
        let span = add(sub(end, start), sub(step, 1i64));
        let n = int_cast::<u64, i64, _>(div(span, step));
        ctx.bind(select(lt(start, end), n, 0u64))
    }

    fn nth(ctx: &mut Ctx, start: Var<i64>, step: Var<i64>, index: Var<u64>) -> Var<i64> {
        ctx.bind(add(start, mul(int_cast::<i64, u64, _>(index), step)))
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

// Hand-written so the derive does not demand `T: Clone` — `T` is a staged
// *marker*, not a runtime value.
impl<T, Start: Clone, End: Clone, Step: Clone> Clone for RangeIter<T, Start, End, Step> {
    fn clone(&self) -> Self {
        RangeIter {
            start: self.start.clone(),
            end: self.end.clone(),
            step: self.step.clone(),
            _phantom: PhantomData,
        }
    }
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
    type Cursor = RangeCursor<T>;

    fn open(self, ctx: &mut Ctx) -> RangeCursor<T> {
        // Bounds are evaluated once, like `std`'s `Range`: a body that changes
        // the variable the range was built from does not move its end.
        let start = ctx.bind(self.start);
        let end = ctx.bind(self.end);
        let step = ctx.bind(self.step);
        let counter = ctx.var(start);
        RangeCursor {
            start,
            end,
            step,
            counter,
        }
    }
}

/// The cursor of a [`RangeIter`]: a counter from `start` towards `end`.
pub struct RangeCursor<T: RangeStep> {
    start: Var<T>,
    end: Var<T>,
    step: Var<T>,
    counter: Var<T>,
}

impl<T: RangeStep> Cursor for RangeCursor<T> {
    type Item = T;
    type Close = ();

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<T>, ()) {
        let i = self.counter;
        ctx.exit_if(ge(i, self.end), done);
        // Hand out a *copy* of the counter: a consumer that stores to its item
        // must not be able to change the iteration. Single-def, so the frontend
        // resolves it with no extra instruction.
        let item = ctx.bind(i);
        ctx.store(i, add(i, self.step));
        (item, ())
    }

    fn indexed_len(&mut self, ctx: &mut Ctx) -> Option<Var<u64>> {
        Some(T::count(ctx, self.start, self.end, self.step))
    }

    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<T>, ()) {
        (T::nth(ctx, self.start, self.step, index), ())
    }
}

/// The element count of a `u64` range (see `IndexedStagedIterator::len`).
pub type RangeLen<Start, End, Step> =
    Select<Lt<Start, End>, Div<Add<Sub<End, Start>, Sub<Step, Const<u64>>>, Step>, Const<u64>>;

// The *typed* random-access traits (`IndexedStagedIterator`/`IndexedSource`,
// behind `rev`) are provided for `u64` ranges, whose element and length share a
// type. Every range — `i64` included — has random access through its cursor
// (`RangeStep::count`/`nth`), which is what `zip` uses.
impl<Start, End, Step> IndexedStagedIterator for RangeIter<u64, Start, End, Step>
where
    Start: Staged<Out = u64> + Clone + 'static,
    End: Staged<Out = u64> + Clone + 'static,
    Step: Staged<Out = u64> + Clone + 'static,
{
    // Number of elements with a forward step: ceil((end - start) / step),
    // computed as (end - start + step - 1) / step, and 0 for an empty range
    // (where `end - start` would wrap). Reduces to `end - start` when
    // `step == 1` (Cranelift folds the constants).
    type LenExpr = RangeLen<Start, End, Step>;

    fn len(&self) -> Self::LenExpr {
        let span = sub(self.end.clone(), self.start.clone());
        let step_minus_1 = sub(self.step.clone(), Const::<u64>::new(1));
        let n = (span + step_minus_1) / self.step.clone();
        select(lt(self.start.clone(), self.end.clone()), n, 0u64)
    }
}

/// Random access into a range: element `k` is `start + k * step`. Makes a range
/// usable as a `zip` source and as the input to `rev`.
impl<Start, End, Step> IndexedSource for RangeIter<u64, Start, End, Step>
where
    Start: Staged<Out = u64> + Clone + 'static,
    End: Staged<Out = u64> + Clone + 'static,
    Step: Staged<Out = u64> + Clone + 'static,
{
    type Item = u64;
    type LenExpr = RangeLen<Start, End, Step>;
    type GetExpr = Add<Start, Mul<Var<u64>, Step>>;

    fn count(&self) -> Self::LenExpr {
        IndexedStagedIterator::len(self)
    }

    unsafe fn get_at(self, index: Var<u64>) -> Self::GetExpr {
        // Total for any `index`; the `unsafe` contract is about staying inside
        // `count`, which only affects whether the value is *in* the range.
        add(self.start, mul(index, self.step))
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
