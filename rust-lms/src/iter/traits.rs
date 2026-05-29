//! Core traits for staged iteration.

use crate::func::Ctx;
use crate::num::{add, gt, lt, SupportsAdd, SupportsComparison};
use crate::staged::{Const, Staged, Var};
use crate::types::{BoolType, ConstantType, CopyType, StagedType, U64Type};

use super::{Filter, Map, Zip};

// =============================================================================
// MinMax sentinels for min/max reductions
// =============================================================================

pub trait MinMax: Copy {
    fn min_sentinel() -> Self; // smallest value, used as max-reduction start
    fn max_sentinel() -> Self; // largest value, used as min-reduction start
}

impl MinMax for i64 {
    fn min_sentinel() -> Self {
        i64::MIN
    }
    fn max_sentinel() -> Self {
        i64::MAX
    }
}
impl MinMax for u64 {
    fn min_sentinel() -> Self {
        u64::MIN
    }
    fn max_sentinel() -> Self {
        u64::MAX
    }
}
impl MinMax for f64 {
    fn min_sentinel() -> Self {
        f64::NEG_INFINITY
    }
    fn max_sentinel() -> Self {
        f64::INFINITY
    }
}
impl MinMax for i32 {
    fn min_sentinel() -> Self {
        i32::MIN
    }
    fn max_sentinel() -> Self {
        i32::MAX
    }
}
impl MinMax for u32 {
    fn min_sentinel() -> Self {
        u32::MIN
    }
    fn max_sentinel() -> Self {
        u32::MAX
    }
}

// =============================================================================
// StagedIterator
// =============================================================================

/// A staged iterator that generates imperative loop code.
///
/// The consumer closure is called **once at staging time** to build the loop
/// body. Call side-effecting methods on the `Ctx` it receives (`ctx.assign`,
/// `ctx.if_then`, etc.) to emit per-iteration code.
///
/// # Example
/// ```ignore
/// let sum = ctx.var(0.0f64);
/// arr.staged_iter().for_each(ctx, move |ctx, elem| {
///     ctx.assign(sum, add(sum, elem));
/// });
/// // `sum` now holds the accumulated result
/// ```
pub trait StagedIterator: Sized {
    type Item: StagedType;

    /// Drive a loop over all elements.
    ///
    /// `consumer` is called once at staging time; it emits the per-element
    /// body via the `Ctx` it receives. No `Clone` constraint required.
    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static;

    // =========================================================================
    // Combinators
    // =========================================================================

    fn map<U, F, MapOut>(self, f: F) -> Map<Self, F, U>
    where
        U: StagedType,
        F: Fn(Var<Self::Item>) -> MapOut,
        MapOut: Staged<Out = U>,
    {
        Map::new(self, f)
    }

    fn filter<P, Cond>(self, p: P) -> Filter<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = BoolType>,
    {
        Filter::new(self, p)
    }

    // =========================================================================
    // Terminal operations
    // =========================================================================

    /// Sum all elements. Accumulator starts at `T::RuntimeValue::default()` (zero).
    fn sum(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: StagedType + ConstantType + CopyType + SupportsAdd + 'static,
        <Self::Item as StagedType>::RuntimeValue: Default,
    {
        let acc = ctx.var(Const::<Self::Item>::new(Default::default()));
        self.for_each(ctx, move |ctx, elem| {
            ctx.store(acc, add::<Self::Item, _, _>(acc, elem));
        });
        acc
    }

    /// Count elements passing through (including any upstream filter).
    fn count(self, ctx: &mut Ctx) -> Var<U64Type>
    where
        Self::Item: 'static,
    {
        let acc = ctx.var(0u64);
        self.for_each(ctx, move |ctx, _elem| {
            ctx.store(acc, add::<U64Type, _, _>(acc, 1u64));
        });
        acc
    }

    /// Find the minimum element. Starts at the type's maximum sentinel.
    fn min(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: StagedType + ConstantType + CopyType + SupportsComparison + 'static,
        <Self::Item as StagedType>::RuntimeValue: MinMax,
    {
        let sentinel = <Self::Item as StagedType>::RuntimeValue::max_sentinel();
        let acc = ctx.var(Const::<Self::Item>::new(sentinel));
        self.for_each(ctx, move |ctx, elem| {
            ctx.if_then(lt::<Self::Item, _, _>(elem, acc), move |ctx| {
                ctx.store(acc, elem);
            });
        });
        acc
    }

    /// Find the maximum element. Starts at the type's minimum sentinel.
    fn max(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: StagedType + ConstantType + CopyType + SupportsComparison + 'static,
        <Self::Item as StagedType>::RuntimeValue: MinMax,
    {
        let sentinel = <Self::Item as StagedType>::RuntimeValue::min_sentinel();
        let acc = ctx.var(Const::<Self::Item>::new(sentinel));
        self.for_each(ctx, move |ctx, elem| {
            ctx.if_then(gt::<Self::Item, _, _>(elem, acc), move |ctx| {
                ctx.store(acc, elem);
            });
        });
        acc
    }

    /// Fold with a user-managed accumulator.
    ///
    /// `acc` is any `Copy + 'static` value — typically a `Var<T>` or a tuple
    /// of `Var`s you declared via `ctx.var()` before calling `fold`.
    /// `f` receives `(ctx, acc, elem)` and emits the per-iteration update.
    ///
    /// After `fold` returns the accumulator vars hold the final result.
    ///
    /// # Example — count + sum simultaneously
    /// ```ignore
    /// let count = ctx.var(0u64);
    /// let sum   = ctx.var(0.0f64);
    /// slice.staged_iter().fold(ctx, (count, sum), |ctx, (c, s), elem| {
    ///     ctx.assign(c, add(c, 1u64));
    ///     ctx.assign(s, add(s, elem));
    /// });
    /// ```
    fn fold<Acc, F>(self, ctx: &mut Ctx, acc: Acc, f: F)
    where
        Acc: Copy + 'static,
        F: FnOnce(&mut Ctx, Acc, Var<Self::Item>) + 'static,
    {
        self.for_each(ctx, move |ctx, elem| {
            f(ctx, acc, elem);
        });
    }
}

// =============================================================================
// IndexedStagedIterator
// =============================================================================

/// A staged iterator that tracks element positions, enabling `zip`.
pub trait IndexedStagedIterator: StagedIterator {
    /// The type of the length expression (e.g. `SliceLen<S>`, `Sub<End, Start>`).
    type LenExpr: Staged<Out = U64Type> + Clone + 'static;

    /// Return the number of elements as a staged expression.
    fn len(&self) -> Self::LenExpr;

    /// Zip this (indexed) iterator with a secondary random-access source.
    ///
    /// Both sources are accessed at the same 0-based position each iteration.
    /// The primary source's length drives the loop; the caller must ensure the
    /// secondary has at least that many elements.
    ///
    /// Only available for iterators that also implement `IndexedSource` (slice
    /// iterators and slice variable references).
    fn zip<S>(self, other: S) -> Zip<Self, S>
    where
        Self: IndexedSource,
        S: IndexedSource,
    {
        Zip::new(self, other)
    }
}

// =============================================================================
// IndexedSource
// =============================================================================

/// A random-access data source usable as the secondary input to `zip`, or as
/// the primary when it also implements `IndexedStagedIterator`.
pub trait IndexedSource: Clone + 'static {
    type Item: StagedType;
    type LenExpr: Staged<Out = U64Type> + Clone + 'static;
    type GetExpr: Staged<Out = Self::Item> + 'static;

    fn len(&self) -> Self::LenExpr;
    fn get_at(self, index: Var<U64Type>) -> Self::GetExpr;
}

// =============================================================================
// IntoStagedIterator
// =============================================================================

pub trait IntoStagedIterator {
    type Iter: StagedIterator;
    fn staged_iter(self) -> Self::Iter;
}
