//! Core traits for staged iteration.

use crate::control::not;
use crate::func::Ctx;
use crate::num::{add, gt, lt, select, Num};
use crate::staged::{Const, Staged, Var};
use crate::staged_opt::StagedOpt;
use crate::types::{BoolType, ConstantType, CopyType, StagedType, U64Type};

use super::{Filter, FilterMap, Map, Scan, SkipWhile, TakeWhile, Zip};

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

    /// Stateful map: thread a mutable `Var<St>` accumulator (initialized to
    /// `init`) through the iteration. `f(ctx, state, elem)` updates `state`;
    /// the post-update `state` is emitted as each element.
    ///
    /// Example — prefix sums:
    /// `iter.scan(0i64, |ctx, acc, x| ctx.store(acc, acc + x))`.
    fn scan<St, Init, F>(self, init: Init, f: F) -> Scan<Self, St, Init, F>
    where
        St: StagedType + crate::types::CopyType + 'static,
        Init: crate::staged::IntoStaged<St>,
        F: Fn(&mut Ctx, Var<St>, Var<Self::Item>) + 'static,
    {
        Scan::new(self, init, f)
    }

    /// Map-and-filter fused: `f` returns a [`StagedOpt`] (typically via
    /// `cond.then_some(value)`); `Some` payloads are kept, `None` dropped. No
    /// `Option` is materialized — one branch per element, value in a register.
    fn filter_map<F, O>(self, f: F) -> FilterMap<Self, F>
    where
        F: Fn(Var<Self::Item>) -> O,
        O: StagedOpt,
    {
        FilterMap::new(self, f)
    }

    /// Yield elements while `p` holds; stop the whole iteration at the first
    /// element where it fails (short-circuits via `break_loop`).
    fn take_while<P, Cond>(self, p: P) -> TakeWhile<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = BoolType>,
    {
        TakeWhile::new(self, p)
    }

    /// Skip leading elements while `p` holds; yield the rest (starting at the
    /// first element where `p` fails).
    fn skip_while<P, Cond>(self, p: P) -> SkipWhile<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = BoolType>,
    {
        SkipWhile::new(self, p)
    }

    // =========================================================================
    // Terminal operations
    // =========================================================================

    /// Sum all elements. Accumulator starts at `T::RuntimeValue::default()` (zero).
    fn sum(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: Num,
        <Self::Item as StagedType>::RuntimeValue: Default,
    {
        let acc = ctx.var(Const::<Self::Item>::new(Default::default()));
        self.for_each(ctx, move |ctx, elem| {
            ctx.store(acc, acc + elem);
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
            ctx.store(acc, acc + 1u64);
        });
        acc
    }

    /// Branchless count of elements satisfying `pred`.
    ///
    /// Equivalent to `self.filter(pred).count(ctx)` but adds a predicated
    /// `0/1` (via cmov) every iteration instead of branching — so the loop
    /// body has no data-dependent branch and stays vectorizable.
    fn count_if<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<U64Type>
    where
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = BoolType> + 'static,
    {
        let acc = ctx.var(0u64);
        self.for_each(ctx, move |ctx, elem| {
            ctx.store(acc, acc + select(pred(elem), 1u64, 0u64));
        });
        acc
    }

    /// Branchless sum of elements satisfying `pred`.
    ///
    /// Equivalent to `self.filter(pred).sum(ctx)` but adds `select(pred, elem,
    /// 0)` every iteration instead of branching.
    fn sum_if<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<Self::Item>
    where
        Self::Item: Num,
        <Self::Item as StagedType>::RuntimeValue: Default,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = BoolType> + 'static,
    {
        let acc = ctx.var(Const::<Self::Item>::new(Default::default()));
        self.for_each(ctx, move |ctx, elem| {
            let zero = Const::<Self::Item>::new(Default::default());
            ctx.store(acc, acc + select(pred(elem), elem, zero));
        });
        acc
    }

    /// Find the minimum element. Starts at the type's maximum sentinel.
    fn min(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: Num,
        <Self::Item as StagedType>::RuntimeValue: MinMax,
    {
        let sentinel = <Self::Item as StagedType>::RuntimeValue::max_sentinel();
        let acc = ctx.var(Const::<Self::Item>::new(sentinel));
        self.for_each(ctx, move |ctx, elem| {
            // Branchless: unconditional store of a cmov. Both arms are already
            // in registers, so there's no downside, and the body stays
            // vectorizable/unrollable.
            ctx.store(acc, select(lt(elem, acc), elem, acc));
        });
        acc
    }

    /// Find the maximum element. Starts at the type's minimum sentinel.
    fn max(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: Num,
        <Self::Item as StagedType>::RuntimeValue: MinMax,
    {
        let sentinel = <Self::Item as StagedType>::RuntimeValue::min_sentinel();
        let acc = ctx.var(Const::<Self::Item>::new(sentinel));
        self.for_each(ctx, move |ctx, elem| {
            // Branchless: see `min`.
            ctx.store(acc, select(gt(elem, acc), elem, acc));
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

    // =========================================================================
    // Short-circuiting terminals
    //
    // These drive the ordinary `for_each` loop but `break_loop` out of it once
    // the answer is known. Because combinators (`map`/`filter`/…) introduce
    // only `if_then`s — never loops — the `break_loop` always targets the
    // source's single iteration loop, so these compose freely after `filter`,
    // `map`, etc. (Eager terminals like `sum`/`fold` emit no break and keep
    // their optimal loop body.)
    // =========================================================================

    /// `true` as soon as any element satisfies `pred` (short-circuits).
    fn any<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<bool>
    where
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = BoolType> + 'static,
    {
        let found = ctx.var(false);
        self.for_each(ctx, move |ctx, elem| {
            ctx.if_then(pred(elem), move |ctx| {
                ctx.store(found, true);
                ctx.break_loop();
            });
        });
        found
    }

    /// `true` only if every element satisfies `pred` (short-circuits on the
    /// first failure).
    fn all<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<bool>
    where
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = BoolType> + 'static,
    {
        let result = ctx.var(true);
        self.for_each(ctx, move |ctx, elem| {
            ctx.if_then(not(pred(elem)), move |ctx| {
                ctx.store(result, false);
                ctx.break_loop();
            });
        });
        result
    }

    /// Index (in this iterator's sequence, i.e. after any `filter`) of the
    /// first element satisfying `pred`, or the total element count if none
    /// match (short-circuits).
    fn position<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<U64Type>
    where
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = BoolType> + 'static,
    {
        // `idx` counts elements seen; on a match we break *before* incrementing,
        // so it holds the match position. With no match it ends at the count.
        let idx = ctx.var(0u64);
        self.for_each(ctx, move |ctx, elem| {
            ctx.if_then(pred(elem), move |ctx| {
                ctx.break_loop();
            });
            ctx.store(idx, add(idx, 1u64));
        });
        idx
    }

    /// First `Some` produced by `f`, short-circuiting. Returns `(value, found)`:
    /// when `found` is `false` the iterator was exhausted and `value` holds the
    /// default — check `found` before using `value`. No `Option` is
    /// materialized; `f` is typically `|x| cond.then_some(mapped)`.
    fn find_map<F, O>(self, ctx: &mut Ctx, f: F) -> (Var<O::Item>, Var<bool>)
    where
        F: Fn(Var<Self::Item>) -> O + 'static,
        O: StagedOpt + 'static,
        O::Item: ConstantType + CopyType + 'static,
        <O::Item as StagedType>::RuntimeValue: Default,
    {
        let result = ctx.var(Const::<O::Item>::new(Default::default()));
        let found = ctx.var(false);
        self.for_each(ctx, move |ctx, elem| {
            f(elem).eliminate(
                ctx,
                move |ctx, v| {
                    ctx.store(result, v);
                    ctx.store(found, true);
                    ctx.break_loop();
                },
                |_| {},
            );
        });
        (result, found)
    }
}

// =============================================================================
// IndexedStagedIterator
// =============================================================================

/// A staged iterator that tracks element positions, enabling `zip`.
#[allow(clippy::len_without_is_empty)]
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

    /// Alias for [`staged_iter`](Self::staged_iter), reading more naturally at
    /// call sites that consume `self`: `arr.into_staged_iter()`.
    fn into_staged_iter(self) -> Self::Iter
    where
        Self: Sized,
    {
        self.staged_iter()
    }
}
