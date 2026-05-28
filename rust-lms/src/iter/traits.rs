//! Core traits for staged iteration.

use crate::control::if_then;
use crate::func::VarBuilder;
use crate::num::{add, gt, lt};
use crate::num::{SupportsAdd, SupportsComparison};
use crate::staged::{assign, Const, LetVar, Staged, Var};
use crate::types::{BoolType, ConstantType, CopyType, StagedType, U64Type, UnitType};

use super::accumulator::{Accumulator, FoldExpr, IntoAccumulatorUpdate};
use super::{Filter, Map, Zip};

// =============================================================================
// MinMax helper — initial values for min/max reductions
// =============================================================================

pub trait MinMax: Copy {
    fn min_sentinel() -> Self; // starting value for max-reduction (smallest possible)
    fn max_sentinel() -> Self; // starting value for min-reduction (largest possible)
}

impl MinMax for i64 {
    fn min_sentinel() -> Self { i64::MIN }
    fn max_sentinel() -> Self { i64::MAX }
}
impl MinMax for u64 {
    fn min_sentinel() -> Self { u64::MIN }
    fn max_sentinel() -> Self { u64::MAX }
}
impl MinMax for f64 {
    fn min_sentinel() -> Self { f64::NEG_INFINITY }
    fn max_sentinel() -> Self { f64::INFINITY }
}
impl MinMax for i32 {
    fn min_sentinel() -> Self { i32::MIN }
    fn max_sentinel() -> Self { i32::MAX }
}
impl MinMax for u32 {
    fn min_sentinel() -> Self { u32::MIN }
    fn max_sentinel() -> Self { u32::MAX }
}

// =============================================================================
// StagedIterator
// =============================================================================

/// A push-based staged iterator. The iterator controls the loop structure;
/// consumers provide a callback that is called at staging time to build the
/// per-iteration body expression.
pub trait StagedIterator: Sized {
    type Item: StagedType;

    /// Drive a loop over all elements, calling `consumer` once at staging time
    /// to build the per-element body expression.
    ///
    /// `Body: Clone` is required because the loop codegen clones the body each
    /// time it needs to emit it into a Cranelift block.
    fn consume<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType> + use<Self, F, Body>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType> + Clone;

    // =========================================================================
    // Combinators
    // =========================================================================

    fn map<F, U, MapOut>(self, f: F) -> Map<Self, F, U>
    where
        F: Fn(Var<Self::Item>) -> MapOut,
        MapOut: Staged<Out = U>,
        U: StagedType,
    {
        Map::new(self, f)
    }

    fn filter<P, Cond>(self, predicate: P) -> Filter<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = BoolType>,
    {
        Filter::new(self, predicate)
    }

    // =========================================================================
    // Terminal operations
    // =========================================================================

    /// Execute a side-effecting operation for each element.
    fn for_each<F, Body>(self, builder: &mut VarBuilder, f: F) -> impl Staged<Out = UnitType> + use<Self, F, Body>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        self.consume(builder, f)
    }

    /// Reduce elements with a fold function.
    ///
    /// Returns a Rust tuple `(loop_expr, result_vars)`. Include `loop_expr` in
    /// the staged sequence (as a statement) and read from `result_vars` after.
    ///
    /// # Example
    /// ```ignore
    /// let (fold_loop, (count_var, sum_var)) = slice
    ///     .staged_iter()
    ///     .fold(ctx, (0u64, 0.0f64), |(count, sum), elem| {
    ///         (add(count, 1u64), add(sum, elem))
    ///     });
    /// staged_block! {
    ///     fold_loop;
    ///     sum_var
    /// }
    /// ```
    fn fold<Acc, FoldFn, Update>(
        self,
        builder: &mut VarBuilder,
        init: Acc,
        fold_fn: FoldFn,
    ) -> (impl Staged<Out = UnitType> + use<Self, Acc, FoldFn, Update>, Acc::Vars)
    where
        Acc: Accumulator,
        Acc::Vars: Clone,
        FoldFn: FnOnce(Acc::Refs, Var<Self::Item>) -> Update,
        Update: IntoAccumulatorUpdate<Acc>,
        Update::Update: Clone,
    {
        let (acc_init, vars) = Acc::create_vars(builder, init);
        let vars_clone = vars.clone();
        let loop_expr = self.consume(builder, move |elem| {
            let refs = Acc::as_refs(&vars_clone);
            let update = fold_fn(refs, elem);
            update.apply_update(vars_clone.clone())
        });
        ((acc_init, loop_expr), vars)
    }

    /// Sum all elements. Initializes accumulator to `T::RuntimeValue::default()` (zero).
    ///
    /// Returns a staged expression that evaluates to the final sum.
    fn sum(self, builder: &mut VarBuilder) -> impl Staged<Out = Self::Item> + use<Self>
    where
        Self::Item: StagedType + ConstantType + CopyType + SupportsAdd,
        <Self::Item as StagedType>::RuntimeValue: Default,
    {
        let init_lv = builder.let_var(Const::<Self::Item>::new(Default::default()));
        let acc = init_lv.var();
        let loop_expr = self.consume(builder, move |elem| {
            assign(acc, add::<Self::Item, _, _>(acc, elem))
        });
        FoldExpr { init: init_lv, loop_expr, result_var: acc }
    }

    /// Count the number of elements that pass through (including filtered ones
    /// if a filter is upstream).
    fn count(self, builder: &mut VarBuilder) -> impl Staged<Out = U64Type> + use<Self> {
        let init_lv = builder.let_var(0u64);
        let acc = init_lv.var();
        let loop_expr = self.consume(builder, move |_elem| {
            assign(acc, add::<U64Type, _, _>(acc, 1u64))
        });
        FoldExpr { init: init_lv, loop_expr, result_var: acc }
    }

    /// Find the minimum element. Initializes to the type's maximum sentinel value.
    fn min(self, builder: &mut VarBuilder) -> impl Staged<Out = Self::Item> + use<Self>
    where
        Self::Item: StagedType + ConstantType + CopyType + SupportsComparison,
        <Self::Item as StagedType>::RuntimeValue: MinMax,
    {
        let sentinel = <Self::Item as StagedType>::RuntimeValue::max_sentinel();
        let init_lv = builder.let_var(Const::<Self::Item>::new(sentinel));
        let acc = init_lv.var();
        let loop_expr = self.consume(builder, move |elem| {
            if_then(lt::<Self::Item, _, _>(elem, acc), assign(acc, elem))
        });
        FoldExpr { init: init_lv, loop_expr, result_var: acc }
    }

    /// Find the maximum element. Initializes to the type's minimum sentinel value.
    fn max(self, builder: &mut VarBuilder) -> impl Staged<Out = Self::Item> + use<Self>
    where
        Self::Item: StagedType + ConstantType + CopyType + SupportsComparison,
        <Self::Item as StagedType>::RuntimeValue: MinMax,
    {
        let sentinel = <Self::Item as StagedType>::RuntimeValue::min_sentinel();
        let init_lv = builder.let_var(Const::<Self::Item>::new(sentinel));
        let acc = init_lv.var();
        let loop_expr = self.consume(builder, move |elem| {
            if_then(gt::<Self::Item, _, _>(elem, acc), assign(acc, elem))
        });
        FoldExpr { init: init_lv, loop_expr, result_var: acc }
    }
}

// =============================================================================
// IndexedStagedIterator
// =============================================================================

/// A staged iterator that tracks element position, enabling `zip`.
///
/// All slice and range iterators implement this. `filter` breaks the index
/// correspondence so filtered iterators do NOT implement this.
pub trait IndexedStagedIterator: StagedIterator {
    type LenExpr: Staged<Out = U64Type>;

    fn len(&self, builder: &mut VarBuilder) -> LetVar<U64Type, Self::LenExpr>;

    fn consume_indexed<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType> + use<Self, F, Body>
    where
        F: FnOnce(Var<U64Type>, Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType> + Clone;

    /// Zip this iterator with a secondary random-access source.
    ///
    /// Both sources are accessed at the same index on each iteration. The
    /// primary iterator's length drives the loop; the caller must ensure the
    /// secondary source has at least that many elements.
    fn zip<S>(self, other: S) -> Zip<Self, S>
    where
        S: IndexedSource,
    {
        Zip::new(self, other)
    }
}

// =============================================================================
// IndexedSource
// =============================================================================

/// A random-access data source usable as the secondary input to `zip`.
///
/// Implemented by `Var<SRef<Slice<T>>>`.
pub trait IndexedSource: Clone {
    type Item: StagedType;
    type LenExpr: Staged<Out = U64Type>;
    /// The expression type returned by `get_unchecked`. Must be Clone because
    /// the body containing it is cloned by loop codegen.
    type GetExpr: Staged<Out = Self::Item> + Clone;

    fn len(&self, builder: &mut VarBuilder) -> LetVar<U64Type, Self::LenExpr>;
    fn get_at(self, index: Var<U64Type>) -> Self::GetExpr;
}

// =============================================================================
// IntoStagedIterator
// =============================================================================

pub trait IntoStagedIterator {
    type Iter: StagedIterator;
    fn staged_iter(self) -> Self::Iter;
}
