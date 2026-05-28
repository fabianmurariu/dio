//! Core traits for staged iteration.

use crate::func::VarBuilder;
use crate::staged::{LetVar, Staged, Var};
use crate::types::{BoolType, StagedType, U64Type, UnitType};

use super::{Filter, Map};

/// A staged iterator that generates loop code when consumed.
///
/// This is a push-based (internal) iterator. The iterator controls the loop
/// structure, and consumers provide what to do with each element via callbacks.
///
/// Callbacks are Rust closures that manipulate staged expressions. They are
/// called at **staging time** to build the computation graph, not at runtime.
pub trait StagedIterator: Sized {
    /// The staged type of elements produced by this iterator.
    type Item: StagedType;

    /// Consume this iterator, generating a loop that processes each element.
    ///
    /// The `consumer` closure receives a `Var<Self::Item>` representing the
    /// current element. It returns a staged expression for the loop body.
    ///
    /// # Implementation Note
    /// - Source iterators implement this to generate the actual loop structure.
    /// - Combinator iterators delegate to inner iterator with a wrapped consumer.
    fn consume<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType> + Clone;

    // =========================================================================
    // Combinators - Transform the iterator, return a new iterator
    // =========================================================================

    /// Transform each element using the given function.
    ///
    /// # Example
    /// ```ignore
    /// slice.staged_iter()
    ///     .map(|x| add(*x, 2.0))
    ///     .sum(builder)
    /// ```
    fn map<F, U, MapOut>(self, f: F) -> Map<Self, F, U>
    where
        F: Fn(Var<Self::Item>) -> MapOut,
        MapOut: Staged<Out = U>,
        U: StagedType,
    {
        Map::new(self, f)
    }

    /// Keep only elements that satisfy the predicate.
    ///
    /// **Note:** Filter breaks index correspondence - cannot be zipped.
    ///
    /// # Example
    /// ```ignore
    /// slice.staged_iter()
    ///     .filter(|x| gt(*x, 0.0))
    ///     .sum(builder)
    /// ```
    fn filter<P, Cond>(self, predicate: P) -> Filter<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = BoolType>,
    {
        Filter::new(self, predicate)
    }

    // =========================================================================
    // Terminal Operations - Consume the iterator, return results
    // =========================================================================

    /// Execute a side-effecting operation for each element.
    fn for_each<F, Body>(self, builder: &mut VarBuilder, f: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType> + Clone,
    {
        self.consume(builder, f)
    }

    // TODO: fold-based methods disabled pending Clone bound fix for apply_update return type
    // The issue is that apply_update returns `impl Staged<Out = UnitType>` which doesn't
    // prove Clone, but consume requires Body: Clone.
    //
    // /// Reduce elements to accumulator value(s).
    // fn fold<Acc, F, Update>(...) -> Acc::Vars
    //
    // /// Sum all elements.
    // fn sum(self, builder: &mut VarBuilder) -> Var<Self::Item>
    //
    // /// Count the number of elements.
    // fn count(self, builder: &mut VarBuilder) -> Var<U64Type>
    //
    // /// Find the minimum element.
    // fn min(self, builder: &mut VarBuilder) -> Var<Self::Item>
    //
    // /// Find the maximum element.
    // fn max(self, builder: &mut VarBuilder) -> Var<Self::Item>
}

/// Helper trait for min/max bounds.
pub trait MinMax: Copy {
    fn min_value() -> Self;
    fn max_value() -> Self;
}

impl MinMax for i64 {
    fn min_value() -> Self { i64::MIN }
    fn max_value() -> Self { i64::MAX }
}

impl MinMax for u64 {
    fn min_value() -> Self { u64::MIN }
    fn max_value() -> Self { u64::MAX }
}

impl MinMax for f64 {
    fn min_value() -> Self { f64::NEG_INFINITY }
    fn max_value() -> Self { f64::INFINITY }
}

impl MinMax for i32 {
    fn min_value() -> Self { i32::MIN }
    fn max_value() -> Self { i32::MAX }
}

impl MinMax for u32 {
    fn min_value() -> Self { u32::MIN }
    fn max_value() -> Self { u32::MAX }
}

/// A staged iterator that maintains index correspondence with its source.
///
/// Indexed iterators support operations that require knowing the current
/// position: `zip`, `enumerate`, `take`, `skip`, `unroll`.
///
/// # Index Correspondence
/// Element N in output corresponds to element N in input. Operations that break this:
/// - `filter` - skipped elements shift indices
/// - `flat_map` - one input produces multiple outputs
///
/// # Like Rayon's IndexedParallelIterator
/// This is analogous to Rayon's `IndexedParallelIterator` - only indexed
/// iterators can be zipped.
pub trait IndexedStagedIterator: StagedIterator {
    /// The concrete type returned by len().
    /// This allows implementations to return their specific LetVar type.
    type LenExpr: Staged<Out = U64Type>;

    /// Get the length of this iterator as a variable.
    ///
    /// Returns a LetVar that materializes the length into a variable.
    /// The returned LetVar derefs to Var<U64Type> which is Copy.
    fn len(&self, builder: &mut VarBuilder) -> LetVar<U64Type, Self::LenExpr>;

    /// Consume with access to the current index.
    fn consume_indexed<F, Body>(
        self,
        builder: &mut VarBuilder,
        consumer: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>, Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType> + Clone;

    // TODO: zip method disabled pending Zip implementation fix
    // /// Zip with another indexed source.
    // fn zip<S>(self, other: S) -> super::Zip<Self, S>
    // where
    //     S: IndexedSource,
    // {
    //     super::Zip::new(self, other)
    // }
}

/// A data source that supports random access by index.
///
/// Used by `zip` to access elements from a secondary source using the
/// primary iterator's index.
pub trait IndexedSource: Clone {
    type Item: StagedType;

    /// The concrete type returned by len().
    type LenExpr: Staged<Out = U64Type>;

    /// The concrete type returned by get_unchecked().
    /// Clone is required because the body expression is cloned in while loops.
    type GetExpr: Staged<Out = Self::Item> + Clone;

    /// Get the number of elements as a variable.
    fn len(&self, builder: &mut VarBuilder) -> LetVar<U64Type, Self::LenExpr>;

    /// Get element at index without bounds checking.
    /// Takes self by value since IndexedSource is Clone.
    /// Returns the staged expression directly (no LetVar) since this is used
    /// inside loop bodies where we don't have access to builder.
    fn get_unchecked(self, index: Var<U64Type>) -> Self::GetExpr;
}

/// Extension trait for ergonomic `.staged_iter()` call.
pub trait IntoStagedIterator {
    type Iter: StagedIterator;

    fn staged_iter(self) -> Self::Iter;
}
