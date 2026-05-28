# Staged Iterator API Design

## Overview

This document specifies a push-based (internal) staged iterator system for rust-lms.
The design enables composable, fused iteration that compiles to efficient single-loop code.

**Key principles:**
1. **Push-based iteration** - Iterator controls the loop, consumer provides callback
2. **CPS-style fusion** - Combinators wrap callbacks, enabling single-loop compilation
3. **IndexedStagedIterator** - Subset of iterators that support `zip` (like Rayon)
4. **Abstract over sources** - Slices, ranges, and future sources share the same API

## Target API

```rust
slice.staged_iter()
    .map(|x| add(*x, 2.0))
    .filter(|x| gt(*x, 3.0))
    .fold(ctx, (0u64, 0.0f64), |(count, sum), x| (add(*count, 1u64), add(*sum, *x)))
```

This compiles to a **single fused loop** - no intermediate allocations.

## Architecture

```
                    ┌─────────────────────────┐
                    │   StagedIterator        │
                    │   (base trait)          │
                    │                         │
                    │ + map()                 │
                    │ + filter()              │
                    │ + flat_map()            │
                    │ + fold()                │
                    │ + for_each()            │
                    │ + sum(), count(), etc.  │
                    └───────────┬─────────────┘
                                │
                    ┌───────────┴─────────────┐
                    │                         │
        ┌───────────▼───────────┐ ┌───────────▼───────────┐
        │ IndexedStagedIterator │ │ (Non-indexed)         │
        │ (extends base)        │ │                       │
        │                       │ │ - FlatMap             │
        │ + zip()               │ │ - Filter              │
        │ + enumerate()         │ │                       │
        │ + take(), skip()      │ │ *These break index    │
        │ + unroll()            │ │  correspondence       │
        └───────────┬───────────┘ └───────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
┌───────▼───────┐       ┌───────▼───────┐
│  SliceIter    │       │  RangeIter    │
│  (source)     │       │  (source)     │
└───────────────┘       └───────────────┘
```

## Core Insight: CPS-Style Fusion

Each combinator transforms what happens to each element:

```
fold provides: "update accumulators with element"
  ↓
filter wraps: "if predicate, then [fold's callback]"
  ↓
map wraps: "transform element, pass to [filter's callback]"
  ↓
source generates: "for each element in slice, call [final continuation]"
```

Result: The whole chain compiles to ONE loop with all operations inlined.

---

## Core Traits

### StagedIterator (Base Trait)

```rust
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
        ctx: &mut CompilerContext,
        consumer: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType>;

    // =========================================================================
    // Combinators - Transform the iterator, return a new iterator
    // =========================================================================

    /// Transform each element using the given function.
    ///
    /// # Example
    /// ```ignore
    /// slice.staged_iter()
    ///     .map(|x| add(*x, 2.0))  // x: Var<F64Type>, returns Add<...>
    ///     .sum(ctx)
    /// ```
    fn map<F, U, MapOut>(self, f: F) -> Map<Self, F, U>
    where
        F: Fn(Var<Self::Item>) -> MapOut,
        MapOut: Staged<Out = U>,
        U: StagedType + CopyType,
    {
        Map { inner: self, map_fn: f, _phantom: PhantomData }
    }

    /// Keep only elements that satisfy the predicate.
    ///
    /// **Note:** Filter breaks index correspondence - `FilteredIter` does NOT
    /// implement `IndexedStagedIterator` and cannot be zipped.
    ///
    /// # Example
    /// ```ignore
    /// slice.staged_iter()
    ///     .filter(|x| gt(*x, 0.0))
    ///     .sum(ctx)
    /// ```
    fn filter<P, Cond>(self, predicate: P) -> Filter<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = BoolType>,
    {
        Filter { inner: self, predicate }
    }

    /// Map each element to an iterator and flatten the results.
    ///
    /// This generates **nested loops** in the output code.
    ///
    /// **Note:** FlatMap breaks index correspondence - cannot be zipped.
    ///
    /// # Example
    /// ```ignore
    /// // For each row, iterate its elements
    /// matrix.staged_iter()
    ///     .flat_map(|row| row.staged_iter())
    ///     .sum(ctx)
    ///
    /// // Generates:
    /// // for i in 0..matrix.len() {
    /// //     for j in 0..matrix[i].len() {
    /// //         sum += matrix[i][j];
    /// //     }
    /// // }
    /// ```
    fn flat_map<F, Inner>(self, f: F) -> FlatMap<Self, F, Inner>
    where
        F: Fn(Var<Self::Item>) -> Inner,
        Inner: StagedIterator,
    {
        FlatMap { inner: self, flat_map_fn: f, _phantom: PhantomData }
    }

    // =========================================================================
    // Terminal Operations - Consume the iterator, return results
    // =========================================================================

    /// Execute a side-effecting operation for each element.
    fn for_each<F, Body>(self, ctx: &mut CompilerContext, f: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        self.consume(ctx, f)
    }

    /// Reduce elements to accumulator value(s).
    ///
    /// # Example
    /// ```ignore
    /// // Single accumulator
    /// let sum = slice.staged_iter().fold(ctx, 0.0f64, |acc, x| add(*acc, *x));
    ///
    /// // Multiple accumulators (tuple)
    /// let (count, sum) = slice.staged_iter()
    ///     .fold(ctx, (0u64, 0.0f64), |(c, s), x| (add(*c, 1u64), add(*s, *x)));
    /// ```
    fn fold<Acc, F, Update>(self, ctx: &mut CompilerContext, init: Acc, f: F) -> Acc::Vars
    where
        Acc: Accumulator,
        F: FnOnce(Acc::Refs, Var<Self::Item>) -> Update,
        Update: IntoAccumulatorUpdate<Acc>;

    /// Sum all elements.
    fn sum(self, ctx: &mut CompilerContext) -> Var<Self::Item>
    where
        Self::Item: StagedNumeric,
    {
        self.fold(ctx, Self::Item::ZERO, |acc, x| add(*acc, *x))
    }

    /// Count the number of elements.
    fn count(self, ctx: &mut CompilerContext) -> Var<U64Type> {
        self.fold(ctx, 0u64, |acc, _x| add(*acc, 1u64))
    }

    /// Find the minimum element.
    fn min(self, ctx: &mut CompilerContext) -> Var<Self::Item>
    where
        Self::Item: StagedOrd + StagedBounded,
    {
        self.fold(ctx, Self::Item::MAX, |acc, x| select(lt(*x, *acc), *x, *acc))
    }

    /// Find the maximum element.
    fn max(self, ctx: &mut CompilerContext) -> Var<Self::Item>
    where
        Self::Item: StagedOrd + StagedBounded,
    {
        self.fold(ctx, Self::Item::MIN, |acc, x| select(gt(*x, *acc), *x, *acc))
    }
}
```

### IndexedStagedIterator (Extended Trait)

```rust
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
    /// Get the length of this iterator.
    fn len(&self) -> impl Staged<Out = U64Type>;

    /// Consume with access to the current index.
    fn consume_indexed<F, Body>(
        self,
        ctx: &mut CompilerContext,
        consumer: F,
    ) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>, Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType>;

    /// Zip with another indexed source.
    ///
    /// Both sources are iterated in lockstep using the same index.
    /// Length is the minimum of both.
    ///
    /// # Example
    /// ```ignore
    /// // Dot product
    /// let dot = vec_a.staged_iter()
    ///     .zip(vec_b)
    ///     .map(|(a, b)| mul(*a, *b))
    ///     .sum(ctx);
    /// ```
    fn zip<S>(self, other: S) -> Zip<Self, S>
    where
        S: IndexedSource,
    {
        Zip { iter: self, other }
    }

    /// Attach the index to each element.
    ///
    /// # Example
    /// ```ignore
    /// slice.staged_iter()
    ///     .enumerate()
    ///     .for_each(ctx, |(i, x)| { /* i is index, x is element */ })
    /// ```
    fn enumerate(self) -> Enumerate<Self> {
        Enumerate { inner: self }
    }

    /// Take only the first `n` elements.
    fn take<N: Staged<Out = U64Type>>(self, n: N) -> Take<Self, N> {
        Take { inner: self, count: n }
    }

    /// Skip the first `n` elements.
    fn skip<N: Staged<Out = U64Type>>(self, n: N) -> Skip<Self, N> {
        Skip { inner: self, count: n }
    }

    /// Request loop unrolling by factor `N`.
    fn unroll<const N: usize>(self) -> Unroll<Self, N> {
        Unroll { inner: self }
    }
}
```

### IndexedSource (For Zip)

```rust
/// A data source that supports random access by index.
///
/// Used by `zip` to access elements from a secondary source using the
/// primary iterator's index.
pub trait IndexedSource {
    type Item: StagedType;

    /// Get the number of elements.
    fn len(&self) -> impl Staged<Out = U64Type>;

    /// Get element at index without bounds checking.
    fn get_unchecked(&self, index: impl Staged<Out = U64Type>) -> impl Staged<Out = Self::Item>;
}

// Slices are indexed sources
impl<'a, T: StagedType + CopyType> IndexedSource for Var<SRef<'a, Slice<T>>> {
    type Item = T;

    fn len(&self) -> impl Staged<Out = U64Type> {
        (*self).len()
    }

    fn get_unchecked(&self, index: impl Staged<Out = U64Type>) -> impl Staged<Out = T> {
        (*self).get_unchecked(index)
    }
}
```

---

## Combinator Implementations

### Map

```rust
/// Iterator adapter that transforms each element.
///
/// Preserves `IndexedStagedIterator` if inner iterator has it.
pub struct Map<I, F, U> {
    inner: I,
    map_fn: F,
    _phantom: PhantomData<U>,
}

impl<I, F, U, MapOut> StagedIterator for Map<I, F, U>
where
    I: StagedIterator,
    F: Fn(Var<I::Item>) -> MapOut,
    MapOut: Staged<Out = U>,
    U: StagedType + CopyType,
{
    type Item = U;

    fn consume<G, Body>(self, ctx: &mut CompilerContext, consumer: G) -> impl Staged<Out = UnitType>
    where
        G: FnOnce(Var<U>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        // Create variable for mapped value
        let mapped_var = ctx.let_var(U::default_value());

        // Build outer body using mapped variable
        let outer_body = consumer(*mapped_var);

        // Delegate to inner iterator
        self.inner.consume(ctx, move |inner_elem| {
            let mapped_expr = (self.map_fn)(inner_elem);
            (
                mapped_var,                           // Initialize variable
                assign(*mapped_var, mapped_expr),     // Compute mapped value
                outer_body,                           // Run consumer
            )
        })
    }
}

// Map preserves IndexedStagedIterator
impl<I, F, U, MapOut> IndexedStagedIterator for Map<I, F, U>
where
    I: IndexedStagedIterator,
    F: Fn(Var<I::Item>) -> MapOut,
    MapOut: Staged<Out = U>,
    U: StagedType + CopyType,
{
    fn len(&self) -> impl Staged<Out = U64Type> {
        self.inner.len()
    }

    fn consume_indexed<G, Body>(self, ctx: &mut CompilerContext, consumer: G) -> impl Staged<Out = UnitType>
    where
        G: FnOnce(Var<U64Type>, Var<U>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        let mapped_var = ctx.let_var(U::default_value());

        self.inner.consume_indexed(ctx, move |idx, inner_elem| {
            let mapped_expr = (self.map_fn)(inner_elem);
            let outer_body = consumer(idx, *mapped_var);
            (mapped_var, assign(*mapped_var, mapped_expr), outer_body)
        })
    }
}
```

### Filter

```rust
/// Iterator adapter that keeps only elements matching a predicate.
///
/// **Does NOT implement `IndexedStagedIterator`** - filter breaks index correspondence.
pub struct Filter<I, P> {
    inner: I,
    predicate: P,
}

impl<I, P, Cond> StagedIterator for Filter<I, P>
where
    I: StagedIterator,
    P: Fn(Var<I::Item>) -> Cond,
    Cond: Staged<Out = BoolType>,
{
    type Item = I::Item;

    fn consume<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        self.inner.consume(ctx, move |elem| {
            let condition = (self.predicate)(elem);
            let body = consumer(elem);
            if_then(condition, body)  // Only execute if predicate passes
        })
    }
}

// Filter does NOT implement IndexedStagedIterator - this is intentional!
```

### FlatMap

```rust
/// Iterator adapter that maps to iterators and flattens.
///
/// Generates nested loops. **Does NOT implement `IndexedStagedIterator`**.
pub struct FlatMap<I, F, Inner> {
    inner: I,
    flat_map_fn: F,
    _phantom: PhantomData<Inner>,
}

impl<I, F, Inner> StagedIterator for FlatMap<I, F, Inner>
where
    I: StagedIterator,
    F: Fn(Var<I::Item>) -> Inner,
    Inner: StagedIterator,
{
    type Item = Inner::Item;

    fn consume<G, Body>(self, ctx: &mut CompilerContext, consumer: G) -> impl Staged<Out = UnitType>
    where
        G: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        // Outer loop
        self.inner.consume(ctx, move |outer_elem| {
            // Create inner iterator from outer element
            let inner_iter = (self.flat_map_fn)(outer_elem);
            // Inner loop (nested)
            inner_iter.consume(ctx, consumer)
        })
    }
}

// FlatMap does NOT implement IndexedStagedIterator - this is intentional!
```

### Zip

```rust
/// Iterator that combines two sources element-wise using the same index.
pub struct Zip<I, S> {
    iter: I,   // Primary iterator (controls the loop)
    other: S,  // Secondary source (accessed by index)
}

impl<I, S> StagedIterator for Zip<I, S>
where
    I: IndexedStagedIterator,
    S: IndexedSource,
    I::Item: CopyType,
    S::Item: CopyType,
{
    // Note: Item would ideally be a tuple, but we handle this specially
    type Item = I::Item;  // See consume signature for how we handle pairs

    fn consume<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        // Consumer receives TWO Vars, not a tuple Var
        F: FnOnce(Var<I::Item>, Var<S::Item>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        // Create variable for secondary element
        let other_elem = ctx.let_var(S::Item::default_value());

        // Consume primary iterator with index access
        self.iter.consume_indexed(ctx, move |idx, primary_elem| {
            // Load from secondary source using same index
            let other_expr = self.other.get_unchecked(idx);
            let body = consumer(primary_elem, *other_elem);
            (other_elem, assign(*other_elem, other_expr), body)
        })
    }
}

impl<I, S> IndexedStagedIterator for Zip<I, S>
where
    I: IndexedStagedIterator,
    S: IndexedSource,
    I::Item: CopyType,
    S::Item: CopyType,
{
    fn len(&self) -> impl Staged<Out = U64Type> {
        min(self.iter.len(), self.other.len())
    }

    fn consume_indexed<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>, Var<I::Item>, Var<S::Item>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        let other_elem = ctx.let_var(S::Item::default_value());

        self.iter.consume_indexed(ctx, move |idx, primary_elem| {
            let other_expr = self.other.get_unchecked(idx);
            let body = consumer(idx, primary_elem, *other_elem);
            (other_elem, assign(*other_elem, other_expr), body)
        })
    }
}
```

### Enumerate

```rust
/// Iterator that attaches indices to elements.
pub struct Enumerate<I> {
    inner: I,
}

impl<I: IndexedStagedIterator> StagedIterator for Enumerate<I> {
    type Item = I::Item;  // Consumer receives (idx, elem) as separate args

    fn consume<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>, Var<I::Item>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        self.inner.consume_indexed(ctx, consumer)
    }
}
```

### Unroll

```rust
/// Iterator adapter that requests loop unrolling.
pub struct Unroll<I, const N: usize> {
    inner: I,
}

impl<I: IndexedStagedIterator, const N: usize> StagedIterator for Unroll<I, N>
where
    I::Item: CopyType,
{
    type Item = I::Item;

    fn consume<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: Fn(Var<Self::Item>) -> Body,  // Note: Fn not FnOnce (called N times)
        Body: Staged<Out = UnitType> + Clone,
    {
        // Generates:
        // while (i + N <= len) {
        //     consumer(data[i+0])
        //     consumer(data[i+1])
        //     ...
        //     consumer(data[i+N-1])
        //     i += N
        // }
        // while (i < len) {
        //     consumer(data[i])
        //     i += 1
        // }

        UnrolledConsume { iter: self.inner, consumer, _n: PhantomData::<N> }
    }
}
```

---

## Source Iterators

### SliceIter

```rust
/// Iterator over elements of a staged slice.
pub struct SliceIter<'a, T, S>
where
    T: StagedType,
{
    slice: S,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T, S> SliceIter<'a, T, S>
where
    T: StagedType + CopyType,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
{
    pub fn new(slice: S) -> Self {
        SliceIter { slice, _phantom: PhantomData }
    }
}

impl<'a, T, S> StagedIterator for SliceIter<'a, T, S>
where
    T: StagedType + CopyType,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
{
    type Item = T;

    fn consume<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<T>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        let i = ctx.let_var(0u64);
        let elem = ctx.let_var(T::default_value());
        let body = consumer(*elem);

        SliceIterLoop {
            index: i,
            elem_var: elem,
            slice: self.slice,
            body,
        }
    }
}

impl<'a, T, S> IndexedStagedIterator for SliceIter<'a, T, S>
where
    T: StagedType + CopyType,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
{
    fn len(&self) -> impl Staged<Out = U64Type> {
        self.slice.clone().len()
    }

    fn consume_indexed<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>, Var<T>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        let i = ctx.let_var(0u64);
        let elem = ctx.let_var(T::default_value());
        let body = consumer(*i, *elem);

        SliceIterLoop { index: i, elem_var: elem, slice: self.slice, body }
    }
}

/// Internal: The actual loop structure for slice iteration.
struct SliceIterLoop<I, E, S, Body> {
    index: I,
    elem_var: E,
    slice: S,
    body: Body,
}

impl<I, E, S, Body, T> Staged for SliceIterLoop<I, E, S, Body>
where
    I: Staged<Out = UnitType>,
    E: Staged<Out = UnitType>,
    S: Staged<Out = SRef<'_, Slice<T>>> + Clone,
    Body: Staged<Out = UnitType> + Clone,
    T: StagedType,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Initialize variables
        self.index.codegen(ctx);
        self.elem_var.codegen(ctx);

        // Extract Var references (implementation detail)
        let i_var: Var<U64Type> = /* from index */;
        let elem_var: Var<T> = /* from elem_var */;

        // Generate: while (i < len) { elem = slice[i]; body; i++; }
        while_loop(
            lt(*i_var, self.slice.clone().len()),
            (
                assign(*elem_var, self.slice.clone().get_unchecked(*i_var)),
                self.body.clone(),
                assign(*i_var, add(*i_var, 1u64)),
            )
        ).codegen(ctx)
    }
}

// Extension trait for ergonomic .staged_iter() call
pub trait IntoStagedIterator {
    type Iter: StagedIterator;
    fn staged_iter(self) -> Self::Iter;
}

impl<'a, T, S> IntoStagedIterator for S
where
    T: StagedType + CopyType,
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone,
{
    type Iter = SliceIter<'a, T, S>;
    fn staged_iter(self) -> Self::Iter { SliceIter::new(self) }
}
```

### RangeIter

```rust
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

    fn consume<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        // Initialize index to start value
        let i = ctx.let_var_expr(self.start);
        let body = consumer(*i);

        RangeIterLoop { index: i, end: self.end, body }
    }
}

impl<Start, End> IndexedStagedIterator for RangeIter<Start, End>
where
    Start: Staged<Out = U64Type> + Clone,
    End: Staged<Out = U64Type> + Clone,
{
    fn len(&self) -> impl Staged<Out = U64Type> {
        sub(self.end.clone(), self.start.clone())
    }

    fn consume_indexed<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<U64Type>, Var<U64Type>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        let i = ctx.let_var_expr(self.start);
        // For range, index and value are the same
        let body = consumer(*i, *i);
        RangeIterLoop { index: i, end: self.end, body }
    }
}

struct RangeIterLoop<I, End, Body> {
    index: I,
    end: End,
    body: Body,
}

impl<I, End, Body> Staged for RangeIterLoop<I, End, Body>
where
    I: Staged<Out = UnitType>,
    End: Staged<Out = U64Type> + Clone,
    Body: Staged<Out = UnitType> + Clone,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.index.codegen(ctx);
        let i_var: Var<U64Type> = /* extract */;

        while_loop(
            lt(*i_var, self.end.clone()),
            (self.body.clone(), assign(*i_var, add(*i_var, 1u64)))
        ).codegen(ctx)
    }
}
```

---

## Accumulator Traits

```rust
/// A type that can serve as a fold accumulator.
pub trait Accumulator: Sized {
    /// The staged variable type(s).
    type Vars: Clone;
    /// References to variables for the fold function.
    type Refs: Clone;

    fn create_vars(ctx: &mut CompilerContext, init: Self) -> Self::Vars;
    fn as_refs(vars: &Self::Vars) -> Self::Refs;
}

/// Update expression for accumulators.
pub trait IntoAccumulatorUpdate<Acc: Accumulator> {
    fn apply_update(self, vars: Acc::Vars) -> impl Staged<Out = UnitType>;
}

// Single value implementation
impl<T: StagedType + ConstantType + CopyType> Accumulator for T::RuntimeValue
where T::RuntimeValue: Copy
{
    type Vars = Var<T>;
    type Refs = Var<T>;

    fn create_vars(ctx: &mut CompilerContext, init: Self) -> Self::Vars {
        *ctx.let_var(init)
    }
    fn as_refs(vars: &Self::Vars) -> Self::Refs { *vars }
}

// 2-tuple implementation
impl<A, B> Accumulator for (A::RuntimeValue, B::RuntimeValue)
where
    A: StagedType + ConstantType + CopyType,
    B: StagedType + ConstantType + CopyType,
    A::RuntimeValue: Copy,
    B::RuntimeValue: Copy,
{
    type Vars = (Var<A>, Var<B>);
    type Refs = (Var<A>, Var<B>);

    fn create_vars(ctx: &mut CompilerContext, (a, b): Self) -> Self::Vars {
        (*ctx.let_var(a), *ctx.let_var(b))
    }
    fn as_refs(vars: &Self::Vars) -> Self::Refs { *vars }
}

// Update implementations...
impl<T, Expr> IntoAccumulatorUpdate<T::RuntimeValue> for Expr
where
    T: StagedType + ConstantType + CopyType,
    T::RuntimeValue: Copy,
    Expr: Staged<Out = T>,
{
    fn apply_update(self, var: Var<T>) -> impl Staged<Out = UnitType> {
        assign(var, self)
    }
}

impl<A, B, ExprA, ExprB> IntoAccumulatorUpdate<(A::RuntimeValue, B::RuntimeValue)> for (ExprA, ExprB)
where
    A: StagedType + ConstantType + CopyType,
    B: StagedType + ConstantType + CopyType,
    A::RuntimeValue: Copy,
    B::RuntimeValue: Copy,
    ExprA: Staged<Out = A>,
    ExprB: Staged<Out = B>,
{
    fn apply_update(self, (va, vb): (Var<A>, Var<B>)) -> impl Staged<Out = UnitType> {
        (assign(va, self.0), assign(vb, self.1))
    }
}
```

---

## Usage Examples

### Basic Operations

```rust
// Sum
let sum = slice.staged_iter().sum(ctx);

// Count with filter
let count = slice.staged_iter()
    .filter(|x| gt(*x, threshold))
    .count(ctx);

// Min/Max
let minimum = slice.staged_iter().min(ctx);
```

### Map + Filter + Fold

```rust
let (count, sum) = slice.staged_iter()
    .map(|x| add(*x, 2.0))
    .filter(|x| gt(*x, 3.0))
    .fold(ctx, (0u64, 0.0f64), |(c, s), x| {
        (add(*c, 1u64), add(*s, *x))
    });

let avg = div(*sum, cast::<_, F64Type>(*count));
```

### Zip (Dot Product)

```rust
let dot = vec_a.staged_iter()
    .zip(vec_b)
    .map(|(a, b)| mul(*a, *b))
    .sum(ctx);
```

### FlatMap (Nested Loops)

```rust
// Sum all elements in 2D structure
let total = matrix.staged_iter()
    .flat_map(|row| row.staged_iter())
    .sum(ctx);
```

### Range

```rust
// Sum of squares 1..n
let sum_sq = range(1u64, n)
    .map(|i| mul(*i, *i))
    .fold(ctx, 0u64, |acc, x| add(*acc, x));
```

### Enumerate

```rust
// Weighted sum: sum(i * x[i])
let weighted = slice.staged_iter()
    .enumerate()
    .map(|(i, x)| mul(cast::<_, F64Type>(*i), *x))
    .sum(ctx);
```

---

## Generated Code Examples

### Map + Filter + Fold

```rust
slice.staged_iter()
    .map(|x| add(*x, 2.0))
    .filter(|x| gt(*x, 3.0))
    .fold(ctx, (0u64, 0.0f64), |(c, s), x| (add(*c, 1), add(*s, *x)))
```

**Generated Cranelift IR:**
```
block0:
    v_i = iconst.i64 0
    v_count = iconst.i64 0
    v_sum = f64const 0.0
    jump block1(v_i, v_count, v_sum)

block1(i: i64, count: i64, sum: f64):
    v_cond = icmp ult i, len
    brif v_cond, block2, block_exit(count, sum)

block2:
    v_elem = load.f64 [ptr + i*8]
    v_mapped = fadd v_elem, 2.0          ; map: x + 2
    v_pred = fcmp gt v_mapped, 3.0       ; filter: > 3
    brif v_pred, block3, block4(count, sum)

block3:
    v_new_count = iadd count, 1          ; fold: count + 1
    v_new_sum = fadd sum, v_mapped       ; fold: sum + x
    jump block4(v_new_count, v_new_sum)

block4(c: i64, s: f64):
    v_next = iadd i, 1
    jump block1(v_next, c, s)

block_exit(final_count: i64, final_sum: f64):
    ; results available
```

### Zip

```rust
a.staged_iter().zip(b).map(|(x, y)| mul(*x, *y)).sum(ctx)
```

**Generated:**
```
block1(i: i64, sum: f64):
    brif (i < min_len), block2, exit

block2:
    v_a = load.f64 [ptr_a + i*8]
    v_b = load.f64 [ptr_b + i*8]   ; Same index!
    v_prod = fmul v_a, v_b
    v_new = fadd sum, v_prod
    jump block1(i+1, v_new)
```

### FlatMap (Nested)

```rust
outer.staged_iter().flat_map(|row| row.staged_iter()).sum(ctx)
```

**Generated:**
```
block_outer(i: i64, sum: f64):
    brif (i < outer_len), block_outer_body, exit

block_outer_body:
    v_row_ptr = load [outer + i*16]
    v_row_len = load [outer + i*16 + 8]
    jump block_inner(0, sum)

block_inner(j: i64, inner_sum: f64):
    brif (j < v_row_len), block_inner_body, block_inner_done

block_inner_body:
    v_elem = load [v_row_ptr + j*8]
    v_new = fadd inner_sum, v_elem
    jump block_inner(j+1, v_new)

block_inner_done:
    jump block_outer(i+1, inner_sum)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure
- [ ] `StagedIterator` trait with `consume`
- [ ] `Accumulator` traits
- [ ] Single-value fold

### Phase 2: Source Iterators
- [ ] `SliceIter` (indexed)
- [ ] `RangeIter` (indexed)
- [ ] `IntoStagedIterator` extension

### Phase 3: Basic Combinators
- [ ] `Map` (preserves indexed)
- [ ] `Filter` (non-indexed)
- [ ] `sum`, `count`, `for_each`

### Phase 4: Tuple Accumulators
- [ ] 2, 3, 4-tuple support

### Phase 5: Indexed Operations
- [ ] `IndexedStagedIterator` trait
- [ ] `Zip`
- [ ] `Enumerate`
- [ ] `IndexedSource` trait

### Phase 6: Advanced
- [ ] `FlatMap`
- [ ] `Take`, `Skip`
- [ ] `min`, `max`, `any`, `all`

### Phase 7: Unrolling
- [ ] `Unroll` combinator
- [ ] Main + remainder loops

---

## Open Questions

1. **Tuple element access in zip**: Consumer receives `(Var<A>, Var<B>)` as separate args vs single `Var<(A,B)>`?

2. **Early exit**: `find`/`any`/`all` ideally break early - requires loop break support

3. **collect()**: Materialize iterator to slice - needs memory allocation strategy
