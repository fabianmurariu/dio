# Staged Iterator API Design

## Goal

Enable expressive, composable iteration that compiles to efficient fused loops:

```rust
slice.staged_iter()
    .map(|x| add(x, 2.0))
    .filter(|x| gt(x, 3.0))
    .fold((0u64, 0.0f64), |(count, sum), x| (add(count, 1u64), add(sum, x)))
```

This should generate a **single fused loop** equivalent to:

```rust
let mut count = 0u64;
let mut sum = 0.0f64;
for i in 0..len {
    let x = data[i];
    let mapped = x + 2.0;
    if mapped > 3.0 {
        count += 1;
        sum += mapped;
    }
}
(count, sum)
```

## Core Insight: CPS-Style Fusion

Each combinator transforms what happens to each element:

```
fold provides: "update accumulators with element"
  ↓
filter wraps: "if predicate, then [inner continuation]"
  ↓
map wraps: "transform element, pass to [inner continuation]"
  ↓
source generates: "for each element in slice, call [final continuation]"
```

This is Continuation-Passing Style (CPS). The whole chain compiles to one loop!

## Type Design

### Core Trait

```rust
/// A staged iterator that can be consumed to generate loop code.
///
/// This is an "internal" (push-based) iterator - the iterator controls
/// the loop structure, and consumers provide what to do with each element.
pub trait StagedIterator: Sized {
    /// The staged type of elements produced
    type Item: StagedType;

    /// Consume this iterator, generating a loop that calls `consumer` for each element.
    ///
    /// The consumer receives a `Var<Self::Item>` representing the current element.
    /// Returns a staged expression that, when codegen'd, produces the full loop.
    fn consume<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType>;

    // =========== Combinators ===========

    fn map<F, U, Out>(self, f: F) -> Map<Self, F, U>
    where
        F: Fn(Var<Self::Item>) -> Out,
        Out: Staged<Out = U>,
        U: StagedType + CopyType,
    {
        Map { inner: self, map_fn: f, _phantom: PhantomData }
    }

    fn filter<P, Cond>(self, pred: P) -> Filter<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = BoolType>,
    {
        Filter { inner: self, pred_fn: pred }
    }

    fn unroll<const N: usize>(self) -> Unroll<Self, N> {
        Unroll { inner: self }
    }

    // =========== Terminal Operations ===========

    fn for_each<F, Body>(self, ctx: &mut CompilerContext, f: F) -> impl Staged<Out = UnitType>
    where
        F: FnOnce(Var<Self::Item>) -> Body,
        Body: Staged<Out = UnitType>,
    {
        self.consume(ctx, f)
    }

    fn fold<Acc, F, Update>(
        self,
        ctx: &mut CompilerContext,
        init: Acc,
        f: F,
    ) -> Acc::Vars
    where
        Acc: FoldAccumulator,
        F: FnOnce(Acc::Vars, Var<Self::Item>) -> Update,
        Update: AccumulatorUpdate<Acc>,
    {
        let acc_vars = Acc::create_vars(ctx, init);
        let loop_expr = self.consume(ctx, |elem| {
            let updates = f(acc_vars.clone(), elem);
            updates.apply(acc_vars.clone())
        });
        // The fold returns the accumulator variables
        // User can dereference them after the loop
        acc_vars
    }

    fn sum(self, ctx: &mut CompilerContext) -> Var<Self::Item>
    where
        Self::Item: StagedNumeric,
    {
        self.fold(ctx, Self::Item::zero(), |acc, x| add(*acc, *x))
    }

    fn count(self, ctx: &mut CompilerContext) -> Var<U64Type> {
        self.fold(ctx, 0u64, |acc, _x| add(*acc, 1u64))
    }
}
```

### Source Iterator (Slice)

```rust
/// Iterator over a staged slice
pub struct SliceIter<'a, T: StagedType, S> {
    slice: S,
    _phantom: PhantomData<&'a T>,
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
        // Create loop index
        let i = ctx.let_var(0u64);

        // Create variable for current element
        let elem = ctx.let_var(T::default_value());

        // Build consumer body using the element variable
        let body = consumer(*elem);

        // Generate the loop
        SliceIterLoop {
            index: i,
            elem_var: elem,
            slice: self.slice,
            body,
        }
    }
}

/// The actual loop structure for slice iteration
struct SliceIterLoop<S, Body> {
    index: InitVar<U64Type, ...>,
    elem_var: InitVar<T, ...>,
    slice: S,
    body: Body,
}

impl<S, Body, T> Staged for SliceIterLoop<S, Body>
where
    S: Staged<Out = SRef<'_, Slice<T>>> + Clone,
    Body: Staged<Out = UnitType>,
    T: StagedType,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Initialize index and element variables
        self.index.codegen(ctx);
        self.elem_var.codegen(ctx);

        // Generate: while (i < len) { elem = slice[i]; body; i++; }
        let condition = lt(*self.index, self.slice.clone().len());
        let loop_body = (
            assign(*self.elem_var, self.slice.clone().get_unchecked(*self.index)),
            self.body.clone(),  // Execute consumer's body
            assign(*self.index, add(*self.index, 1u64)),
        );

        while_loop(condition, loop_body).codegen(ctx)
    }
}
```

### Map Combinator

```rust
/// Mapped iterator - transforms each element
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
        // Create variable for mapped result
        let mapped_var = ctx.let_var(U::default_value());

        // Build the outer body that uses the mapped value
        let outer_body = consumer(*mapped_var);

        // Consume inner iterator, inserting map operation
        self.inner.consume(ctx, |inner_elem| {
            // Apply map function to get mapped expression
            let mapped_expr = (self.map_fn)(inner_elem);
            // Sequence: compute mapped value, then run outer body
            (assign(*mapped_var, mapped_expr), outer_body)
        })
    }
}
```

### Filter Combinator

```rust
/// Filtered iterator - only passes elements matching predicate
pub struct Filter<I, P> {
    inner: I,
    pred_fn: P,
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
        // Consume inner iterator, wrapping body in conditional
        self.inner.consume(ctx, |elem| {
            // Compute predicate
            let cond = (self.pred_fn)(elem);
            // Only execute consumer body if predicate is true
            let body = consumer(elem);
            if_then(cond, body)
        })
    }
}
```

### Unroll Combinator

```rust
/// Unrolled iterator - processes N elements per loop iteration
pub struct Unroll<I, const N: usize> {
    inner: I,
}

impl<I, const N: usize> StagedIterator for Unroll<I, N>
where
    I: StagedIterator,
    I::Item: CopyType,
{
    type Item = I::Item;

    fn consume<F, Body>(self, ctx: &mut CompilerContext, consumer: F) -> impl Staged<Out = UnitType>
    where
        F: Fn(Var<Self::Item>) -> Body,  // Note: Fn not FnOnce!
        Body: Staged<Out = UnitType>,
    {
        // For unrolling, we need the consumer to be callable multiple times
        // Generate: main unrolled loop + remainder loop

        // This requires the inner iterator to support unrolled consumption
        // We delegate to a specialized method
        UnrolledLoop::<I, F, N> {
            inner: self.inner,
            consumer,
            ctx,
        }
    }
}
```

### Fold Accumulator Trait

```rust
/// Trait for types that can be fold accumulators
pub trait FoldAccumulator {
    /// The staged variable representation
    type Vars: Clone;

    /// Create variables initialized to the given values
    fn create_vars(ctx: &mut CompilerContext, init: Self) -> Self::Vars;
}

/// Trait for accumulator update expressions
pub trait AccumulatorUpdate<Acc: FoldAccumulator> {
    /// Apply this update to the accumulator variables
    fn apply(self, vars: Acc::Vars) -> impl Staged<Out = UnitType>;
}

// Implementation for single values
impl<T: StagedType + ConstantType> FoldAccumulator for T::RuntimeValue {
    type Vars = Var<T>;

    fn create_vars(ctx: &mut CompilerContext, init: Self) -> Self::Vars {
        *ctx.let_var(init)
    }
}

// Implementation for tuples (2 elements)
impl<A, B> FoldAccumulator for (A, B)
where
    A: Into<SomeConstType>,  // Can create staged constant
    B: Into<SomeConstType>,
{
    type Vars = (Var<A::Type>, Var<B::Type>);

    fn create_vars(ctx: &mut CompilerContext, (a, b): Self) -> Self::Vars {
        (*ctx.let_var(a), *ctx.let_var(b))
    }
}

// Update implementation for tuples
impl<A, B, ExprA, ExprB> AccumulatorUpdate<(A, B)> for (ExprA, ExprB)
where
    ExprA: Staged<Out = A::Type>,
    ExprB: Staged<Out = B::Type>,
{
    fn apply(self, (var_a, var_b): (Var<...>, Var<...>)) -> impl Staged<Out = UnitType> {
        (assign(var_a, self.0), assign(var_b, self.1))
    }
}
```

## Usage Examples

### Simple Sum

```rust
let sum_var = slice.staged_iter()
    .sum(ctx);

// Use sum_var in subsequent code
let result = mul(*sum_var, 2.0);
```

### Count with Filter

```rust
let count = slice.staged_iter()
    .filter(|x| gt(*x, threshold))
    .count(ctx);
```

### Complex Aggregation

```rust
let (count_var, sum_var) = slice.staged_iter()
    .map(|x| add(*x, 2.0))
    .filter(|x| gt(*x, 3.0))
    .fold(ctx, (0u64, 0.0f64), |(count, sum), x| {
        (add(*count, 1u64), add(*sum, *x))
    });

// Compute average after the loop
let avg = div(*sum_var, cast::<_, F64Type>(*count_var));
```

### Unrolled Iteration

```rust
let sum = slice.staged_iter()
    .unroll::<4>()
    .fold(ctx, 0.0f64, |acc, x| add(*acc, *x));
```

### Min/Max with Filtering

```rust
let (min_var, max_var) = slice.staged_iter()
    .filter(|x| gt(*x, threshold))
    .fold(ctx, (f64::INFINITY, f64::NEG_INFINITY), |(min, max), x| {
        (select(lt(*x, *min), *x, *min),
         select(gt(*x, *max), *x, *max))
    });
```

## Generated Code Example

For:
```rust
slice.staged_iter()
    .map(|x| add(*x, 2.0))
    .filter(|x| gt(*x, 3.0))
    .fold(ctx, (0u64, 0.0f64), |(count, sum), x| (add(*count, 1u64), add(*sum, *x)))
```

Generated Cranelift IR (conceptual):
```
block0:
    v_i = iconst.i64 0
    v_count = iconst.i64 0
    v_sum = f64const 0.0
    jump block_header(v_i, v_count, v_sum)

block_header(i: i64, count: i64, sum: f64):
    v_cond = icmp.i64 ult i, len
    brif v_cond, block_body, block_exit(count, sum)

block_body:
    ; Load element
    v_elem = load.f64 [ptr + i*8]

    ; Map: x + 2.0
    v_mapped = fadd v_elem, 2.0

    ; Filter: mapped > 3.0
    v_pred = fcmp.f64 gt v_mapped, 3.0
    brif v_pred, block_update, block_continue(count, sum)

block_update:
    ; Fold: count + 1, sum + x
    v_new_count = iadd count, 1
    v_new_sum = fadd sum, v_mapped
    jump block_continue(v_new_count, v_new_sum)

block_continue(new_count: i64, new_sum: f64):
    v_next_i = iadd i, 1
    jump block_header(v_next_i, new_count, new_sum)

block_exit(final_count: i64, final_sum: f64):
    ; Results available in final_count, final_sum
```

## Implementation Phases

### Phase 1: Core Infrastructure
- `StagedIterator` trait with `consume` method
- `SliceIter` source iterator
- Basic `fold` and `for_each` terminals

### Phase 2: Combinators
- `Map` combinator
- `Filter` combinator
- Tuple accumulator support

### Phase 3: Convenience Methods
- `sum`, `count`, `min`, `max`
- `enumerate` for index access
- `zip` for parallel iteration

### Phase 4: Unrolling
- `Unroll` combinator
- Remainder loop handling
- Accumulator splitting for ILP

### Phase 5: Advanced
- `take`, `skip` for bounded iteration
- `scan` for running accumulator access
- `flat_map` for nested iteration

## Key Design Decisions

1. **Push-based (internal) iteration** - Better for fusion and code generation
2. **Variables for element access** - `Var<T>` allows natural staged expression composition
3. **Closures store functions** - `Fn` closures called during codegen, not runtime
4. **Trait-based accumulators** - Extensible to tuples, custom types
5. **Explicit ctx parameter** - Needed for variable creation; could hide in thread-local

## Open Questions

1. **Parallel iteration (zip)** - How to handle different-length slices?
2. **Early exit (find/any/all)** - Need break/return in generated loop
3. **Nested iteration** - flat_map requires nested loop generation
4. **Error handling** - How to surface codegen errors?
