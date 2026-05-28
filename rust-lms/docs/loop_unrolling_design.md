# Loop Unrolling and Staged Iterators Design

## Overview

This document outlines a design for loop unrolling and staged iterator abstractions
in rust-lms. The goal is to:

1. Reduce loop overhead through unrolling
2. Enable better instruction-level parallelism (ILP)
3. Provide ergonomic APIs that abstract away loop complexity
4. Lay groundwork for future SIMD vectorization

## What is Loop Unrolling?

Loop unrolling executes multiple iterations of a loop body in a single loop cycle,
reducing branch overhead and enabling better CPU pipelining.

### Example: Sum Reduction

**Normal loop:**
```
for i in 0..n:
    sum += data[i]    // Data dependency: each add waits for previous
```

**Unrolled by 4:**
```
// Main loop (handles n - n%4 elements)
for i in (0..n).step_by(4):
    sum += data[i]
    sum += data[i+1]
    sum += data[i+2]
    sum += data[i+3]

// Remainder loop (handles n%4 elements)
for i in (n - n%4)..n:
    sum += data[i]
```

**Unrolled with accumulator splitting (best for ILP):**
```
// 4 independent accumulators - no data dependencies between them!
sum0, sum1, sum2, sum3 = 0, 0, 0, 0
for i in (0..n).step_by(4):
    sum0 += data[i]      // Independent
    sum1 += data[i+1]    // Independent
    sum2 += data[i+2]    // Independent
    sum3 += data[i+3]    // Independent

// Combine at the end
total = sum0 + sum1 + sum2 + sum3
```

### Benefits

1. **Reduced loop overhead**: Fewer condition checks and jumps
2. **Instruction-level parallelism**: CPU can execute independent ops simultaneously
3. **Better pipelining**: More instructions available for out-of-order execution
4. **Enables vectorization**: Unrolled loops map naturally to SIMD

### Costs

1. **Code size**: Unrolled code is larger (cache pressure)
2. **Register pressure**: More simultaneous values need registers
3. **Remainder handling**: Extra code for non-divisible lengths

## Current Loop Pattern in rust-lms

```rust
// Current manual loop construction
let i = ctx.let_var(0u64);
let sum = ctx.let_var(0.0f64);

let loop_body = while_loop(lt(*i, data.len()), {
    (
        assign(*sum, add(*sum, data.get_unchecked(*i))),
        assign(*i, add(*i, 1u64)),
    )
});

(i, sum, loop_body, *sum)
```

This is verbose and error-prone. Users must:
- Create index variable manually
- Remember to increment index
- Construct the while_loop correctly

## Proposed Abstractions

### Level 1: Simple Loop Helpers

Reduce boilerplate for common patterns without changing the underlying model.

```rust
// for_range: iterate index from start to end
let sum = ctx.let_var(0.0f64);
for_range(ctx, 0u64, data.len(), |i| {
    assign(*sum, add(*sum, data.get_unchecked(i)))
})

// for_each: iterate over slice elements
for_each(ctx, data, |i, val| {
    assign(*sum, add(*sum, val))
})
```

**Implementation:**

```rust
/// Iterate index from start to end (exclusive)
pub fn for_range<END, F, BODY>(
    ctx: &mut CompilerContext,
    start: u64,
    end: END,
    body_fn: F,
) -> impl Staged<Out = UnitType>
where
    END: Staged<Out = U64Type> + Clone,
    F: FnOnce(Var<U64Type>) -> BODY,
    BODY: Staged<Out = UnitType>,
{
    let i = ctx.let_var(start);
    let body = body_fn(*i);
    (
        i,
        while_loop(lt(*i, end), (body, assign(*i, add(*i, 1u64))))
    )
}
```

### Level 2: Fold/Reduce Operations

Higher-level abstraction for reductions with explicit accumulator management.

```rust
// fold: reduce slice to single value
let sum = slice_fold(ctx, data, 0.0f64, |acc, val| {
    add(acc, val)
});

// fold with predicate
let filtered_sum = slice_fold_if(ctx, data, 0.0f64,
    |val| gt(val, threshold),
    |acc, val| add(acc, val)
);
```

**Implementation:**

```rust
/// Fold over slice elements
pub fn slice_fold<'a, T, ACC, F, BODY>(
    ctx: &mut CompilerContext,
    slice: impl Staged<Out = SRef<'a, Slice<T>>> + Clone,
    init: ACC::RuntimeValue,
    fold_fn: F,
) -> Var<ACC>
where
    T: StagedType + CopyType,
    ACC: StagedType + ConstantType,
    F: FnOnce(Var<ACC>, SliceGetUnchecked<...>) -> BODY,
    BODY: Staged<Out = ACC>,
{
    let i = ctx.let_var(0u64);
    let acc = ctx.let_var(init);

    let loop_body = while_loop(lt(*i, slice.len()), {
        let val = slice.get_unchecked(*i);
        (
            assign(*acc, fold_fn(*acc, val)),
            assign(*i, add(*i, 1u64)),
        )
    });

    // Return the accumulator variable
    acc
}
```

### Level 3: Unrolled Loops

The key challenge: for unrolling, we need to generate the body multiple times
with different index expressions. This requires a different API.

```rust
// Unrolled for_range - body_fn is called N times per iteration
for_range_unrolled::<4>(ctx, 0u64, data.len(), |offset| {
    // offset is Add<Var<U64Type>, Const<U64Type>> for values 0,1,2,3
    let idx = add(*i, offset);
    assign(*sum, add(*sum, data.get_unchecked(idx)))
})
```

**The Body Regeneration Problem:**

For unrolling, we can't use `FnOnce` - we need to call the body function
multiple times with different offsets. Options:

1. **`Fn` closure**: `F: Fn(impl Staged<Out = U64Type>) -> BODY`
2. **Trait-based body**: User implements `UnrollBody` trait
3. **Macro-based**: `unroll!(4, |i| body)` expands to repeated code

Option 1 is simplest:

```rust
/// Unrolled loop over range
pub fn for_range_unrolled<const N: usize, END, F, BODY>(
    ctx: &mut CompilerContext,
    start: u64,
    end: END,
    body_fn: F,
) -> impl Staged<Out = UnitType>
where
    END: Staged<Out = U64Type> + Clone,
    F: Fn(/* index expr */) -> BODY,
    BODY: Staged<Out = UnitType>,
{
    let i = ctx.let_var(start);
    let len = end;  // Capture end expression

    // Main unrolled loop
    let unroll_const = Const::<U64Type>::new(N as u64);
    let main_loop = while_loop(
        lt(add(*i, unroll_const), len.clone()),
        {
            // Generate N copies of body with offsets 0..N
            let bodies: [BODY; N] = std::array::from_fn(|offset| {
                body_fn(add(*i, Const::new(offset as u64)))
            });
            (
                bodies,  // Execute all N bodies
                assign(*i, add(*i, unroll_const)),
            )
        }
    );

    // Remainder loop
    let remainder_loop = while_loop(
        lt(*i, len),
        (
            body_fn(*i),
            assign(*i, add(*i, 1u64)),
        )
    );

    (i, main_loop, remainder_loop)
}
```

### Level 4: Staged Iterators

Full iterator abstraction with method chaining.

```rust
// Staged iterator with chaining
data.staged_iter()
    .filter(|val| gt(val, threshold))
    .map(|val| mul(val, 2.0))
    .unroll::<4>()
    .fold(0.0, |acc, val| add(acc, val))
```

**Core Types:**

```rust
/// A staged iterator over a slice
pub struct StagedSliceIter<'a, T: StagedType, S> {
    slice: S,
    _phantom: PhantomData<&'a T>,
}

/// Filtered iterator
pub struct FilteredIter<I, P> {
    inner: I,
    predicate: P,
}

/// Mapped iterator
pub struct MappedIter<I, F> {
    inner: I,
    map_fn: F,
}

/// Unrolled iterator
pub struct UnrolledIter<I, const N: usize> {
    inner: I,
}

/// Trait for staged iteration
pub trait StagedIterator {
    type Item: StagedType;

    fn filter<P>(self, pred: P) -> FilteredIter<Self, P>
    where P: Fn(/* item */) -> impl Staged<Out = BoolType>;

    fn map<F, O>(self, f: F) -> MappedIter<Self, F>
    where F: Fn(/* item */) -> impl Staged<Out = O>;

    fn unroll<const N: usize>(self) -> UnrolledIter<Self, N>;

    fn for_each<F>(self, f: F) -> impl Staged<Out = UnitType>
    where F: Fn(/* item */) -> impl Staged<Out = UnitType>;

    fn fold<ACC, F>(self, init: ACC, f: F) -> impl Staged<Out = ACC>
    where F: Fn(Var<ACC>, /* item */) -> impl Staged<Out = ACC>;
}
```

## Accumulator Splitting for ILP

For maximum performance in reductions, we want multiple independent accumulators:

```rust
// With accumulator splitting (unroll factor 4)
let (sum0, sum1, sum2, sum3) = slice_fold_split::<4>(
    ctx, data, 0.0f64,
    |acc, val| add(acc, val)
);
let total = add(add(sum0, sum1), add(sum2, sum3));
```

**Generated code pattern:**
```
acc0, acc1, acc2, acc3 = init, init, init, init
i = 0
while i + 4 <= len:
    acc0 = f(acc0, data[i])
    acc1 = f(acc1, data[i+1])
    acc2 = f(acc2, data[i+2])
    acc3 = f(acc3, data[i+3])
    i += 4

// Remainder
while i < len:
    acc0 = f(acc0, data[i])
    i += 1

// Combine
result = combine(acc0, acc1, acc2, acc3)
```

## Implementation Phases

### Phase 1: Simple Loop Helpers
- `for_range(ctx, start, end, |i| body)` - basic range loop
- `for_each_slice(ctx, slice, |i, val| body)` - iterate over slice
- No unrolling, just ergonomic wrappers

### Phase 2: Fold/Reduce
- `slice_fold(ctx, slice, init, |acc, val| expr)` - basic fold
- `slice_fold_if(ctx, slice, init, pred, |acc, val| expr)` - filtered fold
- Proper accumulator management

### Phase 3: Unrolled Loops
- `for_range_unrolled::<N>(ctx, start, end, |i| body)` - unrolled range
- `slice_fold_unrolled::<N>(ctx, slice, init, |acc, val| expr)` - unrolled fold
- Main loop + remainder handling

### Phase 4: Accumulator Splitting
- `slice_fold_split::<N>(ctx, slice, init, |acc, val| expr)`
- Multiple independent accumulators
- Final combination step

### Phase 5: Full Staged Iterators (Optional)
- Method chaining API
- filter/map/fold composition
- Lazy evaluation and fusion

## Generated IR Examples

### Non-unrolled sum:
```
block0:
    v0 = iconst.i64 0       ; i = 0
    v1 = f64const 0.0       ; sum = 0.0
    jump block1(v0, v1)

block1(v2: i64, v3: f64):   ; loop header
    v4 = icmp ult v2, v_len
    brif v4, block2, block3

block2:                      ; loop body
    v5 = load.f64 [ptr + v2*8]
    v6 = fadd v3, v5        ; sum += val (data dependency!)
    v7 = iadd v2, 1
    jump block1(v7, v6)

block3:                      ; exit
    return v3
```

### Unrolled by 4 with accumulator splitting:
```
block0:
    v0 = iconst.i64 0       ; i = 0
    v1 = f64const 0.0       ; sum0 = 0.0
    v2 = f64const 0.0       ; sum1 = 0.0
    v3 = f64const 0.0       ; sum2 = 0.0
    v4 = f64const 0.0       ; sum3 = 0.0
    jump block1(v0, v1, v2, v3, v4)

block1(vi: i64, vs0: f64, vs1: f64, vs2: f64, vs3: f64):
    v_check = iadd vi, 4
    v_cond = icmp ule v_check, v_len
    brif v_cond, block2, block3

block2:                      ; unrolled body - 4 independent loads + adds
    v10 = load.f64 [ptr + vi*8]
    v11 = load.f64 [ptr + vi*8 + 8]
    v12 = load.f64 [ptr + vi*8 + 16]
    v13 = load.f64 [ptr + vi*8 + 24]
    v20 = fadd vs0, v10     ; Independent!
    v21 = fadd vs1, v11     ; Independent!
    v22 = fadd vs2, v12     ; Independent!
    v23 = fadd vs3, v13     ; Independent!
    v_next = iadd vi, 4
    jump block1(v_next, v20, v21, v22, v23)

block3:                      ; remainder + combine
    ; ... handle remaining elements with vs0 ...
    v_total = fadd (fadd vs0, vs1), (fadd vs2, vs3)
    return v_total
```

## API Design Considerations

### Closure vs Trait

**Closure approach** (simpler API):
```rust
for_range_unrolled::<4>(ctx, 0, n, |i| {
    assign(*sum, add(*sum, data.get_unchecked(i)))
})
```

**Trait approach** (more flexible):
```rust
struct SumBody { sum: Var<F64Type>, data: Var<SRef<Slice<F64Type>>> }

impl UnrollableBody for SumBody {
    fn body(&self, i: impl Staged<Out = U64Type>) -> impl Staged<Out = UnitType> {
        assign(*self.sum, add(*self.sum, self.data.get_unchecked(i)))
    }
}

for_range_unrolled::<4>(ctx, 0, n, SumBody { sum, data })
```

Recommendation: Start with closure approach, add trait for advanced use cases.

### Const Generics vs Runtime Unroll Factor

**Const generics** (compile-time):
```rust
for_range_unrolled::<4>(...)  // Unroll factor known at Rust compile time
```

**Runtime parameter**:
```rust
for_range_unrolled(4, ...)    // Unroll factor can vary at runtime
```

Recommendation: Use const generics - enables better type checking and the
unroll factor is typically a compile-time decision anyway.

## Future: SIMD Vectorization

Once loop unrolling is in place, SIMD is a natural extension:

```rust
// Future API
data.staged_iter()
    .vectorize::<4, f64x4>()  // Use SIMD f64x4 type
    .fold(f64x4::splat(0.0), |acc, val| acc + val)
    .horizontal_sum()
```

This would generate Cranelift SIMD instructions (F64X2 on most platforms).

## Summary

| Level | API | Benefit |
|-------|-----|---------|
| 1 | `for_range`, `for_each` | Reduced boilerplate |
| 2 | `slice_fold` | Proper accumulator handling |
| 3 | `*_unrolled` | Reduced loop overhead |
| 4 | `*_split` | Instruction-level parallelism |
| 5 | Staged iterators | Composable, ergonomic API |

The staged iterator approach can help abstract loop complexities by:
1. Hiding index management
2. Enabling declarative loop transformations (unroll, vectorize)
3. Providing a familiar Rust iterator-like API
4. Allowing optimization hints without changing semantics
