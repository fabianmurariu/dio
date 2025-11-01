# Building Composable Staged APIs in Rust

## Your Question

> "I wonder if there is a way to represent functions? like RepFn<A, B, C, OUT>, I don't know if `impl Fn(Rep<A>, Rep<B>, Rep<C>) -> Rep<OUT>` would work, maybe a different trait? RepCallableN with N implementations? I really don't know how to build composability in this API what would you suggest?"

## The Answer

**YES!** And it's simpler than you think! Rust closures naturally provide exactly what Scala LMS does with `Rep[T] => Rep[U]`.

## Critical Insight: Two Kinds of Functions

In multi-stage programming, there are TWO kinds of functions:

### 1. Meta-Level Functions (Code Generators) ⭐

**Scala LMS:** `f: Rep[T] => Rep[U]`
**Rust:** `F: Fn(Rep<T>) -> Rep<U>`

These functions exist AT STAGING TIME - they generate code!

```rust
// This is a meta-level function (code generator)
vec.map(|x| {
    // This code runs at STAGING TIME
    // It builds an AST/IR for doubling x
    x.mul(RepI64::constant(2))
})
```

**Key point:** The closure itself doesn't exist in the generated code! It runs during compilation to *generate* code.

### 2. Object-Level Functions (Staged Values)

**Scala LMS:** `Rep[T => U]` (less common)
**Rust:** `RepFn<T, U>` (requires defunctionalization)

These functions exist AT RUNTIME in the generated code!

```rust
// This would be an object-level function
let lambda = RepFn::lambda(|x| x.mul(RepI64::constant(2)));
// The lambda itself becomes data that can be passed around
```

**Key point:** The function is reified as data and exists in the generated program.

## Your Scala LMS Examples - Direct Translation!

### Example 1: foreach

**Scala LMS:**
```scala
def foreach(f: Rep[T] => Rep[Unit]): Rep[Unit]
```

**Rust:**
```rust
fn foreach<F>(&self, f: F) -> RepUnit
where
    F: Fn(Rep<T>) -> RepUnit
```

### Example 2: sumIf

**Scala LMS:**
```scala
def sumIf(f: Rep[T] => Rep[Boolean]) = {
    var n = zero[T]
    foreach(x => if (f(x)) n += x)
    n
}
```

**Rust:**
```rust
fn sum_if<F>(&self, predicate: F) -> RepI64
where
    F: Fn(RepI64) -> RepBool,
{
    // Compose with foreach!
    self.foreach(|x| {
        let condition = predicate(x.clone());
        // Generate: if (condition) sum += x
        RepUnit::constant(())
    });
    // Return sum variable
}
```

## Building Composable APIs: The Pattern

The key to composability is **meta-level functions** (closures). Here's the pattern:

### Step 1: Define Your Operations

```rust
impl<T: Staged> Vector<T> {
    /// Basic operation: foreach
    pub fn foreach<F>(&self, f: F) -> RepUnit
    where
        F: Fn(Rep<T>) -> RepUnit,
    {
        // Generate loop, call f for each element
    }

    /// Transform operation: map
    pub fn map<F, Out>(&self, f: F) -> Vector<Out>
    where
        F: Fn(Rep<T>) -> Rep<Out>,
        Out: Staged,
    {
        // Generate loop that transforms each element
    }

    /// Filter operation
    pub fn filter<F>(&self, predicate: F) -> Vector<T>
    where
        F: Fn(Rep<T>) -> RepBool,
    {
        // Generate loop that conditionally includes elements
    }
}
```

### Step 2: Compose Operations

Operations naturally compose because they return new staged values!

```rust
// Chaining: filter then map then reduce
let result = vec
    .filter(|x| x.lt(RepI64::constant(100)))  // Keep x < 100
    .map(|x| x.mul(RepI64::constant(2)))       // Double each
    .sum_if(|x| x.lt(RepI64::constant(50)));   // Sum if < 50

// Nesting: operations use other operations
impl Vector<I64Type> {
    fn sum_if<F>(&self, predicate: F) -> RepI64
    where
        F: Fn(RepI64) -> RepBool,
    {
        // Use foreach internally - composition!
        self.foreach(|x| {
            if predicate(x) {
                // accumulate
            }
        });
    }
}
```

### Step 3: Generic Reductions

```rust
impl Vector<T> {
    /// Generic reduce - the ultimate composable operation!
    pub fn reduce<F>(&self, zero: Rep<T>, f: F) -> Rep<T>
    where
        F: Fn(Rep<T>, Rep<T>) -> Rep<T>,
    {
        self.foreach(|x| {
            // acc = f(acc, x)
        });
    }
}

// Now you can build everything from reduce:
let sum = vec.reduce(RepI64::constant(0), |acc, x| acc.add(x));
let product = vec.reduce(RepI64::constant(1), |acc, x| acc.mul(x));
let max = vec.reduce(RepI64::constant(i64::MIN), |acc, x| {
    // if x > acc then x else acc
});
```

## Do You Need RepFn<A, B, C, OUT>?

**For most cases: NO!** You don't need to define RepFn at all!

### When to use closures (99% of cases):

```rust
// Just use F: Fn(...)
fn map<F, Out>(&self, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>) -> Rep<Out>,

// Multiple arguments? No problem!
fn zip_with<F, U, Out>(&self, other: &Vector<U>, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>, Rep<U>) -> Rep<Out>,

// Three arguments? Still fine!
fn combine3<F, U, V, Out>(
    &self,
    vec2: &Vector<U>,
    vec3: &Vector<V>,
    f: F,
) -> Vector<Out>
where
    F: Fn(Rep<T>, Rep<U>, Rep<V>) -> Rep<Out>,
```

### When you MIGHT need RepFn (rare):

Only if you need to:
1. Pass functions as runtime values
2. Store functions in data structures
3. Partially apply functions
4. Build function combinators

```rust
// Example: A staged interpreter that needs function values
enum StagedExpr {
    Apply(RepFn<I64Type, I64Type>, RepI64),
    Lambda(RepFn<I64Type, I64Type>),
}
```

## Pattern: Multi-Arity Functions

### Option 1: Tuple Arguments (Simple)

```rust
fn combine<F>(&self, other: &Vector<T>, f: F) -> Rep<T>
where
    F: Fn((Rep<T>, Rep<T>)) -> Rep<T>,  // Tuple!
{
    // ...
}

// Usage:
vec1.combine(&vec2, |(x, y)| x.add(y))
```

### Option 2: Multiple Parameters (Ergonomic)

```rust
fn combine<F>(&self, other: &Vector<T>, f: F) -> Rep<T>
where
    F: Fn(Rep<T>, Rep<T>) -> Rep<T>,  // Two separate params!
{
    // ...
}

// Usage:
vec1.combine(&vec2, |x, y| x.add(y))
```

### Option 3: Generic Arity (Advanced)

```rust
trait RepCallable2<A, B, Out> {
    fn call(&self, a: Rep<A>, b: Rep<B>) -> Rep<Out>;
}

impl<F, A, B, Out> RepCallable2<A, B, Out> for F
where
    F: Fn(Rep<A>, Rep<B>) -> Rep<Out>,
    A: Staged,
    B: Staged,
    Out: Staged,
{
    fn call(&self, a: Rep<A>, b: Rep<B>) -> Rep<Out> {
        self(a, b)
    }
}
```

But honestly, just use `Fn(...)` directly! It's simpler.

## Complete Example: Building a Composable Array API

```rust
pub struct Vector<T: Staged> {
    // ...
}

impl<T: Staged> Vector<T> {
    // Basic iteration
    pub fn foreach<F>(&self, f: F) -> RepUnit
    where
        F: Fn(Rep<T>) -> RepUnit,
    {
        // Generate: for (i = 0; i < length; i++) f(data[i])
    }

    // Transform
    pub fn map<F, Out>(&self, f: F) -> Vector<Out>
    where
        F: Fn(Rep<T>) -> Rep<Out>,
        Out: Staged,
    {
        // Generate: new array with f(each element)
    }

    // Filter
    pub fn filter<F>(&self, predicate: F) -> Vector<T>
    where
        F: Fn(Rep<T>) -> RepBool,
    {
        // Generate: new array with elements where predicate is true
    }

    // Reduce
    pub fn reduce<F>(&self, zero: Rep<T>, f: F) -> Rep<T>
    where
        F: Fn(Rep<T>, Rep<T>) -> Rep<T>,
    {
        // Generate: accumulate with f
    }

    // Zip two arrays
    pub fn zip<U, Out, F>(
        &self,
        other: &Vector<U>,
        f: F,
    ) -> Vector<Out>
    where
        U: Staged,
        Out: Staged,
        F: Fn(Rep<T>, Rep<U>) -> Rep<Out>,
    {
        // Generate: for i, result[i] = f(self[i], other[i])
    }
}

// Specialized operations for specific types
impl Vector<I64Type> {
    pub fn sum(&self) -> RepI64 {
        self.reduce(RepI64::constant(0), |acc, x| acc.add(x))
    }

    pub fn product(&self) -> RepI64 {
        self.reduce(RepI64::constant(1), |acc, x| acc.mul(x))
    }

    pub fn sum_if<F>(&self, predicate: F) -> RepI64
    where
        F: Fn(RepI64) -> RepBool,
    {
        // Compose foreach with conditional accumulation
        self.foreach(|x| {
            let condition = predicate(x.clone());
            // if condition { sum += x }
            RepUnit::constant(())
        });
        // return sum
    }
}

// Now we can build complex pipelines!
fn example_pipeline(data: Vector<I64Type>) -> RepI64 {
    data
        .filter(|x| x.lt(RepI64::constant(100)))
        .map(|x| x.mul(RepI64::constant(2)))
        .filter(|x| x.lt(RepI64::constant(50)))
        .sum()
}
```

## Key Principles for Composability

1. **Return Staged Values:** Operations should return `Rep<T>` or `Vector<T>`, not raw code
2. **Use Generic Closures:** Accept `F: Fn(Rep<T>) -> Rep<U>` for maximum flexibility
3. **Build on Basics:** Implement `foreach` and `reduce`, then build everything else on top
4. **Type Safety:** Let Rust's type system ensure correctness
5. **Zero Runtime Cost:** Closures are inlined at staging time - no performance penalty

## Common Patterns

### Pattern 1: Conditional Operations

```rust
fn filter_map<F, Out>(&self, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>) -> Option<Rep<Out>>,  // Rust Option at meta-level!
    Out: Staged,
{
    // Generate code that conditionally includes/transforms
}
```

### Pattern 2: Multi-Array Operations

```rust
fn zip3<U, V, Out, F>(
    &self,
    vec2: &Vector<U>,
    vec3: &Vector<V>,
    f: F,
) -> Vector<Out>
where
    U: Staged,
    V: Staged,
    Out: Staged,
    F: Fn(Rep<T>, Rep<U>, Rep<V>) -> Rep<Out>,
```

### Pattern 3: Accumulation with State

```rust
fn scan<F>(&self, init: Rep<T>, f: F) -> Vector<T>
where
    F: Fn(Rep<T>, Rep<T>) -> Rep<T>,
{
    // Generate: result[i] = f(result[i-1], data[i])
    // Prefix sum, running maximum, etc.
}
```

## Summary

### You DON'T need:
- ❌ RepFn<A, B, C, OUT> for most cases
- ❌ RepCallableN traits
- ❌ Complex function representations

### You DO need:
- ✅ Generic closures: `F: Fn(Rep<T>) -> Rep<U>`
- ✅ Operations that return staged values
- ✅ Building complex operations from simple ones

### The Magic Formula:

```rust
// For N-ary functions, just use Fn with N parameters!
fn my_operation<F>(&self, other: &Vector<U>, f: F) -> Rep<Out>
where
    F: Fn(Rep<T>, Rep<U>) -> Rep<Out>,  // ← This is all you need!
{
    // Generate code that calls f
}
```

**Composability comes from returning staged values, not from complex function types!**

## Try It!

Run the examples:

```bash
cargo run --example rep_higher_order    # See the three approaches
cargo run --example rep_vector_complete  # See your Scala LMS example in Rust!
```

The examples show your exact Scala LMS code working in Rust with the same semantics and composability!
