# Functions and Composability in Staged Rust: Complete Answer

## Your Questions

1. **Is there a way to represent functions?** Like `RepFn<A, B, C, OUT>`?
2. **Would `impl Fn(Rep<A>, Rep<B>, Rep<C>) -> Rep<OUT>` work?**
3. **Do we need a different trait? RepCallableN with N implementations?**
4. **How do I build composability in this API?**

## The Answers

### 1. Do you need RepFn?

**For most cases: NO!**

Your Scala LMS examples use `Rep[T] => Rep[U]`, which is a **meta-level function** (code generator), not a staged value. In Rust, regular closures provide exactly this:

```rust
// Scala LMS
def foreach(f: Rep[T] => Rep[Unit]): Rep[Unit]

// Rust - DIRECT TRANSLATION!
fn foreach<F>(&self, f: F) -> RepUnit
where
    F: Fn(Rep<T>) -> RepUnit,
```

### 2. Does `impl Fn(...)` work?

**YES! Perfectly!**

```rust
// One argument:
fn map<F, Out>(&self, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>) -> Rep<Out>,

// Two arguments:
fn zip_with<F, U, Out>(&self, other: &Vector<U>, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>, Rep<U>) -> Rep<Out>,

// Three arguments:
fn combine3<F, U, V, Out>(
    &self,
    vec2: &Vector<U>,
    vec3: &Vector<V>,
    f: F,
) -> Vector<Out>
where
    F: Fn(Rep<T>, Rep<U>, Rep<V>) -> Rep<Out>,
```

This is **exactly** what Scala LMS does! The closure runs at staging time to generate code.

### 3. Do you need RepCallableN traits?

**NO!** Just use `Fn(...)` directly.

The only time you'd need a custom trait is if you want to:
- Abstract over both closures AND custom types
- Build trait objects
- Have more control over the interface

But for 99% of cases, `F: Fn(...)` is perfect.

### 4. How to build composability?

**Three principles:**

1. **Operations accept closures:** `F: Fn(Rep<T>) -> Rep<U>`
2. **Operations return staged values:** `Rep<T>` or `Vector<T>`
3. **Build complex from simple:** Use `foreach` to implement `map`, `filter`, `sum_if`, etc.

Example:
```rust
// Basic building block
fn foreach<F>(&self, f: F) -> RepUnit
where
    F: Fn(Rep<T>) -> RepUnit;

// Build sum_if using foreach (COMPOSITION!)
fn sum_if<F>(&self, predicate: F) -> RepI64
where
    F: Fn(RepI64) -> RepBool,
{
    self.foreach(|x| {
        if predicate(x) {
            // accumulate
        }
    })
}

// Now chain operations!
vec.filter(|x| x > 0)
   .map(|x| x * 2)
   .sum_if(|x| x < 100)
```

## Working Example: Your Scala LMS Code in Rust!

**Scala LMS:**
```scala
class Vector[T](val data: Rep[Array[T]]) {
  def foreach(f: Rep[T] => Rep[Unit]): Rep[Unit] =
    for (i <- 0 until data.length) f(data(i))

  def sumIf(f: Rep[T] => Rep[Boolean]) = {
    var n = zero[T]
    foreach(x => if (f(x)) n += x)
    n
  }
}
```

**Rust (DIRECT TRANSLATION):**
```rust
pub struct Vector<T: Staged> {
    // array and length references
}

impl Vector<I64Type> {
    pub fn foreach<F>(&self, f: F) -> RepUnit
    where
        F: Fn(RepI64) -> RepUnit,
    {
        // Generate: for (i = 0; i < length; i++) f(data[i])
    }

    pub fn sum_if<F>(&self, predicate: F) -> RepI64
    where
        F: Fn(RepI64) -> RepBool,
    {
        // var sum = 0
        self.foreach(|x| {
            // if (predicate(x)) sum += x
        });
        // return sum
    }
}
```

**Usage (looks like Scala!):**
```rust
let vec = Vector::new(...);

// Simple foreach
vec.foreach(|x| {
    // process each element
});

// Composable operations!
let sum = vec.sum_if(|x| x.lt(RepI64::constant(10)));

// Chaining!
let result = vec
    .filter(|x| x.lt(RepI64::constant(100)))
    .map(|x| x.mul(RepI64::constant(2)))
    .sum();
```

## The Critical Distinction

### Meta-Level Functions (What Scala LMS Uses)

```rust
// The closure is a CODE GENERATOR
vec.map(|x| x.mul(RepI64::constant(2)))
//      ^^
//      This runs at STAGING TIME to build the computation

// After staging, you get generated code like:
// for (i = 0; i < length; i++)
//     result[i] = data[i] * 2;
```

The closure doesn't exist in the final code!

### Object-Level Functions (Rare)

```rust
// A function as a RUNTIME VALUE
let lambda = RepFn::lambda(param, body);
let result = lambda.call(arg);

// After staging, this becomes an actual function call in generated code
```

You only need this if functions are first-class values in your language.

## Complete API Example

```rust
pub struct Vector<T: Staged> {
    // ...
}

impl<T: Staged> Vector<T> {
    /// Core iteration
    pub fn foreach<F>(&self, f: F) -> RepUnit
    where
        F: Fn(Rep<T>) -> RepUnit,
    {
        // Generate: for each element, call f
    }

    /// Transform elements
    pub fn map<F, Out>(&self, f: F) -> Vector<Out>
    where
        F: Fn(Rep<T>) -> Rep<Out>,
        Out: Staged,
    {
        // Generate: new array with f(each element)
    }

    /// Filter elements
    pub fn filter<F>(&self, predicate: F) -> Vector<T>
    where
        F: Fn(Rep<T>) -> RepBool,
    {
        // Generate: new array with elements where predicate is true
    }

    /// Generic reduction
    pub fn reduce<F>(&self, zero: Rep<T>, f: F) -> Rep<T>
    where
        F: Fn(Rep<T>, Rep<T>) -> Rep<T>,
    {
        // Generate: acc = zero; for each x: acc = f(acc, x)
    }

    /// Zip two arrays
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
        // Generate: for i: result[i] = f(self[i], other[i])
    }
}

// Specialized operations
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
        self.foreach(|x| {
            if predicate(x) {
                // accumulate
            }
        });
        // return sum
    }
}
```

## Composability in Action

```rust
fn complex_pipeline(data: Vector<I64Type>) -> RepI64 {
    data
        // Filter: keep only positive values
        .filter(|x| RepI64::constant(0).lt(x))

        // Map: square each value
        .map(|x| x.mul(x.clone()))

        // Filter: keep values < 1000
        .filter(|x| x.lt(RepI64::constant(1000)))

        // Sum: add them all up
        .sum()
}

fn multi_array_operation(
    vec1: Vector<I64Type>,
    vec2: Vector<I64Type>,
) -> Vector<I64Type> {
    // Zip two arrays with addition
    vec1.zip(&vec2, |x, y| x.add(y))
}

fn custom_reduction(data: Vector<I64Type>) -> RepI64 {
    // Generic reduce can implement anything!
    data.reduce(RepI64::constant(i64::MIN), |max, x| {
        // max = if x > max then x else max
        // (simplified - you'd use if_then_else in real code)
        x
    })
}
```

## Run The Examples!

We've implemented your exact Scala LMS examples:

```bash
# See your Vector class in Rust!
cargo run --example rep_vector_complete

# See three approaches to functions
cargo run --example rep_higher_order
```

Output shows:
- ✅ `foreach` working
- ✅ `map` working
- ✅ `filter` working
- ✅ `sumIf` working (your example!)
- ✅ Chained operations: `filter().map().sumIf()`
- ✅ Full composability

## Summary

| Question | Answer |
|----------|--------|
| Need RepFn<A,B,C,OUT>? | **No** - use `F: Fn(Rep<A>, Rep<B>, Rep<C>) -> Rep<OUT>` |
| Does `impl Fn(...)` work? | **Yes!** - Perfect for meta-level functions |
| Need RepCallableN? | **No** - `Fn(...)` handles all arities |
| How to build composability? | Return staged values, accept closures, chain operations |

## Key Insight

**Scala LMS's `Rep[T] => Rep[U]` is a meta-level function (code generator), not a staged value!**

In Rust, this is just: `F: Fn(Rep<T>) -> Rep<U>`

No need for complex machinery - Rust's closure system provides exactly what you need!

## Documentation Index

- `QUICK_REFERENCE.md` - One-page cheat sheet
- `rep_functions.md` - Deep dive on function representations
- `composable_api_guide.md` - Complete guide with examples
- `rep_design.md` - Rep<T> design and philosophy
- `rep_summary.md` - Rep<T> quick start

## Examples

- `examples/rep_higher_order.rs` - Three approaches to functions
- `examples/rep_vector_complete.rs` - Your Scala LMS Vector in Rust!
- `examples/rep_working_demo.rs` - Proof that Rep<T> compiles and runs

---

**Bottom line:** Rust's closure system + type system give you the same power as Scala LMS, with the same composability, and simpler syntax!
