# Staged Functions: Rep&lt;T&gt; and Higher-Order Programming

## The Challenge

How do we represent functions in a staged setting? In Scala LMS:

```scala
class Vector[T](val data: Rep[Array[T]]) {
  def foreach(f: Rep[T] => Rep[Unit]): Rep[Unit] = ...
  def sumIf(f: Rep[T] => Rep[Boolean]) = ...
}
```

The key question: What does `Rep[T] => Rep[Unit]` mean, and how do we do it in Rust?

## Three Approaches

### Approach 1: Closures as Code Generators (Scala LMS Style)
**Key insight:** `f: Rep[T] => Rep[Unit]` is NOT a staged function value. It's a meta-level function (code generator).

### Approach 2: First-Class Staged Functions (Defunctionalization)
**Key insight:** Represent functions as data structures that can be staged.

### Approach 3: Trait-Based Callable (Extensible)
**Key insight:** Define a trait for things that can be called, implement for specific functions.

## Approach 1: Closures as Code Generators ⭐ (Recommended)

This is what Scala LMS actually does!

```rust
impl<T: Staged> Rep<Array<T>> {
    /// foreach takes a closure that generates code for each element
    /// f: Rep[T] => Rep[Unit] in Scala
    /// f: Fn(Rep<T>) -> Rep<Unit> in Rust
    pub fn foreach<F>(&self, f: F) -> Rep<Unit>
    where
        F: Fn(Rep<T>) -> Rep<Unit>,
    {
        // Generate a loop that calls f on each element
        // f is a code generator, not staged code itself!
    }
}
```

**Critical distinction:**
- `Fn(Rep<T>) -> Rep<Unit>` = "code generator" (meta-level)
- `Rep<Fn(T) -> Unit>` = "staged function" (object-level)

The Scala LMS example uses the first form! The function `f` exists at staging time, not at runtime.

## Approach 2: First-Class Staged Functions

If you want functions as runtime values (lambdas), you need to reify them:

```rust
// Represent a staged function as data
pub enum RepFn<In, Out>
where
    In: Staged,
    Out: Staged,
{
    // A named function (refers to a compiled function)
    Named(FunctionId),

    // An inline lambda (captured as an expression tree)
    Lambda {
        param: Variable,
        body: Box<Rep<Out>>,
    },

    // A closure (captures environment)
    Closure {
        param: Variable,
        body: Box<Rep<Out>>,
        captures: Vec<Rep<In>>,
    },
}

impl<In, Out> RepFn<In, Out>
where
    In: Staged,
    Out: Staged,
{
    pub fn call(&self, arg: Rep<In>) -> Rep<Out> {
        // Generate function call code
    }
}
```

## Approach 3: Trait-Based Callable

```rust
// A trait for things that can be called
pub trait RepCallable<Args, Out> {
    fn call(&self, args: Args) -> Out;
}

// Implement for specific arities
pub trait RepCallable1<A, Out> {
    fn call(&self, a: A) -> Out;
}

pub trait RepCallable2<A, B, Out> {
    fn call(&self, a: A, b: B) -> Out;
}

// Implement for closures
impl<F, A, Out> RepCallable1<Rep<A>, Rep<Out>> for F
where
    F: Fn(Rep<A>) -> Rep<Out>,
    A: Staged,
    Out: Staged,
{
    fn call(&self, a: Rep<A>) -> Rep<Out> {
        self(a)
    }
}
```

## Comparison

| Approach | Pros | Cons | Use Case |
|----------|------|------|----------|
| **Closures as Code Generators** | Simple, natural, Scala LMS style | Functions don't exist at runtime | Most cases: map, filter, foreach |
| **First-Class Staged Functions** | Functions are runtime values | Complex, requires reification | Need to pass functions as data |
| **Trait-Based** | Extensible, type-safe | Boilerplate for each arity | Custom callable types |

## Recommendation for Different Scenarios

### For `foreach`, `map`, `filter` → Use Closures (Approach 1)

These don't need runtime function values - they're compile-time code generators.

### For function composition, currying → Use First-Class (Approach 2)

When you need to build and pass around functions as values.

### For custom operators → Use Traits (Approach 3)

When you have domain-specific callable types.

## Critical Insight: Two Levels

Understanding multi-stage programming requires distinguishing:

1. **Meta-level (staging time):** Where code generation happens
2. **Object-level (runtime):** Where generated code runs

```rust
// Meta-level function (code generator)
fn generate_loop<F>(f: F) -> Rep<Unit>
where
    F: Fn(Rep<i64>) -> Rep<Unit>  // ← Meta-level
{
    // f is a Rust closure that exists at staging time
    // It takes Rep<i64> and produces Rep<Unit>
    // The generated code will NOT contain f itself
}

// Object-level function (staged value)
fn create_staged_function() -> Rep<Function> {
    // This would be a function that exists at RUNTIME
    // in the generated code
}
```

Scala LMS primarily uses meta-level functions!

## Next Steps

Let me show working examples of all three approaches...
