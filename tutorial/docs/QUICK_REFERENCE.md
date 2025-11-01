# Quick Reference: Staged Functions in Rust

## TL;DR

**Q:** How do I represent functions like `RepFn<A, B, C, OUT>`?

**A:** You don't! Use Rust closures: `F: Fn(Rep<A>, Rep<B>, Rep<C>) -> Rep<OUT>`

## Scala LMS → Rust Translation

| Scala LMS | Rust | Meaning |
|-----------|------|---------|
| `f: Rep[T] => Rep[U]` | `F: Fn(Rep<T>) -> Rep<U>` | Meta-level function (code generator) |
| `Rep[T => U]` | `RepFn<T, U>` | Object-level function (staged value) |

## The Pattern

### ✅ DO THIS (Simple, Composable)

```rust
// For single-arg functions:
fn map<F, Out>(&self, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>) -> Rep<Out>,

// For two-arg functions:
fn zip<F, U, Out>(&self, other: &Vector<U>, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>, Rep<U>) -> Rep<Out>,

// For three-arg functions:
fn combine3<F, U, V, Out>(&self, vec2: &Vector<U>, vec3: &Vector<V>, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>, Rep<U>, Rep<V>) -> Rep<Out>,
```

### ❌ DON'T DO THIS (Overengineered)

```rust
// You don't need these!
enum RepFn<A, B, C, OUT> { ... }
trait RepCallable1<A, OUT> { ... }
trait RepCallable2<A, B, OUT> { ... }
// etc.
```

## Your Scala LMS Examples

### foreach

**Scala:**
```scala
def foreach(f: Rep[T] => Rep[Unit]): Rep[Unit]
```

**Rust:**
```rust
fn foreach<F>(&self, f: F) -> RepUnit
where
    F: Fn(Rep<T>) -> RepUnit,
```

### sumIf

**Scala:**
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
    self.foreach(|x| {
        let condition = predicate(x.clone());
        // if condition { sum += x }
    });
    // return sum
}
```

## Composability = Chaining

```rust
// This just works!
let result = vec
    .filter(|x| x.lt(RepI64::constant(100)))
    .map(|x| x.mul(RepI64::constant(2)))
    .sum_if(|x| x.lt(RepI64::constant(50)));
```

**Why?** Because each operation returns a staged value (`Vector<T>` or `Rep<T>`), which can be used by the next operation.

## The Key Insight

**Closures are code generators, not staged code!**

```rust
vec.map(|x| x.mul(RepI64::constant(2)))
       ^                ^
       |                |
       |                +-- This builds an AST at staging time
       +------------------- This closure runs at staging time
```

The closure doesn't exist in the generated code - it runs during compilation to *produce* the code!

## When You Actually Need RepFn

Only if you need **runtime function values**:

```rust
// Rare: passing functions as data at runtime
enum StagedExpr {
    Apply(RepFn<I64Type, I64Type>, RepI64),
    Lambda(RepFn<I64Type, I64Type>),
}

// Most cases: just use closures!
fn map<F>(&self, f: F) -> Vector<Out>
where
    F: Fn(Rep<T>) -> Rep<Out>,  // ← This is enough!
```

## API Design Checklist

- [ ] Operations accept `F: Fn(Rep<T>) -> Rep<U>`
- [ ] Operations return staged values (`Rep<T>`, `Vector<T>`, etc.)
- [ ] Basic operations (foreach, reduce) implemented
- [ ] Complex operations built from basic ones
- [ ] Operations can be chained naturally

## Examples

See working code in:
- `tutorial/examples/rep_higher_order.rs` - Three approaches to functions
- `tutorial/examples/rep_vector_complete.rs` - Your Scala LMS example in Rust!

Run:
```bash
cargo run --example rep_vector_complete
```

## Further Reading

- `tutorial/docs/rep_functions.md` - Detailed explanation of function representations
- `tutorial/docs/composable_api_guide.md` - Complete guide to building composable APIs
- `tutorial/docs/rep_design.md` - Deep dive on Rep<T> design
