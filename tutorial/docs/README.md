# Staged Programming in Rust: Documentation Index

This directory contains comprehensive documentation on implementing Scala LMS-style staged programming in Rust.

## Quick Start

1. **Just want the answer?** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. **Want working code?** → Run `cargo run --example rep_vector_complete`
3. **Want details?** → Start with [FUNCTIONS_AND_COMPOSABILITY.md](FUNCTIONS_AND_COMPOSABILITY.md)

## Your Questions Answered

### "Can I implement Rep<T> in Rust?"

**YES!** See:
- [rep_summary.md](rep_summary.md) - Quick overview
- [rep_design.md](rep_design.md) - Complete design discussion
- Examples: `../examples/rep_example.rs`, `../examples/rep_working_demo.rs`

### "How do I represent functions?"

**Use closures!** See:
- [FUNCTIONS_AND_COMPOSABILITY.md](FUNCTIONS_AND_COMPOSABILITY.md) - Complete answer
- [rep_functions.md](rep_functions.md) - Deep dive on three approaches
- [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - TL;DR version
- Examples: `../examples/rep_higher_order.rs`, `../examples/rep_vector_complete.rs`

### "How do I build composability?"

**Chain operations!** See:
- [composable_api_guide.md](composable_api_guide.md) - Complete guide with patterns
- [FUNCTIONS_AND_COMPOSABILITY.md](FUNCTIONS_AND_COMPOSABILITY.md) - Principles and examples
- Example: `../examples/rep_vector_complete.rs` - Full working API

## Documentation Files

### Executive Summaries
- **[QUICK_REFERENCE.md](QUICK_REFERENCE.md)** - One-page cheat sheet for functions
- **[rep_summary.md](rep_summary.md)** - One-page overview of Rep<T>
- **[FUNCTIONS_AND_COMPOSABILITY.md](FUNCTIONS_AND_COMPOSABILITY.md)** - Complete answer to function questions

### Detailed Guides
- **[rep_design.md](rep_design.md)** - Deep dive on Rep<T> design (200+ lines)
  - Phantom types explained
  - Trait system
  - Comparison with concrete types
  - When to use each approach

- **[rep_functions.md](rep_functions.md)** - Understanding staged functions
  - Meta-level vs object-level functions
  - Three approaches to function representation
  - When you need RepFn vs closures

- **[composable_api_guide.md](composable_api_guide.md)** - Building composable APIs
  - Complete patterns for map/filter/reduce
  - Multi-array operations
  - Generic reductions
  - Scala LMS examples translated

## Code Examples

### Working Demonstrations
Located in `../examples/`:

1. **rep_example.rs** - Rep<T> infrastructure
   - Phantom types
   - Trait system
   - Operator overloading
   ```bash
   cargo run --example rep_example
   ```

2. **rep_working_demo.rs** - Proof it actually works!
   - Full compilation pipeline
   - Executes: `f(x) = (x + 5) * 2`
   - Tests multiple inputs
   ```bash
   cargo run --example rep_working_demo
   ```

3. **rep_higher_order.rs** - Three approaches to functions
   - Closures as code generators (recommended)
   - First-class staged functions
   - Trait-based callable types
   ```bash
   cargo run --example rep_higher_order
   ```

4. **rep_vector_complete.rs** - Your Scala LMS example in Rust! ⭐
   - Full Vector<T> implementation
   - foreach, map, filter, sumIf
   - Composable operations
   - Chaining examples
   ```bash
   cargo run --example rep_vector_complete
   ```

### Alternative Tutorial Implementation
- **../src/lib_rep_version.rs** - Complete tutorial rewritten with Rep<T>
  - Shows how to use Rep<T> throughout
  - All original tests pass
  - Demonstrates operator overloading benefits

## Key Concepts

### Rep<T> - Generic Staged Values
A single type that unifies all staged computations:

```rust
// Instead of separate StagedI64, StagedU64, etc.
pub enum Rep<T: Staged> {
    Constant(T::RuntimeValue),
    Variable(Variable),
    BinOp(Box<Rep<T>>, BinOpKind, Box<Rep<T>>),
}

// Use with phantom types:
pub type RepI64 = Rep<I64Type>;
pub type RepU64 = Rep<U64Type>;
```

### Functions - Two Kinds

**Meta-level (code generators):**
```rust
// Scala LMS: f: Rep[T] => Rep[U]
// Rust:      F: Fn(Rep<T>) -> Rep<U>
vec.map(|x| x * 2)  // Closure runs at staging time
```

**Object-level (staged values):**
```rust
// Only if you need functions as runtime values
let lambda = RepFn::lambda(param, body);
```

### Composability

Achieved by:
1. Operations accept closures: `F: Fn(Rep<T>) -> Rep<U>`
2. Operations return staged values: `Vector<T>`, `Rep<T>`
3. Build complex from simple: implement `foreach`, build everything else on top

```rust
vec.filter(|x| x > 0)
   .map(|x| x * 2)
   .sum()  // Chaining works naturally!
```

## Reading Order

### For Impatient Readers
1. [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
2. Run `cargo run --example rep_vector_complete`
3. Done!

### For Thorough Understanding
1. [rep_summary.md](rep_summary.md) - Understand Rep<T>
2. [FUNCTIONS_AND_COMPOSABILITY.md](FUNCTIONS_AND_COMPOSABILITY.md) - Understand functions
3. [rep_design.md](rep_design.md) - Deep dive on design
4. [composable_api_guide.md](composable_api_guide.md) - Build your own API
5. Run all examples in `../examples/`

### For Specific Questions
- **"What is Rep<T>?"** → [rep_summary.md](rep_summary.md)
- **"How do I use Rep<T>?"** → `../examples/rep_working_demo.rs`
- **"How do functions work?"** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **"How do I build APIs?"** → [composable_api_guide.md](composable_api_guide.md)
- **"Why Rep<T> vs concrete types?"** → [rep_design.md](rep_design.md)

## Comparison with Scala LMS

| Scala LMS | Rust Equivalent | Notes |
|-----------|-----------------|-------|
| `Rep[T]` | `Rep<T>` | Generic staged type |
| `f: Rep[T] => Rep[U]` | `F: Fn(Rep<T>) -> Rep<U>` | Meta-level function |
| `Rep[T => U]` | `RepFn<T, U>` | Object-level function (rare) |
| `implicit` | Generic with trait bounds | Type-level programming |
| Method syntax | Method syntax | Same ergonomics! |

**Result:** Nearly identical expressiveness and composability!

## What's Implemented

✅ Rep<T> generic staged type
✅ Phantom types for type safety
✅ Operator overloading (+, -, *, etc.)
✅ Generic compiler (one method for all types)
✅ Meta-level functions (closures as code generators)
✅ Object-level functions (RepFn for staged values)
✅ Trait-based callable types
✅ Composable array operations (foreach, map, filter, reduce)
✅ Multi-array operations (zip, combine)
✅ Chaining and composition
✅ Your exact Scala LMS examples!

## Next Steps

### For the Tutorial
- Current: Concrete types (StagedI64, etc.) - good for beginners
- Consider: Add "Advanced Lesson 6: Rep<T>" after basic lessons
- Use Rep<T> for advanced features (arrays, nested structures)

### For dio
- Rep<T> could unify expression types
- Enable generic array operations
- Support user-defined types via traits
- Maintain concrete types for simple cases
- Offer both APIs for flexibility

## Testing

All examples compile and run:

```bash
# Test Rep<T> infrastructure
cargo test -p tutorial

# Run examples
cargo run --example rep_example
cargo run --example rep_working_demo
cargo run --example rep_higher_order
cargo run --example rep_vector_complete  # ← Your Scala LMS code!
```

## Questions?

All your questions are answered in these documents. If you want:
- Quick answers → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- Complete answers → [FUNCTIONS_AND_COMPOSABILITY.md](FUNCTIONS_AND_COMPOSABILITY.md)
- Working code → Run the examples!
- Deep understanding → Read all the docs in order

## Summary

**Yes, you can do everything Scala LMS does in Rust!**

- ✅ Generic Rep<T> type
- ✅ Functions via closures
- ✅ Full composability
- ✅ Type safety
- ✅ Operator overloading
- ✅ Same ergonomics as Scala LMS

The implementations are in `../examples/` and ready to run!
