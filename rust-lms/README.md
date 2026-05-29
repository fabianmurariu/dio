# rust-lms

**Type-Safe Staged Computation in Rust**

A Rust implementation of multi-stage programming inspired by [Scala LMS](https://scala-lms.github.io/) (Lightweight Modular Staging).

## Features

- ✅ **Compile-time type safety**: Invalid operations caught at compile time
- ✅ **Zero-cost abstractions**: `Var<T>` and `Const<T>` are `Copy` when possible
- ✅ **Heterogeneous operations**: Operations can change output types (e.g., comparison → bool)
- ✅ **Full composability**: Any `Staged` value works anywhere a `Staged` value is expected
- ✅ **Dynamic dispatch support**: Boxing via `.boxed()` when needed
- ✅ **Cranelift backend**: Generates efficient machine code via Cranelift JIT

## Quick Example

```rust
use rust_lms::prelude::*;
use cranelift_frontend::Variable;

// Create variables and constants
let x = Var::<i64>::new(Variable::from_u32(0));
let five = Const::<i64>::new(5);
let two = Const::<i64>::new(2);

// Build expressions: (x + 5) * 2
let expr = mul(add(x, five), two);

// x is Copy, so we can reuse it!
let expr2 = add(x, x);

// Comparisons change type to Bool
let comparison = lt(x, Const::new(100));

// This won't compile - type mismatch caught at compile time!
// let bad = add(x, comparison);  // ERROR: can't add I64 and Bool
```

## Architecture

### Core Concepts

The design separates **values** from **operations**:

- **Values**: `Var<T>`, `Const<T>` - represent pure values (variables/constants)
- **Operations**: `Add<L,R>`, `Lt<L,R>` - separate structs that implement `Staged`

This separation enables:

1. Type-level constraints on operations
2. Heterogeneous operations (changing types)
3. Copy semantics for lightweight values

### Type System

```rust
// Foundation trait
trait Staged {
    type Out: StagedType;
    fn codegen(&self, builder: &mut FunctionBuilder) -> Value;
}

// Type markers
struct i64;    // i64 values
struct U64Type;    // u64 values
struct F64Type;    // f64 values
struct BoolType;   // boolean values
```

### Operations with Type Constraints

```rust
// Add requires both operands to have the same type T that supports addition
impl<L, R, T> Staged for Add<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsAdd,
{
    type Out = T;  // Output is also T
}

// Lt changes the type: inputs are T, output is Bool
impl<L, R, T> Staged for Lt<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsComparison,
{
    type Out = BoolType;  // Always returns Bool!
}
```

## Module Organization

```
rust-lms/
├── src/
│   ├── lib.rs          # Main library entry point
│   ├── staged.rs       # Core Staged trait, Var<T>, Const<T>
│   ├── types.rs        # Type system (StagedType, i64, etc.)
│   └── num/
│       ├── mod.rs      # Numeric operations module
│       ├── traits.rs   # Capability traits (SupportsAdd, etc.)
│       └── ops.rs      # Operation structs (Add, Sub, Lt, etc.)
└── examples/
    └── basic_usage.rs  # Comprehensive example
```

## Why This Design?

### Problem with Traditional Approach

Traditional expression trees mix values and operations:

```rust
enum Rep<T> {
    Constant(T),
    Variable(Variable),
    BinOp(Box<Rep<T>>, BinOpKind, Box<Rep<T>>),  // ❌ Mixed!
}
```

Issues:

- Can't be `Copy` (contains `Box`)
- Hard to enforce type constraints
- Difficult to support heterogeneous operations

### Our Solution

Separate values and operations:

```rust
// Values only
struct Var<T> { var: Variable, ... }     // ✅ Copy when T is Copy!
struct Const<T> { value: T::RuntimeValue, ... }  // ✅ Copy when T is Copy!

// Operations separately
struct Add<L, R> { left: L, right: R }   // ✅ Type constraints via traits!
struct Lt<L, R> { left: L, right: R }    // ✅ Can change output type!
```

Benefits:

- Values can be `Copy`
- Type constraints enforced at compile time
- Operations are first-class values
- Full type safety

## Examples

Run the basic example:

```bash
cargo run --example basic_usage -p rust-lms
```

## Integration with Dio

This library is designed to integrate with the Dio JIT compiler project, providing a type-safe frontend for building staged computations that compile to efficient Cranelift IR.

## License

Part of the Dio project.
