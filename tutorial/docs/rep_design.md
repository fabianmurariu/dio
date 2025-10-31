# Rep&lt;T&gt; Design: Scala LMS-style Staging in Rust

## Overview

This document explores implementing a generic `Rep<T>` abstraction in Rust, similar to Scala LMS's `Rep[T]`. We compare it to the current concrete-types approach used in the tutorial.

## The Question

Can we have a generic `Rep<T>` in Rust where:
1. You can add two `Rep<T>` variables together using `+` operator
2. Addition only works if there's a codegen implementation for type `T`
3. Assignment and other operations work naturally
4. The compiler becomes more generic
5. All tutorial tests pass

**Short answer: YES! This is absolutely possible and quite elegant in Rust.**

## Two Approaches

### Approach 1: Concrete Types (Current Tutorial)

```rust
// Separate type for each staged type
pub enum StagedI64 {
    Constant(i64),
    Variable(Variable),
    Add(Box<StagedI64>, Box<StagedI64>),
    Sub(Box<StagedI64>, Box<StagedI64>),
}

// Explicit methods
impl StagedI64 {
    pub fn add(left: StagedI64, right: StagedI64) -> Self {
        StagedI64::Add(Box::new(left), Box::new(right))
    }
}

// Usage
let result = StagedI64::add(x, y);
```

### Approach 2: Generic Rep&lt;T&gt; (LMS-style)

```rust
// Single generic type for all staged computations
pub enum Rep<T: Staged> {
    Constant(T::RuntimeValue),
    Variable(Variable),
    BinOp(Box<Rep<T>>, BinOpKind, Box<Rep<T>>),
}

// Phantom types as markers
pub struct I64Type;
pub struct U64Type;

// Type alias for convenience
pub type RepI64 = Rep<I64Type>;

// Operator overloading
impl<T: SupportsBinOp> Add for Rep<T> {
    type Output = Rep<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Add, Box::new(rhs))
    }
}

// Usage
let result = x + y;  // Natural syntax!
```

## Detailed Design: How Rep&lt;T&gt; Works

### 1. Phantom Types as Type Tags

The key insight is using **phantom types** as type-level tags:

```rust
// These types have zero size and exist only at compile time
#[derive(Clone)]
pub struct I64Type;

#[derive(Clone)]
pub struct U64Type;

#[derive(Clone)]
pub struct BoolType;
```

These are **marker types** - they carry no data but provide type information to the compiler. They're similar to Rust's `PhantomData` but even simpler.

### 2. The Staged Trait

Types must implement `Staged` to work with `Rep<T>`:

```rust
pub trait Staged: 'static + Clone {
    // The actual runtime type (e.g., i64 for I64Type)
    type RuntimeValue: Clone;

    // Cranelift type representation
    fn cranelift_type() -> cranelift_codegen::ir::Type;

    // How to generate code for constants
    fn codegen_constant(value: &Self::RuntimeValue, builder: &mut FunctionBuilder) -> Value;
}
```

Example implementation:

```rust
impl Staged for I64Type {
    type RuntimeValue = i64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn codegen_constant(value: &i64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value)
    }
}
```

### 3. Conditional Operations via Traits

We only implement operations for types that support them:

```rust
// Not all types support binary operations
pub trait SupportsBinOp: Staged {
    fn codegen_binop(
        kind: BinOpKind,
        left: Value,
        right: Value,
        builder: &mut FunctionBuilder,
    ) -> Value;
}

// Now we can conditionally implement Add
impl<T: SupportsBinOp> Add for Rep<T> {
    type Output = Rep<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Add, Box::new(rhs))
    }
}
```

**This provides type safety!** You can add `RepI64` values, but not `RepBool` values (unless we implement `SupportsBinOp` for `BoolType`, which we don't).

### 4. Generic Compiler

The compiler becomes completely generic:

```rust
impl Compiler {
    // ONE method that works for ALL types!
    pub fn compile_nary<T>(
        &mut self,
        num_params: usize,
        body: impl FnOnce(&mut FunctionBuilder, &[Variable]) -> Rep<T>,
    ) -> Result<CompiledNary<T>, StagingError>
    where
        T: SupportsBinOp,
    {
        // Generic compilation logic...
        let result_expr = body(&mut builder, &param_vars);
        let result_val = result_expr.codegen(&mut builder);
        // ...
    }
}
```

Instead of `compile_nary_i64`, `compile_nary_u64`, etc., we have one `compile_nary<T>` that works for all types!

## Answering Your Specific Questions

### Q1: Can we add two Rep&lt;T&gt; variables using `+`?

**Yes!** We implement the `Add` trait:

```rust
impl<T: SupportsBinOp> Add for Rep<T> {
    type Output = Rep<T>;
    fn add(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Add, Box::new(rhs))
    }
}

// Usage:
let x = RepI64::variable(vars[0]);
let y = RepI64::constant(5);
let result = x + y;  // Works!
```

### Q2: Does addition only work when there's a codegen implementation?

**Yes!** This is enforced at compile time via trait bounds:

```rust
// Add is only implemented for types that implement SupportsBinOp
impl<T: SupportsBinOp> Add for Rep<T> { ... }

// I64Type implements SupportsBinOp
impl SupportsBinOp for I64Type {
    fn codegen_binop(...) -> Value { ... }
}

// BoolType does NOT implement SupportsBinOp
// So Rep<BoolType> does NOT implement Add
// This code won't compile:
let b1 = RepBool::constant(true);
let b2 = RepBool::constant(false);
let result = b1 + b2;  // ERROR: no implementation of `Add` for `Rep<BoolType>`
```

### Q3: Does assignment work?

**Yes!** `Rep<T>` is just a regular Rust type:

```rust
let x = RepI64::variable(vars[0]);
let five = RepI64::constant(5);
let sum = x + five;          // assignment works
let product = sum * two;     // chaining works
return product;              // returning works
```

Because `Rep<T>` derives `Clone`, we can also clone values:

```rust
let x = RepI64::variable(vars[0]);
let square = x.clone() * x;  // Need to clone since * consumes self
```

### Q4: What traits must T implement?

At minimum:
- `Staged` - Core trait defining how to stage type `T`
- `'static` - T must not contain references (required for trait objects)
- `Clone` - Allows cloning the marker type

For operations:
- `SupportsBinOp` - For arithmetic operations (+, -, *, /)
- Custom traits like `SupportsComparison` for `<`, `>`, `==`, etc.

```rust
// Minimal example:
#[derive(Clone)]
pub struct MyCustomType;

impl Staged for MyCustomType {
    type RuntimeValue = i32;
    fn cranelift_type() -> cranelift_codegen::ir::Type { types::I32 }
    fn codegen_constant(value: &i32, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I32, *value as i64)
    }
}

impl SupportsBinOp for MyCustomType {
    fn codegen_binop(kind: BinOpKind, left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        match kind {
            BinOpKind::Add => builder.ins().iadd(left, right),
            // ... other operations
        }
    }
}

// Now Rep<MyCustomType> works!
type RepMyCustom = Rep<MyCustomType>;
```

### Q5: Does it make the compiler more generic?

**Absolutely!** Compare:

**Before (concrete types):**
```rust
fn compile_nary_i64(...) -> Result<CompiledNaryI64, ...> { ... }
fn compile_nary_u64(...) -> Result<CompiledNaryU64, ...> { ... }
fn compile_nary_f64(...) -> Result<CompiledNaryF64, ...> { ... }
// ... one method per type
```

**After (generic):**
```rust
fn compile_nary<T: SupportsBinOp>(
    ...
) -> Result<CompiledNary<T>, ...>
where
    T: SupportsBinOp,
{
    // ONE implementation that works for ALL types!
}

// Convenience methods:
fn compile_nary_i64(...) -> Result<CompiledNaryI64, ...> {
    self.compile_nary::<I64Type>(...)  // Delegates to generic version
}
```

### Q6: Is it applicable to the tutorial?

**Yes, but with tradeoffs.**

## Comparison: Pros and Cons

### Concrete Types (Current Approach)

**Pros:**
- ✅ **Simpler for beginners**: No phantom types, no trait bounds
- ✅ **Clearer error messages**: Errors mention `StagedI64` directly
- ✅ **Explicit control**: Easy to add type-specific operations
- ✅ **Gradual learning**: Each lesson adds one concept at a time
- ✅ **No advanced Rust features**: No need to understand phantom types

**Cons:**
- ❌ **Code duplication**: Similar code for `StagedI64`, `StagedU64`, etc.
- ❌ **Verbose operations**: `StagedI64::add(x, y)` instead of `x + y`
- ❌ **Less generic compiler**: Need separate compile methods per type
- ❌ **Not idiomatic Rust**: Missing operator overloading

### Generic Rep&lt;T&gt; (LMS Approach)

**Pros:**
- ✅ **More generic**: One type, one compiler method for all types
- ✅ **Better ergonomics**: `x + y` instead of `StagedI64::add(x, y)`
- ✅ **Less code duplication**: Shared implementation across types
- ✅ **Extensible**: Easy to add new types by implementing traits
- ✅ **Idiomatic Rust**: Uses operator overloading and trait system
- ✅ **More powerful**: Can write generic functions over all staged types

**Cons:**
- ❌ **Steeper learning curve**: Requires understanding phantom types
- ❌ **Complex error messages**: Errors mention `Rep<I64Type>` and trait bounds
- ❌ **More advanced Rust**: Needs trait bounds, associated types, phantoms
- ❌ **Might confuse beginners**: What is `I64Type`? Why is it in `Rep<>`?

## Usage Comparison

### Creating Values

**Concrete:**
```rust
let x = StagedI64::variable(vars[0]);
let five = StagedI64::constant(5);
```

**Generic:**
```rust
let x = RepI64::variable(vars[0]);
let five = RepI64::constant(5);
// Or more explicitly:
let x = Rep::<I64Type>::variable(vars[0]);
```

### Operations

**Concrete:**
```rust
let sum = StagedI64::add(x, five);
let product = StagedI64::mul(sum, two);
```

**Generic:**
```rust
let sum = x + five;
let product = sum * two;
// Or even:
let result = (x + five) * two;
```

### Generic Functions

**Concrete:**
```rust
// Can't write generic functions easily
fn square_i64(x: StagedI64) -> StagedI64 {
    StagedI64::mul(x.clone(), x)
}

fn square_u64(x: StagedU64) -> StagedU64 {
    StagedU64::mul(x.clone(), x)
}
```

**Generic:**
```rust
// One function for all types!
fn square<T: SupportsBinOp>(x: Rep<T>) -> Rep<T> {
    x.clone() * x
}

// Works for any type:
let squared_i64 = square(RepI64::constant(5));
let squared_u64 = square(RepU64::constant(5));
```

## Test Compatibility

Both approaches can pass the same tests! Here's a side-by-side:

**Concrete:**
```rust
#[test]
fn test_lesson1_constant_addition() {
    let mut compiler = Compiler::new().unwrap();
    let compiled = compiler
        .compile_nary_i64(1, |_builder, vars| {
            let x = StagedI64::variable(vars[0]);
            let five = StagedI64::constant(5);
            StagedI64::add(x, five)
        })
        .unwrap();

    assert_eq!(compiled.call(&[10]), 15);
}
```

**Generic:**
```rust
#[test]
fn test_lesson1_constant_addition() {
    let mut compiler = Compiler::new().unwrap();
    let compiled = compiler
        .compile_nary_i64(1, |_builder, vars| {
            let x = RepI64::variable(vars[0]);
            let five = RepI64::constant(5);
            x + five  // Natural operator syntax!
        })
        .unwrap();

    assert_eq!(compiled.call(&[10]), 15);
}
```

The changes are minimal! Just:
1. `StagedI64` → `RepI64`
2. `StagedI64::add(x, five)` → `x + five`

## Recommendation for the Tutorial

I'd recommend a **hybrid approach**:

### Phase 1: Start with Concrete Types (Lessons 1-3)
- Easier for beginners to grasp
- Clear connection between concepts and code
- No "magic" - everything is explicit

### Phase 2: Introduce Rep&lt;T&gt; (Advanced Lesson)
- After students understand staging fundamentals
- Show the more elegant abstraction
- Demonstrate Rust's type system power
- Could be "Lesson 6: Advanced Abstractions"

### Structure:
```
Lesson 1-5: Concrete types (StagedI64, etc.)
  ↓
Lesson 6: "Advanced: Generic Staging with Rep<T>"
  - Explain phantom types
  - Show how to unify StagedI64, StagedU64 under Rep<T>
  - Demonstrate operator overloading
  - Compare both approaches
  ↓
Lesson 7+: Use Rep<T> for advanced features
  - Arrays: Rep<ArrayType<T>>
  - Functions: Rep<FnType<Args, Ret>>
  - Complex types benefit more from generics
```

## Real-World Application

This design is directly applicable to your `dio` project! You could:

1. **Keep current concrete types for core expressions**
   - They're working well for basic use cases
   - Clear and debuggable

2. **Add Rep&lt;T&gt; for advanced features**
   - Generic array operations
   - User-defined types
   - Complex staging scenarios

3. **Provide both APIs**
   - Concrete types for simple cases
   - Generic types for advanced users

## Further Reading

- **Scala LMS Papers**: "Lightweight Modular Staging" by Rompf & Odersky
- **Rust Phantom Types**: [Rust Book - PhantomData](https://doc.rust-lang.org/std/marker/struct.PhantomData.html)
- **Multi-stage Programming**: MetaOCaml, Terra, etc.

## Conclusion

Yes, Rust can absolutely support a Scala LMS-style `Rep<T>` abstraction! The type system is powerful enough to express:

1. ✅ Generic staged computations
2. ✅ Operator overloading (`x + y`)
3. ✅ Conditional trait implementations (operations only when supported)
4. ✅ Type-safe codegen
5. ✅ Generic compiler

The main question is **pedagogical**: Is this the right abstraction for teaching beginners?

- For a tutorial aimed at teaching staging concepts: **Start concrete**
- For a tutorial aimed at advanced Rust features: **Use Rep&lt;T&gt;**
- For production code: **Consider both**, depending on use case

The provided `lib_rep_version.rs` shows a complete working implementation that passes all the tutorial tests!