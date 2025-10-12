# Tutorial: Building a Partial Evaluation Compiler

Welcome! This hands-on tutorial teaches you how to build a **JIT compiler** using **partial evaluation** and **Futamura projections** in Rust. You'll create a system similar to [Scala LMS](https://scala-lms.github.io/) that generates optimized machine code via Cranelift.

## What You'll Build

A **multi-stage programming** framework where:
1. You write Rust code that describes computations (staging)
2. Your code generates Cranelift IR at "compile time"
3. Cranelift compiles that IR to machine code
4. The result is a fast, specialized function

This is the **first Futamura projection**: specializing an interpreter (our staging framework) with a program (your expressions) to get a compiler!

## Quick Start

```bash
# See what you need to implement
cargo test -p tutorial

# Work on a specific lesson
cargo test -p tutorial test_lesson1
cargo test -p tutorial test_lesson2

# See detailed output
cargo test -p tutorial -- --nocapture
```

## Tutorial Structure

### ✅ Lesson 1: Simple Addition (Example - Complete)
**Concept**: Staging basics - building code generators instead of computing

**What you learn**:
- The `Staged` trait for code generation
- `StagedI64` for representing integer computations
- `Compiler` for JIT compilation
- The difference between compile-time and runtime

**Status**: All tests pass! Study this as your example.

### 📝 Lesson 2: Constants (Exercise - Your Turn!)
**Concept**: Partial evaluation - fixing inputs at compile time

**Your task**: Implement `StagedI64::sub()` for subtraction

**What you learn**:
- How constants enable optimization
- Specializing functions by fixing parameters
- The power of partial evaluation

**Key insight**: `f(x) = 100 - 42` can be compiled with NO runtime inputs!

### 📝 Lesson 3: Variables (Exercise - Your Turn!)
**Concept**: Combining constants and variables for flexible specialization

**Your task**:
1. Add `Mul` variant to `StagedI64` enum
2. Implement `StagedI64::mul()`
3. Handle `Mul` in the `codegen()` method

**What you learn**:
- Variables represent unknown runtime values
- Mixing constants and variables creates specialized code
- How to extend the expression language

**Example**: `f(x) = (x + 5) * (x - 2)` - constants 5 and 2 are baked in!

### 📝 Lesson 4: Mixed Type Operations (Exercise - Your Turn!)
**Concept**: Multiple types in a staged language

**Your task**:
1. Implement `StagedU64` enum (like `StagedI64` but for `u64`)
2. Implement the `Staged` trait for `StagedU64`
3. Add `compile_unary_u64()` method to `Compiler`
4. Create `CompiledUnaryU64` struct

**What you learn**:
- Different types need different representations
- Type-specific code generation
- Expanding the type system

### 📝 Lesson 5: Boolean Operations (Exercise - Your Turn!)
**Concept**: Comparisons and logical operations for control flow

**Your task**:
1. Implement `StagedBool` enum with:
   - `Constant(bool)`
   - `LessThan(StagedI64, StagedI64)`
   - `GreaterThan(StagedI64, StagedI64)`
   - `Equal(StagedI64, StagedI64)`
   - `And`, `Or`, `Not` for logical operations
2. Implement the `Staged` trait (bools are `i8` in Cranelift)
3. Add comparison methods
4. Add `compile_unary_i64_to_bool()` to `Compiler`

**What you learn**:
- Booleans as `i8` values (0 or 1)
- Comparison operations (`icmp` in Cranelift)
- Logical operations (`band`, `bor`, `bnot`)
- Foundation for conditionals (coming in future lessons)

## Learning Strategy

1. **Read the working example** (Lesson 1) carefully
2. **Study the pattern**: Enum variants → codegen implementation
3. **Follow the hints** in the TODO comments
4. **Run tests frequently** to get feedback
5. **Check the Cranelift IR** in test output to understand generated code

## Key Concepts

### Staging

**The Big Idea**: Instead of computing values immediately, build a data structure that describes the computation. Later, compile this description to machine code.

```rust
// Immediate computation (normal programming)
let x = 5 + 3;  // Computes 8 right now

// Staged computation (our framework)
let five = StagedI64::constant(5);
let three = StagedI64::constant(3);
let sum = StagedI64::add(five, three);  // Builds a description!

// Later: compile the description to machine code
let compiled = compiler.compile(...);
let result = compiled.call(x);  // NOW it computes!
```

### Partial Evaluation

**The Big Idea**: If you know some inputs at compile time, you can specialize the code for those specific values.

```rust
// General function (takes 2 inputs)
fn add(x: i64, y: i64) -> i64 { x + y }

// Specialized function (x is fixed to 42)
fn add_42(y: i64) -> i64 { 42 + y }
```

The specialized version:
- Has fewer parameters (only `y`, not `x`)
- Can be optimized better (the constant 42 is visible)
- Runs faster (one less memory access)

This is **partial evaluation** - we "partially evaluated" the function with `x=42`.

### Futamura Projections

**The Big Idea**: Compilers are just programs that transform other programs. We can apply partial evaluation to compilers themselves!

**First Projection** (what we're building):
```
interpreter(program, input) → result
specialize(interpreter, program) → compiled_program
compiled_program(input) → result  // Faster!
```

Fix the program, get a specialized function for that program.

**Second Projection** (future work):
```
specialize(interpreter, interpreter) → compiler
```

Specialize the interpreter with itself to get a compiler generator!

**Third Projection** (advanced):
```
specialize(specialize, interpreter) → compiler_generator
```

This generates compiler generators automatically!

### Cranelift IR

**The Target Language**: Our staged computations generate Cranelift IR, which looks like:

```
function add_five(i64) -> i64 {
block0(v0: i64):          ; parameter
    v1 = iconst.i64 5     ; constant 5
    v2 = iadd v0, v1      ; add them
    return v2             ; return result
}
```

Your staging code automatically generates this!

## Cranelift Instruction Reference

Here are the key instructions you'll use:

### Constants
- `iconst.i64 <value>` - Create a 64-bit integer constant
- `iconst.i8 <value>` - Create an 8-bit integer constant (for bools)

### Arithmetic (Signed)
- `iadd <a> <b>` - Add two integers
- `isub <a> <b>` - Subtract two integers
- `imul <a> <b>` - Multiply two integers
- `sdiv <a> <b>` - Signed division
- `srem <a> <b>` - Signed remainder (modulo)

### Arithmetic (Unsigned)
- `udiv <a> <b>` - Unsigned division
- `urem <a> <b>` - Unsigned remainder

### Comparisons
- `icmp eq <a> <b>` - Equal
- `icmp ne <a> <b>` - Not equal
- `icmp slt <a> <b>` - Signed less than
- `icmp sle <a> <b>` - Signed less than or equal
- `icmp sgt <a> <b>` - Signed greater than
- `icmp sge <a> <b>` - Signed greater than or equal
- `icmp ult <a> <b>` - Unsigned less than
- `icmp ugt <a> <b>` - Unsigned greater than

### Logical Operations
- `band <a> <b>` - Bitwise AND
- `bor <a> <b>` - Bitwise OR
- `bxor <a> <b>` - Bitwise XOR
- `bnot <a>` - Bitwise NOT

### Control Flow (Future lessons)
- `br <block>` - Unconditional branch
- `brif <cond> <then_block> <else_block>` - Conditional branch
- `jump <block> <args>` - Jump to block with arguments
- `return <value>` - Return from function

## Debugging Tips

### Enable Cranelift IR output

Modify a test to print the generated IR:

```rust
#[test]
fn debug_my_code() {
    let mut compiler = Compiler::new().unwrap();
    let mut ctx = Context::new();

    // ... build your function in ctx.func ...

    println!("Generated IR:\n{}", ctx.func.display());
}
```

### Check compilation errors

The errors tell you what's wrong:

```rust
error: Type mismatch: expected I64, got I8
```

This means you're trying to use a bool (I8) where an integer (I64) is expected.

### Use smaller tests

Don't try to implement everything at once! Start with:
1. Just the enum variant
2. Just the constructor function
3. Just the codegen case
4. Then the full test

## What's Next?

After completing these 5 lessons, you'll be ready for:

### Lesson 6-8: Arrays and Loops
- `StagedArray<T>` for array types
- Loop code generation
- Index bounds checking
- The root function signature: `(input_arrays, input_scalars, output_array, length)`

### Lesson 9-11: Control Flow
- `if-then-else` conditionals
- `while` loops
- `for` loops
- Block parameters for SSA form

### Lesson 12-14: Functions
- Function calls and inlining
- Closures and higher-order functions
- Recursive functions
- Tail call optimization

### Lesson 15+: Advanced Topics
- SIMD operations
- Memory layout optimization
- Auto-vectorization
- Query operator fusion (like dio4)

## Architecture Connection

This tutorial builds on the dio3 and dio4 codebases:

### dio3 Style (AST → SSA → Machine Code)
```
Expr → SsaProgram → Cranelift IR → Machine Code
```

### dio4 Style (Direct Staging - What We're Building)
```
Staged Rust Code → Cranelift IR → Machine Code
```

Our tutorial follows the dio4 approach: your Rust code directly generates Cranelift IR through the `Staged` trait's `codegen()` method.

### Comparison

| Aspect | dio3 | dio4 | tutorial |
|--------|------|------|----------|
| Intermediate IR | SSA | None | None |
| Entry point | Parse AST | Staging API | Staging API |
| Flexibility | Fixed AST | Extensible | Extensible |
| Complexity | Higher | Lower | Lowest |
| Use case | Query language | Query operators | Learning |

The tutorial is the simplest form - perfect for learning the concepts!

## Common Mistakes

### 1. Computing instead of staging

```rust
// ❌ WRONG - This computes at compile time
fn wrong(x: i64) -> i64 {
    x + 5  // This is immediate computation!
}

// ✅ RIGHT - This stages the computation
fn right(x: Variable) -> StagedI64 {
    let x_staged = StagedI64::variable(x);
    let five = StagedI64::constant(5);
    StagedI64::add(x_staged, five)  // Description of computation!
}
```

### 2. Wrong Cranelift type

```rust
// ❌ WRONG - Using I32 for i64
builder.ins().iconst(types::I32, value)

// ✅ RIGHT - Use I64 for i64
builder.ins().iconst(types::I64, value)
```

### 3. Forgetting to clone

```rust
// ❌ WRONG - Can't use `x` twice, it moved!
let sum = StagedI64::add(x, x);

// ✅ RIGHT - Clone staged values
let sum = StagedI64::add(x.clone(), x);
```

### 4. Wrong instruction for types

```rust
// ❌ WRONG - Using signed compare for unsigned
// (when working with StagedU64)
builder.ins().icmp(IntCC::SignedLessThan, a, b)

// ✅ RIGHT - Use unsigned compare for unsigned
builder.ins().icmp(IntCC::UnsignedLessThan, a, b)
```

## Resources

### Documentation
- [Tutorial Overview](../docs/tutorial/00-overview.md) - Conceptual introduction
- [Cranelift Docs](https://docs.rs/cranelift/) - IR and API reference
- [Scala LMS](https://scala-lms.github.io/) - Similar multi-stage programming

### Code Examples
- `dio4/src/staging.rs` - Production staging implementation
- `dio4/src/operators.rs` - Advanced operator fusion
- `dio3/src/cranelift_backend.rs` - SSA to Cranelift compilation

### Academic Papers
- [Futamura Projections](https://www.brics.dk/~hosc/local/HOSC-11-4-pp381-391.pdf) - Original paper
- [LMS: Lightweight Modular Staging](https://scala-lms.github.io/) - Scala LMS papers

## Getting Help

1. **Read the comments** in `src/lib.rs` - they contain detailed explanations
2. **Study the working example** (Lesson 1) - it shows the complete pattern
3. **Check the Cranelift docs** - for instruction reference
4. **Look at dio4** - for production examples of staging

## Success Criteria

You've completed the tutorial when:
- ✅ All 9 tests pass
- ✅ You understand the `Staged` trait
- ✅ You can extend the language with new operations
- ✅ You understand partial evaluation vs. full evaluation
- ✅ You can read Cranelift IR

Then you're ready to build real compilers!

## Next Steps After Completion

1. **Extend the language** - Add division, modulo, bitwise ops
2. **Add more types** - f64, i32, u32, etc.
3. **Implement arrays** - Start working on Lessons 6-8
4. **Study dio4** - See how it builds query operators
5. **Read LMS papers** - Understand the theory deeper

Good luck! Remember: **Programming is just data, and compilers are just programs that transform data.**
