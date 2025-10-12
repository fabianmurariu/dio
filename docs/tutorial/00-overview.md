# Tutorial: Building a Partial Evaluation Compiler with Futamura Projections

## Overview

This tutorial teaches you how to build a **partial evaluation compiler** using **Futamura projections** in Rust, similar to Scala LMS but targeting Cranelift for code generation. You'll learn by building a progressively more sophisticated JIT compiler from scratch.

## What You'll Learn

1. **Staged Computation** - Understanding the difference between compile-time and runtime values
2. **Futamura Projections** - How specialization creates optimized code
3. **Partial Evaluation** - Reducing computation by fixing some inputs at compile time
4. **JIT Compilation** - Generating machine code at runtime using Cranelift
5. **Multi-Stage Programming** - Building abstractions for code generation

## Architecture Summary

### dio3 - SSA IR Compiler
- **Purpose**: Compiles expressions to SSA IR then to machine code
- **Pipeline**: AST → SSA IR → Cranelift → Machine Code
- **Key Files**:
  - `ast.rs` - Expression AST with types
  - `ssa.rs` - SSA v2 IR (Static Single Assignment)
  - `cranelift_backend.rs` - Cranelift code generation
  - `execution.rs` - JIT execution runtime

### dio4 - Staged Compilation Framework
- **Purpose**: Multi-stage programming with callback-based operators
- **Pipeline**: Staged Values → Cranelift IR → Machine Code (direct, no SSA)
- **Key Files**:
  - `staging.rs` - Core staging types (`StagedU64`, `StagedBool`, etc.)
  - `operators.rs` - Callback-based query operators
  - `pipeline.rs` - Query pipeline construction
  - `compiler.rs` - JIT compilation

### Key Differences

**dio3** uses an intermediate SSA representation - you build an AST, convert to SSA, then compile to Cranelift:
```
Expr → SSA IR → Cranelift IR → Machine Code
```

**dio4** uses direct staging - your Rust code generates Cranelift IR directly at "compile time":
```
Staged Rust Code → Cranelift IR → Machine Code
```

dio4 is closer to Scala LMS in that staging happens in the host language (Rust), not through an intermediate IR.

## The Three Futamura Projections

Given:
- `prog` - A program to run
- `input` - Runtime input data
- `interpreter(prog, input)` - Executes prog with input

### First Projection
**Specialize an interpreter with a program**

```
compiler = specialize(interpreter, prog)
result = compiler(input)
```

You fix the program, get a compiled function that just needs input.

### Second Projection
**Create a compiler from an interpreter**

```
compiler_generator = specialize(interpreter, interpreter)
compiler = compiler_generator(prog)
```

Specialize the interpreter with itself to get a compiler generator.

### Third Projection
**Create a compiler generator from an interpreter**

```
compiler_generator_generator = specialize(interpreter, specialize)
compiler_generator = compiler_generator_generator(interpreter)
```

This is what we're building! A framework where you write staged code and get specialized machine code.

## Tutorial Structure

This tutorial follows a learning-by-doing approach:

1. **Lesson 1-3**: Basic Expression Language (Scalars)
   - Constants and variables
   - Arithmetic operations
   - Type system (i64, u64, bool)

2. **Lesson 4-6**: Arrays and Loops (Coming in future prompts)
   - Array types and indexing
   - Loop generation
   - Conditions and control flow

3. **Lesson 7-9**: Functions and Optimization (Coming in future prompts)
   - Function calls
   - Inlining and specialization
   - SIMD operations

## The `tutorial` Crate

Located in `tutorial/`, this crate provides:

- **Working examples** - Complete implementations with passing tests
- **Student exercises** - Incomplete implementations with failing tests
- **Progressive difficulty** - Each lesson builds on previous concepts
- **Test-driven learning** - Fix tests to learn concepts

## Running the Tutorial

```bash
# Run all tests to see what needs to be implemented
cargo test -p tutorial

# Run tests for a specific lesson
cargo test -p tutorial test_lesson1

# Run tests with debug output
cargo test -p tutorial -- --nocapture
```

Each failing test teaches a concept. Your goal: make all tests pass!

## Key Concepts

### Staging

**Two-level execution**:
- **Compile time (now)**: Build Cranelift IR
- **Runtime (later)**: Execute compiled code

```rust
// Compile time: construct IR
let x = StagedI64::constant(42);
let y = StagedI64::variable(var);
let sum = x.add(y); // This builds IR, doesn't compute!

// Runtime: execute compiled function
let result = compiled_func(10); // Returns 52
```

### Partial Evaluation

Fix some inputs at compile time to specialize code:

```rust
// General: add(x, y)
fn add(x: i64, y: i64) -> i64 { x + y }

// Specialized: add_42(y) where x=42
fn add_42(y: i64) -> i64 { 42 + y }
```

The specialized version has one less parameter and can be optimized better!

### Cranelift IR

Low-level representation before machine code:

```
function add(i64, i64) -> i64 {
block0(v0: i64, v1: i64):
    v2 = iadd v0, v1
    return v2
}
```

Our staging framework generates this automatically from Rust code!

## Next Steps

Start with `tutorial/src/lib.rs` and work through the lessons. Each lesson has:
1. An explanation of the concept
2. A working example with a passing test
3. Student exercises with failing tests

Good luck! Remember: compilation is just another program, and partial evaluation is how we make it fast.

## References

- [Futamura Projections Explained](https://blog.sigplan.org/2015/05/15/futamura-projections/)
- [Scala LMS](https://scala-lms.github.io/) - Similar multi-stage programming in Scala
- [Cranelift](https://cranelift.dev/) - Code generator we target
- Original dio4 design: `docs/dio4.md`
