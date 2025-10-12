# Getting Started with the Tutorial

This quick start guide gets you up and running with the partial evaluation compiler tutorial.

## Prerequisites

- Rust toolchain (1.70+)
- Basic understanding of Rust syntax
- Familiarity with the concept of compilation

No prior knowledge of compilers, JIT, or partial evaluation is required!

## Running the Tutorial

From the dio project root:

```bash
# See all tests and current status
cargo test -p tutorial

# Expected output:
# - Lesson 1 tests: PASS (3/3) ✅
# - Lesson 2 tests: FAIL (0/2) - Your turn!
# - Lesson 3 tests: FAIL (0/2) - Your turn!
# - Lesson 4 tests: FAIL (0/1) - Your turn!
# - Lesson 5 tests: FAIL (0/1) - Your turn!
```

## Learning Path

### Step 1: Study the Working Example

Read `tutorial/src/lib.rs` starting from Lesson 1. This shows a complete working example of:
- The `StagedI64` type
- The `Staged` trait implementation
- The `Compiler` that generates machine code
- Tests that verify the compiled code works

**Time: 15-20 minutes**

**Goal**: Understand the pattern you'll repeat for other operations.

### Step 2: Implement Subtraction (Lesson 2)

Find the `TODO` for `StagedI64::sub()`:

```rust
pub fn sub(left: StagedI64, right: StagedI64) -> Self {
    // TODO: YOUR CODE HERE
    todo!("Implement subtraction for StagedI64")
}
```

**Hints**:
1. Add a `Sub` variant to the `StagedI64` enum (around line 107)
2. Implement the `sub` method following the `add` pattern
3. Handle the `Sub` case in `codegen()` using `builder.ins().isub()`

**Time: 10-15 minutes**

**Verify**: Run `cargo test -p tutorial test_lesson2` - both tests should pass!

### Step 3: Implement Multiplication (Lesson 3)

Similar to subtraction, but you'll add the `Mul` variant and use `builder.ins().imul()`.

**Time: 10-15 minutes**

**Verify**: Run `cargo test -p tutorial test_lesson3` - both tests should pass!

### Step 4: Implement Unsigned Integers (Lesson 4)

Create a new type `StagedU64` following the `StagedI64` pattern:

```rust
#[derive(Debug, Clone)]
pub enum StagedU64 {
    Constant(u64),
    Variable(Variable),
    Add(Box<StagedU64>, Box<StagedU64>),
}
```

Then:
1. Implement `Staged` trait
2. Add `compile_unary_u64()` to `Compiler`
3. Create `CompiledUnaryU64` struct

**Time: 20-30 minutes**

**Verify**: Run `cargo test -p tutorial test_lesson4`

### Step 5: Implement Booleans (Lesson 5)

Create `StagedBool` with comparisons and logical operations:

```rust
#[derive(Debug, Clone)]
pub enum StagedBool {
    Constant(bool),
    LessThan(Box<StagedI64>, Box<StagedI64>),
    // ... more variants
}
```

This is the most complex lesson!

**Time: 30-45 minutes**

**Verify**: Run `cargo test -p tutorial test_lesson5`

## Total Time Estimate

- Reading and understanding: 15-20 minutes
- Implementing lessons: 1.5-2 hours
- Experimentation and debugging: 30 minutes

**Total: ~2-3 hours** for a complete understanding of staged compilation!

## Key Files

```
tutorial/
├── Cargo.toml           # Dependencies
├── README.md            # Comprehensive guide
└── src/
    └── lib.rs          # Tutorial code and tests
```

All the code, explanations, and tests are in `lib.rs`. It's designed to be read top to bottom!

## Testing Strategy

### Run All Tests
```bash
cargo test -p tutorial
```

### Run Specific Lesson
```bash
cargo test -p tutorial test_lesson1
cargo test -p tutorial test_lesson2
# etc.
```

### See Detailed Output
```bash
cargo test -p tutorial -- --nocapture
```

This shows:
- Which tests passed/failed
- Assertion values
- Any debug output

### Debug a Single Test
```bash
cargo test -p tutorial test_lesson1_constant_addition -- --exact --nocapture
```

## Understanding Test Output

### Passing Test (Example)
```
test tests::test_lesson1_constant_addition ... ok
```

This means your code:
1. Compiled successfully
2. Generated correct Cranelift IR
3. Executed and produced expected results

### Failing Test (Exercise)
```
test tests::test_lesson2_simple_subtraction - should panic ... ok
```

The `should panic` means the test expects to fail (because you haven't implemented it yet). When you implement it correctly, it will show as just `ok` without the `should panic`.

### Error Messages

If you see:
```
error: Type mismatch: expected I64, got I8
```

This means you're mixing types incorrectly (e.g., using a bool where an integer is expected).

## Debugging Tips

### 1. Print Generated IR

Add this to a test:

```rust
#[test]
fn debug_my_code() {
    use cranelift_codegen::Context;

    // ... your code ...

    println!("Generated IR:\n{}", ctx.func.display());
}
```

### 2. Check Compilation Errors

The compiler tells you what's wrong:
- `Type mismatch` - using wrong types
- `not yet implemented` - you need to implement something
- `Compilation failed` - invalid Cranelift IR

### 3. Start Simple

Don't try to implement everything at once:

1. Add the enum variant
2. Make it compile (use `todo!()` in codegen)
3. Implement codegen
4. Run the tests

### 4. Compare with Working Examples

When stuck:
- Look at how `Add` is implemented
- Follow the same pattern for your operation
- Check the Cranelift instruction reference in the README

## Common Issues

### Issue: "not yet implemented" panic

**Solution**: You need to implement the `todo!()` marker. Remove the `todo!()` and add real code.

### Issue: Type mismatch error

**Solution**: Make sure you're using the right Cranelift types:
- `types::I64` for i64/u64
- `types::I8` for booleans

### Issue: Test still marked "should panic"

**Solution**: Update the test to remove the `#[should_panic]` attribute once you've implemented the feature.

## What You'll Learn

By completing this tutorial, you'll understand:

1. **Staging** - Building code generators instead of computing values
2. **Partial Evaluation** - Specializing code by fixing some inputs
3. **JIT Compilation** - Generating machine code at runtime
4. **Futamura Projections** - How specialization creates compilers
5. **Cranelift IR** - Low-level representation before machine code

## Next Steps

After completing the tutorial:

1. **Extend it** - Add more operations (division, modulo, shifts)
2. **Study dio4** - See production staging in `dio4/src/staging.rs`
3. **Read the papers** - Understand the theory behind partial evaluation
4. **Build something** - Create your own staged DSL!

## Getting Help

If you're stuck:

1. Read the comments in `lib.rs` - they contain detailed hints
2. Study Lesson 1 - it's a complete working example
3. Check `tutorial/README.md` - comprehensive reference
4. Look at dio4 - production examples

## Success Indicators

You've mastered the tutorial when you can:

- ✅ Explain the difference between staged and immediate computation
- ✅ Implement new operations following the pattern
- ✅ Read and understand Cranelift IR
- ✅ Debug compilation errors
- ✅ Extend the type system with new types

Then you're ready to build real compilers!

## Philosophy

This tutorial teaches **learning by doing**. Rather than reading about compilers, you're building one! The tests guide you, the examples show you the pattern, and the hints keep you on track.

**Remember**: Every expert was once a beginner. Take it one step at a time, and you'll understand these powerful concepts before you know it!

Happy hacking! 🚀
