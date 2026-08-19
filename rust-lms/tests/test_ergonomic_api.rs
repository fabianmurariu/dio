//! Tests for ergonomic API improvements
//!
//! These tests demonstrate the improved ergonomics from using IntoStaged plus
//! std::ops impls:
//! - `ctx.store(var, 42i64)` instead of `assign(var, Const::<i64>::new(42))`
//! - `x + 5i64` instead of `add(x, Const::<i64>::new(5))`
//! - `lt(x, 100i64)` instead of `lt(x, Const::<i64>::new(100))`
//! - `while_loop(true, ...)` instead of `while_loop(Const::<bool>::new(true), ...)`
//! - `arr.get_unchecked(0u64)` instead of `arr.get_unchecked(Const::<u64>::new(0))`

use rust_lms::prelude::*;

#[test]
fn test_ergonomic_arithmetic() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("arith", |ctx| {
        let x = ctx.let_var(10i64);
        (x, *x + 5i64)
    });

    let compiled = compiler.compile(call0(f)).expect("compilation failed");
    assert_eq!(compiled.run(), 15);
}

#[test]
fn test_ergonomic_let_var() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("let_var_test", |ctx| {
        let x = ctx.var(42i64);
        let y = ctx.var(8i64);
        x + y
    });

    let compiled = compiler.compile(call0(f)).expect("compilation failed");
    assert_eq!(compiled.run(), 50);
}

#[test]
fn test_ergonomic_comparison() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("clamp_max", |_ctx, x: Var<i64>| {
        // Ergonomic comparison and conditional
        if_then_else(
            lt(x, 100i64),
            x,
            Const::<i64>::new(100), // Constants in return positions need type annotation
        )
    });

    let compiled = compiler.compile(f).expect("compilation failed");
    let clamp = compiled.as_fn();

    assert_eq!(clamp.call(50), 50); // Below max
    assert_eq!(clamp.call(150), 100); // Above max, clamped
}

#[test]
fn test_ergonomic_while_loop() {
    let mut compiler = Compiler::new();

    let count_to_n = compiler.fun1("count_to_n", |ctx, n: Var<i64>| {
        let i = ctx.var(0i64);
        let sum = ctx.var(0i64);
        ctx.while_loop(lt(i, n), move |ctx| {
            ctx.store(sum, sum + i);
            ctx.store(i, i + 1i64);
        });
        sum
    });

    let compiled = compiler.compile(count_to_n).expect("compilation failed");
    let f = compiled.as_fn();

    assert_eq!(f.call(5), 0 + 1 + 2 + 3 + 4); // Sum of 0..5
    assert_eq!(f.call(3), 0 + 1 + 2); // Sum of 0..3
    assert_eq!(f.call(55), (54 * 55) / 2); // Sum of 0..55
}

#[test]
fn test_ergonomic_slice_operations() {
    let mut compiler = Compiler::new();

    // fn get_second(arr: &[i64]) -> i64
    let get_second = compiler.fun1("get_second", |_ctx, arr: Var<SRef<Slice<i64>>>| {
        // SAFETY: this test calls the kernel only with slices of length >= 2.
        unsafe { arr.get_unchecked(1u64) } // Ergonomic index - no Const needed!
    });

    let compiled = compiler.compile(get_second).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 5] = [10, 20, 30, 40, 50];
    let slice: &[i64] = &data;

    let result = f.call(slice);
    assert_eq!(result, 20);
}

#[test]
fn test_ergonomic_slice_set() {
    let mut compiler = Compiler::new();

    // fn set_first(arr: &mut [i64])
    let set_first = compiler.fun1("set_first", |_ctx, arr: Var<SRefMut<Slice<i64>>>| {
        // SAFETY: this test calls the kernel only with non-empty slices.
        unsafe { arr.set_unchecked(0u64, 999i64) } // Both arguments are ergonomic.
    });

    let compiled = compiler.compile(set_first).expect("compilation failed");
    let f = compiled.as_fn();

    let mut data: [i64; 3] = [10, 20, 30];
    let slice: &mut [i64] = &mut data;

    f.call(slice);
    assert_eq!(data[0], 999);
}

#[test]
fn test_ergonomic_slice_subslice() {
    let mut compiler = Compiler::new();

    let sum_middle = compiler.fun1("sum_middle", |ctx, arr: Var<SRef<Slice<i64>>>| {
        let i = ctx.var(0u64);
        let total = ctx.var(0i64);
        // SAFETY: this test calls the kernel only with slices of length >= 4.
        let sub = unsafe { arr.slice_unchecked(1u64, 4u64) };
        ctx.while_loop(lt(i, sub.clone().len()), move |ctx| {
            // SAFETY: the loop condition proves `i < sub.len()`.
            ctx.store(total, total + unsafe { sub.clone().get_unchecked(i) });
            ctx.store(i, i + 1u64);
        });
        total
    });

    let compiled = compiler.compile(sum_middle).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 6] = [100, 10, 20, 30, 200, 300];
    let slice: &[i64] = &data;

    // arr[1..4] = [10, 20, 30], sum = 60
    let result = f.call(slice);
    assert_eq!(result, 60);
}

#[test]
fn test_ergonomic_mixed_operations() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("mixed_ops", |ctx| {
        let x = ctx.var(5i64);
        let y = ctx.var(10i64);
        if_then_else(lt(x, y), x * 2i64, y / 2i64)
    });

    let compiled = compiler.compile(call0(f)).expect("compilation failed");
    assert_eq!(compiled.run(), 10); // x < y, so x * 2 = 10
}

#[test]
fn test_ergonomic_f64_operations() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("compute", |_ctx, x: Var<f64>| x * 2.5f64 + 3.5f64);

    let compiled = compiler.compile(f).expect("compilation failed");
    let compute = compiled.as_fn();

    assert!((compute.call(2.0) - 8.5).abs() < 0.0001); // 2.0 * 2.5 + 3.5 = 8.5
}

// =============================================================================
// Imperative Ctx API tests
// =============================================================================

#[test]
fn test_imperative_basic_var() {
    let mut compiler = Compiler::new();

    // ctx.var() declares + inits inline; no tuple sequencing needed.
    let f = compiler.fun0("imp_basic", |ctx| {
        let x = ctx.var(42i64);
        let y = ctx.var(8i64);
        x + y
    });

    let compiled = compiler.compile(call0(f)).expect("compilation failed");
    assert_eq!(compiled.run(), 50);
}

#[test]
fn test_imperative_while_loop() {
    let mut compiler = Compiler::new();

    let count_to_n = compiler.fun1("count_to_n_imp", |ctx, n: Var<i64>| {
        let i = ctx.var(0i64);
        let sum = ctx.var(0i64);
        ctx.while_loop(lt(i, n), move |ctx| {
            ctx.store(sum, sum + i);
            ctx.store(i, i + 1i64);
        });
        sum
    });

    let compiled = compiler.compile(count_to_n).expect("compilation failed");
    let f = compiled.as_fn();

    assert_eq!(f.call(5), 0 + 1 + 2 + 3 + 4);
    assert_eq!(f.call(10), 45);
}

#[test]
fn test_imperative_bind() {
    // ctx.bind evaluates a complex expression once and binds to a Copy Var.
    let mut compiler = Compiler::new();

    let f = compiler.fun1("bind_test_imp", |ctx, n: Var<i64>| {
        let doubled = ctx.bind(n + n);
        doubled + doubled
    });

    let compiled = compiler.compile(f).expect("compilation failed");
    let f = compiled.as_fn();
    assert_eq!(f.call(5), 20); // (5+5) + (5+5)
    assert_eq!(f.call(3), 12);
}

#[test]
fn test_imperative_fibonacci() {
    let mut compiler = Compiler::new();

    let fib_iter = compiler.fun1("fib_iter_imp", |ctx, n: Var<i64>| {
        let i = ctx.var(2i64);
        let a = ctx.var(0i64);
        let b = ctx.var(1i64);
        let temp = ctx.var(0i64);
        if_then_else(lt(n, 2i64), n, {
            ctx.while_loop(lt(i, n + 1i64), move |ctx| {
                ctx.store(temp, a + b);
                ctx.store(a, b);
                ctx.store(b, temp);
                ctx.store(i, i + 1i64);
            });
            b
        })
    });

    let compiled = compiler.compile(fib_iter).expect("compilation failed");
    let fib = compiled.as_fn();

    assert_eq!(fib.call(0), 0);
    assert_eq!(fib.call(1), 1);
    assert_eq!(fib.call(5), 5);
    assert_eq!(fib.call(10), 55);
}
