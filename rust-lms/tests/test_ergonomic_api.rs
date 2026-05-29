//! Tests for ergonomic API improvements
//!
//! These tests demonstrate the improved ergonomics from using IntoStaged:
//! - assign(var, 42i64) instead of assign(var, Const::<i64>::new(42))
//! - add(x, 5i64) instead of add(x, Const::<i64>::new(5))
//! - lt(x, 100i64) instead of lt(x, Const::<i64>::new(100))
//! - while_loop(true, ...) instead of while_loop(Const::<BoolType>::new(true), ...)
//! - arr.get_unchecked(0u64) instead of arr.get_unchecked(Const::<U64Type>::new(0))

use rust_lms::prelude::*;

#[test]
fn test_ergonomic_arithmetic() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("arith", |ctx| {
        let x = ctx.let_var(10i64);
        (x, add(*x, 5i64))
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
        add::<i64, _, _>(x, y)
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
            lt::<i64, _, _>(x, 100i64), // Ergonomic lt
            x,
            Const::<i64>::new(100), // Constants in return positions need type annotation
        )
    });

    let compiled = compiler.compile(f).expect("compilation failed");
    let clamp = compiled.as_fn();

    assert_eq!(clamp(50), 50); // Below max
    assert_eq!(clamp(150), 100); // Above max, clamped
}

#[test]
fn test_ergonomic_while_loop() {
    let mut compiler = Compiler::new();

    let count_to_n = compiler.fun1("count_to_n", |ctx, n: Var<i64>| {
        let i = ctx.let_var(0i64);
        let sum = ctx.let_var(0i64);
        (
            i,
            sum,
            while_loop(
                lt::<i64, _, _>(*i, n),
                (assign(*sum, add(*sum, *i)), assign(*i, add(*i, 1i64))),
            ),
            *sum,
        )
    });

    let compiled = compiler.compile(count_to_n).expect("compilation failed");
    let f = compiled.as_fn();

    assert_eq!(f(5), 0 + 1 + 2 + 3 + 4); // Sum of 0..5
    assert_eq!(f(3), 0 + 1 + 2); // Sum of 0..3
    assert_eq!(f(55), (54 * 55) / 2); // Sum of 0..55
}

#[test]
fn test_ergonomic_slice_operations() {
    let mut compiler = Compiler::new();

    // fn get_second(arr: &[i64]) -> i64
    let get_second = compiler.fun1("get_second", |_ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.get_unchecked(1u64) // Ergonomic index - no Const needed!
    });

    let compiled = compiler.compile(get_second).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 5] = [10, 20, 30, 40, 50];
    let slice: &[i64] = &data;

    let result = f(slice);
    assert_eq!(result, 20);
}

#[test]
fn test_ergonomic_slice_set() {
    let mut compiler = Compiler::new();

    // fn set_first(arr: &mut [i64])
    let set_first = compiler.fun1("set_first", |_ctx, arr: Var<SRefMut<Slice<i64>>>| {
        arr.set_unchecked(0u64, 999i64) // Both index and value are ergonomic!
    });

    let compiled = compiler.compile(set_first).expect("compilation failed");
    let f = compiled.as_fn();

    let mut data: [i64; 3] = [10, 20, 30];
    let slice: &mut [i64] = &mut data;

    f(slice);
    assert_eq!(data[0], 999);
}

#[test]
fn test_ergonomic_slice_subslice() {
    let mut compiler = Compiler::new();

    let sum_middle = compiler.fun1("sum_middle", |ctx, arr: Var<SRef<Slice<i64>>>| {
        let i = ctx.let_var(0u64);
        let total = ctx.let_var(0i64);
        let sub = arr.slice_unchecked(1u64, 4u64);
        (
            i,
            total,
            while_loop(
                lt(*i, sub.len()),
                (
                    assign(*total, add(*total, sub.get_unchecked(*i))),
                    assign(*i, add(*i, 1u64)),
                ),
            ),
            *total,
        )
    });

    let compiled = compiler.compile(sum_middle).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 6] = [100, 10, 20, 30, 200, 300];
    let slice: &[i64] = &data;

    // arr[1..4] = [10, 20, 30], sum = 60
    let result = f(slice);
    assert_eq!(result, 60);
}

#[test]
fn test_ergonomic_mixed_operations() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("mixed_ops", |ctx| {
        let x = ctx.let_var(5i64);
        let y = ctx.let_var(10i64);
        (x, y, if_then_else(lt(*x, *y), mul(*x, 2i64), div(*y, 2i64)))
    });

    let compiled = compiler.compile(call0(f)).expect("compilation failed");
    assert_eq!(compiled.run(), 10); // x < y, so x * 2 = 10
}

#[test]
fn test_ergonomic_f64_operations() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("compute", |_ctx, x: Var<F64Type>| {
        add(mul(x, 2.5f64), 3.5f64)
    });

    let compiled = compiler.compile(f).expect("compilation failed");
    let compute = compiled.as_fn();

    assert!((compute(2.0) - 8.5).abs() < 0.0001); // 2.0 * 2.5 + 3.5 = 8.5
}

// =============================================================================
// Imperative Ctx API tests (replaces old staged_block! tests)
// =============================================================================

#[test]
fn test_imperative_basic_var() {
    let mut compiler = Compiler::new();

    // ctx.var() declares + inits inline; no tuple sequencing needed.
    let f = compiler.fun0("imp_basic", |ctx| {
        let x = ctx.var(42i64);
        let y = ctx.var(8i64);
        add::<i64, _, _>(x, y)
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
            ctx.store(sum, add(sum, i));
            ctx.store(i, add(i, 1i64));
        });
        sum
    });

    let compiled = compiler.compile(count_to_n).expect("compilation failed");
    let f = compiled.as_fn();

    assert_eq!(f(5), 0 + 1 + 2 + 3 + 4);
    assert_eq!(f(10), 45);
}

#[test]
fn test_imperative_bind() {
    // ctx.bind evaluates a complex expression once and binds to a Copy Var.
    let mut compiler = Compiler::new();

    let f = compiler.fun1("bind_test_imp", |ctx, n: Var<i64>| {
        let doubled = ctx.bind(add(n, n));
        add(doubled, doubled)
    });

    let compiled = compiler.compile(f).expect("compilation failed");
    let f = compiled.as_fn();
    assert_eq!(f(5), 20); // (5+5) + (5+5)
    assert_eq!(f(3), 12);
}

#[test]
fn test_imperative_fibonacci() {
    let mut compiler = Compiler::new();

    let fib_iter = compiler.fun1("fib_iter_imp", |ctx, n: Var<i64>| {
        let i = ctx.var(2i64);
        let a = ctx.var(0i64);
        let b = ctx.var(1i64);
        let temp = ctx.var(0i64);
        if_then_else(lt(n, 2), n, {
            ctx.while_loop(lt(i, add(n, 1i64)), move |ctx| {
                ctx.store(temp, add(a, b));
                ctx.store(a, b);
                ctx.store(b, temp);
                ctx.store(i, add(i, 1i64));
            });
            b
        })
    });

    let compiled = compiler.compile(fib_iter).expect("compilation failed");
    let fib = compiled.as_fn();

    assert_eq!(fib(0), 0);
    assert_eq!(fib(1), 1);
    assert_eq!(fib(5), 5);
    assert_eq!(fib(10), 55);
}
