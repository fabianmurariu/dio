//! Tests for ergonomic API improvements
//!
//! These tests demonstrate the improved ergonomics from using IntoStaged:
//! - assign(var, 42i64) instead of assign(var, Const::<I64Type>::new(42))
//! - add(x, 5i64) instead of add(x, Const::<I64Type>::new(5))
//! - lt(x, 100i64) instead of lt(x, Const::<I64Type>::new(100))
//! - while_loop(true, ...) instead of while_loop(Const::<BoolType>::new(true), ...)
//! - arr.get_unchecked(0u64) instead of arr.get_unchecked(Const::<U64Type>::new(0))

use rust_lms::prelude::*;

#[test]
fn test_ergonomic_arithmetic() {
    let mut compiler = Compiler::new();

    let x = compiler.let_var(10i64);

    // All arithmetic operations accept primitives directly
    let expr = (x, add(*x, 5i64)); // Ergonomic add;

    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 15);
}

#[test]
fn test_ergonomic_let_var() {
    let mut compiler = Compiler::new();

    // let_var returns InitVar directly - no tuple unpacking needed!
    let x = compiler.let_var(42i64);
    let y = compiler.let_var(8i64);

    // x and y automatically initialize when used in the tuple, use *x to get Var
    let expr = (x, y, add::<I64Type, _, _>(*x, *y));

    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 50);
}

#[test]
fn test_ergonomic_comparison() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("clamp_max", |_ctx, x: Var<I64Type>| {
        // Ergonomic comparison and conditional
        if_then_else(
            lt::<I64Type, _, _>(x, 100i64), // Ergonomic lt
            x,
            Const::<I64Type>::new(100), // Constants in return positions need type annotation
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

    let i = compiler.let_var(0i64);
    let sum = compiler.let_var(0i64);

    let count_to_n = compiler.fun1("count_to_n", |_ctx, n: Var<I64Type>| {
        (
            (i, sum),
            while_loop(
                lt::<I64Type, _, _>(*i, n), // Ergonomic condition
                (
                    assign(*sum, add(*sum, *i)),
                    assign(*i, add(*i, 1i64)), // Ergonomic increment
                ),
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
    let get_second = compiler.fun1("get_second", |_ctx, arr: Var<SRef<Slice<I64Type>>>| {
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
    let set_first = compiler.fun1("set_first", |_ctx, arr: Var<SRefMut<Slice<I64Type>>>| {
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

    let i = compiler.let_var(0u64);
    let total = compiler.let_var(0i64);

    // fn sum_middle(arr: &[i64]) -> i64
    let sum_middle = compiler.fun1("sum_middle", |_ctx, arr: Var<SRef<Slice<I64Type>>>| {
        let sub = arr.slice_unchecked(1u64, 4u64); // Ergonomic slice indices!
        (
            (i, total),
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

    let x = compiler.let_var(5i64);
    let y = compiler.let_var(10i64);

    // Mix of all ergonomic features - variables auto-initialize
    let expr = (
        x,
        y,
        if_then_else(
            lt(*x, *y),
            mul(*x, 2i64),
            div(*y, 2i64),
        ),
    );

    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 10); // x < y, so x * 2 = 10
}

#[test]
fn test_ergonomic_f64_operations() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("compute", |_ctx, x: Var<F64Type>| {
        add(
            mul(x, 2.5f64),
            3.5f64,
        )
    });

    let compiled = compiler.compile(f).expect("compilation failed");
    let compute = compiled.as_fn();

    assert!((compute(2.0) - 8.5).abs() < 0.0001); // 2.0 * 2.5 + 3.5 = 8.5
}
