//! Integration tests for slice support
//!
//! These tests verify that slices work correctly:
//! - `Var<SRef<Slice<T>>>` = `&[T]` - immutable slice reference
//! - `Var<SRefMut<Slice<T>>>` = `&mut [T]` - mutable slice reference

use rust_lms::prelude::*;
use rust_lms::control::while_loop;
use rust_lms::num::{gt, lt};

#[test]
fn test_slice_len() {
    let mut compiler = Compiler::new();

    // fn get_len(arr: &[i64]) -> u64
    let get_len = compiler.fun1("get_len", |arr: Var<SRef<Slice<I64Type>>>| {
        arr.len()
    });

    let compiled = compiler.compile(get_len).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 5] = [10, 20, 30, 40, 50];
    let slice: &[i64] = &data;

    let result = f(slice);
    assert_eq!(result, 5);
}

#[test]
fn test_slice_len_empty() {
    let mut compiler = Compiler::new();

    let get_len = compiler.fun1("get_len", |arr: Var<SRef<Slice<I64Type>>>| {
        arr.len()
    });

    let compiled = compiler.compile(get_len).expect("compilation failed");
    let f = compiled.as_fn();

    let empty: &[i64] = &[];
    assert_eq!(f(empty), 0);
}

#[test]
fn test_slice_get_unchecked() {
    let mut compiler = Compiler::new();

    // fn get_second(arr: &[i64]) -> i64
    // Returns arr[1]
    let get_second = compiler.fun1("get_second", |arr: Var<SRef<Slice<I64Type>>>| {
        arr.get_unchecked(1u64)
    });

    let compiled = compiler.compile(get_second).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 5] = [10, 20, 30, 40, 50];
    let slice: &[i64] = &data;

    let result = f(slice);
    assert_eq!(result, 20);
}

#[test]
fn test_slice_sum() {
    let mut compiler = Compiler::new();

    // Variables for the loop
    let i = compiler.var_unchecked::<U64Type>();
    let total = compiler.var_unchecked::<I64Type>();

    // fn sum(arr: &[i64]) -> i64
    let sum = compiler.fun1("sum", |arr: Var<SRef<Slice<I64Type>>>| {
        (
            assign(i, 0u64),
            assign(total, 0i64),
            while_loop(
                lt(i, arr.len()),
                (
                    assign(total, add(total, arr.get_unchecked(i))),
                    assign(i, add::<U64Type, _, _>(i, 1u64)),
                )
            ),
            total
        )
    });

    let compiled = compiler.compile(sum).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 5] = [10, 20, 30, 40, 50];
    let slice: &[i64] = &data;

    let result = f(slice);
    assert_eq!(result, 150); // 10 + 20 + 30 + 40 + 50 = 150
}

#[test]
fn test_slice_mutable_set() {
    let mut compiler = Compiler::new();

    // fn set_first(arr: &mut [i64], val: i64)
    // Sets arr[0] = val
    let set_first = compiler.fun1("set_first", |arr: Var<SRefMut<Slice<I64Type>>>| {
        arr.set_unchecked(0u64, 999i64)
    });

    let compiled = compiler.compile(set_first).expect("compilation failed");
    let f = compiled.as_fn();

    let mut data: [i64; 3] = [10, 20, 30];
    let slice: &mut [i64] = &mut data;

    f(slice);
    assert_eq!(data[0], 999);
    assert_eq!(data[1], 20);
    assert_eq!(data[2], 30);
}

#[test]
fn test_slice_mutable_fill() {
    let mut compiler = Compiler::new();

    let i = compiler.var_unchecked::<U64Type>();

    // fn fill_with_42(arr: &mut [i64])
    let fill = compiler.fun1("fill", |arr: Var<SRefMut<Slice<I64Type>>>| {
        (
            assign(i, 0u64),
            while_loop(
                lt::<U64Type, _, _>(i, arr.len()),
                (
                    arr.set_unchecked(i, 42i64),
                    assign(i, add::<U64Type, _, _>(i, 1u64)),
                )
            ),
        )
    });

    let compiled = compiler.compile(fill).expect("compilation failed");
    let f = compiled.as_fn();

    let mut data: [i64; 4] = [0, 0, 0, 0];
    let slice: &mut [i64] = &mut data;

    f(slice);
    assert_eq!(data, [42, 42, 42, 42]);
}

#[test]
fn test_slice_subslice() {
    let mut compiler = Compiler::new();

    // fn sum_middle(arr: &[i64]) -> i64
    // Returns sum of arr[1..4]
    let i = compiler.var_unchecked::<U64Type>();
    let total = compiler.var_unchecked::<I64Type>();

    let sum_middle = compiler.fun1("sum_middle", |arr: Var<SRef<Slice<I64Type>>>| {
        let sub = arr.slice_unchecked(1u64, 4u64);
        (
            assign(i, 0u64),
            assign(total, 0i64),
            while_loop(
                lt::<U64Type, _, _>(i, sub.len()),
                (
                    assign(total, add::<I64Type, _, _>(total, sub.get_unchecked(i))),
                    assign(i, add::<U64Type, _, _>(i, 1u64)),
                )
            ),
            total
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
fn test_slice_sum_all_larger_than_3(){
    let mut compiler = Compiler::new();

    let i = compiler.var_unchecked::<U64Type>();
    let count = compiler.var_unchecked::<U64Type>();

    // fn count_greater_than_3(arr: &[i64]) -> u64
    let count_greater_than_3 = compiler.fun1("count_greater_than_3", |arr: Var<SRef<Slice<I64Type>>>| {
        (
            assign(i, 0u64),
            assign(count, 0u64),
            while_loop(
                lt(i, arr.len()),
                (
                    if_then(
                        gt::<I64Type, _, _>(arr.get_unchecked(i), 3i64),
                        assign(count, add::<U64Type, _, _>(count, 1u64)),
                    ),
                    assign(i, add::<U64Type, _, _>(i, 1u64)),
                )
            ),
            count
        )
    });

    let compiled = compiler.compile(count_greater_than_3).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 5] = [1, 4, 5, 2, 6];
    let slice: &[i64] = &data;

    let result = f(slice);
    assert_eq!(result, 3); // 4, 5, 6 are greater than 3
}

#[test]
fn test_slice_f64() {
    let mut compiler = Compiler::new();

    let i = compiler.var_unchecked::<U64Type>();
    let total = compiler.var_unchecked::<F64Type>();

    // fn sum_f64(arr: &[f64]) -> f64
    let sum_f64 = compiler.fun1("sum_f64", |arr: Var<SRef<Slice<F64Type>>>| {
        (
            assign(i, 0u64),
            assign(total, 0.0f64),
            while_loop(
                lt(i, arr.len()),
                (
                    assign(total, add(total, arr.get_unchecked(i))),
                    assign(i, add::<U64Type, _, _>(i, 1u64)),
                )
            ),
            total
        )
    });

    let compiled = compiler.compile(sum_f64).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [f64; 4] = [1.5, 2.5, 3.0, 4.0];
    let slice: &[f64] = &data;

    let result = f(slice);
    assert!((result - 11.0).abs() < 0.0001);
}

#[test]
fn test_slice_return_subslice_len() {
    let mut compiler = Compiler::new();

    // fn get_half_len(arr: &[i64]) -> u64
    // Returns length of first half of array
    let get_half_len = compiler.fun1("get_half_len", |arr: Var<SRef<Slice<I64Type>>>| {
        let half = div::<U64Type, _, _>(arr.len(), 2u64);
        let sub = arr.slice_unchecked(0u64, half);
        sub.len()
    });

    let compiled = compiler.compile(get_half_len).expect("compilation failed");
    let f = compiled.as_fn();

    let data: [i64; 10] = [0; 10];
    let slice: &[i64] = &data;

    let result = f(slice);
    assert_eq!(result, 5);
}
