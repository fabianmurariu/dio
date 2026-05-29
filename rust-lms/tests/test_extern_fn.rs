//! Integration tests for external function calling via #[extern_fn]

use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

// =============================================================================
// Simple external functions
// =============================================================================

/// Simple addition function
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_add(x: i64, y: i64) -> i64 {
    x + y
}

/// Simple multiplication
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_mul(x: i64, y: i64) -> i64 {
    x * y
}

/// Square a number
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_square(x: i64) -> i64 {
    x * x
}

/// Function with no return value
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_noop() {
    // Do nothing
}

// =============================================================================
// FatSlice external functions
// =============================================================================

/// Sum elements of a slice
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_sum_slice(data: FatSlice<i64>) -> i64 {
    unsafe { data.as_slice().iter().sum() }
}

/// Get length of slice
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_slice_len(data: FatSlice<i64>) -> i64 {
    data.len as i64
}

/// Double each element in a mutable slice
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_double_slice(mut data: FatSliceMut<i64>) {
    unsafe {
        for x in data.as_slice_mut() {
            *x *= 2;
        }
    }
}

// =============================================================================
// Tests
// =============================================================================

#[test]
fn test_extern_fn_simple_add() {
    let mut compiler = Compiler::new();

    // Register the external function
    let add_fn = compiler.extern_fn::<ExtAddExtern>();

    // Create a staged function that calls the external function
    let test_fn = compiler.fun2("test", |_ctx, x: Var<i64>, y: Var<i64>| {
        call_extern2::<_, _, _, i64, i64, i64>(add_fn, x, y)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    assert_eq!(f(10, 32), 42);
    assert_eq!(f(-5, 5), 0);
    assert_eq!(f(100, 200), 300);
}

#[test]
fn test_extern_fn_simple_square() {
    let mut compiler = Compiler::new();

    let square_fn = compiler.extern_fn::<ExtSquareExtern>();

    let test_fn = compiler.fun1("test", |_ctx, x: Var<i64>| {
        call_extern1::<_, _, i64, i64>(square_fn, x)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    assert_eq!(f(5), 25);
    assert_eq!(f(7), 49);
    assert_eq!(f(-3), 9);
}

#[test]
fn test_extern_fn_chained() {
    let mut compiler = Compiler::new();

    let add_fn = compiler.extern_fn::<ExtAddExtern>();
    let square_fn = compiler.extern_fn::<ExtSquareExtern>();

    // Compute square(x + y)
    let test_fn = compiler.fun2("test", |_ctx, x: Var<i64>, y: Var<i64>| {
        let sum = call_extern2::<_, _, _, i64, i64, i64>(add_fn, x, y);
        call_extern1::<_, _, i64, i64>(square_fn, sum)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    // (3 + 4)^2 = 49
    assert_eq!(f(3, 4), 49);
    // (10 + 0)^2 = 100
    assert_eq!(f(10, 0), 100);
}

#[test]
fn test_extern_fn_with_internal() {
    let mut compiler = Compiler::new();

    let ext_add = compiler.extern_fn::<ExtAddExtern>();

    // Mix internal and external function calls
    let internal_double = compiler.fun1("double", |_ctx, x: Var<i64>| add(x, x));

    let test_fn = compiler.fun2("test", |_ctx, x: Var<i64>, y: Var<i64>| {
        // double(ext_add(x, y))
        let sum = call_extern2::<_, _, _, i64, i64, i64>(ext_add, x, y);
        call1(internal_double, sum)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    // (10 + 32) * 2 = 84
    assert_eq!(f(10, 32), 84);
}

#[test]
fn test_extern_fn_sum_slice() {
    let mut compiler = Compiler::new();

    let sum_fn = compiler.extern_fn::<ExtSumSliceExtern>();

    // Function that takes a FatSlice and returns the sum
    let test_fn = compiler.fun1("test", |_ctx, data: Var<FatSliceType<i64>>| {
        call_extern1::<_, _, FatSliceType<i64>, i64>(sum_fn, data)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    let data = [1i64, 2, 3, 4, 5];
    let fat_slice = FatSlice::from_slice(&data);
    assert_eq!(f(fat_slice), 15); // 1+2+3+4+5 = 15

    let data2 = [10i64, 20, 30];
    let fat_slice2 = FatSlice::from_slice(&data2);
    assert_eq!(f(fat_slice2), 60); // 10+20+30 = 60
}

#[test]
fn test_extern_fn_slice_len() {
    let mut compiler = Compiler::new();

    let len_fn = compiler.extern_fn::<ExtSliceLenExtern>();

    let test_fn = compiler.fun1("test", |_ctx, data: Var<FatSliceType<i64>>| {
        call_extern1::<_, _, FatSliceType<i64>, i64>(len_fn, data)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    let data = [1i64, 2, 3, 4, 5];
    let fat_slice = FatSlice::from_slice(&data);
    assert_eq!(f(fat_slice), 5);

    let empty: [i64; 0] = [];
    let fat_empty = FatSlice::from_slice(&empty);
    assert_eq!(f(fat_empty), 0);
}
