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

/// An unsafe callback must use `call_extern1_unchecked`.
#[extern_fn]
#[no_mangle]
pub unsafe extern "C" fn ext_read_i64(ptr: *const i64) -> i64 {
    // SAFETY: required by this function's contract.
    unsafe { *ptr }
}

/// A safe shared-reference callback retains a staged `SRef` signature.
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_read_ref(value: &i64) -> i64 {
    *value
}

/// A safe mutable-reference callback retains a staged `SRefMut` signature.
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_add_assign(value: &mut i64, delta: i64) -> i64 {
    *value += delta;
    *value
}

/// Slice references preserve their pointer-and-length signature, but are not
/// considered safe extern calls because Rust does not define their C ABI.
#[allow(improper_ctypes_definitions)]
#[extern_fn]
#[no_mangle]
pub extern "C" fn ext_ref_slice_len(data: &[i64]) -> usize {
    data.len()
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
fn test_extern_marker_carries_the_complete_signature() {
    fn assert_add_signature<S>()
    where
        S: ExternFn<Args = (i64, i64), Ret = i64> + SafeExternFn,
    {
    }

    fn assert_noop_signature<S>()
    where
        S: ExternFn<Args = (), Ret = ()> + SafeExternFn,
    {
    }

    fn assert_slice_signature<S>()
    where
        S: ExternFn<Args = (FatSliceType<i64>,), Ret = i64> + SafeExternFn,
    {
    }

    fn assert_ref_signature<S>()
    where
        S: ExternFn<Args = (SRef<'static, Opaque<i64>>,), Ret = i64> + SafeExternFn,
    {
    }

    fn assert_mut_ref_signature<S>()
    where
        S: ExternFn<Args = (SRefMut<'static, Opaque<i64>>, i64), Ret = i64> + SafeExternFn,
    {
    }

    fn assert_ref_slice_signature<S>()
    where
        S: ExternFn<Args = (SRef<'static, Slice<i64>>,), Ret = u64>,
    {
    }

    assert_add_signature::<ExtAddExtern>();
    assert_noop_signature::<ExtNoopExtern>();
    assert_slice_signature::<ExtSumSliceExtern>();
    assert_ref_signature::<ExtReadRefExtern>();
    assert_mut_ref_signature::<ExtAddAssignExtern>();
    assert_ref_slice_signature::<ExtRefSliceLenExtern>();
}

#[test]
fn test_safe_extern_shared_reference() {
    let mut compiler = Compiler::new();
    let read = compiler.extern_fn::<ExtReadRefExtern>();
    let test_fn = compiler.fun1("read_ref", |_ctx, value: Var<SRef<Opaque<i64>>>| {
        call_extern1(read, value)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let value = 42i64;
    assert_eq!(compiled.call(&value), 42);
}

#[test]
fn test_safe_extern_mut_reference_reborrows_sequentially() {
    let mut compiler = Compiler::new();
    let add_assign = compiler.extern_fn::<ExtAddAssignExtern>();
    let test_fn = compiler.fun1(
        "add_assign_twice",
        |ctx, mut value: Var<SRefMut<Opaque<i64>>>| {
            let _first = ctx.bind(call_extern2(add_assign, &mut value, Const::<i64>::new(1)));
            ctx.bind(call_extern2(add_assign, &mut value, Const::<i64>::new(2)))
        },
    );

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let mut value = 10i64;
    assert_eq!(compiled.call(&mut value), 13);
    assert_eq!(value, 13);
}

#[test]
fn test_extern_slice_reference_uses_split_parameter_values() {
    let mut compiler = Compiler::new();
    let len = compiler.extern_fn::<ExtRefSliceLenExtern>();
    let test_fn = compiler.fun1("ref_slice_len", |_ctx, data: Var<SRef<Slice<i64>>>| {
        // SAFETY: `data` is a valid shared slice reference. This call is
        // unchecked only because Rust slice references have no stable C ABI.
        unsafe { call_extern1_unchecked(len, data) }
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let data = [3i64, 5, 8, 13];
    assert_eq!(compiled.call(&data), 4);
}

#[test]
fn test_unsafe_extern_requires_explicit_constructor() {
    let mut compiler = Compiler::new();
    let read = compiler.extern_fn::<ExtReadI64Extern>();
    let test_fn = compiler.fun1("test", |_ctx, ptr: Var<SPtr<i64>>| {
        // SAFETY: the generated function forwards its caller-provided pointer;
        // this test supplies a live, aligned `i64` below.
        unsafe { call_extern1_unchecked(read, ptr) }
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let function = compiled.as_fn();
    let value = 42i64;
    assert_eq!(function.call(&value), 42);
}

#[test]
fn test_extern_fn_simple_add() {
    let mut compiler = Compiler::new();

    // Register the external function
    let add_fn = compiler.extern_fn::<ExtAddExtern>();

    // Create a staged function that calls the external function
    let test_fn = compiler.fun2("test", |_ctx, x: Var<i64>, y: Var<i64>| {
        call_extern2(add_fn, x, y)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    assert_eq!(f.call(10, 32), 42);
    assert_eq!(f.call(-5, 5), 0);
    assert_eq!(f.call(100, 200), 300);
}

#[test]
fn test_extern_fn_simple_square() {
    let mut compiler = Compiler::new();

    let square_fn = compiler.extern_fn::<ExtSquareExtern>();

    let test_fn = compiler.fun1("test", |_ctx, x: Var<i64>| call_extern1(square_fn, x));

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    assert_eq!(f.call(5), 25);
    assert_eq!(f.call(7), 49);
    assert_eq!(f.call(-3), 9);
}

#[test]
fn test_extern_fn_chained() {
    let mut compiler = Compiler::new();

    let add_fn = compiler.extern_fn::<ExtAddExtern>();
    let square_fn = compiler.extern_fn::<ExtSquareExtern>();

    // Compute square(x + y)
    let test_fn = compiler.fun2("test", |_ctx, x: Var<i64>, y: Var<i64>| {
        let sum = call_extern2(add_fn, x, y);
        call_extern1(square_fn, sum)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    // (3 + 4)^2 = 49
    assert_eq!(f.call(3, 4), 49);
    // (10 + 0)^2 = 100
    assert_eq!(f.call(10, 0), 100);
}

#[test]
fn test_extern_fn_with_internal() {
    let mut compiler = Compiler::new();

    let ext_add = compiler.extern_fn::<ExtAddExtern>();

    // Mix internal and external function calls
    let internal_double = compiler.fun1("double", |_ctx, x: Var<i64>| x + x);

    let test_fn = compiler.fun2("test", |_ctx, x: Var<i64>, y: Var<i64>| {
        // double(ext_add(x, y))
        let sum = call_extern2(ext_add, x, y);
        call1(internal_double, sum)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    // (10 + 32) * 2 = 84
    assert_eq!(f.call(10, 32), 84);
}

#[test]
fn test_extern_fn_sum_slice() {
    let mut compiler = Compiler::new();

    let sum_fn = compiler.extern_fn::<ExtSumSliceExtern>();

    // Function that takes a FatSlice and returns the sum
    let test_fn = compiler.fun1("test", |_ctx, data: Var<FatSliceType<i64>>| {
        call_extern1(sum_fn, data)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    let data = [1i64, 2, 3, 4, 5];
    let fat_slice = FatSlice::from_slice(&data);
    assert_eq!(f.call(fat_slice), 15); // 1+2+3+4+5 = 15

    let data2 = [10i64, 20, 30];
    let fat_slice2 = FatSlice::from_slice(&data2);
    assert_eq!(f.call(fat_slice2), 60); // 10+20+30 = 60
}

#[test]
fn test_extern_fn_slice_len() {
    let mut compiler = Compiler::new();

    let len_fn = compiler.extern_fn::<ExtSliceLenExtern>();

    let test_fn = compiler.fun1("test", |_ctx, data: Var<FatSliceType<i64>>| {
        call_extern1(len_fn, data)
    });

    let compiled = compiler.compile(test_fn).expect("compilation failed");
    let f = compiled.as_fn();

    let data = [1i64, 2, 3, 4, 5];
    let fat_slice = FatSlice::from_slice(&data);
    assert_eq!(f.call(fat_slice), 5);

    let empty: [i64; 0] = [];
    let fat_empty = FatSlice::from_slice(&empty);
    assert_eq!(f.call(fat_empty), 0);
}
