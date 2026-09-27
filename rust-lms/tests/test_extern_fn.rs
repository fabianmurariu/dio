//! Integration tests for external function calling via #[extern_fn]

use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

mod common;
use common::for_each_backend;

// =============================================================================
// Simple external functions
// =============================================================================

/// Simple addition function
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_add(x: i64, y: i64) -> i64 {
    x + y
}

/// Simple multiplication
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_mul(x: i64, y: i64) -> i64 {
    x * y
}

/// Square a number
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_square(x: i64) -> i64 {
    x * x
}

/// Function with no return value
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_noop() {
    // Do nothing
}

/// # Safety
/// An unsafe callback must use `call_extern1_unchecked`.
#[extern_fn]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ext_read_i64(ptr: *const i64) -> i64 {
    unsafe { *ptr }
}

/// A safe shared-reference callback retains a staged `SRef` signature.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_read_ref(value: &i64) -> i64 {
    *value
}

/// A safe mutable-reference callback retains a staged `SRefMut` signature.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_add_assign(value: &mut i64, delta: i64) -> i64 {
    *value += delta;
    *value
}

/// Slice references preserve their pointer-and-length signature, but are not
/// considered safe extern calls because Rust does not define their C ABI.
#[allow(improper_ctypes_definitions)]
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_ref_slice_len(data: &[i64]) -> usize {
    data.len()
}

// =============================================================================
// FatSlice external functions
// =============================================================================

/// Sum elements of a slice
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_sum_slice(data: FatSlice<i64>) -> i64 {
    unsafe { data.as_slice().iter().sum() }
}

/// Get length of slice
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_slice_len(data: FatSlice<i64>) -> i64 {
    data.len as i64
}

#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn ext_identity_slice(data: FatSlice<i64>) -> FatSlice<i64> {
    data
}

/// Double each element in a mutable slice
#[extern_fn]
#[unsafe(no_mangle)]
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
        S: ExternFn<Args = (RawSlice<i64>,), Ret = i64> + SafeExternFn,
    {
    }

    fn assert_ref_signature<S>()
    where
        S: ExternFn<Args = (SRef<Opaque<i64>>,), Ret = i64> + SafeExternFn,
    {
    }

    fn assert_mut_ref_signature<S>()
    where
        S: ExternFn<Args = (SRefMut<Opaque<i64>>, i64), Ret = i64> + SafeExternFn,
    {
    }

    fn assert_ref_slice_signature<S>()
    where
        S: ExternFn<Args = (SRef<Slice<i64>>,), Ret = u64>,
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
    for_each_backend(|mut compiler| {
        let read = compiler.extern_fn::<ExtReadRefExtern>();
        let test_fn = compiler.fun1("read_ref", |_ctx, value: Var<SRef<Opaque<i64>>>| {
            call_extern1(read, value)
        });

        let compiled = compiler.compile(test_fn).expect("compilation failed");
        let value = 42i64;
        assert_eq!(compiled.call(&value), 42);
    });
}

#[test]
fn test_safe_extern_mut_reference_reborrows_sequentially() {
    for_each_backend(|mut compiler| {
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
    });
}

#[test]
fn test_extern_slice_reference_uses_split_parameter_values() {
    for_each_backend(|mut compiler| {
        let len = compiler.extern_fn::<ExtRefSliceLenExtern>();
        let test_fn = compiler.fun1("ref_slice_len", |_ctx, data: Var<SRef<Slice<i64>>>| {
            // SAFETY: `data` is a valid shared slice reference. This call is
            // unchecked only because Rust slice references have no stable C ABI.
            unsafe { call_extern1_unchecked(len, data) }
        });

        let compiled = compiler.compile(test_fn).expect("compilation failed");
        let data = [3i64, 5, 8, 13];
        assert_eq!(compiled.call(&data), 4);
    });
}

#[test]
fn test_unsafe_extern_requires_explicit_constructor() {
    for_each_backend(|mut compiler| {
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
    });
}

#[test]
fn test_extern_fn_simple_add() {
    for_each_backend(|mut compiler| {
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
    });
}

#[test]
fn test_extern_fn_simple_square() {
    for_each_backend(|mut compiler| {
        let square_fn = compiler.extern_fn::<ExtSquareExtern>();

        let test_fn = compiler.fun1("test", |_ctx, x: Var<i64>| call_extern1(square_fn, x));

        let compiled = compiler.compile(test_fn).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(5), 25);
        assert_eq!(f.call(7), 49);
        assert_eq!(f.call(-3), 9);
    });
}

#[test]
fn test_extern_fn_chained() {
    for_each_backend(|mut compiler| {
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
    });
}

#[test]
fn test_extern_fn_with_internal() {
    for_each_backend(|mut compiler| {
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
    });
}

#[test]
fn test_extern_fn_sum_slice() {
    for_each_backend(|mut compiler| {
        let sum_fn = compiler.extern_fn::<ExtSumSliceExtern>();

        // Function that takes a FatSlice and returns the sum
        let test_fn = compiler.fun1("test", |_ctx, data: Var<RawSlice<i64>>| {
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
    });
}

#[test]
fn test_extern_fn_slice_len() {
    for_each_backend(|mut compiler| {
        let len_fn = compiler.extern_fn::<ExtSliceLenExtern>();

        let test_fn = compiler.fun1("test", |_ctx, data: Var<RawSlice<i64>>| {
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
    });
}

#[test]
fn test_extern_call_preserves_fat_slice_return() {
    for_each_backend(|mut compiler| {
        let identity = compiler.extern_fn::<ExtIdentitySliceExtern>();
        let test_fn = compiler.fun1("identity_len", |ctx, data: Var<RawSlice<i64>>| {
            let returned: Var<RawSlice<i64>> = ctx.bind(call_extern1(identity, data));
            returned.len()
        });

        let compiled = compiler.compile(test_fn).expect("compilation failed");
        let data = [1i64, 2, 3];
        assert_eq!(compiled.call(FatSlice::from_slice(&data)), 3);
    });
}

// =============================================================================
// from_fn: a pull-based source that reports its own exhaustion
// =============================================================================

const STREAM: [i64; 4] = [5, 7, 11, 13];

/// One host-side cursor per test: these are process globals and the test
/// harness runs tests in parallel, so sharing one between two tests would let
/// them observe each other's pulls.
static mut DRAIN_CURSOR: usize = 0;
static mut CAPPED_CURSOR: usize = 0;

/// Returns the next value, or the `0` sentinel once drained — the shape an FFI
/// stream callback has.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn drain_next() -> i64 {
    // SAFETY: only `from_fn_pulls_until_the_sentinel` touches this cursor, and
    // it drives one kernel at a time.
    unsafe {
        if DRAIN_CURSOR >= STREAM.len() {
            return 0;
        }
        let v = STREAM[DRAIN_CURSOR];
        DRAIN_CURSOR += 1;
        v
    }
}

#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn capped_next() -> i64 {
    // SAFETY: only `from_fn_producer_can_break_before_pulling` touches this one.
    unsafe {
        if CAPPED_CURSOR >= STREAM.len() {
            return 0;
        }
        let v = STREAM[CAPPED_CURSOR];
        CAPPED_CURSOR += 1;
        v
    }
}

/// `from_fn` drives the loop until the producer yields `None`, with no length
/// known up front.
#[test]
fn from_fn_pulls_until_the_sentinel() {
    for_each_backend(|mut compiler| {
        let next = compiler.extern_fn::<DrainNextExtern>();
        let f = compiler.fun0("drain", move |ctx| {
            let acc = ctx.var(0i64);
            from_fn(move |ctx| {
                let v = ctx.bind(call_extern0(next));
                ne(v, 0i64).then_some(v)
            })
            .for_each(ctx, move |ctx, v| {
                ctx.store(acc, add(acc, v));
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile");

        // SAFETY: this test owns `DRAIN_CURSOR`; reset before each backend run.
        unsafe { DRAIN_CURSOR = 0 };
        assert_eq!(compiled.call(), 36); // 5 + 7 + 11 + 13
    });
}

/// The producer may emit a guard *before* pulling — the case a `take_while` on
/// the produced item cannot express, because by then the pull has happened.
#[test]
fn from_fn_producer_can_break_before_pulling() {
    for_each_backend(|mut compiler| {
        let next = compiler.extern_fn::<CappedNextExtern>();
        let f = compiler.fun1("drain_capped", move |ctx, cap: Var<i64>| {
            let acc = ctx.var(0i64);
            let pulls = ctx.var(0i64);
            from_fn(move |ctx| {
                // Stop before consuming another item once the cap is reached.
                ctx.if_then(ge(pulls, cap), |ctx| ctx.break_loop());
                ctx.store(pulls, add(pulls, 1i64));
                let v = ctx.bind(call_extern0(next));
                ne(v, 0i64).then_some(v)
            })
            .for_each(ctx, move |ctx, v| {
                ctx.store(acc, add(acc, v));
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile");

        // SAFETY: this test owns `CAPPED_CURSOR`.
        unsafe { CAPPED_CURSOR = 0 };
        // Cap of 2 stops after 5 + 7, leaving 11 and 13 unread.
        assert_eq!(compiled.call(2), 12);
        // SAFETY: as above — the cursor proves only two items were pulled.
        assert_eq!(unsafe { CAPPED_CURSOR }, 2);
    });
}

// =============================================================================
// Generic extern functions: one marker instantiation per type argument
// =============================================================================

/// Widen any small integer to `i64` and double it. Each instantiation of the
/// generated `GenericDoubleExtern<T>` gets its own monomorphic thunk. `T` is
/// used directly as a parameter type, so it must be `StagedType` itself.
#[extern_fn]
pub extern "C" fn generic_double<T: StagedType + Copy + Into<i64>>(x: T) -> i64 {
    2 * x.into()
}

#[test]
fn generic_extern_fn_instantiates_per_type() {
    for_each_backend(|mut compiler| {
        let from_i32 = compiler.extern_fn::<GenericDoubleExtern<i32>>();
        let from_u8 = compiler.extern_fn::<GenericDoubleExtern<u8>>();
        let f = compiler.fun2("double_both", move |_ctx, a: Var<i32>, b: Var<u8>| {
            add(call_extern1(from_i32, a), call_extern1(from_u8, b))
        });
        let compiled = compiler.compile(f).expect("compile");
        assert_eq!(compiled.call(-7, 200), 2 * -7 + 2 * 200);
    });
}
