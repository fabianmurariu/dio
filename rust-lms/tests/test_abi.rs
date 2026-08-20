//! Regression coverage for the canonical storage-pointer ABI.

use std::sync::atomic::{AtomicUsize, Ordering};

use rust_lms::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C)]
pub struct PartialWord {
    pub a: u32,
    pub b: u32,
    pub c: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C)]
pub struct FloatPair {
    pub x: f64,
    pub y: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C, align(16))]
pub struct Aligned16 {
    pub high: u64,
    pub low: u32,
    pub tag: u16,
    pub flags: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C)]
pub struct Nested {
    pub inner: PartialWord,
    pub tail: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C)]
pub struct Large {
    pub integer: u64,
    pub float: f64,
    pub count: u64,
    pub code: u32,
}

#[extern_fn]
pub extern "C" fn echo_partial(value: PartialWord) -> PartialWord {
    value
}

#[extern_fn]
pub extern "C" fn echo_float_pair(value: FloatPair) -> FloatPair {
    value
}

#[extern_fn]
pub extern "C" fn echo_aligned(value: Aligned16) -> Aligned16 {
    value
}

#[extern_fn]
pub extern "C" fn echo_nested(value: Nested) -> Nested {
    value
}

#[extern_fn]
pub extern "C" fn echo_large(value: Large) -> Large {
    value
}

static UNIT_CALLS: AtomicUsize = AtomicUsize::new(0);

#[extern_fn]
pub extern "C" fn record_unit_call() {
    UNIT_CALLS.fetch_add(1, Ordering::SeqCst);
}

macro_rules! assert_internal_round_trip {
    ($name:literal, $staged:ty, $value:expr $(,)?) => {{
        let value = $value;
        let mut compiler = Compiler::new();
        let identity = compiler.fun1($name, |_ctx, input: Var<$staged>| input);
        let compiled = compiler.compile(identity).expect("compilation failed");
        assert_eq!(compiled.call(value), value);
    }};
}

#[test]
fn internal_aggregate_round_trips_use_exact_layouts() {
    assert_eq!(std::mem::size_of::<PartialWord>(), 12);
    assert_eq!(std::mem::align_of::<Aligned16>(), 16);
    assert!(std::mem::size_of::<Large>() > 16);

    assert_internal_round_trip!(
        "partial_identity",
        PartialWord,
        PartialWord { a: 1, b: 2, c: 3 }
    );
    assert_internal_round_trip!("float_identity", FloatPair, FloatPair { x: 1.25, y: -9.5 });
    assert_internal_round_trip!(
        "aligned_identity",
        Aligned16,
        Aligned16 {
            high: 11,
            low: 22,
            tag: 33,
            flags: 44,
        },
    );
    assert_internal_round_trip!(
        "nested_identity",
        Nested,
        Nested {
            inner: PartialWord { a: 4, b: 5, c: 6 },
            tail: 7,
        },
    );
    assert_internal_round_trip!(
        "large_identity",
        Large,
        Large {
            integer: 8,
            float: 3.5,
            count: 9,
            code: 10,
        },
    );
}

macro_rules! assert_extern_round_trip {
    ($marker:ty, $staged:ty, $value:expr $(,)?) => {{
        let value = $value;
        let mut compiler = Compiler::new();
        let external = compiler.extern_fn::<$marker>();
        let forward = compiler.fun1("extern_round_trip", move |_ctx, input: Var<$staged>| {
            call_extern1(external, input)
        });
        let compiled = compiler.compile(forward).expect("compilation failed");
        assert_eq!(compiled.call(value), value);
    }};
}

#[test]
fn rust_thunks_handle_every_aggregate_shape() {
    assert_extern_round_trip!(
        EchoPartialExtern,
        PartialWord,
        PartialWord {
            a: 101,
            b: 102,
            c: 103,
        }
    );
    assert_extern_round_trip!(
        EchoFloatPairExtern,
        FloatPair,
        FloatPair { x: -0.25, y: 64.5 }
    );
    assert_extern_round_trip!(
        EchoAlignedExtern,
        Aligned16,
        Aligned16 {
            high: 201,
            low: 202,
            tag: 203,
            flags: 204,
        }
    );
    assert_extern_round_trip!(
        EchoNestedExtern,
        Nested,
        Nested {
            inner: PartialWord {
                a: 301,
                b: 302,
                c: 303,
            },
            tail: 304,
        }
    );
    assert_extern_round_trip!(
        EchoLargeExtern,
        Large,
        Large {
            integer: 401,
            float: -2.75,
            count: 402,
            code: 403,
        }
    );
}

#[test]
fn unit_has_no_runtime_return_value() {
    let compiler = Compiler::new();
    compiler.compile(unit()).expect("compilation failed").run();

    UNIT_CALLS.store(0, Ordering::SeqCst);
    let mut compiler = Compiler::new();
    let external = compiler.extern_fn::<RecordUnitCallExtern>();
    let invoke = compiler.fun0("invoke_unit", move |_ctx| call_extern0(external));
    let compiled = compiler.compile(invoke).expect("compilation failed");
    compiled.call();
    assert_eq!(UNIT_CALLS.load(Ordering::SeqCst), 1);
}
