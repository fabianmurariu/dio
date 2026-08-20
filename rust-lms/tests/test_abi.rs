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

/// Smallest partial word: a 1-byte aggregate exercises the exact-size copy at the
/// finest granularity (a rounded 8-byte load would read 7 bytes past the object).
#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C)]
pub struct Byte1 {
    pub flag: bool,
}

/// 4-byte partial word — the other sub-word size the suite previously skipped.
#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C)]
pub struct Word4 {
    pub a: u32,
}

/// Small (<=16B) mixed integer+float aggregate. This is the classic SysV/Win64
/// split-class case (one INTEGER eightbyte, one SSE eightbyte). It is moot under
/// the canonical by-pointer ABI, which is exactly why it is worth pinning: an
/// accidental return to register-class classification would regress here first.
#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C)]
pub struct MixedSmall {
    pub integer: u64,
    pub float: f64,
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

#[extern_fn]
pub extern "C" fn echo_byte1(value: Byte1) -> Byte1 {
    value
}

#[extern_fn]
pub extern "C" fn echo_word4(value: Word4) -> Word4 {
    value
}

#[extern_fn]
pub extern "C" fn echo_mixed_small(value: MixedSmall) -> MixedSmall {
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
    assert_eq!(std::mem::size_of::<Byte1>(), 1);
    assert_eq!(std::mem::size_of::<Word4>(), 4);
    assert_eq!(std::mem::size_of::<PartialWord>(), 12);
    assert_eq!(std::mem::size_of::<MixedSmall>(), 16);
    assert_eq!(std::mem::align_of::<Aligned16>(), 16);
    assert!(std::mem::size_of::<Large>() > 16);

    assert_internal_round_trip!("byte1_identity", Byte1, Byte1 { flag: true });
    assert_internal_round_trip!("word4_identity", Word4, Word4 { a: 0xDEAD_BEEF });
    assert_internal_round_trip!(
        "mixed_small_identity",
        MixedSmall,
        MixedSmall {
            integer: 0x0102_0304_0506_0708,
            float: -1.5,
        }
    );
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
    assert_extern_round_trip!(EchoByte1Extern, Byte1, Byte1 { flag: true });
    assert_extern_round_trip!(EchoWord4Extern, Word4, Word4 { a: 0x0BAD_F00D });
    assert_extern_round_trip!(
        EchoMixedSmallExtern,
        MixedSmall,
        MixedSmall {
            integer: 0x1122_3344_5566_7788,
            float: 2.25,
        }
    );
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
fn coption_roundtrips_over_aligned_payload_end_to_end() {
    // `COption<Aligned16>` places its payload at offset 16 (`align_up(8, 16)`) —
    // the one path where `COptionType::payload_offset()` actually feeds codegen.
    // The isolated layout test in `option.rs` checks the arithmetic; this drives a
    // Some (store + load at offset 16) and a None (default) through the JIT so a
    // hard-coded `8` or a mis-derived offset would fail end to end.
    assert_eq!(std::mem::align_of::<Aligned16>(), 16);
    assert_eq!(
        std::mem::align_of::<rust_lms::option::COption<Aligned16>>(),
        16
    );

    let value = Aligned16 {
        high: 0xAAAA,
        low: 0xBBBB,
        tag: 0xCCCC,
        flags: 0xDDDD,
    };
    let default = Aligned16 {
        high: 1,
        low: 2,
        tag: 3,
        flags: 4,
    };

    // Some(value) -> unwrap_or(default) == value
    let mut compiler = Compiler::new();
    let some_rt = compiler.fun2(
        "coption_some_aligned",
        |_ctx, x: Var<Aligned16>, d: Var<Aligned16>| unwrap_or(c_some(x), d),
    );
    let compiled = compiler.compile(some_rt).expect("compilation failed");
    assert_eq!(compiled.call(value, default), value);

    // None -> unwrap_or(default) == default
    let mut compiler = Compiler::new();
    let none_rt = compiler.fun2(
        "coption_none_aligned",
        |_ctx, _x: Var<Aligned16>, d: Var<Aligned16>| unwrap_or(c_none::<Aligned16>(), d),
    );
    let compiled = compiler.compile(none_rt).expect("compilation failed");
    assert_eq!(compiled.call(value, default), default);
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
