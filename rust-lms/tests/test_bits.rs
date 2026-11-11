//! The integer bit operations — `swap_bytes`, `count_ones`, `leading_zeros`,
//! `trailing_zeros`, shifts — at every width, and `u128`: unsigned order, constants and
//! casts. Each result is checked against Rust's method of the same name.

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

/// [`samples`], truncated to `T`.
fn samples_of<T: Truncate>() -> Vec<T> {
    samples(T::truncate)
}

/// `as` from a `u128`, for the generic [`check`].
trait Truncate: Copy {
    fn truncate(v: u128) -> Self;
}

macro_rules! truncate {
    ($($ty:ty),*) => { $(impl Truncate for $ty { fn truncate(v: u128) -> Self { v as $ty } })* };
}
truncate!(u8, i8, u16, u32, i32, u64, i64, u128, i128);

/// Values whose bits sit at both ends and in the middle, for any width.
fn samples<T: Copy>(from_u128: impl Fn(u128) -> T) -> Vec<T> {
    [
        0u128,
        1,
        0x80,
        0xff,
        0x0102,
        0x0102_0304,
        0x0102_0304_0506_0708,
        0x0102_0304_0506_0708_090a_0b0c_0d0e_0f10,
        1 << 63,
        1 << 127,
        u128::MAX,
        u128::MAX >> 1,
    ]
    .into_iter()
    .map(from_u128)
    .collect()
}

/// `op` of each sample, compiled, against `host`.
fn check<T>(op: BitOpKind, host: impl Fn(T) -> T)
where
    T: Int + Truncate + Copy + std::fmt::LowerHex + PartialEq + std::fmt::Debug,
    T: for<'call> RuntimeParam<Arg<'call> = T> + for<'call> RuntimeResult<Output<'call> = T>,
    T: StagedType<RuntimeValue = T>,
{
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("op", move |_ctx, x: Var<T>| match op {
            BitOpKind::TrailingZeros => trailing_zeros(x),
            BitOpKind::LeadingZeros => leading_zeros(x),
            BitOpKind::CountOnes => count_ones(x),
            BitOpKind::SwapBytes => swap_bytes(x),
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        for x in samples_of::<T>() {
            assert_eq!(f.call(x), host(x), "{op:?}({x:#x})");
        }
    });
}

/// Check the four bit operations of one integer type against Rust's.
macro_rules! bit_ops_match_rust {
    ($name:ident, $ty:ty) => {
        #[test]
        fn $name() {
            check::<$ty>(BitOpKind::SwapBytes, |x| x.swap_bytes());
            check::<$ty>(BitOpKind::CountOnes, |x| x.count_ones() as $ty);
            check::<$ty>(BitOpKind::LeadingZeros, |x| x.leading_zeros() as $ty);
            check::<$ty>(BitOpKind::TrailingZeros, |x| x.trailing_zeros() as $ty);
        }
    };
}

bit_ops_match_rust!(bit_ops_u8, u8);
bit_ops_match_rust!(bit_ops_i8, i8);
bit_ops_match_rust!(bit_ops_u16, u16);
bit_ops_match_rust!(bit_ops_u32, u32);
bit_ops_match_rust!(bit_ops_i32, i32);
bit_ops_match_rust!(bit_ops_u64, u64);
bit_ops_match_rust!(bit_ops_i64, i64);
bit_ops_match_rust!(bit_ops_u128, u128);
bit_ops_match_rust!(bit_ops_i128, i128);

#[test]
fn u128_orders_unsigned() {
    for_each_backend(|mut compiler| {
        // `lt` and `ge` of every pair, as two bits.
        let f = compiler.fun2("cmp", |_ctx, a: Var<u128>, b: Var<u128>| {
            select(lt(a, b), 1u8, 0u8) | select(ge(a, b), 2u8, 0u8)
        });
        let compiled = compiler.compile(f).expect("compile");
        let cmp = compiled.as_fn();
        let xs = samples(|v| v);
        for &a in &xs {
            for &b in &xs {
                let want = (a < b) as u8 | (((a >= b) as u8) << 1);
                assert_eq!(cmp.call(a, b), want, "{a:#x} vs {b:#x}");
            }
        }
    });
}

#[test]
fn u128_constants_and_shifts() {
    for_each_backend(|mut compiler| {
        // `(x << 32) | 0xffff_ffff_0000_0000_0000_0000_0000_0000 >> 96`: both shift
        // directions and a constant wider than 64 bits.
        let f = compiler.fun1("key", |_ctx, x: Var<u128>| {
            (x << Const::new(32u128)) | (Const::new(u128::MAX << 96) >> Const::new(96u128))
        });
        let compiled = compiler.compile(f).expect("compile");
        let key = compiled.as_fn();
        for x in samples(|v| v) {
            assert_eq!(key.call(x), (x << 32) | ((u128::MAX << 96) >> 96));
        }
    });
}

#[test]
fn u128_casts_to_its_halves_and_back() {
    for x in samples(|v| v) {
        for_each_backend(|mut compiler| {
            let f = compiler.fun1("halves", |_ctx, x: Var<u128>| {
                // The low half, the high half shifted down, widened back and put together.
                let lo = int_cast::<u64, u128, _>(x);
                let hi = int_cast::<u64, u128, _>(x >> Const::new(64u128));
                int_cast::<u128, u64, _>(lo) | (int_cast::<u128, u64, _>(hi) << Const::new(64u128))
            });
            let compiled = compiler.compile(f).expect("compile");
            assert_eq!(compiled.as_fn().call(x), x);
        });
        for_each_backend(|mut compiler| {
            let f = compiler.fun1("signed", |_ctx, x: Var<i64>| int_cast::<i128, i64, _>(x));
            let compiled = compiler.compile(f).expect("compile");
            assert_eq!(compiled.as_fn().call(x as i64), x as i64 as i128);
        });
    }
}

/// Arrow's inline key for a short byte view: the data bytes big-endian above the length,
/// so one unsigned compare orders two inline strings.
#[test]
fn inline_key_orders_as_the_bytes() {
    fn view(s: &[u8]) -> u128 {
        let mut bytes = [0u8; 16];
        bytes[..4].copy_from_slice(&(s.len() as u32).to_le_bytes());
        bytes[4..4 + s.len()].copy_from_slice(s);
        u128::from_le_bytes(bytes)
    }
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("lt", |_ctx, a: Var<u128>, b: Var<u128>| {
            let key = |v: Var<u128>| {
                (swap_bytes(v) << Const::new(32u128))
                    | int_cast::<u128, u32, _>(int_cast::<u32, u128, _>(v))
            };
            lt(key(a), key(b))
        });
        let compiled = compiler.compile(f).expect("compile");
        let lt = compiled.as_fn();
        let strings: [&[u8]; 8] = [
            b"",
            b"a",
            b"ab",
            b"b",
            b"bar",
            b"bar\0",
            b"\xff",
            b"abcdefghijkl",
        ];
        for a in strings {
            for b in strings {
                assert_eq!(lt.call(view(a), view(b)), a < b, "{a:?} < {b:?}");
            }
        }
    });
}
