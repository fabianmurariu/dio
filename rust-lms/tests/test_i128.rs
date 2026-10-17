//! `i128`: an `Ordered` scalar without arithmetic, and `pack_i128`, whose signed order
//! is the order of its `(hi: i64, lo: u64)` parts.

use proptest::prelude::*;
use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

#[derive(Clone, Copy, Debug, PartialEq, StagedType)]
#[repr(C)]
pub struct Keyed {
    pub tag: u32,
    pub key: i128,
}

const PARTS: [(i64, u64); 9] = [
    (i64::MIN, 0),
    (i64::MIN, u64::MAX),
    (-1, 0),
    (-1, u64::MAX),
    (0, 0),
    (0, 1 << 63),
    (0, u64::MAX),
    (1, 0),
    (i64::MAX, u64::MAX),
];

#[test]
fn packed_order_is_the_order_of_the_parts() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun4(
            "packed_lt",
            |_ctx, t1: Var<i64>, i1: Var<u64>, t2: Var<i64>, i2: Var<u64>| {
                lt(pack_i128(t1, i1), pack_i128(t2, i2))
            },
        );
        let compiled = compiler.compile(f).expect("compile failed");
        let lt = compiled.as_fn();
        for a in PARTS {
            for b in PARTS {
                assert_eq!(lt.call(a.0, a.1, b.0, b.1), a < b, "{a:?} < {b:?}");
            }
        }
    });
}

#[test]
fn packs_to_the_host_value() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("pack", |_ctx, t: Var<i64>, i: Var<u64>| pack_i128(t, i));
        let compiled = compiler.compile(f).expect("compile failed");
        let pack = compiled.as_fn();
        for (t, i) in PARTS {
            assert_eq!(pack.call(t, i), ((t as i128) << 64) | i as i128);
        }
    });
}

#[test]
fn constants_wider_than_64_bits() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("below_max", |_ctx, x: Var<i128>| {
            lt(x, Const::new(i128::MAX))
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let below_max = compiled.as_fn();
        assert!(below_max.call(i128::MAX - 1));
        assert!(below_max.call(i128::MIN));
        assert!(!below_max.call(i128::MAX));
    });
}

#[test]
fn min_of_a_slice() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("slice_min", |ctx, xs: Var<SRef<Slice<i128>>>| {
            let best = ctx.var(Const::new(i128::MAX));
            xs.staged_iter().for_each(ctx, move |ctx, x| {
                ctx.store(best, min(x, best));
            });
            best
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let slice_min = compiled.as_fn();
        let xs = [5i128 << 70, -(3i128 << 90), 7, i128::MIN + 1, 1 << 64];
        assert_eq!(slice_min.call(&xs[..]), i128::MIN + 1);
        assert_eq!(slice_min.call(&[][..]), i128::MAX);
    });
}

#[test]
fn struct_field() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("max_key", |ctx, xs: Var<SRef<Slice<Keyed>>>| {
            let best = ctx.var(Const::new(i128::MIN));
            xs.staged_iter().for_each(ctx, move |ctx, x| {
                ctx.store(best, max(x.get(KeyedType::key()), best));
            });
            best
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let max_key = compiled.as_fn();
        let xs = [
            Keyed { tag: 1, key: -1 },
            Keyed {
                tag: 2,
                key: 1 << 100,
            },
            Keyed {
                tag: 3,
                key: 1 << 64,
            },
        ];
        assert_eq!(max_key.call(&xs[..]), 1 << 100);
    });
}

proptest! {
    #[test]
    fn packed_order_matches_tuples(a in any::<(i64, u64)>(), b in any::<(i64, u64)>()) {
        let mut compiler = Compiler::new();
        let f = compiler.fun4(
            "packed_cmp",
            |ctx, t1: Var<i64>, i1: Var<u64>, t2: Var<i64>, i2: Var<u64>| {
                let (x, y) = (ctx.bind(pack_i128(t1, i1)), ctx.bind(pack_i128(t2, i2)));
                select(lt(x, y), -1i64, select(eq(x, y), 0i64, 1i64))
            },
        );
        let compiled = compiler.compile(f).expect("compile failed");
        prop_assert_eq!(compiled.as_fn().call(a.0, a.1, b.0, b.1), a.cmp(&b) as i64);
    }
}
