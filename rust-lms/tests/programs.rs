//! Test for mutable reference handling in compute_stats program
//!
//! Run with: RUST_LMS_DEBUG_IR=1 cargo test --test programs -- --nocapture

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

/// Test compute_stats with mutable reference parameters using store_ref/load_ref_mut.
#[test]
fn test_compute_stats() {
    for_each_backend(|mut compiler| {
        let stats_fn = compiler.fun6(
            "compute_stats",
            |ctx,
             data: Var<SRef<Slice<f64>>>,
             v: Var<f64>,
             mut count_ptr: Var<SRefMut<u64>>,
             mut min_ptr: Var<SRefMut<f64>>,
             mut max_ptr: Var<SRefMut<f64>>,
             mut sum_ptr: Var<SRefMut<f64>>| {
                // Accumulators kept in register-resident locals; values flushed to
                // the output pointers only at the end.
                let count = ctx.var(0u64);
                let min = ctx.var(f64::INFINITY);
                let max = ctx.var(f64::NEG_INFINITY);
                let sum = ctx.var(0.0f64);

                data.staged_iter().for_each(ctx, move |ctx, val| {
                    ctx.if_then(gt(val, v), move |ctx| {
                        ctx.store(count, count + 1u64);
                        ctx.store(sum, sum + val);
                        ctx.if_then(lt(val, min), move |ctx| ctx.store(min, val));
                        ctx.if_then(gt(val, max), move |ctx| ctx.store(max, val));
                    });
                });

                // Emit the 4 store_refs as side effects, then return ().
                ctx.emit(store_ref(&mut count_ptr, count));
                ctx.emit(store_ref(&mut min_ptr, min));
                ctx.emit(store_ref(&mut max_ptr, max));
                ctx.emit(store_ref(&mut sum_ptr, sum));
                Const::<()>::new(())
            },
        );

        let compiled = compiler
            .compile(stats_fn)
            .expect("Failed to compile rust-lms function");
        let func = compiled.as_fn();

        let data = [1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0];
        let threshold = 4.0; // Values > 4.0: 5.0, 8.0, 9.0

        let mut count = 0u64;
        let mut min = 0.0;
        let mut max = 0.0;
        let mut sum = 0.0;

        func.call(
            &data[..],
            threshold,
            &mut count,
            &mut min,
            &mut max,
            &mut sum,
        );

        assert_eq!(count, 3, "count mismatch");
        assert_eq!(min, 5.0, "min mismatch");
        assert_eq!(max, 9.0, "max mismatch");
        assert_eq!(sum, 22.0, "sum mismatch");
    });
}

/// Simple test for store_ref/load_ref_mut with a single mutable reference.
#[test]
fn test_simple_store_load() {
    for_each_backend(|mut compiler| {
        let inc_fn = compiler.fun1("increment", |_ctx, mut ptr: Var<SRefMut<u64>>| {
            let current = load_ref_mut(&mut ptr);
            store_ref(&mut ptr, current + 1u64)
        });

        let compiled = compiler.compile(inc_fn).expect("Failed to compile");
        let func = compiled.as_fn();

        let mut value = 41u64;
        func.call(&mut value);
        assert_eq!(value, 42);
    });
}

// =============================================================================
// ne / le / ge
// =============================================================================

/// 1*ne + 2*le + 4*ge, so each result occupies its own bit of the answer.
#[test]
fn ne_le_ge_on_signed() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("cmps", |ctx, a: Var<i64>, b: Var<i64>| {
            let n = ctx.bind(select(ne(a, b), 1i64, 0i64));
            let l = ctx.bind(select(le(a, b), 2i64, 0i64));
            let g = ctx.bind(select(ge(a, b), 4i64, 0i64));
            add(add(n, l), g)
        });
        let compiled = compiler.compile(f).expect("compile");
        let cmp = compiled.as_fn();
        assert_eq!(cmp.call(1, 2), 1 + 2); // ne, le
        assert_eq!(cmp.call(2, 1), 1 + 4); // ne, ge
        assert_eq!(cmp.call(2, 2), 2 + 4); // le, ge
        assert_eq!(cmp.call(-3, 1), 1 + 2); // signed: -3 < 1
    });
}

/// The unsigned predicates must be picked, not the signed ones: a value with
/// the top bit set compares *greater*, not negative.
#[test]
fn ne_le_ge_on_unsigned() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("cmps_u", |ctx, a: Var<u64>, b: Var<u64>| {
            let n = ctx.bind(select(ne(a, b), 1u64, 0u64));
            let l = ctx.bind(select(le(a, b), 2u64, 0u64));
            let g = ctx.bind(select(ge(a, b), 4u64, 0u64));
            add(add(n, l), g)
        });
        let compiled = compiler.compile(f).expect("compile");
        let cmp = compiled.as_fn();
        assert_eq!(cmp.call(u64::MAX, 1), 1 + 4);
        assert_eq!(cmp.call(7, 7), 2 + 4);
    });
}

/// NaN is why `le` is not `!gt` and `ne` is not an ordered comparison: every
/// ordered predicate is false against NaN, while `ne` must be true.
#[test]
fn float_comparisons_follow_ieee_on_nan() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("fcmps", |ctx, a: Var<f64>, b: Var<f64>| {
            let n = ctx.bind(select(ne(a, b), 1i64, 0i64));
            let l = ctx.bind(select(le(a, b), 2i64, 0i64));
            let g = ctx.bind(select(ge(a, b), 4i64, 0i64));
            add(add(n, l), g)
        });
        let compiled = compiler.compile(f).expect("compile");
        let cmp = compiled.as_fn();

        assert_eq!(cmp.call(1.0, 2.0), 1 + 2);
        assert_eq!(cmp.call(2.0, 2.0), 2 + 4);
        // NaN: `ne` true, both ordered comparisons false — same as Rust.
        assert_eq!(cmp.call(f64::NAN, 1.0), 1);
        assert_eq!(cmp.call(f64::NAN, f64::NAN), 1);
    });
}
