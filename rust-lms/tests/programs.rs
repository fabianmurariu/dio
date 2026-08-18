//! Test for mutable reference handling in compute_stats program
//!
//! Run with: RUST_LMS_DEBUG_IR=1 cargo test --test programs -- --nocapture

use rust_lms::prelude::*;

/// Test compute_stats with mutable reference parameters using store_ref/load_ref_mut.
#[test]
fn test_compute_stats() {
    let mut compiler = Compiler::new();

    let stats_fn = compiler.fun6(
        "compute_stats",
        |ctx,
         data: Var<SRef<Slice<f64>>>,
         v: Var<f64>,
         count_ptr: Var<SRefMut<u64>>,
         min_ptr: Var<SRefMut<f64>>,
         max_ptr: Var<SRefMut<f64>>,
         sum_ptr: Var<SRefMut<f64>>| {
            let i = ctx.var(0u64);
            // Accumulators kept in register-resident locals; values flushed to
            // the output pointers only at the end.
            let count = ctx.var(0u64);
            let min = ctx.var(f64::INFINITY);
            let max = ctx.var(f64::NEG_INFINITY);
            let sum = ctx.var(0.0f64);
            let val = ctx.var(0.0f64);

            ctx.while_loop(lt(i, data.clone().len()), move |ctx| {
                ctx.store(val, data.clone().get_unchecked(i));
                ctx.if_then(gt(val, v), move |ctx| {
                    ctx.store(count, count + 1u64);
                    ctx.store(sum, sum + val);
                    ctx.if_then(lt(val, min), move |ctx| ctx.store(min, val));
                    ctx.if_then(gt(val, max), move |ctx| ctx.store(max, val));
                });
                ctx.store(i, i + 1u64);
            });

            // Emit the 4 store_refs as side effects, then return ().
            ctx.emit(store_ref(count_ptr, count));
            ctx.emit(store_ref(min_ptr, min));
            ctx.emit(store_ref(max_ptr, max));
            ctx.emit(store_ref(sum_ptr, sum));
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
}

/// Simple test for store_ref/load_ref_mut with a single mutable reference.
#[test]
fn test_simple_store_load() {
    let mut compiler = Compiler::new();

    let inc_fn = compiler.fun1("increment", |_ctx, ptr: Var<SRefMut<u64>>| {
        store_ref(ptr, load_ref_mut(ptr) + 1u64)
    });

    let compiled = compiler.compile(inc_fn).expect("Failed to compile");
    let func = compiled.as_fn();

    let mut value = 41u64;
    func.call(&mut value);
    assert_eq!(value, 42);
}
