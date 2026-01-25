//! Test for mutable reference handling in compute_stats program
//!
//! Run with: RUST_LMS_DEBUG_IR=1 cargo test --test programs -- --nocapture

use rust_lms::num::gt;
use rust_lms::prelude::*;

/// Test compute_stats with mutable reference parameters using store_ref/load_ref_mut
#[test]
fn test_compute_stats() {
    let mut compiler = Compiler::new();

    let stats_fn = compiler.fun6(
        "compute_stats",
        |ctx,
         data: Var<SRef<Slice<F64Type>>>,
         v: Var<F64Type>,
         count_ptr: Var<SRefMut<U64Type>>,
         min_ptr: Var<SRefMut<F64Type>>,
         max_ptr: Var<SRefMut<F64Type>>,
         sum_ptr: Var<SRefMut<F64Type>>| {
            // Create loop variable
            let i = ctx.let_var(0u64);

            // Create LOCAL VARIABLES for accumulators - keeps values in registers!
            // This avoids load/store on every iteration
            let count = ctx.let_var(0u64);
            let min = ctx.let_var(f64::INFINITY);
            let max = ctx.let_var(f64::NEG_INFINITY);
            let sum = ctx.let_var(0.0f64);

            // Create a variable to hold the current value (avoid repeated loads)
            let val = ctx.let_var(0.0f64);

            // Main loop - all operations on local variables (registers)
            let loop_body = while_loop(lt(*i, data.len()), {
                // Load value into local variable ONCE per iteration
                (
                    assign(*val, data.get_unchecked(*i)),
                    // Check if value > v
                    if_then(gt(*val, v), {
                        (
                            assign(*count, add(*count, 1u64)),
                            assign(*sum, add(*sum, *val)),
                            if_then(lt(*val, *min), assign(*min, *val)),
                            if_then(gt(*val, *max), assign(*max, *val)),
                        )
                    }),
                    assign(*i, add(*i, 1u64)),
                )
            });

            // Store final results to output pointers (only ONCE after loop)
            let store_results = (
                store_ref(count_ptr, *count),
                store_ref(min_ptr, *min),
                store_ref(max_ptr, *max),
                store_ref(sum_ptr, *sum),
            );

            (i, count, min, max, sum, val, loop_body, store_results)
        },
    );

    let compiled = compiler
        .compile(stats_fn)
        .expect("Failed to compile rust-lms function");
    let func = compiled.as_fn();

    // Test data
    let data = vec![1.0, 5.0, 3.0, 8.0, 2.0, 9.0, 4.0];
    let threshold = 4.0; // Values > 4.0: 5.0, 8.0, 9.0

    let mut count = 0u64;
    let mut min = 0.0;
    let mut max = 0.0;
    let mut sum = 0.0;

    func(
        &data[..],
        threshold,
        &mut count,
        &mut min,
        &mut max,
        &mut sum,
    );

    // Expected: count=3, min=5.0, max=9.0, sum=22.0
    assert_eq!(count, 3, "count mismatch");
    assert_eq!(min, 5.0, "min mismatch");
    assert_eq!(max, 9.0, "max mismatch");
    assert_eq!(sum, 22.0, "sum mismatch");
}

/// Simple test for store_ref/load_ref_mut with a single mutable reference
#[test]
fn test_simple_store_load() {
    let mut compiler = Compiler::new();

    let inc_fn = compiler.fun1("increment", |_ctx, ptr: Var<SRefMut<U64Type>>| {
        store_ref(ptr, add(load_ref_mut(ptr), Const::new(1u64)))
    });

    let compiled = compiler.compile(inc_fn).expect("Failed to compile");
    let func = compiled.as_fn();

    let mut value = 41u64;
    func(&mut value);
    assert_eq!(value, 42);
}
