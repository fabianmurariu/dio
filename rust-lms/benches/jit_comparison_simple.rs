use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::time::Duration;

// ============================================================================
// Data Generation
// ============================================================================

fn generate_test_data(size: usize) -> (Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let mut data: Vec<f64> = (0..size)
        .map(|_| rng.gen_range(-10000.0..10000.0))
        .collect();

    // Find median (50th percentile) as predicate value
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = data[size / 2];

    // Shuffle back to random order
    use rand::seq::SliceRandom;
    data.shuffle(&mut rng);

    (data, median)
}

// ============================================================================
// Native Rust Baseline
// ============================================================================

fn compute_stats_native(
    data: &[f64],
    v: f64,
    count: &mut u64,
    min: &mut f64,
    max: &mut f64,
    sum: &mut f64,
    avg: &mut f64,
) {
    *count = 0;
    *min = f64::INFINITY;
    *max = f64::NEG_INFINITY;
    *sum = 0.0;

    for &value in data {
        if value > v {
            *count += 1;
            *sum += value;
            if value < *min {
                *min = value;
            }
            if value > *max {
                *max = value;
            }
        }
    }

    *avg = if *count > 0 {
        *sum / (*count as f64)
    } else {
        0.0
    };
}

// ============================================================================
// rust-lms Implementation
// ============================================================================

use rust_lms::prelude::*;
use rust_lms::num::gt;

fn compile_and_run_rust_lms(
    data: &[f64],
    v: f64,
    count: &mut u64,
    min: &mut f64,
    max: &mut f64,
    sum: &mut f64,
) {
    let mut compiler = Compiler::new();

    let stats_fn = compiler.fun6(
        "compute_stats",
        |ctx,
         data: Var<SRef<Slice<F64Type>>>,
         v: Var<F64Type>,
         count_ptr: Var<SRefMut<U64Type>>,
         min_ptr: Var<SRefMut<F64Type>>,
         max_ptr: Var<SRefMut<F64Type>>,
         sum_ptr: Var<SRefMut<F64Type>>,
         | {
            // Create loop variable
            let i = ctx.let_var(0u64);

            // Create LOCAL VARIABLES for accumulators - keeps values in registers!
            let l_count = ctx.let_var(0u64);
            let l_min = ctx.let_var(f64::INFINITY);
            let l_max = ctx.let_var(f64::NEG_INFINITY);
            let l_sum = ctx.let_var(0.0f64);

            // Create a variable for the current value
            let val = ctx.let_var(0.0f64);

            // Main loop - all operations on local variables (registers)
            let loop_body = while_loop(
                lt(*i, data.len()),
                {
                    // Load value once
                    let load_val = assign(*val, data.get_unchecked(*i));

                    // Check if value > v
                    let process = if_then(
                        gt(*val, v),
                        {
                            // All operations on register variables
                            let inc_count = assign(*l_count, add(*l_count, 1u64));
                            let add_sum = assign(*l_sum, add(*l_sum, *val));
                            let update_min = if_then(lt(*val, *l_min), assign(*l_min, *val));
                            let update_max = if_then(gt(*val, *l_max), assign(*l_max, *val));
                            (inc_count, add_sum, update_min, update_max)
                        }
                    );

                    // Increment loop counter
                    (load_val, process, assign(*i, add(*i, 1u64)))
                }
            );

            // Store final results to output pointers (only ONCE after loop)
            let store_results = (
                store_ref(count_ptr, *l_count),
                store_ref(min_ptr, *l_min),
                store_ref(max_ptr, *l_max),
                store_ref(sum_ptr, *l_sum),
            );

            (i, l_count, l_min, l_max, l_sum, val, loop_body, store_results)
        }
    );

    let compiled = compiler.compile(stats_fn).expect("Failed to compile rust-lms function");
    let func = compiled.as_fn();
    func(data, v, count, min, max, sum);
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_native(c: &mut Criterion, data_size: usize) {
    let (data, v) = generate_test_data(data_size);

    let mut group = c.benchmark_group(format!("native/{}", data_size));
    group.bench_function("execution", |b| {
        let mut count = 0u64;
        let mut min = 0.0;
        let mut max = 0.0;
        let mut sum = 0.0;
        let mut avg = 0.0;

        b.iter(|| {
            compute_stats_native(
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
                &mut avg,
            );
            black_box((count, min, max, sum, avg))
        });
    });
    group.finish();
}

fn bench_rust_lms(c: &mut Criterion, data_size: usize) {
    let (data, v) = generate_test_data(data_size);

    let mut group = c.benchmark_group(format!("rust-lms/{}", data_size));

    // Cold start: compile + execute
    group.bench_function("cold_start", |b| {
        b.iter(|| {
            let mut count = 0u64;
            let mut min = 0.0;
            let mut max = 0.0;
            let mut sum = 0.0;

            compile_and_run_rust_lms(
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
            );
            black_box((count, min, max, sum))
        });
    });

    // Warm execution: compile once, then measure only execution
    // Use raw pointers (SMutPtr) to avoid lifetime issues with criterion's b.iter()
    let mut compiler = Compiler::new();

    let stats_fn = compiler.fun6(
        "compute_stats_ptr",
        |ctx,
         data: Var<SPtr<F64Type>>,       // raw pointer to data
         len: Var<I64Type>,               // length of data (i64)
         v: Var<F64Type>,
         count_ptr: Var<SMutPtr<I64Type>>,  // use i64 for count too
         min_ptr: Var<SMutPtr<F64Type>>,
         max_ptr: Var<SMutPtr<F64Type>>,
         | {
            let i = ctx.let_var(0i64);

            // Local variables for accumulators - kept in registers
            let l_count = ctx.let_var(0i64);
            let l_min = ctx.let_var(f64::INFINITY);
            let l_max = ctx.let_var(f64::NEG_INFINITY);

            // Variable for current value
            let val = ctx.let_var(0.0f64);

            let loop_body = while_loop(
                lt(*i, len),
                {
                    // Load value into the variable
                    let load_val = assign(*val, array_index(data, *i));

                    let process = if_then(
                        gt(*val, v),
                        {
                            // All operations on register variables
                            let inc_count = assign(*l_count, add(*l_count, 1i64));
                            let update_min = if_then(lt(*val, *l_min), assign(*l_min, *val));
                            let update_max = if_then(gt(*val, *l_max), assign(*l_max, *val));
                            (inc_count, update_min, update_max)
                        }
                    );

                    (load_val, process, assign(*i, add(*i, 1i64)))
                }
            );

            // Store final results (only once after loop)
            let store_results = (
                store(count_ptr, *l_count),
                store(min_ptr, *l_min),
                store(max_ptr, *l_max),
            );

            (i, l_count, l_min, l_max, val, loop_body, store_results)
        }
    );

    let compiled = compiler.compile(stats_fn).expect("Failed to compile rust-lms function");
    let func = compiled.as_fn();

    group.bench_function("warm_execution", |b| {
        let mut count = 0i64;
        let mut min = 0.0;
        let mut max = 0.0;

        b.iter(|| {
            func(
                black_box(data.as_ptr()),
                black_box(data.len() as i64),
                black_box(v),
                black_box(&mut count as *mut i64),
                black_box(&mut min as *mut f64),
                black_box(&mut max as *mut f64),
            );
        });
    });

    group.finish();
}

fn benchmark_all_sizes(c: &mut Criterion) {
    for &size in &[1_000, 10_000, 100_000] {
        bench_native(c, size);
        bench_rust_lms(c, size);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets = benchmark_all_sizes
}

criterion_main!(benches);
