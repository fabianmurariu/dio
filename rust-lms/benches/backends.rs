//! Backend comparison: **native Rust vs Cranelift-JIT vs LLVM/MLIR-JIT** on slice kernels
//! shaped like sql-gen's columnar work. Steady-state (warm) execution — each JIT function is
//! compiled once, outside the measured loop.
//!
//! Kernels:
//! - `filtered_sum(a, z)` — sum of `a[i]` where `a[i] > z` (one columnar slice).
//! - `sum_above_median(a, b)` — `m = a[len/2]` (the median of the pre-sorted `a`), then sum
//!   `a[i] + b[i]` over `i` where `max(a[i], b[i]) > m` ("one of the two values exceeds the
//!   median of the first array").
//!
//! Run:
//!   cargo bench --bench backends                        # native + Cranelift
//!   MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm LLVM_SYS_220_PREFIX=$MLIR_SYS_220_PREFIX \
//!     PATH="$MLIR_SYS_220_PREFIX/bin:$PATH" DYLD_LIBRARY_PATH="$MLIR_SYS_220_PREFIX/lib" \
//!     cargo bench --bench backends --features llvm      # + LLVM/MLIR

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Duration;

use rust_lms::func::JitBackend;
use rust_lms::prelude::*;

const SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

type FilteredSum = Box<dyn Fn(&[i64], i64) -> i64>;
type SumAboveMedian = Box<dyn Fn(&[i64], &[i64]) -> i64>;

// ============================================================================
// Data — `a` is sorted so `a[len/2]` is genuinely its median.
// ============================================================================

fn generate(size: usize, seed: u64) -> (Vec<i64>, Vec<i64>) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut a: Vec<i64> = (0..size)
        .map(|_| rng.random_range(-1_000i64..1_000))
        .collect();
    a.sort_unstable();
    let b: Vec<i64> = (0..size)
        .map(|_| rng.random_range(-1_000i64..1_000))
        .collect();
    (a, b)
}

// ============================================================================
// Native baselines
// ============================================================================

#[inline(never)]
fn filtered_sum_native(a: &[i64], z: i64) -> i64 {
    let mut acc = 0i64;
    for &v in a {
        if v > z {
            acc = acc.wrapping_add(v);
        }
    }
    acc
}

#[inline(never)]
fn sum_above_median_native(a: &[i64], b: &[i64]) -> i64 {
    let n = a.len().min(b.len());
    let m = a[n / 2];
    let mut acc = 0i64;
    for i in 0..n {
        if a[i].max(b[i]) > m {
            acc = acc.wrapping_add(a[i].wrapping_add(b[i]));
        }
    }
    acc
}

// ============================================================================
// Staged kernels — compiled once, callable as `Fn`. Same source for both backends.
// ============================================================================

fn build_filtered_sum(backend: JitBackend) -> FilteredSum {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun2(
        "filtered_sum",
        |ctx, a: Var<SRef<Slice<i64>>>, z: Var<i64>| {
            let acc = ctx.var(0i64);
            let i = ctx.var(0u64);
            ctx.while_loop(lt(i, a.count()), move |ctx| {
                // SAFETY: the loop condition proves `i < a.len()`.
                let v = ctx.bind(unsafe { a.get_unchecked(i) });
                ctx.if_then(gt(v, z), move |ctx| {
                    ctx.store(acc, add(acc, v));
                });
                ctx.store(i, add(i, 1u64));
            });
            acc
        },
    );
    let compiled = compiler.compile(f).expect("filtered_sum compile");
    Box::new(move |a, z| compiled.as_fn().call(a, z))
}

fn build_sum_above_median(backend: JitBackend) -> SumAboveMedian {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun2(
        "sum_above_median",
        |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
            let n = ctx.bind(a.count());
            let mid = ctx.bind(div(n, 2u64));
            // Median of the (pre-sorted) first array.
            let m = ctx.bind(unsafe { a.get_unchecked(mid) });
            let acc = ctx.var(0i64);
            let i = ctx.var(0u64);
            ctx.while_loop(lt(i, n), move |ctx| {
                // SAFETY: `i < n <= len(a)` and the caller passes `len(b) >= len(a)`.
                let ai = ctx.bind(unsafe { a.get_unchecked(i) });
                let bi = ctx.bind(unsafe { b.get_unchecked(i) });
                // "one of the two values exceeds the median" == max(ai, bi) > m.
                ctx.if_then(gt(max(ai, bi), m), move |ctx| {
                    ctx.store(acc, add(acc, add(ai, bi)));
                });
                ctx.store(i, add(i, 1u64));
            });
            acc
        },
    );
    let compiled = compiler.compile(f).expect("sum_above_median compile");
    Box::new(move |a, b| compiled.as_fn().call(a, b))
}

/// The JIT backends available in this build: always Cranelift, plus LLVM under `--features llvm`.
fn backends() -> Vec<(&'static str, JitBackend)> {
    #[allow(unused_mut)]
    let mut v = vec![("cranelift", JitBackend::Cranelift)];
    #[cfg(feature = "llvm")]
    v.push(("llvm", JitBackend::Llvm));
    v
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_filtered_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered_sum");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    let jits: Vec<(&str, FilteredSum)> = backends()
        .into_iter()
        .map(|(name, b)| (name, build_filtered_sum(b)))
        .collect();

    for &size in SIZES {
        let (a, _) = generate(size, 42);
        let z = 0i64; // ~50% selectivity
        let expected = filtered_sum_native(&a, z);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("native", size), &size, |bch, _| {
            bch.iter(|| black_box(filtered_sum_native(black_box(&a[..]), black_box(z))))
        });
        for (name, f) in &jits {
            assert_eq!(f(&a[..], z), expected, "{name} filtered_sum mismatch");
            group.bench_with_input(BenchmarkId::new(*name, size), &size, |bch, _| {
                bch.iter(|| black_box(f(black_box(&a[..]), black_box(z))))
            });
        }
    }
    group.finish();
}

fn bench_sum_above_median(c: &mut Criterion) {
    let mut group = c.benchmark_group("sum_above_median");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    let jits: Vec<(&str, SumAboveMedian)> = backends()
        .into_iter()
        .map(|(name, b)| (name, build_sum_above_median(b)))
        .collect();

    for &size in SIZES {
        let (a, b) = generate(size, 7);
        let expected = sum_above_median_native(&a, &b);
        group.throughput(Throughput::Elements(size as u64));

        group.bench_with_input(BenchmarkId::new("native", size), &size, |bch, _| {
            bch.iter(|| {
                black_box(sum_above_median_native(
                    black_box(&a[..]),
                    black_box(&b[..]),
                ))
            })
        });
        for (name, f) in &jits {
            assert_eq!(
                f(&a[..], &b[..]),
                expected,
                "{name} sum_above_median mismatch"
            );
            group.bench_with_input(BenchmarkId::new(*name, size), &size, |bch, _| {
                bch.iter(|| black_box(f(black_box(&a[..]), black_box(&b[..]))))
            });
        }
    }
    group.finish();
}

criterion_group!(benches, bench_filtered_sum, bench_sum_above_median);
criterion_main!(benches);
