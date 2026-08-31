//! **Iterator vs hand-rolled `while_loop`**, same kernel, same backend.
//!
//! Each shape is built twice from the same source data — once with an explicit
//! index loop, once with the staged iterator API — and the two are measured
//! side by side. The iterators are *fused*: `for_each` emits the loop directly
//! rather than materialising anything, so the claim under test is that the
//! emitted code is the same and the abstraction is free.
//!
//! Every pair is checked for equal results before being timed, so a benchmark
//! that silently computed the wrong thing fails rather than looks fast.
//!
//! Timings confirm the shapes match within noise, but the stronger evidence is
//! the emitted IR: for `sum`, the manual and iterator kernels lower to
//! *identical* Cranelift IR down to SSA numbering. Check any pair with
//! `RUST_LMS_DEBUG_IR=1` if a change here ever looks suspicious — a real
//! regression shows up as an extra instruction in the loop body, which is far
//! easier to read than a 5% timing delta.
//!
//! Run:
//!   cargo bench --bench iter                        # Cranelift
//!   MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm LLVM_SYS_220_PREFIX=$MLIR_SYS_220_PREFIX \
//!     PATH="$MLIR_SYS_220_PREFIX/bin:$PATH" DYLD_LIBRARY_PATH="$MLIR_SYS_220_PREFIX/lib" \
//!     cargo bench --bench iter --features llvm      # + LLVM/MLIR

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Duration;

use rust_lms::func::JitBackend;
use rust_lms::prelude::*;

const SIZES: &[usize] = &[10_000, 1_000_000];

type SliceFn = Box<dyn Fn(&[i64]) -> i64>;
type SlicePairFn = Box<dyn Fn(&[i64], &[i64]) -> i64>;
type CountFn = Box<dyn Fn(u64) -> u64>;

/// One labelled way of building a slice kernel, for side-by-side comparison.
type SliceVariant = (&'static str, fn(JitBackend) -> SliceFn);

fn data(size: usize, seed: u64) -> Vec<i64> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size)
        .map(|_| rng.random_range(-1_000i64..1_000))
        .collect()
}

/// The JIT backends available in this build.
fn backends() -> Vec<(&'static str, JitBackend)> {
    #[allow(unused_mut)]
    let mut v = vec![("cranelift", JitBackend::Cranelift)];
    #[cfg(feature = "llvm")]
    v.push(("llvm", JitBackend::Llvm));
    v
}

// ============================================================================
// 1. sum over a slice
// ============================================================================

fn sum_manual(backend: JitBackend) -> SliceFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("sum_manual", |ctx, a: Var<SRef<Slice<i64>>>| {
        let acc = ctx.var(0i64);
        let i = ctx.var(0u64);
        let n = ctx.bind(a.len());
        ctx.while_loop(lt(i, n), move |ctx| {
            // SAFETY: the loop condition proves `i < a.len()`.
            let v = ctx.bind(unsafe { a.get_unchecked(i) });
            ctx.store(acc, add(acc, v));
            ctx.store(i, add(i, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a| compiled.as_fn().call(a))
}

fn sum_iter(backend: JitBackend) -> SliceFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("sum_iter", |ctx, a: Var<SRef<Slice<i64>>>| {
        a.staged_iter().sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a| compiled.as_fn().call(a))
}

// ============================================================================
// 2. filtered sum — a branch inside the loop body
// ============================================================================

fn filtered_manual(backend: JitBackend) -> SliceFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("filtered_manual", |ctx, a: Var<SRef<Slice<i64>>>| {
        let acc = ctx.var(0i64);
        let i = ctx.var(0u64);
        let n = ctx.bind(a.len());
        ctx.while_loop(lt(i, n), move |ctx| {
            // SAFETY: the loop condition proves `i < a.len()`.
            let v = ctx.bind(unsafe { a.get_unchecked(i) });
            ctx.if_then(gt(v, 0i64), move |ctx| {
                ctx.store(acc, add(acc, v));
            });
            ctx.store(i, add(i, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a| compiled.as_fn().call(a))
}

fn filtered_iter(backend: JitBackend) -> SliceFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("filtered_iter", |ctx, a: Var<SRef<Slice<i64>>>| {
        let acc = ctx.var(0i64);
        a.staged_iter()
            .filter(|v| gt(v, 0i64))
            .for_each(ctx, move |ctx, v| {
                ctx.store(acc, add(acc, v));
            });
        acc
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a| compiled.as_fn().call(a))
}

/// The branchless form: `select(pred, v, 0)` every iteration instead of a
/// branch. Same answer, no misprediction.
fn filtered_branchless(backend: JitBackend) -> SliceFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("filtered_branchless", |ctx, a: Var<SRef<Slice<i64>>>| {
        a.staged_iter().sum_if(ctx, |v| gt(v, 0i64))
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a| compiled.as_fn().call(a))
}

// ============================================================================
// 3. map + filter chain — two fused combinators before the terminal
// ============================================================================

fn chain_manual(backend: JitBackend) -> SliceFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("chain_manual", |ctx, a: Var<SRef<Slice<i64>>>| {
        let acc = ctx.var(0i64);
        let i = ctx.var(0u64);
        let n = ctx.bind(a.len());
        ctx.while_loop(lt(i, n), move |ctx| {
            // SAFETY: the loop condition proves `i < a.len()`.
            let v = ctx.bind(unsafe { a.get_unchecked(i) });
            ctx.if_then(gt(v, 0i64), move |ctx| {
                let doubled = ctx.bind(mul(v, 2i64));
                ctx.store(acc, add(acc, doubled));
            });
            ctx.store(i, add(i, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a| compiled.as_fn().call(a))
}

fn chain_iter(backend: JitBackend) -> SliceFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("chain_iter", |ctx, a: Var<SRef<Slice<i64>>>| {
        a.staged_iter()
            .filter(|v| gt(v, 0i64))
            .map(|v| mul(v, 2i64))
            .sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a| compiled.as_fn().call(a))
}

// ============================================================================
// 4. zip — two slices walked in lockstep
// ============================================================================

fn zip_manual(backend: JitBackend) -> SlicePairFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun2(
        "zip_manual",
        |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
            let acc = ctx.var(0i64);
            let i = ctx.var(0u64);
            let n = ctx.bind(a.len());
            ctx.while_loop(lt(i, n), move |ctx| {
                // SAFETY: `i < len(a)`, and the caller passes `len(b) >= len(a)`.
                let ai = ctx.bind(unsafe { a.get_unchecked(i) });
                // SAFETY: as above.
                let bi = ctx.bind(unsafe { b.get_unchecked(i) });
                ctx.store(acc, add(acc, mul(ai, bi)));
                ctx.store(i, add(i, 1u64));
            });
            acc
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a, b| compiled.as_fn().call(a, b))
}

fn zip_iter(backend: JitBackend) -> SlicePairFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun2(
        "zip_iter",
        |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
            let acc = ctx.var(0i64);
            a.staged_iter().zip(b).for_each(ctx, move |ctx, ai, bi| {
                ctx.store(acc, add(acc, mul(ai, bi)));
            });
            acc
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |a, b| compiled.as_fn().call(a, b))
}

// ============================================================================
// 5. counted range — no slice, just the loop
//
// Accumulates with `xor`, not `+`. With `+` this is a polynomial recurrence and
// LLVM replaces the whole loop with a closed form — it reported the same 3.8 ns
// for 10_000 and 1_000_000 elements, which is a measurement of nothing.
// ============================================================================

fn range_manual(backend: JitBackend) -> CountFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("range_manual", |ctx, n: Var<u64>| {
        let acc = ctx.var(0u64);
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, n), move |ctx| {
            ctx.store(acc, bitxor(acc, mul(i, i)));
            ctx.store(i, add(i, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |n| compiled.as_fn().call(n))
}

fn range_iter_(backend: JitBackend) -> CountFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun1("range_iter", |ctx, n: Var<u64>| {
        let acc = ctx.var(0u64);
        range(0u64, n).for_each(ctx, move |ctx, i| {
            ctx.store(acc, bitxor(acc, mul(i, i)));
        });
        acc
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |n| compiled.as_fn().call(n))
}

// ============================================================================
// Benchmarks
// ============================================================================

/// Time several builds of one slice-consuming shape against each other, after
/// checking they all agree. The first variant is the reference.
fn compare_slice(c: &mut Criterion, name: &str, variants: &[SliceVariant]) {
    let mut group = c.benchmark_group(name);
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    let built: Vec<(&str, Vec<(&str, SliceFn)>)> = backends()
        .into_iter()
        .map(|(b, backend)| {
            (
                b,
                variants
                    .iter()
                    .map(|(l, f)| (*l, f(backend)))
                    .collect::<Vec<_>>(),
            )
        })
        .collect();

    for &size in SIZES {
        let a = data(size, 42);
        group.throughput(Throughput::Elements(size as u64));

        for (backend, vs) in &built {
            let expected = vs[0].1(&a[..]);
            for (label, f) in vs {
                assert_eq!(
                    f(&a[..]),
                    expected,
                    "{name}/{backend}/{label}: results differ"
                );
                group.bench_with_input(
                    BenchmarkId::new(format!("{backend}/{label}"), size),
                    &size,
                    |bch, _| bch.iter(|| black_box(f(black_box(&a[..])))),
                );
            }
        }
    }
    group.finish();
}

fn bench_sum(c: &mut Criterion) {
    compare_slice(c, "sum", &[("manual", sum_manual), ("iter", sum_iter)]);
}

fn bench_filtered(c: &mut Criterion) {
    compare_slice(
        c,
        "filtered_sum",
        &[
            ("manual", filtered_manual),
            ("iter", filtered_iter),
            ("branchless", filtered_branchless),
        ],
    );
}

fn bench_chain(c: &mut Criterion) {
    compare_slice(
        c,
        "filter_map_sum",
        &[("manual", chain_manual), ("iter", chain_iter)],
    );
}

fn bench_zip(c: &mut Criterion) {
    let mut group = c.benchmark_group("zip_dot");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    let built: Vec<(&str, SlicePairFn, SlicePairFn)> = backends()
        .into_iter()
        .map(|(b, backend)| (b, zip_manual(backend), zip_iter(backend)))
        .collect();

    for &size in SIZES {
        let (a, b) = (data(size, 1), data(size, 2));
        group.throughput(Throughput::Elements(size as u64));

        for (backend, m, it) in &built {
            assert_eq!(
                m(&a[..], &b[..]),
                it(&a[..], &b[..]),
                "zip_dot/{backend}: results differ"
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{backend}/manual"), size),
                &size,
                |bch, _| bch.iter(|| black_box(m(black_box(&a[..]), black_box(&b[..])))),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{backend}/iter"), size),
                &size,
                |bch, _| bch.iter(|| black_box(it(black_box(&a[..]), black_box(&b[..])))),
            );
        }
    }
    group.finish();
}

fn bench_range(c: &mut Criterion) {
    let mut group = c.benchmark_group("range_sum_squares");
    group.measurement_time(Duration::from_secs(3));
    group.sample_size(30);

    let built: Vec<(&str, CountFn, CountFn)> = backends()
        .into_iter()
        .map(|(b, backend)| (b, range_manual(backend), range_iter_(backend)))
        .collect();

    for &size in SIZES {
        let n = size as u64;
        group.throughput(Throughput::Elements(n));

        for (backend, m, it) in &built {
            assert_eq!(m(n), it(n), "range/{backend}: results differ");
            group.bench_with_input(
                BenchmarkId::new(format!("{backend}/manual"), size),
                &size,
                |bch, _| bch.iter(|| black_box(m(black_box(n)))),
            );
            group.bench_with_input(
                BenchmarkId::new(format!("{backend}/iter"), size),
                &size,
                |bch, _| bch.iter(|| black_box(it(black_box(n)))),
            );
        }
    }
    group.finish();
}

criterion_group!(
    benches,
    bench_sum,
    bench_filtered,
    bench_chain,
    bench_zip,
    bench_range
);
criterion_main!(benches);
