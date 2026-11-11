//! k-way merge of each node's runs, in the shape a query compiles to: `for n in nodes:
//! for x in merge(runs of n)`, so each merge is opened, primed and closed in a tight
//! loop over small runs. Synthetic, to compare the merges on layouts the graph500 data
//! of `graph_kmerge.rs` does not have; that bench is the one to read for real data.
//!
//! Every variant folds the merged stream into an order-sensitive checksum
//! (`acc * 31 + x`), checked against itertools before it is timed.
//!
//! Groups are `{group}/{layout}/deg{D}`, ids `{backend}/{variant}/{K}`:
//!
//! - `K`: the runs per node (fixed at staging time for `kmerge_by`, a run-time loop
//!   for `heap`).
//! - `D`: the nodes' mean degree, uniform in `0..=2D`; a node's elements are split
//!   over its `K` runs, so a run averages `D / K` elements.
//! - layout `random`: each element in a random run, so runs interleave;
//!   `disjoint`: each run a contiguous range of the node's keys, so one runs ahead.
//!
//! Variants:
//!
//! - `nested`: a balanced tree of two-way `merge_by`s (`K` in 2, 3, 4, 8).
//! - `chain`, `linear`: `kmerge_by` with each plan (a chain of compare branches, a pass
//!   over all heads); code for every run.
//! - `heap`: `kmerge_rows`, the runs pushed by a staged loop: the same code for any `K`.
//! - `host_itertools`: the same merge in Rust.
//!
//! `kmerge_nodes` has `K` up to 8 and every variant; `kmerge_nodes_wide` has `K` 16 to
//! 64, runs mostly empty or of one element as a node's rows over many layers' runs
//! are, and only `linear` (O(K) per element) and `heap` (O(log K)).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Duration;

use rust_lms::func::JitBackend;
use rust_lms::iter::{Plan, kmerge_rows};
use rust_lms::prelude::*;

/// Elements per graph.
const N: usize = 1 << 20;

type S = SRef<Slice<u64>>;

fn backends() -> Vec<(&'static str, JitBackend)> {
    #[allow(unused_mut)]
    let mut v = vec![("cranelift", JitBackend::Cranelift)];
    #[cfg(feature = "llvm")]
    v.push(("llvm", JitBackend::Llvm));
    v
}

fn checksum(acc: u64, x: u64) -> u64 {
    acc.wrapping_mul(31).wrapping_add(x)
}

// =============================================================================
// JIT kernels
// =============================================================================

/// The `k` runs of `values` bounded by `offsets`, as slices.
fn part_slices(ctx: &Ctx, values: Var<S>, offsets: Var<S>, k: usize) -> Vec<Var<S>> {
    (0..k as u64)
        .map(|j| {
            // SAFETY: the host passes `k + 1` ascending offsets within `values`.
            unsafe {
                let start = offsets.get_unchecked(j);
                let end = offsets.get_unchecked(j + 1);
                ctx.bind(values.subslice_unchecked(start, end))
            }
        })
        .collect()
}

fn fold_into<I: StagedIterator>(
    ctx: &Ctx,
    acc: Var<u64>,
    it: I,
    value: impl Fn(&Ctx, Var<I::Item>) -> Var<u64>,
) {
    it.for_each(ctx, move |ctx, x| {
        let x = value(ctx, x);
        ctx.store(acc, acc * 31u64 + x);
    });
}

type M2<I, P> = MergeBy<I, I, P>;
type M4<I, P> = MergeBy<M2<I, P>, M2<I, P>, P>;
type M8<I, P> = MergeBy<M4<I, P>, M4<I, P>, P>;

type Item<J> = <<J as IntoStagedIterator>::Iter as StagedIterator>::Item;

fn m2<J, P, C>(p: &mut impl Iterator<Item = J>, le: P) -> M2<J::Iter, P>
where
    J: IntoStagedIterator,
    Item<J>: CopyType + 'static,
    P: Fn(Var<Item<J>>, Var<Item<J>>) -> C + Copy,
    C: Staged<Out = bool> + 'static,
{
    let a = p.next().unwrap().staged_iter();
    a.merge_by(p.next().unwrap(), le)
}

macro_rules! tree {
    ($name:ident, $out:ident, $half:ident) => {
        fn $name<J, P, C>(p: &mut impl Iterator<Item = J>, le: P) -> $out<J::Iter, P>
        where
            J: IntoStagedIterator,
            Item<J>: CopyType + 'static,
            P: Fn(Var<Item<J>>, Var<Item<J>>) -> C + Copy,
            C: Staged<Out = bool> + 'static,
        {
            let a = $half(p, le);
            a.merge_by($half(p, le), le)
        }
    };
}
tree!(m4, M4, m2);
tree!(m8, M8, m4);

/// `runs` merged by a balanced tree of `merge_by`s, folded into `acc`.
fn nested_into(ctx: &Ctx, acc: Var<u64>, runs: Vec<Var<S>>) {
    let p = &mut runs.into_iter();
    let id = |_: &Ctx, x| x;
    match p.len() {
        2 => fold_into(ctx, acc, m2(p, le), id),
        3 => fold_into(ctx, acc, m2(p, le).merge_by(p.next().unwrap(), le), id),
        4 => fold_into(ctx, acc, m4(p, le), id),
        8 => fold_into(ctx, acc, m8(p, le), id),
        k => unreachable!("{k} runs"),
    }
}

/// The parts of `values` bounded by `offsets` (`k + 1` of them), merged by
/// `kmerge_rows` with a heap of `k` entries: one staged push, in a loop over the parts.
fn heap_merge(
    values: Var<S>,
    offsets: Var<S>,
    k: usize,
) -> impl StagedIterator<Item = ZipItem<u64, u64>> {
    // SAFETY: the host passes `k + 1` ascending offsets within `values`, whose parts
    // are sorted, so there are at most `k` rows.
    unsafe {
        kmerge_rows(k, move |ctx, heap| {
            range(0u64, k as u64).for_each(ctx, |ctx, j| {
                let start = ctx.bind(offsets.get_unchecked(j));
                let end = ctx.bind(offsets.get_unchecked(j + 1u64));
                let part = ctx.bind(values.subslice_unchecked(start, end));
                heap.push(ctx, part, part);
            })
        })
    }
}

/// `merge` with the plan of a variant named `chain` or `linear`.
fn configure<I, F>(merge: KMergeBy<I, F>, variant: &str) -> KMergeBy<I, F> {
    merge.with_plan(match variant {
        "chain" => Plan::Chain,
        "linear" => Plan::Linear,
        v => unreachable!("{v}"),
    })
}

// =============================================================================
// Nodes
// =============================================================================

const KS: &[usize] = &[2, 3, 4, 8];
const DEGREES: &[usize] = &[4, 32];
const LAYOUTS: &[&str] = &["random", "disjoint"];
const VARIANTS: &[&str] = &["nested", "chain", "linear", "heap"];
const WIDE_KS: &[usize] = &[16, 32, 64];
const WIDE_DEGREES: &[usize] = &[8, 64];
const WIDE_VARIANTS: &[&str] = &["linear", "heap"];

/// `N` keys over nodes of degree uniform in `0..=2 * degree`, each node's keys split
/// over `k` runs: node `n`'s run `j` is `values[offsets[n * k + j]..offsets[n * k + j + 1]]`.
/// Keys are `i / 2 * 3`, `i` the node's first position in `values` plus the element's
/// index within the node: ascending within a node, with gaps and ties.
struct Nodes {
    values: Vec<u64>,
    offsets: Vec<u64>,
    nodes: u64,
}

impl Nodes {
    fn new(layout: &str, k: usize, degree: usize) -> Self {
        let mut rng = StdRng::seed_from_u64((k * 1000 + degree) as u64);
        let (mut values, mut offsets, mut nodes) = (Vec::with_capacity(N), vec![0u64], 0);
        while values.len() < N {
            let d = rng.random_range(0..=2 * degree).min(N - values.len());
            let runs: Vec<usize> = match layout {
                "random" => (0..d).map(|_| rng.random_range(0..k)).collect(),
                "disjoint" => (0..d).map(|i| i * k / d.max(1)).collect(),
                _ => unreachable!(),
            };
            for j in 0..k {
                let start = values.len();
                values.extend(
                    (0..d)
                        .filter(|&i| runs[i] == j)
                        .map(|i| ((start + i) / 2 * 3) as u64),
                );
                offsets.push(values.len() as u64);
            }
            nodes += 1;
        }
        Nodes {
            values,
            offsets,
            nodes,
        }
    }

    /// Each node's runs merged by itertools: `merge` for two runs, `kmerge` above.
    fn host(&self, k: usize) -> u64 {
        (0..self.nodes as usize).fold(0, |acc, n| {
            let offsets = &self.offsets[n * k..=n * k + k];
            let run = |j: usize| {
                self.values[offsets[j] as usize..offsets[j + 1] as usize]
                    .iter()
                    .copied()
            };
            let merged: Box<dyn Iterator<Item = u64>> = match k {
                2 => Box::new(run(0).merge(run(1))),
                _ => Box::new((0..k).map(run).kmerge()),
            };
            merged.fold(acc, checksum)
        })
    }
}

type NodesFn = Box<dyn Fn(&Nodes) -> u64>;

/// `for n in nodes: for x in merge(runs of n): acc = acc * 31 + x`.
fn kernel(backend: JitBackend, variant: &str, k: usize) -> NodesFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let variant = variant.to_string();
    let f = compiler.fun3(
        "kmerge_nodes",
        move |ctx, values: Var<S>, offsets: Var<S>, nodes: Var<u64>| {
            let acc = ctx.var(0u64);
            range(0u64, nodes).for_each(ctx, |ctx, n| {
                let base = ctx.bind(n * k as u64);
                // SAFETY: the host passes `nodes * k + 1` ascending offsets within `values`.
                let offsets = unsafe {
                    let end = ctx.bind(base + (k as u64 + 1));
                    ctx.bind(offsets.subslice_unchecked(base, end))
                };
                // The heap pushes its runs from a staged loop; the others take them as
                // slices staged once each.
                let runs = || part_slices(ctx, values, offsets, k);
                match variant.as_str() {
                    "heap" => fold_into(ctx, acc, heap_merge(values, offsets, k), |ctx, x| {
                        ctx.bind(x.first())
                    }),
                    "nested" => nested_into(ctx, acc, runs()),
                    v => fold_into(ctx, acc, configure(kmerge_by(runs(), |_, x| x), v), |_, x| x),
                }
            });
            acc
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |p| {
        compiled
            .as_fn()
            .call(&p.values[..], &p.offsets[..], p.nodes)
    })
}

// =============================================================================
// Benches
// =============================================================================

fn bench_nodes_of(
    c: &mut Criterion,
    name: &str,
    ks: &[usize],
    degrees: &[usize],
    variants: &[&str],
) {
    for &layout in LAYOUTS {
        for &degree in degrees {
            let mut group = c.benchmark_group(format!("{name}/{layout}/deg{degree}"));
            group.warm_up_time(Duration::from_millis(1200));
            group.measurement_time(Duration::from_secs(10));
            group.sample_size(50);
            group.throughput(Throughput::Elements(N as u64));
            for &k in ks {
                let nodes = Nodes::new(layout, k, degree);
                let expected = nodes.host(k);
                group.bench_with_input(BenchmarkId::new("host_itertools", k), &k, |b, _| {
                    b.iter(|| black_box(&nodes).host(k))
                });
                for (backend_name, backend) in backends() {
                    for &variant in variants {
                        let f = kernel(backend, variant, k);
                        assert_eq!(
                            f(&nodes),
                            expected,
                            "{name}/{layout}/{backend_name}/{variant}/{k}"
                        );
                        group.bench_with_input(
                            BenchmarkId::new(format!("{backend_name}/{variant}"), k),
                            &k,
                            |b, _| b.iter(|| f(black_box(&nodes))),
                        );
                    }
                }
            }
            group.finish();
        }
    }
}

fn bench_nodes(c: &mut Criterion) {
    bench_nodes_of(c, "kmerge_nodes", KS, DEGREES, VARIANTS);
    bench_nodes_of(c, "kmerge_nodes_wide", WIDE_KS, WIDE_DEGREES, WIDE_VARIANTS);
}

criterion_group!(benches, bench_nodes);
criterion_main!(benches);
