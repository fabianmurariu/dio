//! **Graph neighbour access: visible slices vs chunked vs per-element FFI.**
//!
//! One CSR graph (`offsets`, `targets`, per-edge `ts`) sits behind every variant;
//! only *how staged code reaches a node's neighbours* changes:
//!
//! | variant       | what the kernel sees                                            |
//! |---------------|-----------------------------------------------------------------|
//! | `native`      | plain Rust over the CSR arrays (reference line, not staged)     |
//! | `csr`         | the CSR slices themselves: pointer + offset loops, no FFI       |
//! | `slice_ffi`   | one extern call per node returning its neighbours as a slice    |
//! | `chunked`     | [`ChunkedIter`]: stack-slot chunks; one call per list ≤ 64      |
//! | `unbuffered`  | [`ReusedOpaqueIter`]: one indirect extern call per neighbour    |
//!
//! Each workload is written once, generic over the stage-0 [`Nbrs`] strategy, so
//! every variant runs the same algorithm:
//!
//! - `sum` — Σ of all neighbour ids.
//! - `dst_gt_src` — count of edges with `dst > src` (predicate uses the outer node).
//! - `any_mod16` — nodes with a neighbour divisible by 16 (short-circuits: the
//!   break-and-drop path).
//! - `window` — Σ of neighbour ids over edges with `ts < WINDOW`. The staged
//!   variants filter a visible `ts` slice; the iterator variants filter inside
//!   the Rust iterator, which staged code cannot see.
//! - `two_hop` — for the first `TWO_HOP_SOURCES` nodes `n`, count 2-hop paths
//!   `n → m → w` with `w > n` (two nested neighbour loops).
//!
//! Graphs: 1M nodes with power-law out-degrees (Lomax, mean ≈ 10, hubs capped at
//! 100k — mostly tiny lists, occasionally huge), and a uniform degree-10 control.
//!
//! Every kernel's result is checked against `native` before it is timed.
//! `native` is a reference line, not a like-for-like baseline: its `sum` is one
//! flat (vectorized) pass over `targets` that ignores node boundaries.
//!
//! Run:
//!   cargo bench -p rust-lms-std --bench graph_iter                  # Cranelift
//!   cargo bench -p rust-lms-std --bench graph_iter --features llvm  # + LLVM (needs LLVM 22)
//!   cargo bench -p rust-lms-std --bench graph_iter -- 'two_hop'     # one workload
//!
//! With Homebrew LLVM on macOS, prefix the `llvm` run with
//! `LIBRARY_PATH=/opt/homebrew/lib` (LLVM links `-lzstd`, which Apple's linker
//! does not look for there).

#![allow(clippy::missing_safety_doc)]

use std::hint::black_box;
use std::ops::Range;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rust_lms::prelude::*;
use rust_lms_std::{ChunkStart, ChunkedIterFns, ChunkedIterKind};

const NODES: usize = 1_000_000;
const MAX_DEGREE: usize = 100_000;
const WINDOW: u64 = 250; // `ts` is uniform in 0..1000, so ~25% of edges pass
const TWO_HOP_SOURCES: u64 = 100_000;

// =============================================================================
// Graph
// =============================================================================

pub struct Graph {
    offsets: Vec<u64>,
    targets: Vec<u64>,
    ts: Vec<u64>,
}

impl Graph {
    fn build(degrees: impl Iterator<Item = usize>, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut offsets = vec![0u64];
        let mut targets = Vec::new();
        for d in degrees {
            targets.extend((0..d).map(|_| rng.random_range(0..NODES as u64)));
            offsets.push(targets.len() as u64);
        }
        let ts = (0..targets.len())
            .map(|_| rng.random_range(0..1000))
            .collect();
        Graph {
            offsets,
            targets,
            ts,
        }
    }

    /// Lomax (Pareto II) out-degrees: `3 · (u^(-1/1.3) − 1)`, mean ≈ 10.
    fn power_law(seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let degrees: Vec<usize> = (0..NODES)
            .map(|_| {
                let u: f64 = rng.random_range(f64::EPSILON..1.0);
                ((3.0 * (u.powf(-1.0 / 1.3) - 1.0)) as usize).min(MAX_DEGREE)
            })
            .collect();
        Self::build(degrees.into_iter(), seed + 1)
    }

    fn uniform(seed: u64) -> Self {
        Self::build(std::iter::repeat_n(10, NODES), seed)
    }

    fn range(&self, n: u64) -> Range<usize> {
        self.offsets[n as usize] as usize..self.offsets[n as usize + 1] as usize
    }

    fn nbrs(&self, n: u64) -> &[u64] {
        &self.targets[self.range(n)]
    }

    fn window_nbrs(&self, n: u64) -> impl Iterator<Item = u64> + '_ {
        let r = self.range(n);
        self.targets[r.clone()]
            .iter()
            .zip(&self.ts[r])
            .filter(|(_, t)| **t < WINDOW)
            .map(|(d, _)| *d)
    }

    fn describe(&self, name: &str) {
        let n = self.offsets.len() - 1;
        let max = (0..n as u64)
            .map(|v| self.range(v).len())
            .max()
            .unwrap_or(0);
        let small = (0..n as u64).filter(|&v| self.range(v).len() <= 4).count();
        eprintln!(
            "graph {name}: {n} nodes, {} edges, mean degree {:.1}, max {max}, {:.0}% of nodes with ≤4 neighbours",
            self.targets.len(),
            self.targets.len() as f64 / n as f64,
            100.0 * small as f64 / n as f64,
        );
    }
}

// =============================================================================
// Host externs for the FFI variants
// =============================================================================

type G = SRef<Opaque<Graph>>;

/// `slice_ffi`: a node's neighbours as a slice into the CSR arrays.
#[extern_fn]
pub extern "C" fn nbrs_slice(g: &Graph, n: u64) -> FatSlice<u64> {
    FatSlice::from_slice(g.nbrs(n))
}

/// `slice_ffi` (window): the matching `ts` slice.
#[extern_fn]
pub extern "C" fn ts_slice(g: &Graph, n: u64) -> FatSlice<u64> {
    FatSlice::from_slice(&g.ts[g.range(n)])
}

#[extern_fn]
pub unsafe extern "C" fn reused_nbrs(g: &Graph, n: u64, slot: *mut ()) {
    // SAFETY: `slot` is the reused slot the kernel reserved for this level, and
    // the graph outlives every kernel call.
    unsafe { emplace_iter(slot as *mut OpaqueIterSlot<u64>, g.nbrs(n).iter().copied()) };
}

#[extern_fn]
pub unsafe extern "C" fn reused_window(g: &Graph, n: u64, slot: *mut ()) {
    // SAFETY: as in `reused_nbrs`.
    unsafe { emplace_iter(slot as *mut OpaqueIterSlot<u64>, g.window_nbrs(n)) };
}

#[extern_fn]
pub unsafe extern "C" fn chunked_nbrs(g: &Graph, n: u64, slot: &mut ChunkStart<u64>) {
    // SAFETY: the graph outlives every kernel call.
    unsafe { slot.start_borrowed(g.nbrs(n).iter().copied()) };
}

#[extern_fn]
pub unsafe extern "C" fn chunked_window(g: &Graph, n: u64, slot: &mut ChunkStart<u64>) {
    // SAFETY: the graph outlives every kernel call.
    unsafe { slot.start_borrowed(g.window_nbrs(n)) };
}

struct ChunkedNbrs;
impl ChunkedIterKind for ChunkedNbrs {
    type Item = u64;
    type Init = ChunkedNbrsExtern;
}

struct ChunkedWindow;
impl ChunkedIterKind for ChunkedWindow {
    type Item = u64;
    type Init = ChunkedWindowExtern;
}

struct ReusedNbrs;
unsafe impl ReusedOpaqueIterKind for ReusedNbrs {
    type Item = u64;
    type Init = ReusedNbrsExtern;
}

struct ReusedWindow;
unsafe impl ReusedOpaqueIterKind for ReusedWindow {
    type Item = u64;
    type Init = ReusedWindowExtern;
}

// =============================================================================
// Stage 0: how a kernel reaches a node's neighbours
// =============================================================================

/// A neighbour-access strategy. `ctx` is the body the iterator will run in, so a
/// strategy can bind per-node values (CSR bounds, a returned slice) once.
trait Nbrs {
    fn nbrs(&self, ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_;
    /// Neighbours over edges with `ts < WINDOW`.
    fn window_nbrs(&self, ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_;
}

type U64s = Var<SRef<Slice<u64>>>;

/// `csr`: the kernel holds the CSR slices.
struct Csr {
    offsets: U64s,
    targets: U64s,
    ts: U64s,
}

impl Csr {
    fn bounds(&self, ctx: &mut Ctx, n: Var<u64>) -> (Var<u64>, Var<u64>) {
        // SAFETY: `n < NODES` and `offsets` has `NODES + 1` entries.
        let start = ctx.bind(unsafe { self.offsets.get_unchecked(n) });
        let end = ctx.bind(unsafe { self.offsets.get_unchecked(add(n, 1u64)) });
        (start, end)
    }
}

impl Nbrs for Csr {
    fn nbrs(&self, ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_ {
        let (start, end) = self.bounds(ctx, n);
        let targets = self.targets;
        // SAFETY: CSR offsets are within `targets`.
        range(start, end).map(move |j| unsafe { targets.get_unchecked(j) })
    }

    fn window_nbrs(&self, ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_ {
        let (start, end) = self.bounds(ctx, n);
        let (targets, ts) = (self.targets, self.ts);
        // SAFETY: CSR offsets are within `targets` and `ts`.
        range(start, end)
            .filter(move |j| lt(unsafe { ts.get_unchecked(j) }, WINDOW))
            .map(move |j| unsafe { targets.get_unchecked(j) })
    }
}

/// `slice_ffi`: one extern call per node returns its neighbour slice.
struct SliceFfi {
    g: Var<G>,
    nbrs: ExternRef<NbrsSliceExtern>,
    ts: ExternRef<TsSliceExtern>,
}

impl Nbrs for SliceFfi {
    fn nbrs(&self, ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_ {
        let s = ctx.bind(call_extern2(self.nbrs, self.g, n));
        // SAFETY: `j < s.len()`, and the slice points into the live graph.
        range(0u64, s.len()).map(move |j| unsafe { s.get_unchecked(j) })
    }

    fn window_nbrs(&self, ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_ {
        let s = ctx.bind(call_extern2(self.nbrs, self.g, n));
        let t = ctx.bind(call_extern2(self.ts, self.g, n));
        // SAFETY: both slices have the node's degree as length, into the live graph.
        range(0u64, s.len())
            .filter(move |j| lt(unsafe { t.get_unchecked(j) }, WINDOW))
            .map(move |j| unsafe { s.get_unchecked(j) })
    }
}

/// `chunked`: chunks through a stack slot per nesting level.
struct Chunked {
    g: Var<G>,
    nbrs: ChunkedIterFns<ChunkedNbrs>,
    window: ChunkedIterFns<ChunkedWindow>,
}

impl Nbrs for Chunked {
    fn nbrs(&self, _ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_ {
        self.nbrs.iter2(self.g, n)
    }

    fn window_nbrs(&self, _ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_ {
        self.window.iter2(self.g, n)
    }
}

/// `unbuffered`: one indirect extern call per neighbour.
struct Unbuffered {
    g: Var<G>,
    nbrs: ReusedOpaqueIterFns<ReusedNbrs>,
    window: ReusedOpaqueIterFns<ReusedWindow>,
}

impl Nbrs for Unbuffered {
    fn nbrs(&self, _ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_ {
        self.nbrs.iter2(self.g, n)
    }

    fn window_nbrs(&self, _ctx: &mut Ctx, n: Var<u64>) -> impl StagedIterator<Item = u64> + '_ {
        self.window.iter2(self.g, n)
    }
}

// =============================================================================
// Workloads (written once, for every strategy)
// =============================================================================

#[derive(Clone, Copy, Debug)]
enum Workload {
    Sum,
    DstGtSrc,
    AnyMod16,
    Window,
    TwoHop,
}

impl Workload {
    fn name(self) -> &'static str {
        match self {
            Workload::Sum => "sum",
            Workload::DstGtSrc => "dst_gt_src",
            Workload::AnyMod16 => "any_mod16",
            Workload::Window => "window",
            Workload::TwoHop => "two_hop",
        }
    }

    fn stage<S: Nbrs>(self, ctx: &mut Ctx, s: &S) -> Var<u64> {
        let acc = ctx.var(0u64);
        let nodes = NODES as u64;
        match self {
            Workload::Sum => range(0u64, nodes).for_each(ctx, |ctx, n| {
                s.nbrs(ctx, n)
                    .for_each(ctx, move |ctx, d| ctx.store(acc, add(acc, d)));
            }),
            Workload::DstGtSrc => range(0u64, nodes).for_each(ctx, |ctx, n| {
                let c = s.nbrs(ctx, n).count_if(ctx, move |d| gt(d, n));
                ctx.store(acc, add(acc, c));
            }),
            Workload::AnyMod16 => range(0u64, nodes).for_each(ctx, |ctx, n| {
                let hit = s.nbrs(ctx, n).any(ctx, |d| eq(bitand(d, 15u64), 0u64));
                ctx.store(acc, add(acc, select(hit, 1u64, 0u64)));
            }),
            Workload::Window => range(0u64, nodes).for_each(ctx, |ctx, n| {
                s.window_nbrs(ctx, n)
                    .for_each(ctx, move |ctx, d| ctx.store(acc, add(acc, d)));
            }),
            Workload::TwoHop => range(0u64, TWO_HOP_SOURCES).for_each(ctx, |ctx, n| {
                s.nbrs(ctx, n).for_each(ctx, |ctx, m| {
                    let c = s.nbrs(ctx, m).count_if(ctx, move |w| gt(w, n));
                    ctx.store(acc, add(acc, c));
                });
            }),
        }
        acc
    }

    fn native(self, g: &Graph) -> u64 {
        let nodes = 0..NODES as u64;
        match self {
            Workload::Sum => g.targets.iter().sum(),
            Workload::DstGtSrc => nodes
                .map(|n| g.nbrs(n).iter().filter(|&&d| d > n).count() as u64)
                .sum(),
            Workload::AnyMod16 => nodes
                .filter(|&n| g.nbrs(n).iter().any(|d| d & 15 == 0))
                .count() as u64,
            Workload::Window => nodes.map(|n| g.window_nbrs(n).sum::<u64>()).sum(),
            Workload::TwoHop => (0..TWO_HOP_SOURCES)
                .map(|n| {
                    g.nbrs(n)
                        .iter()
                        .map(|&m| g.nbrs(m).iter().filter(|&&w| w > n).count() as u64)
                        .sum::<u64>()
                })
                .sum(),
        }
    }

    /// Neighbour visits per run, for throughput (an upper bound for `any_mod16`).
    fn visits(self, g: &Graph) -> u64 {
        match self {
            Workload::TwoHop => (0..TWO_HOP_SOURCES)
                .flat_map(|n| g.nbrs(n))
                .map(|&m| g.nbrs(m).len() as u64)
                .sum(),
            _ => g.targets.len() as u64,
        }
    }
}

// =============================================================================
// Kernels
// =============================================================================

type Kernel = Box<dyn Fn(&Graph) -> u64>;

#[derive(Clone, Copy)]
enum Variant {
    Csr,
    SliceFfi,
    Chunked,
    Unbuffered,
}

impl Variant {
    fn name(self) -> String {
        match self {
            Variant::Csr => "csr".into(),
            Variant::SliceFfi => "slice_ffi".into(),
            Variant::Chunked => "chunked".into(),
            Variant::Unbuffered => "unbuffered".into(),
        }
    }

    fn build(self, backend: JitBackend, w: Workload) -> Kernel {
        let mut compiler = Compiler::new().with_backend(backend);
        match self {
            Variant::Csr => {
                let f = compiler.fun3("csr", move |ctx, offsets: U64s, targets: U64s, ts: U64s| {
                    w.stage(
                        ctx,
                        &Csr {
                            offsets,
                            targets,
                            ts,
                        },
                    )
                });
                let k = compiler.compile(f).expect("compile");
                Box::new(move |g| k.as_fn().call(&g.offsets, &g.targets, &g.ts))
            }
            Variant::SliceFfi => {
                let nbrs = compiler.extern_fn();
                let ts = compiler.extern_fn();
                let f = compiler.fun1("slice_ffi", move |ctx, g: Var<G>| {
                    w.stage(ctx, &SliceFfi { g, nbrs, ts })
                });
                let k = compiler.compile(f).expect("compile");
                Box::new(move |g| k.as_fn().call(g))
            }
            Variant::Chunked => {
                let nbrs = ChunkedIterFns::register(&mut compiler);
                let window = ChunkedIterFns::register(&mut compiler);
                let f = compiler.fun1("chunked", move |ctx, g: Var<G>| {
                    w.stage(ctx, &Chunked { g, nbrs, window })
                });
                let k = compiler.compile(f).expect("compile");
                Box::new(move |g| k.as_fn().call(g))
            }
            Variant::Unbuffered => {
                let nbrs = compiler.reused_opaque_iter_fns();
                let window = compiler.reused_opaque_iter_fns();
                let f = compiler.fun1("unbuffered", move |ctx, g: Var<G>| {
                    w.stage(ctx, &Unbuffered { g, nbrs, window })
                });
                let k = compiler.compile(f).expect("compile");
                Box::new(move |g| k.as_fn().call(g))
            }
        }
    }
}

fn backends() -> Vec<(&'static str, JitBackend)> {
    #[allow(unused_mut)]
    let mut v = vec![("cranelift", JitBackend::Cranelift)];
    #[cfg(feature = "llvm")]
    v.push(("llvm", JitBackend::Llvm));
    v
}

// =============================================================================
// Benchmark
// =============================================================================

const WORKLOADS: &[Workload] = &[
    Workload::Sum,
    Workload::DstGtSrc,
    Workload::AnyMod16,
    Workload::Window,
    Workload::TwoHop,
];

fn bench_graph_iter(c: &mut Criterion) {
    let graphs = [
        ("power_law", Graph::power_law(7)),
        ("uniform", Graph::uniform(11)),
    ];
    for (name, g) in &graphs {
        g.describe(name);
    }

    for &w in WORKLOADS {
        for (graph_name, g) in &graphs {
            let expected = w.native(g);
            let mut group = c.benchmark_group(format!("graph_iter/{}/{graph_name}", w.name()));
            group.throughput(Throughput::Elements(w.visits(g)));

            group.bench_function("native", |b| b.iter(|| black_box(w.native(black_box(g)))));

            let variants = [
                Variant::Csr,
                Variant::SliceFfi,
                Variant::Chunked,
                Variant::Unbuffered,
            ];
            for v in variants {
                for (backend_name, backend) in backends() {
                    let k = v.build(backend, w);
                    assert_eq!(
                        k(g),
                        expected,
                        "{}/{backend_name} computed the wrong {} on {graph_name}",
                        v.name(),
                        w.name()
                    );
                    group.bench_function(BenchmarkId::new(backend_name, v.name()), |b| {
                        b.iter(|| black_box(k(black_box(g))))
                    });
                }
            }
            group.finish();
        }
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3));
    targets = bench_graph_iter
}
criterion_main!(benches);
