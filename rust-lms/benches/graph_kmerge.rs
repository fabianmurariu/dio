//! **A node's neighbours over layers and runs, merged: staged vs Rust.**
//!
//! The shape a query compiles to: scan nodes, and for each, merge its adjacency rows
//! across the picked layers and each layer's runs into one stream sorted by
//! neighbour, dropping repeats (an edge in several picked layers), and fold it. The
//! merge's state is set up for every node, inside the scan loop.
//!
//! **Graph.** The graph500 scale-18 edge list in `benches/` (174,147 nodes, 7.6M
//! directed edges, each in both directions; degree median 12, mean 44 per direction,
//! max 25,707), so every run is on the same data. Each edge, in `(src, dst)` order,
//! is given id = its position, then a random layer of `L` and a random run of `R`
//! (seeded, so the same split every time); with `overlap`, 30% of edges are also in
//! a second random layer (another random run), with the same id. Each (layer, run)
//! is a matrix of its own: per node, a row of neighbours ascending and their edge
//! ids, at `nbrs[offsets[m * (V + 1) + n]..offsets[m * (V + 1) + n + 1]]` for matrix
//! `m = layer * R + run`. Graphs: `L` in 1, 2, 4 by `R` in 1, 2, 4, 8, and with
//! `L >= 2`, with and without overlap.
//!
//! **Scans.** `all`: every node, in id order. `hubs`: only the 5% of nodes with the
//! largest degree (8,707 nodes, degree at least 182 per direction, 64% of the edges),
//! in id order: long rows.
//!
//! **Variants.** Each folds `(nbr, eid)` into an order-sensitive checksum, checked
//! against `rust/flat` before it is timed.
//!
//! | id               | the merge of a node's rows                                       |
//! |------------------|------------------------------------------------------------------|
//! | `rust/flat`      | itertools `kmerge_by` over all `L x R` rows, then `dedup_by`      |
//! | `rust/nested`    | `kmerge_by` per layer over its `R` rows, then `kmerge_by` of those |
//! | `{be}/flat`      | `kmerge_rows` over all `L x R` rows (a loop pushes them), dedup   |
//! | `{be}/nested`    | `kmerge_rows` per layer, `kmerge_by` of the `L` layers, dedup     |
//!
//! `{be}` is `cranelift`, and `llvm` with `--features llvm`. `flat` is what
//! `layers([..]).neighbours` compiles to in db4 today; `nested` is a merge per
//! layer, as a layer at a time would give. The nested variants need `L >= 2`.
//!
//! Throughput is the edges popped (repeats included). The line printed before each
//! group gives the nodes scanned, the edges popped, the non-empty rows a node has
//! on average, and the share of popped edges dropped as repeats.
//!
//! Run:
//!   cargo bench -p rust-lms --bench graph_kmerge
//!   LIBRARY_PATH=/opt/homebrew/lib cargo bench -p rust-lms --bench graph_kmerge --features llvm
//!   cargo bench -p rust-lms --bench graph_kmerge -- 'hubs/L4'   # a subset

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use flate2::read::GzDecoder;
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::io::{BufRead, BufReader};
use std::sync::OnceLock;
use std::time::Duration;

use rust_lms::func::JitBackend;
use rust_lms::iter::{Heap, KMergeRows, kmerge_rows};
use rust_lms::prelude::*;

const EDGE_LIST: &str = "benches/graph500-scale18-ef16_adj.edges.gz";
const LAYERS: &[usize] = &[1, 2, 4];
const RUNS: &[usize] = &[1, 2, 4, 8];
/// The share of edges, in percent, also in a second layer when overlapping.
const OVERLAP: u32 = 30;
/// The share of nodes, by largest degree, the `hubs` scan visits, in percent.
const HUBS: usize = 5;

type S = SRef<Slice<u64>>;

fn backends() -> Vec<(&'static str, JitBackend)> {
    #[allow(unused_mut)]
    let mut v = vec![("cranelift", JitBackend::Cranelift)];
    #[cfg(feature = "llvm")]
    v.push(("llvm", JitBackend::Llvm));
    v
}

fn checksum(acc: u64, nbr: u64, eid: u64) -> u64 {
    acc.wrapping_mul(31)
        .wrapping_add(nbr.wrapping_mul(3).wrapping_add(eid))
}

// =============================================================================
// Data
// =============================================================================

/// The edge list, sorted by `(src, dst)`, and the number of nodes.
fn edges() -> &'static (Vec<(u32, u32)>, usize) {
    static EDGES: OnceLock<(Vec<(u32, u32)>, usize)> = OnceLock::new();
    EDGES.get_or_init(|| {
        let path = format!("{}/{EDGE_LIST}", env!("CARGO_MANIFEST_DIR"));
        let file = std::fs::File::open(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
        let mut edges = Vec::with_capacity(7_600_696);
        for line in BufReader::new(GzDecoder::new(file)).lines() {
            let line = line.unwrap();
            let mut ids = line
                .split_ascii_whitespace()
                .map(|x| x.parse::<u32>().unwrap());
            edges.push((ids.next().unwrap(), ids.next().unwrap()));
        }
        edges.sort_unstable();
        let nodes = edges.iter().map(|&(s, d)| s.max(d)).max().unwrap() as usize + 1;
        (edges, nodes)
    })
}

/// The edge list split into `layers x runs` matrices (see the module docs).
struct Graph {
    layers: usize,
    runs: usize,
    overlap: bool,
    /// `V + 1`: the offsets of a matrix.
    stride: u64,
    offsets: Vec<u64>,
    nbrs: Vec<u64>,
    eids: Vec<u64>,
}

impl Graph {
    fn new(layers: usize, runs: usize, overlap: bool) -> Self {
        let (edges, nodes) = edges();
        let mut rng = StdRng::seed_from_u64((layers * 100 + runs * 10 + overlap as usize) as u64);
        let matrices = layers * runs;
        // The matrices each edge is in.
        let placed: Vec<(usize, Option<usize>)> = edges
            .iter()
            .map(|_| {
                let layer = rng.random_range(0..layers);
                let first = layer * runs + rng.random_range(0..runs);
                let second = (overlap && rng.random_range(0..100) < OVERLAP).then(|| {
                    let other = (layer + rng.random_range(1..layers)) % layers;
                    other * runs + rng.random_range(0..runs)
                });
                (first, second)
            })
            .collect();
        let stride = nodes + 1;
        let mut counts = vec![0u64; matrices * stride];
        for (&(src, _), &(first, second)) in edges.iter().zip(&placed) {
            for m in std::iter::once(first).chain(second) {
                counts[m * stride + src as usize + 1] += 1;
            }
        }
        let mut offsets = counts;
        for i in 1..offsets.len() {
            offsets[i] += offsets[i - 1];
        }
        let total = *offsets.last().unwrap() as usize;
        let (mut nbrs, mut eids) = (vec![0u64; total], vec![0u64; total]);
        let mut next = offsets.clone();
        // Edges are in `(src, dst)` order, so each row fills ascending.
        for (eid, (&(src, dst), &(first, second))) in edges.iter().zip(&placed).enumerate() {
            for m in std::iter::once(first).chain(second) {
                let at = &mut next[m * stride + src as usize];
                nbrs[*at as usize] = dst as u64;
                eids[*at as usize] = eid as u64;
                *at += 1;
            }
        }
        Graph {
            layers,
            runs,
            overlap,
            stride: stride as u64,
            offsets,
            nbrs,
            eids,
        }
    }

    fn name(&self) -> String {
        let overlap = if self.overlap { "_overlap" } else { "" };
        format!("L{}_R{}{overlap}", self.layers, self.runs)
    }

    fn row(&self, m: usize, n: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
        let at = m * self.stride as usize + n as usize;
        let range = self.offsets[at] as usize..self.offsets[at + 1] as usize;
        self.nbrs[range.clone()]
            .iter()
            .copied()
            .zip(self.eids[range].iter().copied())
    }

    fn row_len(&self, m: usize, n: u64) -> u64 {
        let at = m * self.stride as usize + n as usize;
        self.offsets[at + 1] - self.offsets[at]
    }
}

/// The nodes each scan visits, ascending.
fn scans() -> Vec<(&'static str, Vec<u64>)> {
    let (edges, nodes) = edges();
    let mut degree = vec![0usize; *nodes];
    for &(src, _) in edges {
        degree[src as usize] += 1;
    }
    let mut by_degree: Vec<u64> = (0..*nodes as u64).collect();
    by_degree.sort_by_key(|&n| std::cmp::Reverse(degree[n as usize]));
    let mut hubs = by_degree[..nodes * HUBS / 100].to_vec();
    hubs.sort_unstable();
    vec![("all", (0..*nodes as u64).collect()), ("hubs", hubs)]
}

/// (edges popped, non-empty rows per node, share of popped edges that are repeats).
fn describe(graph: &Graph, scan: &[u64]) -> (u64, f64, f64) {
    let (mut popped, mut rows, mut kept) = (0u64, 0u64, 0u64);
    for &n in scan {
        for m in 0..graph.layers * graph.runs {
            let len = graph.row_len(m, n);
            popped += len;
            rows += (len > 0) as u64;
        }
        kept += rust_flat(graph, n).count() as u64;
    }
    let repeats = 1.0 - kept as f64 / popped.max(1) as f64;
    (popped, rows as f64 / scan.len() as f64, repeats)
}

// =============================================================================
// Rust
// =============================================================================

fn rust_flat(graph: &Graph, n: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
    (0..graph.layers * graph.runs)
        .map(move |m| graph.row(m, n))
        .kmerge_by(|a, b| a.0 < b.0)
        .dedup_by(|a, b| a.0 == b.0)
}

fn rust_nested(graph: &Graph, n: u64) -> impl Iterator<Item = (u64, u64)> + '_ {
    (0..graph.layers)
        .map(move |l| {
            (0..graph.runs)
                .map(move |r| graph.row(l * graph.runs + r, n))
                .kmerge_by(|a, b| a.0 < b.0)
        })
        .kmerge_by(|a, b| a.0 < b.0)
        .dedup_by(|a, b| a.0 == b.0)
}

fn rust_scan<'g, I: Iterator<Item = (u64, u64)>>(
    graph: &'g Graph,
    scan: &[u64],
    merge: impl Fn(&'g Graph, u64) -> I,
) -> u64 {
    scan.iter().fold(0, |acc, &n| {
        merge(graph, n).fold(acc, |acc, (nbr, eid)| checksum(acc, nbr, eid))
    })
}

// =============================================================================
// Staged
// =============================================================================

/// The graph's arrays and the scan, as a kernel sees them.
#[derive(Clone, Copy)]
struct Args {
    nbrs: Var<S>,
    eids: Var<S>,
    offsets: Var<S>,
    stride: Var<u64>,
    runs: Var<u64>,
}

impl Args {
    /// Push node `n`'s row of matrix `m` onto `heap`.
    fn push(&self, ctx: &Ctx, heap: &Heap<u64, u64>, m: Var<u64>, n: Var<u64>) {
        let at = ctx.bind(m * self.stride + n);
        // SAFETY: `m` is a matrix and `n` a node of the graph, so `at + 1` is within
        // `offsets`, whose entries bound rows within `nbrs` and `eids`.
        unsafe {
            let start = ctx.bind(self.offsets.get_unchecked(at));
            let end = ctx.bind(self.offsets.get_unchecked(at + 1u64));
            heap.push(
                ctx,
                ctx.bind(self.nbrs.subslice_unchecked(start, end)),
                ctx.bind(self.eids.subslice_unchecked(start, end)),
            );
        }
    }

    /// Node `n`'s rows over matrices `first..first + count`, merged by a heap of
    /// `capacity`.
    fn merge(
        self,
        capacity: usize,
        first: Var<u64>,
        count: Var<u64>,
        n: Var<u64>,
    ) -> KMergeRows<u64, u64, impl FnOnce(&Ctx, &Heap<u64, u64>)> {
        // SAFETY: the heap gets `count <= capacity` rows, each sorted, with as many
        // edge ids as neighbours, in the graph's arrays, which outlive the kernel.
        unsafe {
            kmerge_rows(capacity, move |ctx, heap| {
                range(0u64, count).for_each(ctx, |ctx, j| {
                    self.push(ctx, heap, ctx.bind(first + j), n);
                })
            })
        }
    }
}

type Kernel = Box<dyn Fn(&Graph, &[u64]) -> u64>;

fn kernel(backend: JitBackend, nested: bool, layers: usize, runs: usize) -> Kernel {
    let mut compiler = Compiler::new().with_backend(backend);
    let f = compiler.fun6(
        "graph_kmerge",
        move |ctx,
              nbrs: Var<S>,
              eids: Var<S>,
              offsets: Var<S>,
              scan: Var<S>,
              stride: Var<u64>,
              runs_now: Var<u64>| {
            let args = Args {
                nbrs,
                eids,
                offsets,
                stride,
                runs: runs_now,
            };
            let acc = ctx.var(0u64);
            let fold = move |ctx: &Ctx, x: Var<ZipItem<u64, u64>>| {
                ctx.store(acc, acc * 31u64 + (x.first() * 3u64 + x.second()));
            };
            range(0u64, scan.len()).for_each(ctx, |ctx, i| {
                // SAFETY: `i` is below the scan's length.
                let n = ctx.bind(unsafe { scan.get_unchecked(i) });
                let key = |_: &Ctx, x: Var<ZipItem<u64, u64>>| x.first();
                if nested {
                    let per_layer: Vec<_> = (0..layers as u64)
                        .map(|l| {
                            let first = ctx.bind(args.runs * l);
                            args.merge(runs, first, args.runs, n)
                        })
                        .collect();
                    kmerge_by(per_layer, key)
                        .dedup_by_key(key)
                        .for_each(ctx, fold);
                } else {
                    let zero = ctx.var(0u64);
                    let count = ctx.bind(args.runs * layers as u64);
                    args.merge(layers * runs, zero, count, n)
                        .dedup_by_key(key)
                        .for_each(ctx, fold);
                }
            });
            acc
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |g, scan| {
        compiled.as_fn().call(
            &g.nbrs[..],
            &g.eids[..],
            &g.offsets[..],
            scan,
            g.stride,
            g.runs as u64,
        )
    })
}

// =============================================================================
// Benches
// =============================================================================

fn bench_graph_kmerge(c: &mut Criterion) {
    let scans = scans();
    for &layers in LAYERS {
        for &runs in RUNS {
            for overlap in [false, true] {
                if overlap && layers < 2 {
                    continue;
                }
                let graph = Graph::new(layers, runs, overlap);
                for (scan_name, scan) in &scans {
                    let (popped, rows, repeats) = describe(&graph, scan);
                    eprintln!(
                        "graph_kmerge {scan_name}/{}: {} nodes, {popped} edges popped, \
                         {rows:.2} rows/node, {:.1}% repeats",
                        graph.name(),
                        scan.len(),
                        repeats * 100.0
                    );
                    let mut group =
                        c.benchmark_group(format!("graph_kmerge/{scan_name}/{}", graph.name()));
                    group.warm_up_time(Duration::from_millis(500));
                    group.measurement_time(Duration::from_secs(3));
                    group.sample_size(10);
                    group.throughput(Throughput::Elements(popped));

                    let expected = rust_scan(&graph, scan, rust_flat);
                    let nested_ok = layers >= 2;
                    group.bench_function(BenchmarkId::new("rust", "flat"), |b| {
                        b.iter(|| rust_scan(black_box(&graph), black_box(scan), rust_flat))
                    });
                    if nested_ok {
                        assert_eq!(
                            rust_scan(&graph, scan, rust_nested),
                            expected,
                            "rust/nested"
                        );
                        group.bench_function(BenchmarkId::new("rust", "nested"), |b| {
                            b.iter(|| rust_scan(black_box(&graph), black_box(scan), rust_nested))
                        });
                    }
                    for (backend_name, backend) in backends() {
                        for nested in [false, true] {
                            if nested && !nested_ok {
                                continue;
                            }
                            let name = if nested { "nested" } else { "flat" };
                            let f = kernel(backend, nested, layers, runs);
                            assert_eq!(f(&graph, scan), expected, "{backend_name}/{name}");
                            group.bench_function(BenchmarkId::new(backend_name, name), |b| {
                                b.iter(|| f(black_box(&graph), black_box(scan)))
                            });
                        }
                    }
                    group.finish();
                }
            }
        }
    }
}

criterion_group!(benches, bench_graph_kmerge);
criterion_main!(benches);
