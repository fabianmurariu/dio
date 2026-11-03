//! k-way merge of sorted parts (`relational/jit_stage3.md` §7.1).
//!
//! Every variant folds the merged stream into an order-sensitive checksum
//! (`acc * 31 + x`), checked against a host merge before it is timed.
//!
//! - `nested`: a balanced tree of two-way `merge_by`s, the baseline.
//! - `chain`, `linear`: `kmerge_by` with each plan (a chain of compare branches, a pass
//!   over all keys).
//! - `host_itertools`: the same merge in Rust, for reference.
//!
//! `kmerge_nodes_*`: the shape a query compiles to, `for n in nodes: for x in
//! merge(runs of n)`, so each merge is opened, primed and closed in a tight loop over
//! small parts. `ffi_itertools` gets each node's merged iterator from itertools (`merge` for
//! two runs, `kmerge` above) through an extern call, and pulls and folds it in the kernel.
//! Events are `EventTime`-shaped structs `{t, i}`, merged by the `i128` packing of `(t, i)`.
//!
//! Layouts: `random` (each element in a random part), `disjoint` (each part a contiguous
//! range of keys), `clustered` (runs of 8 to 64 consecutive keys, each in a random part).
//! Keys: `u64`, and events `(t: i64, i: u64)`, which the JIT merges packed to `i128`
//! (`kmerge_by`) or compared as a tuple (`nested`).

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use itertools::Itertools;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::hint::black_box;
use std::time::Duration;

use rust_lms::func::JitBackend;
use rust_lms::iter::{CHAIN_MAX, Plan};
use rust_lms::prelude::*;

const N: usize = 1 << 20;
const KS: &[usize] = &[2, 4, 8];
const LAYOUTS: &[&str] = &["random", "disjoint", "clustered"];

type S = SRef<Slice<u64>>;
type T = SRef<Slice<i64>>;

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
// Data
// =============================================================================

/// The part of each of `N` ascending keys.
fn assignment(layout: &str, k: usize, rng: &mut StdRng) -> Vec<usize> {
    match layout {
        "random" => (0..N).map(|_| rng.random_range(0..k)).collect(),
        "disjoint" => (0..N).map(|i| k - 1 - i * k / N).collect(),
        "clustered" => {
            let mut parts = Vec::with_capacity(N);
            while parts.len() < N {
                let (part, len) = (rng.random_range(0..k), rng.random_range(8..=64));
                parts.extend(std::iter::repeat_n(part, len.min(N - parts.len())));
            }
            parts
        }
        _ => unreachable!(),
    }
}

/// `k` parts of ascending keys, laid out end to end, with `k + 1` offsets.
struct Parts<K> {
    values: Vec<K>,
    offsets: Vec<u64>,
}

impl<K: Copy> Parts<K> {
    fn new(keys: impl Iterator<Item = K>, assignment: &[usize], k: usize) -> Self {
        let mut parts = vec![Vec::new(); k];
        for (key, &j) in keys.zip(assignment) {
            parts[j].push(key);
        }
        let mut offsets = vec![0u64];
        for part in &parts {
            offsets.push(offsets.last().unwrap() + part.len() as u64);
        }
        Parts {
            values: parts.concat(),
            offsets,
        }
    }

    fn slices(&self) -> Vec<&[K]> {
        self.offsets
            .windows(2)
            .map(|w| &self.values[w[0] as usize..w[1] as usize])
            .collect()
    }
}

/// Keys with gaps and ties: `i / 2 * 3`.
fn u64_parts(layout: &str, k: usize) -> Parts<u64> {
    let mut rng = StdRng::seed_from_u64(k as u64);
    let assignment = assignment(layout, k, &mut rng);
    Parts::new((0..N as u64).map(|i| i / 2 * 3), &assignment, k)
}

/// Events: times with ties around zero, ids unique and ascending within a time.
fn event_parts(layout: &str, k: usize) -> (Parts<i64>, Parts<u64>) {
    let mut rng = StdRng::seed_from_u64(k as u64);
    let assignment = assignment(layout, k, &mut rng);
    let times = (0..N as i64).map(|i| i / 4 - (N as i64) / 8);
    (
        Parts::new(times, &assignment, k),
        Parts::new(0..N as u64, &assignment, k),
    )
}

// =============================================================================
// JIT kernels
// =============================================================================

/// The `k` parts of `values` as slices.
fn part_slices<E: StagedType + 'static>(
    ctx: &Ctx,
    values: Var<SRef<Slice<E>>>,
    offsets: Var<S>,
    k: usize,
) -> Vec<Var<SRef<Slice<E>>>> {
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

fn fold<I: StagedIterator>(
    ctx: &Ctx,
    it: I,
    value: impl Fn(&Ctx, Var<I::Item>) -> Var<u64>,
) -> Var<u64> {
    let acc = ctx.var(0u64);
    fold_into(ctx, acc, it, value);
    acc
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
type M16<I, P> = MergeBy<M8<I, P>, M8<I, P>, P>;
type M32<I, P> = MergeBy<M16<I, P>, M16<I, P>, P>;

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
tree!(m16, M16, m8);
tree!(m32, M32, m16);

/// `fold` over a balanced tree of `merge_by`s of `parts`.
fn nested<J, P, C>(
    ctx: &Ctx,
    parts: Vec<J>,
    le: P,
    value: impl Fn(&Ctx, Var<Item<J>>) -> Var<u64>,
) -> Var<u64>
where
    J: IntoStagedIterator,
    Item<J>: CopyType + 'static,
    P: Fn(Var<Item<J>>, Var<Item<J>>) -> C + Copy,
    C: Staged<Out = bool> + 'static,
{
    let acc = ctx.var(0u64);
    nested_into(ctx, acc, parts, le, value);
    acc
}

fn nested_into<J, P, C>(
    ctx: &Ctx,
    acc: Var<u64>,
    parts: Vec<J>,
    le: P,
    value: impl Fn(&Ctx, Var<Item<J>>) -> Var<u64>,
) where
    J: IntoStagedIterator,
    Item<J>: CopyType + 'static,
    P: Fn(Var<Item<J>>, Var<Item<J>>) -> C + Copy,
    C: Staged<Out = bool> + 'static,
{
    let p = &mut parts.into_iter();
    match p.len() {
        2 => fold_into(ctx, acc, m2(p, le), value),
        3 => {
            let m = m2(p, le).merge_by(p.next().unwrap(), le);
            fold_into(ctx, acc, m, value)
        }
        4 => fold_into(ctx, acc, m4(p, le), value),
        8 => fold_into(ctx, acc, m8(p, le), value),
        16 => fold_into(ctx, acc, m16(p, le), value),
        32 => fold_into(ctx, acc, m32(p, le), value),
        k => unreachable!("{k} parts"),
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

type U64Fn = Box<dyn Fn(&Parts<u64>) -> u64>;
type EventFn = Box<dyn Fn(&Parts<i64>, &Parts<u64>) -> u64>;

fn u64_kernel(backend: JitBackend, variant: &str, k: usize) -> U64Fn {
    let mut compiler = Compiler::new().with_backend(backend);
    let variant = variant.to_string();
    let f = compiler.fun2("kmerge_u64", move |ctx, values: Var<S>, offsets: Var<S>| {
        let parts = part_slices(ctx, values, offsets, k);
        match variant.as_str() {
            "nested" => nested(ctx, parts, le, |_, x| x),
            v => fold(ctx, configure(kmerge_by(parts, |_, x| x), v), |_, x| x),
        }
    });
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |p| compiled.as_fn().call(&p.values[..], &p.offsets[..]))
}

fn event_kernel(backend: JitBackend, variant: &str, k: usize) -> EventFn {
    let mut compiler = Compiler::new().with_backend(backend);
    let variant = variant.to_string();
    let f = compiler.fun3(
        "kmerge_events",
        move |ctx, times: Var<T>, ids: Var<S>, offsets: Var<S>| {
            let parts: Vec<_> = part_slices(ctx, times, offsets, k)
                .into_iter()
                .zip(part_slices(ctx, ids, offsets, k))
                .map(|(t, i)| t.staged_iter().zip(i))
                .collect();
            let value = |ctx: &Ctx, e: Var<ZipItem<i64, u64>>| ctx.bind(e.second());
            match variant.as_str() {
                "nested" => nested(
                    ctx,
                    parts,
                    |x, y| {
                        // TODO: we need Ordered for ZipItem
                        lt(x.first(), y.first())
                            | (eq(x.first(), y.first()) & le(x.second(), y.second()))
                    },
                    value,
                ),
                v => fold(
                    ctx,
                    configure(kmerge_by(parts, |_, e| pack_i128(e.first(), e.second())), v),
                    value,
                ),
            }
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |t, i| {
        compiled
            .as_fn()
            .call(&t.values[..], &i.values[..], &t.offsets[..])
    })
}

// =============================================================================
// Host references
// =============================================================================

fn host_itertools<K: Copy + Ord>(parts: &[&[K]], value: impl Fn(usize, K) -> u64) -> u64 {
    parts
        .iter()
        .enumerate()
        .map(|(j, p)| p.iter().map(move |&x| (x, j)))
        .kmerge_by(|a, b| a < b)
        .fold(0, |acc, (x, j)| checksum(acc, value(j, x)))
}

// =============================================================================
// Benches
// =============================================================================

const VARIANTS: &[&str] = &["nested", "chain", "linear"];

fn group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.warm_up_time(Duration::from_millis(1200));
    group.measurement_time(Duration::from_millis(10000));
    group.sample_size(50);
    group.throughput(Throughput::Elements(N as u64));
    group
}

fn bench_u64(c: &mut Criterion) {
    for &layout in LAYOUTS {
        let mut group = group(c, &format!("kmerge_u64/{layout}"));
        for &k in KS {
            let parts = u64_parts(layout, k);
            let slices = parts.slices();
            let expected = host_itertools(&slices, |_, x| x);
            group.bench_with_input(BenchmarkId::new("host_itertools", k), &k, |b, _| {
                b.iter(|| host_itertools(black_box(&slices), |_, x| x))
            });
            for (backend_name, backend) in backends() {
                for &variant in VARIANTS {
                    if variant == "chain" && k > CHAIN_MAX {
                        continue;
                    }
                    let f = u64_kernel(backend, variant, k);
                    assert_eq!(f(&parts), expected, "{layout}/{backend_name}/{variant}/{k}");
                    group.bench_with_input(
                        BenchmarkId::new(format!("{backend_name}/{variant}"), k),
                        &k,
                        |b, _| b.iter(|| f(black_box(&parts))),
                    );
                }
            }
        }
        group.finish();
    }
}

fn bench_events(c: &mut Criterion) {
    for &layout in LAYOUTS {
        let mut group = group(c, &format!("kmerge_events/{layout}"));
        for &k in KS {
            let (times, ids) = event_parts(layout, k);
            let events: Vec<Vec<(i64, u64)>> = times
                .slices()
                .into_iter()
                .zip(ids.slices())
                .map(|(t, i)| t.iter().copied().zip(i.iter().copied()).collect())
                .collect();
            let slices: Vec<&[(i64, u64)]> = events.iter().map(|p| &p[..]).collect();
            let id = |_, (_, i): (i64, u64)| i;
            let expected = host_itertools(&slices, id);
            group.bench_with_input(BenchmarkId::new("host_itertools", k), &k, |b, _| {
                b.iter(|| host_itertools(black_box(&slices), id))
            });
            for (backend_name, backend) in backends() {
                for &variant in VARIANTS {
                    if variant == "chain" && k > CHAIN_MAX {
                        continue;
                    }
                    let f = event_kernel(backend, variant, k);
                    assert_eq!(
                        f(&times, &ids),
                        expected,
                        "{layout}/{backend_name}/{variant}/{k}"
                    );
                    group.bench_with_input(
                        BenchmarkId::new(format!("{backend_name}/{variant}"), k),
                        &k,
                        |b, _| b.iter(|| f(black_box(&times), black_box(&ids))),
                    );
                }
            }
        }
        group.finish();
    }
}

// =============================================================================
// Per node: many small merges in one kernel
// =============================================================================

/// An event as `EventTime` lays it out.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, StagedType)]
#[repr(C)]
pub struct Ev {
    pub t: i64,
    pub i: u64,
}

const NODE_KS: &[usize] = &[2, 3, 4, 8];
const DEGREES: &[usize] = &[4, 32];
const NODE_LAYOUTS: &[&str] = &["random", "disjoint"];
const NODE_VARIANTS: &[&str] = &["ffi_itertools", "nested", "chain", "linear"];

/// `N` elements over nodes of degree uniform in `0..=2 * degree`, each node's elements
/// split over `k` runs: node `n`'s run `j` is `values[offsets[n * k + j]..offsets[n * k + j + 1]]`.
struct Nodes<K> {
    values: Vec<K>,
    offsets: Vec<u64>,
    nodes: u64,
}

impl<K: Copy + Ord> Nodes<K> {
    /// `key(i)` is ascending in `i` within a node.
    fn new(layout: &str, k: usize, degree: usize, key: impl Fn(usize) -> K) -> Self {
        let mut rng = StdRng::seed_from_u64((k * 1000 + degree) as u64);
        let (mut values, mut offsets, mut nodes) = (Vec::with_capacity(N), vec![0u64], 0);
        while values.len() < N {
            let d = rng.random_range(0..=2 * degree).min(N - values.len());
            let parts: Vec<usize> = match layout {
                "random" => (0..d).map(|_| rng.random_range(0..k)).collect(),
                "disjoint" => (0..d).map(|i| i * k / d.max(1)).collect(),
                _ => unreachable!(),
            };
            for j in 0..k {
                let start = values.len();
                values.extend((0..d).filter(|&i| parts[i] == j).map(|i| key(start + i)));
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
}

/// One node's runs merged by itertools: `merge` for two runs, `kmerge` above; `offsets`
/// are the node's `k + 1` offsets into `values`.
fn itertools_node<'a, K: Copy + Ord + 'a>(
    values: &'a [K],
    offsets: &'a [u64],
) -> Box<dyn Iterator<Item = K> + 'a> {
    let part = move |j: usize| {
        values[offsets[j] as usize..offsets[j + 1] as usize]
            .iter()
            .copied()
    };
    match offsets.len() - 1 {
        2 => Box::new(part(0).merge(part(1))),
        k => Box::new((0..k).map(part).kmerge()),
    }
}

fn host_nodes<K: Copy + Ord>(nodes: &Nodes<K>, k: usize, value: impl Fn(K) -> u64 + Copy) -> u64 {
    (0..nodes.nodes as usize).fold(0, |acc, n| {
        let offsets = &nodes.offsets[n * k..=n * k + k];
        itertools_node(&nodes.values, offsets).fold(acc, |acc, x| checksum(acc, value(x)))
    })
}

/// [`itertools_node`] as a handle, which the kernel drives as a `DynIter`.
#[allow(improper_ctypes_definitions)]
#[rust_lms_derive::extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn itertools_node_u64(values: &[u64], offsets: &[u64]) -> *mut () {
    // SAFETY: the kernel hands the handle to `DynIter<u64>`, which drops it once.
    unsafe { box_dyn_iter(itertools_node(values, offsets)).into_raw() }
}

#[allow(improper_ctypes_definitions)]
#[rust_lms_derive::extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn itertools_node_events(values: &[Ev], offsets: &[u64]) -> *mut () {
    // SAFETY: the kernel hands the handle to `DynIter<Ev>`, which drops it once.
    unsafe { box_dyn_iter(itertools_node(values, offsets)).into_raw() }
}

/// `for n in nodes: for x in merge(runs of n): acc = acc * 31 + value(x)`.
fn node_loop<E: CopyType + 'static>(
    ctx: &Ctx,
    values: Var<SRef<Slice<E>>>,
    offsets: Var<S>,
    nodes: Var<u64>,
    k: usize,
    mut node: impl FnMut(&Ctx, Var<u64>, Vec<Var<SRef<Slice<E>>>>, Var<S>),
) -> Var<u64> {
    let acc = ctx.var(0u64);
    range(0u64, nodes).for_each(ctx, |ctx, n| {
        let base = ctx.bind(n * k as u64);
        // SAFETY: the host passes `nodes * k + 1` ascending offsets within `values`.
        let (offsets, parts) = unsafe {
            let end = ctx.bind(base + (k as u64 + 1));
            let offsets = ctx.bind(offsets.subslice_unchecked(base, end));
            (offsets, part_slices(ctx, values, offsets, k))
        };
        node(ctx, acc, parts, offsets);
    });
    acc
}

type NodesFn<K> = Box<dyn Fn(&Nodes<K>) -> u64>;

fn nodes_u64_kernel(backend: JitBackend, variant: &str, k: usize) -> NodesFn<u64> {
    let mut compiler = Compiler::new().with_backend(backend);
    let itertools = compiler.extern_fn::<ItertoolsNodeU64Extern>();
    let merged = compiler.opaque_iter_fns::<DynIter<u64>>();
    let variant = variant.to_string();
    let f = compiler.fun3(
        "nodes_u64",
        move |ctx, values: Var<S>, offsets: Var<S>, nodes: Var<u64>| {
            node_loop(
                ctx,
                values,
                offsets,
                nodes,
                k,
                |ctx, acc, parts, offsets| {
                    match variant.as_str() {
                        "ffi_itertools" => {
                            // SAFETY: both slices are valid for the call, which returns
                            // a fresh `DynIter` handle.
                            let merged = unsafe {
                                merged.iter(call_extern2_unchecked(itertools, values, offsets))
                            };
                            fold_into(ctx, acc, merged, |_, x| x)
                        }
                        "nested" => nested_into(ctx, acc, parts, le, |_, x| x),
                        v => {
                            fold_into(ctx, acc, configure(kmerge_by(parts, |_, x| x), v), |_, x| x)
                        }
                    }
                },
            )
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |p| {
        compiled
            .as_fn()
            .call(&p.values[..], &p.offsets[..], p.nodes)
    })
}

fn nodes_events_kernel(backend: JitBackend, variant: &str, k: usize) -> NodesFn<Ev> {
    let mut compiler = Compiler::new().with_backend(backend);
    let itertools = compiler.extern_fn::<ItertoolsNodeEventsExtern>();
    let merged = compiler.opaque_iter_fns::<DynIter<Ev>>();
    let variant = variant.to_string();
    let f = compiler.fun3(
        "nodes_events",
        move |ctx, values: Var<SRef<Slice<Ev>>>, offsets: Var<S>, nodes: Var<u64>| {
            node_loop(
                ctx,
                values,
                offsets,
                nodes,
                k,
                |ctx, acc, parts, offsets| {
                    let key = |e: Var<Ev>| pack_i128(e.get(EvType::t()), e.get(EvType::i()));
                    let value = |ctx: &Ctx, e: Var<Ev>| ctx.bind(e.get(EvType::i()));
                    match variant.as_str() {
                        "ffi_itertools" => {
                            // SAFETY: both slices are valid for the call, which returns
                            // a fresh `DynIter` handle.
                            let merged = unsafe {
                                merged.iter(call_extern2_unchecked(itertools, values, offsets))
                            };
                            fold_into(ctx, acc, merged, value)
                        }
                        "nested" => {
                            nested_into(ctx, acc, parts, move |x, y| le(key(x), key(y)), value)
                        }
                        v => fold_into(
                            ctx,
                            acc,
                            configure(kmerge_by(parts, move |_, e| key(e)), v),
                            value,
                        ),
                    }
                },
            )
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    Box::new(move |p| {
        compiled
            .as_fn()
            .call(&p.values[..], &p.offsets[..], p.nodes)
    })
}

fn bench_nodes_of<K: Copy + Ord>(
    c: &mut Criterion,
    name: &str,
    key: impl Fn(usize) -> K + Copy,
    value: impl Fn(K) -> u64 + Copy,
    kernel: impl Fn(JitBackend, &str, usize) -> NodesFn<K>,
) {
    for &layout in NODE_LAYOUTS {
        for &degree in DEGREES {
            let mut group = group(c, &format!("{name}/{layout}/d{degree}"));
            for &k in NODE_KS {
                let nodes = Nodes::new(layout, k, degree, key);
                let expected = host_nodes(&nodes, k, value);
                group.bench_with_input(BenchmarkId::new("host_itertools", k), &k, |b, _| {
                    b.iter(|| host_nodes(black_box(&nodes), k, value))
                });
                for (backend_name, backend) in backends() {
                    for &variant in NODE_VARIANTS {
                        if variant == "nested" && !matches!(k, 2 | 3 | 4 | 8) {
                            continue;
                        }
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
    bench_nodes_of(
        c,
        "kmerge_nodes_u64",
        |i| (i / 2 * 3) as u64,
        |x| x,
        nodes_u64_kernel,
    );
    bench_nodes_of(
        c,
        "kmerge_nodes_events",
        |i| Ev {
            t: (i / 4) as i64 - (N as i64) / 8,
            i: i as u64,
        },
        |e| e.i,
        nodes_events_kernel,
    );
}

criterion_group!(benches, bench_u64, bench_events, bench_nodes);
criterion_main!(benches);
