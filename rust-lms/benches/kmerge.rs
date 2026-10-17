//! k-way merge of sorted parts (`relational/jit_stage3.md` §7.1).
//!
//! Every variant folds the merged stream into an order-sensitive checksum
//! (`acc * 31 + x`), checked against a host merge before it is timed.
//!
//! - `nested`: a balanced tree of two-way `merge_by`s, the baseline.
//! - `linear`, `tree`: `kmerge_by` with each plan (a pass over all keys, a loser tree),
//!   with the runner-up shortcut; `_noruns` without it.
//! - `host_itertools`, `host_loser_tree`: the same merge in Rust, for reference.
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
use rust_lms::iter::Plan;
use rust_lms::prelude::*;

const N: usize = 1 << 20;
const KS: &[usize] = &[2, 4, 8, 16, 32];
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
    it.for_each(ctx, move |ctx, x| {
        let x = value(ctx, x);
        ctx.store(acc, acc * 31u64 + x);
    });
    acc
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
    let p = &mut parts.into_iter();
    match p.len() {
        2 => fold(ctx, m2(p, le), value),
        4 => fold(ctx, m4(p, le), value),
        8 => fold(ctx, m8(p, le), value),
        16 => fold(ctx, m16(p, le), value),
        32 => fold(ctx, m32(p, le), value),
        k => unreachable!("{k} parts"),
    }
}

/// `merge` set up for a variant named `linear`, `tree`, `linear_noruns` or `tree_noruns`.
fn configure<I, F>(merge: KMergeBy<I, F>, variant: &str) -> KMergeBy<I, F> {
    let (plan, runs) = match variant {
        "linear" => (Plan::Linear, true),
        "tree" => (Plan::Tree, true),
        "linear_noruns" => (Plan::Linear, false),
        "tree_noruns" => (Plan::Tree, false),
        v => unreachable!("{v}"),
    };
    let merge = merge.with_plan(plan);
    if runs { merge } else { merge.without_runs() }
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

/// A loser tree over `parts`: `tree[0]` is the winner, `tree[1..k]` the losers, leaves at
/// `k..2k` in heap order. Ties go to the lower part.
fn host_loser_tree<K: Copy + Ord>(parts: &[&[K]], max: K, value: impl Fn(usize, K) -> u64) -> u64 {
    let k = parts.len();
    let mut pos = vec![0usize; k];
    let key = |j: usize, pos: &[usize]| (parts[j].get(pos[j]).copied().unwrap_or(max), j);
    let mut tree = vec![0usize; k];
    let mut win = vec![0usize; 2 * k];
    for j in 0..k {
        win[k + j] = j;
    }
    for i in (1..k).rev() {
        let (a, b) = (win[2 * i], win[2 * i + 1]);
        let (w, l) = if key(a, &pos) <= key(b, &pos) {
            (a, b)
        } else {
            (b, a)
        };
        win[i] = w;
        tree[i] = l;
    }
    tree[0] = win[1];
    let mut acc = 0;
    loop {
        let mut w = tree[0];
        let (x, _) = key(w, &pos);
        if x == max {
            return acc;
        }
        acc = checksum(acc, value(w, x));
        pos[w] += 1;
        let mut node = (w + k) / 2;
        while node >= 1 {
            let o = tree[node];
            if key(o, &pos) < key(w, &pos) {
                tree[node] = w;
                w = o;
            }
            node /= 2;
        }
        tree[0] = w;
    }
}

// =============================================================================
// Benches
// =============================================================================

const VARIANTS: &[&str] = &["nested", "linear", "tree", "linear_noruns", "tree_noruns"];

fn group<'a>(
    c: &'a mut Criterion,
    name: &str,
) -> criterion::BenchmarkGroup<'a, criterion::measurement::WallTime> {
    let mut group = c.benchmark_group(name);
    group.measurement_time(Duration::from_secs(2));
    group.sample_size(20);
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
            assert_eq!(host_loser_tree(&slices, u64::MAX, |_, x| x), expected);
            group.bench_with_input(BenchmarkId::new("host_itertools", k), &k, |b, _| {
                b.iter(|| host_itertools(black_box(&slices), |_, x| x))
            });
            group.bench_with_input(BenchmarkId::new("host_loser_tree", k), &k, |b, _| {
                b.iter(|| host_loser_tree(black_box(&slices), u64::MAX, |_, x| x))
            });
            for (backend_name, backend) in backends() {
                for &variant in VARIANTS {
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
            assert_eq!(host_loser_tree(&slices, (i64::MAX, u64::MAX), id), expected);
            group.bench_with_input(BenchmarkId::new("host_itertools", k), &k, |b, _| {
                b.iter(|| host_itertools(black_box(&slices), id))
            });
            group.bench_with_input(BenchmarkId::new("host_loser_tree", k), &k, |b, _| {
                b.iter(|| host_loser_tree(black_box(&slices), (i64::MAX, u64::MAX), id))
            });
            for (backend_name, backend) in backends() {
                for &variant in VARIANTS {
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

criterion_group!(benches, bench_u64, bench_events);
criterion_main!(benches);
