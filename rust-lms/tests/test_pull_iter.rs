//! The pull protocol (docs/pull_iter.md): `zip` and `chain` over any iterators —
//! extern streams included — plus `take`/`skip`, and resource safety: every
//! extern handle opened by a pipeline is dropped exactly once, on every exit.

use std::cell::Cell;

use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

mod common;
use common::for_each_backend;

// =============================================================================
// An extern "graph" whose iterators count how many are alive
// =============================================================================

pub struct Graph {
    nodes: Vec<u64>,
    other: Vec<u64>,
    opened: Cell<u64>,
    live: Cell<i64>,
}

fn graph(nodes: &[u64], other: &[u64]) -> Graph {
    Graph {
        nodes: nodes.to_vec(),
        other: other.to_vec(),
        opened: Cell::new(0),
        live: Cell::new(0),
    }
}

/// A node iterator that reports its own drop back to the graph.
struct Counted<'g> {
    inner: std::iter::Copied<std::slice::Iter<'g, u64>>,
    live: &'g Cell<i64>,
}

impl Iterator for Counted<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        self.inner.next()
    }
}

impl Drop for Counted<'_> {
    fn drop(&mut self) {
        self.live.set(self.live.get() - 1);
    }
}

fn open_counted<'g>(g: &'g Graph, items: &'g [u64]) -> *mut () {
    g.opened.set(g.opened.get() + 1);
    g.live.set(g.live.get() + 1);
    let it = Counted {
        inner: items.iter().copied(),
        live: &g.live,
    };
    // SAFETY: the generated traversal transfers this handle to the matching
    // `DynIter<u64>` next/drop exactly once, within the kernel call that `g`
    // outlives.
    unsafe { box_dyn_iter(it).into_raw() }
}

#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn pull_graph_nodes(g: &Graph) -> *mut () {
    open_counted(g, &g.nodes)
}

#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn pull_graph_other(g: &Graph) -> *mut () {
    open_counted(g, &g.other)
}

type G = SRef<Opaque<Graph>>;
type S = SRef<Slice<u64>>;

/// Every opened handle was dropped (and none twice).
fn assert_all_dropped(g: &Graph, opened: u64) {
    assert_eq!(g.opened.get(), opened, "handles opened");
    assert_eq!(g.live.get(), 0, "handles still alive after the kernel");
}

// =============================================================================
// zip
// =============================================================================

#[test]
fn zip_extern_with_slice() {
    for_each_backend(|mut compiler| {
        let nodes = compiler.extern_fn::<PullGraphNodesExtern>();
        let it = compiler.opaque_iter_fns::<DynIter<u64>>();
        let f = compiler.fun2("dot", move |ctx, g: Var<G>, w: Var<S>| {
            // SAFETY: a fresh `DynIter<u64>` handle from the producer.
            unsafe { it.iter(call_extern1(nodes, g)) }
                .zip(w)
                .map(|p| p.first() * p.second())
                .sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();

        let g = graph(&[1, 2, 3], &[]);
        assert_eq!(f.call(&g, &[10u64, 20, 30][..]), 140);
        assert_all_dropped(&g, 1);

        // Stops at the shorter side, whichever it is.
        let g = graph(&[1, 2, 3], &[]);
        assert_eq!(f.call(&g, &[10u64][..]), 10);
        assert_all_dropped(&g, 1);
        let g = graph(&[1], &[]);
        assert_eq!(f.call(&g, &[10u64, 20, 30][..]), 10);
        assert_all_dropped(&g, 1);
    });
}

#[test]
fn zip_slice_with_extern_three_arg_consumer() {
    for_each_backend(|mut compiler| {
        let nodes = compiler.extern_fn::<PullGraphNodesExtern>();
        let it = compiler.opaque_iter_fns::<DynIter<u64>>();
        let f = compiler.fun2("weighted", move |ctx, g: Var<G>, w: Var<S>| {
            let acc = ctx.var(0u64);
            // SAFETY: a fresh `DynIter<u64>` handle from the producer.
            let nodes = unsafe { it.iter(call_extern1(nodes, g)) };
            w.staged_iter().zip(nodes).for_each(ctx, move |ctx, wi, n| {
                ctx.store(acc, acc * 100u64 + wi * n);
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        let g = graph(&[1, 2, 3], &[]);
        assert_eq!(f.call(&g, &[5u64, 6][..]), 512);
        assert_all_dropped(&g, 1);
    });
}

#[test]
fn zip_extern_with_extern() {
    for_each_backend(|mut compiler| {
        let nodes = compiler.extern_fn::<PullGraphNodesExtern>();
        let other = compiler.extern_fn::<PullGraphOtherExtern>();
        let it = compiler.opaque_iter_fns::<DynIter<u64>>();
        let f = compiler.fun1("pairs", move |ctx, g: Var<G>| {
            // SAFETY: fresh `DynIter<u64>` handles from the producers.
            let (a, b) = unsafe {
                (
                    it.iter(call_extern1(nodes, g)),
                    it.iter(call_extern1(other, g)),
                )
            };
            // Count pairs where the two streams agree.
            a.zip(b).count_if(ctx, |p| eq(p.first(), p.second()))
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        let g = graph(&[1, 2, 3, 4, 5], &[1, 0, 3, 4]);
        assert_eq!(f.call(&g), 3);
        assert_all_dropped(&g, 2);
    });
}

#[test]
fn zip_filtered_with_indexed() {
    // A filtered stream is not indexed, so this is the pull plan: the filter
    // skips on the left while the right advances once per pair.
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("evens_by_pos", |ctx, a: Var<S>, b: Var<S>| {
            let acc = ctx.var(0u64);
            a.staged_iter()
                .filter(|x| eq(rem(x, 2u64), 0u64))
                .zip(b)
                .for_each(ctx, move |ctx, x, y| {
                    ctx.store(acc, acc * 100u64 + x * y);
                });
            acc
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        // evens of a: 2, 4, 6 ; b: 1, 2 → 2*1, 4*2
        assert_eq!(f.call(&[1u64, 2, 3, 4, 5, 6][..], &[1u64, 2][..]), 208);
    });
}

#[test]
fn zip_of_i64_range_and_slice_is_indexed() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("signed", |ctx, s: Var<SRef<Slice<i64>>>| {
            range(-2i64, 10i64)
                .zip(s)
                .map(|p| p.first() * p.second())
                .sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        // (-2)*1 + (-1)*1 + 0*1 + 1*1 = -2
        assert_eq!(f.call(&[1i64, 1, 1, 1][..]), -2);
        assert_eq!(f.call(&[][..]), 0);
    });
}

#[test]
fn zip_of_mapped_slices() {
    // `map` keeps random access, so this is the shared-counter plan (it did not
    // compile before the pull redesign).
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("mapped", |ctx, a: Var<S>, b: Var<S>| {
            a.staged_iter()
                .map(|x| x + 1u64)
                .zip(b.staged_iter().map(|y| y * 2u64))
                .map(|p| p.first() * p.second())
                .sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        // (1+1)*(3*2) + (2+1)*(4*2) = 12 + 24
        assert_eq!(f.call(&[1u64, 2][..], &[3u64, 4, 5][..]), 36);
    });
}

// =============================================================================
// chain
// =============================================================================

#[test]
fn chain_of_slices_preserves_order() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("digits", |ctx, a: Var<S>, b: Var<S>| {
            let acc = ctx.var(0u64);
            a.staged_iter().chain(b).for_each(ctx, move |ctx, x| {
                ctx.store(acc, acc * 10u64 + x);
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1u64, 2][..], &[3u64, 4, 5][..]), 12345);
        assert_eq!(f.call(&[][..], &[7u64][..]), 7);
        assert_eq!(f.call(&[8u64][..], &[][..]), 8);
        assert_eq!(f.call(&[][..], &[][..]), 0);
    });
}

#[test]
fn chain_extern_then_slice() {
    for_each_backend(|mut compiler| {
        let nodes = compiler.extern_fn::<PullGraphNodesExtern>();
        let it = compiler.opaque_iter_fns::<DynIter<u64>>();
        let f = compiler.fun2("mem_then_disk", move |ctx, g: Var<G>, disk: Var<S>| {
            // SAFETY: a fresh `DynIter<u64>` handle from the producer.
            unsafe { it.iter(call_extern1(nodes, g)) }
                .chain(disk)
                .enumerate()
                .map(|p| p.first() * p.second())
                .sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        let g = graph(&[5, 6], &[]);
        // 0*5 + 1*6 + 2*7 + 3*8
        assert_eq!(f.call(&g, &[7u64, 8][..]), 44);
        assert_all_dropped(&g, 1);
    });
}

#[test]
fn take_while_after_chain_stops_everything() {
    // A downstream `take_while` failing in the *first* half must end the whole
    // chain — the second half must not run (the push-model `break_loop` bug).
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("tw", |ctx, a: Var<S>, b: Var<S>| {
            a.staged_iter()
                .chain(b)
                .take_while(|x| lt(x, 10u64))
                .sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1u64, 2, 10, 3][..], &[5u64, 6][..]), 3);
        assert_eq!(f.call(&[1u64, 2][..], &[5u64, 60, 6][..]), 8);
    });
}

#[test]
fn take_while_inside_chain_moves_on() {
    // `a.take_while(p).chain(b)`: `a` ends at its first failure, then `b` runs.
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("tw_inner", |ctx, a: Var<S>, b: Var<S>| {
            a.staged_iter()
                .take_while(|x| lt(x, 10u64))
                .chain(b)
                .sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1u64, 2, 10, 3][..], &[5u64][..]), 8);
    });
}

#[test]
fn early_exit_in_first_half_drops_both_handles() {
    for_each_backend(|mut compiler| {
        let nodes = compiler.extern_fn::<PullGraphNodesExtern>();
        let other = compiler.extern_fn::<PullGraphOtherExtern>();
        let it = compiler.opaque_iter_fns::<DynIter<u64>>();
        let f = compiler.fun2("any_big", move |ctx, g: Var<G>, t: Var<u64>| {
            // SAFETY: fresh `DynIter<u64>` handles from the producers.
            let (a, b) = unsafe {
                (
                    it.iter(call_extern1(nodes, g)),
                    it.iter(call_extern1(other, g)),
                )
            };
            a.chain(b).any(ctx, move |x| gt(x, t))
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();

        let g = graph(&[1, 50, 2], &[3, 4]);
        assert!(f.call(&g, 10)); // found in the first half
        assert_all_dropped(&g, 2);

        let g = graph(&[1, 2], &[3, 40]);
        assert!(f.call(&g, 10)); // found in the second half
        assert_all_dropped(&g, 2);

        let g = graph(&[1, 2], &[3, 4]);
        assert!(!f.call(&g, 10)); // exhausted
        assert_all_dropped(&g, 2);
    });
}

// =============================================================================
// take / skip
// =============================================================================

#[test]
fn take_and_skip_on_a_slice() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun3("window", |ctx, s: Var<S>, k: Var<u64>, n: Var<u64>| {
            s.staged_iter().skip(k).take(n).sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        let s = [1u64, 2, 3, 4, 5];
        assert_eq!(f.call(&s[..], 1, 3), 9);
        assert_eq!(f.call(&s[..], 0, 0), 0);
        assert_eq!(f.call(&s[..], 4, 10), 5);
        assert_eq!(f.call(&s[..], 9, 2), 0);
    });
}

#[test]
fn take_and_skip_on_an_extern_stream() {
    for_each_backend(|mut compiler| {
        let nodes = compiler.extern_fn::<PullGraphNodesExtern>();
        let it = compiler.opaque_iter_fns::<DynIter<u64>>();
        let f = compiler.fun1("mid", move |ctx, g: Var<G>| {
            let acc = ctx.var(0u64);
            // SAFETY: a fresh `DynIter<u64>` handle from the producer.
            unsafe { it.iter(call_extern1(nodes, g)) }
                .skip(1u64)
                .take(2u64)
                .for_each(ctx, move |ctx, x| ctx.store(acc, acc * 10u64 + x));
            acc
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        let g = graph(&[1, 2, 3, 4], &[]);
        assert_eq!(f.call(&g), 23);
        assert_all_dropped(&g, 1);
    });
}

#[test]
fn skip_and_take_keep_the_shared_counter_in_zip() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("shifted", |ctx, a: Var<S>, b: Var<S>| {
            let acc = ctx.var(0u64);
            a.staged_iter()
                .skip(1u64)
                .zip(b.staged_iter().take(2u64))
                .for_each(ctx, move |ctx, x, y| {
                    ctx.store(acc, acc * 100u64 + x * y);
                });
            acc
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        // a[1..] = 2, 3, 4 ; b.take(2) = 5, 6 → 10, 18
        assert_eq!(f.call(&[1u64, 2, 3, 4][..], &[5u64, 6, 7][..]), 1018);
    });
}

// =============================================================================
// rev on an empty range (length used to wrap)
// =============================================================================

#[test]
fn rev_of_an_empty_range_is_empty() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("rev_count", |ctx, a: Var<u64>, b: Var<u64>| {
            range(a, b).rev().count(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let f = compiled.as_fn();
        assert_eq!(f.call(2, 5), 3);
        assert_eq!(f.call(5, 3), 0);
    });
}
