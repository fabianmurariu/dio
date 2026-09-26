//! Pooled, chunked external iterators: a Rust iterator is driven from staged
//! code a chunk at a time through reused per-nesting-level slots that stage 0
//! assigns (nested loops get distinct slots, siblings share one).

#![allow(clippy::missing_safety_doc)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use rust_lms::prelude::*;
use rust_lms_std::{
    ChunkSlot, HostVec, IterPool, IterPoolRef, PooledIterFns, PooledIterKind, SVec, StagedIterPool,
    SvecGrowExtern,
};

mod common;
use common::for_each_backend;

// --- counting allocator: per thread, so tests running in parallel don't count
// each other's allocations ---
thread_local! {
    static ALLOCS: Cell<usize> = const { Cell::new(0) };
}
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.with(|a| a.set(a.get() + 1));
        unsafe { System.alloc(l) }
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}
#[global_allocator]
static GA: Counting = Counting;

/// A trivial adjacency-list graph, opaque to staged code.
pub struct Graph {
    adj: Vec<Vec<u64>>,
    /// Counts refill-driving `next` calls past the end, to prove an exhausted
    /// iterator is never polled again.
    polls_after_end: Cell<u64>,
}

impl Graph {
    fn new(adj: Vec<Vec<u64>>) -> Self {
        Graph {
            adj,
            polls_after_end: Cell::new(0),
        }
    }
}

/// Node ids `0..len`.
#[extern_fn]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pool_nodes(g: &Graph, slot: &mut ChunkSlot<u64>) {
    unsafe { slot.emplace(0u64..g.adj.len() as u64) };
}

/// Neighbours of `n`.
#[extern_fn]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pool_neighbours(g: &Graph, n: u64, slot: &mut ChunkSlot<u64>) {
    unsafe { slot.emplace(g.adj[n as usize].iter().copied()) };
}

/// `0..n`, but records any `next` after it has returned `None`.
#[extern_fn]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pool_counted(g: &Graph, n: u64, slot: &mut ChunkSlot<u64>) {
    let mut i = 0u64;
    let mut ended = false;
    let polls = &g.polls_after_end;
    unsafe {
        slot.emplace(std::iter::from_fn(move || {
            if ended {
                polls.set(polls.get() + 1);
                return None;
            }
            if i < n {
                i += 1;
                Some(i - 1)
            } else {
                ended = true;
                None
            }
        }))
    };
}

/// Emplaces nothing: an empty iteration.
#[extern_fn]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn pool_nothing(_g: &Graph, _slot: &mut ChunkSlot<u64>) {}

struct Nodes;
impl PooledIterKind for Nodes {
    type Item = u64;
    type Init = PoolNodesExtern;
}

struct Neighbours;
impl PooledIterKind for Neighbours {
    type Item = u64;
    type Init = PoolNeighboursExtern;
}

struct Counted;
impl PooledIterKind for Counted {
    type Item = u64;
    type Init = PoolCountedExtern;
}

struct Nothing;
impl PooledIterKind for Nothing {
    type Item = u64;
    type Init = PoolNothingExtern;
}

type G = SRef<Opaque<Graph>>;

fn sample() -> Graph {
    Graph::new(vec![
        vec![1, 2, 3],
        vec![4],
        vec![5, 6, 7, 8, 9, 10, 11],
        vec![],
        vec![12, 13],
    ])
}

fn neighbour_sum(g: &Graph) -> u64 {
    g.adj.iter().flatten().sum()
}

/// Nested `nodes → neighbours`, summing neighbour ids. Chunk sizes that do and
/// don't divide the set sizes, including 1.
#[test]
fn nested_traversal_sums_neighbours() {
    for chunk in [1u64, 2, 3, 1024] {
        for_each_backend(|mut compiler| {
            let nodes = PooledIterFns::<Nodes>::register(&mut compiler);
            let neigh = PooledIterFns::<Neighbours>::register(&mut compiler);
            let f = compiler.fun2("sum", move |ctx, g: Var<G>, pool: Var<IterPoolRef>| {
                let pool = &StagedIterPool::new(ctx, pool);
                let total = ctx.var(0u64);
                nodes.iter1(pool, g, chunk).for_each(ctx, move |ctx, n| {
                    neigh.iter2(pool, g, n, chunk).for_each(ctx, move |ctx, b| {
                        ctx.store(total, add(total, b));
                    });
                });
                assert_eq!(pool.slots_used(), 2);
                total
            });
            let compiled = compiler.compile(f).expect("compile");
            let g = sample();
            let mut pool = IterPool::new();
            assert_eq!(compiled.call(&g, &mut pool), neighbour_sum(&g));
            assert_eq!(pool.num_slots(), 2);
        });
    }
}

/// Two sibling loops inside one outer loop share slot 1; the pool never grows
/// past the nesting depth.
#[test]
fn sibling_loops_share_a_slot() {
    for_each_backend(|mut compiler| {
        let nodes = PooledIterFns::<Nodes>::register(&mut compiler);
        let neigh = PooledIterFns::<Neighbours>::register(&mut compiler);
        let f = compiler.fun2("siblings", move |ctx, g: Var<G>, pool: Var<IterPoolRef>| {
            let pool = &StagedIterPool::new(ctx, pool);
            let total = ctx.var(0u64);
            nodes.iter1(pool, g, 2).for_each(ctx, move |ctx, n| {
                neigh.iter2(pool, g, n, 2).for_each(ctx, move |ctx, b| {
                    ctx.store(total, add(total, b));
                });
                // Second sibling counts the same neighbours again.
                let c = neigh.iter2(pool, g, n, 3).count(ctx);
                ctx.store(total, add(total, mul(c, 1000u64)));
            });
            assert_eq!(pool.slots_used(), 2);
            total
        });
        let compiled = compiler.compile(f).expect("compile");
        let g = sample();
        let mut pool = IterPool::new();
        let edges: u64 = g.adj.iter().map(|a| a.len() as u64).sum();
        assert_eq!(
            compiled.call(&g, &mut pool),
            neighbour_sum(&g) + 1000 * edges
        );
        assert_eq!(pool.num_slots(), 2);
    });
}

/// `any` breaks out mid-chunk: the answer is right, and the iterator is dropped
/// on the break path, so the slot is empty for the next call.
#[test]
fn any_short_circuits_mid_chunk() {
    for_each_backend(|mut compiler| {
        let counted = PooledIterFns::<Counted>::register(&mut compiler);
        let f = compiler.fun3(
            "any",
            move |ctx, g: Var<G>, n: Var<u64>, pool: Var<IterPoolRef>| {
                let pool = &StagedIterPool::new(ctx, pool);
                counted.iter2(pool, g, n, 4).any(ctx, |x| eq(x, 5u64))
            },
        );
        let compiled = compiler.compile(f).expect("compile");
        let g = sample();
        let mut pool = IterPool::new();
        assert!(compiled.call(&g, 100, &mut pool));
        assert!(!compiled.call(&g, 3, &mut pool));
        assert!(compiled.call(&g, 6, &mut pool));
    });
}

/// A short final chunk ends the iteration without polling the exhausted
/// iterator again.
#[test]
fn exhausted_iterator_is_not_polled_again() {
    for_each_backend(|mut compiler| {
        let counted = PooledIterFns::<Counted>::register(&mut compiler);
        let f = compiler.fun3(
            "count",
            move |ctx, g: Var<G>, n: Var<u64>, pool: Var<IterPoolRef>| {
                let pool = &StagedIterPool::new(ctx, pool);
                counted.iter2(pool, g, n, 4).count(ctx)
            },
        );
        let compiled = compiler.compile(f).expect("compile");
        let g = sample();
        let mut pool = IterPool::new();
        for n in [0u64, 1, 3, 4, 5, 8, 9] {
            assert_eq!(compiled.call(&g, n, &mut pool), n);
        }
        assert_eq!(g.polls_after_end.get(), 0);
    });
}

/// A producer that emplaces nothing is an empty iteration.
#[test]
fn empty_producer_yields_nothing() {
    for_each_backend(|mut compiler| {
        let nothing = PooledIterFns::<Nothing>::register(&mut compiler);
        let f = compiler.fun2("none", move |ctx, g: Var<G>, pool: Var<IterPoolRef>| {
            let pool = &StagedIterPool::new(ctx, pool);
            nothing.iter1(pool, g, 8).count(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let g = sample();
        let mut pool = IterPool::new();
        assert_eq!(compiled.call(&g, &mut pool), 0);
    });
}

/// Combinators fuse over the pooled source, and results are written out to an
/// output `SVec` — the pipeline shape: read chunks in, push rows out.
#[test]
fn filter_map_into_output_svec() {
    for_each_backend(|mut compiler| {
        let mut out = HostVec::<u64>::new();
        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `out` owns a u64 control block and outlives every call.
        let mut svec = unsafe { SVec::<u64>::new(out.handle(), grow) };
        let nodes = PooledIterFns::<Nodes>::register(&mut compiler);
        let neigh = PooledIterFns::<Neighbours>::register(&mut compiler);
        let f = compiler.fun2("pipeline", move |ctx, g: Var<G>, pool: Var<IterPoolRef>| {
            let pool = &StagedIterPool::new(ctx, pool);
            nodes.iter1(pool, g, 3).for_each(ctx, |ctx, n| {
                neigh
                    .iter2(pool, g, n, 2)
                    .filter(|b| eq(bitand(b, 1u64), 1u64))
                    .map(move |b| add(mul(n, 100u64), b))
                    .for_each(ctx, |ctx, row| svec.push(ctx, row));
            });
            svec.len(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let g = sample();
        let mut pool = IterPool::new();
        let expected: Vec<u64> = g
            .adj
            .iter()
            .enumerate()
            .flat_map(|(n, a)| {
                a.iter()
                    .filter(|b| *b & 1 == 1)
                    .map(move |b| n as u64 * 100 + b)
            })
            .collect();
        assert_eq!(compiled.call(&g, &mut pool), expected.len() as u64);
        assert_eq!(out.as_slice(), expected.as_slice());
    });
}

/// Once the pool has warmed up, a nested traversal allocates nothing: iterators
/// live inline in their slots and chunk buffers keep their capacity.
#[test]
fn warm_pool_traversal_does_not_allocate() {
    for_each_backend(|mut compiler| {
        let nodes = PooledIterFns::<Nodes>::register(&mut compiler);
        let neigh = PooledIterFns::<Neighbours>::register(&mut compiler);
        let f = compiler.fun2("sum", move |ctx, g: Var<G>, pool: Var<IterPoolRef>| {
            let pool = &StagedIterPool::new(ctx, pool);
            let total = ctx.var(0u64);
            nodes.iter1(pool, g, 2).for_each(ctx, move |ctx, n| {
                neigh.iter2(pool, g, n, 4).for_each(ctx, move |ctx, b| {
                    ctx.store(total, add(total, b));
                });
            });
            total
        });
        let compiled = compiler.compile(f).expect("compile");
        let kernel = compiled.as_fn();
        let g = sample();
        let mut pool = IterPool::new();
        assert_eq!(kernel.call(&g, &mut pool), neighbour_sum(&g)); // warm-up

        let before = ALLOCS.with(Cell::get);
        let got = kernel.call(&g, &mut pool);
        let allocs = ALLOCS.with(Cell::get) - before;
        assert_eq!(got, neighbour_sum(&g));
        assert_eq!(allocs, 0, "warm pooled traversal allocated {allocs} times");
    });
}
