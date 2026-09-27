//! Chunked external iterators: a Rust iterator driven from staged code a chunk at
//! a time through a stack-frame slot per nesting level. The chunk boundary
//! (`CHUNK` items) is where the protocol can go wrong, so lengths straddle it.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

use rust_lms::prelude::*;
use rust_lms_std::{
    CHUNK, ChunkStart, ChunkedIterFns, ChunkedIterKind, HostVec, SVec, SvecGrowExtern,
};

mod common;
use common::for_each_backend;

// --- counting allocator: per thread, so parallel tests don't count each other ---
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
    /// `next` calls after an iterator returned `None`.
    polls_after_end: Cell<u64>,
    /// Iterators started minus iterators dropped.
    live: Cell<i64>,
}

impl Graph {
    fn new(adj: Vec<Vec<u64>>) -> Self {
        Graph {
            adj,
            polls_after_end: Cell::new(0),
            live: Cell::new(0),
        }
    }
}

/// Lengths straddling the chunk boundary, with distinct ids.
fn boundary_graph() -> Graph {
    let lens = [0, 1, 3, CHUNK - 1, CHUNK, CHUNK + 1, 2 * CHUNK, 200];
    let mut next = 0u64;
    Graph::new(
        lens.iter()
            .map(|&len| {
                let v: Vec<u64> = (next..next + len as u64).collect();
                next += len as u64;
                v
            })
            .collect(),
    )
}

/// Node ids `0..len`. The range owns its bounds, so this producer is safe.
#[extern_fn]
pub extern "C" fn ch_nodes(g: &Graph, slot: &mut ChunkStart<u64>) {
    slot.start(0u64..g.adj.len() as u64);
}

/// Neighbours of `n`.
///
/// # Safety
///
/// `g` must outlive the kernel call that runs the traversal.
#[extern_fn]
pub unsafe extern "C" fn ch_neighbours(g: &Graph, n: u64, slot: &mut ChunkStart<u64>) {
    unsafe { slot.start_borrowed(g.adj[n as usize].iter().copied()) };
}

/// An iterator that tracks its own lifetime, and reports polls after its end.
/// Its size hint is unknown, so a full final chunk needs one more fill.
struct Tracked<'g> {
    g: &'g Graph,
    items: std::vec::IntoIter<u64>,
    ended: bool,
}

impl Iterator for Tracked<'_> {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        if self.ended {
            self.g.polls_after_end.set(self.g.polls_after_end.get() + 1);
            return None;
        }
        let x = self.items.next();
        self.ended = x.is_none();
        x
    }
}

impl Drop for Tracked<'_> {
    fn drop(&mut self) {
        self.g.live.set(self.g.live.get() - 1);
    }
}

/// Neighbours of `n` through `Tracked`.
///
/// # Safety
///
/// `g` must outlive the kernel call that runs the traversal.
#[extern_fn]
pub unsafe extern "C" fn ch_tracked(g: &Graph, n: u64, slot: &mut ChunkStart<u64>) {
    g.live.set(g.live.get() + 1);
    let items = g.adj[n as usize].clone().into_iter();
    unsafe {
        slot.start_borrowed(Tracked {
            g,
            items,
            ended: false,
        })
    };
}

/// Starts nothing: an empty iteration.
#[extern_fn]
pub extern "C" fn ch_nothing(_g: &Graph, _slot: &mut ChunkStart<u64>) {}

struct Nodes;
impl ChunkedIterKind for Nodes {
    type Item = u64;
    type Init = ChNodesExtern;
}

struct Neighbours;
impl ChunkedIterKind for Neighbours {
    type Item = u64;
    type Init = ChNeighboursExtern;
}

struct TrackedK;
impl ChunkedIterKind for TrackedK {
    type Item = u64;
    type Init = ChTrackedExtern;
}

struct Nothing;
impl ChunkedIterKind for Nothing {
    type Item = u64;
    type Init = ChNothingExtern;
}

type G = SRef<Opaque<Graph>>;

fn neighbour_sum(g: &Graph) -> u64 {
    g.adj.iter().flatten().sum()
}

#[test]
fn nested_traversal_across_chunk_boundaries() {
    for_each_backend(|mut compiler| {
        let nodes = ChunkedIterFns::<Nodes>::register(&mut compiler);
        let neigh = ChunkedIterFns::<Neighbours>::register(&mut compiler);
        let f = compiler.fun1("sum", move |ctx, g: Var<G>| {
            let total = ctx.var(0u64);
            nodes.iter1(g).for_each(ctx, move |ctx, n| {
                neigh.iter2(g, n).for_each(ctx, move |ctx, b| {
                    ctx.store(total, add(total, b));
                });
            });
            total
        });
        let compiled = compiler.compile(f).expect("compile");
        let g = boundary_graph();
        assert_eq!(compiled.call(&g), neighbour_sum(&g));
    });
}

/// Every length, counted separately: a miscounted chunk shows up as a wrong
/// count for exactly one length.
#[test]
fn counts_each_length_exactly() {
    for_each_backend(|mut compiler| {
        let tracked = ChunkedIterFns::<TrackedK>::register(&mut compiler);
        let f = compiler.fun2("count", move |ctx, g: Var<G>, n: Var<u64>| {
            tracked.iter2(g, n).count(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let g = boundary_graph();
        for (n, list) in g.adj.iter().enumerate() {
            assert_eq!(compiled.call(&g, n as u64), list.len() as u64, "list {n}");
        }
        assert_eq!(
            g.polls_after_end.get(),
            0,
            "an exhausted iterator was polled"
        );
        assert_eq!(g.live.get(), 0, "an iterator was leaked or double-dropped");
    });
}

/// `any` breaks out mid-chunk and mid-list: the answer is right and the
/// iterator is dropped on the break path.
#[test]
fn any_breaks_and_drops() {
    for_each_backend(|mut compiler| {
        let tracked = ChunkedIterFns::<TrackedK>::register(&mut compiler);
        let f = compiler.fun3("any", move |ctx, g: Var<G>, n: Var<u64>, x: Var<u64>| {
            tracked.iter2(g, n).any(ctx, move |v| eq(v, x))
        });
        let compiled = compiler.compile(f).expect("compile");
        let g = boundary_graph();
        for (n, list) in g.adj.iter().enumerate() {
            for probe in [
                list.first(),
                list.get(CHUNK - 1),
                list.get(CHUNK),
                list.last(),
            ]
            .into_iter()
            .flatten()
            {
                assert!(
                    compiled.call(&g, n as u64, *probe),
                    "list {n}, value {probe}"
                );
            }
            assert!(
                !compiled.call(&g, n as u64, u64::MAX),
                "list {n}, absent value"
            );
        }
        assert_eq!(g.live.get(), 0, "an iterator was leaked or double-dropped");
    });
}

/// Sibling loops at the same depth each get their own slot.
#[test]
fn sibling_loops() {
    for_each_backend(|mut compiler| {
        let nodes = ChunkedIterFns::<Nodes>::register(&mut compiler);
        let neigh = ChunkedIterFns::<Neighbours>::register(&mut compiler);
        let f = compiler.fun1("siblings", move |ctx, g: Var<G>| {
            let total = ctx.var(0u64);
            nodes.iter1(g).for_each(ctx, move |ctx, n| {
                neigh.iter2(g, n).for_each(ctx, move |ctx, b| {
                    ctx.store(total, add(total, b));
                });
                let c = neigh.iter2(g, n).count(ctx);
                ctx.store(total, add(total, mul(c, 1_000_000u64)));
            });
            total
        });
        let compiled = compiler.compile(f).expect("compile");
        let g = boundary_graph();
        let edges: u64 = g.adj.iter().map(|a| a.len() as u64).sum();
        assert_eq!(compiled.call(&g), neighbour_sum(&g) + 1_000_000 * edges);
    });
}

#[test]
fn empty_producer_yields_nothing() {
    for_each_backend(|mut compiler| {
        let nothing = ChunkedIterFns::<Nothing>::register(&mut compiler);
        let f = compiler.fun1("none", move |ctx, g: Var<G>| nothing.iter1(g).count(ctx));
        let compiled = compiler.compile(f).expect("compile");
        let g = boundary_graph();
        assert_eq!(compiled.call(&g), 0);
    });
}

/// The pipeline shape: read chunks in, filter and map, push rows out.
#[test]
fn filter_map_into_output_svec() {
    for_each_backend(|mut compiler| {
        let mut out = HostVec::<u64>::new();
        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `out` owns a u64 control block and outlives every call.
        let mut svec = unsafe { SVec::<u64>::new(out.handle(), grow) };
        let nodes = ChunkedIterFns::<Nodes>::register(&mut compiler);
        let neigh = ChunkedIterFns::<Neighbours>::register(&mut compiler);
        let f = compiler.fun1("pipeline", move |ctx, g: Var<G>| {
            nodes.iter1(g).for_each(ctx, |ctx, n| {
                neigh
                    .iter2(g, n)
                    .filter(|b| eq(bitand(b, 1u64), 1u64))
                    .map(move |b| add(mul(n, 1000u64), b))
                    .for_each(ctx, |ctx, row| svec.push(ctx, row));
            });
            svec.len(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        let g = boundary_graph();
        let expected: Vec<u64> = g
            .adj
            .iter()
            .enumerate()
            .flat_map(|(n, a)| {
                a.iter()
                    .filter(|b| *b & 1 == 1)
                    .map(move |b| n as u64 * 1000 + b)
            })
            .collect();
        assert_eq!(compiled.call(&g), expected.len() as u64);
        assert_eq!(out.as_slice(), expected.as_slice());
    });
}

/// Slots live in the kernel's stack frame and iterators inline in them: a
/// nested traversal allocates nothing, not even on the first call.
#[test]
fn traversal_does_not_allocate() {
    for_each_backend(|mut compiler| {
        let nodes = ChunkedIterFns::<Nodes>::register(&mut compiler);
        let neigh = ChunkedIterFns::<Neighbours>::register(&mut compiler);
        let f = compiler.fun1("sum", move |ctx, g: Var<G>| {
            let total = ctx.var(0u64);
            nodes.iter1(g).for_each(ctx, move |ctx, n| {
                neigh.iter2(g, n).for_each(ctx, move |ctx, b| {
                    ctx.store(total, add(total, b));
                });
            });
            total
        });
        let compiled = compiler.compile(f).expect("compile");
        let kernel = compiled.as_fn();
        let g = boundary_graph();
        let before = ALLOCS.with(Cell::get);
        let got = kernel.call(&g);
        let allocs = ALLOCS.with(Cell::get) - before;
        assert_eq!(got, neighbour_sum(&g));
        assert_eq!(allocs, 0, "chunked traversal allocated {allocs} times");
    });
}
