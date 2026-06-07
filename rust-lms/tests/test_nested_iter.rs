//! Reused-storage nested iterators: `for n in nodes { for dst in neighbours(n) }`
//! with one slot reserved per level (the hand-rolled mini-vtable), so the inner
//! iterator is reconstructed in place each time — no per-set allocation.
//!
//! A counting global allocator proves the JIT'd nested traversal performs *zero*
//! heap allocations (the slice/range iterators fit the inline budget).

#![allow(clippy::missing_safety_doc)]

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

// --- counting allocator (this is an integration test = its own binary) ---
static ALLOCS: AtomicUsize = AtomicUsize::new(0);
struct Counting;
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        ALLOCS.fetch_add(1, Ordering::Relaxed);
        System.alloc(l)
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        System.dealloc(p, l)
    }
}
#[global_allocator]
static GA: Counting = Counting;

/// A trivial adjacency-list "graph", opaque to staged code.
pub struct Nums {
    adj: Vec<Vec<u64>>,
}

/// Outer producer: node ids `0..len`. `Range<u64>` (16 B) → lives inline.
#[extern_fn]
#[no_mangle]
pub unsafe extern "C" fn nums_nodes(g: &Nums, slot: *mut ()) {
    emplace_iter(slot as *mut OpaqueIterSlot<u64>, 0u64..g.adj.len() as u64);
}

/// Inner producer: neighbours of `n`. `Copied<slice::Iter<u64>>` (16 B) → inline.
#[extern_fn]
#[no_mangle]
pub unsafe extern "C" fn nums_neighbours(g: &Nums, n: u64, slot: *mut ()) {
    emplace_iter(
        slot as *mut OpaqueIterSlot<u64>,
        g.adj[n as usize].iter().copied(),
    );
}

struct NodesK;
impl ReusedOpaqueIterKind for NodesK {
    type Item = u64;
    type Init = NumsNodesExtern;
}

struct NeighK;
impl ReusedOpaqueIterKind for NeighK {
    type Item = u64;
    type Init = NumsNeighboursExtern;
}

#[test]
fn nested_traversal_sums_neighbours_with_zero_allocation() {
    let mut compiler = Compiler::new();
    let nodes = compiler.reused_opaque_iter_fns::<NodesK>();
    let neigh = compiler.reused_opaque_iter_fns::<NeighK>();

    // sum over all nodes of (sum of that node's neighbour ids)
    let f = compiler.fun1("sum_neighbours", move |ctx, g: Var<SRef<Opaque<Nums>>>| {
        let total = ctx.var(0u64);
        nodes.iter1(g).for_each(ctx, move |ctx, n| {
            neigh.iter2(g, n).for_each(ctx, move |ctx, dst| {
                ctx.store(total, add(total, dst));
            });
        });
        total
    });
    let kernel = compiler.compile(f).expect("compile").as_fn();

    let g = Nums {
        adj: vec![vec![1, 2, 3], vec![4], vec![5, 6], vec![]],
    };
    let expected: u64 = g.adj.iter().flatten().copied().sum(); // 1+2+3+4+5+6

    // Measure allocations *during* the JIT'd nested traversal only.
    let before = ALLOCS.load(Ordering::Relaxed);
    let got = kernel(&g);
    let allocs = ALLOCS.load(Ordering::Relaxed) - before;

    assert_eq!(got, expected, "nested traversal produced the wrong sum");
    assert_eq!(
        allocs, 0,
        "nested traversal allocated {allocs} times (expected 0 — iterators live in the reused slots)"
    );
}
