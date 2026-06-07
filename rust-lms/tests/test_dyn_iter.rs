//! Generic `Box<dyn Iterator<Item = T>>` / `Box<dyn ExactSizeIterator>` support:
//! the library supplies next/drop/len/next_value generically, so a user writes
//! only the domain producer that boxes their iterator. Covers `u64` and `f64`
//! (the latter exercises the COption FP-register ABI fix).

#![allow(clippy::missing_safety_doc)]

use either::Either;
use raphtory::{
    core::entities::{EID, VID},
    db::{
        api::{
            mutation::AdditionOps,
            view::{EdgeViewOps, GraphViewOps, NodeViewOps},
        },
        graph::{edge::EdgeView, graph::Graph},
    },
    storage::{core_ops::CoreGraphOps, graph::edges::edge_storage_ops::EdgeStorageOps},
};
use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

// Producers: box a (borrowing) iterator into a thin `*mut ()` handle. The
// library's DynIter<T> provides everything else.
#[extern_fn]
#[no_mangle]
pub extern "C" fn all_node_internal_ids(g: &Graph) -> *mut () {
    box_dyn_iter(g.nodes().iter().map(|node| node.node.as_u64()))
}

#[extern_fn]
#[no_mangle]
pub extern "C" fn neighbours(g: &Graph, n: u64) -> *mut () {
    let vid = VID(n as usize);
    box_dyn_iter(
        g.node(vid)
            .into_iter()
            .flat_map(|n| n.neighbours().into_iter().map(|n| n.node.0 as u64)),
    )
}

#[extern_fn]
#[no_mangle]
pub extern "C" fn edges(g: &Graph, n: u64) -> *mut () {
    let vid = VID(n as usize);
    box_dyn_iter(g.node(vid).into_iter().flat_map(|n| {
        n.edges().into_iter().map(|e| {
            (
                e.edge.pid().0 as u64,
                e.edge.src().0 as u64,
                e.edge.dst().0 as u64,
            )
        })
    }))
}

#[extern_fn]
#[no_mangle]
pub extern "C" fn weight(g: &Graph, e: u64) -> COption<i32> {
    let eid = EID(e as usize);
    let edge_ref = g
        .core_edge(Either::Left(eid))
        .as_ref()
        .edge_ref(raphtory::core::entities::edges::edge_ref::Dir::Out);
    let edge_view = EdgeView::new(g, edge_ref);
    let val = edge_view
        .properties()
        .temporal()
        .get("weight")
        .and_then(|props| props.latest().and_then(|p| p.cast_num::<i32>()));
    val.into()
}

// Raphtory edge weights are `i32` (see `weight`); these synthetic f64 producers
// only exist to keep the COption<f64> register-consume / ExactSize ABI coverage
// (the real weight path is exercised by the traversal work).
#[extern_fn]
#[no_mangle]
pub extern "C" fn g_weights(_g: &Graph) -> *mut () {
    box_dyn_iter([1.5f64, 2.5, 3.0].into_iter())
}

#[extern_fn]
#[no_mangle]
pub extern "C" fn g_weights_exact(_g: &Graph) -> *mut () {
    box_dyn_exact_iter([1.5f64, 2.5, 3.0].into_iter())
}

fn graph() -> Graph {
    let g = Graph::new();
    g.add_edge(-3, 1, 2, [("weight", 4i32)], None).unwrap();
    g.add_edge(0, 2, 3, [("weight", 2i32)], None).unwrap();
    g.add_edge(3, 1, 4, [("weight", 1i32)], None).unwrap();
    g.add_edge(7, 4, 3, [("weight", 3i32)], None).unwrap();
    g.add_edge(8, 1, 3, [("weight", 13i32)], None).unwrap();
    g
}

#[test]
fn dyn_iter_u64_sum_and_filter() {
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<AllNodeInternalIdsExtern>();
    let nodes = compiler.opaque_iter_fns::<DynIter<u64>>();

    let f = compiler.fun1("sum_ids_gt2", move |ctx, g: Var<SRef<Opaque<Graph>>>| {
        nodes
            .iter(call_extern1::<_, _, SRef<Opaque<Graph>>, OpaqueHandle>(
                producer, g,
            ))
            .filter(|x| lt(2u64, x))
            .sum(ctx)
    });
    let kernel = compiler.compile(f).expect("compile").as_fn();

    // Compare the JIT'd traversal against the same computation run host-side
    // (node VIDs > 2, summed), so it's correct regardless of raphtory's VID map.
    let g = graph();
    let expected: u64 = g
        .nodes()
        .iter()
        .map(|n| n.node.as_u64())
        .filter(|&x| x > 2)
        .sum();
    assert_eq!(kernel(&g), expected);
}

#[test]
fn dyn_iter_f64_sum_register_consume() {
    // Plain next/drop path over f64 — the COption<f64> payload must come back in
    // an FP register (the ABI fix), then feed the accumulator.
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<GWeightsExtern>();
    let w = compiler.opaque_iter_fns::<DynIter<f64>>();

    let f = compiler.fun1("sum_weights", move |ctx, g: Var<SRef<Opaque<Graph>>>| {
        w.iter(call_extern1::<_, _, SRef<Opaque<Graph>>, OpaqueHandle>(
            producer, g,
        ))
        .sum(ctx)
    });
    let kernel = compiler.compile(f).expect("compile").as_fn();
    assert!((kernel(&graph()) - 7.0).abs() < 1e-9); // 1.5 + 2.5 + 3.0
}

#[test]
fn dyn_exact_iter_f64_count_and_sum() {
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<GWeightsExactExtern>();
    let w = compiler.exact_opaque_iter_fns::<DynExactIter<f64>>();

    // O(1) count via len(it).
    let count_fn = compiler.fun1("count_w", move |ctx, g: Var<SRef<Opaque<Graph>>>| {
        w.iter(call_extern1::<_, _, SRef<Opaque<Graph>>, OpaqueHandle>(
            producer, g,
        ))
        .count(ctx)
    });
    let count = compiler.compile(count_fn).expect("compile").as_fn();
    assert_eq!(count(&graph()), 3);

    // Counted-loop sum via len + next_value.
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<GWeightsExactExtern>();
    let w = compiler.exact_opaque_iter_fns::<DynExactIter<f64>>();
    let sum_fn = compiler.fun1("sum_w_exact", move |ctx, g: Var<SRef<Opaque<Graph>>>| {
        w.iter(call_extern1::<_, _, SRef<Opaque<Graph>>, OpaqueHandle>(
            producer, g,
        ))
        .sum(ctx)
    });
    let sum = compiler.compile(sum_fn).expect("compile").as_fn();
    assert!((sum(&graph()) - 7.0).abs() < 1e-9);
}
