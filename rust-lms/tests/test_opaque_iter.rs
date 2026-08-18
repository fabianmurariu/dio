//! Concept 2: opaque external iterators driven from staged code.
//!
//! Stands in for a graph library (e.g. raphtory): `graph_iter_nodes(&Graph)`
//! returns a boxed `IntoIter<u64>` as an opaque `*mut ()` handle, and the staged
//! iterator drives it via `next`/`drop` (register-consume loop) or, on the
//! ExactSize path, `len`/`next_value` (counted loop, O(1) `count`).

// Test-helper extern fns deref the opaque `*mut ()` handle by contract.
#![allow(clippy::missing_safety_doc)]

use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

pub struct Graph {
    nodes: Vec<u64>,
}

type NodeIter = std::vec::IntoIter<u64>;

/// Producer: a fresh owned iterator over the node ids, boxed to a thin handle.
#[extern_fn]
#[no_mangle]
pub extern "C" fn graph_iter_nodes(g: &Graph) -> *mut () {
    let it: Box<NodeIter> = Box::new(g.nodes.clone().into_iter());
    Box::into_raw(it) as *mut ()
}

#[extern_fn]
#[no_mangle]
pub unsafe extern "C" fn graph_nodes_next(it: *mut ()) -> COption<u64> {
    (*(it as *mut NodeIter)).next().into()
}

#[extern_fn]
#[no_mangle]
pub unsafe extern "C" fn graph_nodes_next_value(it: *mut ()) -> u64 {
    (*(it as *mut NodeIter))
        .next()
        .expect("next_value past len")
}

#[extern_fn]
#[no_mangle]
pub unsafe extern "C" fn graph_nodes_len(it: *mut ()) -> u64 {
    (*(it as *mut NodeIter)).len() as u64
}

#[extern_fn]
#[no_mangle]
pub unsafe extern "C" fn graph_nodes_drop(it: *mut ()) {
    drop(Box::from_raw(it as *mut NodeIter));
}

/// The kind: bundles next/drop (and len/next_value for the ExactSize path).
struct NodesKind;
impl OpaqueIterKind for NodesKind {
    type Item = u64;
    type Next = GraphNodesNextExtern;
    type Drop = GraphNodesDropExtern;
}
impl ExactSizeOpaqueIterKind for NodesKind {
    type Len = GraphNodesLenExtern;
    type NextValue = GraphNodesNextValueExtern;
}

fn graph(nodes: Vec<u64>) -> Graph {
    Graph { nodes }
}

#[test]
fn plain_count_via_next_drop() {
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<GraphIterNodesExtern>();
    let nodes = compiler.opaque_iter_fns::<NodesKind>();

    // fn(&Graph) -> u64 : drive the external iterator and count it.
    let f = compiler.fun1("count_nodes", move |ctx, g: Var<SRef<Opaque<Graph>>>| {
        let handle = call_extern1(producer, g);
        nodes.iter(handle).count(ctx)
    });
    let compiled = compiler.compile(f).expect("compile");
    let kernel = compiled.as_fn();

    let g = graph(vec![10, 20, 30, 40, 50]);
    assert_eq!(kernel.call(&g), 5);
    assert_eq!(kernel.call(&graph(vec![])), 0);
}

#[test]
fn plain_sum_and_filter() {
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<GraphIterNodesExtern>();
    let nodes = compiler.opaque_iter_fns::<NodesKind>();

    // Sum the node ids > 2, branchlessly.
    let f = compiler.fun1("sum_big", move |ctx, g: Var<SRef<Opaque<Graph>>>| {
        let handle = call_extern1(producer, g);
        nodes.iter(handle).filter(|x| lt(2u64, x)).sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile");
    let kernel = compiled.as_fn();

    let g = graph(vec![1, 2, 3, 4, 5]);
    assert_eq!(kernel.call(&g), 3 + 4 + 5);
}

#[test]
fn exact_size_count_is_o1_and_sum_works() {
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<GraphIterNodesExtern>();
    let nodes = compiler.exact_opaque_iter_fns::<NodesKind>();

    // count(): O(1), just len(it).
    let count_fn = compiler.fun1("count_exact", move |ctx, g: Var<SRef<Opaque<Graph>>>| {
        let handle = call_extern1(producer, g);
        nodes.iter(handle).count(ctx)
    });
    let compiled = compiler.compile(count_fn).expect("compile");
    let count = compiled.as_fn();
    assert_eq!(count.call(&graph(vec![7, 8, 9])), 3);

    // sum() over the counted loop (len + next_value).
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<GraphIterNodesExtern>();
    let nodes = compiler.exact_opaque_iter_fns::<NodesKind>();
    let sum_fn = compiler.fun1("sum_exact", move |ctx, g: Var<SRef<Opaque<Graph>>>| {
        let handle = call_extern1(producer, g);
        nodes.iter(handle).sum(ctx)
    });
    let compiled = compiler.compile(sum_fn).expect("compile");
    let sum = compiled.as_fn();
    assert_eq!(sum.call(&graph(vec![10, 20, 30, 40])), 100);
}
