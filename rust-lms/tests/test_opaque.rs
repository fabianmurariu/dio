//! Concept 1: opaque external borrows (`Opaque<T>` behind `SRef`/`SRefMut`).
//!
//! Staged code receives an opaque `&Graph` / `&mut Graph`, never inspects it,
//! and hands it to `extern "C"` functions. The compiled kernel's ABI is
//! `extern "C" fn(&Graph) -> _`, so a real `&graph` is passed directly.

use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

/// A stand-in for an external graph library (e.g. raphtory). Opaque to staged
/// code — it never implements `StagedType`.
pub struct Graph {
    nodes: Vec<u64>,
}

#[extern_fn]
#[no_mangle]
pub extern "C" fn graph_node_count(g: &Graph) -> u64 {
    g.nodes.len() as u64
}

#[extern_fn]
#[no_mangle]
pub extern "C" fn graph_push_node(g: &mut Graph, id: u64) {
    g.nodes.push(id);
}

#[test]
fn opaque_ref_param_round_trips() {
    let mut compiler = Compiler::new();
    let node_count = compiler.extern_fn::<GraphNodeCountExtern>();

    // fn(&Graph) -> u64 : just delegate to the extern.
    let f = compiler.fun1("count_nodes", move |_ctx, g: Var<SRef<Opaque<Graph>>>| {
        call_extern1(node_count, g)
    });
    let compiled = compiler.compile(f).expect("compile");
    let kernel = compiled.as_fn();

    let g = Graph {
        nodes: vec![10, 20, 30, 40],
    };
    assert_eq!(kernel.call(&g), 4); // &g passed straight through
}

#[test]
fn opaque_mut_ref_param() {
    let mut compiler = Compiler::new();
    let push = compiler.extern_fn::<GraphPushNodeExtern>();
    let node_count = compiler.extern_fn::<GraphNodeCountExtern>();

    // fn(&mut Graph) -> u64 : push one node, return the new count.
    let f = compiler.fun1(
        "push_and_count",
        move |ctx, g: Var<SRefMut<Opaque<Graph>>>| {
            ctx.emit(call_extern2(push, g, Const::<u64>::new(99)));
            call_extern1(node_count, ref_as_const(g))
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    let kernel = compiled.as_fn();

    let mut g = Graph { nodes: vec![1, 2] };
    assert_eq!(kernel.call(&mut g), 3);
    assert_eq!(g.nodes, vec![1, 2, 99]);
}
