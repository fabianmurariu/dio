//! Concept 1: opaque external borrows.
//!
//! The compiled entry point receives a real `&Graph` / `&mut Graph`, exposes its
//! address as a raw opaque pointer, and uses an unsafe staging boundary when it
//! hands that pointer to an `extern "C"` function.

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
        // SAFETY: the compiled entry point requires a live shared `&Graph`.
        unsafe { call_extern1_unchecked(node_count, ref_as_ptr(g)) }
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
            let g_ptr = ctx.bind(ref_mut_as_ptr(g));
            // SAFETY: the entry point receives an exclusive `&mut Graph`, and
            // these calls are sequenced without concurrent access.
            ctx.emit(unsafe { call_extern2_unchecked(push, g_ptr, Const::<u64>::new(99)) });
            unsafe { call_extern1_unchecked(node_count, ptr_as_const(g_ptr)) }
        },
    );
    let compiled = compiler.compile(f).expect("compile");
    let kernel = compiled.as_fn();

    let mut g = Graph { nodes: vec![1, 2] };
    assert_eq!(kernel.call(&mut g), 3);
    assert_eq!(g.nodes, vec![1, 2, 99]);
}
