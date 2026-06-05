//! Generic `Box<dyn Iterator<Item = T>>` / `Box<dyn ExactSizeIterator>` support:
//! the library supplies next/drop/len/next_value generically, so a user writes
//! only the domain producer that boxes their iterator. Covers `u64` and `f64`
//! (the latter exercises the COption FP-register ABI fix).

#![allow(clippy::missing_safety_doc)]

use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

pub struct Graph {
    ids: Vec<u64>,
    weights: Vec<f64>,
}

// Producers: box a (borrowing) iterator into a thin `*mut ()` handle. The
// library's DynIter<T> provides everything else.
#[extern_fn]
#[no_mangle]
pub extern "C" fn g_ids(g: &Graph) -> *mut () {
    box_dyn_iter(g.ids.iter().copied())
}

#[extern_fn]
#[no_mangle]
pub extern "C" fn g_weights(g: &Graph) -> *mut () {
    box_dyn_iter(g.weights.iter().copied())
}

#[extern_fn]
#[no_mangle]
pub extern "C" fn g_weights_exact(g: &Graph) -> *mut () {
    box_dyn_exact_iter(g.weights.iter().copied())
}

fn graph() -> Graph {
    Graph {
        ids: vec![1, 2, 3, 4, 5],
        weights: vec![1.5, 2.5, 3.0],
    }
}

#[test]
fn dyn_iter_u64_sum_and_filter() {
    let mut compiler = Compiler::new();
    let producer = compiler.extern_fn::<GIdsExtern>();
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
    assert_eq!(kernel(&graph()), 3 + 4 + 5);
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
