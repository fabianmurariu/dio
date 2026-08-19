//! `SVec` round-trips: a JIT kernel pushes into a host-backed vector that *grows*
//! mid-run (reallocating its buffer), and we verify the data survives — proving the
//! control-block handle indirection keeps the baked pointer valid across growth.

use rust_lms::prelude::*;
use rust_lms_std::{HostVec, SVec, SvecGrowExtern};

#[test]
#[should_panic(expected = "HostVec does not support zero-sized element types")]
fn zero_sized_elements_are_rejected() {
    let _ = HostVec::<()>::new();
}

/// Push `0,10,20,…` for `i in 0..n` into an `SVec`, forcing several grows
/// (`cap` 0→4→8→16), then check the host reads back every value and the kernel
/// returns the right length.
#[test]
fn push_grows_and_reads_back() {
    let mut host = HostVec::<i64>::new();

    let mut compiler = Compiler::new();
    let grow = compiler.extern_fn::<SvecGrowExtern>();
    // SAFETY: `host` owns an i64 control block and outlives compilation and
    // every call through `compiled`.
    let svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

    let fill = compiler.fun1("fill", move |ctx, n: Var<u64>| {
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, n), move |ctx| {
            let v = ctx.bind(mul(int_cast::<i64, u64, _>(i), 10i64));
            svec.push(ctx, v);
            ctx.store(i, add(i, 1u64));
        });
        svec.len(ctx)
    });
    let compiled = compiler.compile(fill).expect("compile");

    let len = compiled.call(10);
    assert_eq!(len, 10);
    assert_eq!(host.len(), 10);
    assert_eq!(host.as_slice(), &[0, 10, 20, 30, 40, 50, 60, 70, 80, 90]);
}

/// Push `0..n`, then read them all back with `get` inside the *same* kernel and
/// return their sum — exercises `get` and confirms the data is intact after the
/// grows that happened during the push loop.
#[test]
fn get_after_grow() {
    let mut host = HostVec::<i64>::new();

    let mut compiler = Compiler::new();
    let grow = compiler.extern_fn::<SvecGrowExtern>();
    // SAFETY: `host` owns an i64 control block and outlives compilation and
    // every call through `compiled`.
    let svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

    let sum_fn = compiler.fun1("sum", move |ctx, n: Var<u64>| {
        // push 0..n
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, n), move |ctx| {
            let v = ctx.bind(int_cast::<i64, u64, _>(i));
            svec.push(ctx, v);
            ctx.store(i, add(i, 1u64));
        });
        // sum via get
        let acc = ctx.var(0i64);
        let j = ctx.var(0u64);
        let count = svec.len(ctx);
        ctx.while_loop(lt(j, count), move |ctx| {
            // SAFETY: the loop condition proves `j < count == svec.len()`.
            let e = unsafe { svec.get(ctx, j) };
            ctx.store(acc, add(acc, e));
            ctx.store(j, add(j, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(sum_fn).expect("compile");

    // 0+1+…+99 = 4950
    let sum = compiled.call(100);
    assert_eq!(sum, 4950);
    assert_eq!(host.len(), 100);
}

/// An `SVec` of `u64`, and `set` overwriting earlier elements — a second element
/// type plus the mutable path.
#[test]
fn set_overwrites() {
    let mut host = HostVec::<u64>::new();

    let mut compiler = Compiler::new();
    let grow = compiler.extern_fn::<SvecGrowExtern>();
    // SAFETY: `host` owns a u64 control block and outlives compilation and
    // every call through `compiled`.
    let svec = unsafe { SVec::<u64>::new(host.handle(), grow) };

    let build = compiler.fun0("build", move |ctx| {
        // push 5 zeros
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, 5u64), move |ctx| {
            let zero = ctx.var(0u64);
            svec.push(ctx, zero);
            ctx.store(i, add(i, 1u64));
        });
        // set[2] = 42, set[4] = 7
        let (two, forty_two) = (ctx.var(2u64), ctx.var(42u64));
        // SAFETY: five initialized elements were pushed above.
        unsafe { svec.set(ctx, two, forty_two) };
        let (four, seven) = (ctx.var(4u64), ctx.var(7u64));
        // SAFETY: five initialized elements were pushed above.
        unsafe { svec.set(ctx, four, seven) };
        svec.len(ctx)
    });
    let compiled = compiler.compile(build).expect("compile");

    let len = compiled.call();
    assert_eq!(len, 5);
    assert_eq!(host.as_slice(), &[0, 0, 42, 0, 7]);
}
