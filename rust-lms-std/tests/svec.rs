//! `SVec` round-trips: a JIT kernel pushes into a host-backed vector that *grows*
//! mid-run (reallocating its buffer), and we verify the data survives — proving the
//! control-block handle indirection keeps the baked pointer valid across growth.

use rust_lms::prelude::*;
mod common;
use common::for_each_backend;

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
    for_each_backend(|mut compiler| {
        let mut host = HostVec::<i64>::new();

        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `host` owns an i64 control block and outlives compilation and
        // every call through `compiled`.
        let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

        let fill = compiler.fun1("fill", move |ctx, n: Var<u64>| {
            let i = ctx.var(0u64);
            ctx.while_loop(lt(i, n), |ctx| {
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
    });
}

/// Push `0..n`, then read them all back with `get` inside the *same* kernel and
/// return their sum — exercises `get` and confirms the data is intact after the
/// grows that happened during the push loop.
#[test]
fn get_after_grow() {
    for_each_backend(|mut compiler| {
        let mut host = HostVec::<i64>::new();

        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `host` owns an i64 control block and outlives compilation and
        // every call through `compiled`.
        let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

        let sum_fn = compiler.fun1("sum", move |ctx, n: Var<u64>| {
            // push 0..n
            let i = ctx.var(0u64);
            ctx.while_loop(lt(i, n), |ctx| {
                let v = ctx.bind(int_cast::<i64, u64, _>(i));
                svec.push(ctx, v);
                ctx.store(i, add(i, 1u64));
            });
            // sum via get
            let acc = ctx.var(0i64);
            let j = ctx.var(0u64);
            let count = svec.len(ctx);
            ctx.while_loop(lt(j, count), |ctx| {
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
    });
}

/// An `SVec` of `u64`, and `set` overwriting earlier elements — a second element
/// type plus the mutable path.
#[test]
fn set_overwrites() {
    for_each_backend(|mut compiler| {
        let mut host = HostVec::<u64>::new();

        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `host` owns a u64 control block and outlives compilation and
        // every call through `compiled`.
        let mut svec = unsafe { SVec::<u64>::new(host.handle(), grow) };

        let build = compiler.fun0("build", move |ctx| {
            // push 5 zeros
            let i = ctx.var(0u64);
            ctx.while_loop(lt(i, 5u64), |ctx| {
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
    });
}

// =============================================================================
// Borrow discipline — refactor plan row 7
// =============================================================================
//
// A view of an `SVec` and a `push` through it cannot overlap: growth may move
// the buffer. The rule is enforced by an ordinary Rust borrow — `as_slice`
// borrows the handle, `as_mut_slice` borrows it mutably, `push` takes
// `&mut self` — so violations are *compile* errors, proven by `compile_fail`
// doctests on `SVec::push`.
//
// The two cases worth exercising at runtime are the ones that must *work*.

#[test]
fn shared_views_coexist() {
    let mut host = HostVec::<i64>::new();
    let mut compiler = Compiler::new();
    let grow = compiler.extern_fn::<SvecGrowExtern>();
    // SAFETY: `host` owns the control block and outlives this staging scope.
    let svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

    let f = compiler.fun0("three_views", |ctx| {
        let a = svec.as_slice();
        let b = svec.as_slice();
        let c = a;
        // All three are live at once, like three `&[T]`.
        // `len()` here is `SliceOps::len` — the same method a function-parameter
        // slice uses, reached through the guard's `Deref`.
        ctx.bind(add(add(a.len(), b.len()), c.len()))
    });
    let compiled = compiler.compile(f).expect("compile");
    assert_eq!(compiled.call(), 0); // three views of an empty vec

    // The shared borrows ended at their last use, so a unique view is available.
    let mut host2 = HostVec::<i64>::new();
    let mut compiler2 = Compiler::new();
    let grow2 = compiler2.extern_fn::<SvecGrowExtern>();
    // SAFETY: `host2` owns the control block and outlives this scope.
    let mut svec2 = unsafe { SVec::<i64>::new(host2.handle(), grow2) };
    let _unique = svec2.as_mut_slice();
}

/// Take a view, finish with it, then grow. Non-lexical lifetimes end the borrow
/// at the view's last use, so this is accepted — the case the original spike
/// wrongly concluded was unreachable (it had made the *view itself* `Staged`,
/// which forced `ctx.bind`'s `'static` bound onto the borrow).
#[test]
fn view_released_then_grow() {
    for_each_backend(|mut compiler| {
        let mut host = HostVec::<i64>::new();
        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `host` owns the control block and outlives compilation and the call.
        let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

        let f = compiler.fun0("grow_after_view", |ctx| {
            let n = {
                let view = svec.as_slice();
                ctx.bind(view.len())
            }; // borrow ends here
            let v = ctx.bind(add(int_cast::<i64, u64, _>(n), 7i64));
            svec.push(ctx, v); // allowed again
            svec.len(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        assert_eq!(compiled.call(), 1);
        assert_eq!(host.as_slice(), &[7i64]);
    });
}

// =============================================================================
// SVec views join the common slice API — refactor plan row 8
// =============================================================================

/// Sum any trusted `i64` slice. Written once against the capability traits,
/// with no idea where its storage came from.
///
/// Binds once and reborrows rather than taking `S: Clone`: unique slice
/// expressions are deliberately not `Clone`, so a `Clone` bound would silently
/// restrict this to shared origins.
fn total<S>(ctx: &mut Ctx, s: S) -> Var<i64>
where
    S: Staged + 'static,
    S::Out: TrustedSliceType<Elem = i64>,
{
    let mut v = ctx.bind(s);
    let n = ctx.bind(v.reborrow().len());
    let acc = ctx.var(0i64);
    let i = ctx.var(0u64);
    ctx.while_loop(lt(i, n), move |ctx| {
        // SAFETY: the loop condition proves `i < len`.
        ctx.store(acc, add(acc, unsafe { v.reborrow().get_unchecked(i) }));
        ctx.store(i, add(i, 1u64));
    });
    acc
}

/// The row-8 exit criterion: the *same* helper drives a function parameter and
/// an `SVec` view, unchanged, on every backend.
#[test]
fn one_helper_serves_a_parameter_and_an_svec_view() {
    for_each_backend(|mut compiler| {
        let mut host = HostVec::<i64>::new();
        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `host` outlives compilation and the call below.
        let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

        let f = compiler.fun1("both", |ctx, arg: Var<SRef<Slice<i64>>>| {
            for k in 1..=3i64 {
                let v = ctx.var(k);
                svec.push(ctx, v);
            }
            let from_param = total(ctx, arg);
            let from_view = {
                let view = svec.as_slice();
                total(ctx, *view)
            };
            add(from_param, from_view)
        });

        let compiled = compiler.compile(f).expect("compile");
        // 30 from the parameter, 1+2+3 from the SVec view
        assert_eq!(compiled.call(&[10i64, 20]), 36);
    });
}

/// The unique half: a mutable view reaches the writing ops by the same route,
/// and the host sees the writes.
#[test]
fn mutable_view_writes_through_the_common_api() {
    for_each_backend(|mut compiler| {
        let mut host = HostVec::<i64>::new();
        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `host` outlives compilation and the call below.
        let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

        let f = compiler.fun0("fill", |ctx| {
            for _ in 0..3 {
                let z = ctx.var(0i64);
                svec.push(ctx, z);
            }
            let n = {
                let view = svec.as_mut_slice();
                let mut m = ctx.bind(*view);
                // SAFETY: three elements were pushed above.
                ctx.emit(unsafe { m.reborrow().set_unchecked(0u64, 7i64) });
                // SAFETY: as above.
                ctx.emit(unsafe { m.reborrow().set_unchecked(2u64, 9i64) });
                ctx.bind(m.len())
            };
            n
        });

        let compiled = compiler.compile(f).expect("compile");
        assert_eq!(compiled.call(), 3);
        assert_eq!(host.as_slice(), &[7i64, 0, 9]);
    });
}

/// A view is *reloading*: an expression copied out of a guard, used after a
/// growth has been emitted, observes the new buffer rather than a stale
/// pointer. This is what makes the `Copy`, lifetime-free deref target safe.
#[test]
fn view_expression_reloads_after_growth() {
    for_each_backend(|mut compiler| {
        let mut host = HostVec::<i64>::new();
        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `host` outlives compilation and the call below.
        let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

        let f = compiler.fun0("reload", |ctx| {
            // One element, then copy the expression out and drop the guard.
            let first = ctx.var(1i64);
            svec.push(ctx, first);
            let expr = {
                let view = svec.as_slice();
                *view
            };
            // Force several growths (capacity starts at 4).
            for k in 2..=8i64 {
                let v = ctx.var(k);
                svec.push(ctx, v);
            }
            // The copied expression re-reads `(ptr, len)`, so it sees all 8.
            total(ctx, expr)
        });

        let compiled = compiler.compile(f).expect("compile");
        assert_eq!(compiled.call(), 36); // 1+2+…+8
        assert_eq!(host.len(), 8);
    });
}

/// **G10, `SVec` half.** A view iterates through exactly the same
/// `staged_iter` a function parameter uses — no `SVec`-specific iterator.
#[test]
fn svec_view_iterates() {
    for_each_backend(|mut compiler| {
        let mut host = HostVec::<i64>::new();
        let grow = compiler.extern_fn::<SvecGrowExtern>();
        // SAFETY: `host` outlives compilation and the call below.
        let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };

        let f = compiler.fun0("sum_view", |ctx| {
            for k in 1..=5i64 {
                let v = ctx.var(k);
                svec.push(ctx, v);
            }
            let view = svec.as_slice();
            (*view).staged_iter().sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile");
        assert_eq!(compiled.call(), 15); // 1+2+3+4+5
    });
}
