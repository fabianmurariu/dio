//! Selected problems from the 99 Lisp/Haskell list problems, implemented as
//! integration tests for the JIT compiler.
//!
//! Tests marked `#[ignore]` document features missing from rust-lms.

use rust_lms::prelude::*;

// =============================================================================
// P01: Find the last element of a list.
//
// rust-lms has no sum/option result type plumbed through `fun*` returns yet,
// so we follow the convention "last element, or 0 if the slice is empty".
// =============================================================================

#[test]
fn p99_01_last_element() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("p01_last", |ctx, arr: Var<SRef<Slice<i64>>>| {
        let n = ctx.var(0u64);
        ctx.store(n, arr.len());
        let result = ctx.var(0i64);
        ctx.if_then(gt(n, 0u64), move |ctx| {
            ctx.store(result, arr.get_unchecked(sub(n, 1u64)));
        });
        result
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(&[1, 2, 3, 4][..]), 4);
    assert_eq!(g(&[42][..]), 42);
    assert_eq!(g(&[][..]), 0);
}

// =============================================================================
// P05: Reverse a list. We reverse the slice in place using two-pointer swaps.
// =============================================================================

#[test]
fn p99_05_reverse_in_place() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("p05_reverse", |ctx, arr: Var<SRefMut<Slice<i64>>>| {
        let n = ctx.var(0u64);
        ctx.store(n, arr.len());
        // Two-pointer swap. We track `lo` and `hi+1` so unsigned arithmetic
        // never underflows (the loop exits before they cross).
        let lo = ctx.var(0u64);
        let hi_plus_1 = ctx.var(0u64);
        ctx.store(hi_plus_1, n);
        ctx.while_loop(lt(add(lo, 1u64), hi_plus_1), move |ctx| {
            let hi = ctx.var(0u64);
            ctx.store(hi, sub(hi_plus_1, 1u64));
            let a = ctx.var(0i64);
            let b = ctx.var(0i64);
            ctx.store(a, arr.get_unchecked(lo));
            ctx.store(b, arr.get_unchecked(hi));
            ctx.emit(arr.set_unchecked(lo, b));
            ctx.emit(arr.set_unchecked(hi, a));
            ctx.store(lo, add(lo, 1u64));
            ctx.store(hi_plus_1, hi);
        });
        Const::<UnitType>::new(())
    });

    let compiled = compiler.compile(f).expect("compile");
    // Note: `as_fn()` returns a fn pointer whose `&mut [i64]` parameter is
    // bound to a single lifetime, not HRTB. Calling `compiled.as_fn()(...)`
    // inline gives each call a fresh borrow.
    let mut data = vec![1i64, 2, 3, 4, 5];
    compiled.as_fn()(&mut data[..]);
    assert_eq!(data, vec![5, 4, 3, 2, 1]);

    let mut even = vec![10i64, 20, 30, 40];
    compiled.as_fn()(&mut even[..]);
    assert_eq!(even, vec![40, 30, 20, 10]);

    let mut single = vec![7i64];
    compiled.as_fn()(&mut single[..]);
    assert_eq!(single, vec![7]);

    let mut empty: Vec<i64> = vec![];
    compiled.as_fn()(&mut empty[..]);
    assert_eq!(empty, Vec::<i64>::new());
}

// =============================================================================
// P31: isPrime. Returns 1 (true) or 0 (false) since there is no boolean
// return type plumbed through `as_fn` directly.
// =============================================================================

#[test]
fn p99_31_is_prime() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("p31_isprime", |ctx, n: Var<i64>| {
        let result = ctx.var(0i64);
        ctx.if_then(gt(n, 1i64), move |ctx| {
            let prime = ctx.var(1i64);
            let d = ctx.var(2i64);
            ctx.while_loop(lt(mul(d, d), add(n, 1i64)), move |ctx| {
                ctx.if_then(eq(rem(n, d), 0i64), move |ctx| {
                    ctx.store(prime, 0i64);
                });
                ctx.store(d, add(d, 1i64));
            });
            ctx.store(result, prime);
        });
        result
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    for &(n, expected) in &[
        (-1, 0i64),
        (0, 0),
        (1, 0),
        (2, 1),
        (3, 1),
        (4, 0),
        (7, 1),
        (9, 0),
        (97, 1),
        (100, 0),
        (1_000_003, 1),
    ] {
        assert_eq!(g(n), expected, "isPrime({n}) was wrong");
    }
}

// =============================================================================
// P32: GCD via the Euclidean algorithm.
// =============================================================================

#[test]
fn p99_32_gcd() {
    let mut compiler = Compiler::new();

    let f = compiler.fun2("p32_gcd", |ctx, a: Var<i64>, b: Var<i64>| {
        let x = ctx.var(0i64);
        let y = ctx.var(0i64);
        let t = ctx.var(0i64);
        ctx.store(x, a);
        ctx.store(y, b);
        ctx.while_loop(gt(y, 0i64), move |ctx| {
            ctx.store(t, rem(x, y));
            ctx.store(x, y);
            ctx.store(y, t);
        });
        x
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(36, 63), 9);
    assert_eq!(g(123456, 7890), 6);
    assert_eq!(g(100, 0), 100);
    assert_eq!(g(0, 25), 25);
    assert_eq!(g(17, 13), 1);
}

// =============================================================================
// P34: Euler's totient φ(n) via the naive coprime count.
// =============================================================================

#[test]
fn p99_34_totient() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("p34_totient", |ctx, n: Var<i64>| {
        let result = ctx.var(0i64);
        // φ(1) = 1 by convention.
        ctx.if_then(eq(n, 1i64), move |ctx| {
            ctx.store(result, 1i64);
        });
        ctx.if_then(gt(n, 1i64), move |ctx| {
            let count = ctx.var(0i64);
            let k = ctx.var(1i64);
            ctx.while_loop(lt(k, n), move |ctx| {
                // gcd(k, n) via Euclidean
                let x = ctx.var(0i64);
                let y = ctx.var(0i64);
                let t = ctx.var(0i64);
                ctx.store(x, k);
                ctx.store(y, n);
                ctx.while_loop(gt(y, 0i64), move |ctx| {
                    ctx.store(t, rem(x, y));
                    ctx.store(x, y);
                    ctx.store(y, t);
                });
                ctx.if_then(eq(x, 1i64), move |ctx| {
                    ctx.store(count, add(count, 1i64));
                });
                ctx.store(k, add(k, 1i64));
            });
            ctx.store(result, count);
        });
        result
    });

    let compiled = compiler.compile(f).expect("compile");
    let g = compiled.as_fn();

    assert_eq!(g(1), 1);
    assert_eq!(g(10), 4); // 1, 3, 7, 9
    assert_eq!(g(9), 6);
    assert_eq!(g(13), 12); // prime
    assert_eq!(g(100), 40);
}
