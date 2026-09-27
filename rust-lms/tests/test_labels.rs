//! Scoped labels: `block`/`join`/`repeat`/`iterate` and their jumps
//! (`exit`/`goto`/`again`). See `docs/pull_iter.md` §4.1 for the model: they are
//! Rust's labelled `break`/`continue`, and only `iterate` is a loop scope.

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

// =============================================================================
// join + goto: several exits, one merged value
// =============================================================================

#[test]
fn join_merges_gotos_and_fall_through() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("sign", |ctx, x: Var<i64>| {
            ctx.join(|ctx, out| {
                ctx.if_then(lt(x, 0i64), |ctx| ctx.goto(out, -1i64));
                ctx.if_then(gt(x, 0i64), |ctx| ctx.goto(out, 1i64));
                0i64
            })
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let sign = compiled.as_fn();
        assert_eq!(sign.call(-5), -1);
        assert_eq!(sign.call(0), 0);
        assert_eq!(sign.call(7), 1);
    });
}

#[test]
fn join_carries_a_fat_value() {
    // The merged value is a slice: two block parameters (ptr, len).
    for_each_backend(|mut compiler| {
        let f = compiler.fun3(
            "pick_sum",
            |ctx, pick_a: Var<bool>, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
                let s = ctx.join(|ctx, out| {
                    ctx.if_then(pick_a, move |ctx| ctx.goto(out, a));
                    b
                });
                s.staged_iter().sum(ctx)
            },
        );
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        let (a, b) = ([1i64, 2, 3], [10i64, 20]);
        assert_eq!(f.call(true, &a[..], &b[..]), 6);
        assert_eq!(f.call(false, &a[..], &b[..]), 30);
    });
}

// =============================================================================
// block + exit: a value-less forward label, exited from deep nesting
// =============================================================================

#[test]
fn block_exit_skips_the_rest_of_the_block() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("classify", |ctx, x: Var<i64>| {
            let r = ctx.var(0i64);
            ctx.block(|ctx, out| {
                ctx.if_then(gt(x, 10i64), |ctx| {
                    ctx.if_then(gt(x, 100i64), |ctx| {
                        ctx.store(r, 2i64);
                        ctx.exit(out);
                    });
                    ctx.store(r, 1i64);
                    ctx.exit(out);
                });
                ctx.store(r, -1i64);
            });
            r
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(5), -1);
        assert_eq!(f.call(50), 1);
        assert_eq!(f.call(500), 2);
    });
}

// =============================================================================
// iterate + exit_if: the iteration loop
// =============================================================================

#[test]
fn iterate_runs_until_exit() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("sum_below", |ctx, n: Var<u64>| {
            let i = ctx.var(0u64);
            let acc = ctx.var(0u64);
            ctx.iterate(|ctx, done| {
                ctx.exit_if(ge(i, n), done);
                ctx.store(acc, acc + i);
                ctx.store(i, i + 1u64);
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(0), 0);
        assert_eq!(f.call(5), 10);
    });
}

// =============================================================================
// repeat + again: a retry point that is not a loop scope
// =============================================================================

#[test]
fn repeat_retries_until_accepted() {
    // Collatz steps: `repeat` re-runs its body until the value reaches 1.
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("collatz", |ctx, x: Var<u64>| {
            let v = ctx.var(x);
            let steps = ctx.var(0u64);
            ctx.repeat(|ctx, again| {
                ctx.if_then(ne(v, 1u64), |ctx| {
                    ctx.if_then_else(
                        eq(rem(v, 2u64), 0u64),
                        |ctx| ctx.store(v, div(v, 2u64)),
                        |ctx| ctx.store(v, v * 3u64 + 1u64),
                    );
                    ctx.store(steps, steps + 1u64);
                    ctx.again(again);
                });
            });
            steps
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(1), 0);
        assert_eq!(f.call(6), 8);
        assert_eq!(f.call(27), 111);
    });
}

#[test]
fn break_loop_passes_through_repeat() {
    // `slice.filter(p).any(q)` written by hand (docs/pull_iter.md §4.1): the
    // retry lives in a `repeat`, and `break_loop` from the consumer must leave
    // the whole iteration, not just the retry. `visited` proves the early exit.
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "filter_any",
            |ctx, s: Var<SRef<Slice<i64>>>, out_visited: Var<SRefMut<u64>>| {
                let n = ctx.bind(s.len());
                let i = ctx.var(0u64);
                let found = ctx.var(false);
                let visited = ctx.var(0u64);
                ctx.iterate(|ctx, done| {
                    // Filter::next: keep even elements.
                    let x = ctx.repeat(|ctx, again| {
                        ctx.exit_if(ge(i, n), done);
                        // SAFETY: i < n was checked just above.
                        let x = ctx.bind(unsafe { s.get_unchecked(i) });
                        ctx.store(i, i + 1u64);
                        ctx.store(visited, visited + 1u64);
                        ctx.again_if(ne(rem(x, 2i64), 0i64), again);
                        x
                    });
                    // any's consumer: an even element greater than 10.
                    ctx.if_then(gt(x, 10i64), |ctx| {
                        ctx.store(found, true);
                        ctx.break_loop();
                    });
                });
                ctx.emit(store_ref(out_visited, visited));
                found
            },
        );
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let mut visited = 0u64;
        // 13 is odd (retried), 12 is the first hit: 5 elements visited, not 7.
        let data = [1i64, 4, 3, 13, 12, 14, 7];
        assert!(f.call(&data[..], &mut visited));
        assert_eq!(visited, 5);

        let data = [1i64, 4, 3, 13];
        assert!(!f.call(&data[..], &mut visited));
        assert_eq!(visited, 4);
    });
}

// =============================================================================
// chain, by hand: a private "done" label per side, one consumer
// =============================================================================

#[test]
fn hand_written_chain_emits_the_consumer_once() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "chain_sum",
            |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
                let (na, nb) = (ctx.bind(a.len()), ctx.bind(b.len()));
                let (i, j) = (ctx.var(0u64), ctx.var(0u64));
                let in_b = ctx.var(false);
                let acc = ctx.var(0i64);
                ctx.iterate(|ctx, done| {
                    let x = ctx.join(|ctx, got| {
                        ctx.if_then(not(in_b), |ctx| {
                            ctx.block(|ctx, a_done| {
                                ctx.exit_if(ge(i, na), a_done);
                                // SAFETY: i < na.
                                let x = ctx.bind(unsafe { a.get_unchecked(i) });
                                ctx.store(i, i + 1u64);
                                ctx.goto(got, x);
                            });
                            ctx.store(in_b, true);
                        });
                        ctx.exit_if(ge(j, nb), done);
                        // SAFETY: j < nb.
                        let y = ctx.bind(unsafe { b.get_unchecked(j) });
                        ctx.store(j, j + 1u64);
                        y
                    });
                    // The consumer, emitted once for both sides.
                    ctx.store(acc, acc * 10i64 + x);
                });
                acc
            },
        );
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1i64, 2][..], &[3i64, 4, 5][..]), 12345);
        assert_eq!(f.call(&[][..], &[7i64][..]), 7);
        assert_eq!(f.call(&[8i64][..], &[][..]), 8);
    });
}

#[test]
fn nested_iterate_scopes_are_independent() {
    // An inner `iterate`'s `done` and `break_loop` leave only the inner loop.
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("tri", |ctx, n: Var<u64>| {
            let i = ctx.var(0u64);
            let acc = ctx.var(0u64);
            ctx.iterate(|ctx, outer_done| {
                ctx.exit_if(ge(i, n), outer_done);
                let j = ctx.var(0u64);
                ctx.iterate(|ctx, inner_done| {
                    ctx.exit_if(gt(j, i), inner_done);
                    ctx.store(acc, acc + 1u64);
                    ctx.store(j, j + 1u64);
                });
                ctx.store(i, i + 1u64);
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        // sum_{i<4} (i+1) = 10
        assert_eq!(f.call(4), 10);
    });
}
