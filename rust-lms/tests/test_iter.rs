//! Integration tests for the staged iterator API: sum, count, min, max, fold, zip.
//! All use the imperative Ctx API: ctx.var(), ctx.assign(), etc.

use rust_lms::prelude::*;

// =============================================================================
// sum
// =============================================================================

#[test]
fn test_iter_sum_i64() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("sum_i64", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().sum(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let sum = compiled.as_fn();

    let data: [i64; 5] = [10, 20, 30, 40, 50];
    assert_eq!(sum(&data[..]), 150);
}

#[test]
fn test_iter_sum_f64() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("sum_f64", |ctx, arr: Var<SRef<Slice<F64Type>>>| {
        arr.staged_iter().sum(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let sum = compiled.as_fn();

    let data: [f64; 4] = [1.5, 2.5, 3.0, 4.0];
    assert!((sum(&data[..]) - 11.0).abs() < 1e-9);
}

#[test]
fn test_iter_sum_with_map() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("doubled_sum", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().map(|x| x * 2i64).sum(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let data: [i64; 4] = [1, 2, 3, 4]; // 2+4+6+8 = 20
    assert_eq!(f(&data[..]), 20);
}

#[test]
fn test_iter_sum_with_filter() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("positive_sum", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().filter(|x| lt(0i64, x)).sum(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let data: [i64; 6] = [-3, 5, -1, 8, 0, 2];
    assert_eq!(f(&data[..]), 15); // 5+8+2
}

// =============================================================================
// count
// =============================================================================

#[test]
fn test_iter_count_all() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("count_all", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().count(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let data: [i64; 7] = [1, 2, 3, 4, 5, 6, 7];
    assert_eq!(f(&data[..]), 7u64);
}

#[test]
fn test_iter_count_filtered() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("count_gt3", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().filter(|x| lt(3i64, x)).count(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let data: [i64; 5] = [1, 4, 5, 2, 6];
    assert_eq!(f(&data[..]), 3u64); // 4, 5, 6
}

// =============================================================================
// min / max
// =============================================================================

#[test]
fn test_iter_min_i64() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("min_i64", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().min(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let data: [i64; 5] = [30, 10, 50, 20, 40];
    assert_eq!(f(&data[..]), 10);
}

#[test]
fn test_iter_max_f64() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("max_f64", |ctx, arr: Var<SRef<Slice<F64Type>>>| {
        arr.staged_iter().max(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let data: [f64; 5] = [1.5, 9.9, 3.3, 7.7, 2.2];
    assert!((f(&data[..]) - 9.9).abs() < 1e-9);
}

#[test]
fn test_iter_min_max_filtered() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("min_positive", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().filter(|x| lt(0i64, x)).min(ctx)
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let data: [i64; 6] = [-5, 3, -1, 7, 2, 9];
    assert_eq!(f(&data[..]), 2); // min of {3, 7, 2, 9}
}

// =============================================================================
// fold (multi-accumulator via user-managed vars)
// =============================================================================

#[test]
fn test_iter_fold_count_and_sum() {
    let mut compiler = Compiler::new();

    let f = compiler.fun1("count_and_sum", |ctx, arr: Var<SRef<Slice<F64Type>>>| {
        // Declare accumulator vars BEFORE fold
        let count = ctx.var(0u64);
        let sum = ctx.var(0.0f64);

        // fold uses user-managed vars — no Accumulator trait needed
        arr.staged_iter()
            .fold(ctx, (count, sum), |ctx, (c, s), elem| {
                ctx.store(c, c + 1u64);
                ctx.store(s, s + elem);
            });

        count // return count as the function result
    });

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    assert_eq!(f(&data[..]), 4u64);
}

// =============================================================================
// zip
// =============================================================================

#[test]
fn test_iter_zip_dot_product() {
    let mut compiler = Compiler::new();

    // Dot product: sum of a[i] * b[i]
    let f = compiler.fun2(
        "dot_product",
        |ctx, a: Var<SRef<Slice<F64Type>>>, b: Var<SRef<Slice<F64Type>>>| {
            let acc = ctx.var(0.0f64);

            a.staged_iter().zip(b).for_each(ctx, move |ctx, ai, bi| {
                ctx.store(acc, acc + ai * bi);
            });

            acc
        },
    );

    let compiled = compiler.compile(f).expect("compile failed");
    let dot = compiled.as_fn();

    let a: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
    let b: [f64; 4] = [4.0, 3.0, 2.0, 1.0];
    // dot = 1*4 + 2*3 + 3*2 + 4*1 = 20
    assert!((dot(&a[..], &b[..]) - 20.0).abs() < 1e-9);
}

#[test]
fn test_iter_zip_element_wise_sum() {
    let mut compiler = Compiler::new();

    // sum of (a[i] + b[i]) for all i
    let f = compiler.fun2(
        "zip_sum",
        |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
            let total = ctx.var(0i64);

            a.staged_iter().zip(b).for_each(ctx, move |ctx, ai, bi| {
                ctx.store(total, total + ai + bi);
            });

            total
        },
    );

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    let a: [i64; 4] = [1, 2, 3, 4];
    let b: [i64; 4] = [10, 20, 30, 40];
    // (1+10)+(2+20)+(3+30)+(4+40) = 11+22+33+44 = 110
    assert_eq!(f(&a[..], &b[..]), 110);
}

// =============================================================================
// range iterator with sum
// =============================================================================

#[test]
fn test_range_sum() {
    let mut compiler = Compiler::new();

    // Sum of range [0, n)
    let f = compiler.fun1("range_sum", |ctx, n: Var<U64Type>| range(0u64, n).sum(ctx));

    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();

    assert_eq!(f(10u64), 45u64); // 0+1+...+9 = 45
    assert_eq!(f(5u64), 10u64); // 0+1+2+3+4 = 10
}

#[test]
fn test_range_step_sum() {
    let mut compiler = Compiler::new();
    // Sum of [0, n) stepping by 2: 0 + 2 + 4 + ...
    let f = compiler.fun1("range_step_sum", |ctx, n: Var<U64Type>| {
        range_step(0u64, n, 2u64).sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    assert_eq!(f(10u64), 20u64); // 0+2+4+6+8
    assert_eq!(f(9u64), 20u64); // 0+2+4+6+8
}

#[test]
fn test_range_i64_sum() {
    let mut compiler = Compiler::new();
    // i64 range works as a StagedIterator (no zip/len, but sum/fold do).
    let f = compiler.fun1("range_i64_sum", |ctx, n: Var<i64>| range(0i64, n).sum(ctx));
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    assert_eq!(f(5i64), 10i64); // 0+1+2+3+4
}

#[test]
fn test_range_into_staged_iter() {
    let mut compiler = Compiler::new();
    let f = compiler.fun1("range_into_iter", |ctx, n: Var<U64Type>| {
        range(0u64, n).into_staged_iter().map(|x| x * 3u64).sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    assert_eq!(f(4u64), 18u64); // 3*(0+1+2+3)
}

#[test]
fn test_iter_any() {
    let mut compiler = Compiler::new();
    let f = compiler.fun1("any_gt_4", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().any(ctx, |x| gt(x, 4i64))
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    assert!(f(&[1i64, 2, 3, 5][..])); // 5 > 4
    assert!(!f(&[1i64, 2, 3, 4][..])); // none > 4
    assert!(!f(&[][..]));
}

#[test]
fn test_iter_all() {
    let mut compiler = Compiler::new();
    let f = compiler.fun1("all_positive", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().all(ctx, |x| gt(x, 0i64))
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    assert!(f(&[1i64, 2, 3][..]));
    assert!(!f(&[1i64, -2, 3][..]));
    assert!(f(&[][..])); // vacuously true
}

#[test]
fn test_iter_position() {
    let mut compiler = Compiler::new();
    let f = compiler.fun1("pos_of_3", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().position(ctx, |x| eq(x, 3i64))
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    assert_eq!(f(&[10i64, 20, 3, 40][..]), 2); // index 2
    assert_eq!(f(&[10i64, 20][..]), 2); // not found -> len (2)
}

#[test]
fn test_iter_any_after_filter() {
    // Early exit now composes after a combinator (was indexed-only before).
    let mut compiler = Compiler::new();
    let f = compiler.fun1("any_even_gt_4", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter()
            .filter(|x| eq(x % 2i64, 0i64))
            .any(ctx, |x| gt(x, 4i64))
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    assert!(f(&[1i64, 3, 6][..])); // 6 is even and > 4
    assert!(!f(&[1i64, 3, 4, 5][..])); // only even is 4, not > 4
}

#[test]
fn test_iter_position_after_map() {
    let mut compiler = Compiler::new();
    let f = compiler.fun1("pos_after_map", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter()
            .map(|x| x * 2i64)
            .position(ctx, |x| eq(x, 6i64))
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    assert_eq!(f(&[1i64, 2, 3, 4][..]), 2); // 3*2 == 6 at index 2
}

#[test]
fn test_iter_scan_prefix_sum() {
    // Running (prefix) sum via scan, then total via sum of the running values'
    // last == grand total; here we just sum the prefix sums for a check value.
    let mut compiler = Compiler::new();
    let f = compiler.fun1("scan_prefix", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter()
            .scan(0i64, |ctx, acc, x| ctx.store(acc, acc + x))
            .sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    // prefix sums of [1,2,3,4] = [1,3,6,10]; their sum = 20
    assert_eq!(f(&[1i64, 2, 3, 4][..]), 20);
}

#[test]
fn test_iter_take_while() {
    let mut compiler = Compiler::new();
    let f = compiler.fun1("take_while_lt5", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().take_while(|x| lt(x, 5i64)).sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    // takes 1,2,3 then stops at 5; sum = 6 (the 4 after 5 is not reached)
    assert_eq!(f(&[1i64, 2, 3, 5, 4][..]), 6);
}

#[test]
fn test_iter_nested_break() {
    // Count elements of `a` that appear in `b`. The inner `any` builds its own
    // loop and breaks out of it; the outer loop must continue unaffected.
    let mut compiler = Compiler::new();
    let f = compiler.fun2(
        "count_in_both",
        |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
            let count = ctx.var(0u64);
            a.staged_iter().for_each(ctx, move |ctx, ai| {
                let present = b.staged_iter().any(ctx, move |bj| eq(bj, ai));
                ctx.if_then(present, move |ctx| {
                    ctx.store(count, count + 1u64);
                });
            });
            count
        },
    );
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    let a: [i64; 4] = [1, 2, 3, 7];
    let b: [i64; 3] = [2, 3, 4];
    assert_eq!(f(&a[..], &b[..]), 2); // 2 and 3 are in both
}

#[test]
fn test_iter_skip_while() {
    let mut compiler = Compiler::new();
    let f = compiler.fun1("skip_while_lt5", |ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.staged_iter().skip_while(|x| lt(x, 5i64)).sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile failed");
    let f = compiled.as_fn();
    // skips 1,2,3; yields 5,4,6 from the first >=5; sum = 15
    assert_eq!(f(&[1i64, 2, 3, 5, 4, 6][..]), 15);
}
