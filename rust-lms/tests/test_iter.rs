//! Integration tests for the staged iterator API: sum, count, min, max, fold, zip.
//! All use the imperative Ctx API: ctx.var(), ctx.assign(), etc.

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

// =============================================================================
// sum
// =============================================================================

#[test]
fn test_iter_sum_i64() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("sum_i64", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().sum(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let sum = compiled.as_fn();

        let data: [i64; 5] = [10, 20, 30, 40, 50];
        assert_eq!(sum.call(&data[..]), 150);
    });
}

#[test]
fn test_iter_sum_f64() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("sum_f64", |ctx, arr: Var<SRef<Slice<f64>>>| {
            arr.staged_iter().sum(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let sum = compiled.as_fn();

        let data: [f64; 4] = [1.5, 2.5, 3.0, 4.0];
        assert!((sum.call(&data[..]) - 11.0).abs() < 1e-9);
    });
}

#[test]
fn test_iter_sum_with_map() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("doubled_sum", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().map(|x| x * 2i64).sum(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let data: [i64; 4] = [1, 2, 3, 4]; // 2+4+6+8 = 20
        assert_eq!(f.call(&data[..]), 20);
    });
}

#[test]
fn test_iter_sum_with_filter() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("positive_sum", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().filter(|x| lt(0i64, x)).sum(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let data: [i64; 6] = [-3, 5, -1, 8, 0, 2];
        assert_eq!(f.call(&data[..]), 15); // 5+8+2
    });
}

// =============================================================================
// count
// =============================================================================

#[test]
fn test_iter_count_all() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("count_all", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().count(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let data: [i64; 7] = [1, 2, 3, 4, 5, 6, 7];
        assert_eq!(f.call(&data[..]), 7u64);
    });
}

#[test]
fn test_iter_count_filtered() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("count_gt3", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().filter(|x| lt(3i64, x)).count(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let data: [i64; 5] = [1, 4, 5, 2, 6];
        assert_eq!(f.call(&data[..]), 3u64); // 4, 5, 6
    });
}

// =============================================================================
// min / max
// =============================================================================

#[test]
fn test_iter_min_i64() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("min_i64", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().min(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let data: [i64; 5] = [30, 10, 50, 20, 40];
        assert_eq!(f.call(&data[..]), 10);
    });
}

#[test]
fn test_iter_max_f64() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("max_f64", |ctx, arr: Var<SRef<Slice<f64>>>| {
            arr.staged_iter().max(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let data: [f64; 5] = [1.5, 9.9, 3.3, 7.7, 2.2];
        assert!((f.call(&data[..]) - 9.9).abs() < 1e-9);
    });
}

#[test]
fn test_iter_min_max_filtered() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("min_positive", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().filter(|x| lt(0i64, x)).min(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let data: [i64; 6] = [-5, 3, -1, 7, 2, 9];
        assert_eq!(f.call(&data[..]), 2); // min of {3, 7, 2, 9}
    });
}

// =============================================================================
// fold / fold_if
// =============================================================================

#[test]
fn test_iter_fold_returns_the_accumulator() {
    for_each_backend(|mut compiler| {
        // Horner: digits → number.
        let f = compiler.fun1("horner", |ctx, arr: Var<SRef<Slice<u64>>>| {
            arr.staged_iter().fold(ctx, 0u64, |acc, d| acc * 10u64 + d)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1u64, 2, 3][..]), 123);
        assert_eq!(f.call(&[][..]), 0);
    });
}

#[test]
fn test_iter_fold_with_a_bool_accumulator() {
    // The accumulator need not be numeric: "is the slice sorted?".
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("sorted", |ctx, arr: Var<SRef<Slice<i64>>>| {
            let prev = ctx.var(i64::MIN);
            arr.staged_iter()
                .scan(true, move |ctx, ok, x| {
                    ctx.store(ok, select(gt(prev, x), false, ok));
                    ctx.store(prev, x);
                })
                .fold(ctx, true, |_, ok| ok)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert!(f.call(&[1i64, 2, 2, 5][..]));
        assert!(!f.call(&[1i64, 3, 2][..]));
        assert!(f.call(&[][..]));
    });
}

#[test]
fn test_iter_fold_if_matches_filter_fold() {
    for_each_backend(|mut compiler| {
        let branchless = compiler.fun1("fold_if", |ctx, arr: Var<SRef<Slice<f64>>>| {
            arr.staged_iter()
                .fold_if(ctx, 0.0f64, |x| gt(x, 0.0f64), |acc, x| acc + x * x)
        });
        let compiled = compiler.compile(branchless).expect("compile failed");
        let data = [1.5f64, -2.0, 3.0, -0.5, 2.0];
        let expected: f64 = data.iter().filter(|&&x| x > 0.0).map(|x| x * x).sum();
        assert!((compiled.as_fn().call(&data[..]) - expected).abs() < 1e-12);
    });
}

// =============================================================================
// several accumulators: for_each over vars declared before the loop
// =============================================================================

#[test]
fn test_iter_for_each_count_and_sum() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("count_and_sum", |ctx, arr: Var<SRef<Slice<f64>>>| {
            // Several accumulators at once: vars declared before the loop,
            // updated by `for_each`.
            let count = ctx.var(0u64);
            let sum = ctx.var(0.0f64);
            arr.staged_iter().for_each(ctx, move |ctx, elem| {
                ctx.store(count, count + 1u64);
                ctx.store(sum, sum + elem);
            });

            count // return count as the function result
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let data: [f64; 4] = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(f.call(&data[..]), 4u64);
    });
}

// =============================================================================
// zip
// =============================================================================

#[test]
fn test_iter_zip_dot_product() {
    for_each_backend(|mut compiler| {
        // Dot product: sum of a[i] * b[i]
        let f = compiler.fun2(
            "dot_product",
            |ctx, a: Var<SRef<Slice<f64>>>, b: Var<SRef<Slice<f64>>>| {
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
        assert!((dot.call(&a[..], &b[..]) - 20.0).abs() < 1e-9);
    });
}

#[test]
fn test_iter_zip_stops_at_shorter_secondary() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "zip_short_secondary",
            |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
                let total = ctx.var(0i64);
                a.staged_iter().zip(b).for_each(ctx, move |ctx, ai, bi| {
                    ctx.store(total, total + ai + bi);
                });
                total
            },
        );
        let compiled = compiler.compile(f).expect("compile failed");

        assert_eq!(compiled.call(&[1, 2, 3, 4][..], &[10, 20][..]), 33);
    });
}

#[test]
fn test_iter_zip_stops_at_shorter_primary_through_combinator() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "zip_short_primary",
            |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
                a.staged_iter()
                    .zip(b)
                    .map(|pair| pair.first() * pair.second())
                    .sum(ctx)
            },
        );
        let compiled = compiler.compile(f).expect("compile failed");

        assert_eq!(compiled.call(&[2, 3][..], &[10, 20, 30, 40][..]), 80);
    });
}

#[test]
fn test_iter_zip_element_wise_sum() {
    for_each_backend(|mut compiler| {
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
        assert_eq!(f.call(&a[..], &b[..]), 110);
    });
}

#[test]
fn test_iter_zip_composes_with_map_and_sum() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "zip_map_sum",
            |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
                a.staged_iter()
                    .zip(b)
                    .map(|pair| pair.first() * pair.second())
                    .sum(ctx)
            },
        );

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let a: [i64; 4] = [1, 2, 3, 4];
        let b: [i64; 4] = [10, 20, 30, 40];
        assert_eq!(f.call(&a[..], &b[..]), 300);
    });
}

#[test]
fn test_iter_zip_for_each_yields_pair_item() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "zip_pair_for_each",
            |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>| {
                let total = ctx.var(0i64);

                StagedIterator::for_each(a.staged_iter().zip(b), ctx, move |ctx, pair| {
                    ctx.store(total, total + pair.first() + pair.second());
                });

                total
            },
        );

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let a: [i64; 3] = [1, 2, 3];
        let b: [i64; 3] = [10, 20, 30];
        assert_eq!(f.call(&a[..], &b[..]), 66);
    });
}

#[test]
fn test_iter_zip_is_indexed_source_for_nested_zip() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun3(
            "nested_zip_sum",
            |ctx, a: Var<SRef<Slice<i64>>>, b: Var<SRef<Slice<i64>>>, c: Var<SRef<Slice<i64>>>| {
                let total = ctx.var(0i64);

                a.staged_iter()
                    .zip(b)
                    .zip(c)
                    .for_each(ctx, move |ctx, ab, c_value| {
                        let a_value = ctx.bind(ab.first());
                        let b_value = ctx.bind(ab.second());
                        ctx.store(total, total + a_value + b_value + c_value);
                    });

                total
            },
        );

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        let a: [i64; 3] = [1, 2, 3];
        let b: [i64; 3] = [10, 20, 30];
        let c: [i64; 3] = [100, 200, 300];
        assert_eq!(f.call(&a[..], &b[..], &c[..]), 666);
    });
}

// =============================================================================
// range iterator with sum
// =============================================================================

#[test]
fn test_range_sum() {
    for_each_backend(|mut compiler| {
        // Sum of range [0, n)
        let f = compiler.fun1("range_sum", |ctx, n: Var<u64>| range(0u64, n).sum(ctx));

        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(10u64), 45u64); // 0+1+...+9 = 45
        assert_eq!(f.call(5u64), 10u64); // 0+1+2+3+4 = 10
    });
}

#[test]
fn test_range_step_sum() {
    for_each_backend(|mut compiler| {
        // Sum of [0, n) stepping by 2: 0 + 2 + 4 + ...
        let f = compiler.fun1("range_step_sum", |ctx, n: Var<u64>| {
            range_step(0u64, n, 2u64).sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(10u64), 20u64); // 0+2+4+6+8
        assert_eq!(f.call(9u64), 20u64); // 0+2+4+6+8
    });
}

#[test]
fn test_range_i64_sum() {
    for_each_backend(|mut compiler| {
        // i64 range works as a StagedIterator (no zip/len, but sum/fold do).
        let f = compiler.fun1("range_i64_sum", |ctx, n: Var<i64>| range(0i64, n).sum(ctx));
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(5i64), 10i64); // 0+1+2+3+4
    });
}

#[test]
fn test_range_into_staged_iter() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("range_into_iter", |ctx, n: Var<u64>| {
            range(0u64, n).into_staged_iter().map(|x| x * 3u64).sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(4u64), 18u64); // 3*(0+1+2+3)
    });
}

#[test]
fn test_iter_filter_map() {
    for_each_backend(|mut compiler| {
        // keep evens, map to x*10, sum: [1,2,3,4] -> 2,4 -> 20,40 -> 60
        let f = compiler.fun1("fm_sum", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter()
                .filter_map(|x| eq(x % 2i64, 0i64).then_some(x * 10i64))
                .sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1i64, 2, 3, 4][..]), 60);
        assert_eq!(f.call(&[1i64, 3, 5][..]), 0); // no evens
    });
}

#[test]
fn test_iter_find_map() {
    for_each_backend(|mut compiler| {
        // first element > 3, mapped to x*10; -1 if none found
        let f = compiler.fun1("find_map", |ctx, arr: Var<SRef<Slice<i64>>>| {
            let (val, found) = arr
                .staged_iter()
                .find_map(ctx, |x| gt(x, 3i64).then_some(x * 10i64));
            select(found, val, -1i64)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1i64, 2, 5, 4][..]), 50); // first > 3 is 5 -> 50
        assert_eq!(f.call(&[1i64, 2, 3][..]), -1); // none > 3
    });
}

#[test]
fn test_iter_count_if() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("count_gt_3", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().count_if(ctx, |x| gt(x, 3i64))
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1i64, 4, 5, 2, 6][..]), 3u64); // 4,5,6
        assert_eq!(f.call(&[][..]), 0u64);
    });
}

#[test]
fn test_iter_sum_if() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("sum_even", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().sum_if(ctx, |x| eq(x % 2i64, 0i64))
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1i64, 2, 3, 4, 5, 6][..]), 12); // 2+4+6
    });
}

#[test]
fn test_iter_any() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("any_gt_4", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().any(ctx, |x| gt(x, 4i64))
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert!(f.call(&[1i64, 2, 3, 5][..])); // 5 > 4
        assert!(!f.call(&[1i64, 2, 3, 4][..])); // none > 4
        assert!(!f.call(&[][..]));
    });
}

#[test]
fn test_iter_all() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("all_positive", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().all(ctx, |x| gt(x, 0i64))
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert!(f.call(&[1i64, 2, 3][..]));
        assert!(!f.call(&[1i64, -2, 3][..]));
        assert!(f.call(&[][..])); // vacuously true
    });
}

#[test]
fn test_iter_position() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("pos_of_3", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().position(ctx, |x| eq(x, 3i64))
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[10i64, 20, 3, 40][..]), 2); // index 2
        assert_eq!(f.call(&[10i64, 20][..]), 2); // not found -> len (2)
    });
}

#[test]
fn test_iter_any_after_filter() {
    // Early exit now composes after a combinator (was indexed-only before).
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("any_even_gt_4", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter()
                .filter(|x| eq(x % 2i64, 0i64))
                .any(ctx, |x| gt(x, 4i64))
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert!(f.call(&[1i64, 3, 6][..])); // 6 is even and > 4
        assert!(!f.call(&[1i64, 3, 4, 5][..])); // only even is 4, not > 4
    });
}

#[test]
fn test_iter_position_after_map() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("pos_after_map", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter()
                .map(|x| x * 2i64)
                .position(ctx, |x| eq(x, 6i64))
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        assert_eq!(f.call(&[1i64, 2, 3, 4][..]), 2); // 3*2 == 6 at index 2
    });
}

#[test]
fn test_iter_scan_prefix_sum() {
    // Running (prefix) sum via scan, then total via sum of the running values'
    // last == grand total; here we just sum the prefix sums for a check value.
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("scan_prefix", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter()
                .scan(0i64, |ctx, acc, x| ctx.store(acc, acc + x))
                .sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        // prefix sums of [1,2,3,4] = [1,3,6,10]; their sum = 20
        assert_eq!(f.call(&[1i64, 2, 3, 4][..]), 20);
    });
}

#[test]
fn test_iter_take_while() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("take_while_lt5", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().take_while(|x| lt(x, 5i64)).sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        // takes 1,2,3 then stops at 5; sum = 6 (the 4 after 5 is not reached)
        assert_eq!(f.call(&[1i64, 2, 3, 5, 4][..]), 6);
    });
}

#[test]
fn test_iter_nested_break() {
    // Count elements of `a` that appear in `b`. The inner `any` builds its own
    // loop and breaks out of it; the outer loop must continue unaffected.
    for_each_backend(|mut compiler| {
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
        assert_eq!(f.call(&a[..], &b[..]), 2); // 2 and 3 are in both
    });
}

#[test]
fn test_iter_skip_while() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("skip_while_lt5", |ctx, arr: Var<SRef<Slice<i64>>>| {
            arr.staged_iter().skip_while(|x| lt(x, 5i64)).sum(ctx)
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let f = compiled.as_fn();
        // skips 1,2,3; yields 5,4,6 from the first >=5; sum = 15
        assert_eq!(f.call(&[1i64, 2, 3, 5, 4, 6][..]), 15);
    });
}

// =============================================================================
// enumerate
// =============================================================================

/// The three-argument path: index and element arrive as separate vars.
#[test]
fn test_iter_enumerate_weighted_sum() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("weighted", |ctx, a: Var<SRef<Slice<i64>>>| {
            let acc = ctx.var(0i64);
            a.staged_iter().enumerate().for_each(ctx, move |ctx, i, x| {
                ctx.store(acc, acc + x * int_cast::<i64, u64, _>(i));
            });
            acc
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let weighted = compiled.as_fn();
        // 10*0 + 20*1 + 30*2 + 40*3 = 200
        let data: [i64; 4] = [10, 20, 30, 40];
        assert_eq!(weighted.call(&data[..]), 200);
    });
}

/// The combinator path: `Item = ZipItem<u64, _>`, so `enumerate` composes with
/// `map` and the terminals like any other adapter.
#[test]
fn test_iter_enumerate_composes_with_map() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("indexed", |ctx, a: Var<SRef<Slice<i64>>>| {
            a.staged_iter()
                .enumerate()
                .map(|pair| pair.second() * int_cast::<i64, u64, _>(pair.first()))
                .sum(ctx)
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let indexed = compiled.as_fn();
        let data: [i64; 4] = [10, 20, 30, 40];
        assert_eq!(indexed.call(&data[..]), 200);
    });
}

/// The index counts *elements the consumer sees*, so a preceding `filter`
/// renumbers from zero rather than reporting source positions.
#[test]
fn test_iter_enumerate_after_filter_renumbers() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("after_filter", |ctx, a: Var<SRef<Slice<i64>>>| {
            let acc = ctx.var(0u64);
            a.staged_iter()
                .filter(|x| gt(x, 15i64))
                .enumerate()
                .for_each(ctx, move |ctx, i, _x| {
                    ctx.store(acc, acc + i);
                });
            acc
        });

        let compiled = compiler.compile(f).expect("compile failed");
        let after_filter = compiled.as_fn();
        // 20, 30, 40 survive and are numbered 0,1,2 -> 0+1+2 = 3
        let data: [i64; 4] = [10, 20, 30, 40];
        assert_eq!(after_filter.call(&data[..]), 3u64);
    });
}

/// A consumer receives a *copy* of the loop counter, not the counter itself, so
/// writing to its item cannot change the iteration. (Before this was fixed,
/// `range` handed out its own counter and a stray store silently skipped
/// elements.)
#[test]
fn test_iter_range_item_is_a_copy_of_the_counter() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("stomp", |ctx, n: Var<u64>| {
            let acc = ctx.var(0u64);
            range(0u64, n).for_each(ctx, move |ctx, i| {
                ctx.store(acc, acc + i);
                // Deliberately stomp the item; the range must be unaffected.
                ctx.store(i, i + 100u64);
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let stomp = compiled.as_fn();
        // 0+1+…+9 = 45, and all 10 iterations must run.
        assert_eq!(stomp.call(10), 45u64);
    });
}

// =============================================================================
// rev
// =============================================================================

/// A reversed range visits the same elements back to front. Weighted by
/// position so the *order* is observable, not just the set.
#[test]
fn test_iter_rev_range() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("rev_range", |ctx, n: Var<u64>| {
            let acc = ctx.var(0u64);
            range(0u64, n).rev().for_each(ctx, move |ctx, i| {
                // acc = acc * 10 + i : encodes the visit order in the digits.
                ctx.store(acc, acc * 10u64 + i);
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let rev = compiled.as_fn();
        assert_eq!(rev.call(4), 3210u64);
        assert_eq!(rev.call(1), 0u64);
        assert_eq!(rev.call(0), 0u64); // empty: the body never runs
    });
}

/// `rev` is keyed on `IndexedSource`, so it reverses a slice as readily as a
/// range — same method, different origin.
#[test]
fn test_iter_rev_slice() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("rev_slice", |ctx, a: Var<SRef<Slice<i64>>>| {
            let acc = ctx.var(0i64);
            a.staged_iter().rev().for_each(ctx, move |ctx, x| {
                ctx.store(acc, acc * 10i64 + x);
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile failed");
        let rev = compiled.as_fn();
        let data: [i64; 4] = [1, 2, 3, 4];
        assert_eq!(rev.call(&data[..]), 4321);
    });
}
