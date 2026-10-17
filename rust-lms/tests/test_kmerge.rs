//! `kmerge_by`: k sorted parts merged into one stable sorted stream, checked against a
//! host stable sort, by a linear pass and by a loser tree, with and without the
//! runner-up shortcut.

use proptest::prelude::*;
use rust_lms::prelude::*;

mod common;
use common::{for_each_backend, with_backends};
use rust_lms::iter::Plan;

const PLANS: [Plan; 2] = [Plan::Linear, Plan::Tree];

type S = SRef<Slice<u64>>;
type Out = SRefMut<Slice<u64>>;
type Kernel = Compiled<rust_lms::func::FunType3<S, S, Out, u64>>;

/// Items are `key * 16 + part`, so the output shows both the order and, on equal keys,
/// which part came first.
fn tag(parts: &[Vec<u64>]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let mut values = Vec::new();
    let mut offsets = vec![0u64];
    for (j, part) in parts.iter().enumerate() {
        values.extend(part.iter().map(|k| k * 16 + j as u64));
        offsets.push(values.len() as u64);
    }
    let mut expected = values.clone();
    expected.sort_by_key(|x| x >> 4);
    (values, offsets, expected)
}

/// Writes the merge of `k` parts of `values` (bounds in `offsets`) to `out`, up to
/// `limit` elements, and returns how many it wrote.
fn compile_merge(
    mut compiler: Compiler,
    k: usize,
    plan: Plan,
    runs: bool,
    limit: Option<u64>,
) -> Kernel {
    let f = compiler.fun3(
        "kmerge",
        move |ctx, values: Var<S>, offsets: Var<S>, out: Var<Out>| {
            let parts: Vec<_> = (0..k as u64)
                .map(|j| {
                    // SAFETY: the host passes `k + 1` ascending offsets within `values`.
                    unsafe {
                        let start = offsets.get_unchecked(j);
                        let end = offsets.get_unchecked(j + 1);
                        ctx.bind(values.subslice_unchecked(start, end))
                    }
                })
                .collect();
            let merge = kmerge_by(parts, |_, x| shr(x, 4u64));
            let merge = kmerge_with(merge, plan, runs);
            let n = ctx.var(0u64);
            let body = move |ctx: &Ctx, x: Var<u64>| {
                // SAFETY: `out` has room for every element.
                ctx.emit(unsafe { out.set_unchecked(n, x) });
                ctx.store(n, n + 1u64);
            };
            match limit {
                Some(limit) => merge.take(limit).for_each(ctx, body),
                None => merge.for_each(ctx, body),
            }
            n
        },
    );
    compiler.compile(f).expect("compile failed")
}

fn kmerge_with<I, F>(merge: KMergeBy<I, F>, plan: Plan, runs: bool) -> KMergeBy<I, F> {
    let merge = merge.with_plan(plan);
    if runs { merge } else { merge.without_runs() }
}

fn run(kernel: &Kernel, parts: &[Vec<u64>]) -> Vec<u64> {
    let (values, offsets, _) = tag(parts);
    let mut out = vec![0u64; values.len()];
    let n = kernel.as_fn().call(&values[..], &offsets[..], &mut out[..]);
    out.truncate(n as usize);
    out
}

#[test]
fn merges_interleaved_parts() {
    with_backends(|make| {
        for plan in PLANS {
            let parts = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];
            let kernel = compile_merge(make(), 3, plan, true, None);
            assert_eq!(run(&kernel, &parts), tag(&parts).2);
        }
    });
}

#[test]
fn equal_keys_keep_part_order() {
    with_backends(|make| {
        for plan in PLANS {
            let parts = vec![vec![2, 2], vec![1, 2], vec![2, 3], vec![]];
            let kernel = compile_merge(make(), 4, plan, true, None);
            let merged = run(&kernel, &parts);
            assert_eq!(merged, tag(&parts).2);
            assert_eq!(
                merged.iter().map(|x| x & 15).collect::<Vec<_>>(),
                [1, 0, 0, 1, 2, 2]
            );
        }
    });
}

#[test]
fn no_parts_and_empty_parts() {
    with_backends(|make| {
        for plan in PLANS {
            let kernel = compile_merge(make(), 0, plan, true, None);
            assert_eq!(run(&kernel, &[]), Vec::<u64>::new());
            let kernel = compile_merge(make(), 3, plan, true, None);
            assert_eq!(run(&kernel, &[vec![], vec![], vec![]]), Vec::<u64>::new());
            let parts = vec![vec![], vec![5], vec![]];
            assert_eq!(run(&kernel, &parts), tag(&parts).2);
        }
    });
}

#[test]
fn stops_early() {
    with_backends(|make| {
        for plan in PLANS {
            let parts = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];
            let kernel = compile_merge(make(), 3, plan, true, Some(4));
            assert_eq!(run(&kernel, &parts), tag(&parts).2[..4]);
        }
    });
}

#[test]
fn non_indexed_parts() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("odd_merge", |ctx, a: Var<S>, b: Var<S>| {
            let odd = |s: Var<S>| s.staged_iter().filter(|x| eq(rem(x, 2u64), 1u64));
            let acc = ctx.var(0u64);
            kmerge_by(vec![odd(a), odd(b)], |_, x| x).for_each(ctx, move |ctx, x| {
                ctx.store(acc, acc * 10u64 + x);
            });
            acc
        });
        let compiled = compiler.compile(f).expect("compile failed");
        assert_eq!(
            compiled
                .as_fn()
                .call(&[1u64, 2, 5, 9][..], &[3u64, 4, 7][..]),
            13579
        );
    });
}

#[test]
fn event_keys_packed_to_i128() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun3(
            "events",
            |ctx, times: Var<SRef<Slice<i64>>>, ids: Var<S>, split: Var<u64>| {
                // SAFETY: `split <= len` for both columns, which the host keeps equal.
                let part = |lo: Var<u64>, hi: Var<u64>| unsafe {
                    let t = ctx.bind(times.subslice_unchecked(lo, hi));
                    let i = ctx.bind(ids.subslice_unchecked(lo, hi));
                    t.staged_iter().zip(i)
                };
                let (zero, len) = (ctx.bind(Const::new(0u64)), ctx.bind(ids.len()));
                let acc = ctx.var(0u64);
                kmerge_by(vec![part(zero, split), part(split, len)], |_, e| {
                    pack_i128(e.first(), e.second())
                })
                .for_each(ctx, move |ctx, e| ctx.store(acc, acc * 10u64 + e.second()));
                acc
            },
        );
        let compiled = compiler.compile(f).expect("compile failed");
        let times = [-5i64, 0, 0, 7, -5, 0, 3];
        let ids = [1u64, 3, 6, 9, 2, 4, 8];
        // (-5,1) (-5,2) (0,3) (0,4) (0,6) (3,8) (7,9)
        assert_eq!(compiled.as_fn().call(&times[..], &ids[..], 4), 1234689);
    });
}

fn sorted_parts() -> impl Strategy<Value = Vec<Vec<u64>>> {
    prop::collection::vec(
        prop::collection::vec(0u64..40, 0..12).prop_map(|mut p| {
            p.sort();
            p
        }),
        0..12,
    )
}

fn plan() -> impl Strategy<Value = Plan> {
    prop_oneof![Just(Plan::Linear), Just(Plan::Tree)]
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(128))]

    #[test]
    fn matches_a_stable_sort(parts in sorted_parts(), plan in plan(), runs in any::<bool>()) {
        let kernel = compile_merge(Compiler::new(), parts.len(), plan, runs, None);
        prop_assert_eq!(run(&kernel, &parts), tag(&parts).2);
    }

    #[test]
    fn disjoint_parts(
        lens in prop::collection::vec(0u64..20, 0..12),
        plan in plan(),
        runs in any::<bool>(),
    ) {
        let mut start = 0;
        let mut parts: Vec<Vec<u64>> = lens
            .iter()
            .map(|&n| {
                let part = (start..start + n).collect();
                start += n;
                part
            })
            .collect();
        parts.reverse();
        let kernel = compile_merge(Compiler::new(), parts.len(), plan, runs, None);
        prop_assert_eq!(run(&kernel, &parts), tag(&parts).2);
    }
}
