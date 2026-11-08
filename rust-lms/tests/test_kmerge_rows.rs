//! `kmerge_rows`: a number of sorted rows known only at run time, merged by a heap in
//! stack arrays, by one compiled kernel whatever the number; and `dedup_by_key` after
//! it. Cranelift only.

use proptest::prelude::*;
use rust_lms::func::JitBackend;
use rust_lms::iter::kmerge_rows;
use rust_lms::prelude::*;

type S = SRef<Slice<u64>>;
type Out = SRefMut<Slice<u64>>;
type Kernel = Compiled<rust_lms::func::FunType4<S, S, S, Out, u64>>;

const CAPACITY: usize = 16;

/// Merges the rows of `keys`, with `values` alongside, whose bounds are `offsets`, and
/// writes each value to `out`; with `dedup`, only the first of equal keys.
fn compile(dedup: bool) -> Kernel {
    let mut compiler = Compiler::new().with_backend(JitBackend::Cranelift);
    let f = compiler.fun4(
        "kmerge_rows",
        move |ctx, keys: Var<S>, values: Var<S>, offsets: Var<S>, out: Var<Out>| {
            let rows = ctx.bind(offsets.len() - 1u64);
            // SAFETY: the host passes at most CAPACITY rows of sorted keys, with
            // ascending offsets within both slices.
            let merge = unsafe {
                kmerge_rows(CAPACITY, move |ctx, heap| {
                    range(0u64, rows).for_each(ctx, |ctx, j| {
                        let (start, end) = (
                            ctx.bind(offsets.get_unchecked(j)),
                            ctx.bind(offsets.get_unchecked(j + 1u64)),
                        );
                        heap.push(
                            ctx,
                            ctx.bind(keys.subslice_unchecked(start, end)),
                            ctx.bind(values.subslice_unchecked(start, end)),
                        );
                    })
                })
            };
            let n = ctx.var(0u64);
            let body = move |ctx: &Ctx, pair: Var<ZipItem<u64, u64>>| {
                // SAFETY: `out` has room for every element.
                ctx.emit(unsafe { out.set_unchecked(n, pair.second()) });
                ctx.store(n, n + 1u64);
            };
            if dedup {
                merge.dedup_by_key(|_, pair| pair.first()).for_each(ctx, body);
            } else {
                merge.for_each(ctx, body);
            }
            n
        },
    );
    compiler.compile(f).expect("compilation failed")
}

/// The merged values; a value is `key * 16 + row`.
fn run(kernel: &Kernel, rows: &[Vec<u64>]) -> Vec<u64> {
    let mut keys = Vec::new();
    let mut values = Vec::new();
    let mut offsets = vec![0u64];
    for (j, row) in rows.iter().enumerate() {
        keys.extend(row);
        values.extend(row.iter().map(|k| k * 16 + j as u64));
        offsets.push(keys.len() as u64);
    }
    let mut out = vec![0u64; keys.len()];
    let n = kernel
        .as_fn()
        .call(&keys[..], &values[..], &offsets[..], &mut out[..]);
    out.truncate(n as usize);
    out
}

fn sorted_rows() -> impl Strategy<Value = Vec<Vec<u64>>> {
    prop::collection::vec(
        prop::collection::vec(0u64..40, 0..12).prop_map(|mut row| {
            row.sort();
            row
        }),
        0..=CAPACITY,
    )
}

#[test]
fn merges_any_number_of_rows_with_one_kernel() {
    let kernel = compile(false);
    assert_eq!(run(&kernel, &[]), Vec::<u64>::new());
    assert_eq!(run(&kernel, &[vec![], vec![]]), Vec::<u64>::new());
    let rows = vec![vec![1, 4, 7], vec![2, 5, 8], vec![3, 6, 9]];
    let keys: Vec<u64> = run(&kernel, &rows).iter().map(|v| v >> 4).collect();
    assert_eq!(keys, (1..=9).collect::<Vec<_>>());
}

proptest! {
    #[test]
    fn merge_is_the_sorted_union(rows in sorted_rows()) {
        let kernel = compile(false);
        let merged = run(&kernel, &rows);
        let keys: Vec<u64> = merged.iter().map(|v| v >> 4).collect();
        prop_assert!(keys.windows(2).all(|w| w[0] <= w[1]), "sorted: {keys:?}");
        let mut expected: Vec<u64> = rows
            .iter()
            .enumerate()
            .flat_map(|(j, row)| row.iter().map(move |k| k * 16 + j as u64))
            .collect();
        expected.sort();
        let mut got = merged.clone();
        got.sort();
        prop_assert_eq!(got, expected);
    }

    #[test]
    fn dedup_keeps_each_key_once(rows in sorted_rows()) {
        let kernel = compile(true);
        let keys: Vec<u64> = run(&kernel, &rows).iter().map(|v| v >> 4).collect();
        let mut expected: Vec<u64> = rows.iter().flatten().copied().collect();
        expected.sort();
        expected.dedup();
        prop_assert_eq!(keys, expected);
    }
}
