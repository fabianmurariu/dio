//! `Boolean` columns, a bit per row, read through `FfiBoolArray` against Arrow: nulls,
//! several bytes, and slices starting at any bit.

use arrow::array::BooleanArray;

use arrow_lms::{ArrayBatchOps, ArrowArrayOps, FfiArray, FfiBoolArray, prepare_dyn_arrays};
use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

/// Counts the rows that are valid and true, and the row count.
fn count_true(mut compiler: Compiler) -> impl Fn(&[FfiArray]) -> (u64, u64) {
    let f = compiler.fun2(
        "count_true",
        |ctx, batch: Var<SRef<Slice<FfiArray>>>, out: Var<SRefMut<Slice<u64>>>| {
            // SAFETY: every batch passed is one Boolean column.
            let col = ctx.bind(unsafe { batch.column::<FfiBoolArray>(0) });
            let len = ctx.bind(col.len());
            let count = range(0u64, len).count_if(ctx, move |_, i| {
                // SAFETY: `i < len`, the row count and the validity's length.
                if_then_else(
                    unsafe { col.validity().is_valid(i) },
                    unsafe { col.value_unchecked(i) },
                    Const::<bool>::new(false),
                )
            });
            let mut out = out;
            // SAFETY: the host passes two `u64`s.
            ctx.emit(unsafe { out.reborrow().set_unchecked(0u64, count) });
            ctx.emit(unsafe { out.set_unchecked(1u64, len) });
            unit()
        },
    );
    let compiled = compiler.compile(f).unwrap();
    move |arrays| {
        let mut out = [0u64; 2];
        compiled.as_fn().call(arrays, &mut out);
        (out[0], out[1])
    }
}

fn arrow_count(array: &BooleanArray) -> (u64, u64) {
    let trues = array.iter().filter(|v| *v == Some(true)).count();
    (trues as u64, array.len() as u64)
}

#[test]
fn boolean_values_are_read_bit_by_bit() {
    for_each_backend(|compiler| {
        let count = count_true(compiler);
        // More than one byte, so the byte index matters.
        let values: Vec<bool> = (0..21).map(|i| i % 3 == 0 || i == 20).collect();
        let array = BooleanArray::from(values);
        let prepared = prepare_dyn_arrays(&[&array]).unwrap();
        assert_eq!(prepared.arrays()[0].len(), 21);
        assert_eq!(count(prepared.arrays()), arrow_count(&array));

        let nullable = BooleanArray::from(vec![Some(true), None, Some(false), Some(true), None]);
        let prepared = prepare_dyn_arrays(&[&nullable]).unwrap();
        assert_eq!(count(prepared.arrays()), (2, 5));

        let empty = BooleanArray::from(Vec::<bool>::new());
        let prepared = prepare_dyn_arrays(&[&empty]).unwrap();
        assert_eq!(count(prepared.arrays()), (0, 0));
    });
}

/// A slice starts at its first row, whatever bit that is: on a byte, mid-byte, and with
/// nulls whose bitmap starts elsewhere.
#[test]
fn slices_start_at_their_first_row() {
    for_each_backend(|compiler| {
        let count = count_true(compiler);
        let array = BooleanArray::from(
            (0..40)
                .map(|i| (i % 5 != 2).then_some(i >= 16 || i % 3 == 0))
                .collect::<Vec<_>>(),
        );
        for (offset, len) in [(8, 16), (3, 4), (3, 30), (13, 27)] {
            let sliced = array.slice(offset, len);
            let prepared = prepare_dyn_arrays(&[&sliced]).unwrap();
            assert_eq!(
                count(prepared.arrays()),
                arrow_count(&sliced),
                "{offset}..+{len}"
            );
        }
    });
}
