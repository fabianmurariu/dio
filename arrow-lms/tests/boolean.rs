//! `Boolean` arrays: bit-packed values read through `BooleanArrayView`, verified
//! against Arrow, including nulls and slices at a byte offset.

use arrow::array::BooleanArray;

use arrow_lms::{ArrayBatchOps, FfiArray, FfiError, prepare_dyn_arrays};
use rust_lms::prelude::*;

/// Counts the rows that are valid and true.
fn count_true() -> impl Fn(&[FfiArray]) -> u64 {
    let mut compiler = Compiler::new();
    let f = compiler.fun1("count_true", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
        // SAFETY: every batch passed below is one `BooleanArray`.
        let col = unsafe { batch.boolean(0) };
        range(0u64, col.len()).count_if(ctx, move |_, i| {
            // SAFETY: `i < len`, the descriptor's row count and validity length.
            if_then_else(
                unsafe { col.validity().is_valid(i) },
                unsafe { col.value_unchecked(i) },
                Const::<bool>::new(false),
            )
        })
    });
    let compiled = compiler.compile(f).unwrap();
    move |arrays| compiled.as_fn().call(arrays)
}

fn arrow_count(array: &BooleanArray) -> u64 {
    array.iter().filter(|v| *v == Some(true)).count() as u64
}

#[test]
fn boolean_values_are_read_bit_by_bit() {
    let count = count_true();
    // More than one byte, so the byte index matters.
    let values: Vec<bool> = (0..21).map(|i| i % 3 == 0 || i == 20).collect();
    let array = BooleanArray::from(values);
    let prepared = prepare_dyn_arrays(&[&array]).unwrap();
    assert_eq!(prepared.arrays()[0].len(), 21);
    assert_eq!(count(prepared.arrays()), arrow_count(&array));

    let nullable = BooleanArray::from(vec![Some(true), None, Some(false), Some(true), None]);
    let prepared = prepare_dyn_arrays(&[&nullable]).unwrap();
    assert_eq!(count(prepared.arrays()), 2);

    let empty = BooleanArray::from(Vec::<bool>::new());
    let prepared = prepare_dyn_arrays(&[&empty]).unwrap();
    assert_eq!(count(prepared.arrays()), 0);
}

#[test]
fn byte_aligned_slices_start_at_their_first_row() {
    let count = count_true();
    let array = BooleanArray::from((0..24).map(|i| i >= 16 || i == 3).collect::<Vec<_>>());
    let sliced = array.slice(8, 16);
    let prepared = prepare_dyn_arrays(&[&sliced]).unwrap();
    assert_eq!(count(prepared.arrays()), 8);
    assert_eq!(arrow_count(&sliced), 8);

    let unaligned = array.slice(3, 4);
    assert!(matches!(
        prepare_dyn_arrays(&[&unaligned]),
        Err(FfiError::UnalignedBooleanValues {
            index: 0,
            bit_offset: 3
        })
    ));
}
