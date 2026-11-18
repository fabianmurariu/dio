//! Primitive columns read in kernels through `FfiPrimArray<T>` (`values`,
//! `value_unchecked`, `validity`, `len`), against Arrow.

use std::sync::Arc;

use arrow::{
    array::{Array, Int16Array, Int32Array},
    buffer::NullBuffer,
    datatypes::{DataType, Field, Schema},
    record_batch::RecordBatch,
};

use arrow_lms::{
    ArrayBatchOps, ArrowArrayOps, FfiArray, FfiPrimArray, FfiPrimArrayOps, prepare_dyn_arrays,
    prepare_record_batch,
};
use rust_lms::prelude::*;

mod common;
#[cfg(test)]
use common::for_each_backend;

type BatchFn<R> = FunType1<SRef<Slice<FfiArray>>, R>;

fn batch(values: Vec<i32>) -> RecordBatch {
    let col = Int32Array::from(values);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
}

/// The sum of an `Int32` batch's only column.
fn sum_kernel(mut compiler: Compiler) -> Compiled<BatchFn<i32>> {
    let f = compiler.fun1("sum", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
        // SAFETY: every batch passed is one Int32 column.
        let col = ctx.bind(unsafe { batch.column::<FfiPrimArray<i32>>(0) });
        col.values().staged_iter().sum(ctx)
    });
    compiler.compile(f).unwrap()
}

#[test]
fn sum_i32_column() {
    for_each_backend(|compiler| {
        let sum = sum_kernel(compiler);
        let rb = batch(vec![1, 2, 3, 4, 5]);
        let prepared = prepare_record_batch(&rb).unwrap();
        let arrow_sum =
            arrow::compute::sum(rb.column(0).as_any().downcast_ref::<Int32Array>().unwrap())
                .unwrap();
        assert_eq!(sum.as_fn().call(prepared.arrays()), 15);
        assert_eq!(sum.as_fn().call(prepared.arrays()), arrow_sum);
    });
}

#[test]
fn empty_column_sums_to_zero() {
    for_each_backend(|compiler| {
        let sum = sum_kernel(compiler);
        let rb = batch(vec![]);
        let prepared = prepare_record_batch(&rb).unwrap();
        assert_eq!(sum.as_fn().call(prepared.arrays()), 0);
    });
}

/// A slice starts at its first row: its values are offset, not just its validity.
#[test]
fn a_sliced_column_starts_at_its_first_row() {
    for_each_backend(|compiler| {
        let sum = sum_kernel(compiler);
        let array = Int32Array::from(vec![1000, 1, 2, 3, 2000]);
        let sliced = array.slice(1, 3);
        let prepared = prepare_dyn_arrays(&[&sliced]).unwrap();
        assert_eq!(sum.as_fn().call(prepared.arrays()), 6);
    });
}

#[test]
fn sum_skips_nulls_via_is_valid() {
    for_each_backend(|mut compiler| {
        let col = Int32Array::new(
            vec![10, 999, 30].into(),
            Some(NullBuffer::from(vec![true, false, true])),
        );
        let prepared = prepare_dyn_arrays(&[&col]).unwrap();
        let f = compiler.fun1("sum_valid", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
            // SAFETY: the batch is one Int32 column.
            let col = ctx.bind(unsafe { batch.column::<FfiPrimArray<i32>>(0) });
            let acc = ctx.var(0i32);
            col.values()
                .staged_iter()
                .enumerate()
                .for_each(ctx, move |ctx, pair| {
                    let i = ctx.bind(pair.first());
                    let v = ctx.bind(pair.second());
                    // SAFETY: `i` is a position the iterator produced, so a row.
                    let valid = ctx.bind(unsafe { col.validity().is_valid(i) });
                    ctx.if_then(valid, move |ctx| ctx.store(acc, add(acc, v)));
                });
            acc
        });
        let compiled = compiler.compile(f).unwrap();
        assert_eq!(compiled.as_fn().call(prepared.arrays()), 40);
    });
}

#[test]
fn is_valid_respects_sliced_bitmap_offset() {
    for_each_backend(|mut compiler| {
        let col = Int32Array::new(
            vec![5, 10, 20, 30, 40].into(),
            Some(NullBuffer::from(vec![true, false, true, false, true])),
        );
        // [10, 20, 30] with validity [false, true, false] (bit offset 1).
        let sliced = col.slice(1, 3);
        let prepared = prepare_dyn_arrays(&[&sliced]).unwrap();
        let f = compiler.fun1("count_valid", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
            // SAFETY: the batch is one Int32 column.
            let col = ctx.bind(unsafe { batch.column::<FfiPrimArray<i32>>(0) });
            // Only the bitmap is read, so a range over positions drives this one.
            let valid = range(0u64, col.len()).count_if(ctx, move |_, i| {
                // SAFETY: `i < col.len()`, the range's bound.
                unsafe { col.validity().is_valid(i) }
            });
            int_cast::<i64, u64, _>(valid)
        });
        let compiled = compiler.compile(f).unwrap();
        assert_eq!(compiled.as_fn().call(prepared.arrays()), 1);
    });
}

#[test]
fn read_i16_by_row() {
    for_each_backend(|mut compiler| {
        let col = Int16Array::from(vec![2i16, 3, 5]);
        let prepared = prepare_dyn_arrays(&[&col]).unwrap();
        let f = compiler.fun1("sum_i16", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
            // SAFETY: the batch is one Int16 column.
            let col = ctx.bind(unsafe { batch.column::<FfiPrimArray<i16>>(0) });
            let acc = ctx.var(0i64);
            range(0u64, col.len()).for_each(ctx, |ctx, i| {
                // SAFETY: `i < len`.
                let v = int_cast::<i64, i16, _>(unsafe { col.value_unchecked(i) });
                ctx.store(acc, add(acc, v));
            });
            acc
        });
        let compiled = compiler.compile(f).unwrap();
        assert_eq!(compiled.as_fn().call(prepared.arrays()), 10);
    });
}
