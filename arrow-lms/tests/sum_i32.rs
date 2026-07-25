//! End-to-end prototype: pass Arrow primitive arrays into JIT'd kernels and
//! operate on them through the `rust-lms` iterator framework.

use std::sync::Arc;

use arrow::array::{Array, Int16Array, Int32Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use arrow_lms::{prepare_dyn_arrays, prepare_record_batch, FfiArrayBatch, FfiArrayBatchOps};
use rust_lms::prelude::*;

fn batch(values: Vec<i32>) -> RecordBatch {
    let col = Int32Array::from(values);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
}

fn nullable_batch(values: Vec<i32>, validity: Vec<bool>) -> RecordBatch {
    let col = Int32Array::new(values.into(), Some(NullBuffer::from(validity)));
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
}

#[test]
fn sum_i32_column_from_batch() {
    let rb = batch(vec![1, 2, 3, 4, 5]);
    let prepared = prepare_record_batch(&rb).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum_i32", |ctx, batch: Var<SRef<FfiArrayBatch>>| {
        batch.primitive::<i32>(0).physical_values().sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile");
    let sum = compiled.as_fn();

    assert_eq!(sum(&ffi), 15);

    let arrow_sum =
        arrow::compute::sum(rb.column(0).as_any().downcast_ref::<Int32Array>().unwrap()).unwrap();
    assert_eq!(sum(&ffi), arrow_sum);
}

#[test]
fn empty_column_sums_to_zero() {
    let rb = batch(vec![]);
    let prepared = prepare_record_batch(&rb).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum_i32", |ctx, batch: Var<SRef<FfiArrayBatch>>| {
        batch.primitive::<i32>(0).physical_values().sum(ctx)
    });
    let sum = compiler.compile(f).unwrap().as_fn();

    assert_eq!(sum(&ffi), 0);
}

#[test]
fn combinators_compose_over_physical_values() {
    let rb = batch(vec![1, 2, 3, 4, 5, 6]);
    let prepared = prepare_record_batch(&rb).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let count_gt3 = compiler.fun1("count_gt3", |ctx, batch: Var<SRef<FfiArrayBatch>>| {
        batch
            .primitive::<i32>(0)
            .physical_values()
            .count_if(ctx, |x| gt(x, 3i32))
    });
    let count = compiler.compile(count_gt3).unwrap().as_fn();
    assert_eq!(count(&ffi), 3);

    let mut compiler = Compiler::new();
    let sum_even_sq = compiler.fun1("sum_even_sq", |ctx, batch: Var<SRef<FfiArrayBatch>>| {
        batch
            .primitive::<i32>(0)
            .physical_values()
            .filter(|x| eq(x % 2i32, 0i32))
            .map(|x| x * x)
            .sum(ctx)
    });
    let sum = compiler.compile(sum_even_sq).unwrap().as_fn();
    assert_eq!(sum(&ffi), 4 + 16 + 36);
}

#[test]
fn non_null_values_skip_invalid_rows() {
    let rb = nullable_batch(vec![10, 999, 30], vec![true, false, true]);
    let prepared = prepare_record_batch(&rb).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let f = compiler.fun1(
        "sum_non_null_i32",
        |ctx, batch: Var<SRef<FfiArrayBatch>>| batch.primitive::<i32>(0).non_null_values().sum(ctx),
    );
    let sum = compiler.compile(f).unwrap().as_fn();

    assert_eq!(sum(&ffi), 40);
}

#[test]
fn primitive_staged_iter_composes_with_standard_combinators() {
    let rb = nullable_batch(vec![10, 999, 30], vec![true, false, true]);
    let prepared = prepare_record_batch(&rb).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let f = compiler.fun1(
        "sum_staged_iter_i32",
        |ctx, batch: Var<SRef<FfiArrayBatch>>| {
            batch
                .primitive::<i32>(0)
                .staged_iter()
                .filter(|row| row.second())
                .map(|row| row.first())
                .sum(ctx)
        },
    );
    let sum = compiler.compile(f).unwrap().as_fn();

    assert_eq!(sum(&ffi), 40);
}

#[test]
fn validity_iter_zips_with_physical_values() {
    let rb = nullable_batch(vec![10, 999, 30, 777], vec![true, false, true, false]);
    let prepared = prepare_record_batch(&rb).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let f = compiler.fun1(
        "manual_valid_sum",
        |ctx, batch: Var<SRef<FfiArrayBatch>>| {
            let arr = batch.primitive::<i32>(0);
            let acc = ctx.var(0i32);

            arr.physical_values().zip(arr.validity().iter()).for_each(
                ctx,
                move |ctx, value, is_valid| {
                    ctx.if_then(is_valid, move |ctx| {
                        ctx.store(acc, acc + value);
                    });
                },
            );

            acc
        },
    );
    let sum = compiler.compile(f).unwrap().as_fn();

    assert_eq!(sum(&ffi), 40);
}

#[test]
fn validity_iter_respects_sliced_bitmap_offsets() {
    let col = Int32Array::new(
        vec![5, 10, 20, 30, 40].into(),
        Some(NullBuffer::from(vec![true, false, true, false, true])),
    );
    let sliced = col.slice(1, 3);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let rb = RecordBatch::try_new(schema, vec![Arc::new(sliced)]).unwrap();
    let prepared = prepare_record_batch(&rb).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("count_valid", |ctx, batch: Var<SRef<FfiArrayBatch>>| {
        batch
            .primitive::<i32>(0)
            .validity()
            .iter()
            .count_if(ctx, |is_valid| is_valid)
    });
    let count_valid = compiler.compile(f).unwrap().as_fn();

    assert_eq!(count_valid(&ffi), 1);
}

#[test]
fn primitive_view_works_from_erased_dyn_arrays() {
    let col = Int16Array::from(vec![2i16, 3, 5]);
    let arrays: [&dyn Array; 1] = [&col];
    let prepared = prepare_dyn_arrays(&arrays).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum_i16", |ctx, batch: Var<SRef<FfiArrayBatch>>| {
        let total = batch.primitive::<i16>(0).physical_values().sum(ctx);
        int_cast::<i64, i16, _>(total)
    });
    let sum = compiler.compile(f).unwrap().as_fn();

    assert_eq!(sum(&ffi), 10);
}
