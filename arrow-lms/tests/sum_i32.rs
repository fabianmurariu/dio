//! Read-path smoke tests: pass Arrow primitive arrays into JIT'd kernels and
//! read them through the staged column view (`primitive`/`value_unchecked`/
//! `is_valid`/`len`), verified against Arrow.

use std::sync::Arc;

use arrow::array::{Array, Int16Array, Int32Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use arrow_lms::{prepare_dyn_arrays, prepare_record_batch, ArrayBatchOps, FfiArray};
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
fn sum_i32_column() {
    let rb = batch(vec![1, 2, 3, 4, 5]);
    let prepared = prepare_record_batch(&rb).unwrap();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
        // SAFETY: `batch()` constructs a one-column Int32 batch.
        let col = unsafe { batch.primitive::<i32>(0) };
        let acc = ctx.var(0i32);
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, col.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < col.len()`.
            let v = ctx.bind(unsafe { col.value_unchecked(i) });
            ctx.store(acc, add(acc, v));
            ctx.store(i, add(i, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(f).unwrap();
    let sum = compiled.as_fn();

    let arrow_sum =
        arrow::compute::sum(rb.column(0).as_any().downcast_ref::<Int32Array>().unwrap()).unwrap();
    assert_eq!(sum.call(prepared.arrays()), 15);
    assert_eq!(sum.call(prepared.arrays()), arrow_sum);
}

#[test]
fn empty_column_sums_to_zero() {
    let rb = batch(vec![]);
    let prepared = prepare_record_batch(&rb).unwrap();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
        // SAFETY: `batch()` constructs a one-column Int32 batch.
        let col = unsafe { batch.primitive::<i32>(0) };
        let acc = ctx.var(0i32);
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, col.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < col.len()`.
            let v = ctx.bind(unsafe { col.value_unchecked(i) });
            ctx.store(acc, add(acc, v));
            ctx.store(i, add(i, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(f).unwrap();
    let sum = compiled.as_fn();

    assert_eq!(sum.call(prepared.arrays()), 0);
}

#[test]
fn sum_skips_nulls_via_is_valid() {
    let rb = nullable_batch(vec![10, 999, 30], vec![true, false, true]);
    let prepared = prepare_record_batch(&rb).unwrap();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum_valid", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
        // SAFETY: `nullable_batch()` constructs a one-column Int32 batch.
        let col = unsafe { batch.primitive::<i32>(0) };
        let acc = ctx.var(0i32);
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, col.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < col.len()`.
            let valid = ctx.bind(unsafe { col.validity().is_valid(i) });
            ctx.if_then(valid, move |ctx| {
                // SAFETY: the surrounding loop proves `i < col.len()`.
                let v = ctx.bind(unsafe { col.value_unchecked(i) });
                ctx.store(acc, add(acc, v));
            });
            ctx.store(i, add(i, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(f).unwrap();
    let sum = compiled.as_fn();

    assert_eq!(sum.call(prepared.arrays()), 40);
}

#[test]
fn is_valid_respects_sliced_bitmap_offset() {
    let col = Int32Array::new(
        vec![5, 10, 20, 30, 40].into(),
        Some(NullBuffer::from(vec![true, false, true, false, true])),
    );
    // slice [10, 20, 30] with validity [false, true, false] (bit_offset = 1)
    let sliced = col.slice(1, 3);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
    let rb = RecordBatch::try_new(schema, vec![Arc::new(sliced)]).unwrap();
    let prepared = prepare_record_batch(&rb).unwrap();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("count_valid", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
        // SAFETY: the prepared batch has one Int32 column.
        let col = unsafe { batch.primitive::<i32>(0) };
        let acc = ctx.var(0i64);
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, col.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < col.len()`.
            let valid = ctx.bind(unsafe { col.validity().is_valid(i) });
            ctx.if_then(valid, move |ctx| ctx.store(acc, add(acc, 1i64)));
            ctx.store(i, add(i, 1u64));
        });
        acc
    });
    let compiled = compiler.compile(f).unwrap();
    let count_valid = compiled.as_fn();

    assert_eq!(count_valid.call(prepared.arrays()), 1);
}

#[test]
fn read_i16_from_erased_dyn_arrays() {
    let col = Int16Array::from(vec![2i16, 3, 5]);
    let arrays: [&dyn Array; 1] = [&col];
    let prepared = prepare_dyn_arrays(&arrays).unwrap();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum_i16", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
        // SAFETY: `arrays` contains one Int16 array.
        let col = unsafe { batch.primitive::<i16>(0) };
        let acc = ctx.var(0i16);
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, col.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < col.len()`.
            let v = ctx.bind(unsafe { col.value_unchecked(i) });
            ctx.store(acc, add(acc, v));
            ctx.store(i, add(i, 1u64));
        });
        int_cast::<i64, i16, _>(acc)
    });
    let compiled = compiler.compile(f).unwrap();
    let sum = compiled.as_fn();

    assert_eq!(sum.call(prepared.arrays()), 10);
}
