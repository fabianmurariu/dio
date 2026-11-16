//! A record batch read through `SchemaBatch`: columns of different types in one kernel,
//! each read as its schema type; a read of the wrong type is a staging panic. Every
//! kernel on each backend.

use std::sync::Arc;

use arrow::array::{Array, BooleanArray, Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_lms::{
    ArrowArrayOps, FfiBatch, FfiBoolArray, FfiByteViewArray, FfiPrimArray, FfiRecordBatch,
    SchemaBatch,
};
use rust_lms::prelude::*;

mod common;
use common::backends;

fn schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("n", DataType::Int64, false),
        Field::new("b", DataType::Boolean, true),
        Field::new("s", DataType::Utf8View, true),
    ]))
}

/// Forty rows: `n` is the row, `b` and `s` have nulls, and `s` has strings either side
/// of the twelve bytes a view holds inline.
fn record_batch() -> RecordBatch {
    let rows = 0..40i64;
    let ints = Int64Array::from_iter_values(rows.clone());
    let bools: BooleanArray = rows
        .clone()
        .map(|i| (i % 5 != 2).then_some(i % 3 == 0))
        .collect();
    let strings: StringViewArray = rows
        .map(|i| (i % 7 != 3).then(|| "abcdefghijklmnopqrstuvwxyz"[..(i % 23) as usize].to_owned()))
        .collect();
    RecordBatch::try_new(
        schema(),
        vec![Arc::new(ints), Arc::new(bools), Arc::new(strings)],
    )
    .unwrap()
}

/// The sum of `n` over the rows whose `b` is true and whose `s` is longer than twelve
/// bytes, reading all three columns in one kernel.
#[test]
fn a_kernel_reads_columns_of_three_types() {
    let rb = record_batch();
    let ints = rb.column(0).as_any().downcast_ref::<Int64Array>().unwrap();
    let bools = rb
        .column(1)
        .as_any()
        .downcast_ref::<BooleanArray>()
        .unwrap();
    let strings = rb
        .column(2)
        .as_any()
        .downcast_ref::<StringViewArray>()
        .unwrap();
    let want: i64 = (0..rb.num_rows())
        .filter(|&i| bools.is_valid(i) && bools.value(i))
        .filter(|&i| strings.is_valid(i) && strings.value(i).len() > 12)
        .map(|i| ints.value(i))
        .sum();
    assert!(want > 0);

    let batch = FfiRecordBatch::new(rb, &schema()).unwrap();
    for backend in backends() {
        let mut compiler = Compiler::new().with_backend(backend);
        let fun = compiler.fun1("sum_n", |ctx, b: Var<SRef<FfiBatch>>| {
            // SAFETY: every batch passed is an `FfiRecordBatch` of `schema()`.
            let batch = unsafe { SchemaBatch::new(b, schema()) };
            let n = ctx.bind(batch.column::<FfiPrimArray<i64>>(0));
            let flag = ctx.bind(batch.column::<FfiBoolArray>(1));
            let s = ctx.bind(batch.column::<FfiByteViewArray<Utf8>>(2));
            let sum = ctx.var(0i64);
            range(0u64, batch.num_rows()).for_each(ctx, |ctx, i| {
                // SAFETY: `i` is a row; each value is read only when its row is valid.
                let flagged = ctx.bind(unsafe { flag.validity().is_valid(i) });
                ctx.if_then(flagged, |ctx| {
                    ctx.if_then(unsafe { flag.value_unchecked(i) }, |ctx| {
                        ctx.if_then(unsafe { s.validity().is_valid(i) }, |ctx| {
                            let value = ctx.bind(unsafe { s.value_unchecked(i) });
                            ctx.if_then(gt(value.byte_len(), 12u64), |ctx| {
                                ctx.store(sum, sum + unsafe { n.value_unchecked(i) });
                            });
                        });
                    });
                });
            });
            sum
        });
        let compiled = compiler.compile(fun).expect("compile");
        assert_eq!(compiled.as_fn().call(batch.batch()), want);
    }
}

/// A column read as a type its schema field is not described by panics while staging.
#[test]
#[should_panic(expected = "column 2 (s) is Utf8View, not read as")]
fn a_column_read_as_another_type_panics_while_staging() {
    let mut compiler = Compiler::new();
    compiler.fun1("wrong", |ctx, b: Var<SRef<FfiBatch>>| {
        // SAFETY: never run.
        let batch = unsafe { SchemaBatch::new(b, schema()) };
        let n = ctx.bind(batch.column::<FfiPrimArray<i64>>(2));
        n.len()
    });
}

/// A column past the schema's panics while staging.
#[test]
#[should_panic(expected = "column 3 read from a batch of 3 columns")]
fn a_column_past_the_schema_panics_while_staging() {
    let mut compiler = Compiler::new();
    compiler.fun1("past", |ctx, b: Var<SRef<FfiBatch>>| {
        // SAFETY: never run.
        let batch = unsafe { SchemaBatch::new(b, schema()) };
        let n = ctx.bind(batch.column::<FfiPrimArray<i64>>(3));
        n.len()
    });
}
