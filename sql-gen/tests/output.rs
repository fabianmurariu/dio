//! Output materialization: parse SQL, JIT a kernel that writes surviving rows
//! into pre-allocated Arrow buffers, and check the resulting `RecordBatch`.

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArrayBatch, FfiMutableArrays, PreparedOutput, prepare_record_batch};
use rust_lms::prelude::*;
use sql_gen::{gen_collect, sql_to_operator};

fn batch(a: Int32Array, b: Int64Array) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int32, a.null_count() > 0),
        Field::new("b", DataType::Int64, b.null_count() > 0),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap()
}

/// Parse `sql`, JIT a materializing kernel, and return the output `RecordBatch`.
fn jit_collect(sql: &str, rb: &RecordBatch) -> RecordBatch {
    let op = sql_to_operator(sql, "t", rb.schema()).expect("lower sql");
    let out_schema = op.output_schema();

    let prepared_in = prepare_record_batch(rb).unwrap();
    let ffi_in = prepared_in.as_ffi();
    let mut out = PreparedOutput::alloc(out_schema, rb.num_rows());

    let mut compiler = Compiler::new();
    let f = compiler.fun2(
        "q",
        |ctx, batch: Var<SRef<FfiArrayBatch>>, sink: Var<SRefMut<FfiMutableArrays>>| {
            gen_collect(ctx, batch, sink, &op, &op.output_schema())
        },
    );
    let compiled = compiler.compile(f).expect("compile");

    let mut ffi_out = out.as_ffi_mut();
    let n = compiled.as_fn()(&ffi_in, &mut ffi_out);
    out.into_record_batch(n as usize)
}

fn i32s(rb: &RecordBatch, col: usize) -> &Int32Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}
fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}

#[test]
fn filter_then_passthrough_columns() {
    let rb = batch(
        Int32Array::from(vec![1, 2, 3, 4]),
        Int64Array::from(vec![10, 20, 30, 40]),
    );
    let out = jit_collect("SELECT a, b FROM t WHERE a < 3", &rb);

    assert_eq!(out.num_rows(), 2);
    assert_eq!(i32s(&out, 0).values(), &[1, 2]);
    assert_eq!(i64s(&out, 1).values(), &[10, 20]);
}

#[test]
fn nullable_column_preserves_nulls() {
    let a = Int32Array::new(
        vec![10, 0, 30].into(),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let rb = batch(a, Int64Array::from(vec![0, 0, 0]));
    let out = jit_collect("SELECT a FROM t", &rb);

    assert_eq!(out.num_rows(), 3);
    let col = i32s(&out, 0);
    assert_eq!(col.null_count(), 1);
    assert!(col.is_valid(0) && col.is_null(1) && col.is_valid(2));
    assert_eq!(col.value(0), 10);
    assert_eq!(col.value(2), 30);
}

#[test]
fn null_predicate_drops_row() {
    // a > 15 with a = [10, null, 30]: the null row's predicate is unknown -> dropped.
    let a = Int32Array::new(
        vec![10, 0, 30].into(),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let rb = batch(a, Int64Array::from(vec![0, 0, 0]));
    let out = jit_collect("SELECT a FROM t WHERE a > 15", &rb);

    assert_eq!(out.num_rows(), 1);
    assert_eq!(i32s(&out, 0).value(0), 30);
}

#[test]
fn computed_projection() {
    let rb = batch(
        Int32Array::from(vec![1, 2, 3]),
        Int64Array::from(vec![0, 0, 0]),
    );
    // a + 1 -> datafusion types this as Int64 (widened literal).
    let out = jit_collect("SELECT a + 1 FROM t", &rb);

    assert_eq!(out.num_rows(), 3);
    assert_eq!(i64s(&out, 0).values(), &[2, 3, 4]);
}
