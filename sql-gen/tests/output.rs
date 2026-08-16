//! End-to-end: run SQL through `exec_jit` (parse → lower → JIT → run) and check
//! the resulting `RecordBatch`, for both row materialization and scalar
//! aggregates.

use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int32Array, Int64Array};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use sql_gen::exec_jit;

/// Two columns: `a: Int32`, `b: Int64` (nullability inferred from the arrays).
fn batch(a: Int32Array, b: Int64Array) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int32, a.null_count() > 0),
        Field::new("b", DataType::Int64, b.null_count() > 0),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap()
}

fn run(sql: &str, rb: &RecordBatch) -> RecordBatch {
    exec_jit(sql, "t", rb).expect("exec_jit")
}

fn i32s(rb: &RecordBatch, col: usize) -> &Int32Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}
fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}
fn f64s(rb: &RecordBatch, col: usize) -> &Float64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}

// -----------------------------------------------------------------------------
// Row materialization
// -----------------------------------------------------------------------------

#[test]
fn filter_then_passthrough_columns() {
    let rb = batch(
        Int32Array::from(vec![1, 2, 3, 4]),
        Int64Array::from(vec![10, 20, 30, 40]),
    );
    let out = run("SELECT a, b FROM t WHERE a < 3", &rb);

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
    let out = run("SELECT a FROM t", &rb);

    let col = i32s(&out, 0);
    assert_eq!(col.null_count(), 1);
    assert!(col.is_valid(0) && col.is_null(1) && col.is_valid(2));
    assert_eq!(col.value(0), 10);
    assert_eq!(col.value(2), 30);
}

#[test]
fn null_predicate_drops_row() {
    let a = Int32Array::new(
        vec![10, 0, 30].into(),
        Some(NullBuffer::from(vec![true, false, true])),
    );
    let rb = batch(a, Int64Array::from(vec![0, 0, 0]));
    let out = run("SELECT a FROM t WHERE a > 15", &rb);

    assert_eq!(out.num_rows(), 1);
    assert_eq!(i32s(&out, 0).value(0), 30);
}

#[test]
fn computed_projection() {
    let rb = batch(
        Int32Array::from(vec![1, 2, 3]),
        Int64Array::from(vec![0, 0, 0]),
    );
    let out = run("SELECT a + 1 FROM t", &rb); // datafusion types a+1 as Int64
    assert_eq!(i64s(&out, 0).values(), &[2, 3, 4]);
}

// -----------------------------------------------------------------------------
// Scalar aggregates (one-row output)
// -----------------------------------------------------------------------------

#[test]
fn count_star() {
    let rb = batch(
        Int32Array::from(vec![1, 2, 3, 4, 5]),
        Int64Array::from(vec![0, 0, 0, 0, 0]),
    );
    let out = run("SELECT count(*) FROM t", &rb);
    assert_eq!(out.num_rows(), 1);
    assert_eq!(i64s(&out, 0).value(0), 5);
}

#[test]
fn count_with_filter() {
    let rb = batch(
        Int32Array::from(vec![1, 2, 3, 4, 5, 6]),
        Int64Array::from(vec![0, 0, 0, 0, 0, 0]),
    );
    let out = run("SELECT count(*) FROM t WHERE a < 4", &rb);
    assert_eq!(i64s(&out, 0).value(0), 3);
}

#[test]
fn count_col_skips_nulls() {
    let a = Int32Array::new(
        vec![10, 0, 30, 0].into(),
        Some(NullBuffer::from(vec![true, false, true, false])),
    );
    let rb = batch(a, Int64Array::from(vec![0, 0, 0, 0]));
    let out = run("SELECT count(a) FROM t", &rb);
    assert_eq!(i64s(&out, 0).value(0), 2);
}

#[test]
fn count_nonnull_col_is_row_count() {
    // A non-nullable column: `count(b)` (null_count 0) equals `count(*)`.
    let rb = batch(
        Int32Array::from(vec![1, 2, 3, 4, 5]),
        Int64Array::from(vec![0, 0, 0, 0, 0]),
    );
    let out = run("SELECT count(b) FROM t", &rb);
    assert_eq!(i64s(&out, 0).value(0), 5);
}

#[test]
fn count_all_null_col_is_zero() {
    let a = Int32Array::new(
        vec![0, 0, 0].into(),
        Some(NullBuffer::from(vec![false, false, false])),
    );
    let rb = batch(a, Int64Array::from(vec![0, 0, 0]));
    let out = run("SELECT count(a) FROM t", &rb);
    assert_eq!(i64s(&out, 0).value(0), 0);
}

#[test]
fn count_col_with_filter_still_correct() {
    // A `WHERE` between the aggregate and the scan forces the per-row path (the
    // batch-level null_count no longer applies); result must still be right.
    let a = Int32Array::new(
        vec![10, 0, 30, 40, 0].into(),
        Some(NullBuffer::from(vec![true, false, true, true, false])),
    );
    let rb = batch(a, Int64Array::from(vec![1, 2, 3, 4, 5]));
    // rows with b > 2: b in {3,4,5} → a = {30(non-null), 40(non-null), null} → 2.
    let out = run("SELECT count(a) FROM t WHERE b > 2", &rb);
    assert_eq!(i64s(&out, 0).value(0), 2);
}

#[test]
fn min_max_sum() {
    let rb = batch(
        Int32Array::from(vec![3, 1, 4, 1, 5]),
        Int64Array::from(vec![10, 20, 30, 40, 50]),
    );
    let out = run("SELECT min(a), max(a), sum(b) FROM t", &rb);
    assert_eq!(out.num_rows(), 1);
    // min(a) / max(a) keep a's Int32 type; sum(b) widens to Int64.
    assert_eq!(i32s(&out, 0).value(0), 1);
    assert_eq!(i32s(&out, 1).value(0), 5);
    assert_eq!(i64s(&out, 2).value(0), 150);
}

#[test]
fn sum_over_expr() {
    let rb = batch(
        Int32Array::from(vec![3, 1, 4, 1, 5]),
        Int64Array::from(vec![10, 20, 30, 40, 50]),
    );
    let out = run("SELECT min(a), max(a), sum(b+1) FROM t", &rb);
    assert_eq!(out.num_rows(), 1);
    // min(a) / max(a) keep a's Int32 type; sum(b) widens to Int64.
    assert_eq!(i32s(&out, 0).value(0), 1);
    assert_eq!(i32s(&out, 1).value(0), 5);
    assert_eq!(i64s(&out, 2).value(0), 155);
}

#[test]
fn aggregate_over_empty_input() {
    let rb = batch(
        Int32Array::from(Vec::<i32>::new()),
        Int64Array::from(Vec::<i64>::new()),
    );
    // count -> 0 (non-null); min/sum -> NULL over empty input.
    let out = run("SELECT count(*), min(a), sum(b) FROM t", &rb);
    assert_eq!(out.num_rows(), 1);
    assert_eq!(i64s(&out, 0).value(0), 0);
    assert!(i32s(&out, 1).is_null(0));
    assert!(i64s(&out, 2).is_null(0));
}

#[test]
fn min_skips_nulls() {
    let a = Int32Array::new(
        vec![9, 0, 2, 0].into(),
        Some(NullBuffer::from(vec![true, false, true, false])),
    );
    let rb = batch(a, Int64Array::from(vec![0, 0, 0, 0]));
    let out = run("SELECT min(a) FROM t", &rb);
    assert_eq!(i32s(&out, 0).value(0), 2);
}

#[test]
fn avg_of_int_column() {
    let rb = batch(
        Int32Array::from(vec![3, 1, 4, 1, 5]),
        Int64Array::from(vec![0, 1, 2, 13, 40]),
    );
    // avg(a) = 14 / 5 = 2.8 (Float64, even over an integer column)
    let out = run("SELECT avg(a), sum(b) FROM t", &rb);
    assert_eq!(f64s(&out, 0).value(0), 2.8);
    assert_eq!(i64s(&out, 1).value(0), 56);
}

#[test]
fn avg_skips_nulls() {
    let a = Int32Array::new(
        vec![9, 0, 2, 0].into(),
        Some(NullBuffer::from(vec![true, false, true, false])),
    );
    let rb = batch(a, Int64Array::from(vec![0, 0, 0, 0]));
    // avg over the two valid values (9, 2) = 5.5
    let out = run("SELECT avg(a) FROM t", &rb);
    assert_eq!(f64s(&out, 0).value(0), 5.5);
}

#[test]
fn avg_over_empty_is_null() {
    let rb = batch(
        Int32Array::from(Vec::<i32>::new()),
        Int64Array::from(Vec::<i64>::new()),
    );
    let out = run("SELECT avg(a) FROM t", &rb);
    assert!(f64s(&out, 0).is_null(0));
}
