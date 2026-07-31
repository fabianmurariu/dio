//! End-to-end vertical slice: parse SQL with datafusion, lower it to our push
//! operators, JIT a `count(*)` kernel, and check the result matches the
//! plain-Rust reference interpreter (the Futamura equivalence).

use std::sync::Arc;

use arrow::array::{Int32Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArray, prepare_record_batch};
use rust_lms::prelude::*;
use sql_gen::exec::exec_count;
use sql_gen::{Operator, gen_count, sql_to_operator};

/// Two columns: `a: Int32`, `b: Int64`.
fn batch(a: Vec<i32>, b: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Int32, false),
        Field::new("b", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![Arc::new(Int32Array::from(a)), Arc::new(Int64Array::from(b))],
    )
    .unwrap()
}

/// Compile `op` into a `count(*)` kernel and run it over `rb`.
fn jit_count(op: &Operator, rb: &RecordBatch) -> i64 {
    let prepared = prepare_record_batch(rb).unwrap();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("q", |ctx, batch: Var<SRef<Slice<FfiArray>>>| {
        gen_count(ctx, batch, op)
    });
    let compiled = compiler.compile(f).expect("compile");
    compiled.as_fn()(prepared.arrays())
}

/// Run `sql` end-to-end, asserting the JIT and reference interpreter agree, and
/// return the row count.
fn run_sql(sql: &str, rb: &RecordBatch) -> i64 {
    let op = sql_to_operator(sql, "t", rb.schema()).expect("lower sql");
    let reference = exec_count(&op, rb);
    let jit = jit_count(&op, rb);
    assert_eq!(jit, reference, "jit vs reference for `{sql}`");
    jit
}

#[test]
fn select_star_counts_all_rows() {
    let rb = batch(vec![1, 2, 3, 4, 5], vec![10, 20, 30, 40, 50]);
    assert_eq!(run_sql("SELECT * FROM t", &rb), 5);
}

#[test]
fn filter_on_i32_column() {
    let rb = batch(vec![1, 2, 3, 4, 5, 6], vec![0, 0, 0, 0, 0, 0]);
    assert_eq!(run_sql("SELECT * FROM t WHERE a < 4", &rb), 3);
}

#[test]
fn filter_on_i64_column() {
    let rb = batch(vec![0, 0, 0, 0], vec![10, 25, 30, 100]);
    assert_eq!(run_sql("SELECT * FROM t WHERE b >= 30", &rb), 2);
}

#[test]
fn conjunction_across_columns() {
    let rb = batch(vec![1, 5, 5, 9, 5], vec![10, 20, 40, 40, 40]);
    // a = 5 AND b > 30
    assert_eq!(run_sql("SELECT a FROM t WHERE a = 5 AND b > 30", &rb), 2);
}

#[test]
fn disjunction() {
    let rb = batch(vec![1, 2, 3, 4, 5], vec![0, 0, 0, 0, 0]);
    assert_eq!(run_sql("SELECT * FROM t WHERE a = 1 OR a = 5", &rb), 2);
}

#[test]
fn not_equal() {
    let rb = batch(vec![7, 7, 1, 7, 2], vec![0, 0, 0, 0, 0]);
    assert_eq!(run_sql("SELECT * FROM t WHERE a <> 7", &rb), 2);
}

#[test]
fn empty_result() {
    let rb = batch(vec![1, 2, 3], vec![0, 0, 0]);
    assert_eq!(run_sql("SELECT * FROM t WHERE a > 100", &rb), 0);
}
