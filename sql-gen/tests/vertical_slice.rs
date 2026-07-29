//! End-to-end vertical slice: build an `Operator` plan, JIT it against an Arrow
//! `RecordBatch`, and check the result matches the plain-Rust reference
//! interpreter (the Futamura equivalence).

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArrayBatch, prepare_record_batch};
use rust_lms::prelude::*;
use sql_gen::exec::exec_count;
use sql_gen::{Expr, Operator, Predicate, Schema, gen_count};

fn one_col_i32(vals: Vec<i32>) -> RecordBatch {
    let col = Int32Array::from(vals);
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "a",
        DataType::Int32,
        false,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
}

/// Compile `plan` into a `count(*)` kernel and run it over `rb`.
fn jit_count(plan: &Operator, rb: &RecordBatch) -> i64 {
    let schema = Schema::from_record_batch(rb);
    let prepared = prepare_record_batch(rb).unwrap();
    let ffi = prepared.as_ffi();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("q", |ctx, batch: Var<SRef<FfiArrayBatch>>| {
        gen_count(ctx, batch, plan, &schema)
    });
    let compiled = compiler.compile(f).expect("compile");
    compiled.as_fn()(&ffi)
}

/// Assert the JIT and the reference interpreter agree, and (optionally) match an
/// expected count.
fn assert_agree(plan: &Operator, rb: &RecordBatch, expected: i64) {
    let reference = exec_count(plan, rb);
    let jit = jit_count(plan, rb);
    assert_eq!(reference, expected, "reference interpreter");
    assert_eq!(jit, expected, "JIT kernel");
}

#[test]
fn scan_counts_all_rows() {
    let rb = one_col_i32(vec![1, 2, 3, 4, 5]);
    assert_agree(&Operator::Scan, &rb, 5);
}

#[test]
fn filter_lt_counts_matches() {
    let rb = one_col_i32(vec![1, 2, 3, 4, 5, 6]);
    // where a < 4
    let plan = Operator::Filter(
        Predicate::Lt(Expr::Col(0), Expr::LitI32(4)),
        Box::new(Operator::Scan),
    );
    assert_agree(&plan, &rb, 3);
}

#[test]
fn filter_eq_then_project() {
    let rb = one_col_i32(vec![7, 7, 1, 7, 2]);
    // select a where a = 7
    let plan = Operator::Project(
        vec![0],
        Box::new(Operator::Filter(
            Predicate::Eq(Expr::Col(0), Expr::LitI32(7)),
            Box::new(Operator::Scan),
        )),
    );
    assert_agree(&plan, &rb, 3);
}

#[test]
fn empty_batch_counts_zero() {
    let rb = one_col_i32(vec![]);
    assert_agree(&Operator::Scan, &rb, 0);
}
