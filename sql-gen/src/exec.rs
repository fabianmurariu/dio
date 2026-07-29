//! Plain-Rust reference interpreter — the un-staged twin of [`crate::gen`].
//!
//! It runs the same `Operator` tree directly over a `RecordBatch`. Its only job
//! is to give tests an independent oracle: the JIT kernel must produce the same
//! result (the Futamura equivalence). `gen.rs` mirrors this file one-to-one.

use arrow::array::{Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

use crate::plan::{Expr, Operator, Predicate};

/// A host-side (un-staged) column value — the reference twin of
/// [`crate::value::ColVal`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Val {
    I32(i32),
    I64(i64),
    F64(f64),
}

/// Count the rows produced by `plan` over `rb`.
pub fn exec_count(plan: &Operator, rb: &RecordBatch) -> i64 {
    let mut n = 0i64;
    exec_op(plan, rb, &mut |_row| n += 1);
    n
}

fn exec_op(op: &Operator, rb: &RecordBatch, yld: &mut dyn FnMut(&[Val])) {
    match op {
        Operator::Scan => {
            for i in 0..rb.num_rows() {
                let row: Vec<Val> = (0..rb.num_columns()).map(|c| read(rb, c, i)).collect();
                yld(&row);
            }
        }
        Operator::Filter(pred, parent) => {
            exec_op(parent, rb, &mut |row| {
                if eval_pred(pred, row) {
                    yld(row);
                }
            });
        }
        Operator::Project(cols, parent) => {
            exec_op(parent, rb, &mut |row| {
                let projected: Vec<Val> = cols.iter().map(|&c| row[c]).collect();
                yld(&projected);
            });
        }
    }
}

fn read(rb: &RecordBatch, col: usize, i: usize) -> Val {
    let a = rb.column(col);
    match a.data_type() {
        DataType::Int32 => Val::I32(a.as_any().downcast_ref::<Int32Array>().unwrap().value(i)),
        DataType::Int64 => Val::I64(a.as_any().downcast_ref::<Int64Array>().unwrap().value(i)),
        DataType::Float64 => Val::F64(a.as_any().downcast_ref::<Float64Array>().unwrap().value(i)),
        other => panic!("unsupported column type: {other}"),
    }
}

fn eval_expr(e: &Expr, row: &[Val]) -> Val {
    match e {
        Expr::Col(c) => row[*c],
        Expr::LitI32(v) => Val::I32(*v),
    }
}

fn eval_pred(p: &Predicate, row: &[Val]) -> bool {
    match p {
        Predicate::Eq(a, b) => as_i32(eval_expr(a, row)) == as_i32(eval_expr(b, row)),
        Predicate::Lt(a, b) => as_i32(eval_expr(a, row)) < as_i32(eval_expr(b, row)),
    }
}

fn as_i32(v: Val) -> i32 {
    match v {
        Val::I32(x) => x,
        other => panic!("slice: expected i32, got {other:?}"),
    }
}
