//! Plain-Rust reference interpreter — the un-staged twin of [`crate::codegen`].
//!
//! It runs the same [`Operator`] tree directly over a `RecordBatch`, evaluating
//! the same datafusion [`Expr`] subset. Its only job is to be an independent
//! oracle for tests: the JIT kernel must agree with it (Futamura equivalence).
//! Temporary — it will be dropped once codegen is trusted / too costly to mirror.

use arrow::array::{Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::ScalarValue;
use datafusion_expr::{BinaryExpr, Expr, Operator as DfOp};

use crate::plan::Operator;

/// A host-side (un-staged) value — the reference twin of [`crate::value::ColVal`].
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Val {
    I32(i32),
    I64(i64),
    F64(f64),
    Bool(bool),
}

/// Count the rows produced by `plan` over `rb`.
pub fn exec_count(plan: &Operator, rb: &RecordBatch) -> i64 {
    let mut n = 0i64;
    exec_op(plan, rb, &mut |_row| n += 1);
    n
}

fn exec_op(op: &Operator, rb: &RecordBatch, yld: &mut dyn FnMut(&[Val])) {
    match op {
        Operator::Scan { schema } => {
            for i in 0..rb.num_rows() {
                let row: Vec<Val> = (0..schema.fields().len())
                    .map(|c| read(rb, c, schema.field(c).data_type(), i))
                    .collect();
                yld(&row);
            }
        }
        Operator::Filter { predicate, input } => {
            let schema = input.output_schema();
            exec_op(input, rb, &mut |row| {
                if as_bool(eval_expr(predicate, &schema, row)) {
                    yld(row);
                }
            });
        }
        Operator::Project { exprs, input } => {
            let schema = input.output_schema();
            exec_op(input, rb, &mut |row| {
                let projected: Vec<Val> =
                    exprs.iter().map(|e| eval_expr(e, &schema, row)).collect();
                yld(&projected);
            });
        }
    }
}

fn read(rb: &RecordBatch, col: usize, dt: &DataType, i: usize) -> Val {
    let a = rb.column(col);
    match dt {
        DataType::Int32 => Val::I32(a.as_any().downcast_ref::<Int32Array>().unwrap().value(i)),
        DataType::Int64 => Val::I64(a.as_any().downcast_ref::<Int64Array>().unwrap().value(i)),
        DataType::Float64 => Val::F64(a.as_any().downcast_ref::<Float64Array>().unwrap().value(i)),
        other => panic!("unsupported column type: {other}"),
    }
}

fn eval_expr(e: &Expr, schema: &SchemaRef, row: &[Val]) -> Val {
    match e {
        Expr::Column(c) => {
            let idx = schema
                .index_of(&c.name)
                .unwrap_or_else(|_| panic!("unknown column: {}", c.name));
            row[idx]
        }
        Expr::Literal(sv, _) => literal(sv),
        Expr::BinaryExpr(be) => binary(be, schema, row),
        other => panic!("unsupported expression: {other:?}"),
    }
}

fn literal(sv: &ScalarValue) -> Val {
    match sv {
        ScalarValue::Int32(Some(v)) => Val::I32(*v),
        ScalarValue::Int64(Some(v)) => Val::I64(*v),
        ScalarValue::Float64(Some(v)) => Val::F64(*v),
        ScalarValue::Boolean(Some(v)) => Val::Bool(*v),
        other => panic!("unsupported literal: {other:?}"),
    }
}

fn binary(be: &BinaryExpr, schema: &SchemaRef, row: &[Val]) -> Val {
    let l = eval_expr(&be.left, schema, row);
    let r = eval_expr(&be.right, schema, row);
    let out = match be.op {
        DfOp::Eq => num_cmp(l, r, |o| o == std::cmp::Ordering::Equal),
        DfOp::NotEq => num_cmp(l, r, |o| o != std::cmp::Ordering::Equal),
        DfOp::Lt => num_cmp(l, r, |o| o == std::cmp::Ordering::Less),
        DfOp::Gt => num_cmp(l, r, |o| o == std::cmp::Ordering::Greater),
        DfOp::LtEq => num_cmp(l, r, |o| o != std::cmp::Ordering::Greater),
        DfOp::GtEq => num_cmp(l, r, |o| o != std::cmp::Ordering::Less),
        DfOp::And => as_bool(l) && as_bool(r),
        DfOp::Or => as_bool(l) || as_bool(r),
        other => panic!("unsupported binary operator: {other:?}"),
    };
    Val::Bool(out)
}

fn num_cmp(l: Val, r: Val, pred: impl Fn(std::cmp::Ordering) -> bool) -> bool {
    let ord = if let (Val::F64(x), Val::F64(y)) = (l, r) {
        x.partial_cmp(&y).expect("NaN comparison")
    } else {
        to_i64(l).cmp(&to_i64(r))
    };
    pred(ord)
}

fn to_i64(v: Val) -> i64 {
    match v {
        Val::I64(x) => x,
        Val::I32(x) => x as i64,
        other => panic!("expected integer operand, got {other:?}"),
    }
}

fn as_bool(v: Val) -> bool {
    match v {
        Val::Bool(b) => b,
        other => panic!("expected bool operand, got {other:?}"),
    }
}
