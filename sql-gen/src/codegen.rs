//! The staged twin of [`crate::exec`]: `gen_*` mirrors `exec_*` one-to-one but
//! *emits* rust-lms staged ops instead of running them. Specializing this
//! interpreter to an [`Operator`] tree yields a JIT kernel — the first Futamura
//! projection for relational algebra (per `docs/sql_to_c.pdf`).
//!
//! Scalar expressions are datafusion [`Expr`] values; we evaluate the subset our
//! "very simple queries" produce (columns, integer/float/bool literals,
//! comparison and boolean operators).

use arrow::datatypes::{DataType, SchemaRef};
use arrow_lms::{FfiArrayBatch, FfiArrayBatchOps};
use datafusion_common::ScalarValue;
use datafusion_expr::{BinaryExpr, Expr, Operator as DfOp};
use rust_lms::prelude::*;

use crate::plan::Operator;
use crate::value::{ColVal, Row};

/// A staged expression yielding the input batch descriptor. Everything inside
/// the JIT kernel is `'static` (the descriptor is passed by reference at call
/// time), so a plain `Var<SRef<FfiArrayBatch>>` satisfies this.
pub trait BatchSource:
    Staged<Out = SRef<'static, FfiArrayBatch<'static, 'static>>> + Copy + 'static
{
}

impl<T> BatchSource for T where
    T: Staged<Out = SRef<'static, FfiArrayBatch<'static, 'static>>> + Copy + 'static
{
}

/// Downstream continuation, invoked once per emitted row at code-generation
/// time. It owns its captures so it can move through `'static` staged
/// loops/branches; the [`Row`] rides by value (cheap `Copy` handles).
type Yld = Box<dyn FnOnce(&mut Ctx, Row) + 'static>;

/// Emit a kernel counting the rows `plan` produces over `batch` — the staged
/// twin of [`crate::exec::exec_count`], and a `count(*)` terminal.
pub fn gen_count<B: BatchSource>(ctx: &mut Ctx, batch: B, plan: &Operator) -> Var<i64> {
    let acc = ctx.var(0i64);
    gen_op(
        plan,
        ctx,
        batch,
        Box::new(move |ctx, _row| ctx.store(acc, add(acc, 1i64))),
    );
    acc
}

fn gen_op<B: BatchSource>(op: &Operator, ctx: &mut Ctx, batch: B, yld: Yld) {
    match op {
        Operator::Scan { schema } => gen_scan(ctx, batch, schema.clone(), yld),

        Operator::Filter { predicate, input } => {
            let predicate = predicate.clone();
            let schema = input.output_schema();
            gen_op(
                input,
                ctx,
                batch,
                Box::new(move |ctx, row| {
                    let cond = gen_predicate(ctx, &predicate, &schema, &row);
                    ctx.if_then(cond, move |ctx| yld(ctx, row));
                }),
            );
        }

        Operator::Project { exprs, input } => {
            let exprs = exprs.clone();
            let schema = input.output_schema();
            gen_op(
                input,
                ctx,
                batch,
                Box::new(move |ctx, row| {
                    let projected: Row = exprs
                        .iter()
                        .map(|e| gen_expr(ctx, e, &schema, &row))
                        .collect();
                    yld(ctx, projected);
                }),
            );
        }
    }
}

/// Drive the row loop over the input batch, building a mixed-stage [`Row`] per
/// index. This is where static `DataType` → staged read dispatch lives.
fn gen_scan<B: BatchSource>(ctx: &mut Ctx, batch: B, schema: SchemaRef, yld: Yld) {
    let len = gen_len(ctx, batch, schema.field(0).data_type());
    let i = ctx.var(0u64);
    let fields = schema.fields().clone();

    ctx.while_loop(lt(i, len), move |ctx| {
        let row: Row = fields
            .iter()
            .enumerate()
            .map(|(col, f)| gen_read(ctx, batch, col, f.data_type(), i))
            .collect();
        yld(ctx, row);
        ctx.store(i, add(i, 1u64));
    });
}

/// Bind the batch length (element count of column 0; all columns share it).
fn gen_len<B: BatchSource>(ctx: &mut Ctx, batch: B, dt: &DataType) -> Var<u64> {
    match dt {
        DataType::Int32 => ctx.bind(batch.primitive::<i32>(0).len()),
        DataType::Int64 => ctx.bind(batch.primitive::<i64>(0).len()),
        DataType::Float64 => ctx.bind(batch.primitive::<f64>(0).len()),
        other => panic!("unsupported column type: {other}"),
    }
}

/// Read column `col` at row `i` as a typed staged value.
fn gen_read<B: BatchSource>(
    ctx: &mut Ctx,
    batch: B,
    col: usize,
    dt: &DataType,
    i: Var<u64>,
) -> ColVal {
    match dt {
        DataType::Int32 => ColVal::I32(ctx.bind(batch.primitive::<i32>(col).value_unchecked(i))),
        DataType::Int64 => ColVal::I64(ctx.bind(batch.primitive::<i64>(col).value_unchecked(i))),
        DataType::Float64 => ColVal::F64(ctx.bind(batch.primitive::<f64>(col).value_unchecked(i))),
        other => panic!("unsupported column type: {other}"),
    }
}

fn gen_predicate(ctx: &mut Ctx, e: &Expr, schema: &SchemaRef, row: &Row) -> Var<bool> {
    match gen_expr(ctx, e, schema, row) {
        ColVal::Bool(v) => v,
        _ => panic!("predicate did not evaluate to bool: {e:?}"),
    }
}

fn gen_expr(ctx: &mut Ctx, e: &Expr, schema: &SchemaRef, row: &Row) -> ColVal {
    match e {
        Expr::Column(c) => {
            let idx = schema
                .index_of(&c.name)
                .unwrap_or_else(|_| panic!("unknown column: {}", c.name));
            row[idx]
        }
        Expr::Literal(sv, _) => gen_literal(ctx, sv),
        Expr::BinaryExpr(be) => gen_binary(ctx, be, schema, row),
        other => panic!("unsupported expression: {other:?}"),
    }
}

fn gen_literal(ctx: &mut Ctx, sv: &ScalarValue) -> ColVal {
    match sv {
        ScalarValue::Int32(Some(v)) => ColVal::I32(ctx.var(*v)),
        ScalarValue::Int64(Some(v)) => ColVal::I64(ctx.var(*v)),
        ScalarValue::Float64(Some(v)) => ColVal::F64(ctx.var(*v)),
        ScalarValue::Boolean(Some(v)) => ColVal::Bool(ctx.bind(Const::<bool>::new(*v))),
        other => panic!("unsupported literal: {other:?}"),
    }
}

fn gen_binary(ctx: &mut Ctx, be: &BinaryExpr, schema: &SchemaRef, row: &Row) -> ColVal {
    let l = gen_expr(ctx, &be.left, schema, row);
    let r = gen_expr(ctx, &be.right, schema, row);
    let out = match be.op {
        DfOp::Eq => num_cmp(ctx, Cmp::Eq, l, r),
        DfOp::NotEq => {
            let e = num_cmp(ctx, Cmp::Eq, l, r);
            ctx.bind(not(e))
        }
        DfOp::Lt => num_cmp(ctx, Cmp::Lt, l, r),
        DfOp::Gt => num_cmp(ctx, Cmp::Gt, l, r),
        DfOp::LtEq => {
            let e = num_cmp(ctx, Cmp::Gt, l, r);
            ctx.bind(not(e))
        }
        DfOp::GtEq => {
            let e = num_cmp(ctx, Cmp::Lt, l, r);
            ctx.bind(not(e))
        }
        // Branchless logical connectives on bool operands.
        DfOp::And => {
            let (a, b) = (as_bool(l), as_bool(r));
            ctx.bind(select(a, b, Const::<bool>::new(false)))
        }
        DfOp::Or => {
            let (a, b) = (as_bool(l), as_bool(r));
            ctx.bind(select(a, Const::<bool>::new(true), b))
        }
        other => panic!("unsupported binary operator: {other:?}"),
    };
    ColVal::Bool(out)
}

enum Cmp {
    Eq,
    Lt,
    Gt,
}

/// Compare two numeric column values, producing a `bool`. Floats compare as
/// `f64`; everything else is widened to `i64`.
fn num_cmp(ctx: &mut Ctx, kind: Cmp, l: ColVal, r: ColVal) -> Var<bool> {
    if let (ColVal::F64(x), ColVal::F64(y)) = (l, r) {
        return match kind {
            Cmp::Eq => ctx.bind(eq(x, y)),
            Cmp::Lt => ctx.bind(lt(x, y)),
            Cmp::Gt => ctx.bind(gt(x, y)),
        };
    }
    let x = to_i64(ctx, l);
    let y = to_i64(ctx, r);
    match kind {
        Cmp::Eq => ctx.bind(eq(x, y)),
        Cmp::Lt => ctx.bind(lt(x, y)),
        Cmp::Gt => ctx.bind(gt(x, y)),
    }
}

fn to_i64(ctx: &mut Ctx, c: ColVal) -> Var<i64> {
    match c {
        ColVal::I64(v) => v,
        ColVal::I32(v) => ctx.bind(int_cast::<i64, i32, _>(v)),
        other => panic!("expected integer operand, got {}", tag(other)),
    }
}

fn as_bool(c: ColVal) -> Var<bool> {
    match c {
        ColVal::Bool(v) => v,
        other => panic!("expected bool operand, got {}", tag(other)),
    }
}

fn tag(c: ColVal) -> &'static str {
    match c {
        ColVal::I32(_) => "i32",
        ColVal::I64(_) => "i64",
        ColVal::F64(_) => "f64",
        ColVal::Bool(_) => "bool",
    }
}
