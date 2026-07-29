//! The staged twin of [`crate::exec`]: `gen_*` mirrors `exec_*` one-to-one but
//! *emits* rust-lms staged ops instead of running them. Specializing this
//! interpreter to a given `Operator` tree yields a JIT kernel — the first
//! Futamura projection for relational algebra (per `docs/sql_to_c.pdf`).
//!
//! The push model maps directly: an operator is handed a continuation `Yld` it
//! invokes for each row it emits, exactly like the paper's `yld: Record => Unit`.

use arrow::datatypes::DataType;
use arrow_lms::{FfiArrayBatch, FfiArrayBatchOps};
use rust_lms::prelude::*;

use crate::plan::{Expr, Operator, Predicate, Schema};
use crate::value::{ColVal, Row};

/// A staged expression yielding the input batch descriptor.
///
/// Everything inside the JIT kernel is `'static` (the descriptor is passed by
/// reference at call time), so a plain `Var<SRef<FfiArrayBatch>>` satisfies this.
pub trait BatchSource:
    Staged<Out = SRef<'static, FfiArrayBatch<'static, 'static>>> + Copy + 'static
{
}

impl<T> BatchSource for T where
    T: Staged<Out = SRef<'static, FfiArrayBatch<'static, 'static>>> + Copy + 'static
{
}

/// Downstream continuation, invoked once per emitted row *at code-generation
/// time*. It owns its captures so it can move through `'static` staged
/// loops/branches; the [`Row`] it receives is passed by value (cheap `Copy`
/// handles).
type Yld = Box<dyn FnOnce(&mut Ctx, Row) + 'static>;

/// Emit a kernel counting the rows `plan` produces over `batch` — the staged
/// twin of [`crate::exec::exec_count`], and a `count(*)` terminal.
pub fn gen_count<B: BatchSource>(
    ctx: &mut Ctx,
    batch: B,
    plan: &Operator,
    schema: &Schema,
) -> Var<i64> {
    let acc = ctx.var(0i64);
    gen_op(
        plan,
        ctx,
        batch,
        schema,
        Box::new(move |ctx, _row| ctx.store(acc, add(acc, 1i64))),
    );
    acc
}

fn gen_op<B: BatchSource>(op: &Operator, ctx: &mut Ctx, batch: B, schema: &Schema, yld: Yld) {
    match op {
        Operator::Scan => gen_scan(ctx, batch, schema, yld),

        Operator::Filter(pred, parent) => {
            let pred = pred.clone();
            gen_op(
                parent,
                ctx,
                batch,
                schema,
                Box::new(move |ctx, row| {
                    let cond = gen_pred(ctx, &pred, &row);
                    ctx.if_then(cond, move |ctx| yld(ctx, row));
                }),
            );
        }

        Operator::Project(cols, parent) => {
            let cols = cols.clone();
            gen_op(
                parent,
                ctx,
                batch,
                schema,
                Box::new(move |ctx, row| {
                    let projected: Row = cols.iter().map(|&c| row[c]).collect();
                    yld(ctx, projected);
                }),
            );
        }
    }
}

/// Drive the row loop over the input batch, building a mixed-stage [`Row`] per
/// index and pushing it downstream. This is where static `DataType` → staged
/// read dispatch lives.
fn gen_scan<B: BatchSource>(ctx: &mut Ctx, batch: B, schema: &Schema, yld: Yld) {
    let len = gen_len(ctx, batch, &schema.0[0]);
    let i = ctx.var(0u64);
    let schema = schema.clone();

    ctx.while_loop(lt(i, len), move |ctx| {
        let row: Row = schema
            .0
            .iter()
            .enumerate()
            .map(|(col, dt)| gen_read(ctx, batch, col, dt, i))
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
        other => panic!("unsupported column type in slice: {other}"),
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
        other => panic!("unsupported column type in slice: {other}"),
    }
}

fn gen_expr(ctx: &mut Ctx, e: &Expr, row: &Row) -> ColVal {
    match e {
        Expr::Col(c) => row[*c],
        Expr::LitI32(v) => ColVal::I32(ctx.var(*v)),
    }
}

fn gen_pred(ctx: &mut Ctx, p: &Predicate, row: &Row) -> Var<bool> {
    match p {
        Predicate::Eq(a, b) => {
            let (x, y) = (as_i32(gen_expr(ctx, a, row)), as_i32(gen_expr(ctx, b, row)));
            ctx.bind(eq(x, y))
        }
        Predicate::Lt(a, b) => {
            let (x, y) = (as_i32(gen_expr(ctx, a, row)), as_i32(gen_expr(ctx, b, row)));
            ctx.bind(lt(x, y))
        }
    }
}

fn as_i32(c: ColVal) -> Var<i32> {
    match c {
        ColVal::I32(v) => v,
        _ => panic!("slice: expected i32 operand"),
    }
}
