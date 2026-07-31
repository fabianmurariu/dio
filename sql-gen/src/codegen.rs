//! The staged twin of [`crate::exec`]: `gen_*` mirrors `exec_*` but *emits*
//! rust-lms staged ops. Specializing this interpreter to an [`Operator`] tree
//! yields a JIT kernel — the first Futamura projection for relational algebra
//! (per `docs/sql_to_c.pdf`).
//!
//! Scalar expressions are datafusion [`Expr`] values (columns, literals,
//! comparison / boolean / arithmetic operators). Nulls follow static
//! nullability: only nullable columns/exprs carry an `is_valid` bit, propagated
//! through evaluation; non-nullable ones emit no validity IR.

use arrow::datatypes::{DataType, Field, SchemaRef};
use arrow_lms::{ArrayBatchOps, FfiArray, MutBatchOps, PrimitiveArrayView};
use datafusion_common::ScalarValue;
use datafusion_expr::{BinaryExpr, Expr, Operator as DfOp};
use rust_lms::prelude::*;

use crate::plan::Operator;
use crate::value::{ColVal, Nullness, Row};

/// A staged expression yielding the read-only input batch (`&[FfiArray]`).
pub trait BatchSource: Staged<Out = SRef<'static, Slice<FfiArray>>> + Copy + 'static {}
impl<T> BatchSource for T where T: Staged<Out = SRef<'static, Slice<FfiArray>>> + Copy + 'static {}

/// A staged expression yielding the `&mut` output batch (`&mut [FfiArray]`).
pub trait OutSink: Staged<Out = SRefMut<'static, Slice<FfiArray>>> + Copy + 'static {}
impl<T> OutSink for T where T: Staged<Out = SRefMut<'static, Slice<FfiArray>>> + Copy + 'static {}

/// Downstream continuation, invoked once per emitted row at code-generation
/// time; the [`Row`] rides by value (cheap `Copy` handles).
type Yld = Box<dyn FnOnce(&mut Ctx, Row) + 'static>;

/// Emit a `count(*)` kernel: number of rows `plan` produces over `batch`.
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

/// Emit a materializing kernel: write each surviving projected row into `out`
/// and return the row count. `out_schema` is the output schema (from the plan).
pub fn gen_collect<B: BatchSource, O: OutSink>(
    ctx: &mut Ctx,
    batch: B,
    out: O,
    plan: &Operator,
    out_schema: &SchemaRef,
) -> Var<u64> {
    let n = ctx.var(0u64);
    let fields = out_schema.fields().clone();
    gen_op(
        plan,
        ctx,
        batch,
        Box::new(move |ctx, row| {
            for (c, (cv, field)) in row.iter().zip(fields.iter()).enumerate() {
                write_col(ctx, out, c, field, n, *cv);
            }
            ctx.store(n, add(n, 1u64));
        }),
    );
    n
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
                    let keep = gen_predicate(ctx, &predicate, &schema, &row);
                    ctx.if_then(keep, move |ctx| yld(ctx, row));
                }),
            );
        }

        Operator::Project { exprs, input, .. } => {
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

fn gen_scan<B: BatchSource>(ctx: &mut Ctx, batch: B, schema: SchemaRef, yld: Yld) {
    let len = gen_len(ctx, batch, schema.field(0).data_type());
    let i = ctx.var(0u64);
    let fields = schema.fields().clone();

    ctx.while_loop(lt(i, len), move |ctx| {
        let row: Row = fields
            .iter()
            .enumerate()
            .map(|(col, f)| gen_read(ctx, batch, col, f, i))
            .collect();
        yld(ctx, row);
        ctx.store(i, add(i, 1u64));
    });
}

fn gen_len<B: BatchSource>(ctx: &mut Ctx, batch: B, dt: &DataType) -> Var<u64> {
    match dt {
        DataType::Int32 => ctx.bind(batch.primitive::<i32>(0).len()),
        DataType::Int64 => ctx.bind(batch.primitive::<i64>(0).len()),
        DataType::Float64 => ctx.bind(batch.primitive::<f64>(0).len()),
        other => panic!("unsupported column type: {other}"),
    }
}

/// Read column `col` at row `i`, attaching validity iff the field is nullable.
fn gen_read<B: BatchSource>(
    ctx: &mut Ctx,
    batch: B,
    col: usize,
    field: &Field,
    i: Var<u64>,
) -> ColVal {
    let nullable = field.is_nullable();
    match field.data_type() {
        DataType::Int32 => {
            let view = batch.primitive::<i32>(col);
            let value = ctx.bind(view.value_unchecked(i));
            ColVal::I32(value, read_nullness(ctx, view, nullable, i))
        }
        DataType::Int64 => {
            let view = batch.primitive::<i64>(col);
            let value = ctx.bind(view.value_unchecked(i));
            ColVal::I64(value, read_nullness(ctx, view, nullable, i))
        }
        DataType::Float64 => {
            let view = batch.primitive::<f64>(col);
            let value = ctx.bind(view.value_unchecked(i));
            ColVal::F64(value, read_nullness(ctx, view, nullable, i))
        }
        other => panic!("unsupported column type: {other}"),
    }
}

fn read_nullness<P, M>(
    ctx: &mut Ctx,
    view: PrimitiveArrayView<P, M>,
    nullable: bool,
    i: Var<u64>,
) -> Nullness
where
    P: Staged<Out = SRef<'static, FfiArray>> + Clone + 'static,
    M: StagedType + 'static,
{
    if nullable {
        Nullness::Nullable(ctx.bind(view.validity().is_valid(i)))
    } else {
        Nullness::NonNull
    }
}

// =============================================================================
// Output materialization
// =============================================================================

fn write_col<O: OutSink>(ctx: &mut Ctx, out: O, c: usize, field: &Field, n: Var<u64>, cv: ColVal) {
    match field.data_type() {
        DataType::Int32 => {
            let view = out.column_mut::<i32>(c);
            let v = coerce_i32(ctx, cv);
            view.set(ctx, n, v);
            write_null(ctx, view, field, n, cv);
        }
        DataType::Int64 => {
            let view = out.column_mut::<i64>(c);
            let v = coerce_i64(ctx, cv);
            view.set(ctx, n, v);
            write_null(ctx, view, field, n, cv);
        }
        DataType::Float64 => {
            let view = out.column_mut::<f64>(c);
            let v = coerce_f64(ctx, cv);
            view.set(ctx, n, v);
            write_null(ctx, view, field, n, cv);
        }
        other => panic!("unsupported output column type: {other}"),
    }
}

fn write_null<P, M>(
    ctx: &mut Ctx,
    view: PrimitiveArrayView<P, M>,
    field: &Field,
    n: Var<u64>,
    cv: ColVal,
) where
    P: Staged<Out = SRefMut<'static, FfiArray>> + Clone + 'static,
    M: StagedType + 'static,
{
    // Only nullable outputs need validity writes, and only nullable values can
    // be null (the bitmap starts all-valid, so non-null rows need no write).
    if let (true, Nullness::Nullable(valid)) = (field.is_nullable(), cv.nullness()) {
        ctx.if_then(not(valid), move |ctx| view.set_null(ctx, n));
    }
}

// =============================================================================
// Expression evaluation
// =============================================================================

fn gen_predicate(ctx: &mut Ctx, e: &Expr, schema: &SchemaRef, row: &Row) -> Var<bool> {
    let cv = gen_expr(ctx, e, schema, row);
    let cond = as_bool(cv);
    // SQL: a NULL predicate does not pass the filter -> keep iff (valid && cond).
    match cv.nullness() {
        Nullness::NonNull => cond,
        Nullness::Nullable(valid) => ctx.bind(select(valid, cond, Const::<bool>::new(false))),
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
        ScalarValue::Int32(Some(v)) => ColVal::I32(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Int64(Some(v)) => ColVal::I64(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Float64(Some(v)) => ColVal::F64(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Boolean(Some(v)) => {
            ColVal::Bool(ctx.bind(Const::<bool>::new(*v)), Nullness::NonNull)
        }
        other => panic!("unsupported literal: {other:?}"),
    }
}

fn gen_binary(ctx: &mut Ctx, be: &BinaryExpr, schema: &SchemaRef, row: &Row) -> ColVal {
    let l = gen_expr(ctx, &be.left, schema, row);
    let r = gen_expr(ctx, &be.right, schema, row);
    let null = combine_null(ctx, l.nullness(), r.nullness());
    match be.op {
        DfOp::Eq => ColVal::Bool(num_cmp(ctx, Cmp::Eq, l, r), null),
        DfOp::NotEq => {
            let e = num_cmp(ctx, Cmp::Eq, l, r);
            ColVal::Bool(ctx.bind(not(e)), null)
        }
        DfOp::Lt => ColVal::Bool(num_cmp(ctx, Cmp::Lt, l, r), null),
        DfOp::Gt => ColVal::Bool(num_cmp(ctx, Cmp::Gt, l, r), null),
        DfOp::LtEq => {
            let e = num_cmp(ctx, Cmp::Gt, l, r);
            ColVal::Bool(ctx.bind(not(e)), null)
        }
        DfOp::GtEq => {
            let e = num_cmp(ctx, Cmp::Lt, l, r);
            ColVal::Bool(ctx.bind(not(e)), null)
        }
        DfOp::And => {
            let (a, b) = (as_bool(l), as_bool(r));
            ColVal::Bool(ctx.bind(select(a, b, Const::<bool>::new(false))), null)
        }
        DfOp::Or => {
            let (a, b) = (as_bool(l), as_bool(r));
            ColVal::Bool(ctx.bind(select(a, Const::<bool>::new(true), b)), null)
        }
        DfOp::Plus => arith(ctx, Arith::Add, l, r, null),
        DfOp::Minus => arith(ctx, Arith::Sub, l, r, null),
        DfOp::Multiply => arith(ctx, Arith::Mul, l, r, null),
        other => panic!("unsupported binary operator: {other:?}"),
    }
}

/// `NonNull` unless an operand is nullable; two nullable operands AND their bits.
fn combine_null(ctx: &mut Ctx, a: Nullness, b: Nullness) -> Nullness {
    match (a, b) {
        (Nullness::NonNull, Nullness::NonNull) => Nullness::NonNull,
        (Nullness::Nullable(v), Nullness::NonNull) | (Nullness::NonNull, Nullness::Nullable(v)) => {
            Nullness::Nullable(v)
        }
        (Nullness::Nullable(x), Nullness::Nullable(y)) => {
            Nullness::Nullable(ctx.bind(select(x, y, Const::<bool>::new(false))))
        }
    }
}

enum Cmp {
    Eq,
    Lt,
    Gt,
}

/// Compare two values -> bool. Floats compare as `f64`; ints widen to `i64`.
fn num_cmp(ctx: &mut Ctx, kind: Cmp, l: ColVal, r: ColVal) -> Var<bool> {
    if let (ColVal::F64(x, _), ColVal::F64(y, _)) = (l, r) {
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

enum Arith {
    Add,
    Sub,
    Mul,
}

/// Arithmetic -> numeric ColVal. Floats stay `f64`; ints widen to `i64`.
fn arith(ctx: &mut Ctx, kind: Arith, l: ColVal, r: ColVal, null: Nullness) -> ColVal {
    if let (ColVal::F64(x, _), ColVal::F64(y, _)) = (l, r) {
        let v = match kind {
            Arith::Add => ctx.bind(add(x, y)),
            Arith::Sub => ctx.bind(sub(x, y)),
            Arith::Mul => ctx.bind(mul(x, y)),
        };
        return ColVal::F64(v, null);
    }
    let x = to_i64(ctx, l);
    let y = to_i64(ctx, r);
    let v = match kind {
        Arith::Add => ctx.bind(add(x, y)),
        Arith::Sub => ctx.bind(sub(x, y)),
        Arith::Mul => ctx.bind(mul(x, y)),
    };
    ColVal::I64(v, null)
}

// =============================================================================
// Value extraction / coercion
// =============================================================================

fn to_i64(ctx: &mut Ctx, cv: ColVal) -> Var<i64> {
    match cv {
        ColVal::I64(v, _) => v,
        ColVal::I32(v, _) => ctx.bind(int_cast::<i64, i32, _>(v)),
        other => panic!("expected integer operand, got {}", tag(other)),
    }
}

fn coerce_i32(ctx: &mut Ctx, cv: ColVal) -> Var<i32> {
    match cv {
        ColVal::I32(v, _) => v,
        ColVal::I64(v, _) => ctx.bind(int_cast::<i32, i64, _>(v)),
        other => panic!("cannot coerce {} to i32", tag(other)),
    }
}

fn coerce_i64(ctx: &mut Ctx, cv: ColVal) -> Var<i64> {
    to_i64(ctx, cv)
}

fn coerce_f64(_ctx: &mut Ctx, cv: ColVal) -> Var<f64> {
    match cv {
        ColVal::F64(v, _) => v,
        other => panic!("cannot coerce {} to f64", tag(other)),
    }
}

fn as_bool(cv: ColVal) -> Var<bool> {
    match cv {
        ColVal::Bool(v, _) => v,
        other => panic!("expected bool, got {}", tag(other)),
    }
}

fn tag(cv: ColVal) -> &'static str {
    match cv {
        ColVal::I32(..) => "i32",
        ColVal::I64(..) => "i64",
        ColVal::F64(..) => "f64",
        ColVal::Bool(..) => "bool",
    }
}
