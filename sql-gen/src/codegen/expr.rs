//! Scalar expression evaluation: datafusion [`Expr`] values (columns, literals,
//! comparison / boolean / arithmetic operators) lowered to staged [`ColVal`]s.

use arrow::datatypes::SchemaRef;
use datafusion_common::ScalarValue;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{BinaryExpr, Expr, Operator as DfOp};
use rust_lms::prelude::*;

use crate::value::{ColVal, Nullness, Row, StrVal};

use super::CodegenCtx;
use super::numeric::{as_bool, tag, to_i64};
use super::strings::{gen_str_literal, is_str, str_eq};

pub(crate) fn gen_predicate(
    ctx: &mut Ctx,
    e: &Expr,
    schema: &SchemaRef,
    row: &Row,
    cx: &CodegenCtx,
) -> Var<bool> {
    let cv = gen_expr(ctx, e, schema, row, cx);
    let cond = as_bool(cv);
    // SQL: a NULL predicate does not pass the filter -> keep iff (valid && cond).
    match cv.nullness() {
        Nullness::NonNull => cond,
        Nullness::Nullable(valid) => ctx.bind(select(valid, cond, Const::<bool>::new(false))),
    }
}

pub(crate) fn gen_expr(
    ctx: &mut Ctx,
    e: &Expr,
    schema: &SchemaRef,
    row: &Row,
    cx: &CodegenCtx,
) -> ColVal {
    match e {
        Expr::Column(c) => {
            let idx = schema
                .index_of(&c.name)
                .unwrap_or_else(|_| panic!("unknown column: {}", c.name));
            row[idx]
        }
        Expr::Literal(sv, _) => gen_literal(ctx, sv, cx),
        Expr::BinaryExpr(be) => gen_binary(ctx, be, schema, row, cx),
        Expr::ScalarFunction(f) => gen_scalar_fn(ctx, f, schema, row, cx),
        // An alias is a pure rename: evaluate the inner expression. The output
        // schema already carries the alias name, so nothing else is needed.
        Expr::Alias(a) => gen_expr(ctx, &a.expr, schema, row, cx),
        other => panic!("unsupported expression: {other:?}"),
    }
}

fn gen_scalar_fn(
    ctx: &mut Ctx,
    f: &ScalarFunction,
    schema: &SchemaRef,
    row: &Row,
    cx: &CodegenCtx,
) -> ColVal {
    match f.func.name().to_ascii_lowercase().as_str() {
        // Octet length: the byte length, from the view (`lo & 0xFFFF_FFFF`) for a
        // column or directly from `len` for resolved bytes.
        "octet_length" => match gen_expr(ctx, &f.args[0], schema, row, cx) {
            ColVal::Str(sv, null) => {
                let len = match sv {
                    StrVal::Column { lo, .. } => ctx.bind(bitand::<u64, _, _>(lo, 0xFFFF_FFFFu64)),
                    StrVal::Bytes { len, .. } => len,
                };
                ColVal::I64(ctx.bind(int_cast::<i64, u64, _>(len)), null)
            }
            other => panic!("octet_length expects a string, got {}", tag(other)),
        },
        other => panic!("unsupported scalar function: {other}"),
    }
}
/// Equality: string operands compare by view/bytes, numeric operands by value.
fn gen_eq(ctx: &mut Ctx, l: ColVal, r: ColVal, cx: &CodegenCtx) -> Var<bool> {
    match (l, r) {
        (ColVal::Str(a, _), ColVal::Str(b, _)) => str_eq(ctx, a, b, cx),
        _ if is_str(l) || is_str(r) => panic!("string compared with non-string"),
        _ => num_cmp(ctx, Cmp::Eq, l, r),
    }
}
fn gen_literal(ctx: &mut Ctx, sv: &ScalarValue, cx: &CodegenCtx) -> ColVal {
    match sv {
        ScalarValue::Int32(Some(v)) => ColVal::I32(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Int64(Some(v)) => ColVal::I64(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Float64(Some(v)) => ColVal::F64(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Boolean(Some(v)) => {
            ColVal::Bool(ctx.bind(Const::<bool>::new(*v)), Nullness::NonNull)
        }
        ScalarValue::Utf8(Some(s))
        | ScalarValue::Utf8View(Some(s))
        | ScalarValue::LargeUtf8(Some(s)) => gen_str_literal(ctx, s, cx),
        other => panic!("unsupported literal: {other:?}"),
    }
}
fn gen_binary(
    ctx: &mut Ctx,
    be: &BinaryExpr,
    schema: &SchemaRef,
    row: &Row,
    cx: &CodegenCtx,
) -> ColVal {
    let l = gen_expr(ctx, &be.left, schema, row, cx);
    let r = gen_expr(ctx, &be.right, schema, row, cx);
    let null = combine_null(ctx, l.nullness(), r.nullness());
    match be.op {
        DfOp::Eq => ColVal::Bool(gen_eq(ctx, l, r, cx), null),
        DfOp::NotEq => {
            let e = gen_eq(ctx, l, r, cx);
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
