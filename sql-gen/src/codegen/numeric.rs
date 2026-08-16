//! Numeric value extraction / coercion over the type-erased [`ColVal`].

use rust_lms::prelude::*;

use crate::value::ColVal;

pub(crate) fn to_i64(ctx: &mut Ctx, cv: ColVal) -> Var<i64> {
    match cv {
        ColVal::I64(v, _) => v,
        ColVal::I32(v, _) => ctx.bind(int_cast::<i64, i32, _>(v)),
        other => panic!("expected integer operand, got {}", tag(other)),
    }
}

pub(crate) fn coerce_i32(ctx: &mut Ctx, cv: ColVal) -> Var<i32> {
    match cv {
        ColVal::I32(v, _) => v,
        ColVal::I64(v, _) => ctx.bind(int_cast::<i32, i64, _>(v)),
        other => panic!("cannot coerce {} to i32", tag(other)),
    }
}

pub(crate) fn coerce_f64(_ctx: &mut Ctx, cv: ColVal) -> Var<f64> {
    match cv {
        ColVal::F64(v, _) => v,
        other => panic!("cannot coerce {} to f64", tag(other)),
    }
}

/// Convert any numeric column value to `f64` (int → float). Used by `avg`.
pub(crate) fn to_f64(ctx: &mut Ctx, cv: ColVal) -> Var<f64> {
    match cv {
        ColVal::F64(v, _) => v,
        ColVal::I64(v, _) => ctx.bind(int_to_float::<f64, i64, _>(v)),
        ColVal::I32(v, _) => ctx.bind(int_to_float::<f64, i32, _>(v)),
        other => panic!("cannot convert {} to f64", tag(other)),
    }
}

pub(crate) fn as_bool(cv: ColVal) -> Var<bool> {
    match cv {
        ColVal::Bool(v, _) => v,
        other => panic!("expected bool, got {}", tag(other)),
    }
}

/// Canonicalize an `f64` GROUP BY key before bit-keying it, so bit-equality matches
/// SQL float grouping: map `-0.0` to `+0.0` (they compare equal) and every NaN to one
/// canonical NaN (NaN ≠ NaN, so distinct payloads would otherwise split). Branchless.
pub(crate) fn canonical_f64(ctx: &mut Ctx, key: Var<f64>) -> Var<f64> {
    let is_zero = ctx.bind(eq(key, 0.0f64));
    let no_neg_zero = ctx.bind(select(is_zero, 0.0f64, key));
    let is_number = ctx.bind(eq(no_neg_zero, no_neg_zero)); // false only for NaN
    ctx.bind(select(is_number, no_neg_zero, f64::NAN))
}

pub(crate) fn tag(cv: ColVal) -> &'static str {
    match cv {
        ColVal::I32(..) => "i32",
        ColVal::I64(..) => "i64",
        ColVal::F64(..) => "f64",
        ColVal::Bool(..) => "bool",
        ColVal::Str { .. } => "str",
    }
}
