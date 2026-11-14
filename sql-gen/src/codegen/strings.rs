//! Strings: a `Utf8View` value is a `Var<Str>`, an Arrow view and its table, so every
//! operation is `rust-lms`'s, on a batch's rows, literals and group keys alike.

use rust_lms::prelude::*;

use crate::value::{ColVal, Nullness};

use super::CodegenCtx;
use super::numeric::tag;

/// A string literal, baked in as a constant.
pub(crate) fn gen_str_literal(ctx: &Ctx, s: &str, cx: &CodegenCtx) -> ColVal {
    ColVal::Str(ctx.bind(cx.literal(s)), Nullness::NonNull)
}

pub(crate) fn is_str(cv: ColVal) -> bool {
    matches!(cv, ColVal::Str(..))
}

#[track_caller]
pub(crate) fn as_str(cv: ColVal) -> Var<Str> {
    match cv {
        ColVal::Str(s, _) => s,
        other => panic!("expected a string, got {}", tag(other)),
    }
}
