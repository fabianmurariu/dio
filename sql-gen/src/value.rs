//! The mixed-stage row — a static schema paired with dynamic (stage-1) fields.
//!
//! Rust translation of the paper's `Record(Vector[Rep[T]], Schema)`. Because the
//! query is interpreted at runtime, fields are a type-erased `Vec<ColVal>` rather
//! than a compile-time tuple. Each `ColVal` carries its physical type as a static
//! enum tag, a `Copy` `Var` value, and a static-nullability tag: non-nullable
//! columns/exprs carry **no** validity `Var` at all (zero overhead), nullable
//! ones carry an `is_valid` bit that propagates through expression evaluation.

use rust_lms::prelude::*;

/// Static-nullability tag. `NonNull` emits no validity IR; `Nullable` carries a
/// stage-1 `is_valid` bit.
#[derive(Clone, Copy)]
pub enum Nullness {
    NonNull,
    Nullable(Var<bool>),
}

/// A staged column value: static physical-type tag + `Var` value + nullness.
#[derive(Clone, Copy)]
pub enum ColVal {
    I32(Var<i32>, Nullness),
    I64(Var<i64>, Nullness),
    F64(Var<f64>, Nullness),
    Bool(Var<bool>, Nullness),
}

impl ColVal {
    pub fn nullness(self) -> Nullness {
        match self {
            ColVal::I32(_, n) | ColVal::I64(_, n) | ColVal::F64(_, n) | ColVal::Bool(_, n) => n,
        }
    }
}

/// A row: one [`ColVal`] per column, positional.
pub type Row = Vec<ColVal>;
