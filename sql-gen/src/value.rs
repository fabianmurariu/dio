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
/// `Str` is a `Utf8View` string (see [`StrVal`]).
#[derive(Clone, Copy)]
pub enum ColVal {
    I32(Var<i32>, Nullness),
    I64(Var<i64>, Nullness),
    F64(Var<f64>, Nullness),
    Bool(Var<bool>, Nullness),
    Str(StrVal, Nullness),
}

/// A staged `Utf8View` string. The 16-byte view is carried as two `u64` halves
/// (`lo` = `[len:u32][…]`, `hi` = the rest) — enough for `octet_length`
/// (`lo & 0xFFFF_FFFF`) and inline-string equality (compare both halves).
///
/// A `Column` string also carries the originating array pointer + row so the
/// extern fallback can read its bytes. A `Literal` carries its baked bytes
/// (`bytes`/`blen`) for that fallback; `inline` (stage-0) says whether it fits in
/// a view (`≤12` bytes), so `str_eq` can pick the pure-staged path.
#[derive(Clone, Copy)]
pub enum StrVal {
    Column {
        lo: Var<u64>,
        hi: Var<u64>,
        array: Var<u64>,
        row: Var<u64>,
    },
    Literal {
        lo: Var<u64>,
        hi: Var<u64>,
        bytes: Var<u64>,
        blen: Var<u64>,
        inline: bool,
    },
}

impl StrVal {
    /// The view's first half (`[len:u32][…]`) — present in both variants.
    pub fn lo(self) -> Var<u64> {
        match self {
            StrVal::Column { lo, .. } | StrVal::Literal { lo, .. } => lo,
        }
    }
}

impl ColVal {
    pub fn nullness(self) -> Nullness {
        match self {
            ColVal::I32(_, n)
            | ColVal::I64(_, n)
            | ColVal::F64(_, n)
            | ColVal::Bool(_, n)
            | ColVal::Str(_, n) => n,
        }
    }
}

/// A row: one [`ColVal`] per column, positional.
pub type Row = Vec<ColVal>;
