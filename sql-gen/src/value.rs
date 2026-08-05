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

/// A staged `Utf8View` string, in one of two containers — but every container
/// resolves to a generic `&[u8]` byte reference (`codegen::resolve`), so the
/// runtime is a flat library over bytes.
///
/// - `Column`: a `Utf8View` column row. Carries the 16-byte view as two `u64`
///   halves (`lo` = `[len:u32][…]`, `hi` = the rest) for register-op
///   `octet_length` and the inline-literal `=` fast path, plus the originating
///   array pointer + row so its bytes can be resolved (`arr.value(row)`).
/// - `Bytes`: already-materialized bytes at `(ptr, len)` — a build-time-interned
///   literal or a kernel-produced string. `view` holds the inline view halves iff
///   the bytes are a known `≤12`-byte value (enables the staged `=` fast path);
///   `None` for long literals and produced strings.
#[derive(Clone, Copy)]
pub enum StrVal {
    Column {
        lo: Var<u64>,
        hi: Var<u64>,
        array: Var<u64>,
        row: Var<u64>,
    },
    Bytes {
        ptr: Var<u64>,
        len: Var<u64>,
        view: Option<(u64, u64)>,
    },
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
