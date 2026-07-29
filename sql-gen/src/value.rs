//! The mixed-stage row — a static schema paired with dynamic (stage-1) fields.
//!
//! This is the Rust translation of the paper's `Record(Vector[Rep[T]], Schema)`.
//! Because the query is *interpreted at runtime*, the row's arity and column
//! types are stage-0-dynamic, so the fields cannot be a compile-time tuple; they
//! are a type-erased `Vec<ColVal>`. Each `ColVal` carries its physical type as a
//! (static) enum tag and a `Copy` `Var` handle as the (dynamic) value.
//!
//! Crucially the row lives entirely in stage-0 Rust and never becomes a *staged*
//! aggregate, so it can be moved by value through `'static` staged closures
//! (`while_loop`/`if_then` bodies) — sidestepping the single-register limit of
//! the opaque-iterator item ABI.

use rust_lms::prelude::*;

/// A staged column value: the physical-type tag is static, the `Var` is dynamic.
#[derive(Clone, Copy)]
pub enum ColVal {
    I32(Var<i32>),
    I64(Var<i64>),
    F64(Var<f64>),
}

/// A row: one [`ColVal`] per column, positional (the [`crate::plan::Schema`]
/// supplies the types/order).
pub type Row = Vec<ColVal>;
