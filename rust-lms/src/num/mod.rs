//! Numerical operations for staged computations.
//!
//! This module provides:
//! - Numeric traits: [`Ordered`] (comparisons), [`Num`], [`Int`] (the integers of
//!   every width: shifts, bit counts), [`IntNum`] (adds rem), [`FloatNum`] (marker).
//! - Operation structs: `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Lt`, `Gt`, `Eq`,
//!   `Select`.
//! - Helper functions for ergonomic expression building.
//! - `std::ops::{Add, Sub, Mul, Div, Rem}` impls so `var + 5`, `x % 2`, etc.
//!   work directly on staged carriers.

mod ops;
mod traits;

pub use ops::{
    Add, BitAnd, BitOp, BitOpKind, BitOr, BitXor, Bitcast, Div, Eq, FloatPromote, Ge, Gt, IntCast,
    IntToFloat, Le, Lt, Mul, Ne, PackI128, Rem, Select, Shl, Shr, Sub,
};
pub use traits::{Bits, FloatNum, Int, IntNum, Num, Ordered};

pub use ops::{
    add, bitand, bitcast, bitor, bitxor, count_ones, div, eq, float_promote, ge, gt, i128_hi,
    i128_lo, int_cast, int_to_float, le, leading_zeros, lt, max, min, mul, ne, pack_i128, rem,
    select, shl, shr, sub, swap_bytes, trailing_zeros,
};
