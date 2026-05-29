//! Numerical operations for staged computations.
//!
//! This module provides:
//! - Numeric traits: [`Num`], [`IntNum`] (adds rem), [`FloatNum`] (marker).
//! - Operation structs: `Add`, `Sub`, `Mul`, `Div`, `Rem`, `Lt`, `Gt`, `Eq`,
//!   `Select`.
//! - Helper functions for ergonomic expression building.
//! - `std::ops::{Add, Sub, Mul, Div, Rem}` impls so `var + 5`, `x % 2`, etc.
//!   work directly on staged carriers.

mod ops;
mod traits;

pub use ops::{Add, Div, Eq, Gt, Lt, Mul, Rem, Select, Sub};
pub use traits::{FloatNum, IntNum, Num};

pub use ops::{add, div, eq, gt, lt, max, min, mul, rem, select, sub};
