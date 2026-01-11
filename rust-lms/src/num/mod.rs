//! Numerical operations for staged computations.
//!
//! This module provides:
//! - Capability traits: `SupportsAdd`, `SupportsSub`, `SupportsComparison`, etc.
//! - Operation structs: `Add`, `Sub`, `Mul`, `Div`, `Lt`, `Eq`, etc.
//! - Helper functions for ergonomic expression building

mod ops;
mod traits;

pub use ops::{Add, Div, Eq, Lt, Mul, Sub};
pub use traits::{SupportsAdd, SupportsComparison, SupportsDiv, SupportsMul, SupportsSub};

// Re-export helper functions
pub use ops::{add, div, eq, gt, lt, mul, sub};
