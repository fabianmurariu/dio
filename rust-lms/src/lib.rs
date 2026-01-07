//! # rust-lms: Type-Safe Staged Computation in Rust
//!
//! A Rust implementation of multi-stage programming inspired by Scala LMS (Lightweight Modular Staging).
//!
//! ## Overview
//!
//! This library provides a type-safe framework for building staged computations that compile
//! to efficient machine code via Cranelift. Key features:
//!
//! - **Compile-time type safety**: Invalid operations are caught at compile time
//! - **Zero-cost abstractions**: `Var<T>` and `Const<T>` are Copy when possible
//! - **Heterogeneous operations**: Operations can change output types (e.g., comparison → bool)
//! - **Full composability**: Any `Staged` value works anywhere a `Staged` value is expected
//! - **Dynamic dispatch support**: Boxing via `.boxed()` when needed
//!
//! ## Quick Example
//!
//! ```ignore
//! use rust_lms::prelude::*;
//! use cranelift_frontend::Variable;
//!
//! // Create variables and constants
//! let x = Var::<I64Type>::new(Variable::from_u32(0));
//! let five = Const::<I64Type>::new(5);
//! let two = Const::<I64Type>::new(2);
//!
//! // Build expressions: (x + 5) * 2
//! let expr = mul(add(x, five), two);
//!
//! // x is Copy, so we can reuse it!
//! let expr2 = add(x, x);
//!
//! // Comparisons change type to Bool
//! let comparison = lt(x, Const::new(100));
//!
//! // This won't compile - type mismatch caught at compile time!
//! // let bad = add(x, comparison);  // ERROR: can't add I64 and Bool
//! ```
//!
//! ## Architecture
//!
//! ### Core Traits
//!
//! - [`Staged`](staged::Staged): Anything that can generate runtime code
//! - [`StagedType`](types::StagedType): Types that can participate in staged computation
//!
//! ### Value Types
//!
//! - [`Var<T>`](staged::Var): Typed variable references (Copy-able)
//! - [`Const<T>`](staged::Const): Typed constants (Copy-able)
//!
//! ### Type Markers
//!
//! - [`I64Type`](types::I64Type), [`U64Type`](types::U64Type): Integer types
//! - [`F64Type`](types::F64Type): Floating-point type
//! - [`BoolType`](types::BoolType): Boolean type
//!
//! ### Operations
//!
//! - Arithmetic: [`Add`](num::Add), [`Sub`](num::Sub), [`Mul`](num::Mul), [`Div`](num::Div)
//! - Comparison: [`Lt`](num::Lt), [`Eq`](num::Eq)
//!
//! ## Design Principles
//!
//! ### 1. Separation of Values and Operations
//!
//! Unlike traditional expression trees where operations are part of the value enum,
//! this design separates:
//! - **Values**: `Var<T>`, `Const<T>` - only represent "pure" values
//! - **Operations**: `Add<L,R>`, `Lt<L,R>` - separate structs that also implement `Staged`
//!
//! ### 2. Type-Level Constraints
//!
//! Operations use trait bounds to ensure type safety:
//!
//! ```ignore
//! impl<L, R, T> Staged for Add<L, R>
//! where
//!     L: Staged<Out = T>,  // Left must produce type T
//!     R: Staged<Out = T>,  // Right must produce type T
//!     T: StagedType + SupportsAdd,  // T must support addition
//! {
//!     type Out = T;  // Result is also type T
//! }
//! ```
//!
//! ### 3. Heterogeneous Operations
//!
//! Some operations change types:
//!
//! ```ignore
//! impl<L, R, T> Staged for Lt<L, R>
//! where
//!     L: Staged<Out = T>,
//!     R: Staged<Out = T>,
//!     T: StagedType + SupportsComparison,
//! {
//!     type Out = BoolType;  // Always returns Bool, not T!
//! }
//! ```

pub mod num;
pub mod staged;
pub mod types;

pub mod func;

/// Commonly used types and traits
pub mod prelude {
    pub use crate::func::{fun1, Fun1, FunType1};
    pub use crate::num::{add, div, eq, lt, mul, sub};
    pub use crate::staged::{BoxableStaged, CompilationContext, Const, Staged, Var};
    pub use crate::types::{BoolType, ConstantType, F64Type, I64Type, StagedType, U64Type};
}