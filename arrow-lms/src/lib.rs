//! # arrow-lms: staged Apache Arrow interop for `rust-lms`
//!
//! This crate bridges Apache Arrow columnar data into `rust-lms` staged
//! computations, so a JIT-compiled kernel can read directly out of a
//! [`arrow::record_batch::RecordBatch`]. It is the data-access layer the future
//! SQL executor will codegen against.
//!
//! ## The three layers
//!
//! 1. **FFI descriptor** ([`ffi::FfiArray`]) — a `#[repr(C)]` struct that
//!    flattens one Arrow column into raw pointers + lengths + null metadata.
//!    Host code builds these from a `RecordBatch` *before* calling the JIT'd
//!    function (the pointers borrow into the batch, which must outlive the call).
//! 2. **Host extraction** ([`ffi::get_primitive_i32`]) — plain Rust that
//!    downcasts a column and reads its buffers into an [`ffi::FfiArray`].
//! 3. **Staged view** ([`StagedArrowArrayI32`]) — the stage-0 wrapper. Built from
//!    a `Var<SRef<FfiArray>>` parameter, it binds the column's buffers into local
//!    variables and exposes [`StagedArrowArrayI32::values`], which is a full
//!    [`rust_lms`] iterator source — so `array.values().filter(..).sum(ctx)` and
//!    friends work out of the box.
//!
//! ## Status
//!
//! Prototype: fixed-width `Int32` arrays, values-only iteration. The null buffer
//! is carried end-to-end ([`StagedNullBuffer`]) so null-aware operations and more
//! element types can be layered on without changing the layout.
//!
//! ## Example
//!
//! ```ignore
//! use arrow_lms::{get_primitive_i32, FfiArray, StagedArrowArrayI32};
//! use rust_lms::prelude::*;
//!
//! let ffi = get_primitive_i32(&record_batch, 0);
//!
//! let mut compiler = Compiler::new();
//! let f = compiler.fun1("sum_i32", |ctx, arr: Var<SRef<FfiArray>>| {
//!     let array = StagedArrowArrayI32::load(ctx, arr);
//!     array.values().sum(ctx)            // -> Var<i32>
//! });
//! let sum = compiler.compile(f).unwrap().as_fn();   // extern "C" fn(&FfiArray) -> i32
//! assert_eq!(sum(&ffi), /* sum of column 0 */ 0);
//! ```

// `#[derive(StagedType)]` generates a `FfiArrayType` module of field tokens named
// after the (lowercase) struct fields, which trips the camel-/snake-case lints.
#![allow(non_camel_case_types, non_snake_case)]

mod array;
pub mod ffi;

pub use array::{ArrowArrayOps, AsSlice};
pub use ffi::{ffi_from_int32, get_primitive_i32, FfiArray};
