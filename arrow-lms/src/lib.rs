//! # arrow-lms: staged Apache Arrow interop for `rust-lms`
//!
//! Bridges primitive Apache Arrow arrays into staged `rust-lms` kernels. A batch
//! is a `Slice<FfiArray>`: host code prepares one with [`prepare_record_batch`]
//! (read), and staged code recovers typed columns with `batch.primitive::<T>(idx)`.
//! Output materialization lives in the consumer (`sql-gen`'s `SVec`-backed
//! `OutCols`). Read descriptors and owner-backed writable validity descriptors
//! are separate types.
//!
//! Scope: primitive arrays only. Validity is a first-class staged view.

// `#[derive(StagedType)]` generates field-token modules with lowercase field
// names, which trips the case lints.
#![allow(non_camel_case_types, non_snake_case)]

mod array;
pub mod ffi;
pub mod ffi_mut;

pub use array::{
    ArrayBatchOps, ArraySource, FfiArrayOps, PrimitiveArrayView, ValidityIsValid, ValidityLen,
    ValidityNullCount, ValiditySource, ValidityView,
};
pub use ffi::{
    prepare_array_refs, prepare_arrays, prepare_dyn_arrays, prepare_record_batch,
    prepare_validity_mut, FfiArray, FfiBuffer, FfiError, FfiValidity, FfiValidityMut,
    PreparedFfiBatch, PreparedFfiValidityMut,
};
// `ffi_mut` now holds only the standalone `ValidityView` write ops (re-exported
// from `array`); the host output path moved to the consumer.
