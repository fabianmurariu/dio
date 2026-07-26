//! # arrow-lms: staged Apache Arrow interop for `rust-lms`
//!
//! This crate bridges read-only primitive Apache Arrow arrays into staged
//! `rust-lms` kernels. Host code prepares an erased [`FfiArrayBatch`]; staged
//! code recovers typed primitive views with `batch.primitive::<T>(idx)`.
//!
//! The current scope is intentionally narrow: primitive arrays only, read-only
//! access only. Validity is represented as a first-class staged view, so callers
//! can either iterate non-null values directly or zip physical values with row
//! validity.

// `#[derive(StagedType)]` generates field-token modules with lowercase field
// names, which trips the case lints.
#![allow(non_camel_case_types, non_snake_case)]

mod array;
pub mod ffi;

pub use array::{
    FfiArrayBatchOps, FfiArrayOps, NonNullValues, PrimitiveArrayView, ValidityIsValid,
    ValidityIter, ValidityLen, ValidityNullCount, ValidityView,
};
pub use ffi::{
    ffi_from_primitive, prepare_array_refs, prepare_arrays, prepare_dyn_arrays,
    prepare_record_batch, FfiArray, FfiArrayBatch, FfiBuffer, FfiError, FfiValidity,
    PreparedFfiBatch,
};
