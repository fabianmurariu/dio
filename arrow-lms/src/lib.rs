//! # arrow-lms: staged Apache Arrow interop for `rust-lms`
//!
//! Reads Apache Arrow arrays in staged `rust-lms` kernels, without copying them. Host
//! code describes a batch's columns with [`prepare_record_batch`] (or keeps the batch,
//! [`FfiRecordBatch`]); a kernel reads typed columns through the [`FfiBatch`] it is
//! given (`batch.arrays().primitive::<T>(i)`, `batch.strings(i)`). Read-only: kernels
//! write their output elsewhere (`sql-gen`'s `OutCols`, db4's blocks).
//!
//! Scope: primitive and `Boolean` arrays, and byte-view arrays (`Utf8View`,
//! `BinaryView`) read as `ByteView<K>` rows ([`ByteViewDescriptors`]). Validity is a
//! bitmap ([`FfiBitmap`]) and a null count ([`FfiValidity`]).

// `#[derive(StagedType)]` generates field-token modules with lowercase field
// names, which trips the case lints.
#![allow(non_camel_case_types, non_snake_case)]

mod array;
mod batch;
mod byte_view;
pub mod ffi;
mod validity;

pub use array::{ArrayBatchOps, ArraySource, BooleanArrayView, FfiArrayOps, PrimitiveArrayView};
pub use batch::{FfiBatch, FfiBatchOps, FfiBatchType};
pub use byte_view::{
    ArrowByteKind, ByteViewArrayView, ByteViewDescriptors, FfiByteViewArray, FfiByteViewArrayOps,
};
pub use ffi::{
    FfiArray, FfiBuffer, FfiError, FfiRecordBatch, PreparedFfiBatch, prepare_array_refs,
    prepare_arrays, prepare_dyn_arrays, prepare_record_batch,
};
pub use validity::{BitmapWord, FfiBitmap, FfiValidity, ValidityView};
