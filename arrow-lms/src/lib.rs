//! # arrow-lms: staged Apache Arrow interop for `rust-lms`
//!
//! Reads Apache Arrow arrays in staged `rust-lms` kernels, without copying them. Host
//! code describes a batch's columns, checked against the schema the kernel reads them
//! as, with [`FfiRecordBatch`] (or borrows them unchecked, [`prepare_record_batch`]). A
//! kernel holds the [`FfiBatch`] it is given as a [`SchemaBatch`] of that schema, and
//! reads each column as its own array type (`batch.column::<FfiPrimArray<i64>>(i)`),
//! checked while staging. Read-only: kernels write their output elsewhere (`sql-gen`'s
//! `OutCols`, db4's blocks).
//!
//! The arrays, each built only from an Arrow array of its type:
//! - [`FfiPrimArray<T>`]: the primitive types ([`ArrowNative`]);
//! - [`FfiBoolArray`]: `Boolean`, a bit per row;
//! - [`FfiByteViewArray<K>`]: `Utf8View` and `BinaryView`, read as `ByteView<K>` rows
//!   against a buffer table ([`ByteViewDescriptors`]).
//!
//! Generic code is generic in [`ArrowArray`] (`len`, `validity`, `value_unchecked`, via
//! [`ArrowArrayOps`]). A batch holds [`FfiArray`]s, an enum of the arrays. Validity is a
//! bitmap ([`FfiBitmap`]) and a null count ([`FfiValidity`]). Kernels that work on a
//! bitmap of their own, such as db4's sets of positions, read it as a [`Bitmap`]: a bit,
//! a word, or the set bits of a union in a range.

// `#[derive(StagedType)]` generates field-token modules with lowercase field
// names, which trips the case lints.
#![allow(non_camel_case_types, non_snake_case)]

mod array;
mod batch;
mod byte_view;
pub mod ffi;
mod validity;

pub use array::{
    ArrayBatchOps, ArrowArray, ArrowArrayOps, ArrowNative, FfiArray, FfiArrayOps, FfiBoolArray,
    FfiPrimArray, FfiPrimArrayOps,
};
pub use batch::{FfiBatch, FfiBatchOps, FfiBatchType, SchemaBatch};
pub use byte_view::{ArrowByteKind, ByteViewDescriptors, FfiByteViewArray};
pub use ffi::{
    FfiError, FfiRecordBatch, PreparedFfiBatch, prepare_arrays, prepare_dyn_arrays,
    prepare_record_batch,
};
pub use validity::{Bitmap, BitmapWord, FfiBitmap, FfiValidity, ValidityView};
