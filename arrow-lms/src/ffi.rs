//! FFI descriptor for an Arrow column and the host-side extraction helpers.
//!
//! [`FfiArray`] is a `#[repr(C)]` view of a single fixed-width Arrow array. It is
//! generic over the staged element marker `M` (e.g. `I32Type`): the values
//! buffer is carried *inline* as a [`FatSlice`] (a `(ptr, len)` fat pointer), so
//! the staged side can read it back as a real `Slice<M>` and reuse every
//! `rust-lms` iterator combinator — see [`crate::array`].
//!
//! The lifetime `'a` ties the descriptor to the borrowed `RecordBatch`/array, so
//! the borrow checker enforces "the batch outlives the descriptor".

use std::marker::PhantomData;

use arrow::array::{Array, Int32Array};
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

/// A `#[repr(C)]` view of one fixed-width Arrow column, passed to JIT'd kernels.
///
/// Generic over the staged element marker `M` (`I32Type`, …). The values buffer
/// is an inline [`FatSlice`] so its field address is, bit-for-bit, a staged
/// `Slice<M>` (see [`crate::array::ArrowArrayOps::values`]). The validity bitmap
/// is still carried as raw metadata (null-aware iteration is a later milestone).
///
/// All pointers borrow into the owning `RecordBatch`/array; the `'a` lifetime
/// makes that borrow explicit, so the owner must outlive the descriptor.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct FfiArray<'a, M: StagedType>
where
    <M as StagedType>::RuntimeValue: 'a,
{
    /// Values buffer as an inline fat pointer (already offset-sliced by Arrow).
    #[staged(FatSliceType<M>)]
    values: FatSlice<M::RuntimeValue>,
    /// Validity bitmap's first byte, or null when every element is valid.
    #[staged(U64Type)]
    validity: *const u8,
    /// Bit offset into the validity bitmap (Arrow slices validity by *bit*).
    #[staged(U64Type)]
    validity_bit_offset: u64,
    /// Number of null entries in the (logical) array.
    #[staged(U64Type)]
    null_count: u64,
    /// Ties the descriptor to the borrowed buffers (enforces batch outlives it).
    #[staged(UnitType)]
    _borrow: PhantomData<&'a [M::RuntimeValue]>,
}

impl<'a, M: StagedType> FfiArray<'a, M>
where
    <M as StagedType>::RuntimeValue: 'a,
{
    /// `true` if the array carries no validity bitmap (every element is valid).
    pub fn all_valid(&self) -> bool {
        self.validity.is_null()
    }

    /// Logical element count (host-side accessor).
    pub fn len(&self) -> usize {
        self.values.len
    }

    /// `true` if the array has no elements.
    pub fn is_empty(&self) -> bool {
        self.values.len == 0
    }
}

/// Extract column `col` of `rb` as an [`FfiArray`], expecting an `Int32` column.
///
/// # Panics
/// If the column is not an `Int32Array`.
pub fn get_primitive_i32(rb: &RecordBatch, col: usize) -> FfiArray<'_, I32Type> {
    let array = rb
        .column(col)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("column is not an Int32Array");
    ffi_from_int32(array)
}

/// Build an [`FfiArray`] descriptor from an Arrow `Int32Array`.
///
/// The returned descriptor borrows into `array`'s buffers; the `'_` lifetime ties
/// it to `array`, so it cannot outlive the data it points at.
pub fn ffi_from_int32(array: &Int32Array) -> FfiArray<'_, I32Type> {
    // `values()` is already offset+length-correct for sliced arrays.
    let values = unsafe { FatSlice::from_raw_parts(array.values().as_ptr(), array.len()) };

    let (validity, validity_bit_offset, null_count) = match array.nulls() {
        Some(nulls) => (
            nulls.inner().values().as_ptr(),
            nulls.inner().offset() as u64,
            nulls.null_count() as u64,
        ),
        None => (std::ptr::null::<u8>(), 0, 0),
    };

    FfiArray {
        values,
        validity,
        validity_bit_offset,
        null_count,
        _borrow: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_matches_arrow() {
        let array = Int32Array::from(vec![10, 20, 30]);
        let ffi = ffi_from_int32(&array);

        assert_eq!(ffi.len(), 3);
        assert_eq!(ffi.null_count, 0);
        assert!(ffi.all_valid());
    }

    #[test]
    fn descriptor_tracks_nulls() {
        let array = Int32Array::from(vec![Some(1), None, Some(3)]);
        let ffi = ffi_from_int32(&array);

        assert_eq!(ffi.len(), 3);
        assert_eq!(ffi.null_count, 1);
        assert!(!ffi.all_valid());
    }
}
