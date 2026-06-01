//! FFI descriptor for an Arrow column and the host-side extraction helpers.
//!
//! [`FfiArray`] is a type-erased, `#[repr(C)]` view of a single fixed-width Arrow
//! array: the values buffer (already offset-sliced by Arrow), plus the validity
//! bitmap and its metadata. The element type lives only in the *staged* wrapper
//! ([`crate::StagedArrowArrayI32`]), so one descriptor struct serves every
//! primitive type.

use arrow::array::{Array, Int32Array};
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

/// A `#[repr(C)]` view of one fixed-width Arrow column, passed to JIT'd kernels.
///
/// All pointers borrow into the owning `RecordBatch`/array; that owner **must
/// outlive** any call into a compiled function that receives this descriptor.
///
/// Fields are exposed to staged code as raw `u64` addresses (`#[staged(U64Type)]`)
/// — the staged side does its own typed pointer arithmetic, so the descriptor
/// stays type-erased.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct FfiArray {
    /// Pointer to the first value. Arrow's `values()` is already offset-sliced,
    /// so element `i` lives at `values + i * size_of::<elem>()`.
    #[staged(U64Type)]
    pub values: *const u8,
    /// Logical element count.
    #[staged(U64Type)]
    pub len: u64,
    /// Pointer to the validity bitmap's first byte, or null when there are no
    /// nulls (all elements valid).
    #[staged(U64Type)]
    pub validity: *const u8,
    /// Bit offset into the validity bitmap. Arrow slices validity by *bit*, not
    /// byte, so element `i`'s validity bit is at `validity_bit_offset + i`.
    #[staged(U64Type)]
    pub validity_bit_offset: u64,
    /// Number of null entries in the (logical) array.
    #[staged(U64Type)]
    pub null_count: u64,
}

impl FfiArray {
    /// `true` if the array carries no validity bitmap (every element is valid).
    pub fn all_valid(&self) -> bool {
        self.validity.is_null()
    }
}

/// Extract column `col` of `rb` as an [`FfiArray`], expecting an `Int32` column.
///
/// # Panics
/// If the column is not an `Int32Array`.
pub fn get_primitive_i32(rb: &RecordBatch, col: usize) -> FfiArray {
    let array = rb
        .column(col)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("column is not an Int32Array");
    ffi_from_int32(array)
}

/// Build an [`FfiArray`] descriptor from an Arrow `Int32Array`.
///
/// The returned descriptor borrows into `array`'s buffers; keep `array` (and the
/// `RecordBatch` that owns it) alive for as long as the descriptor is in use.
pub fn ffi_from_int32(array: &Int32Array) -> FfiArray {
    // `values()` is already offset+length-correct for sliced arrays.
    let values = array.values().as_ptr() as *const u8;

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
        len: array.len() as u64,
        validity,
        validity_bit_offset,
        null_count,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptor_matches_arrow() {
        let array = Int32Array::from(vec![10, 20, 30]);
        let ffi = ffi_from_int32(&array);

        assert_eq!(ffi.len, 3);
        assert_eq!(ffi.values, array.values().as_ptr() as *const u8);
        assert_eq!(ffi.null_count, 0);
        assert!(ffi.all_valid());
    }

    #[test]
    fn descriptor_tracks_nulls() {
        let array = Int32Array::from(vec![Some(1), None, Some(3)]);
        let ffi = ffi_from_int32(&array);

        assert_eq!(ffi.len, 3);
        assert_eq!(ffi.null_count, 1);
        assert!(!ffi.all_valid());
    }
}
