//! FFI descriptors for primitive Arrow arrays — read *and* write.
//!
//! The layout is deliberately small and **lifetime-free**: an erased `(ptr,
//! len)` buffer, a validity bitmap over one, and the array = values + validity.
//! Staged code reinterprets these as ordinary `rust-lms` slices. A *batch* is
//! just a slice of [`FfiArray`] — there is no wrapper type: read code takes
//! `SRef<Slice<FfiArray>>` (`&[FfiArray]`), write code `SRefMut<Slice<FfiArray>>`
//! (`&mut [FfiArray]`). Mutability is the reference flavor, not a second type.
//!
//! Descriptors carry no lifetime (raw pointers): a `&mut` is invariant in its
//! pointee's lifetimes, so a lifetimed descriptor could not be handed to a
//! kernel whose ABI type is `'static`. Host safety lives at the *slice* border —
//! `&[FfiArray]` / `&mut [FfiArray]` cannot outlive the `Prepared*` owner — and
//! the raw pointers' "arrow data must outlive the call" is the caller contract.

use std::fmt;
use std::marker::PhantomData;

use arrow::array::types::ArrowPrimitiveType;
use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    PrimitiveArray, StringViewArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

/// Erased `(ptr, len)` buffer. `ptr` is `*mut` to serve both flavors; the read
/// path only ever reaches it through an `SRef`, which emits no writes.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiBuffer {
    #[staged(u64)]
    pub ptr: *mut u8,
    #[staged(u64)]
    pub len: usize,
}

const _: () = {
    assert!(std::mem::size_of::<*mut u8>() == 8);
    assert!(std::mem::size_of::<usize>() == 8);
    assert!(std::mem::offset_of!(FfiBuffer, ptr) == 0);
    assert!(std::mem::offset_of!(FfiBuffer, len) == 8);
};

// SAFETY: FfiBuffer is repr(C), and the assertions above keep its pointer and
// u64-sized element count at the offsets consumed by rust-lms slice codegen.
// Individual buffer validity remains the unsafe conversion caller's contract.
unsafe impl<T: StagedType> SliceRepr<T> for FfiBuffer {}

// SAFETY: FfiBuffer stores a mutable pointer at offset 0. Individual instances
// must still be exclusively writable before they are converted to mutable
// slices.
unsafe impl<T: StagedType> MutSliceRepr<T> for FfiBuffer {}

impl FfiBuffer {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null_mut(),
            len: 0,
        }
    }

    /// # Safety
    /// `ptr` must be valid for the interpretation `len` names, for as long as the
    /// owning descriptor is used.
    pub const unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self { ptr, len }
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr() as *mut u8,
            len: bytes.len(),
        }
    }

    pub fn from_typed_slice<T>(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr() as *mut u8,
            len: slice.len(),
        }
    }
}

/// Arrow validity bitmap descriptor (`1 = valid`, `0 = null`; `null_count == 0`
/// means the bitmap can be ignored). Shared by read (`is_valid`) and write
/// (`set_null`) — see `array.rs` for the staged bit ops.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiValidity {
    #[staged(FfiBuffer)]
    pub bytes: FfiBuffer,
    #[staged(u64)]
    pub bit_offset: u64,
    #[staged(u64)]
    pub bit_len: u64,
    #[staged(u64)]
    pub null_count: u64,
}

impl FfiValidity {
    pub const fn all_valid(len: usize) -> Self {
        Self {
            bytes: FfiBuffer::empty(),
            bit_offset: 0,
            bit_len: len as u64,
            null_count: 0,
        }
    }

    pub fn from_nulls(nulls: Option<&NullBuffer>, len: usize) -> Self {
        match nulls {
            Some(nulls) if nulls.null_count() != 0 => Self {
                bytes: FfiBuffer::from_bytes(nulls.inner().values()),
                bit_offset: nulls.offset() as u64,
                bit_len: nulls.len() as u64,
                null_count: nulls.null_count() as u64,
            },
            _ => Self::all_valid(len),
        }
    }

    pub fn all_valid_host(&self) -> bool {
        self.null_count == 0
    }
}

/// Erased Arrow array descriptor: values + validity, plus an opaque pointer back
/// to the originating arrow array.
///
/// `array` is a type-erased pointer to the concrete arrow array (e.g. a
/// `*const StringViewArray`); it is `null` for fixed-width columns and is passed
/// to extern runtime functions that need the array's own API (e.g. `&str`
/// access, `substring`). It borrows the source, which must outlive the call.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiArray {
    #[staged(FfiBuffer)]
    pub values: FfiBuffer,
    #[staged(FfiValidity)]
    pub validity: FfiValidity,
    #[staged(u64)]
    pub array: *const u8,
}

impl FfiArray {
    pub fn len(&self) -> usize {
        self.values.len
    }

    pub fn is_empty(&self) -> bool {
        self.values.len == 0
    }

    pub fn null_count(&self) -> u64 {
        self.validity.null_count
    }

    pub fn all_valid(&self) -> bool {
        self.validity.all_valid_host()
    }
}

/// Host-owned read descriptors borrowing an Arrow source. The `'data` borrow
/// keeps the source alive; [`arrays`](Self::arrays) hands the kernel `&[FfiArray]`.
#[derive(Debug)]
pub struct PreparedFfiBatch<'data> {
    arrays: Vec<FfiArray>,
    _borrow: PhantomData<&'data [u8]>,
}

impl<'data> PreparedFfiBatch<'data> {
    /// The read batch handed to the kernel: `SRef<Slice<FfiArray>>` at runtime.
    pub fn arrays(&self) -> &[FfiArray] {
        &self.arrays
    }
}

/// Errors produced while preparing Arrow arrays for the staged FFI ABI.
#[derive(Debug)]
pub enum FfiError {
    UnsupportedDataType {
        index: usize,
        data_type: DataType,
    },
    Downcast {
        index: usize,
        data_type: DataType,
    },
    MismatchedLength {
        index: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDataType { index, data_type } => write!(
                f,
                "array {index} has unsupported data type {data_type}; only primitive arrays are supported"
            ),
            Self::Downcast { index, data_type } => {
                write!(f, "array {index} could not be downcast as {data_type}")
            }
            Self::MismatchedLength { index, expected, actual } => {
                write!(f, "array {index} has length {actual}, expected {expected}")
            }
        }
    }
}

impl std::error::Error for FfiError {}

/// Build an erased descriptor from an Arrow primitive array.
pub fn ffi_from_primitive<A: ArrowPrimitiveType>(array: &PrimitiveArray<A>) -> FfiArray {
    FfiArray {
        values: FfiBuffer::from_typed_slice(array.values()),
        validity: FfiValidity::from_nulls(array.nulls(), array.len()),
        array: std::ptr::null(),
    }
}

/// Build an erased descriptor from an Arrow `StringViewArray`.
///
/// `values` holds the **views** buffer (one `u128` per row: `[len:u32][…]`).
/// Staged code reads a view as two `u64` halves; `octet_length` needs only the
/// low 32 bits of the first. The data buffers are not referenced here — that
/// comes with actual byte access (which will `gc` to a single buffer first).
pub fn ffi_from_string_view(array: &StringViewArray) -> FfiArray {
    FfiArray {
        values: FfiBuffer::from_typed_slice(array.views().as_ref()),
        validity: FfiValidity::from_nulls(array.nulls(), array.len()),
        // Opaque pointer to the array itself, for extern `&str`/transform calls.
        array: (array as *const StringViewArray).cast::<u8>(),
    }
}

/// Prepare a `RecordBatch` for a compiled kernel.
pub fn prepare_record_batch(rb: &RecordBatch) -> Result<PreparedFfiBatch<'_>, FfiError> {
    prepare_arrays(rb.columns().iter().map(|array| array.as_ref()))
}

/// Prepare borrowed Arrow [`ArrayRef`] values for a compiled kernel.
pub fn prepare_array_refs(arrays: &[ArrayRef]) -> Result<PreparedFfiBatch<'_>, FfiError> {
    prepare_arrays(arrays.iter().map(|array| array.as_ref()))
}

/// Prepare borrowed dynamic Arrow array references for a compiled kernel.
pub fn prepare_dyn_arrays<'data>(
    arrays: &[&'data dyn Array],
) -> Result<PreparedFfiBatch<'data>, FfiError> {
    prepare_arrays(arrays.iter().copied())
}

/// Prepare borrowed Arrow arrays for a compiled kernel.
pub fn prepare_arrays<'a, I>(arrays: I) -> Result<PreparedFfiBatch<'a>, FfiError>
where
    I: IntoIterator<Item = &'a dyn Array>,
{
    let mut row_count = None;
    let arrays = arrays
        .into_iter()
        .enumerate()
        .map(|(index, array)| {
            match row_count {
                Some(expected) if expected != array.len() => {
                    return Err(FfiError::MismatchedLength {
                        index,
                        expected,
                        actual: array.len(),
                    });
                }
                None => row_count = Some(array.len()),
                _ => {}
            }
            ffi_from_array(index, array)
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(PreparedFfiBatch {
        arrays,
        _borrow: PhantomData,
    })
}

fn ffi_from_array(index: usize, array: &dyn Array) -> Result<FfiArray, FfiError> {
    macro_rules! downcast_primitive {
        ($array_ty:ty) => {{
            let primitive =
                array
                    .as_any()
                    .downcast_ref::<$array_ty>()
                    .ok_or_else(|| FfiError::Downcast {
                        index,
                        data_type: array.data_type().clone(),
                    })?;
            Ok(ffi_from_primitive(primitive))
        }};
    }

    match array.data_type() {
        DataType::Int8 => downcast_primitive!(Int8Array),
        DataType::Int16 => downcast_primitive!(Int16Array),
        DataType::Int32 => downcast_primitive!(Int32Array),
        DataType::Int64 => downcast_primitive!(Int64Array),
        DataType::UInt8 => downcast_primitive!(UInt8Array),
        DataType::UInt16 => downcast_primitive!(UInt16Array),
        DataType::UInt32 => downcast_primitive!(UInt32Array),
        DataType::UInt64 => downcast_primitive!(UInt64Array),
        DataType::Float32 => downcast_primitive!(Float32Array),
        DataType::Float64 => downcast_primitive!(Float64Array),
        DataType::Utf8View => {
            let view = array
                .as_any()
                .downcast_ref::<StringViewArray>()
                .ok_or_else(|| FfiError::Downcast {
                    index,
                    data_type: array.data_type().clone(),
                })?;
            Ok(ffi_from_string_view(view))
        }
        data_type => Err(FfiError::UnsupportedDataType {
            index,
            data_type: data_type.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_slice_representations<T: StagedType>()
    where
        FfiBuffer: SliceRepr<T> + MutSliceRepr<T>,
    {
    }

    #[test]
    fn buffer_is_an_explicit_slice_representation() {
        assert_slice_representations::<u8>();
        assert_slice_representations::<i64>();
    }

    #[test]
    fn descriptor_matches_arrow() {
        let array = Int32Array::from(vec![10, 20, 30]);
        let ffi = ffi_from_primitive(&array);
        assert_eq!(ffi.len(), 3);
        assert_eq!(ffi.null_count(), 0);
        assert!(ffi.all_valid());
    }

    #[test]
    fn descriptor_tracks_nulls() {
        let nulls = NullBuffer::from(vec![true, false, true]);
        let array = Int32Array::new(vec![1, 99, 3].into(), Some(nulls));
        let ffi = ffi_from_primitive(&array);
        assert_eq!(ffi.len(), 3);
        assert_eq!(ffi.null_count(), 1);
        assert!(!ffi.all_valid());
    }
}
