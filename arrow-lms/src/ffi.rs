//! FFI descriptors for read-only primitive Arrow arrays.
//!
//! Host code prepares a [`PreparedFfiBatch`] from a [`RecordBatch`] or borrowed
//! Arrow arrays, then passes the borrowed [`FfiArrayBatch`] view to a JIT'd
//! kernel. The staged side type-specializes columns by position; schema binding
//! is expected to guarantee that `batch.primitive::<T>(idx)` uses the right `T`.

use std::fmt;
use std::marker::PhantomData;
use std::slice;

use arrow::array::types::ArrowPrimitiveType;
use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    PrimitiveArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

/// FFI-safe borrowed slice layout.
///
/// This is the public replacement name for the old "fat slice" concept: a
/// pointer plus a length in elements of `T`.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiSlice<'a, T: 'a> {
    #[staged(u64)]
    pub ptr: *const T,
    #[staged(u64)]
    pub len: usize,
    #[staged(())]
    _borrow: PhantomData<&'a [T]>,
}

impl<'a, T> FfiSlice<'a, T> {
    /// Create a borrowed slice descriptor from raw parts.
    ///
    /// # Safety
    /// `ptr` must be valid for `len` elements for the lifetime represented by
    /// this descriptor.
    pub const unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _borrow: PhantomData,
        }
    }

    /// Create a descriptor from a Rust slice.
    pub fn from_slice(slice: &'a [T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _borrow: PhantomData,
        }
    }

    /// Convert the descriptor back into a Rust slice.
    ///
    /// # Safety
    /// The pointed-to memory must still be valid and obey Rust aliasing rules.
    pub unsafe fn as_slice(&self) -> &'a [T] {
        slice::from_raw_parts(self.ptr, self.len)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, T> From<&'a [T]> for FfiSlice<'a, T> {
    fn from(value: &'a [T]) -> Self {
        Self::from_slice(value)
    }
}

/// Erased primitive values buffer.
///
/// `ptr` points at the first logical primitive value. `len` is the number of
/// primitive elements, not the number of bytes. The staged typed view supplies
/// the element width when loading values.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiBuffer<'a> {
    #[staged(u64)]
    pub ptr: *const u8,
    #[staged(u64)]
    pub len: usize,
    #[staged(())]
    _borrow: PhantomData<&'a [u8]>,
}

impl<'a> FfiBuffer<'a> {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
            _borrow: PhantomData,
        }
    }

    /// Create an erased primitive values descriptor from raw parts.
    ///
    /// # Safety
    /// `ptr` must point at at least `len` primitive elements of the type that
    /// staged code will later use to read this buffer.
    pub const unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Self {
        Self {
            ptr,
            len,
            _borrow: PhantomData,
        }
    }
}

/// FFI-safe Arrow validity bitmap descriptor.
///
/// Arrow validity uses `1 = valid` and `0 = null`. A null `ptr` means the
/// entire logical array is valid.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiValidity<'a> {
    #[staged(u64)]
    pub ptr: *const u8,
    #[staged(u64)]
    pub bit_offset: u64,
    #[staged(u64)]
    pub len: u64,
    #[staged(u64)]
    pub null_count: u64,
    #[staged(())]
    _borrow: PhantomData<&'a [u8]>,
}

impl<'a> FfiValidity<'a> {
    pub const fn all_valid(len: usize) -> Self {
        Self {
            ptr: std::ptr::null(),
            bit_offset: 0,
            len: len as u64,
            null_count: 0,
            _borrow: PhantomData,
        }
    }

    pub fn from_nulls(nulls: Option<&'a NullBuffer>, len: usize) -> Self {
        match nulls {
            Some(nulls) if nulls.null_count() != 0 => Self {
                ptr: nulls.inner().values().as_ptr(),
                bit_offset: nulls.offset() as u64,
                len: nulls.len() as u64,
                null_count: nulls.null_count() as u64,
                _borrow: PhantomData,
            },
            _ => Self::all_valid(len),
        }
    }

    pub fn all_valid_host(&self) -> bool {
        self.ptr.is_null()
    }
}

/// Erased read-only primitive Arrow array descriptor.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiArray<'a> {
    #[staged(FfiBuffer<'a>)]
    pub values: FfiBuffer<'a>,
    #[staged(FfiValidity<'a>)]
    pub validity: FfiValidity<'a>,
}

impl<'a> FfiArray<'a> {
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

/// Borrowed vector of erased primitive array descriptors.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiArrayBatch<'arrays, 'data: 'arrays> {
    #[staged(FfiSlice<'arrays, FfiArray<'data>>)]
    pub arrays: FfiSlice<'arrays, FfiArray<'data>>,
}

impl<'arrays, 'data> FfiArrayBatch<'arrays, 'data> {
    pub fn len(&self) -> usize {
        self.arrays.len
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }
}

/// Host-owned prepared descriptors.
///
/// The descriptors borrow Arrow buffers for `'data`; [`Self::as_ffi`] creates
/// the short-lived batch view that borrows this vector of descriptors.
#[derive(Debug)]
pub struct PreparedFfiBatch<'data> {
    arrays: Vec<FfiArray<'data>>,
}

impl<'data> PreparedFfiBatch<'data> {
    pub fn as_ffi(&self) -> FfiArrayBatch<'_, 'data> {
        FfiArrayBatch {
            arrays: FfiSlice::from_slice(&self.arrays),
        }
    }

    pub fn arrays(&self) -> &[FfiArray<'data>] {
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
            Self::UnsupportedDataType { index, data_type } => {
                write!(
                    f,
                    "array {index} has unsupported data type {data_type}; only primitive arrays are supported"
                )
            }
            Self::Downcast { index, data_type } => {
                write!(f, "array {index} could not be downcast as {data_type}")
            }
            Self::MismatchedLength {
                index,
                expected,
                actual,
            } => write!(f, "array {index} has length {actual}, expected {expected}"),
        }
    }
}

impl std::error::Error for FfiError {}

/// Build an erased FFI descriptor from an Arrow primitive array.
pub fn ffi_from_primitive<A>(array: &PrimitiveArray<A>) -> FfiArray<'_>
where
    A: ArrowPrimitiveType,
{
    FfiArray {
        values: unsafe {
            FfiBuffer::from_raw_parts(array.values().as_ptr().cast::<u8>(), array.len())
        },
        validity: FfiValidity::from_nulls(array.nulls(), array.len()),
    }
}

/// Build an erased FFI descriptor from an Arrow `Int32Array`.
pub fn ffi_from_int32(array: &Int32Array) -> FfiArray<'_> {
    ffi_from_primitive(array)
}

/// Prepare a `RecordBatch` for a compiled kernel.
pub fn prepare_record_batch(rb: &RecordBatch) -> Result<PreparedFfiBatch<'_>, FfiError> {
    prepare_arrays(rb.columns().iter().map(|array| array.as_ref()))
}

/// Prepare borrowed Arrow [`ArrayRef`] values for a compiled kernel.
pub fn prepare_array_refs<'a>(arrays: &'a [ArrayRef]) -> Result<PreparedFfiBatch<'a>, FfiError> {
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

    Ok(PreparedFfiBatch { arrays })
}

fn ffi_from_array(index: usize, array: &dyn Array) -> Result<FfiArray<'_>, FfiError> {
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
        data_type => Err(FfiError::UnsupportedDataType {
            index,
            data_type: data_type.clone(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::buffer::NullBuffer;

    #[test]
    fn descriptor_matches_arrow() {
        let array = Int32Array::from(vec![10, 20, 30]);
        let ffi = ffi_from_int32(&array);

        assert_eq!(ffi.len(), 3);
        assert_eq!(ffi.null_count(), 0);
        assert!(ffi.all_valid());
    }

    #[test]
    fn descriptor_tracks_nulls() {
        let nulls = NullBuffer::from(vec![true, false, true]);
        let array = Int32Array::new(vec![1, 99, 3].into(), Some(nulls));
        let ffi = ffi_from_int32(&array);

        assert_eq!(ffi.len(), 3);
        assert_eq!(ffi.null_count(), 1);
        assert!(!ffi.all_valid());
    }
}
