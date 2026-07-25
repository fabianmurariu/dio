//! FFI descriptors for read-only primitive Arrow arrays.
//!
//! The layout is deliberately small: an erased `(ptr, len)` buffer, a validity
//! bitmap built from that buffer, and an erased array batch. Staged code
//! reinterprets these buffers as ordinary `rust-lms` slices.

use std::fmt;
use std::marker::PhantomData;

use arrow::array::types::ArrowPrimitiveType;
use arrow::array::{
    Array, ArrayRef, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array, Int8Array,
    PrimitiveArray, UInt16Array, UInt32Array, UInt64Array, UInt8Array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

/// Erased FFI slice/buffer layout.
///
/// The meaning of `len` belongs to the owner: primitive values store element
/// count for the eventual typed interpretation, while bitmap buffers store byte
/// count.
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

    /// Create an erased descriptor from raw parts.
    ///
    /// # Safety
    /// `ptr` must be valid for the interpretation attached to `len` by the
    /// owning descriptor.
    pub const unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Self {
        Self {
            ptr,
            len,
            _borrow: PhantomData,
        }
    }

    pub fn from_bytes(bytes: &'a [u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
            _borrow: PhantomData,
        }
    }

    pub fn from_typed_slice<T>(slice: &'a [T]) -> Self {
        Self {
            ptr: slice.as_ptr().cast::<u8>(),
            len: slice.len(),
            _borrow: PhantomData,
        }
    }
}

/// Arrow validity descriptor.
///
/// Arrow validity uses `1 = valid` and `0 = null`. `null_count == 0` means the
/// bitmap can be ignored.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiValidity<'a> {
    #[staged(FfiBuffer<'a>)]
    pub bytes: FfiBuffer<'a>,
    #[staged(u64)]
    pub bit_offset: u64,
    #[staged(u64)]
    pub bit_len: u64,
    #[staged(u64)]
    pub null_count: u64,
}

impl<'a> FfiValidity<'a> {
    pub const fn all_valid(len: usize) -> Self {
        Self {
            bytes: FfiBuffer::empty(),
            bit_offset: 0,
            bit_len: len as u64,
            null_count: 0,
        }
    }

    pub fn from_nulls(nulls: Option<&'a NullBuffer>, len: usize) -> Self {
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

/// Erased batch descriptor: a buffer of [`FfiArray`] descriptors.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiArrayBatch<'arrays, 'data: 'arrays> {
    #[staged(FfiBuffer<'arrays>)]
    pub arrays: FfiBuffer<'arrays>,
    #[staged(())]
    _borrow: PhantomData<&'arrays [FfiArray<'data>]>,
}

impl<'arrays, 'data> FfiArrayBatch<'arrays, 'data> {
    pub fn len(&self) -> usize {
        self.arrays.len
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.len == 0
    }
}

/// Host-owned prepared descriptors.
#[derive(Debug)]
pub struct PreparedFfiBatch<'data> {
    arrays: Vec<FfiArray<'data>>,
}

impl<'data> PreparedFfiBatch<'data> {
    pub fn as_ffi(&self) -> FfiArrayBatch<'_, 'data> {
        FfiArrayBatch {
            arrays: FfiBuffer::from_typed_slice(&self.arrays),
            _borrow: PhantomData,
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
        values: FfiBuffer::from_typed_slice(array.values()),
        validity: FfiValidity::from_nulls(array.nulls(), array.len()),
    }
}

/// Build an erased FFI descriptor from an Arrow `Int32Array`.
pub fn ffi_from_int32(array: &Int32Array) -> FfiArray<'_> {
    ffi_from_primitive(array)
}

/// Extract column `col` of `rb` as an erased primitive descriptor, expecting an
/// `Int32` column.
pub fn get_primitive_i32(rb: &RecordBatch, col: usize) -> FfiArray<'_> {
    let array = rb
        .column(col)
        .as_any()
        .downcast_ref::<Int32Array>()
        .expect("column is not an Int32Array");
    ffi_from_int32(array)
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
