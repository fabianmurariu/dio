//! FFI descriptors for primitive Arrow arrays.
//!
//! The wire structs remain lifetime-free because generated code consumes their
//! `repr(C)` layout. Safe host construction is lifetime-bearing: read batches
//! come from [`PreparedFfiBatch`], while writable bitmap descriptors come from
//! [`PreparedFfiValidityMut`]. Their raw fields are private and read-only and
//! writable buffers are distinct staged types.

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

/// Erased read-only `(ptr, len)` buffer used inside prepared Arrow descriptors.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiBuffer {
    #[staged(SPtr<u8>)]
    ptr: *const u8,
    #[staged(u64)]
    len: usize,
}

const _: () = {
    assert!(std::mem::size_of::<*const u8>() == 8);
    assert!(std::mem::size_of::<usize>() == 8);
    assert!(std::mem::offset_of!(FfiBuffer, ptr) == 0);
    assert!(std::mem::offset_of!(FfiBuffer, len) == 8);
};

// SAFETY: FfiBuffer is repr(C), and the assertions above keep its pointer and
// u64-sized element count at the offsets consumed by rust-lms slice codegen.
// Individual buffer validity remains the unsafe conversion caller's contract.
unsafe impl<T: StagedType> SliceRepr<T> for FfiBuffer {}

impl FfiBuffer {
    const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len(),
        }
    }

    fn from_typed_slice<T>(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr().cast::<u8>(),
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
    bytes: FfiBuffer,
    #[staged(u64)]
    bit_offset: u64,
    #[staged(u64)]
    bit_len: u64,
    #[staged(u64)]
    null_count: u64,
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

    fn from_nulls(nulls: Option<&NullBuffer>, len: usize) -> Self {
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
    values: FfiBuffer,
    #[staged(FfiValidity)]
    validity: FfiValidity,
    #[staged(SPtr<u8>)]
    array: *const u8,
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

/// Mutable `(ptr, len)` buffer used only by owner-backed writable descriptors.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiBufferMut {
    #[staged(SMutPtr<u8>)]
    ptr: *mut u8,
    #[staged(u64)]
    len: usize,
}

// SAFETY: this private repr(C) descriptor stores a writable byte pointer and a
// u64-sized byte count at the offsets consumed by slice lowering.
unsafe impl SliceRepr<u8> for FfiBufferMut {}
unsafe impl MutSliceRepr<u8> for FfiBufferMut {}

/// Writable validity descriptor. Safe construction requires a
/// [`PreparedFfiValidityMut`] owner, so it cannot outlive or escape the mutable
/// bitmap borrow.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiValidityMut {
    #[staged(FfiBufferMut)]
    bytes: FfiBufferMut,
    #[staged(u64)]
    bit_offset: u64,
    #[staged(u64)]
    bit_len: u64,
    #[staged(u64)]
    null_count: u64,
}

/// Host owner for a writable validity descriptor.
pub struct PreparedFfiValidityMut<'data> {
    descriptor: FfiValidityMut,
    _borrow: PhantomData<&'data mut [u8]>,
}

impl PreparedFfiValidityMut<'_> {
    /// Borrow the descriptor for a generated call. The borrow cannot outlive
    /// the original mutable bitmap borrow held by this owner.
    pub fn descriptor_mut(&mut self) -> &mut FfiValidityMut {
        &mut self.descriptor
    }

    pub fn null_count(&self) -> u64 {
        self.descriptor.null_count
    }
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
    BitmapTooShort {
        bit_len: usize,
        byte_len: usize,
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
            Self::BitmapTooShort { bit_len, byte_len } => write!(
                f,
                "validity bitmap has {byte_len} bytes, too short for {bit_len} bits"
            ),
        }
    }
}

impl std::error::Error for FfiError {}

/// Build an erased descriptor from an Arrow primitive array.
fn ffi_from_primitive<A: ArrowPrimitiveType>(array: &PrimitiveArray<A>) -> FfiArray {
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
fn ffi_from_string_view(array: &StringViewArray) -> FfiArray {
    FfiArray {
        values: FfiBuffer::from_typed_slice(array.views().as_ref()),
        validity: FfiValidity::from_nulls(array.nulls(), array.len()),
        // Opaque pointer to the array itself, for extern `&str`/transform calls.
        array: (array as *const StringViewArray).cast::<u8>(),
    }
}

/// Prepare a writable validity bitmap over `bit_len` leading bits.
pub fn prepare_validity_mut(
    bitmap: &mut [u8],
    bit_len: usize,
) -> Result<PreparedFfiValidityMut<'_>, FfiError> {
    let required = bit_len.div_ceil(8);
    if bitmap.len() < required {
        return Err(FfiError::BitmapTooShort {
            bit_len,
            byte_len: bitmap.len(),
        });
    }
    let valid_count = (0..bit_len)
        .filter(|bit| bitmap[*bit / 8] & (1 << (*bit % 8)) != 0)
        .count();
    Ok(PreparedFfiValidityMut {
        descriptor: FfiValidityMut {
            bytes: FfiBufferMut {
                ptr: bitmap.as_mut_ptr(),
                len: required,
            },
            bit_offset: 0,
            bit_len: bit_len as u64,
            null_count: (bit_len - valid_count) as u64,
        },
        _borrow: PhantomData,
    })
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

    fn assert_read_slice_representation<T: StagedType>()
    where
        FfiBuffer: SliceRepr<T>,
    {
    }

    fn assert_mut_slice_representation()
    where
        FfiBufferMut: SliceRepr<u8> + MutSliceRepr<u8>,
    {
    }

    #[test]
    fn buffer_is_an_explicit_slice_representation() {
        assert_read_slice_representation::<u8>();
        assert_read_slice_representation::<i64>();
        assert_mut_slice_representation();
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
