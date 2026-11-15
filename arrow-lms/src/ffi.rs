//! FFI descriptors for Arrow arrays, read-only.
//!
//! The descriptors are lifetime-free because generated code consumes their `repr(C)`
//! layout. Safe host construction is lifetime-bearing: a batch's descriptors come from
//! [`PreparedFfiBatch`] (borrowed) or [`FfiRecordBatch`] (owned).

use std::fmt;
use std::marker::PhantomData;

use arrow::array::types::ArrowPrimitiveType;
use arrow::array::{
    Array, ArrayRef, BooleanArray, Float32Array, Float64Array, Int8Array, Int16Array, Int32Array,
    Int64Array, PrimitiveArray, StringViewArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

use crate::batch::FfiBatch;
use crate::byte_view::ByteViewDescriptors;
use crate::validity::FfiValidity;

/// An array's values as an erased `(ptr, len)`: what [`FfiArray`] holds until arrays are
/// typed (`relational/jit_stage6a.md` M2), and nothing else. Every other descriptor field
/// is an [`FfiSlice<T>`].
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
    pub(crate) const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    pub(crate) fn from_typed_slice<T>(slice: &[T]) -> Self {
        Self {
            ptr: slice.as_ptr().cast::<u8>(),
            len: slice.len(),
        }
    }
}

/// Erased Arrow array descriptor: values and validity. A `Utf8View` column's values are
/// its views, one `u128` per row; its strings are read through the batch's
/// [`FfiByteViewArray`](crate::FfiByteViewArray) for the column (see [`FfiBatch`]).
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiArray {
    #[staged(FfiBuffer)]
    values: FfiBuffer,
    #[staged(FfiValidity)]
    validity: FfiValidity,
}

impl FfiArray {
    /// No rows: what a consumer stands in where a column is missing, so a row lookup
    /// finds no value instead of branching on presence. Its buffers are null.
    pub const fn empty() -> Self {
        Self {
            values: FfiBuffer::empty(),
            validity: FfiValidity::all_valid(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len
    }

    pub fn is_empty(&self) -> bool {
        self.values.len == 0
    }

    pub fn null_count(&self) -> u64 {
        self.validity.null_count()
    }
}

/// One batch's descriptors and what [`FfiBatch`] points to: no lifetime, so its owner
/// keeps the arrays alive (a borrow, or the batch itself).
#[derive(Debug)]
struct Descriptors {
    arrays: Box<[FfiArray]>,
    #[expect(dead_code, reason = "owns what `batch.strings` points to")]
    strings: ByteViewDescriptors<Utf8>,
    batch: FfiBatch,
}

impl Descriptors {
    /// Describe `arrays`, one batch's columns, all of one length.
    fn new<'a>(arrays: impl IntoIterator<Item = &'a dyn Array>) -> Result<Self, FfiError> {
        let arrays: Vec<&dyn Array> = arrays.into_iter().collect();
        let num_rows = arrays.first().map_or(0, |array| array.len());
        let ffi = arrays
            .iter()
            .enumerate()
            .map(|(index, array)| {
                if array.len() != num_rows {
                    return Err(FfiError::MismatchedLength {
                        index,
                        expected: num_rows,
                        actual: array.len(),
                    });
                }
                ffi_from_array(index, *array)
            })
            .collect::<Result<Box<[FfiArray]>, _>>()?;
        let strings = ByteViewDescriptors::new(
            arrays
                .iter()
                .map(|array| array.as_any().downcast_ref::<StringViewArray>()),
        );
        let batch = FfiBatch {
            num_rows: num_rows as u64,
            arrays: FfiSlice::new(&ffi),
            strings: FfiSlice::new(strings.arrays()),
        };
        Ok(Self {
            arrays: ffi,
            strings,
            batch,
        })
    }
}

/// Host-owned read descriptors borrowing an Arrow source. The `'data` borrow keeps the
/// source alive; [`batch`](Self::batch) is what a kernel reads, and
/// [`arrays`](Self::arrays) the columns alone, as `&[FfiArray]`.
#[derive(Debug)]
pub struct PreparedFfiBatch<'data> {
    descriptors: Descriptors,
    _borrow: PhantomData<&'data [u8]>,
}

/// A `RecordBatch` with its descriptors: what [`PreparedFfiBatch`] is for a batch that
/// is kept, not borrowed. Holding the batch keeps its buffers alive.
#[derive(Debug)]
pub struct FfiRecordBatch {
    descriptors: Descriptors,
    record_batch: RecordBatch,
}

impl FfiRecordBatch {
    pub fn new(record_batch: RecordBatch) -> Result<Self, FfiError> {
        let descriptors = Descriptors::new(record_batch.columns().iter().map(|a| a.as_ref()))?;
        Ok(Self {
            descriptors,
            record_batch,
        })
    }

    /// What a kernel reads; it points into `self`, wherever `self` moves.
    pub fn batch(&self) -> &FfiBatch {
        &self.descriptors.batch
    }

    pub fn record_batch(&self) -> &RecordBatch {
        &self.record_batch
    }
}

impl<'data> PreparedFfiBatch<'data> {
    /// The columns' descriptors: `SRef<Slice<FfiArray>>` at runtime.
    pub fn arrays(&self) -> &[FfiArray] {
        &self.descriptors.arrays
    }

    /// The batch: `SRef<FfiBatch>` at runtime.
    pub fn batch(&self) -> &FfiBatch {
        &self.descriptors.batch
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
    /// A `Boolean` array whose values start mid-byte, as a slice at a bit offset that is
    /// not a multiple of 8: its descriptor addresses whole bytes.
    UnalignedBooleanValues {
        index: usize,
        bit_offset: usize,
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
            Self::MismatchedLength {
                index,
                expected,
                actual,
            } => {
                write!(f, "array {index} has length {actual}, expected {expected}")
            }
            Self::UnalignedBooleanValues { index, bit_offset } => write!(
                f,
                "boolean array {index} starts at bit {bit_offset}, not on a byte"
            ),
        }
    }
}

impl std::error::Error for FfiError {}

/// Build an erased descriptor from an Arrow primitive array.
fn ffi_from_primitive<A: ArrowPrimitiveType>(array: &PrimitiveArray<A>) -> FfiArray {
    FfiArray {
        values: FfiBuffer::from_typed_slice(array.values()),
        validity: FfiValidity::from_nulls(array.nulls()),
    }
}

/// Build an erased descriptor from an Arrow `BooleanArray`: `values` addresses the
/// bit-packed values from the array's first bit, its length the row count (see
/// `BooleanArrayView`). Fails if the values start mid-byte.
fn ffi_from_boolean(index: usize, array: &BooleanArray) -> Result<FfiArray, FfiError> {
    let values = array.values();
    let bit_offset = values.offset();
    if !bit_offset.is_multiple_of(8) {
        return Err(FfiError::UnalignedBooleanValues { index, bit_offset });
    }
    let bytes = &values.inner().as_slice()[bit_offset / 8..];
    Ok(FfiArray {
        values: FfiBuffer {
            ptr: bytes.as_ptr(),
            len: array.len(),
        },
        validity: FfiValidity::from_nulls(array.nulls()),
    })
}

/// Build an erased descriptor from an Arrow `StringViewArray`: its views, one `u128` per
/// row, and validity. The strings themselves are read through [`FfiByteViewArray`](crate::FfiByteViewArray).
fn ffi_from_string_view(array: &StringViewArray) -> FfiArray {
    FfiArray {
        values: FfiBuffer::from_typed_slice(array.views().as_ref()),
        validity: FfiValidity::from_nulls(array.nulls()),
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
    Ok(PreparedFfiBatch {
        descriptors: Descriptors::new(arrays)?,
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
        DataType::Boolean => {
            let boolean = array
                .as_any()
                .downcast_ref::<BooleanArray>()
                .ok_or_else(|| FfiError::Downcast {
                    index,
                    data_type: array.data_type().clone(),
                })?;
            ffi_from_boolean(index, boolean)
        }
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
    use arrow::buffer::NullBuffer;

    fn assert_read_slice_representation<T: StagedType>()
    where
        FfiBuffer: SliceRepr<T>,
    {
    }

    #[test]
    fn buffer_is_an_explicit_slice_representation() {
        assert_read_slice_representation::<u8>();
        assert_read_slice_representation::<i64>();
    }

    #[test]
    fn descriptor_matches_arrow() {
        let array = Int32Array::from(vec![10, 20, 30]);
        let ffi = ffi_from_primitive(&array);
        assert_eq!(ffi.len(), 3);
        assert_eq!(ffi.null_count(), 0);
    }

    #[test]
    fn descriptor_tracks_nulls() {
        let nulls = NullBuffer::from(vec![true, false, true]);
        let array = Int32Array::new(vec![1, 99, 3].into(), Some(nulls));
        let ffi = ffi_from_primitive(&array);
        assert_eq!(ffi.len(), 3);
        assert_eq!(ffi.null_count(), 1);
    }
}
