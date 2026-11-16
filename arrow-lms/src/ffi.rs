//! Describing Arrow arrays and batches on the host, read-only.
//!
//! The descriptors are lifetime-free because generated code consumes their `repr(C)`
//! layout. Safe host construction is lifetime-bearing: a batch's descriptors come from
//! [`PreparedFfiBatch`] (borrowed) or [`FfiRecordBatch`] (owned).

use std::fmt;
use std::marker::PhantomData;

use arrow::array::{Array, ArrayRef, AsArray, BinaryViewArray, StringViewArray};
use arrow::datatypes::{
    ArrowPrimitiveType, DataType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
    Int64Type, Schema, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
};
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

use crate::array::{FfiArray, FfiBoolArray, FfiPrimArray};
use crate::batch::FfiBatch;
use crate::byte_view::ByteViewDescriptors;

/// One batch's descriptors and what [`FfiBatch`] points to: no lifetime, so its owner
/// keeps the arrays alive (a borrow, or the batch itself).
#[derive(Debug)]
struct Descriptors {
    arrays: Box<[FfiArray]>,
    #[expect(
        dead_code,
        reason = "owns the buffer table `Utf8View` columns point into"
    )]
    strings: ByteViewDescriptors<Utf8>,
    #[expect(
        dead_code,
        reason = "owns the buffer table `BinaryView` columns point into"
    )]
    binaries: ByteViewDescriptors<Binary>,
    batch: FfiBatch,
}

impl Descriptors {
    /// Describe `arrays`, one batch's columns, all of one length.
    fn new<'a>(arrays: impl IntoIterator<Item = &'a dyn Array>) -> Result<Self, FfiError> {
        let arrays: Vec<&dyn Array> = arrays.into_iter().collect();
        let num_rows = arrays.first().map_or(0, |array| array.len());
        // The byte-view columns' buffer tables first, so their descriptors can be copied
        // from them.
        let strings = ByteViewDescriptors::new(
            arrays
                .iter()
                .map(|array| array.as_any().downcast_ref::<StringViewArray>()),
        );
        let binaries = ByteViewDescriptors::new(
            arrays
                .iter()
                .map(|array| array.as_any().downcast_ref::<BinaryViewArray>()),
        );
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
                match array.data_type() {
                    DataType::Utf8View => Ok(strings.arrays()[index].into()),
                    DataType::BinaryView => Ok(binaries.arrays()[index].into()),
                    _ => describe(index, *array),
                }
            })
            .collect::<Result<Box<[FfiArray]>, _>>()?;
        let batch = FfiBatch {
            num_rows: num_rows as u64,
            arrays: FfiSlice::new(&ffi),
        };
        Ok(Self {
            arrays: ffi,
            strings,
            binaries,
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
///
/// It is built against the schema a kernel reads it as, so its descriptors are what a
/// [`SchemaBatch`](crate::SchemaBatch) of that schema claims.
#[derive(Debug)]
pub struct FfiRecordBatch {
    descriptors: Descriptors,
    record_batch: RecordBatch,
}

impl FfiRecordBatch {
    /// `record_batch`'s descriptors, if its columns have `schema`'s data types. Field
    /// names and nullability are not compared.
    pub fn new(record_batch: RecordBatch, schema: &Schema) -> Result<Self, FfiError> {
        let columns = record_batch.columns();
        if columns.len() != schema.fields().len() {
            return Err(FfiError::ColumnCount {
                expected: schema.fields().len(),
                actual: columns.len(),
            });
        }
        for (index, (column, field)) in columns.iter().zip(schema.fields()).enumerate() {
            if column.data_type() != field.data_type() {
                return Err(FfiError::ColumnType {
                    index,
                    expected: field.data_type().clone(),
                    actual: column.data_type().clone(),
                });
            }
        }
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
    /// The batch has a different number of columns from the schema it is read as.
    ColumnCount {
        expected: usize,
        actual: usize,
    },
    /// Column `index` is not of its schema field's type.
    ColumnType {
        index: usize,
        expected: DataType,
        actual: DataType,
    },
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedDataType { index, data_type } => write!(
                f,
                "array {index} has unsupported data type {data_type}; only primitive, Boolean and byte-view arrays are supported"
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
            Self::ColumnCount { expected, actual } => {
                write!(f, "the batch has {actual} columns, the schema {expected}")
            }
            Self::ColumnType {
                index,
                expected,
                actual,
            } => write!(f, "column {index} is {actual}, the schema's is {expected}"),
        }
    }
}

impl std::error::Error for FfiError {}

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

/// The descriptor of `array`, column `index`, of any type but the byte views (whose
/// descriptors point into their batch's buffer tables).
fn describe(index: usize, array: &dyn Array) -> Result<FfiArray, FfiError> {
    macro_rules! primitive {
        ($arrow:ty) => {
            FfiPrimArray::<<$arrow as ArrowPrimitiveType>::Native>::new(
                array.as_primitive::<$arrow>(),
            )
            .into()
        };
    }
    Ok(match array.data_type() {
        DataType::Int8 => primitive!(Int8Type),
        DataType::Int16 => primitive!(Int16Type),
        DataType::Int32 => primitive!(Int32Type),
        DataType::Int64 => primitive!(Int64Type),
        DataType::UInt8 => primitive!(UInt8Type),
        DataType::UInt16 => primitive!(UInt16Type),
        DataType::UInt32 => primitive!(UInt32Type),
        DataType::UInt64 => primitive!(UInt64Type),
        DataType::Float32 => primitive!(Float32Type),
        DataType::Float64 => primitive!(Float64Type),
        DataType::Boolean => FfiBoolArray::new(array.as_boolean()).into(),
        data_type => {
            return Err(FfiError::UnsupportedDataType {
                index,
                data_type: data_type.clone(),
            });
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{BooleanArray, Int32Array, StringArray};

    /// Every column is described as its own type, a `Utf8View` one included.
    #[test]
    fn each_column_is_its_own_type() {
        let ints = Int32Array::from(vec![10, 20, 30]);
        let bools = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let strings = StringViewArray::from(vec!["a", "b", "a string past twelve bytes"]);
        let prepared = prepare_dyn_arrays(&[&ints, &bools, &strings]).unwrap();
        let arrays = prepared.arrays();
        assert!(matches!(arrays[0], FfiArray::Int32(_)));
        assert!(matches!(arrays[1], FfiArray::Boolean(_)));
        assert!(matches!(arrays[2], FfiArray::Utf8View(_)));
        assert_eq!(arrays[1].null_count(), 1);
        assert_eq!(prepared.batch().num_rows(), 3);
    }

    /// A record batch is refused unless its column types are the schema's, in order.
    #[test]
    fn a_record_batch_is_checked_against_its_schema() {
        use std::sync::Arc;

        use arrow::datatypes::Field;

        let schema = Schema::new(vec![
            Field::new("n", DataType::Int32, false),
            Field::new("s", DataType::Utf8View, true),
        ]);
        let rb = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(StringViewArray::from(vec!["a", "b"])),
            ],
        )
        .unwrap();
        assert!(FfiRecordBatch::new(rb.clone(), &schema).is_ok());

        // Names and nullability are not compared.
        let renamed = Schema::new(vec![
            Field::new("m", DataType::Int32, true),
            Field::new("t", DataType::Utf8View, false),
        ]);
        assert!(FfiRecordBatch::new(rb.clone(), &renamed).is_ok());

        let swapped = Schema::new(vec![
            Field::new("s", DataType::Utf8View, true),
            Field::new("n", DataType::Int32, false),
        ]);
        assert!(matches!(
            FfiRecordBatch::new(rb.clone(), &swapped),
            Err(FfiError::ColumnType {
                index: 0,
                expected: DataType::Utf8View,
                actual: DataType::Int32,
            })
        ));

        let widened = Schema::new(vec![
            Field::new("n", DataType::Int64, false),
            Field::new("s", DataType::Utf8View, true),
        ]);
        assert!(matches!(
            FfiRecordBatch::new(rb.clone(), &widened),
            Err(FfiError::ColumnType { index: 0, .. })
        ));

        let fewer = Schema::new(vec![Field::new("n", DataType::Int32, false)]);
        assert!(matches!(
            FfiRecordBatch::new(rb, &fewer),
            Err(FfiError::ColumnCount {
                expected: 1,
                actual: 2
            })
        ));
    }

    #[test]
    fn other_types_are_refused() {
        let strings = StringArray::from(vec!["a"]);
        assert!(matches!(
            prepare_dyn_arrays(&[&strings]),
            Err(FfiError::UnsupportedDataType { index: 0, .. })
        ));
    }
}
