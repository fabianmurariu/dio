//! Write side: host output allocation + staged writes.
//!
//! There are **no** mutable descriptor types — the write path uses the same
//! lifetime-free [`FfiArray`] as the read path, reached through `SRefMut`. Host
//! code allocates a [`PreparedOutput`] (one value buffer per output column, plus
//! a validity bitmap for nullable ones), hands the kernel `&mut [FfiArray]`, and
//! afterwards assembles a `RecordBatch`. Staged writes go through
//! [`PrimitiveArrayView::set`] / [`set_null`](PrimitiveArrayView::set_null),
//! reached via [`MutBatchOps::column_mut`].
//!
//! Buffers are sized to the worst case (input rows); the kernel fills the first
//! `n` and returns `n`, and the host slices to `n`. No growth yet.

use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array, StringViewBuilder};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

use crate::array::{bit_location, PrimitiveArrayView, ValidityView};
use crate::ffi::{FfiArray, FfiArrayType, FfiBuffer, FfiValidity, FfiValidityType};

// =============================================================================
// Staged writes (SRefMut flavor of the shared view)
// =============================================================================

/// Staged write operations on a mutable batch (`&mut [FfiArray]`).
pub trait MutBatchOps<'r>: Staged<Out = SRefMut<'r, Slice<FfiArray>>> + Sized {
    /// A writable typed view of output column `index`.
    fn column_mut<M>(
        self,
        index: usize,
    ) -> PrimitiveArrayView<impl Staged<Out = SRefMut<'r, FfiArray>> + Clone + 'r, M>
    where
        Self: Clone + 'r,
        M: StagedType + 'r,
    {
        PrimitiveArrayView {
            array: self.get_mut_unchecked(index as u64),
            _elem: PhantomData,
        }
    }
}

impl<'r, B> MutBatchOps<'r> for B where B: Staged<Out = SRefMut<'r, Slice<FfiArray>>> + Sized {}

impl<P, M> PrimitiveArrayView<P, M>
where
    P: Staged<Out = SRefMut<'static, FfiArray>> + Clone + 'static,
    M: StagedType + 'static,
{
    /// Write `value` at output row `n`.
    pub fn set(&self, ctx: &mut Ctx, n: Var<u64>, value: Var<M>) {
        let values = field_addr(self.array.clone(), FfiArrayType::values()).as_mut_slice::<M>();
        ctx.emit(values.set_unchecked(n, value));
    }

    /// A mutable view of this column's validity bitmap (for `set_null` etc.).
    pub fn validity_mut(
        &self,
    ) -> ValidityView<impl Staged<Out = SRefMut<'static, FfiValidity>> + Clone + 'static> {
        ValidityView::new(field_addr(self.array.clone(), FfiArrayType::validity()))
    }
}

/// Mutable validity-bitmap ops — the write twin of the read-side `is_valid`.
/// They live on [`ValidityView`] (the staged counterpart of `FfiValidity`), so a
/// bitmap can be updated independently of any primitive array, and share
/// `bit_location` with `is_valid` so the bit arithmetic exists in one place.
impl<V> ValidityView<V>
where
    V: Staged<Out = SRefMut<'static, FfiValidity>> + Clone + 'static,
{
    /// Mark row `i` null (clear its validity bit): `byte &= ~mask`.
    pub fn set_null(&self, ctx: &mut Ctx, i: Var<u64>) {
        let bytes =
            field_addr(self.validity.clone(), FfiValidityType::bytes()).as_mut_slice::<u8>();
        let (byte_index, mask) = bit_location(self.validity.clone(), i);
        let byte_index = ctx.bind(byte_index);
        let old = int_cast::<u64, u8, _>(bytes.clone().get_unchecked(byte_index));
        let not_mask = bitxor::<u64, _, _>(mask, Const::<u64>::new(u64::MAX));
        let cleared = int_cast::<u8, u64, _>(bitand::<u64, _, _>(old, not_mask));
        ctx.emit(bytes.set_unchecked(byte_index, cleared));
    }

    /// Mark row `i` valid (set its validity bit): `byte |= mask`.
    pub fn set_valid(&self, ctx: &mut Ctx, i: Var<u64>) {
        let bytes =
            field_addr(self.validity.clone(), FfiValidityType::bytes()).as_mut_slice::<u8>();
        let (byte_index, mask) = bit_location(self.validity.clone(), i);
        let byte_index = ctx.bind(byte_index);
        let old = int_cast::<u64, u8, _>(bytes.clone().get_unchecked(byte_index));
        let set = int_cast::<u8, u64, _>(bitor::<u64, _, _>(old, mask));
        ctx.emit(bytes.set_unchecked(byte_index, set));
    }
}

// =============================================================================
// Host side: allocation + RecordBatch assembly
// =============================================================================

/// Host value storage for one output column. Two shapes:
///
/// - **Fixed-width** (`Vec<T>`): the kernel writes values by index into a flat
///   buffer via [`values_ptr`](OutColumn::values_ptr); nulls go in a separate
///   validity bitmap.
/// - **Append** (`StrOutColumn`, `Utf8View`): variable-length, so the kernel
///   *appends* through the column's own builder — reached via
///   [`builder_ptr`](OutColumn::builder_ptr), parked in the descriptor's opaque
///   `array` field — and the builder owns its nulls (no external bitmap).
///
/// Adding a fixed-width type is one arm in `out_column!` + one in [`alloc_column`].
trait OutColumn {
    /// Raw pointer to the flat value buffer (fixed-width columns).
    fn values_ptr(&mut self) -> *mut u8;
    /// Opaque pointer to this column's builder (append columns); null otherwise.
    fn builder_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }
    /// True for append (builder-backed) columns: no flat value buffer, and the
    /// builder owns its own nulls, so no external validity bitmap.
    fn is_append(&self) -> bool {
        false
    }
    fn into_array(self: Box<Self>, n: usize, nulls: Option<NullBuffer>) -> ArrayRef;
}

macro_rules! out_column {
    ($($native:ty => $array:ty),+ $(,)?) => {$(
        impl OutColumn for Vec<$native> {
            fn values_ptr(&mut self) -> *mut u8 {
                self.as_mut_ptr().cast()
            }
            fn into_array(mut self: Box<Self>, n: usize, nulls: Option<NullBuffer>) -> ArrayRef {
                self.truncate(n);
                Arc::new(<$array>::new(ScalarBuffer::from(*self), nulls))
            }
        }
    )+};
}

out_column!(i32 => Int32Array, i64 => Int64Array, f64 => Float64Array);

/// A `Utf8View` output column, backed by an Arrow [`StringViewBuilder`] the
/// kernel appends into. `finish()` produces the `StringViewArray` directly (data
/// buffers + views + nulls), zero-copy — no external buffer plumbing.
struct StrOutColumn {
    builder: StringViewBuilder,
}

impl OutColumn for StrOutColumn {
    fn values_ptr(&mut self) -> *mut u8 {
        std::ptr::null_mut()
    }
    fn builder_ptr(&mut self) -> *mut u8 {
        (&mut self.builder as *mut StringViewBuilder).cast()
    }
    fn is_append(&self) -> bool {
        true
    }
    fn into_array(mut self: Box<Self>, _n: usize, _nulls: Option<NullBuffer>) -> ArrayRef {
        // The builder appended exactly one value per emitted row (== `n`) and
        // owns its own nulls, so ignore the external `nulls`.
        Arc::new(self.builder.finish())
    }
}

fn alloc_column(dt: &DataType, capacity: usize) -> Box<dyn OutColumn> {
    match dt {
        DataType::Int32 => Box::new(vec![0i32; capacity]),
        DataType::Int64 => Box::new(vec![0i64; capacity]),
        DataType::Float64 => Box::new(vec![0.0f64; capacity]),
        DataType::Utf8View => Box::new(StrOutColumn {
            builder: StringViewBuilder::new(),
        }),
        other => panic!("unsupported output column type: {other}"),
    }
}

/// Host-owned output storage. Owns the value/validity buffers so the raw
/// pointers in `descriptors` stay valid for the JIT call; must outlive it.
pub struct PreparedOutput {
    schema: SchemaRef,
    columns: Vec<Box<dyn OutColumn>>,
    /// Per-column validity bitmap (all-valid `0xFF` init); `None` when the
    /// output field is non-nullable.
    validity: Vec<Option<Vec<u8>>>,
    descriptors: Vec<FfiArray>,
}

impl PreparedOutput {
    /// Allocate output buffers for `schema`, each sized to `capacity` rows.
    pub fn alloc(schema: SchemaRef, capacity: usize) -> Self {
        let n = schema.fields().len();
        let mut columns = Vec::with_capacity(n);
        let mut validity = Vec::with_capacity(n);
        let mut descriptors = Vec::with_capacity(n);

        for field in schema.fields() {
            let mut col = alloc_column(field.data_type(), capacity);

            // Append (string) columns don't use a flat value buffer or an external
            // validity bitmap — the kernel appends through the builder pointer
            // parked in the descriptor's `array` field, and the builder owns nulls.
            if col.is_append() {
                let array = col.builder_ptr().cast_const();
                descriptors.push(FfiArray {
                    values: unsafe { FfiBuffer::from_raw_parts(std::ptr::null_mut(), 0) },
                    validity: FfiValidity::all_valid(capacity),
                    array,
                });
                columns.push(col);
                validity.push(None);
                continue;
            }

            let values = unsafe { FfiBuffer::from_raw_parts(col.values_ptr(), capacity) };

            let mut valid = field
                .is_nullable()
                .then(|| vec![0xFFu8; capacity.div_ceil(8)]);
            let validity_desc = match &mut valid {
                Some(bytes) => FfiValidity {
                    bytes: unsafe { FfiBuffer::from_raw_parts(bytes.as_mut_ptr(), bytes.len()) },
                    bit_offset: 0,
                    bit_len: capacity as u64,
                    null_count: 0,
                },
                None => FfiValidity::all_valid(capacity),
            };

            // SAFETY: `col`/`valid` heap buffers stay put once allocated (never
            // grown), and `PreparedOutput` keeps them alive for the call.
            descriptors.push(FfiArray {
                values,
                validity: validity_desc,
                // Output columns are fixed-width for now — no originating array.
                array: std::ptr::null(),
            });
            columns.push(col);
            validity.push(valid);
        }

        Self {
            schema,
            columns,
            validity,
            descriptors,
        }
    }

    /// The mutable batch handed to the kernel: `SRefMut<Slice<FfiArray>>` at
    /// runtime (`&mut [FfiArray]`).
    pub fn as_ffi_mut(&mut self) -> &mut [FfiArray] {
        &mut self.descriptors
    }

    /// Assemble the first `n` rows into a `RecordBatch`.
    pub fn into_record_batch(self, n: usize) -> RecordBatch {
        let arrays = self
            .columns
            .into_iter()
            .zip(self.validity)
            .map(|(col, valid)| {
                let nulls = valid.map(|bytes| {
                    NullBuffer::new(BooleanBuffer::new(Buffer::from_vec(bytes), 0, n))
                });
                col.into_array(n, nulls)
            })
            .collect::<Vec<_>>();
        RecordBatch::try_new(self.schema, arrays).expect("output columns match schema")
    }
}
