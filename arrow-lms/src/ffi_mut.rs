//! Write side: mutable FFI descriptors for materializing output columns.
//!
//! The mirror of [`crate::ffi`]. Host code pre-allocates a [`PreparedOutput`]
//! (one value buffer per output column, plus a validity bitmap for nullable
//! ones), hands the JIT kernel an `&mut FfiMutableArrays`, and afterwards turns
//! the filled buffers into a `RecordBatch`. Staged code writes through
//! [`MutablePrimitiveView::set`] / [`MutablePrimitiveView::set_null`].
//!
//! Unlike the read descriptors these are **lifetime-free** (raw pointers, no
//! `PhantomData` borrow). A `&mut` is invariant in its pointee's lifetimes, so
//! carrying `'a` here would make `&mut FfiMutableArrays<'a>` un-passable to a
//! kernel whose ABI type is `'static`; a lifetime-free pointee coerces cleanly
//! (exactly like `&mut [i64]`). Safety is the caller's contract: the
//! `PreparedOutput` must outlive the kernel call.
//!
//! Buffers are sized to the worst case (input row count); the kernel fills the
//! first `n` rows and returns `n`, and the host slices to `n`. No growth yet.

use std::marker::PhantomData;
use std::sync::Arc;

use arrow::array::{ArrayRef, Float64Array, Int32Array, Int64Array};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer, ScalarBuffer};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use rust_lms::prelude::*;

/// Erased *mutable* `(ptr, len)` buffer — the write-side twin of
/// [`crate::ffi::FfiBuffer`].
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiMutBuffer {
    #[staged(u64)]
    pub ptr: *mut u8,
    #[staged(u64)]
    pub len: usize,
}

impl FfiMutBuffer {
    /// # Safety
    /// `ptr` must be valid for writes of `len` elements under the interpretation
    /// attached by the owning descriptor, for as long as the descriptor is used.
    pub const unsafe fn from_raw_parts(ptr: *mut u8, len: usize) -> Self {
        Self { ptr, len }
    }
}

/// Erased writable primitive output array: a values buffer plus a validity
/// bitmap (`len == 0` for non-nullable columns).
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiMutableArray {
    #[staged(FfiMutBuffer)]
    pub values: FfiMutBuffer,
    #[staged(FfiMutBuffer)]
    pub validity: FfiMutBuffer,
}

/// Erased writable batch: a buffer of [`FfiMutableArray`] descriptors.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiMutableArrays {
    #[staged(FfiMutBuffer)]
    pub arrays: FfiMutBuffer,
}

// =============================================================================
// Host side: allocation + RecordBatch assembly
// =============================================================================

/// Typed value storage for one output column (len = capacity).
enum OutColumn {
    I32(Vec<i32>),
    I64(Vec<i64>),
    F64(Vec<f64>),
}

impl OutColumn {
    fn alloc(dt: &DataType, capacity: usize) -> Self {
        match dt {
            DataType::Int32 => OutColumn::I32(vec![0; capacity]),
            DataType::Int64 => OutColumn::I64(vec![0; capacity]),
            DataType::Float64 => OutColumn::F64(vec![0.0; capacity]),
            other => panic!("unsupported output column type: {other}"),
        }
    }

    fn values_ptr(&mut self) -> *mut u8 {
        match self {
            OutColumn::I32(v) => v.as_mut_ptr().cast(),
            OutColumn::I64(v) => v.as_mut_ptr().cast(),
            OutColumn::F64(v) => v.as_mut_ptr().cast(),
        }
    }

    fn into_array(self, n: usize, nulls: Option<NullBuffer>) -> ArrayRef {
        match self {
            OutColumn::I32(mut v) => {
                v.truncate(n);
                Arc::new(Int32Array::new(ScalarBuffer::from(v), nulls))
            }
            OutColumn::I64(mut v) => {
                v.truncate(n);
                Arc::new(Int64Array::new(ScalarBuffer::from(v), nulls))
            }
            OutColumn::F64(mut v) => {
                v.truncate(n);
                Arc::new(Float64Array::new(ScalarBuffer::from(v), nulls))
            }
        }
    }
}

/// Host-owned output storage. Owns the value/validity buffers so the raw
/// pointers in `descriptors` stay valid for the JIT call; must outlive it.
pub struct PreparedOutput {
    schema: SchemaRef,
    columns: Vec<OutColumn>,
    /// Per-column validity bitmap (all-valid `0xFF` init); `None` when the
    /// output field is non-nullable.
    validity: Vec<Option<Vec<u8>>>,
    descriptors: Vec<FfiMutableArray>,
}

impl PreparedOutput {
    /// Allocate output buffers for `schema`, each sized to `capacity` rows.
    pub fn alloc(schema: SchemaRef, capacity: usize) -> Self {
        let n = schema.fields().len();
        let mut columns = Vec::with_capacity(n);
        let mut validity = Vec::with_capacity(n);
        let mut descriptors = Vec::with_capacity(n);

        for field in schema.fields() {
            let mut col = OutColumn::alloc(field.data_type(), capacity);
            let values_ptr = col.values_ptr();

            let mut valid = field
                .is_nullable()
                .then(|| vec![0xFFu8; capacity.div_ceil(8)]);
            let (validity_ptr, validity_len) = match &mut valid {
                Some(bytes) => (bytes.as_mut_ptr(), bytes.len()),
                None => (std::ptr::null_mut(), 0),
            };

            // SAFETY: `col`/`valid` heap buffers stay put once allocated (never
            // grown), and `PreparedOutput` keeps them alive for the call.
            descriptors.push(FfiMutableArray {
                values: unsafe { FfiMutBuffer::from_raw_parts(values_ptr, capacity) },
                validity: unsafe { FfiMutBuffer::from_raw_parts(validity_ptr, validity_len) },
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

    /// The output descriptor handed to the kernel (raw-pointer-backed; borrows
    /// nothing, so it does not conflict with [`Self::into_record_batch`]).
    pub fn as_ffi_mut(&mut self) -> FfiMutableArrays {
        let ptr = self.descriptors.as_mut_ptr().cast::<u8>();
        let len = self.descriptors.len();
        FfiMutableArrays {
            arrays: unsafe { FfiMutBuffer::from_raw_parts(ptr, len) },
        }
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

// =============================================================================
// Staged write views
// =============================================================================

/// Staged operations on the `&mut FfiMutableArrays` output parameter.
pub trait FfiMutableArraysOps<'r>: Staged<Out = SRefMut<'r, FfiMutableArrays>> + Sized {
    /// A writable view of output column `index`, typed as `M`.
    fn column_mut<M>(
        self,
        index: usize,
    ) -> MutablePrimitiveView<impl Staged<Out = SRefMut<'r, FfiMutableArray>> + Clone + 'r, M>
    where
        Self: Clone + 'r,
        M: StagedType + 'r,
    {
        let array = field_addr(self, FfiMutableArraysType::arrays())
            .as_mut_slice::<FfiMutableArray>()
            .get_mut_unchecked(index as u64);
        MutablePrimitiveView {
            array,
            _elem: PhantomData,
        }
    }
}

impl<'r, B> FfiMutableArraysOps<'r> for B where
    B: Staged<Out = SRefMut<'r, FfiMutableArrays>> + Sized
{
}

/// A writable view of one primitive output column.
pub struct MutablePrimitiveView<P, M> {
    array: P,
    _elem: PhantomData<M>,
}

impl<P: Clone, M> Clone for MutablePrimitiveView<P, M> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, M> Copy for MutablePrimitiveView<P, M> {}

impl<P, M> MutablePrimitiveView<P, M>
where
    P: Staged<Out = SRefMut<'static, FfiMutableArray>> + Clone + 'static,
    M: StagedType + 'static,
{
    /// Write `value` at output row `n`.
    pub fn set(&self, ctx: &mut Ctx, n: Var<u64>, value: Var<M>) {
        let values =
            field_addr(self.array.clone(), FfiMutableArrayType::values()).as_mut_slice::<M>();
        ctx.emit(values.set_unchecked(n, value));
    }

    /// Clear the validity bit at output row `n` (mark it null). Assumes the
    /// bitmap was default-initialized all-valid.
    pub fn set_null(&self, ctx: &mut Ctx, n: Var<u64>) {
        let bitmap =
            field_addr(self.array.clone(), FfiMutableArrayType::validity()).as_mut_slice::<u8>();
        let byte_idx = ctx.bind(shr::<u64, _, _>(n, 3u64));
        let bit = bitand::<u64, _, _>(n, 7u64);
        // clear = old & ~(1 << bit)   (~x computed as x ^ all-ones)
        let old = int_cast::<u64, u8, _>(bitmap.clone().get_unchecked(byte_idx));
        let mask = shl::<u64, _, _>(Const::<u64>::new(1), bit);
        let not_mask = bitxor::<u64, _, _>(mask, Const::<u64>::new(u64::MAX));
        let cleared = int_cast::<u8, u64, _>(bitand::<u64, _, _>(old, not_mask));
        ctx.emit(bitmap.set_unchecked(byte_idx, cleared));
    }
}
