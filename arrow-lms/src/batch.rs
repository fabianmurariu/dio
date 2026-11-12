//! A batch as a kernel reads it: [`FfiBatch`], its row count and a descriptor per column.
//!
//! Every column has an [`FfiArray`] (a `Utf8View` column's describes its views and
//! validity) and a [`FfiByteViewArray<Utf8>`] (empty unless the column is `Utf8View`), at
//! the column's own index in both, so a kernel reads column `i` of either kind without a
//! map from columns to descriptors.

use rust_lms::prelude::*;

use crate::byte_view::{ByteViewArrayView, FfiByteViewArray};
use crate::ffi::{FfiArray, FfiBuffer};

/// One batch's descriptors, built by [`PreparedFfiBatch`](crate::PreparedFfiBatch) or
/// [`FfiRecordBatch`](crate::FfiRecordBatch), which own what it points to.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiBatch {
    #[staged(u64)]
    pub(crate) num_rows: u64,
    /// `SliceRepr<FfiArray>`: one per column.
    #[staged(FfiBuffer)]
    pub(crate) arrays: FfiBuffer,
    /// `SliceRepr<FfiByteViewArray<Utf8>>`: one per column.
    #[staged(FfiBuffer)]
    pub(crate) strings: FfiBuffer,
}

impl FfiBatch {
    pub fn num_rows(&self) -> usize {
        self.num_rows as usize
    }

    /// How many columns.
    pub fn num_columns(&self) -> usize {
        self.arrays.len()
    }
}

/// Staged reads of a `&FfiBatch`.
pub trait FfiBatchOps: Staged<Out = SRef<FfiBatch>> + Sized + Clone + 'static {
    /// The row count.
    fn num_rows(self) -> impl Staged<Out = u64> + Clone {
        // SAFETY: a shared reference to a live descriptor.
        unsafe { load_field_unchecked(ref_as_ptr(self), FfiBatchType::num_rows()) }
    }

    /// The columns' descriptors, as a **trusted** slice: read them with
    /// [`ArrayBatchOps`](crate::ArrayBatchOps).
    fn arrays(self) -> impl Staged<Out = SRef<Slice<FfiArray>>> + Clone {
        // SAFETY: the owner of the batch keeps its descriptors alive and unchanged while
        // the descriptor is read; the read is shared.
        unsafe {
            field_addr(ref_as_ptr(self), FfiBatchType::arrays())
                .into_raw_slice::<FfiArray>()
                .assume_shared()
        }
    }

    /// Column `index` as strings.
    ///
    /// # Safety
    ///
    /// `index` is a column of every batch the kernel reads, and a `Utf8View` one: any
    /// other column reads as an empty array, whose rows are out of bounds.
    unsafe fn strings(
        self,
        index: usize,
    ) -> ByteViewArrayView<impl Staged<Out = SPtr<FfiByteViewArray<Utf8>>> + Clone, Utf8> {
        // SAFETY: one descriptor per column, owned as `arrays`; `index` is a column (the
        // caller).
        let array = unsafe {
            ptr_offset(
                field_addr(ref_as_ptr(self), FfiBatchType::strings())
                    .into_raw_slice::<FfiByteViewArray<Utf8>>()
                    .into_ptr(),
                Const::<i64>::new(index as i64),
            )
        };
        ByteViewArrayView::new(array)
    }
}

impl<B> FfiBatchOps for B where B: Staged<Out = SRef<FfiBatch>> + Sized + Clone + 'static {}
