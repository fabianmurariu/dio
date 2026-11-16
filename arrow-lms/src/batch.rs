//! A batch as a kernel reads it: [`FfiBatch`], its row count and an [`FfiArray`] per
//! column.

use rust_lms::prelude::*;

use crate::array::{ArrayBatchOps, ArrowArray, FfiArray};

/// One batch's descriptors, built by [`PreparedFfiBatch`](crate::PreparedFfiBatch) or
/// [`FfiRecordBatch`](crate::FfiRecordBatch), which own what it points to.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiBatch {
    #[staged(u64)]
    pub(crate) num_rows: u64,
    /// One per column.
    pub(crate) arrays: FfiSlice<FfiArray>,
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

    /// The columns, as a **trusted** slice: read them with
    /// [`ArrayBatchOps`].
    fn arrays(self) -> impl Staged<Out = SRef<Slice<FfiArray>>> + Clone {
        // SAFETY: the owner of the batch keeps its descriptors alive and unchanged while
        // the descriptor is read; the read is shared.
        unsafe {
            field_addr(ref_as_ptr(self), FfiBatchType::arrays())
                .into_raw_slice::<FfiArray>()
                .assume_shared()
        }
    }

    /// Column `index`, read as an `A`.
    ///
    /// # Safety
    ///
    /// `index` is a column of every batch the kernel reads, and holds an `A` in each.
    unsafe fn column<A: ArrowArray>(self, index: usize) -> impl Staged<Out = SRef<A>> + Clone {
        // SAFETY: forwarded from the caller.
        unsafe { self.arrays().column::<A>(index) }
    }
}

impl<B> FfiBatchOps for B where B: Staged<Out = SRef<FfiBatch>> + Sized + Clone + 'static {}
