//! A batch as a kernel reads it: [`FfiBatch`], its row count and an [`FfiArray`] per
//! column, and [`SchemaBatch`], a staged batch whose columns are read as the types its
//! schema gives them.

use std::any::type_name;

use arrow::datatypes::SchemaRef;
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

    /// The columns, as a **trusted** slice. Each is read as the type it holds with
    /// [`ArrayBatchOps::column`], a claim that is `unsafe`; [`SchemaBatch`] makes it
    /// once for the whole batch.
    fn arrays(self) -> impl Staged<Out = SRef<Slice<FfiArray>>> + Clone {
        // SAFETY: the owner of the batch keeps its descriptors alive and unchanged while
        // the descriptor is read; the read is shared.
        unsafe {
            field_addr(ref_as_ptr(self), FfiBatchType::arrays())
                .into_raw_slice::<FfiArray>()
                .assume_shared()
        }
    }
}

impl<B> FfiBatchOps for B where B: Staged<Out = SRef<FfiBatch>> + Sized + Clone + 'static {}

/// A staged batch with the schema of every batch it is at run time, so a column is read
/// as its schema type with no `unsafe` at the read: [`column`](Self::column) checks the
/// type while staging.
///
/// The claim that the batches match the schema is made once, by [`new`](Self::new), where
/// the batch enters the kernel. The host keeps it by building each batch with
/// [`FfiRecordBatch::new`](crate::FfiRecordBatch::new), which refuses a batch whose
/// column types are not the schema's.
#[derive(Clone, Debug)]
pub struct SchemaBatch<B> {
    batch: B,
    schema: SchemaRef,
}

impl<B: FfiBatchOps> SchemaBatch<B> {
    /// `batch`, whose batches have `schema`'s columns.
    ///
    /// # Safety
    ///
    /// At execution, every batch `batch` holds has as many columns as `schema`, each of
    /// its field's data type.
    pub unsafe fn new(batch: B, schema: SchemaRef) -> Self {
        Self { batch, schema }
    }

    /// The schema the batches have.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The row count.
    pub fn num_rows(&self) -> impl Staged<Out = u64> + Clone + use<B> {
        self.batch.clone().num_rows()
    }

    /// Column `index`, read as an `A`.
    ///
    /// # Panics
    ///
    /// While staging, if the schema has no column `index`, or its data type is not one an
    /// `A` describes.
    pub fn column<A: ArrowArray>(
        &self,
        index: usize,
    ) -> impl Staged<Out = SRef<A>> + Clone + use<A, B> {
        let fields = self.schema.fields();
        let Some(field) = fields.get(index) else {
            panic!(
                "column {index} read from a batch of {} columns",
                fields.len()
            );
        };
        assert!(
            A::accepts(field.data_type()),
            "column {index} ({}) is {}, not read as {}",
            field.name(),
            field.data_type(),
            type_name::<A>(),
        );
        // SAFETY: every batch has the schema's columns (`new`), and column `index`'s
        // type is one an `A` describes, so it holds an `A`.
        unsafe { self.batch.clone().arrays().column::<A>(index) }
    }
}
