//! Host-driven input streaming: the kernel pulls batches from [`Inputs`] via the
//! [`scan_next`] extern until each table's stream is exhausted.
//!
//! One [`ScanStream`] per table owns a `Box<dyn Iterator<Item = RecordBatch>>` and
//! the lifecycle of the single live batch: each `next_batch`
//! drops the previous `RecordBatch` (freeing its Arrow buffers) before pulling the
//! next, so a scan over a huge table keeps only **one input batch resident** at a
//! time. `next_batch` returns a pointer to the batch's [`FfiBatch`] (or null at
//! end-of-stream), which the kernel reads its rows through. See
//! `docs/table_scan.md` §3/§6.
//!
//! Single-partition for now; a JOIN's build side (which must retain its batches)
//! and parallelism are later milestones.

use std::ptr;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiBatch, FfiRecordBatch};
use datafusion_common::{DataFusionError, Result};
use rust_lms::prelude::*;

/// One table's batch stream plus the live batch's FFI descriptors.
pub struct ScanStream {
    expected_schema: SchemaRef,
    iter: Box<dyn Iterator<Item = RecordBatch>>,
    /// The batch currently exposed to the kernel, with its descriptors, which stay
    /// valid until the following `next_batch`. Dropping it frees that batch's Arrow
    /// buffers; exactly one input batch is resident at a time.
    current: Option<FfiRecordBatch>,
}

impl ScanStream {
    /// A stream over a boxed iterator of batches.
    pub fn new(expected_schema: SchemaRef, iter: Box<dyn Iterator<Item = RecordBatch>>) -> Self {
        ScanStream {
            expected_schema,
            iter,
            current: None,
        }
    }

    fn validate_schema(&self, rb: &RecordBatch) -> Result<()> {
        if rb.schema().as_ref() == self.expected_schema.as_ref() {
            return Ok(());
        }

        Err(DataFusionError::Execution(format!(
            "input batch schema does not match declared stream schema: expected {:?}, actual {:?}",
            self.expected_schema,
            rb.schema()
        )))
    }

    /// Advance to the next batch and return its descriptors, or `None` when the
    /// stream is exhausted. Drops the previous batch first.
    fn next_batch(&mut self) -> Result<Option<&FfiBatch>> {
        // Release the previous batch (its Arrow buffers are freed here).
        self.current = None;
        let Some(rb) = self.iter.next() else {
            return Ok(None);
        };
        self.validate_schema(&rb)?;
        let batch = FfiRecordBatch::new(rb).map_err(|error| {
            DataFusionError::Execution(format!("failed to prepare input batch: {error}"))
        })?;
        Ok(Some(self.current.insert(batch).batch()))
    }

    fn drain_validated(&mut self) -> Result<Box<dyn Iterator<Item = RecordBatch>>> {
        self.current = None;
        let iter = std::mem::replace(&mut self.iter, Box::new(std::iter::empty()));
        let mut batches = Vec::new();
        for rb in iter {
            self.validate_schema(&rb)?;
            batches.push(rb);
        }
        Ok(Box::new(batches.into_iter()))
    }
}

/// The kernel's input handle: one [`ScanStream`] per table, indexed by table id.
pub struct Inputs {
    streams: Vec<ScanStream>,
    error: Option<DataFusionError>,
}

impl Inputs {
    /// Build inputs from one stream per table (in table-id order).
    pub fn new(streams: Vec<ScanStream>) -> Self {
        Inputs {
            streams,
            error: None,
        }
    }

    /// A single table fed by exactly one in-memory batch (the current
    /// `exec_jit(sql, table, &rb)` shape — a one-element stream).
    pub fn single(rb: RecordBatch) -> Self {
        let schema = rb.schema();
        Inputs::new(vec![ScanStream::new(schema, Box::new(std::iter::once(rb)))])
    }

    /// Take table `table`'s remaining batch iterator, leaving an empty stream in its
    /// slot (so other tables keep their ids). Used to drain a hash join's build side
    /// host-side before the probe kernel runs. Valid only on an un-probed stream.
    pub fn drain_table(&mut self, table: usize) -> Result<Box<dyn Iterator<Item = RecordBatch>>> {
        self.streams
            .get_mut(table)
            .ok_or_else(|| {
                DataFusionError::Execution(format!("input table index {table} is out of bounds"))
            })?
            .drain_validated()
    }

    pub(crate) fn take_error(&mut self) -> Result<()> {
        match self.error.take() {
            Some(error) => Err(error),
            None => Ok(()),
        }
    }

    fn record_error(&mut self, error: DataFusionError) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }
}

/// Pull the next batch of table `table`, returning its descriptors (null = exhausted).
/// `table` is a stage-0 constant in the kernel, so dispatch is free.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn scan_next(inputs: &mut Inputs, table: u64) -> *const FfiBatch {
    let result = usize::try_from(table)
        .map_err(|_| DataFusionError::Execution(format!("input table index {table} is invalid")))
        .and_then(|table| {
            inputs
                .streams
                .get_mut(table)
                .ok_or_else(|| {
                    DataFusionError::Execution(format!(
                        "input table index {table} is out of bounds"
                    ))
                })?
                .next_batch()
                .map(|batch| batch.map(|batch| batch as *const FfiBatch))
        });

    match result {
        Ok(Some(arrays)) => arrays,
        Ok(None) => ptr::null(),
        Err(error) => {
            inputs.record_error(error);
            ptr::null()
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};

    use super::*;

    fn batch(vals: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vals))]).unwrap()
    }

    /// The stream hands out each batch's descriptors (with the right row count),
    /// keeps exactly one batch resident, and returns null once exhausted.
    #[test]
    fn stream_yields_each_batch_then_null() {
        let first = batch(vec![1, 2, 3]);
        let schema = first.schema();
        let mut s = ScanStream::new(schema, Box::new(vec![first, batch(vec![4, 5])].into_iter()));

        let p1 = s.next_batch().unwrap();
        assert_eq!(p1.map(FfiBatch::num_columns), Some(1));
        assert_eq!(p1.map(FfiBatch::num_rows), Some(3)); // batch 1: 3 rows
        assert!(s.current.is_some()); // one batch resident

        let p2 = s.next_batch().unwrap();
        assert_eq!(p2.map(FfiBatch::num_columns), Some(1));
        assert_eq!(p2.map(FfiBatch::num_rows), Some(2)); // batch 2: 2 rows
        assert!(s.current.is_some());

        assert!(s.next_batch().unwrap().is_none()); // exhausted
        assert!(s.current.is_none()); // last batch dropped
    }
}
