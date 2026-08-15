//! Host-driven input streaming: the kernel pulls batches from [`Inputs`] via the
//! [`scan_next`] extern until each table's stream is exhausted.
//!
//! One [`ScanStream`] per table owns a `Box<dyn Iterator<Item = RecordBatch>>` and
//! the lifecycle of the single live batch: each [`next_batch`](ScanStream::next_batch)
//! drops the previous `RecordBatch` (freeing its Arrow buffers) before pulling the
//! next, so a scan over a huge table keeps only **one input batch resident** at a
//! time. `next_batch` returns a pointer to a reused descriptor buffer (or null at
//! end-of-stream); the kernel rebuilds a `&[FfiArray]` batch from it and runs the
//! inner row loop. See `docs/table_scan.md` §3/§6.
//!
//! Single-partition for now; a JOIN's build side (which must retain its batches)
//! and parallelism are later milestones.

use std::ptr;

use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArray, prepare_record_batch};
use rust_lms::prelude::*;

/// One table's batch stream plus the live batch's FFI descriptors.
pub struct ScanStream {
    iter: Box<dyn Iterator<Item = RecordBatch>>,
    /// The batch currently exposed to the kernel. Dropping it frees that batch's
    /// Arrow buffers; exactly one input batch is resident at a time.
    current: Option<RecordBatch>,
    /// Reused descriptor buffer, refilled in place each `next_batch`. Its pointer is
    /// handed to the kernel and stays valid until the following `next_batch`.
    descs: Vec<FfiArray>,
}

impl ScanStream {
    /// A stream over a boxed iterator of batches.
    pub fn new(iter: Box<dyn Iterator<Item = RecordBatch>>) -> Self {
        ScanStream {
            iter,
            current: None,
            descs: Vec::new(),
        }
    }

    /// Advance to the next batch; return a pointer to its descriptor buffer, or
    /// null when the stream is exhausted. Drops the previous batch first.
    fn next_batch(&mut self) -> *const FfiArray {
        // Release the previous batch (its Arrow buffers are freed here).
        self.current = None;
        let Some(rb) = self.iter.next() else {
            return ptr::null();
        };
        // Build the descriptors while borrowing `rb`, copy them into the reused
        // buffer (they're `Copy`, holding only pointers into `rb`'s buffers), then
        // release the borrow and keep `rb` alive so those pointers stay valid.
        let prepared = prepare_record_batch(&rb).expect("prepare input batch");
        self.descs.clear();
        self.descs.extend_from_slice(prepared.arrays());
        drop(prepared);
        self.current = Some(rb);
        self.descs.as_ptr()
    }
}

/// The kernel's input handle: one [`ScanStream`] per table, indexed by table id.
pub struct Inputs {
    streams: Vec<ScanStream>,
}

impl Inputs {
    /// Build inputs from one stream per table (in table-id order).
    pub fn new(streams: Vec<ScanStream>) -> Self {
        Inputs { streams }
    }

    /// A single table fed by exactly one in-memory batch (the current
    /// `exec_jit(sql, table, &rb)` shape — a one-element stream).
    pub fn single(rb: RecordBatch) -> Self {
        Inputs::new(vec![ScanStream::new(Box::new(std::iter::once(rb)))])
    }
}

/// Pull the next batch of table `table`, returning its descriptor pointer (null =
/// exhausted). `table` is a stage-0 constant in the kernel, so dispatch is free.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn scan_next(inputs: &mut Inputs, table: u64) -> *const FfiArray {
    inputs.streams[table as usize].next_batch()
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
        let mut s = ScanStream::new(Box::new(
            vec![batch(vec![1, 2, 3]), batch(vec![4, 5])].into_iter(),
        ));

        let p1 = s.next_batch();
        assert!(!p1.is_null());
        assert_eq!(unsafe { &*p1 }.len(), 3); // batch 1: 3 rows
        assert!(s.current.is_some()); // one batch resident

        let p2 = s.next_batch();
        assert!(!p2.is_null());
        assert_eq!(unsafe { &*p2 }.len(), 2); // batch 2: 2 rows
        assert!(s.current.is_some());

        assert!(s.next_batch().is_null()); // exhausted
        assert!(s.current.is_none()); // last batch dropped
    }
}
