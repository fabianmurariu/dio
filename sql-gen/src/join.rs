//! Host side of the hash join: the materialized **build relation** (the left
//! input, kept as Arrow `RecordBatch`es) plus a hash index over its rows. The
//! probe kernel looks a key up, walks the matching row locators, and `gen_read`s
//! the located build rows straight out of the retained batches. See `docs/joins.md`.
//!
//! Phase 1: inner join, single `Int` key, bare-`Scan` left build (host-side, no
//! JIT — we clone the input batches and index the key column here).

use std::collections::HashMap;
use std::ptr;

use arrow::array::{Array, Int32Array, Int64Array};
use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArray, prepare_record_batch};
use rust_lms::prelude::*;

/// A materialized relation the probe reads located rows out of. Behind a trait so a
/// future spill-to-disk / mmap-backed store can slot in without touching codegen.
pub trait BuildRelation {
    fn num_batches(&self) -> usize;
    /// FFI descriptors for batch `i`, stable for the whole probe (the batch is
    /// retained). The probe rebuilds a `&[FfiArray]` from this pointer and reuses
    /// `gen_read` to lift the located row's columns.
    fn batch_descriptors(&self, i: usize) -> *const FfiArray;
}

/// In-memory build relation: the retained batches plus their FFI descriptors.
/// (Phase 1 fills this by cloning a bare scan's input batches.)
#[derive(Default)]
pub struct InMemoryRelation {
    batches: Vec<RecordBatch>,
    descs: Vec<Vec<FfiArray>>,
}

impl InMemoryRelation {
    fn push(&mut self, rb: RecordBatch) {
        // Descriptors hold raw pointers into `rb`'s Arrow buffers; `rb` is retained
        // in `batches`, so they stay valid for the whole probe.
        let prepared = prepare_record_batch(&rb).expect("prepare build batch");
        let d = prepared.arrays().to_vec();
        drop(prepared);
        self.batches.push(rb);
        self.descs.push(d);
    }
}

impl BuildRelation for InMemoryRelation {
    fn num_batches(&self) -> usize {
        self.batches.len()
    }
    fn batch_descriptors(&self, i: usize) -> *const FfiArray {
        self.descs[i].as_ptr()
    }
}

/// The whole host-side join state: the build relation + a hash index mapping a key
/// to every build-relation row with that key. One baked pointer reaches it from the
/// kernel (like `GroupState`).
pub struct JoinState {
    relation: InMemoryRelation,
    /// key → packed `(batch_idx: u32, row_idx: u32)` locators (a multimap).
    index: HashMap<u64, Vec<u64>>,
    /// Returned for a key with no matches (a stable empty slice).
    empty: Vec<u64>,
}

impl JoinState {
    /// Build from a bare-`Scan` left side: clone each input batch into the relation
    /// and index its `Int` key column (Phase 1). Null keys are not indexed (a `NULL`
    /// join key never matches).
    pub fn build_int(batches: Box<dyn Iterator<Item = RecordBatch>>, key_col: usize) -> Self {
        let mut relation = InMemoryRelation::default();
        let mut index: HashMap<u64, Vec<u64>> = HashMap::new();
        for rb in batches {
            let batch_idx = relation.num_batches() as u64;
            let col = rb.column(key_col);
            for row in 0..rb.num_rows() {
                if let Some(k) = int_at(col, row) {
                    let loc = (batch_idx << 32) | (row as u64);
                    index.entry(k as u64).or_default().push(loc);
                }
            }
            relation.push(rb);
        }
        JoinState {
            relation,
            index,
            empty: Vec::new(),
        }
    }

    fn probe_run(&self, key: u64) -> &[u64] {
        self.index
            .get(&key)
            .map_or(&self.empty[..], |v| v.as_slice())
    }
}

/// Read an `Int32`/`Int64` array element as `i64`, or `None` if the row is null.
fn int_at(col: &dyn Array, row: usize) -> Option<i64> {
    if col.is_null(row) {
        return None;
    }
    if let Some(a) = col.as_any().downcast_ref::<Int64Array>() {
        Some(a.value(row))
    } else if let Some(a) = col.as_any().downcast_ref::<Int32Array>() {
        Some(a.value(row) as i64)
    } else {
        panic!("join key column is not Int32/Int64");
    }
}

/// Number of build rows matching `key` (0 = no match).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn join_probe_count(js: &JoinState, key: u64) -> u64 {
    js.probe_run(key).len() as u64
}

/// Base pointer to `key`'s locator run (null if no match). Stable during the probe
/// (the build phase is finished); the kernel indexes `base[0..count]`.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn join_probe_base(js: &JoinState, key: u64) -> *const u64 {
    let run = js.probe_run(key);
    if run.is_empty() {
        ptr::null()
    } else {
        run.as_ptr()
    }
}

/// Descriptors for build-relation batch `batch_idx`, for the probe's `gen_read`.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn join_left_batch(js: &JoinState, batch_idx: u64) -> *const FfiArray {
    js.relation.batch_descriptors(batch_idx as usize)
}
