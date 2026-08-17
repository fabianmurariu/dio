//! Host side of the hash join: the materialized **build relation** (the left
//! input, kept as Arrow `RecordBatch`es) plus a hash index over its rows. The
//! probe kernel looks a key up, walks the matching row locators, and `gen_read`s
//! the located build rows straight out of the retained batches. See `docs/joins.md`.
//!
//! Phase 1: inner join, single `Int` key, bare-`Scan` left build (host-side, no
//! JIT — we clone the input batches and index the key column here).

use std::collections::HashMap;
use std::ptr;

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
    /// Materialize the build relation from its batches (a bare `Scan` clones its
    /// input RBs; a filtered/projected build passes its one materialized RB). The
    /// key index starts EMPTY — it is populated by the JIT index kernel via
    /// [`join_insert`], not by a host per-value loop.
    pub fn new(batches: Box<dyn Iterator<Item = RecordBatch>>) -> Self {
        let mut relation = InMemoryRelation::default();
        for rb in batches {
            relation.push(rb);
        }
        JoinState {
            relation,
            index: HashMap::new(),
            empty: Vec::new(),
        }
    }

    fn probe_run(&self, key: u64) -> &[u64] {
        self.index
            .get(&key)
            .map_or(&self.empty[..], |v| v.as_slice())
    }
}

/// Number of batches in the build relation — the JIT index kernel's outer loop bound.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn join_rel_count(js: &JoinState) -> u64 {
    js.relation.num_batches() as u64
}

/// Insert a build row's `locator` under `key` (the proxy the JIT index kernel calls
/// per non-null build row, like GROUP BY's `group_upsert`). `key` is the row's
/// join key as `u64`; `locator = (batch_idx << 32) | row_idx`.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn join_insert(js: &mut JoinState, key: u64, locator: u64) {
    js.index.entry(key).or_default().push(locator);
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
