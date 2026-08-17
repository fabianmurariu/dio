//! Host side of the hash join: the materialized **build relation** (the left
//! input, kept as Arrow `RecordBatch`es) plus a hash index (key → row [`Locator`]s)
//! over its rows. Both the build index and the probe are JIT-ed — the build inserts
//! locators via [`join_insert`], the probe walks them and `gen_read`s the located
//! rows straight out of the retained batches. See `docs/joins.md`.

use std::collections::HashMap;
use std::ptr;

use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArray, prepare_record_batch};
use rust_lms::prelude::*;

/// Where a build row lives: its batch index in the relation and its row within it.
/// A typed 2-field record (not a bit-packed `u64`) — the probe reads `.rb_pos` /
/// `.row` as fields. For a single-batch (materialized) relation, `rb_pos` is always
/// 0; a bare-scan clone's relation spans batches.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct Locator {
    #[staged(u32)]
    pub rb_pos: u32,
    #[staged(u32)]
    pub row: u32,
}

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
    /// key → the [`Locator`]s of every build row with that key (a multimap).
    index: HashMap<u64, Vec<Locator>>,
    /// Returned for a key with no matches (a stable empty slice).
    empty: Vec<Locator>,
}

impl JoinState {
    /// A bare-`Scan` build: clone the input RBs into the relation (multi-batch, no
    /// column copy). The key index starts EMPTY — the JIT index kernel fills it via
    /// [`join_insert`] (no host per-value loop).
    pub fn from_clone(batches: Box<dyn Iterator<Item = RecordBatch>>) -> Self {
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

    /// A materialized build: an empty relation whose single batch the fused build
    /// kernel produces (and installs via [`push_batch`](Self::push_batch)) while it
    /// simultaneously fills the index.
    pub fn empty_relation() -> Self {
        JoinState {
            relation: InMemoryRelation::default(),
            index: HashMap::new(),
            empty: Vec::new(),
        }
    }

    /// Install the materialized build batch after the fused kernel finishes (its
    /// locators, inserted during the run, reference row indices into this batch).
    pub fn push_batch(&mut self, rb: RecordBatch) {
        self.relation.push(rb);
    }

    fn probe_run(&self, key: u64) -> &[Locator] {
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

/// Insert a build row's location under `key` — the proxy the JIT build calls per
/// non-null row (like GROUP BY's `group_upsert`). `key` is the row's join key as
/// `u64`; `(rb_pos, row)` locate the row in the relation.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn join_insert(js: &mut JoinState, key: u64, rb_pos: u32, row: u32) {
    js.index
        .entry(key)
        .or_default()
        .push(Locator { rb_pos, row });
}

/// Number of build rows matching `key` (0 = no match).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn join_probe_count(js: &JoinState, key: u64) -> u64 {
    js.probe_run(key).len() as u64
}

/// Base pointer to `key`'s [`Locator`] run (null if no match). Stable during the
/// probe (the build phase is finished); the kernel reads `base[0..count]`.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn join_probe_base(js: &JoinState, key: u64) -> *const Locator {
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
