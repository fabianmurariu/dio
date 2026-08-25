//! GROUP BY support — a Rust-hosted hash table mapping group keys to dense group
//! indices, driven from the JIT kernel.
//!
//! The kernel keeps the *hot* aggregation loop: per input row it computes the
//! group key, calls [`group_find_or_insert`] to get a group index, and JIT-folds each
//! accumulator into an output array slot at that index. This struct owns only the
//! `key -> index` map; the accumulator storage and group-key column are ordinary
//! JIT-written output arrays (sized to the row count, so indices never overflow).
//!
//! Hosting the state in Rust is deliberate: it's the same pattern as the string
//! `BytesPool`, and it's what lets these kernels later become partial aggregates
//! whose `[keys | accumulators]` state is merged across parallel workers.

use ahash::RandomState;
use datafusion_common::{DataFusionError, Result};
use hashbrown::HashTable;
use rust_lms::pool::BytesPool;
use rust_lms::prelude::*;

use crate::status::RuntimeStatus;

/// All host-side state for one GROUP BY, allocated before the kernel is compiled
/// and kept alive across the run (its buffer pointers are baked into the kernel as
/// constants — the same "host outlives the run" contract as the string pool).
///
/// The kernel folds each input row into a group's packed record: `group_upsert` the
/// key → the record pointer, then read-modify-write its fields. Records are `u64`-
/// backed for 8-byte alignment (every field is an `i64`/`f64` cell) and the buffer
/// **grows with the group count** (`O(groups)`, not `O(rows)`) — a new group appends
/// one identity record. This is also the mergeable partial state for parallel
/// aggregation.
pub struct GroupState {
    /// The key → `gidx` map, specialised to the key type (see [`KeyKind`]).
    table: KeyTable,
    /// Packed records, `[key | per-agg value (+ count)]` each, laid out back-to-back
    /// (`num_records = records.len() / template.len()`). Grows as groups are minted.
    records: Vec<u64>,
    /// The identity record (per-field start values — see `codegen::group_template`),
    /// copied in whenever a new group is appended.
    template: Vec<u64>,
    /// The dense index of the *null-key* group (SQL: all NULL keys form one group),
    /// minted lazily on the first null-key row. Kept **out of the hash table** — the
    /// null key is never hashed/interned — but it still occupies a normal record slot,
    /// so the single emit loop covers it. `None` until (and unless) a null key appears.
    null_gidx: Option<u32>,
    /// Reused scratch for building a *variable-length* composite key (columns with a
    /// string): the kernel pushes each column's bytes (`group_key_*`), then
    /// `group_upsert_composite` interns the assembled bytes. Cleared per row.
    key_scratch: Vec<u8>,
    /// First-error sink. A fallible `group_upsert*` records its `DataFusionError`
    /// here and returns a null record pointer; the driver calls
    /// [`RuntimeStatus::check`] after the kernel returns. See [`crate::status`].
    status: RuntimeStatus,
}

/// Which key type a GROUP BY uses — picks the [`GroupTable`] instantiation.
#[derive(Clone, Copy)]
pub enum KeyKind {
    /// A widened `Int32`/`Int64` column, keyed on its `u64` bits.
    Int,
    /// A `Float64` column, keyed on its `u64` *bits* (reuses the `Int` table — the
    /// kernel bitcasts the key; equal floats have equal bits).
    Float,
    /// A `Utf8View` column, keyed on its content bytes (copied into the table's pool).
    Str,
}

/// The group table, monomorphised per key type but type-erased here so [`GroupState`]
/// (and the shared `group_len`/`group_records_base` externs) stay non-generic. Only
/// `group_upsert`/`group_upsert_str` branch on the variant.
enum KeyTable {
    Int(GroupTable<u64>),
    Str(GroupTable<StrKey>),
}

impl GroupState {
    /// A fresh state with no groups; `template` is one identity record's words and
    /// `key` selects the table's key type.
    pub fn new(template: Vec<u64>, key: KeyKind) -> Self {
        let table = match key {
            // `Float` keys on `u64` bits, so they share the `Int` table.
            KeyKind::Int | KeyKind::Float => KeyTable::Int(GroupTable::new()),
            KeyKind::Str => KeyTable::Str(GroupTable::new()),
        };
        GroupState {
            table,
            records: Vec::new(),
            template,
            null_gidx: None,
            key_scratch: Vec::new(),
            status: RuntimeStatus::default(),
        }
    }

    /// Take any error recorded by a fallible `group_upsert*` during the kernel run.
    /// Called by the driver (`run_operator`) after the kernel returns.
    pub fn check(&mut self) -> Result<()> {
        self.status.check()
    }

    fn stride_words(&self) -> usize {
        self.template.len()
    }

    fn num_records(&self) -> usize {
        match self.stride_words() {
            0 => 0,
            w => self.records.len() / w,
        }
    }

    /// Ensure a record exists at `gidx` (append the identity template if `gidx` is
    /// the next new group), then return a mutable view of it. Valid until the next
    /// append reallocates the buffer.
    fn ensure_record(&mut self, gidx: usize) -> Record<'_> {
        if gidx == self.num_records() {
            let template = &self.template;
            self.records.extend_from_slice(template);
        }
        let w = self.stride_words();
        let start = gidx * w;
        // Bounds-checked slice — no pointer arithmetic; `start + w` is in range because
        // `gidx < num_records` after the append above.
        Record {
            cells: &mut self.records[start..start + w],
        }
    }

    /// Base of the (now fully grown) records buffer, for the emit loop.
    fn records_base(&mut self) -> *mut u8 {
        self.records.as_mut_ptr() as *mut u8
    }
}

/// A mutable, host-side view of one group's packed record — its `stride_words`
/// `u64` cells. Every host write into a record (only the group key, today) goes
/// through a named method here, so the `group_upsert*` externs carry no ad-hoc
/// `unsafe` or pointer math. The raw byte pointer escapes exactly once — via
/// [`Record::as_ptr`], handed to the JIT kernel at the extern boundary, which then
/// owns all further (typed) field access through `DynamicRecord`.
struct Record<'a> {
    /// The record's cells; cell 0 (and, for a string key, cell 1) hold the key —
    /// the record's leading field(s), matching `codegen::group_record`.
    cells: &'a mut [u64],
}

impl Record<'_> {
    /// Store an integer key in the record's leading cell (field 0).
    fn set_int_key(&mut self, key: u64) {
        self.cells[0] = key;
    }

    /// Store a string key's pooled `(ptr, len)` in the two leading cells (fields 0/1).
    /// The pointer is stored as its address; the kernel reads it back as an `SPtr<u8>`.
    fn set_str_key(&mut self, ptr: *const u8, len: usize) {
        self.cells[0] = ptr as u64;
        self.cells[1] = len as u64;
    }

    /// The record's byte pointer — handed to the kernel, which owns all further
    /// (typed) access. The single point where a raw pointer leaves this module.
    fn as_ptr(&mut self) -> *mut u8 {
        self.cells.as_mut_ptr() as *mut u8
    }
}

/// A group key we can hash, compare, and **persist** — the specialization seam (the
/// "Tidy Tuples" trick of tailoring hash/equality to the key). `store` copies any
/// variable-length payload (a string's bytes) into the table's pool so the stored
/// key outlives the input batch (needed once inputs stream); it is the identity for
/// scalar keys. Today `u64` and [`StrKey`]; composite keys add impls later.
pub trait GroupKey: Copy {
    /// Hash this (probe or stored) key's content with the table's `state`.
    fn hash_one(&self, state: &RandomState) -> u64;
    /// Content equality (for `StrKey`, a byte compare — *not* pointer equality).
    fn matches(&self, other: &Self) -> bool;
    /// Produce the **stored** form of this probe key, copying any variable data into
    /// `pool` so it stays valid for the table's lifetime. Identity for scalar keys.
    fn store(&self, pool: &mut BytesPool) -> Self;
}

impl GroupKey for u64 {
    fn hash_one(&self, state: &RandomState) -> u64 {
        state.hash_one(*self)
    }
    fn matches(&self, other: &Self) -> bool {
        self == other
    }
    fn store(&self, _pool: &mut BytesPool) -> Self {
        *self
    }
}

/// A `Utf8View` group key: a `(ptr, len)` byte reference. A **probe** points into the
/// live input array; the **stored** form ([`GroupKey::store`]) points into the table's
/// pool. Hash/equality are on content, so equal strings group even when their views
/// (buffer/offset) differ.
#[derive(Clone, Copy)]
pub struct StrKey {
    ptr: *const u8,
    len: usize,
}

impl StrKey {
    /// # Safety
    /// `ptr`/`len` name a valid byte range for the duration of the call.
    fn bytes(&self) -> &[u8] {
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl GroupKey for StrKey {
    fn hash_one(&self, state: &RandomState) -> u64 {
        state.hash_one(self.bytes())
    }
    fn matches(&self, other: &Self) -> bool {
        self.bytes() == other.bytes()
    }
    fn store(&self, pool: &mut BytesPool) -> Self {
        // Copy the key bytes into the pool — the stored key must outlive the input.
        StrKey {
            ptr: pool.append(self.bytes()),
            len: self.len,
        }
    }
}

/// One occupied slot: the group's key (stored inline in the table — decoupled from
/// the records buffer) and its dense index.
struct Entry<K> {
    key: K,
    gidx: u32,
}

/// Maps a group key to its dense index (`0..num_groups`), assigning the next index
/// the first time a key is seen. Built on [`hashbrown::HashTable`] — the *raw* Swiss
/// table where we supply hash + equality per op ([`GroupKey`]), so the key type is
/// open (composite/string keys are new [`GroupKey`] impls, not a rewrite). The value
/// is just the dense index; aggregates live in the separate packed records buffer.
/// The `pool` holds copied variable-length key bytes (empty/unused for scalar keys).
pub struct GroupTable<K> {
    table: HashTable<Entry<K>>,
    state: RandomState,
    pool: BytesPool,
}

impl<K: GroupKey> GroupTable<K> {
    pub fn new() -> Self {
        GroupTable {
            table: HashTable::new(),
            // Fixed seed → deterministic group order for a given input (tests, and
            // reproducible output); grouping correctness doesn't depend on the seed.
            state: RandomState::with_seed(0),
            pool: BytesPool::new(),
        }
    }

    /// Number of distinct groups seen so far.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Find `probe` (returning its `gidx` + stored key), or insert it at `next_gidx`.
    /// The caller supplies `next_gidx` (the records-buffer count) rather than the
    /// table's own `len`, so the gidx space stays dense even when a *null* group —
    /// which bypasses the table — also consumes a record slot.
    pub fn find_or_insert(&mut self, probe: K, next_gidx: u32) -> (u32, K) {
        let hash = probe.hash_one(&self.state);
        if let Some(entry) = self.table.find(hash, |e| e.key.matches(&probe)) {
            return (entry.gidx, entry.key);
        }
        let stored = probe.store(&mut self.pool); // copy variable data (not borrowing `table`)
        let state = &self.state;
        self.table.insert_unique(
            hash,
            Entry {
                key: stored,
                gidx: next_gidx,
            },
            |e| e.key.hash_one(state),
        );
        (next_gidx, stored)
    }
}

impl<K: GroupKey> Default for GroupTable<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Shared entry guard for the fallible `group_upsert*` record-pointer callbacks.
///
/// If the run is already poisoned it returns a null pointer without touching the
/// table, so a failed kernel stops minting groups. Otherwise it runs the checked
/// implementation: on `Ok` it returns the record pointer; on `Err` it records the
/// detailed error in the query's [`RuntimeStatus`] and returns null — the ABI-safe
/// sentinel the generated fold loop branches on. `null` is unambiguous here because
/// a real record pointer is never null.
fn upsert_guard(
    state: &mut GroupState,
    f: impl FnOnce(&mut GroupState) -> Result<*mut u8>,
) -> *mut u8 {
    if state.status.is_failed() {
        return std::ptr::null_mut();
    }
    match f(&mut *state) {
        Ok(ptr) => ptr,
        Err(error) => {
            state.status.record_error(error);
            std::ptr::null_mut()
        }
    }
}

// Test-only override for the `u32` group-count limit (defaults to `u32::MAX`), so a
// test can drive the overflow/error path without minting four billion groups. Set
// per-thread; production builds have no such thread-local. See `group_next_index`.
#[cfg(test)]
thread_local! {
    static GROUP_LIMIT: std::cell::Cell<u64> = const { std::cell::Cell::new(u32::MAX as u64) };
}

/// The dense index the next new group would occupy, checked against `u32` — group
/// indices are stored as `u32` in records and the hash table (PR-17). A stream with
/// more than `u32::MAX` groups is rejected rather than silently wrapping.
fn group_next_index(state: &GroupState) -> Result<u32> {
    let n = state.num_records();
    // The real bound is `u32::MAX`; tests lower it to exercise the same error path.
    #[cfg(test)]
    if n as u64 >= GROUP_LIMIT.with(std::cell::Cell::get) {
        return Err(DataFusionError::Execution(
            "GROUP BY group count exceeds u32::MAX".into(),
        ));
    }
    u32::try_from(n)
        .map_err(|_| DataFusionError::Execution("GROUP BY group count exceeds u32::MAX".into()))
}

/// Find-or-insert an `Int` group by its `u64` key and return its **record pointer** —
/// the per-row proxy call from the kernel (Umbra's `insert`). The table mechanics and
/// records-buffer growth are host code; the extern also writes the key into the
/// record's leading field (offset 0), so the kernel only folds aggregates. The
/// returned pointer is valid until the next `group_upsert*` (which may grow and move
/// the buffer) — the fold uses it immediately, so this holds. Returns null (with the
/// error recorded) if the group count would exceed `u32` (see [`upsert_guard`]).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_upsert(state: &mut GroupState, key: u64) -> *mut u8 {
    upsert_guard(state, |state| {
        let next = group_next_index(state)?;
        let (gidx, stored) = match &mut state.table {
            KeyTable::Int(t) => t.find_or_insert(key, next),
            KeyTable::Str(_) => unreachable!("group_upsert on a string-keyed table"),
        };
        let mut rec = state.ensure_record(gidx as usize);
        rec.set_int_key(stored);
        Ok(rec.as_ptr())
    })
}

/// Find-or-insert the **null-key** group and return its record pointer. The null key
/// bypasses the hash table (see [`GroupState::null_gidx`]); it just gets a record slot
/// like any other group. No key is written — the emit path reads the record's
/// key-valid cell to know it is null.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_upsert_null(state: &mut GroupState) -> *mut u8 {
    upsert_guard(state, |state| {
        let gidx = match state.null_gidx {
            Some(g) => g,
            None => {
                let g = group_next_index(state)?;
                state.null_gidx = Some(g);
                g
            }
        };
        let mut rec = state.ensure_record(gidx as usize);
        Ok(rec.as_ptr())
    })
}

/// Find-or-insert a `Str` group by its content bytes and return its record pointer.
/// The key bytes are copied into the table's pool on a new group; the record's
/// leading `(ptr, len)` fields (offsets 0 and 8) get the **pooled** reference, so the
/// emit loop reads stable bytes.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_upsert_str(state: &mut GroupState, ptr: *const u8, len: u64) -> *mut u8 {
    upsert_guard(state, |state| {
        let probe = StrKey {
            ptr,
            len: len as usize,
        };
        let next = group_next_index(state)?;
        let (gidx, stored) = match &mut state.table {
            KeyTable::Str(t) => t.find_or_insert(probe, next),
            KeyTable::Int(_) => unreachable!("group_upsert_str on an int-keyed table"),
        };
        let mut rec = state.ensure_record(gidx as usize);
        rec.set_str_key(stored.ptr, stored.len);
        Ok(rec.as_ptr())
    })
}

// --- Variable-length composite key builder: the kernel pushes each key column's
// bytes into `key_scratch`, then `group_upsert_composite` interns the assembled bytes
// (a flat byte key, so the `Str` table's content hash/eq apply). ---

/// Start a new composite key (clear the scratch).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_key_reset(state: &mut GroupState) {
    state.key_scratch.clear();
}

/// Append 8 bytes (a fixed-width column's canonicalized bits, or the null bitmap).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_key_push_u64(state: &mut GroupState, v: u64) {
    state.key_scratch.extend_from_slice(&v.to_le_bytes());
}

/// Append a string column: an 8-byte length prefix then the content bytes (so
/// `"ab"+"c"` and `"a"+"bc"` can't collide). A null column pushes length 0 (the null
/// bitmap tells it apart from an empty string).
/// # Safety `(ptr, len)` is a valid byte range produced by staged code for this call.
#[extern_fn]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn group_key_push_bytes(state: &mut GroupState, ptr: *const u8, len: u64) {
    state.key_scratch.extend_from_slice(&len.to_le_bytes());
    if len != 0 {
        let content = unsafe { std::slice::from_raw_parts(ptr, len as usize) };
        state.key_scratch.extend_from_slice(content);
    }
}

/// Intern the assembled composite key (the current `key_scratch`) and return its
/// record pointer — the composite counterpart of `group_upsert_str`. The scratch is a
/// flat byte key, so the `Str` table's content hash/eq apply; it's copied into the
/// pool on a miss.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_upsert_composite(state: &mut GroupState) -> *mut u8 {
    upsert_guard(state, |state| {
        let probe = StrKey {
            ptr: state.key_scratch.as_ptr(),
            len: state.key_scratch.len(),
        };
        let next = group_next_index(state)?;
        let (gidx, stored) = match &mut state.table {
            KeyTable::Str(t) => t.find_or_insert(probe, next),
            KeyTable::Int(_) => unreachable!("group_upsert_composite on an int-keyed table"),
        };
        let mut rec = state.ensure_record(gidx as usize);
        rec.set_str_key(stored.ptr, stored.len);
        Ok(rec.as_ptr())
    })
}

/// Base of the fully-grown records buffer, fetched once for the emit loop.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_records_base(state: &mut GroupState) -> *mut u8 {
    state.records_base()
}

/// The final group count, called once after the fold loop to size the output —
/// the number of materialised records (hash groups plus the null group, if any).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_len(state: &GroupState) -> u64 {
    state.num_records() as u64
}

#[cfg(test)]
mod pr16_tests {
    //! End-to-end coverage for the `group_upsert*` error path (PR-16 + PR-17): a
    //! fallible upsert must record its `DataFusionError`, return the null sentinel,
    //! and have generated code stop — so the query returns `Err` instead of
    //! aborting through the `extern "C"` boundary, dereferencing null, or returning
    //! a truncated/partial result. The real trigger (>`u32::MAX` groups) is induced
    //! via the test-only `GROUP_LIMIT` seam.

    use std::sync::Arc;

    use arrow::array::Int64Array;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    /// Lowers `GROUP_LIMIT` for the current thread and restores `u32::MAX` on drop,
    /// so a panicking assertion cannot leak the low limit into another test.
    struct LimitGuard;
    impl Drop for LimitGuard {
        fn drop(&mut self) {
            super::GROUP_LIMIT.with(|c| c.set(u32::MAX as u64));
        }
    }
    fn with_group_limit(n: u64) -> LimitGuard {
        super::GROUP_LIMIT.with(|c| c.set(n));
        LimitGuard
    }

    fn keyed_batch(keys: Vec<i64>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("key", DataType::Int64, false),
            Field::new("value", DataType::Int64, false),
        ]));
        let values: Vec<i64> = keys.iter().map(|k| k * 10).collect();
        RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(Int64Array::from(values)),
            ],
        )
        .unwrap()
    }

    #[test]
    fn group_overflow_surfaces_as_err_not_abort() {
        // Limit 2 → the 3rd distinct group's upsert fails, exactly as a real
        // >u32::MAX overflow would.
        let _guard = with_group_limit(2);
        let rb = keyed_batch(vec![1, 2, 3, 4]);
        let result = crate::exec_jit("SELECT key, count(*) FROM t GROUP BY key", "t", &rb);
        // The process is still alive (no abort/UB) and we get a structured error.
        let err = result.expect_err("group overflow must surface as Err, not abort/null-deref");
        assert!(
            err.to_string().contains("exceeds"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn group_under_limit_still_succeeds() {
        // Same machinery, comfortably under the limit: the happy path is unaffected.
        let _guard = with_group_limit(1000);
        let rb = keyed_batch(vec![1, 1, 2, 2, 3]);
        let out = crate::exec_jit("SELECT key, count(*) FROM t GROUP BY key", "t", &rb)
            .expect("under the limit, the query must succeed");
        assert_eq!(out.num_rows(), 3, "three distinct groups");
    }
}
