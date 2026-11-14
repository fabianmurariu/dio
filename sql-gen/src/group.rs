//! GROUP BY support — a Rust-hosted hash table mapping group keys to dense group
//! indices, driven from the JIT kernel.
//!
//! The kernel keeps the *hot* aggregation loop: per input row it computes the group
//! key, calls `group_upsert*` to get the group's record, and JIT-folds each accumulator
//! into it. This state owns the key → index map and the records.
//!
//! Every key is fixed-width: an integer or float key is one `u64`, and any other key
//! (strings, several columns) is a **packed** byte key the kernel assembles, a string
//! in it being its view. Strings are interned ([`group_intern`]): each distinct one is
//! stored once, so equal strings have equal views and a packed key's bytes stand for
//! its content.
//!
//! Hosting the state in Rust is deliberate: it's what lets these kernels later become
//! partial aggregates whose `[keys | accumulators]` state is merged across parallel
//! workers.

use ahash::RandomState;
use datafusion_common::{DataFusionError, Result};
use hashbrown::HashTable;
use rust_lms::byte_view::{ByteBuffer, MAX_INLINE_LEN, make_view};
use rust_lms::prelude::*;

use crate::status::RuntimeStatus;

/// All host-side state for one GROUP BY, allocated before the kernel is compiled and
/// kept alive across the run (its pointer is baked into the kernel as a constant).
///
/// The kernel folds each input row into a group's packed record: `group_upsert*` the
/// key → the record pointer, then read-modify-write its fields. Records are `u64`-
/// backed for 8-byte alignment (every field is an `i64`/`f64` cell) and the buffer
/// **grows with the group count** (`O(groups)`, not `O(rows)`) — a new group appends
/// one identity record with its key at the head. This is also the mergeable partial
/// state for parallel aggregation.
pub struct GroupState {
    /// The key → `gidx` map, specialised to the key type (see [`KeyKind`]).
    table: KeyTable,
    /// Packed records, `[key | per-agg value (+ count)]` each, laid out back-to-back
    /// (`num_records = records.len() / template.len()`). Grows as groups are minted.
    records: Vec<u64>,
    /// The identity record (per-field start values — see `codegen::group_template`),
    /// copied in whenever a new group is appended.
    template: Vec<u64>,
    /// The dense index of the *null-key* group of a single integer or float key (SQL:
    /// all NULL keys form one group), minted lazily on the first null-key row. Kept
    /// **out of the hash table**, but it occupies a normal record slot, so the single
    /// emit loop covers it. A packed key carries its nulls in its own bitmap instead.
    null_gidx: Option<u32>,
    /// The string keys, each once.
    strings: Interner,
    /// First-error sink. A fallible `group_upsert*` records its `DataFusionError`
    /// here and returns a null record pointer; the driver calls
    /// [`RuntimeStatus::check`] after the kernel returns. See [`crate::status`].
    status: RuntimeStatus,
}

/// Which key type a GROUP BY uses — picks the [`KeyTable`].
#[derive(Clone, Copy)]
pub enum KeyKind {
    /// A widened `Int32`/`Int64` column, keyed on its `u64` bits.
    Int,
    /// A `Float64` column, keyed on its `u64` *bits* (the `Int` table — the kernel
    /// canonicalizes and bitcasts the key; equal floats have equal bits).
    Float,
    /// A packed key of this many bytes: a string column, or several columns.
    Packed(usize),
}

/// The group table for the query's key type. Only the `group_upsert*` externs branch
/// on the variant.
enum KeyTable {
    Int(IntTable),
    Packed(PackedTable),
}

impl GroupState {
    /// A fresh state with no groups; `template` is one identity record's words and
    /// `key` selects the table's key type.
    pub fn new(template: Vec<u64>, key: KeyKind) -> Self {
        let table = match key {
            KeyKind::Int | KeyKind::Float => KeyTable::Int(IntTable::default()),
            KeyKind::Packed(len) => {
                assert!(
                    len <= template.len() * 8,
                    "a packed key fits at its record's head"
                );
                KeyTable::Packed(PackedTable::new(len))
            }
        };
        GroupState {
            table,
            records: Vec::new(),
            template,
            null_gidx: None,
            strings: Interner::default(),
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
/// `u64` cells. Every host write into a record (only the group key) goes through a
/// named method here. The raw byte pointer escapes exactly once — via
/// [`Record::as_ptr`], handed to the JIT kernel at the extern boundary, which then
/// owns all further (typed) field access through `DynamicRecord`.
struct Record<'a> {
    /// The record's cells; the leading ones hold the key, matching
    /// `codegen::group_record`.
    cells: &'a mut [u64],
}

impl Record<'_> {
    /// Store an integer key in the record's leading cell (field 0).
    fn set_int_key(&mut self, key: u64) {
        self.cells[0] = key;
    }

    /// Store a packed key's bytes at the record's head.
    fn set_packed_key(&mut self, key: &[u8]) {
        record_bytes_mut(self.cells)[..key.len()].copy_from_slice(key);
    }

    /// The record's byte pointer — handed to the kernel, which owns all further
    /// (typed) access. The single point where a raw pointer leaves this module.
    fn as_ptr(&mut self) -> *mut u8 {
        self.cells.as_mut_ptr() as *mut u8
    }
}

/// `cells` as bytes.
fn record_bytes(cells: &[u64]) -> &[u8] {
    // SAFETY: any `u64` is 8 initialized bytes, and `u8` has no alignment.
    unsafe { std::slice::from_raw_parts(cells.as_ptr().cast::<u8>(), cells.len() * 8) }
}

fn record_bytes_mut(cells: &mut [u64]) -> &mut [u8] {
    // SAFETY: as `record_bytes`; any bytes are a valid `u64`.
    unsafe { std::slice::from_raw_parts_mut(cells.as_mut_ptr().cast::<u8>(), cells.len() * 8) }
}

/// Maps an integer key (or a float's bits) to its group's dense index. Fixed seed →
/// deterministic group order for a given input (tests, and reproducible output);
/// grouping correctness doesn't depend on the seed.
struct IntTable {
    table: HashTable<(u64, u32)>,
    state: RandomState,
}

impl Default for IntTable {
    fn default() -> Self {
        Self {
            table: HashTable::new(),
            state: RandomState::with_seed(0),
        }
    }
}

impl IntTable {
    /// `key`'s group, or `next_gidx` for a new one, and whether it is new.
    fn find_or_insert(&mut self, key: u64, next_gidx: u32) -> (u32, bool) {
        let state = &self.state;
        let hash = state.hash_one(key);
        if let Some(&(_, gidx)) = self.table.find(hash, |&(k, _)| k == key) {
            return (gidx, false);
        }
        self.table
            .insert_unique(hash, (key, next_gidx), |&(k, _)| state.hash_one(k));
        (next_gidx, true)
    }
}

/// Maps a packed key to its group's dense index. A key is stored once, at the head of
/// its group's record, so the table holds only indices and compares a probe against
/// the records.
struct PackedTable {
    table: HashTable<u32>,
    state: RandomState,
    /// Every key's length in bytes.
    len: usize,
}

impl PackedTable {
    fn new(len: usize) -> Self {
        Self {
            table: HashTable::new(),
            state: RandomState::with_seed(0),
            len,
        }
    }

    /// Group `gidx`'s key, the head of its record in `records`.
    fn key<'r>(&self, records: &'r [u64], stride_words: usize, gidx: u32) -> &'r [u8] {
        let start = gidx as usize * stride_words;
        &record_bytes(&records[start..start + stride_words])[..self.len]
    }

    /// `probe`'s group, if it has one.
    fn find(&self, records: &[u64], stride_words: usize, probe: &[u8]) -> (u64, Option<u32>) {
        let hash = self.state.hash_one(probe);
        let found = self
            .table
            .find(hash, |&g| self.key(records, stride_words, g) == probe)
            .copied();
        (hash, found)
    }

    /// Add group `gidx`, whose record already holds its key, under `hash`.
    fn insert(&mut self, records: &[u64], stride_words: usize, hash: u64, gidx: u32) {
        let Self { table, state, len } = self;
        table.insert_unique(hash, gidx, |&g| {
            let start = g as usize * stride_words;
            state.hash_one(&record_bytes(&records[start..start + stride_words])[..*len])
        });
    }
}

/// Each distinct string once, so equal strings have equal views: a short one's view is
/// its bytes, and a long one's points into the pool, where it is stored the first time
/// it is seen.
struct Interner {
    pool: BytesPool,
    /// The long strings' views, keyed by their bytes.
    views: HashTable<u128>,
    state: RandomState,
}

impl Default for Interner {
    fn default() -> Self {
        Self {
            pool: BytesPool::new(),
            views: HashTable::new(),
            state: RandomState::with_seed(0),
        }
    }
}

impl Interner {
    /// The view of `bytes`, the same for every equal string.
    fn intern(&mut self, bytes: &[u8]) -> u128 {
        if bytes.len() <= MAX_INLINE_LEN {
            return make_view(bytes, 0, 0);
        }
        let Self { pool, views, state } = self;
        // SAFETY (both reads): every view in `views` is one `pool` returned, read while
        // nothing is appended.
        let hash = state.hash_one(bytes);
        let found = views.find(hash, |&v| {
            let value = unsafe { pool.value(v) };
            (unsafe { value.bytes() }) == bytes
        });
        if let Some(&view) = found {
            return view;
        }
        let view = pool.append(bytes);
        views.insert_unique(hash, view, |&v| {
            let value = unsafe { pool.value(v) };
            state.hash_one(unsafe { value.bytes() })
        });
        view
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
/// records-buffer growth are host code; a new group's record gets the key in its
/// leading field (offset 0), so the kernel only folds aggregates. The returned pointer
/// is valid until the next `group_upsert*` (which may grow and move the buffer) — the
/// fold uses it immediately, so this holds. Returns null (with the error recorded) if
/// the group count would exceed `u32` (see `upsert_guard`).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_upsert(state: &mut GroupState, key: u64) -> *mut u8 {
    upsert_guard(state, |state| {
        let next = group_next_index(state)?;
        let KeyTable::Int(table) = &mut state.table else {
            unreachable!("group_upsert on a packed-key table")
        };
        let (gidx, new) = table.find_or_insert(key, next);
        let mut rec = state.ensure_record(gidx as usize);
        if new {
            rec.set_int_key(key);
        }
        Ok(rec.as_ptr())
    })
}

/// Find-or-insert the **null-key** group of a single integer or float key and return
/// its record pointer. The null key bypasses the hash table (see
/// `GroupState::null_gidx`); it just gets a record slot like any other group. No key is
/// written — the emit path reads the record's key-valid cell to know it is null.
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

/// Find-or-insert a group by its packed key and return its record pointer. A new
/// group's record gets the key's bytes at its head, where the table compares later
/// probes and the emit loop reads the key columns back.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_upsert_packed(state: &mut GroupState, key: FatSlice<u8>) -> *mut u8 {
    upsert_guard(state, |state| {
        // SAFETY: the kernel's packed key, alive for the call.
        let key = unsafe { key.as_slice() };
        let next = group_next_index(state)?;
        let w = state.stride_words();
        let KeyTable::Packed(table) = &state.table else {
            unreachable!("group_upsert_packed on an int-keyed table")
        };
        debug_assert_eq!(
            key.len(),
            table.len,
            "every packed key has the layout's length"
        );
        let (hash, found) = table.find(&state.records, w, key);
        if let Some(gidx) = found {
            return Ok(state.ensure_record(gidx as usize).as_ptr());
        }
        state.ensure_record(next as usize).set_packed_key(key);
        let KeyTable::Packed(table) = &mut state.table else {
            unreachable!()
        };
        table.insert(&state.records, w, hash, next);
        Ok(state.ensure_record(next as usize).as_ptr())
    })
}

/// Intern a string key: write the view every equal string shares to `out`. The kernel
/// calls this for long strings only; a short one's view is already its content.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_intern(state: &mut GroupState, bytes: FatSlice<u8>, out: *mut u128) {
    // SAFETY: the string's bytes, alive for the call; `out` is the kernel's slot for
    // the view.
    unsafe { *out = state.strings.intern(bytes.as_slice()) };
}

/// The table the interned strings' views index, fetched once for the emit loop: nothing
/// is interned after the fold, so it stays where it is.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_strings(state: &GroupState) -> *const ByteBuffer {
    state.strings.pool.buffers().as_ptr()
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
