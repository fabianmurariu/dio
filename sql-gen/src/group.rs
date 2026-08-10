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
use hashbrown::HashTable;
use rust_lms::prelude::*;

/// All host-side state for one GROUP BY, allocated before the kernel is compiled
/// and kept alive across the run (its buffer pointers are baked into the kernel as
/// constants — the same "host outlives the run" contract as the string pool).
///
/// The kernel folds each input row into a group slot: `intern` the key → `gidx`,
/// then read-modify-write `buffers[slot][gidx]`. Buffers are `i64`-typed 8-byte
/// slots (an `f64` accumulator — `avg`'s running sum — reuses the same bytes via a
/// bit-reinterpreting pointer). Sized to the input row count, so `gidx` (dense,
/// `< num_groups ≤ rows`) never overflows. This is also the mergeable partial state
/// for parallel aggregation.
pub struct GroupState {
    pub table: GroupTable<u64>,
    /// One packed record per group (`capacity` records of `template.len()` words),
    /// row-wise: `[key | per-agg value (+ count)]`. `u64`-backed for 8-byte
    /// alignment (every field is an `i64`/`f64` cell). Pre-filled with the identity
    /// record so a group's first fold sees the right start value.
    records: Vec<u64>,
}

impl GroupState {
    /// Allocate `capacity` records, each initialised to the identity `template`
    /// (the per-field start values — see `codegen::group_template`).
    pub fn new(template: &[u64], capacity: usize) -> Self {
        let mut records = Vec::with_capacity(capacity * template.len());
        for _ in 0..capacity {
            records.extend_from_slice(template);
        }
        Self {
            table: GroupTable::new(),
            records,
        }
    }

    /// The group hash table (baked; handed to the `intern`/`len` externs).
    pub fn table_ptr(&mut self) -> *mut GroupTable<u64> {
        &mut self.table
    }

    /// Base of the packed-records buffer (baked; the kernel indexes by `gidx`
    /// through a `RecordLayout`).
    pub fn records_ptr(&mut self) -> *mut u8 {
        self.records.as_mut_ptr() as *mut u8
    }
}

/// A group key we can hash and compare — the specialization seam (the "Tidy
/// Tuples" trick of tailoring hash/equality to the key). Today only `u64` (an
/// `Int32`/`Int64` column widened); string and composite keys add impls later
/// (see [`GroupTable`]'s docs and `docs/path_to_umbra_group_by.md`).
pub trait GroupKey: Copy + PartialEq {
    /// Hash `key` with the table's `state` (host-side, per the paper's proxy model).
    fn hash_one(key: &Self, state: &RandomState) -> u64;
}

impl GroupKey for u64 {
    fn hash_one(key: &Self, state: &RandomState) -> u64 {
        state.hash_one(*key)
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
pub struct GroupTable<K> {
    table: HashTable<Entry<K>>,
    state: RandomState,
}

impl<K: GroupKey> GroupTable<K> {
    pub fn new() -> Self {
        GroupTable {
            table: HashTable::new(),
            // Fixed seed → deterministic group order for a given input (tests, and
            // reproducible output); grouping correctness doesn't depend on the seed.
            state: RandomState::with_seed(0),
        }
    }

    /// Number of distinct groups seen so far.
    pub fn len(&self) -> usize {
        self.table.len()
    }

    pub fn is_empty(&self) -> bool {
        self.table.is_empty()
    }

    /// Find-or-insert `key`, returning its group index — the existing one, or the
    /// next free index (`= current group count`) if `key` is new. Dense and
    /// monotonic, so it doubles as the row cursor into the records buffer.
    pub fn intern(&mut self, key: K) -> u32 {
        let hash = K::hash_one(&key, &self.state);
        if let Some(entry) = self.table.find(hash, |e| e.key == key) {
            return entry.gidx;
        }
        let gidx = self.table.len() as u32;
        let state = &self.state;
        self.table
            .insert_unique(hash, Entry { key, gidx }, |e| K::hash_one(&e.key, state));
        gidx
    }
}

impl<K: GroupKey> Default for GroupTable<K> {
    fn default() -> Self {
        Self::new()
    }
}

/// Find-or-insert `key` (a `u64`-keyed group), returning its dense group index.
/// The per-row proxy call from the kernel (Umbra's `insert` — the table mechanics
/// are host code; the kernel generates the key and the fold).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_find_or_insert(table: &mut GroupTable<u64>, key: u64) -> u64 {
    table.intern(key) as u64
}

/// The final group count, called once after the fold loop to size the output.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_len(table: &GroupTable<u64>) -> u64 {
    table.len() as u64
}
