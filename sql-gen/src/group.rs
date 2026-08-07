//! GROUP BY support — a Rust-hosted hash table mapping group keys to dense group
//! indices, driven from the JIT kernel.
//!
//! The kernel keeps the *hot* aggregation loop: per input row it computes the
//! group key, calls [`group_intern`] to get a group index, and JIT-folds each
//! accumulator into an output array slot at that index. This struct owns only the
//! `key -> index` map; the accumulator storage and group-key column are ordinary
//! JIT-written output arrays (sized to the row count, so indices never overflow).
//!
//! Hosting the state in Rust is deliberate: it's the same pattern as the string
//! `BytesPool`, and it's what lets these kernels later become partial aggregates
//! whose `[keys | accumulators]` state is merged across parallel workers.

use std::collections::HashMap;

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
    pub table: GroupTable,
    buffers: Vec<Vec<i64>>,
}

impl GroupState {
    /// Allocate `capacity`-row buffers, one per slot, each filled with its identity
    /// (`0` for count/sum/avg, `i64::MAX`/`MIN` for min/max — see the codegen layout).
    pub fn new(slot_inits: &[i64], capacity: usize) -> Self {
        let buffers = slot_inits
            .iter()
            .map(|&init| vec![init; capacity])
            .collect();
        Self {
            table: GroupTable::new(),
            buffers,
        }
    }

    /// Address of the group hash table (baked; handed to the `intern` extern).
    pub fn table_ptr(&mut self) -> u64 {
        &mut self.table as *mut GroupTable as u64
    }

    /// Base address of each slot buffer (baked; the kernel indexes by `gidx`).
    pub fn base_ptrs(&self) -> Vec<u64> {
        self.buffers.iter().map(|b| b.as_ptr() as u64).collect()
    }
}

/// Maps a group key to its dense index (`0..num_groups`), assigning the next
/// index the first time a key is seen. Single `i64` key for now (a widened
/// `Int32`/`Int64` group column).
#[derive(Default)]
pub struct GroupTable {
    index: HashMap<i64, u32>,
}

impl GroupTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct groups seen so far.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }
}

/// Intern `key`, returning its group index — the existing one, or the next free
/// index (`= current group count`) if `key` is new. Dense and monotonic, so it
/// doubles as the row cursor into the accumulator arrays.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_intern(table: &mut GroupTable, key: i64) -> u64 {
    let next = table.index.len() as u32;
    *table.index.entry(key).or_insert(next) as u64
}

/// The final group count, called once after the fold loop to size the output.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn group_len(table: &GroupTable) -> u64 {
    table.index.len() as u64
}
