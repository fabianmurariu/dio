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
