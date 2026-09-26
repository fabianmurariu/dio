//! `rust-lms-std` — typed staged data structures for JIT kernels built on
//! [`rust_lms`]. The "Data Structures" and "Tuples" layers of a Tidy-Tuples-style
//! stack (see `docs/path_to_umbra_group_by.md`): growable, host-backed structures a
//! kernel folds into, plus dynamic-but-typed record layouts — reused by GROUP BY,
//! joins, sort, and `DISTINCT`.
//!
//! - [`SVec`] — a growable dynamic array. Proves **handle indirection**: a kernel
//!   structure can grow without dangling the pointer baked into the kernel, because
//!   the baked pointer targets a stable control block, not the movable buffer.
//! - [`RecordLayout`] / [`FieldId`] / [`DynamicRecord`] — a record whose *field set*
//!   is chosen at query-compile time but whose every field *access* stays typed.
//!   Reads/writes take a typed, layout-bound token; the raw `*mut u8` never surfaces.
//! - [`IterPool`] / [`StagedIterPool`] / [`PooledIterFns`] — Rust iterators driven
//!   from staged code a chunk at a time, through reused per-nesting-level slots
//!   assigned at stage 0.

pub mod iter_pool;
pub mod record;
pub mod svec;

pub use iter_pool::{
    Chunk, ChunkSlot, ChunkSlotRef, IterPool, IterPoolRef, PoolSlot, PoolSlotRef, PooledIter,
    PooledIterFns, PooledIterKind, StagedIterPool,
};
pub use record::{DynamicRecord, FieldId, RecordLayout};
pub use svec::{HostVec, HostVecHandle, RawVec, RawVecType, SVec, SvecGrowExtern, svec_grow};
