//! `rust-lms-std` — typed staged data structures for JIT kernels built on
//! [`rust_lms`]. The "Data Structures" and "Tuples" layers of a Tidy-Tuples-style
//! stack (see `docs/path_to_umbra_group_by.md`): growable, host-backed structures a
//! kernel folds into, plus dynamic-but-typed record layouts — reused by GROUP BY,
//! joins, sort, and `DISTINCT`.
//!
//! - [`SVec`] — a growable dynamic array. Proves **handle indirection**: a kernel
//!   structure can grow without dangling the pointer baked into the kernel, because
//!   the baked pointer targets a stable control block, not the movable buffer.
//! - [`RecordLayout`] / [`FieldHandle`] — a record whose *field set* is chosen at
//!   query-compile time but whose every field *access* stays typed. The record is a
//!   type-erased `*mut u8`; the leaves are `SMutPtr<T>`.

pub mod record;
pub mod svec;

pub use record::{FieldHandle, RecordLayout};
pub use svec::{svec_grow, HostVec, RawVec, RawVecType, SVec, SvecGrowExtern};
