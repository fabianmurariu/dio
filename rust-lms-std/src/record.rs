//! Dynamic, but typed, record layouts — the "Tuples" layer of the stack (see
//! `docs/path_to_umbra_group_by.md §4`).
//!
//! A record whose *shape is decided at query-compile time* (e.g. a GROUP BY's
//! aggregate payload: `sum(a)` vs `min(a),max(b),avg(c)` are different tuples)
//! cannot be a Rust `struct`. So we describe it with a [`RecordLayout`] built by
//! reserving fields one at a time — the *set* of fields is dynamic, but each field
//! comes back as a **typed** [`FieldHandle<T>`]. The record as a whole is a
//! type-erased `*mut u8`; every field *access* stays a typed `SMutPtr<T>`.
//!
//! This is Umbra's `store(target + layout[slot].offset, value)`, but the offset is
//! paired with a Rust type at each leaf, so no `u64`-as-pointer escapes.

use std::marker::PhantomData;

use rust_lms::prelude::*;

/// Round `offset` up to a multiple of `align` (a power of two).
fn align_up(offset: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (offset + align - 1) & !(align - 1)
}

/// A record layout being built up field by field. Tracks the running size and the
/// max field alignment, so [`stride`](Self::stride) pads to pack records in an
/// array. Compute the *same* layout from the same field sequence on the host (to
/// size/init the buffer) and in codegen (to access fields) — they agree by
/// construction, exactly like a `#[repr(C)]` struct would.
#[derive(Clone, Copy, Debug, Default)]
pub struct RecordLayout {
    size: usize,
    align: usize,
}

impl RecordLayout {
    pub fn new() -> Self {
        RecordLayout { size: 0, align: 1 }
    }

    /// Reserve the next field of staged type `T`, returning its typed handle. The
    /// field is placed at the next `T`-aligned offset.
    pub fn field<T: StagedType + CopyType + 'static>(&mut self) -> FieldHandle<T> {
        let a = T::align_of().max(1);
        let offset = align_up(self.size, a);
        self.size = offset + T::size_of();
        self.align = self.align.max(a);
        FieldHandle::from_offset(offset)
    }

    /// One record's stride in bytes (size padded to the record's alignment, so a
    /// contiguous array of records keeps every field aligned).
    pub fn stride(&self) -> usize {
        align_up(self.size, self.align.max(1))
    }

    /// The record alignment (max field alignment).
    pub fn align(&self) -> usize {
        self.align.max(1)
    }

    /// A typed pointer to record `index` in a packed array based at `base`:
    /// `base + index * stride`. Bind it once per row and reuse across field
    /// accesses (the record base is loop-invariant within a row).
    pub fn record<B>(&self, ctx: &mut Ctx, base: B, index: Var<u64>) -> Var<SMutPtr<u8>>
    where
        B: Staged<Out = SMutPtr<u8>> + 'static,
    {
        let stride = self.stride() as i64;
        let byte_off = ctx.bind(mul(int_cast::<i64, u64, _>(index), stride));
        ctx.bind(ptr_offset_mut(base, byte_off))
    }
}

/// A typed handle to one field of a record: its byte offset plus the staged type
/// `T`. Reconstructible from a stored `offset` ([`from_offset`](Self::from_offset)),
/// so a query planner can keep offsets in a type-erased list and re-type each field
/// at access time (dispatching on the column's runtime `DataType`).
pub struct FieldHandle<T> {
    offset: usize,
    _t: PhantomData<T>,
}

impl<T> Clone for FieldHandle<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for FieldHandle<T> {}

impl<T: StagedType + CopyType + 'static> FieldHandle<T> {
    /// A handle for a field at byte `offset` (must match a `RecordLayout::field`
    /// placement for type `T`).
    pub fn from_offset(offset: usize) -> Self {
        FieldHandle {
            offset,
            _t: PhantomData,
        }
    }

    pub fn offset(&self) -> usize {
        self.offset
    }

    /// A typed pointer to this field within record `entry`: `(entry + offset) as *mut T`.
    pub fn at<B>(&self, entry: B) -> impl Staged<Out = SMutPtr<T>>
    where
        B: Staged<Out = SMutPtr<u8>> + 'static,
    {
        ptr_cast_mut::<T, u8, _>(ptr_offset_mut(entry, Const::<i64>::new(self.offset as i64)))
    }

    /// Load this field from record `entry`.
    pub fn get<B>(&self, ctx: &mut Ctx, entry: B) -> Var<T>
    where
        B: Staged<Out = SMutPtr<u8>> + 'static,
    {
        ctx.bind(load_ref_mut(self.at(entry)))
    }

    /// Store `v` into this field of record `entry`.
    pub fn set<B>(&self, ctx: &mut Ctx, entry: B, v: Var<T>)
    where
        B: Staged<Out = SMutPtr<u8>> + 'static,
    {
        ctx.emit(store(self.at(entry), v));
    }
}
