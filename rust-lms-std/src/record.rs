//! Dynamic, but typed, record layouts — the "Tuples" layer of the stack (see
//! `docs/path_to_umbra_group_by.md §4`).
//!
//! A record whose *shape is decided at query-compile time* (e.g. a GROUP BY's
//! aggregate payload: `sum(a)` vs `min(a),max(b),avg(c)` are different tuples)
//! cannot be a Rust `struct`. So we describe it with a [`RecordLayout`] built by
//! reserving fields one at a time. Each reservation returns a **typed token**
//! ([`FieldId<T>`]) — the *set* of fields is dynamic, but each field carries its
//! Rust type. Reads/writes go through a [`DynamicRecord`], which hides the raw
//! `*mut u8` entirely and takes tokens: passing the wrong type is a *compile*
//! error, and a token from a different layout panics at stage 0 (a layout brand),
//! so a mis-addressed field can never reach an unchecked pointer op.
//!
//! This is Umbra's `store(target + layout[slot].offset, value)`, but the offset is
//! never handled by the caller — only a typed, layout-bound token is.

use std::marker::PhantomData;
use std::sync::atomic::{AtomicU32, Ordering};

use rust_lms::prelude::*;

/// Round `offset` up to a multiple of `align` (a power of two).
fn align_up(offset: usize, align: usize) -> usize {
    debug_assert!(align.is_power_of_two());
    (offset + align - 1) & !(align - 1)
}

/// Brands each [`RecordLayout`] with a unique id, so a [`FieldId`] can only be used
/// on a record built from the layout that issued it.
static LAYOUT_SEQ: AtomicU32 = AtomicU32::new(1);

/// A record layout being built up field by field. Tracks the running size and the
/// max field alignment, so [`stride`](Self::stride) pads to pack records in an
/// array. Compute the *same* layout from the same field sequence on the host (to
/// size/init the buffer) and in codegen (to access fields) — they agree by
/// construction, exactly like a `#[repr(C)]` struct would. `Copy` (an id + three
/// counters), so it rides freely into codegen closures.
#[derive(Clone, Copy, Debug)]
pub struct RecordLayout {
    id: u32,
    size: usize,
    align: usize,
    count: usize,
}

impl RecordLayout {
    pub fn new() -> Self {
        RecordLayout {
            id: LAYOUT_SEQ.fetch_add(1, Ordering::Relaxed),
            size: 0,
            align: 1,
            count: 0,
        }
    }

    /// Reserve the next field of staged type `T`, returning its typed, layout-bound
    /// token. The field is placed at the next `T`-aligned offset.
    pub fn field<T: StagedType + CopyType + 'static>(&mut self) -> FieldId<T> {
        let a = T::align_of().max(1);
        let offset = align_up(self.size, a);
        let index = self.count;
        self.size = offset + T::size_of();
        self.align = self.align.max(a);
        self.count += 1;
        FieldId {
            layout: self.id,
            index,
            offset,
            _t: PhantomData,
        }
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

    /// Number of fields reserved so far.
    pub fn num_fields(&self) -> usize {
        self.count
    }

    /// A [`DynamicRecord`] for record `index` in a packed array based at `base`
    /// (`base + index * stride`). Use in an emit loop that walks records by index.
    ///
    /// # Safety
    ///
    /// At execution, `base.add(index * self.stride())` must identify a live,
    /// initialized record allocated with this layout and remain writable for
    /// every use of the returned handle.
    pub unsafe fn record<B>(&self, ctx: &mut Ctx, base: B, index: Var<u64>) -> DynamicRecord
    where
        B: Staged<Out = SMutPtr<u8>> + 'static,
    {
        let stride = self.stride() as i64;
        let byte_off = ctx.bind(mul(int_cast::<i64, u64, _>(index), stride));
        // SAFETY: required by this method's contract.
        let ptr = ctx.bind(unsafe { ptr_offset_mut(base, byte_off) });
        // SAFETY: the computed pointer inherits the contract above.
        unsafe { self.wrap(ptr) }
    }

    /// A [`DynamicRecord`] over an already-computed record pointer (e.g. one an
    /// extern handed back). Brands it with this layout so its tokens are accepted.
    ///
    /// # Safety
    ///
    /// `ptr` must identify a live, initialized record allocated with this
    /// layout and remain writable for every use of the returned handle.
    pub unsafe fn wrap(&self, ptr: Var<SMutPtr<u8>>) -> DynamicRecord {
        DynamicRecord {
            layout_id: self.id,
            ptr,
        }
    }
}

impl Default for RecordLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// A typed, layout-bound handle to one field of a record: its byte offset plus the
/// staged type `T`, tagged with the id of the [`RecordLayout`] that issued it.
/// Only a [`RecordLayout::field`] call can make one, so its offset is always valid
/// for that layout; [`DynamicRecord`] checks the brand before touching memory.
pub struct FieldId<T> {
    layout: u32,
    index: usize,
    offset: usize,
    _t: PhantomData<T>,
}

impl<T> Clone for FieldId<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for FieldId<T> {}

impl<T> FieldId<T> {
    /// This field's byte offset within a record (for host-side layout math, e.g.
    /// building an identity template).
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// This field's position in the layout (0-based, reservation order).
    pub fn index(&self) -> usize {
        self.index
    }
}

/// A typed view over one packed record's bytes — the `*mut u8` never surfaces.
/// Field access takes a [`FieldId<T>`]: the type is checked at *compile* time, and
/// the layout brand at stage 0, so you can neither read a field at the wrong type
/// nor address a field the record doesn't have. `Copy` — a brand + a pointer `Var`.
#[derive(Clone, Copy)]
pub struct DynamicRecord {
    layout_id: u32,
    ptr: Var<SMutPtr<u8>>,
}

impl DynamicRecord {
    /// Load field `field` (its type is inferred from the token).
    pub fn get<T: StagedType + CopyType + 'static>(
        &self,
        ctx: &mut Ctx,
        field: FieldId<T>,
    ) -> Var<T> {
        self.check(field.layout, field.index);
        // SAFETY: `DynamicRecord`'s construction contract keeps the record
        // live, while the branded field token proves the offset and type.
        ctx.bind(unsafe { load_mut(self.field_ptr::<T>(field.offset)) })
    }

    /// Store `v` into field `field`.
    pub fn set<T: StagedType + CopyType + 'static>(
        &self,
        ctx: &mut Ctx,
        field: FieldId<T>,
        v: Var<T>,
    ) {
        self.check(field.layout, field.index);
        // SAFETY: `DynamicRecord`'s construction contract keeps the record
        // writable, while the branded field token proves the offset and type.
        ctx.emit(unsafe { store(self.field_ptr::<T>(field.offset), v) });
    }

    /// `(ptr + offset) as *mut T`.
    fn field_ptr<T: StagedType + CopyType + 'static>(
        &self,
        offset: usize,
    ) -> impl Staged<Out = SMutPtr<T>> {
        // SAFETY: all `FieldId` offsets lie within the branded record layout.
        ptr_cast_mut::<T, u8, _>(unsafe {
            ptr_offset_mut(self.ptr, Const::<i64>::new(offset as i64))
        })
    }

    #[track_caller]
    fn check(&self, field_layout: u32, field_index: usize) {
        assert_eq!(
            field_layout, self.layout_id,
            "FieldId (field #{field_index}) is from a different RecordLayout than this record",
        );
    }
}
