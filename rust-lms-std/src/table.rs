//! A hash table that generated code probes and fills itself: [`ChainedTable`].
//!
//! ```text
//!  TableHead                directory: 2^k chain heads    entries: chunks of 256, never moved
//! ┌───────────┐            ┌─────┐
//! │ dir    ───┼───────────▶│  ●──┼──▶ [next ● | hash | key… | state…] ──▶ [next ∅ | …]
//! │ mask      │            │  ∅  │
//! │ len       │            │  ●──┼──▶ [next ∅ | hash | key… | state…]
//! │ chunks ───┼──┐         └─────┘
//! └───────────┘  └──▶ [chunk 0][chunk 1]…   entry k: chunk k / 256, slot k % 256
//! ```
//!
//! An **entry** is a record of an [`EntryLayout`]: the table's own `next` (the chain) and
//! `hash`, then the caller's fields (keys and states). The caller hashes its key and says
//! what equal keys are; the table does the rest in generated code:
//!
//! - **find:** `dir[hash & mask]`, then along the chain, comparing `hash` before the key;
//! - **insert:** the next entry of the current chunk, its `hash` and the caller's fields
//!   set, linked in at the head of its chain, `len + 1`;
//! - **grow:** only when `len` reaches a chunk boundary does the kernel call the host
//!   ([`chained_table_make_room`]), which adds a chunk and, when there is one entry per
//!   chain head, doubles the directory and relinks every entry. Entries stay where they
//!   are, so an entry the kernel holds survives every insert.
//!
//! Like a block, the host owns the memory ([`ChainedTable`], one per kernel call) and the
//! kernel reads its [`TableHead`] through typed fields ([`StagedTable`]), built from the
//! same [`EntryLayout`]:
//!
//! ```ignore
//! let mut entry = EntryLayout::new();
//! let key = entry.field::<u64>();
//! let count = entry.field::<u64>();
//!
//! // generated code
//! let table = unsafe { StagedTable::new(ctx, &entry, table_ref, make_room) };
//! let e = table.find_or_insert(
//!     ctx,
//!     hash_word(ctx, SEED, k),
//!     |ctx, e| ctx.bind(eq(e.get(ctx, key), k)),
//!     |ctx, e| { e.set(ctx, key, k); e.set(ctx, count, ctx.var(0u64)) },
//! );
//! e.set(ctx, count, ctx.bind(add(e.get(ctx, count), 1u64)));
//! table.for_each(ctx, |ctx, e| { /* read e's fields */ });
//!
//! // host, per call
//! let mut table = ChainedTable::new(&entry);
//! compiled.call(&table.table_ref());
//! ```

use std::alloc::{Layout, alloc_zeroed, dealloc, handle_alloc_error};

use rust_lms::prelude::*;

use crate::{DynamicRecord, FieldId, RecordLayout};

/// Entries per chunk, and the directory's smallest size: a power of two, so an entry's
/// chunk and slot are a shift and a mask, and the directory only ever needs to double at a
/// chunk boundary.
pub const CHUNK_ENTRIES: usize = 256;
const CHUNK_SHIFT: u64 = CHUNK_ENTRIES.trailing_zeros() as u64;
const SLOT_MASK: u64 = CHUNK_ENTRIES as u64 - 1;

/// The layout of a table's entries: the table's `next` and `hash`, then the caller's
/// fields. One layout serves the staged side and every host table of a kernel.
#[derive(Clone, Copy)]
pub struct EntryLayout {
    record: RecordLayout,
    next: FieldId<SMutPtr<u8>>,
    hash: FieldId<u64>,
}

impl EntryLayout {
    /// A layout with only the table's own fields.
    pub fn new() -> Self {
        let mut record = RecordLayout::new();
        let next = record.field();
        let hash = record.field();
        Self { record, next, hash }
    }

    /// Reserves one of the caller's fields.
    pub fn field<T: StagedType + CopyType + 'static>(&mut self) -> FieldId<T> {
        self.record.field()
    }
}

impl Default for EntryLayout {
    fn default() -> Self {
        Self::new()
    }
}

/// What the kernel reads of a table: the directory and its mask, the number of entries,
/// and the chunks' addresses. Only [`ChainedTable`] rewrites `dir`, `mask` and `chunks`;
/// the kernel advances `len`.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct TableHead {
    #[staged(SMutPtr<SMutPtr<u8>>)]
    dir: *mut *mut u8,
    #[staged(u64)]
    mask: u64,
    #[staged(u64)]
    len: u64,
    #[staged(SPtr<SMutPtr<u8>>)]
    chunks: *const *mut u8,
}

/// What a kernel takes for a table: its head, and the table for the cold extern. Made by
/// [`ChainedTable::table_ref`].
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct TableRef {
    #[staged(SMutPtr<TableHead>)]
    head: *mut TableHead,
    #[staged(SMutPtr<Opaque<ChainedTable>>)]
    owner: *mut ChainedTable,
}

/// The host owner of a table's memory: the directory and the chunks of entries.
pub struct ChainedTable {
    /// One chunk's allocation: `CHUNK_ENTRIES` entries, aligned for the layout.
    chunk: Layout,
    stride: usize,
    next: usize,
    hash: usize,
    /// Boxed so the address a kernel holds stays put.
    head: Box<TableHead>,
    dir: Vec<*mut u8>,
    chunks: Vec<*mut u8>,
}

// SAFETY: the table owns its directory and chunks, and every pointer in them points into
// its own chunks; the kernel changes it only through `table_ref`, which takes `&mut self`.
unsafe impl Send for ChainedTable {}

impl ChainedTable {
    /// An empty table of `layout`'s entries.
    pub fn new(layout: &EntryLayout) -> Self {
        let stride = layout.record.stride();
        let chunk = Layout::from_size_align(stride * CHUNK_ENTRIES, layout.record.align())
            .expect("a chunk of entries fits in memory");
        let mut table = Self {
            chunk,
            stride,
            next: layout.next.offset(),
            hash: layout.hash.offset(),
            head: Box::new(TableHead {
                dir: std::ptr::null_mut(),
                mask: 0,
                len: 0,
                chunks: std::ptr::null(),
            }),
            dir: Vec::new(),
            chunks: Vec::new(),
        };
        table.relink(CHUNK_ENTRIES);
        table
    }

    /// What a kernel takes to use this table through [`StagedTable`]. Valid while the
    /// table stays where it is and nothing else uses it.
    pub fn table_ref(&mut self) -> TableRef {
        TableRef {
            head: &mut *self.head,
            owner: self,
        }
    }

    /// The number of entries.
    pub fn len(&self) -> usize {
        self.head.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Room for the next entry, at a chunk boundary: a new chunk, and when the directory
    /// has a head per entry, one twice the size.
    fn make_room(&mut self) {
        let len = self.len();
        if len == self.dir.len() {
            self.relink(2 * len);
        }
        if len / CHUNK_ENTRIES == self.chunks.len() {
            // SAFETY: the layout has a non-zero size (every entry holds `next` and `hash`).
            let chunk = unsafe { alloc_zeroed(self.chunk) };
            if chunk.is_null() {
                handle_alloc_error(self.chunk);
            }
            self.chunks.push(chunk);
            self.head.chunks = self.chunks.as_ptr();
        }
    }

    /// A directory of `size` heads with every entry linked into it.
    fn relink(&mut self, size: usize) {
        let mut dir = vec![std::ptr::null_mut::<u8>(); size];
        let mask = size as u64 - 1;
        for k in 0..self.len() {
            let entry = self.entry(k);
            // SAFETY: entry `k` is below `len`, so the kernel wrote its `hash`; `next` is
            // a field of it, aligned for a pointer.
            unsafe {
                let slot = &mut dir[(entry.add(self.hash).cast::<u64>().read() & mask) as usize];
                entry.add(self.next).cast::<*mut u8>().write(*slot);
                *slot = entry;
            }
        }
        self.dir = dir;
        self.head.dir = self.dir.as_mut_ptr();
        self.head.mask = mask;
    }

    /// Entry `k`'s address.
    fn entry(&self, k: usize) -> *mut u8 {
        // SAFETY: chunk `k / CHUNK_ENTRIES` exists for every `k` a caller passes (below
        // `len`), and holds `CHUNK_ENTRIES` entries of `stride` bytes.
        unsafe { self.chunks[k / CHUNK_ENTRIES].add(k % CHUNK_ENTRIES * self.stride) }
    }
}

impl Drop for ChainedTable {
    fn drop(&mut self) {
        for &chunk in &self.chunks {
            // SAFETY: allocated in `make_room` with this layout, freed once.
            unsafe { dealloc(chunk, self.chunk) };
        }
    }
}

/// Makes room in `table` for its next entry: the cold path of
/// [`StagedTable::find_or_insert`], taken once per chunk of entries.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn chained_table_make_room(table: &mut ChainedTable) {
    table.make_room();
}

/// A kernel's handle on a [`ChainedTable`] of `layout`'s entries.
pub struct StagedTable<'l> {
    layout: &'l EntryLayout,
    head: Var<SMutPtr<TableHead>>,
    owner: Var<SMutPtr<Opaque<ChainedTable>>>,
    make_room: ExternRef<ChainedTableMakeRoomExtern>,
}

impl<'l> StagedTable<'l> {
    /// The table `table` refers to.
    ///
    /// # Safety
    ///
    /// At execution, `table` is [`ChainedTable::table_ref`] of a table built from
    /// `layout`, that stays where it is and that nothing else uses during the call.
    pub unsafe fn new(
        ctx: &Ctx,
        layout: &'l EntryLayout,
        table: Var<SRef<TableRef>>,
        make_room: ExternRef<ChainedTableMakeRoomExtern>,
    ) -> Self {
        let table = ref_as_ptr(table);
        // SAFETY: both are plain fields of a live `TableRef` (the caller).
        let (head, owner) = unsafe {
            (
                ctx.bind(load_field_unchecked(table, TableRefType::head())),
                ctx.bind(load_field_unchecked(table, TableRefType::owner())),
            )
        };
        Self {
            layout,
            head,
            owner,
            make_room,
        }
    }

    /// The number of entries.
    pub fn len(&self, ctx: &Ctx) -> Var<u64> {
        // SAFETY: a field of the live head (`new`'s contract).
        ctx.bind(unsafe { load_field_unchecked(self.head, TableHeadType::len()) })
    }

    /// The entry whose key `same_key` accepts among those of `hash`, or a new one, its
    /// `hash` set and its fields set by `init`. `same_key` reads an entry of the same hash;
    /// `init` must set every field the caller reads later (a new entry's fields are zero).
    pub fn find_or_insert(
        &self,
        ctx: &Ctx,
        hash: Var<u64>,
        same_key: impl FnOnce(&Ctx, &DynamicRecord) -> Var<bool>,
        init: impl FnOnce(&Ctx, &DynamicRecord),
    ) -> DynamicRecord {
        let entry = ctx.join(|ctx, found| {
            // SAFETY: a head of the live directory.
            let at = ctx.var(unsafe { load_mut(self.slot(ctx, hash)) });
            ctx.while_loop(not(ptr_is_null(at)), |ctx| {
                let e = self.entry_at(at);
                ctx.if_then(eq(e.get(ctx, self.layout.hash), hash), |ctx| {
                    ctx.if_then(same_key(ctx, &e), |ctx| ctx.goto(found, at));
                });
                ctx.store(at, e.get(ctx, self.layout.next));
            });
            self.insert(ctx, hash, init)
        });
        self.entry_at(entry)
    }

    /// Calls `f` with every entry, in the order they were inserted.
    pub fn for_each(&self, ctx: &Ctx, f: impl FnOnce(&Ctx, DynamicRecord)) {
        let (len, chunks) = (self.len(ctx), self.chunks(ctx));
        range(0u64, len).for_each(ctx, move |ctx, k| {
            f(ctx, self.entry_at(self.entry(ctx, chunks, k)))
        });
    }

    /// A new entry at the end, linked into its chain.
    fn insert(
        &self,
        ctx: &Ctx,
        hash: Var<u64>,
        init: impl FnOnce(&Ctx, &DynamicRecord),
    ) -> Var<SMutPtr<u8>> {
        let len = self.len(ctx);
        let (owner, make_room) = (self.owner, self.make_room);
        ctx.if_then(eq(bitand(len, SLOT_MASK), 0u64), move |ctx| {
            // SAFETY: `owner` is the table of this head, the kernel's alone for the call
            // (`new`'s contract); it rewrites the head, read again below.
            ctx.emit(unsafe { call_extern1_unchecked(make_room, owner) });
        });
        let at = self.entry(ctx, self.chunks(ctx), len);
        let e = self.entry_at(at);
        e.set(ctx, self.layout.hash, hash);
        init(ctx, &e);
        let slot = self.slot(ctx, hash);
        // SAFETY: a head of the live directory.
        e.set(ctx, self.layout.next, ctx.bind(unsafe { load_mut(slot) }));
        // SAFETY: a head of the live directory; `len` is a writable field of the head.
        unsafe {
            ctx.emit(store(slot, at));
            ctx.emit(store(
                field_addr(self.head, TableHeadType::len()),
                add(len, 1u64),
            ));
        }
        at
    }

    /// The directory head `hash` belongs to.
    fn slot(&self, ctx: &Ctx, hash: Var<u64>) -> Var<SMutPtr<SMutPtr<u8>>> {
        // SAFETY: fields of the live head; `hash & mask` is below the directory's size.
        unsafe {
            let dir = ctx.bind(load_field_unchecked(self.head, TableHeadType::dir()));
            let mask = ctx.bind(load_field_unchecked(self.head, TableHeadType::mask()));
            let at = int_cast::<i64, u64, _>(bitand(hash, mask));
            ctx.bind(ptr_offset_mut(dir, at))
        }
    }

    /// The chunks' addresses, read from the head.
    fn chunks(&self, ctx: &Ctx) -> Var<SPtr<SMutPtr<u8>>> {
        // SAFETY: a field of the live head.
        ctx.bind(unsafe { load_field_unchecked(self.head, TableHeadType::chunks()) })
    }

    /// Entry `k`'s address: slot `k % 256` of chunk `k / 256`.
    fn entry(&self, ctx: &Ctx, chunks: Var<SPtr<SMutPtr<u8>>>, k: Var<u64>) -> Var<SMutPtr<u8>> {
        let stride = self.layout.record.stride() as u64;
        // SAFETY: chunk `k / 256` exists (`make_room` ran at `k`'s chunk boundary), and
        // slot `k % 256` is an entry inside it.
        unsafe {
            let chunk = ctx.bind(load(ptr_offset(
                chunks,
                int_cast::<i64, u64, _>(shr(k, CHUNK_SHIFT)),
            )));
            let offset = int_cast::<i64, u64, _>(mul(bitand(k, SLOT_MASK), stride));
            ctx.bind(ptr_offset_mut(chunk, offset))
        }
    }

    fn entry_at(&self, at: Var<SMutPtr<u8>>) -> DynamicRecord {
        // SAFETY: `at` is an entry of this table, of this layout, alive for the call.
        unsafe { self.layout.record.wrap(at) }
    }
}
