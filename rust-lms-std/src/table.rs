//! A chained hash table whose lookups run in generated code: [`ChainedTable`].
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
//! `hash`, then the caller's fields (keys and states). The work is split as in Umbra:
//!
//! - **find, generated, per row:** `dir[hash & mask]`, then along the chain, comparing
//!   `hash` before the caller's key equality;
//! - **insert, host, per new key** ([`chained_table_insert`]): the next entry of the
//!   current chunk, its `hash` set, linked in at the head of its chain, `len + 1`. At a
//!   chunk boundary it first adds a chunk and, when the directory is ¼ full, doubles it and
//!   relinks every entry;
//! - **init, generated:** the caller sets its fields on the new entry.
//!
//! The directory stays ⅛ to ¼ full, so chains are short and a lookup that finds its key
//! rarely walks past another. Entries never move, so an entry the kernel holds survives
//! every insert. Every write to the table's structure (`len`, the directory, the chunks)
//! happens on the host; the kernel only reads [`TableHead`]. `docs/chained_table.md` has
//! the design and the measurements behind it.
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
//! let table = unsafe { StagedTable::new(ctx, &entry, table_ref, insert) };
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

use crate::{DynamicRecord, FieldId, Record, RecordLayout};

/// Entries per chunk: a power of two, so an entry's chunk and slot are a shift and a mask.
pub const CHUNK_ENTRIES: usize = 256;
const CHUNK_SHIFT: u64 = CHUNK_ENTRIES.trailing_zeros() as u64;
const SLOT_MASK: u64 = CHUNK_ENTRIES as u64 - 1;

/// The directory has at least this many chain heads per entry, so it is ⅛ to ¼ full. Its
/// sizes (`SPARSE · 256 · 2^k`) are all multiples of 256, so it only ever fills at a chunk
/// boundary, where `insert` already calls `make_room`.
const SPARSE: usize = 4;

/// The layout of a table's entries: the table's `next` and `hash`, then the caller's
/// fields. One layout serves the staged side and every host table of a kernel.
#[derive(Clone, Copy)]
pub struct EntryLayout {
    record: RecordLayout,
    next: FieldId<SMutPtr<Record>>,
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
/// and the chunks' addresses. Only [`ChainedTable`] writes it.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct TableHead {
    #[staged(SMutPtr<SMutPtr<Record>>)]
    dir: *mut *mut Record,
    #[staged(u64)]
    mask: u64,
    #[staged(u64)]
    len: u64,
    #[staged(SPtr<SMutPtr<Record>>)]
    chunks: *const *mut Record,
}

/// What a kernel takes for a table: its head, and the table for the insert extern. Made
/// by [`ChainedTable::table_ref`].
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct TableRef {
    #[staged(SMutPtr<TableHead>)]
    head: *mut TableHead,
    #[staged(SMutPtr<Opaque<ChainedTable>>)]
    owner: *mut ChainedTable,
}

/// The host owner of a table's memory, and the only writer of its structure: the
/// directory, the chunks of entries, and the entry count.
pub struct ChainedTable {
    /// One chunk's allocation: `CHUNK_ENTRIES` entries, aligned for the layout.
    chunk: Layout,
    stride: usize,
    next: usize,
    hash: usize,
    /// Boxed so the address a kernel holds stays put.
    head: Box<TableHead>,
    dir: Vec<*mut Record>,
    chunks: Vec<*mut Record>,
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
        table.relink(SPARSE * CHUNK_ENTRIES);
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

    /// A new entry of `hash`, linked at the head of its chain. Its other fields are zero.
    fn insert(&mut self, hash: u64) -> *mut Record {
        let len = self.len();
        if len.is_multiple_of(CHUNK_ENTRIES) {
            self.make_room();
        }
        let entry = self.entry(len);
        let slot = &mut self.dir[(hash & self.head.mask) as usize];
        // SAFETY: entry `len` lies in an allocated chunk (made room for above), and
        // `hash` and `next` are fields of it, aligned for their types.
        unsafe {
            Self::field::<u64>(entry, self.hash).write(hash);
            Self::field::<*mut Record>(entry, self.next).write(*slot);
        }
        *slot = entry;
        self.head.len += 1;
        entry
    }

    /// Room for the entry at `len`, a chunk boundary: a new chunk, and when the directory
    /// is ¼ full, one twice the size.
    fn make_room(&mut self) {
        let len = self.len();
        if SPARSE * len == self.dir.len() {
            self.relink(2 * SPARSE * len);
        }
        // SAFETY: the layout has a non-zero size (every entry holds `next` and `hash`).
        let chunk = unsafe { alloc_zeroed(self.chunk) };
        if chunk.is_null() {
            handle_alloc_error(self.chunk);
        }
        self.chunks.push(chunk.cast());
        self.head.chunks = self.chunks.as_ptr();
    }

    /// A directory of `size` heads with every entry linked into it, by its stored hash.
    fn relink(&mut self, size: usize) {
        let mut dir = vec![std::ptr::null_mut::<Record>(); size];
        let mask = size as u64 - 1;
        for k in 0..self.len() {
            let entry = self.entry(k);
            // SAFETY: entry `k` is below `len`, so its `hash` was written by `insert`.
            unsafe {
                let slot = &mut dir[(Self::field::<u64>(entry, self.hash).read() & mask) as usize];
                Self::field::<*mut Record>(entry, self.next).write(*slot);
                *slot = entry;
            }
        }
        self.dir = dir;
        self.head.dir = self.dir.as_mut_ptr();
        self.head.mask = mask;
    }

    /// Entry `k`'s address.
    fn entry(&self, k: usize) -> *mut Record {
        // SAFETY: chunk `k / CHUNK_ENTRIES` exists for every `k` a caller passes (at most
        // `len`, made room for), and holds `CHUNK_ENTRIES` entries of `stride` bytes.
        unsafe {
            self.chunks[k / CHUNK_ENTRIES]
                .cast::<u8>()
                .add(k % CHUNK_ENTRIES * self.stride)
                .cast()
        }
    }

    /// The field at `offset` in `entry`.
    ///
    /// # Safety
    ///
    /// `entry` is an entry of a table and `offset` the offset of a `T` in it.
    unsafe fn field<T>(entry: *mut Record, offset: usize) -> *mut T {
        // SAFETY: forwarded from this function's contract.
        unsafe { entry.cast::<u8>().add(offset).cast() }
    }
}

impl Drop for ChainedTable {
    fn drop(&mut self) {
        for &chunk in &self.chunks {
            // SAFETY: allocated in `make_room` with this layout, freed once.
            unsafe { dealloc(chunk.cast(), self.chunk) };
        }
    }
}

/// A new entry of `hash` in `table`, linked into its chain: the insert of
/// [`StagedTable::find_or_insert`], taken once per new key.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn chained_table_insert(table: &mut ChainedTable, hash: u64) -> *mut Record {
    table.insert(hash)
}

/// A kernel's handle on a [`ChainedTable`] of `layout`'s entries.
pub struct StagedTable<'l> {
    layout: &'l EntryLayout,
    head: Var<SMutPtr<TableHead>>,
    owner: Var<SMutPtr<Opaque<ChainedTable>>>,
    insert: ExternRef<ChainedTableInsertExtern>,
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
        insert: ExternRef<ChainedTableInsertExtern>,
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
            insert,
        }
    }

    /// The number of entries.
    pub fn len(&self, ctx: &Ctx) -> Var<u64> {
        // SAFETY: a field of the live head (`new`'s contract).
        ctx.bind(unsafe { load_field_unchecked(self.head, TableHeadType::len()) })
    }

    /// The entry whose key `same_key` accepts among those of `hash`, or a new one of
    /// `hash` whose fields `init` sets.
    ///
    /// `same_key` reads an entry of the same hash. `init` must set every field the caller
    /// reads later (a new entry's fields are zero) and must not use this table: the new
    /// entry is already linked into its chain when `init` runs.
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
            // SAFETY: `owner` is the table of this head, the kernel's alone for the call
            // (`new`'s contract). It may rewrite the head; `slot` and `chunks` read it
            // again each time.
            let at = ctx.bind(unsafe { call_extern2_unchecked(self.insert, self.owner, hash) });
            init(ctx, &self.entry_at(at));
            at
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

    /// The directory head `hash` belongs to.
    fn slot(&self, ctx: &Ctx, hash: Var<u64>) -> Var<SMutPtr<SMutPtr<Record>>> {
        // SAFETY: fields of the live head; `hash & mask` is below the directory's size.
        unsafe {
            let dir = ctx.bind(load_field_unchecked(self.head, TableHeadType::dir()));
            let mask = ctx.bind(load_field_unchecked(self.head, TableHeadType::mask()));
            let at = int_cast::<i64, u64, _>(bitand(hash, mask));
            ctx.bind(ptr_offset_mut(dir, at))
        }
    }

    /// The chunks' addresses, read from the head.
    fn chunks(&self, ctx: &Ctx) -> Var<SPtr<SMutPtr<Record>>> {
        // SAFETY: a field of the live head.
        ctx.bind(unsafe { load_field_unchecked(self.head, TableHeadType::chunks()) })
    }

    /// Entry `k`'s address: slot `k % 256` of chunk `k / 256`.
    fn entry(
        &self,
        ctx: &Ctx,
        chunks: Var<SPtr<SMutPtr<Record>>>,
        k: Var<u64>,
    ) -> Var<SMutPtr<Record>> {
        let stride = self.layout.record.stride() as u64;
        // SAFETY: `k < len`, so chunk `k / 256` exists, and slot `k % 256` is an entry
        // inside it.
        unsafe {
            let chunk = ctx.bind(load(ptr_offset(
                chunks,
                int_cast::<i64, u64, _>(shr(k, CHUNK_SHIFT)),
            )));
            let offset = int_cast::<i64, u64, _>(mul(bitand(k, SLOT_MASK), stride));
            let bytes = ptr_offset_mut(ptr_cast_mut::<u8, Record, _>(chunk), offset);
            ctx.bind(ptr_cast_mut::<Record, u8, _>(bytes))
        }
    }

    fn entry_at(&self, at: Var<SMutPtr<Record>>) -> DynamicRecord {
        // SAFETY: `at` is an entry of this table, of this layout, alive for the call.
        unsafe { self.layout.record.wrap(at) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A table of `u64` keys, filled on the host the way the extern fills it.
    fn filled(hashes: impl IntoIterator<Item = u64>) -> ChainedTable {
        let mut table = ChainedTable::new(&EntryLayout::new());
        for h in hashes {
            table.insert(h);
        }
        table
    }

    /// The hashes along the chain of `hash`'s slot.
    fn chain(table: &ChainedTable, hash: u64) -> Vec<u64> {
        let mut at = table.dir[(hash & table.head.mask) as usize];
        let mut hashes = vec![];
        while !at.is_null() {
            // SAFETY: `at` is one of the table's entries.
            unsafe {
                hashes.push(ChainedTable::field::<u64>(at, table.hash).read());
                at = ChainedTable::field::<*mut Record>(at, table.next).read();
            }
        }
        hashes
    }

    /// Every entry is in exactly its slot's chain, newest first, after the directory has
    /// doubled twice (1024 → 2048 → 4096 heads), and the head stays ⅛ to ¼ full.
    #[test]
    fn insert_links_every_entry_into_its_chain_across_doublings() {
        let hashes: Vec<u64> = (0..600u64)
            .map(|i| i.wrapping_mul(0x9e37_79b9_7f4a_7c15))
            .collect();
        let table = filled(hashes.iter().copied());
        assert_eq!(table.len(), 600);
        assert_eq!(table.dir.len(), 4096);
        assert_eq!(table.head.mask, 4095);
        for &h in &hashes {
            let expected: Vec<u64> = hashes
                .iter()
                .rev()
                .copied()
                .filter(|&o| o & 4095 == h & 4095)
                .collect();
            assert_eq!(chain(&table, h), expected, "chain of {h:#x}");
        }
    }

    /// An entry's address is the same before and after the table grows around it.
    #[test]
    fn an_entry_does_not_move_when_the_table_grows() {
        let mut table = ChainedTable::new(&EntryLayout::new());
        let first = table.insert(7);
        for h in 0..5000 {
            table.insert(h);
        }
        assert_eq!(table.entry(0), first);
        assert_eq!(chain(&table, 7).last(), Some(&7));
    }
}
