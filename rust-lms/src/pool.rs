//! An append-only pool of byte strings, addressed as Arrow **byte views**, that the
//! host and kernels both append to.
//!
//! [`BytesPool`] stores values the way Arrow's `GenericByteViewBuilder` does: a value of
//! at most [`MAX_INLINE_LEN`] bytes is its view and is not stored at all; a longer one is
//! copied into a **block** and its view is `(len, prefix, buf_idx, offset)` into the
//! pool's blocks. [`BytesPool::buffers`] is the table those views index, so a kernel
//! reads a pool's value as it reads a row of an Arrow array: a [`ByteView`] of the view
//! and the table.
//!
//! **Appending.** The host appends with [`BytesPool::append`]; a kernel with
//! [`StagedPool::copy_in`], through the pool's [`PoolHead`] (the current block's address,
//! bytes used, capacity and index), which it reads and advances itself. Only a full block
//! leaves the kernel, through the cold [`bytes_pool_new_block`] extern. Both write into
//! the block's spare capacity; [`BytesPool::settle`] sets its length afterwards, so read
//! the table after a kernel's appends only once the pool is settled.
//!
//! **Stable addresses.** A block is allocated at its full size and never reallocated,
//! and a value never spans two blocks (one that does not fit starts a new block; one
//! larger than the block size gets a block of its own), so the bytes a view names stay
//! where they are for the pool's lifetime. The *table* grows by one entry per block: its
//! address is stable only while nothing is appended. A plan appends its literals while it
//! is compiled, then bakes the table's address into the kernel.
//!
//! **Lifetime contract (not expressed in the type system; it cannot cross the JIT
//! boundary):** the pool outlives every kernel call that reads its views; a kernel that
//! reads them through a baked table runs only while nothing appends; a kernel that
//! appends is the pool's only user for the call, and the pool stays where it is.
//!
//! [`ByteView`]: crate::byte_view::ByteView

use rust_lms_derive::{StagedType, extern_fn};

use crate::{
    byte_view::relocate_view,
    byte_view::{ByteBuffer, ByteKind, ByteView, MAX_INLINE_LEN, RawByteView, make_view},
    ctx::Ctx,
    ffi::{ExternRef, call_extern2_unchecked},
    num::{add, gt, int_cast},
    opaque::Opaque,
    refer::{SMutPtr, SPtr, SRef, copy_bytes, ptr_offset_mut, ref_as_ptr, store},
    staged::{Const, Var},
    r#struct::{field_addr, load_field_unchecked},
};

/// Default block size (64 KiB): large enough that most values share a block, small
/// enough not to over-allocate for a few literals.
const DEFAULT_BLOCK: usize = 64 * 1024;

/// An append-only pool of byte strings with stable addresses, its values Arrow views.
/// See the [module docs](self).
pub struct BytesPool {
    /// Each block is allocated at its capacity and never reallocated. Values are written
    /// into the last block's spare capacity; its length catches up in [`settle`].
    ///
    /// [`settle`]: Self::settle
    blocks: Vec<Vec<u8>>,
    /// One entry per block, its length current as of the last [`settle`]: the table the
    /// views index.
    ///
    /// [`settle`]: Self::settle
    buffers: Vec<ByteBuffer>,
    /// The capacity of a new block (a larger value gets a block of its own size).
    block_size: usize,
    /// Where the next value goes, boxed so the address a kernel holds stays put.
    head: Box<PoolHead>,
}

// SAFETY: the pool owns everything its pointers point into: the head's `cur` and the
// table's entries are its own blocks, which never move. Appending needs `&mut self` on the
// host, and a kernel appends only through `pool_ref`, which takes `&mut self` too; through
// `&BytesPool` the blocks are only read.
unsafe impl Send for BytesPool {}
unsafe impl Sync for BytesPool {}

/// The block values are appended to: its address, the bytes used of it, its capacity
/// and its index in the table; and the table's address, which moves when a block is
/// added. The host and kernels append through it alike; only [`BytesPool`] starts a block.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct PoolHead {
    #[staged(SMutPtr<u8>)]
    cur: *mut u8,
    #[staged(u64)]
    used: u64,
    #[staged(u64)]
    cap: u64,
    #[staged(u64)]
    buf_idx: u64,
    #[staged(SPtr<ByteBuffer>)]
    buffers: *const ByteBuffer,
}

impl BytesPool {
    /// A pool with the default block size.
    pub fn new() -> Self {
        Self::with_block_size(DEFAULT_BLOCK)
    }

    /// A pool whose blocks are (at least) `block_size` bytes each.
    ///
    /// # Panics
    ///
    /// If `block_size` is over `u32::MAX`: a view's offset is a `u32`.
    pub fn with_block_size(block_size: usize) -> Self {
        assert!(
            block_size <= u32::MAX as usize,
            "a pool block holds at most u32::MAX bytes"
        );
        BytesPool {
            blocks: Vec::new(),
            buffers: Vec::new(),
            block_size: block_size.max(1),
            // No block yet: the first long value starts one.
            head: Box::new(PoolHead {
                cur: std::ptr::null_mut(),
                used: 0,
                cap: 0,
                buf_idx: 0,
                buffers: std::ptr::null(),
            }),
        }
    }

    /// The view of `bytes`: inline when short (nothing stored), else a copy in one block,
    /// addressed by `(buf_idx, offset)` into [`buffers`](Self::buffers). The same steps a
    /// kernel's [`StagedPool::copy_in`] takes.
    pub fn append(&mut self, bytes: &[u8]) -> u128 {
        if bytes.len() <= MAX_INLINE_LEN {
            return make_view(bytes, 0, 0);
        }
        let n = bytes.len();
        if self.head.used as usize + n > self.head.cap as usize {
            self.new_block(n);
        }
        let PoolHead {
            cur, used, buf_idx, ..
        } = *self.head;
        // SAFETY: `used + n <= cap` (a new block if not), so the bytes land in the
        // block's spare capacity, apart from `bytes`.
        unsafe {
            cur.add(used as usize)
                .copy_from_nonoverlapping(bytes.as_ptr(), n)
        };
        self.head.used = used + n as u64;
        self.settle();
        make_view(bytes, buf_idx as u32, used as u32)
    }

    /// Sets the current block's length to the bytes appended to it, and its table entry
    /// with it. A kernel's appends show in [`buffers`](Self::buffers) once this has run.
    pub fn settle(&mut self) {
        if let (Some(block), Some(buffer)) = (self.blocks.last_mut(), self.buffers.last_mut()) {
            let used = self.head.used as usize;
            assert!(
                used <= block.capacity(),
                "more appended to a pool block than it holds"
            );
            // SAFETY: bytes `0..used` were written by `append` or a kernel's `copy_in`,
            // each value end to end from the block's start, within its capacity.
            unsafe { block.set_len(used) };
            *buffer = ByteBuffer::new(block);
        }
    }

    /// Settles the current block and starts one with room for `len` bytes at least.
    fn new_block(&mut self, len: usize) {
        self.settle();
        let size = len.max(self.block_size);
        assert!(
            size <= u32::MAX as usize,
            "a value in a pool is shorter than 4 GiB: a view's offset is a u32"
        );
        let buf_idx = u32::try_from(self.blocks.len()).expect("fewer than 2^32 blocks");
        let mut block = Vec::with_capacity(size);
        self.buffers.push(ByteBuffer::new(&block));
        // The box stays where it is: kernels hold its address.
        *self.head = PoolHead {
            cur: block.as_mut_ptr(),
            used: 0,
            cap: size as u64,
            buf_idx: u64::from(buf_idx),
            buffers: self.buffers.as_ptr(),
        };
        self.blocks.push(block);
    }

    /// What a kernel takes to append to this pool through [`StagedPool`]. Valid while the
    /// pool stays where it is and nothing else appends to it.
    pub fn pool_ref(&mut self) -> PoolRef {
        PoolRef {
            head: &mut *self.head,
            owner: self,
        }
    }

    /// The table the pool's views index: one entry per block, current as of the last
    /// [`settle`](Self::settle).
    pub fn buffers(&self) -> &[ByteBuffer] {
        &self.buffers
    }

    /// `view` with the table it indexes.
    ///
    /// # Safety
    ///
    /// `view` is one this pool returned, and the value is read only while the pool is
    /// alive and nothing is appended to it.
    pub unsafe fn value(&self, view: u128) -> RawByteView {
        // SAFETY: forwarded from this function's contract.
        unsafe { RawByteView::new(view, self.buffers.as_ptr()) }
    }
}

/// What a kernel takes for a pool it appends to: the head, and the pool for the cold
/// extern. Made by [`BytesPool::pool_ref`].
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct PoolRef {
    #[staged(SMutPtr<PoolHead>)]
    head: *mut PoolHead,
    #[staged(SMutPtr<Opaque<BytesPool>>)]
    owner: *mut BytesPool,
}

/// Starts a block of `pool` with room for `len` bytes: the cold path of
/// [`StagedPool::copy_in`], taken when the current block is full.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn bytes_pool_new_block(pool: &mut BytesPool, len: u64) {
    pool.new_block(len as usize);
}

/// A kernel's handle on a [`BytesPool`]: copies long values into its current block, and
/// calls [`bytes_pool_new_block`] when one does not fit.
#[derive(Clone, Copy)]
pub struct StagedPool {
    head: Var<SMutPtr<PoolHead>>,
    owner: Var<SMutPtr<Opaque<BytesPool>>>,
    new_block: ExternRef<BytesPoolNewBlockExtern>,
}

impl StagedPool {
    /// The pool `pool` refers to.
    ///
    /// # Safety
    ///
    /// At execution, `pool` is [`BytesPool::pool_ref`] of a pool that stays where it is
    /// and that nothing else appends to during the call.
    pub unsafe fn new(
        ctx: &Ctx,
        pool: Var<SRef<PoolRef>>,
        new_block: ExternRef<BytesPoolNewBlockExtern>,
    ) -> Self {
        let pool = ref_as_ptr(pool);
        // SAFETY: both are plain fields of a live `PoolRef` (the caller).
        unsafe {
            Self {
                head: ctx.bind(load_field_unchecked(pool, PoolRefType::head())),
                owner: ctx.bind(load_field_unchecked(pool, PoolRefType::owner())),
                new_block,
            }
        }
    }

    /// `view`, one of this pool's, as a value of kind `K`: read through the pool's table as
    /// it is now, so a value appended earlier in the same call is readable.
    ///
    /// # Safety
    ///
    /// At execution, `view` came from this pool (an `append` or a `copy_in`), and for
    /// `Utf8` its bytes are UTF-8.
    pub unsafe fn value<K: ByteKind>(&self, ctx: &Ctx, view: Var<u128>) -> Var<ByteView<K>> {
        // SAFETY: a field of the live head (`new`'s contract); the view's block is an entry
        // of that table (the caller).
        unsafe {
            let buffers = ctx.bind(load_field_unchecked(self.head, PoolHeadType::buffers()));
            ctx.bind(ByteView::from_parts(view, buffers))
        }
    }

    /// `value`'s view in this pool: a short value's view as it is; a long value's bytes
    /// copied to the end of the current block (a new block when they do not fit) and its
    /// view pointed there. [`BytesPool::append`]'s steps, staged.
    pub fn copy_in<K: ByteKind>(&self, ctx: &Ctx, value: Var<ByteView<K>>) -> Var<u128> {
        let view = ctx.var(value.view());
        let n = ctx.bind(value.byte_len());
        let Self {
            head,
            owner,
            new_block,
        } = *self;
        // SAFETY (every head load below): a field of the live head (`new`'s contract).
        ctx.if_then(gt(n, MAX_INLINE_LEN as u64), move |ctx| {
            let used = ctx.bind(unsafe { load_field_unchecked(head, PoolHeadType::used()) });
            let cap = ctx.bind(unsafe { load_field_unchecked(head, PoolHeadType::cap()) });
            ctx.if_then(gt(add(used, n), cap), move |ctx| {
                // SAFETY: `owner` is the pool of this head, alive and appended to by this
                // kernel only (`new`'s contract); the extern rewrites the head, read again
                // below.
                ctx.emit(unsafe { call_extern2_unchecked(new_block, owner, n) });
            });
            let cur = ctx.bind(unsafe { load_field_unchecked(head, PoolHeadType::cur()) });
            let used = ctx.bind(unsafe { load_field_unchecked(head, PoolHeadType::used()) });
            let buf_idx = ctx.bind(unsafe { load_field_unchecked(head, PoolHeadType::buf_idx()) });
            // SAFETY: `used + n <= cap` after the check, so the bytes land in the block's
            // spare capacity; the value is long (this branch), so `long_data` is its bytes,
            // alive for the call (`ByteView`'s contract) and apart from the pool's.
            unsafe {
                let dst = ctx.bind(ptr_offset_mut(cur, int_cast::<i64, u64, _>(used)));
                copy_bytes(ctx, dst, value.long_data(ctx), n);
            }
            let relocated = relocate_view(
                view,
                int_cast::<u32, u64, _>(buf_idx),
                int_cast::<u32, u64, _>(used),
            );
            ctx.store(view, relocated);
            // SAFETY: `used` is a writable field of the live head.
            ctx.emit(unsafe { store(field_addr(head, PoolHeadType::used()), add(used, n)) });
        });
        view
    }
}

impl Default for BytesPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Byte strings a kernel bakes in as constants. A short one is inline and stored nowhere;
/// a long one is copied here with a one-entry table of its own, so adding a value never
/// moves another's and constants can be made while a kernel is staged. Whoever owns the
/// kernel keeps this alive.
#[derive(Default)]
pub struct ByteLiterals {
    /// Each long value's bytes and its table, boxed so they stay where they are.
    values: Vec<(Box<[u8]>, Box<ByteBuffer>)>,
}

// SAFETY: it owns its bytes and never changes a value once added; the pointers inside
// its tables point into those bytes.
unsafe impl Send for ByteLiterals {}
unsafe impl Sync for ByteLiterals {}

impl ByteLiterals {
    pub fn new() -> Self {
        Self::default()
    }

    /// `bytes` as a staged constant of kind `K`.
    ///
    /// # Safety
    ///
    /// Every kernel the constant is baked into runs only while `self` is alive, and for
    /// `Utf8` the bytes are valid UTF-8.
    pub unsafe fn constant<K: ByteKind>(&mut self, bytes: &[u8]) -> Const<ByteView<K>> {
        if let Some(inline) = RawByteView::inline(bytes) {
            return Const::new(inline);
        }
        let bytes: Box<[u8]> = bytes.into();
        let table = Box::new(ByteBuffer::new(&bytes));
        let table_ptr: *const ByteBuffer = &*table;
        let view = make_view(&bytes, 0, 0);
        self.values.push((bytes, table));
        // SAFETY: the view names all of the bytes boxed above, in its one-entry table,
        // and both stay put for as long as `self` lives (this function's contract).
        Const::new(unsafe { RawByteView::new(view, table_ptr) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The bytes of `view`, read through the pool's table.
    fn read(pool: &BytesPool, view: u128) -> Vec<u8> {
        // SAFETY: the pool is alive and unchanged during the read.
        unsafe { pool.value(view).bytes().to_vec() }
    }

    #[test]
    fn short_values_are_inline_and_store_nothing() {
        let mut pool = BytesPool::new();
        let view = pool.append(b"twelve bytes");
        assert!(pool.blocks.is_empty());
        assert_eq!(read(&pool, view), b"twelve bytes");
    }

    #[test]
    fn long_values_share_a_block_until_it_is_full() {
        let mut pool = BytesPool::with_block_size(32);
        let a = pool.append(b"thirteen byte");
        let b = pool.append(b"fourteen bytes");
        let c = pool.append(b"no room left in block 0");

        assert_eq!(pool.blocks.len(), 2);
        assert_eq!(pool.blocks[0], b"thirteen bytefourteen bytes");
        assert_eq!(pool.buffers().len(), 2);
        assert_eq!(pool.buffers()[0], ByteBuffer::new(&pool.blocks[0]));
        assert_eq!((b >> 64) as u32, 0, "b is in block 0");
        assert_eq!((b >> 96) as u32, 13, "after a");
        assert_eq!((c >> 64) as u32, 1, "c starts block 1");
        assert_eq!(read(&pool, a), b"thirteen byte");
        assert_eq!(read(&pool, b), b"fourteen bytes");
        assert_eq!(read(&pool, c), b"no room left in block 0");
    }

    #[test]
    fn blocks_never_move() {
        let mut pool = BytesPool::with_block_size(16);
        let first = pool.append(b"the first long value");
        let block = pool.blocks[0].as_ptr();
        for i in 0..100 {
            pool.append(format!("another long value {i}").as_bytes());
        }
        assert_eq!(pool.blocks[0].as_ptr(), block);
        assert_eq!(read(&pool, first), b"the first long value");
    }

    #[test]
    fn an_oversized_value_gets_its_own_block() {
        let mut pool = BytesPool::with_block_size(16);
        pool.append(b"a long-ish value");
        let big = vec![7u8; 100];
        let view = pool.append(&big);
        assert_eq!(pool.blocks.len(), 2);
        assert_eq!(pool.blocks[1].len(), 100);
        assert_eq!(read(&pool, view), big);
    }
}
