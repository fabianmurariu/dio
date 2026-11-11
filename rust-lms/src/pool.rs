//! A host-owned, append-only pool of byte strings, addressed as Arrow **byte views**.
//!
//! [`BytesPool`] stores values the way Arrow's `GenericByteViewBuilder` does: a value of
//! at most [`MAX_INLINE_LEN`] bytes is its view and is not stored at all; a longer one is
//! copied into a **block** and its view is `(len, prefix, buf_idx, offset)` into the
//! pool's blocks. [`BytesPool::buffers`] is the table those views index, so a kernel
//! reads a pool's value as it reads a row of an Arrow array: a [`ByteView`] of the view
//! and the table.
//!
//! **Stable addresses.** A block is allocated at its full size and never reallocated,
//! and a value never spans two blocks (one that does not fit starts a new block; one
//! larger than the block size gets a block of its own), so the bytes a view names stay
//! where they are for the pool's lifetime. The *table* grows by one entry per block: its
//! address is stable only while nothing is appended. A plan appends its literals while it
//! is compiled, then bakes the table's address into the kernel.
//!
//! **Lifetime contract (not expressed in the type system; it cannot cross the JIT
//! boundary):** the pool outlives every kernel call that reads its views, and nothing is
//! appended while one runs.
//!
//! [`ByteView`]: crate::byte_view::ByteView

use crate::byte_view::{ByteBuffer, MAX_INLINE_LEN, RawByteView, make_view};

/// Default block size (64 KiB): large enough that most values share a block, small
/// enough not to over-allocate for a few literals.
const DEFAULT_BLOCK: usize = 64 * 1024;

/// An append-only pool of byte strings with stable addresses, its values Arrow views.
/// See the [module docs](self).
pub struct BytesPool {
    /// Each block is allocated at its capacity and never reallocated.
    blocks: Vec<Vec<u8>>,
    /// One entry per block, its length kept current: the table the views index.
    buffers: Vec<ByteBuffer>,
    /// The capacity of a new block (a larger value gets a block of its own size).
    block_size: usize,
}

impl BytesPool {
    /// A pool with the default block size.
    pub fn new() -> Self {
        Self::with_block_size(DEFAULT_BLOCK)
    }

    /// A pool whose blocks are (at least) `block_size` bytes each.
    pub fn with_block_size(block_size: usize) -> Self {
        BytesPool {
            blocks: Vec::new(),
            buffers: Vec::new(),
            block_size: block_size.max(1),
        }
    }

    /// The view of `bytes`: inline when short (nothing stored), else a copy in one block,
    /// addressed by `(buf_idx, offset)` into [`buffers`](Self::buffers).
    pub fn append(&mut self, bytes: &[u8]) -> u128 {
        if bytes.len() <= MAX_INLINE_LEN {
            return make_view(bytes, 0, 0);
        }
        let n = bytes.len();
        let fits = self
            .blocks
            .last()
            .is_some_and(|block| block.len() + n <= block.capacity());
        if !fits {
            self.blocks.push(Vec::with_capacity(n.max(self.block_size)));
            self.buffers.push(ByteBuffer::new(&[]));
        }
        let buf_idx = self.blocks.len() - 1;
        let block = &mut self.blocks[buf_idx];
        let offset = block.len();
        // No reallocation: `fits` (or the fresh block) leaves room for `n` more bytes.
        block.extend_from_slice(bytes);
        self.buffers[buf_idx] = ByteBuffer::new(block);
        let buf_idx = u32::try_from(buf_idx).expect("fewer than 2^32 blocks");
        let offset = u32::try_from(offset).expect("a block is smaller than 4 GiB");
        make_view(bytes, buf_idx, offset)
    }

    /// The table the pool's views index: one entry per block.
    pub fn buffers(&self) -> &[ByteBuffer] {
        &self.buffers
    }

    /// `view`, one of this pool's, with the table it indexes.
    pub fn value(&self, view: u128) -> RawByteView {
        RawByteView::new(view, self.buffers.as_ptr())
    }

    /// The blocks, for handing them on without copying.
    pub fn blocks(&self) -> &[Vec<u8>] {
        &self.blocks
    }

    /// Consume the pool, yielding its blocks.
    pub fn into_blocks(self) -> Vec<Vec<u8>> {
        self.blocks
    }
}

impl Default for BytesPool {
    fn default() -> Self {
        Self::new()
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
        assert!(pool.blocks().is_empty());
        assert_eq!(read(&pool, view), b"twelve bytes");
    }

    #[test]
    fn long_values_share_a_block_until_it_is_full() {
        let mut pool = BytesPool::with_block_size(32);
        let a = pool.append(b"thirteen byte");
        let b = pool.append(b"fourteen bytes");
        let c = pool.append(b"no room left in block 0");

        assert_eq!(pool.blocks().len(), 2);
        assert_eq!(pool.blocks()[0], b"thirteen bytefourteen bytes");
        assert_eq!(pool.buffers().len(), 2);
        assert_eq!(pool.buffers()[0], ByteBuffer::new(&pool.blocks()[0]));
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
        let block = pool.blocks()[0].as_ptr();
        for i in 0..100 {
            pool.append(format!("another long value {i}").as_bytes());
        }
        assert_eq!(pool.blocks()[0].as_ptr(), block);
        assert_eq!(read(&pool, first), b"the first long value");
    }

    #[test]
    fn an_oversized_value_gets_its_own_block() {
        let mut pool = BytesPool::with_block_size(16);
        pool.append(b"a long-ish value");
        let big = vec![7u8; 100];
        let view = pool.append(&big);
        assert_eq!(pool.blocks().len(), 2);
        assert_eq!(pool.blocks()[1].len(), 100);
        assert_eq!(read(&pool, view), big);
    }
}
