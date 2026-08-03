//! A host-owned, append-only byte pool — a bump arena with **stable pointers**.
//!
//! Scratch/output memory for JIT kernels that need to produce bytes at runtime
//! (string intermediates, materialized values) without the kernel ever freeing.
//! The *host* owns a [`BytesPool`], the kernel appends into it via the
//! [`pool_append`] extern, and nothing is released until the host drops the pool
//! — or hands its chunks off elsewhere (e.g. an Arrow `Buffer`) for zero-copy
//! recovery.
//!
//! **Why chunked, not one growing `Vec`:** a `Vec` that reallocates on growth
//! would dangle every pointer handed out earlier. [`BytesPool`] instead holds a
//! list of chunks, each pre-sized and **never reallocated** once created, so a
//! returned pointer stays valid for the pool's whole lifetime. Each appended
//! value is stored **contiguously in a single chunk** (a fresh chunk is started
//! if it wouldn't fit), which is also exactly what Arrow's `GenericByteView`
//! requires — a string may not span two buffers.
//!
//! **Lifetime contract (not expressed in the type system — it can't cross the
//! JIT boundary):** the pool must outlive the kernel call and any recovery step.
//! In a compile-and-run driver that holds the pool as a local across
//! build → run → finalize, this is upheld by construction — the same invariant
//! class as `Compiled` outliving its `as_fn` pointer.

use crate::ffi::FatSlice;
use rust_lms_derive::extern_fn;

/// Default chunk size (64 KiB) — large enough that most appends share a chunk,
/// small enough not to over-allocate for tiny pools.
const DEFAULT_CHUNK: usize = 64 * 1024;

/// An append-only bump arena of `u8` with stable pointers. See the [module
/// docs](self).
pub struct BytesPool {
    /// Each chunk is pre-sized to its capacity and never reallocated, so its
    /// `as_ptr()` stays stable for the pool's lifetime.
    chunks: Vec<Vec<u8>>,
    /// Capacity handed to each newly created chunk (an oversized value gets its
    /// own larger chunk instead).
    chunk_size: usize,
}

impl BytesPool {
    /// A pool with the default chunk size.
    pub fn new() -> Self {
        Self::with_chunk_size(DEFAULT_CHUNK)
    }

    /// A pool whose chunks are (at least) `chunk_size` bytes each.
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        BytesPool {
            chunks: Vec::new(),
            chunk_size: chunk_size.max(1),
        }
    }

    /// Copy `bytes` into the pool and return a **stable** pointer to the copy.
    /// The bytes are contiguous within one chunk; the pointer is valid for
    /// `bytes.len()` bytes until the pool is dropped.
    pub fn append(&mut self, bytes: &[u8]) -> *const u8 {
        let n = bytes.len();
        // Start a new chunk unless the current one has contiguous room for `n`.
        let need_new = match self.chunks.last() {
            Some(c) => c.len() + n > c.capacity(),
            None => true,
        };
        if need_new {
            self.chunks.push(Vec::with_capacity(n.max(self.chunk_size)));
        }
        let chunk = self.chunks.last_mut().unwrap();
        let start = chunk.len();
        chunk.extend_from_slice(bytes);
        // SAFETY: `need_new` guaranteed `capacity() >= len() + n`, so
        // `extend_from_slice` did not reallocate and `start` is in bounds.
        unsafe { chunk.as_ptr().add(start) }
    }

    /// The pool's chunks, for zero-copy handoff (e.g. adopting each as an Arrow
    /// `Buffer`). See also [`into_chunks`](Self::into_chunks).
    pub fn chunks(&self) -> &[Vec<u8>] {
        &self.chunks
    }

    /// Consume the pool, yielding ownership of its chunk buffers — hand these to
    /// a consumer that adopts `Vec<u8>` without copying.
    pub fn into_chunks(self) -> Vec<Vec<u8>> {
        self.chunks
    }
}

impl Default for BytesPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Append `bytes` into `pool`, returning a stable pointer (as `u64`) to the
/// stored copy — the kernel-facing entry point for [`BytesPool::append`].
#[extern_fn]
#[no_mangle]
pub extern "C" fn pool_append(pool: &mut BytesPool, bytes: FatSlice<u8>) -> u64 {
    // SAFETY: `bytes` is a valid `(ptr, len)` produced by staged code for the
    // duration of this call.
    let src = unsafe { bytes.as_slice() };
    pool.append(src) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn appends_contiguously_and_starts_new_chunks() {
        let mut pool = BytesPool::with_chunk_size(8);
        let p1 = pool.append(b"aaaa"); // chunk 0
        let p2 = pool.append(b"bbbb"); // fits (4+4 == 8), still chunk 0
        let p3 = pool.append(b"c"); // no room -> chunk 1

        assert_eq!(pool.chunks().len(), 2);
        assert_eq!(pool.chunks()[0], b"aaaabbbb");
        assert_eq!(pool.chunks()[1], b"c");

        // Pointers handed out earlier remain valid and correct.
        unsafe {
            assert_eq!(*p1, b'a');
            assert_eq!(*p2, b'b');
            assert_eq!(*p3, b'c');
            // p2 is 4 bytes past p1 within the same chunk.
            assert_eq!(p2, p1.add(4));
        }
    }

    #[test]
    fn oversized_value_gets_its_own_chunk_contiguously() {
        let mut pool = BytesPool::with_chunk_size(8);
        pool.append(b"x");
        let big = vec![7u8; 100]; // larger than chunk_size
        let pb = pool.append(&big);

        assert_eq!(pool.chunks().len(), 2);
        assert_eq!(pool.chunks()[1].len(), 100);
        // Stored contiguously — the whole value lives in one chunk.
        unsafe {
            assert_eq!(std::slice::from_raw_parts(pb, 100), &big[..]);
        }
    }
}
