//! A host-owned, append-only byte arena with **stable pointers**.
//!
//! Interned string literals and copied GROUP BY string keys live here: each value is
//! copied contiguously into one chunk, chunks are allocated at their full size and never
//! reallocated, so a returned pointer stays valid for the arena's lifetime. Nothing is
//! freed until the arena is dropped.
//!
//! **Lifetime contract (not expressed in the type system; it cannot cross the JIT
//! boundary):** the arena outlives every kernel call that reads a pointer from it.

/// Default chunk size (64 KiB).
const DEFAULT_CHUNK: usize = 64 * 1024;

/// An append-only arena of bytes with stable pointers. See the [module docs](self).
pub struct ByteArena {
    /// Each chunk is allocated at its capacity and never reallocated.
    chunks: Vec<Vec<u8>>,
    chunk_size: usize,
}

impl ByteArena {
    /// An arena with the default chunk size.
    pub fn new() -> Self {
        ByteArena {
            chunks: Vec::new(),
            chunk_size: DEFAULT_CHUNK,
        }
    }

    /// Copy `bytes` into the arena and return a stable pointer to the copy, valid for
    /// `bytes.len()` bytes until the arena is dropped.
    pub fn append(&mut self, bytes: &[u8]) -> *const u8 {
        let n = bytes.len();
        let fits = self
            .chunks
            .last()
            .is_some_and(|chunk| chunk.len() + n <= chunk.capacity());
        if !fits {
            self.chunks.push(Vec::with_capacity(n.max(self.chunk_size)));
        }
        let chunk = self.chunks.last_mut().expect("a chunk with room");
        let start = chunk.len();
        chunk.extend_from_slice(bytes);
        // SAFETY: the chunk had room for `n` more bytes, so it did not reallocate and
        // `start` is in bounds.
        unsafe { chunk.as_ptr().add(start) }
    }
}

impl Default for ByteArena {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointers_stay_valid_across_chunks() {
        let mut arena = ByteArena {
            chunks: Vec::new(),
            chunk_size: 8,
        };
        let p1 = arena.append(b"aaaa");
        let p2 = arena.append(b"bbbb");
        let p3 = arena.append(b"c");
        assert_eq!(arena.chunks.len(), 2);
        unsafe {
            assert_eq!(*p1, b'a');
            assert_eq!(*p2, b'b');
            assert_eq!(*p3, b'c');
            assert_eq!(p2, p1.add(4));
        }
    }
}
