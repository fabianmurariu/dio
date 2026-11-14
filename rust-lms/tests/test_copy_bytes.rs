//! `copy_bytes` (a copy of a run-time length, through `write_unaligned`) and
//! `relocate_view`, the pieces a kernel needs to copy a long byte string into a pool of
//! its own.

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

/// Copies `len` bytes from `src + from` to `dst + to`, and nothing else: every length
/// around the 8-byte word, at unaligned offsets on both sides.
#[test]
fn copy_bytes_copies_exactly_len_bytes_at_any_offset() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun3(
            "copy",
            |ctx, dst: Var<SMutPtr<u8>>, src: Var<SPtr<u8>>, len: Var<u64>| {
                // SAFETY: the host passes `len` readable and writable bytes, apart.
                unsafe { copy_bytes(ctx, dst, src, len) };
                unit()
            },
        );
        let compiled = compiler.compile(f).expect("compile");
        let src: Vec<u8> = (1..=64).collect();
        for len in 0..=40usize {
            for (from, to) in [(0, 0), (1, 3), (5, 7), (3, 0)] {
                let mut dst = vec![0u8; 64];
                compiled
                    .as_fn()
                    .call(dst[to..].as_mut_ptr(), src[from..].as_ptr(), len as u64);
                let mut expect = vec![0u8; 64];
                expect[to..to + len].copy_from_slice(&src[from..from + len]);
                assert_eq!(dst, expect, "len {len}, from {from}, to {to}");
            }
        }
    });
}

/// A long view moved to `(buf_idx, offset)` is `make_view` of its bytes there.
#[test]
fn relocate_view_is_make_view_at_the_new_address() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun3(
            "relocate",
            |_ctx, view: Var<u128>, buf_idx: Var<u32>, offset: Var<u32>| {
                relocate_view(view, buf_idx, offset)
            },
        );
        let compiled = compiler.compile(f).expect("compile");
        for bytes in [
            &b"thirteen byte"[..],
            b"a much longer string \xff\x00 of bytes",
        ] {
            for (buf_idx, offset) in [(0, 0), (1, 13), (u32::MAX, u32::MAX)] {
                let old = make_view(bytes, 7, 99);
                assert_eq!(
                    compiled.as_fn().call(old, buf_idx, offset),
                    make_view(bytes, buf_idx, offset),
                );
            }
        }
    });
}
