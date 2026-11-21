//! `hash_bytes`: a string's hash is its bytes', wherever they are stored, and a long
//! string's covers every byte, not only the length and prefix its view holds.

use std::collections::HashMap;

use rust_lms::prelude::*;
use rust_lms_std::hash::{SEED, hash_bytes};

mod common;
use common::for_each_backend;

/// Every length class, and strings a view alone cannot tell apart: long ones of one
/// length and prefix, differing in the middle, in the last byte, or in the order of
/// their words.
const STRINGS: &[&str] = &[
    "",
    "a",
    "b",
    "abcd",
    "abcde",
    "twelve bytes",
    "twelve bytez",
    "thirteen byte",
    "thirteen bytes",
    "host-0001.example.com/a/very/long/path",
    "host-0001.example.com/a/vary/long/path",
    "host-0001.example.com/a/very/long/patz",
    "abcdefgh12345678",
    "12345678abcdefgh",
    "abcd12345678efgh",
    "abcd\0\0\0\0",
    "abcd\0\0\0\0\0\0\0\0\0",
];

/// Each of `STRINGS`' hashes, its values stored in a pool after `filler` bytes.
fn hashes(filler: usize) -> Vec<Vec<u64>> {
    let mut pool = BytesPool::with_block_size(48);
    pool.append(&vec![b'#'; filler]);
    let views: Vec<u128> = STRINGS.iter().map(|s| pool.append(s.as_bytes())).collect();
    let mut per_backend = Vec::new();
    for_each_backend(|mut compiler| {
        let fun = compiler.fun1("hash", |ctx, s: Var<Str>| hash_bytes(ctx, SEED, s));
        let compiled = compiler.compile(fun).expect("compile");
        let fun = compiled.as_fn();
        // SAFETY: views of `pool`, alive and not appended to while the kernel runs.
        per_backend.push(
            views
                .iter()
                .map(|&view| fun.call(unsafe { pool.value(view) }))
                .collect(),
        );
    });
    per_backend
}

#[test]
fn equal_bytes_hash_alike_wherever_they_are_stored() {
    let (here, there) = (hashes(0), hashes(17));
    assert_eq!(here, there);
    // And on every backend.
    assert!(here.windows(2).all(|w| w[0] == w[1]));
}

#[test]
fn different_bytes_hash_apart() {
    for hashes in hashes(0) {
        let mut seen = HashMap::new();
        for (s, h) in STRINGS.iter().zip(hashes) {
            if let Some(other) = seen.insert(h, s) {
                panic!("{s:?} and {other:?} hash alike");
            }
        }
    }
}
