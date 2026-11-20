//! `ChainedTable`: generated code finds and inserts entries, the host adds chunks and
//! doubles the directory, and the kernel reads every entry back in insertion order. Each
//! test counts keys in a kernel and checks the counts against Rust's.

use std::collections::HashMap;

use rust_lms::prelude::*;
use rust_lms_std::{
    ChainedTable, ChainedTableMakeRoomExtern, EntryLayout, FieldId, StagedTable, TableRef,
    hash::{SEED, hash_word},
};

mod common;
use common::for_each_backend;

/// A table of `key: u64 → count: u64`.
struct Counts {
    layout: EntryLayout,
    key: FieldId<u64>,
    count: FieldId<u64>,
}

impl Counts {
    fn new() -> Self {
        let mut layout = EntryLayout::new();
        let key = layout.field::<u64>();
        let count = layout.field::<u64>();
        Self { layout, key, count }
    }

    /// Counts `keys` in a kernel, hashing each with `hash`, then writes each entry's key
    /// and count, in insertion order, and returns them.
    fn run(&self, keys: &[u64], hash: fn(&Ctx, Var<u64>) -> Var<u64>) -> Vec<(u64, u64)> {
        let mut result = None;
        for_each_backend(|mut compiler| {
            let make_room = compiler.extern_fn::<ChainedTableMakeRoomExtern>();
            let Counts { layout, key, count } = *self;
            let kernel = compiler.fun3(
                "count",
                move |ctx,
                      keys: Var<SRef<Slice<u64>>>,
                      mut out: Var<SRefMut<Slice<u64>>>,
                      table: Var<SRef<TableRef>>| {
                    // SAFETY: the test's table, built from `layout`, used by this call only.
                    let table = unsafe { StagedTable::new(ctx, &layout, table, make_room) };
                    keys.staged_iter().for_each(ctx, |ctx, k| {
                        let e = table.find_or_insert(
                            ctx,
                            hash(ctx, k),
                            |ctx, e| ctx.bind(eq(e.get(ctx, key), k)),
                            |ctx, e| {
                                e.set(ctx, key, k);
                                e.set(ctx, count, ctx.var(0u64));
                            },
                        );
                        e.set(ctx, count, ctx.bind(add(e.get(ctx, count), 1u64)));
                    });
                    let at = ctx.var(0u64);
                    table.for_each(ctx, |ctx, e| {
                        // SAFETY: `out` has two slots per distinct key.
                        unsafe {
                            ctx.emit(out.reborrow().set_unchecked(at, e.get(ctx, key)));
                            ctx.emit(
                                out.reborrow()
                                    .set_unchecked(add(at, 1u64), e.get(ctx, count)),
                            );
                        }
                        ctx.store(at, add(at, 2u64));
                    });
                    table.len(ctx)
                },
            );
            let compiled = compiler.compile(kernel).expect("compile");
            let mut table = ChainedTable::new(&self.layout);
            let mut out = vec![0u64; 2 * keys.len()];
            let len = compiled.as_fn().call(keys, &mut out, &table.table_ref());
            assert_eq!(len as usize, table.len());
            let pairs: Vec<(u64, u64)> = out[..2 * table.len()]
                .chunks(2)
                .map(|p| (p[0], p[1]))
                .collect();
            if let Some(previous) = &result {
                assert_eq!(&pairs, previous, "the backends agree");
            }
            result = Some(pairs);
        });
        result.unwrap()
    }
}

/// Rust's answer: each distinct key, in first-seen order, with its count.
fn expected(keys: &[u64]) -> Vec<(u64, u64)> {
    let mut counts: HashMap<u64, u64> = HashMap::new();
    let mut order = Vec::new();
    for &k in keys {
        *counts.entry(k).or_insert_with(|| {
            order.push(k);
            0
        }) += 1;
    }
    order.into_iter().map(|k| (k, counts[&k])).collect()
}

fn mixed(ctx: &Ctx, k: Var<u64>) -> Var<u64> {
    hash_word(ctx, SEED, k)
}

/// 700 keys over 5000 rows: three chunks of entries and two doublings of the directory
/// (256 → 512 → 1024), with every row finding its key again across them.
#[test]
fn counts_every_key_across_chunks_and_doublings() {
    let keys: Vec<u64> = (0..5000u64).map(|i| (i * 7919) % 700).collect();
    assert_eq!(Counts::new().run(&keys, mixed), expected(&keys));
}

/// Every key has the same hash, so all are one chain: the key equality alone tells them
/// apart.
#[test]
fn equal_hashes_are_told_apart_by_the_key() {
    let keys: Vec<u64> = (0..600u64).map(|i| i % 300).collect();
    let same = |ctx: &Ctx, _: Var<u64>| ctx.var(42u64);
    assert_eq!(Counts::new().run(&keys, same), expected(&keys));
}

/// Hashes that differ only above the directory's mask land in one head until the
/// directory grows past them, then split: relinking must keep every entry findable.
#[test]
fn high_hash_bits_split_chains_when_the_directory_doubles() {
    let keys: Vec<u64> = (0..3000u64).map(|i| i % 1000).collect();
    let high = |ctx: &Ctx, k: Var<u64>| ctx.bind(shl(k, 8u64));
    assert_eq!(Counts::new().run(&keys, high), expected(&keys));
}

/// An entry found before the table grows is still the entry after: a kernel may hold one
/// across any number of inserts.
#[test]
fn an_entry_held_across_growth_stays_put() {
    let Counts { layout, key, count } = Counts::new();
    for_each_backend(|mut compiler| {
        let make_room = compiler.extern_fn::<ChainedTableMakeRoomExtern>();
        let kernel = compiler.fun2(
            "hold",
            move |ctx, n: Var<u64>, table: Var<SRef<TableRef>>| {
                // SAFETY: the test's table, built from `layout`, used by this call only.
                let table = unsafe { StagedTable::new(ctx, &layout, table, make_room) };
                let insert = |ctx: &Ctx, k: Var<u64>| {
                    table.find_or_insert(
                        ctx,
                        mixed(ctx, k),
                        |ctx, e| ctx.bind(eq(e.get(ctx, key), k)),
                        |ctx, e| {
                            e.set(ctx, key, k);
                            e.set(ctx, count, ctx.var(1u64));
                        },
                    )
                };
                let first = insert(ctx, ctx.var(0u64));
                range(1u64, n).for_each(ctx, |ctx, k| {
                    insert(ctx, k);
                });
                first.set(ctx, count, ctx.bind(add(first.get(ctx, count), 100u64)));
                // The first entry, found again, has the count written through `first`.
                let again = insert(ctx, ctx.var(0u64));
                again.get(ctx, count)
            },
        );
        let compiled = compiler.compile(kernel).expect("compile");
        let mut table = ChainedTable::new(&layout);
        assert_eq!(compiled.as_fn().call(2000, &table.table_ref()), 101);
        assert_eq!(table.len(), 2000);
    });
}

/// String keys: each new key's bytes are copied into a pool, equality compares bytes (long
/// keys share their length and first bytes, and differ later), and the stored views read
/// back as the keys.
#[test]
fn string_keys_live_in_a_pool() {
    let words: Vec<String> = (0..40)
        .map(|i| match i % 2 {
            0 => format!("k{i}"),
            _ => format!("a long key that differs only at the end: {i:03}"),
        })
        .collect();
    let rows: Vec<&str> = (0..400).map(|i| words[(i * 13) % 40].as_str()).collect();
    let mut source = BytesPool::new();
    let views: Vec<u128> = rows.iter().map(|r| source.append(r.as_bytes())).collect();
    let source_table = source.buffers().as_ptr();

    let mut layout = EntryLayout::new();
    let text = layout.field::<u128>();
    let count = layout.field::<u64>();
    for_each_backend(|mut compiler| {
        let make_room = compiler.extern_fn::<ChainedTableMakeRoomExtern>();
        let new_block = compiler.extern_fn::<BytesPoolNewBlockExtern>();
        let kernel = compiler.fun4(
            "words",
            move |ctx,
                  rows: Var<SRef<Slice<u128>>>,
                  mut out: Var<SRefMut<Slice<u128>>>,
                  table: Var<SRef<TableRef>>,
                  pool: Var<SRef<PoolRef>>| {
                // SAFETY: the test's table and pool, each used by this call only.
                let (table, pool) = unsafe {
                    (
                        StagedTable::new(ctx, &layout, table, make_room),
                        StagedPool::new(ctx, pool, new_block),
                    )
                };
                let source_table = ctx.bind(const_ptr::<ByteBuffer>(source_table));
                rows.staged_iter().for_each(ctx, |ctx, view| {
                    // SAFETY: a view of `source`, alive for the call.
                    let s = ctx.bind(unsafe { ByteView::<Utf8>::from_parts(view, source_table) });
                    // The length and first four bytes: long keys collide on purpose.
                    let hash = hash_word(ctx, SEED, int_cast::<u64, u128, _>(view));
                    let e = table.find_or_insert(
                        ctx,
                        hash,
                        // SAFETY: an entry's view was copied into `pool`.
                        |ctx, e| {
                            unsafe { pool.value::<Utf8>(ctx, e.get(ctx, text)) }.bytes_eq(ctx, s)
                        },
                        |ctx, e| {
                            e.set(ctx, text, pool.copy_in(ctx, s));
                            e.set(ctx, count, ctx.var(0u64));
                        },
                    );
                    e.set(ctx, count, ctx.bind(add(e.get(ctx, count), 1u64)));
                });
                let at = ctx.var(0u64);
                table.for_each(ctx, |ctx, e| {
                    // SAFETY: `out` has two slots per distinct key.
                    unsafe {
                        ctx.emit(out.reborrow().set_unchecked(at, e.get(ctx, text)));
                        let n = int_cast::<u128, u64, _>(e.get(ctx, count));
                        ctx.emit(out.reborrow().set_unchecked(add(at, 1u64), n));
                    }
                    ctx.store(at, add(at, 2u64));
                });
                table.len(ctx)
            },
        );
        let compiled = compiler.compile(kernel).expect("compile");
        let mut table = ChainedTable::new(&layout);
        let mut pool = BytesPool::with_block_size(128);
        let mut out = vec![0u128; 2 * rows.len()];
        let pool_ref = pool.pool_ref();
        let len = compiled
            .as_fn()
            .call(&views, &mut out, &table.table_ref(), &pool_ref);
        pool.settle();

        let mut counts: HashMap<&str, u64> = HashMap::new();
        for r in &rows {
            *counts.entry(r).or_default() += 1;
        }
        assert_eq!(len as usize, counts.len());
        for pair in out[..2 * counts.len()].chunks(2) {
            // SAFETY: a view of `pool`, settled and not appended to while read.
            let key = unsafe { pool.value(pair[0]).bytes().to_vec() };
            let key = String::from_utf8(key).unwrap();
            assert_eq!(pair[1] as u64, counts[key.as_str()], "{key}");
        }
    });
}

/// `hash_word` is the folded multiply of the hash and word xor-ed, by the spreading
/// constant: checked against the same arithmetic on the host.
#[test]
fn hash_word_is_the_folded_multiply() {
    for_each_backend(|mut compiler| {
        let kernel = compiler.fun2("hash", |ctx, h: Var<u64>, w: Var<u64>| hash_word(ctx, h, w));
        let compiled = compiler.compile(kernel).expect("compile");
        let hash = compiled.as_fn();
        for (h, w) in [
            (SEED, 0),
            (SEED, 1),
            (0, u64::MAX),
            (u64::MAX, 12345),
            (7, 7),
        ] {
            let product = u128::from(h ^ w) * 0x9e37_79b9_7f4a_7c15u128;
            assert_eq!(hash.call(h, w), (product as u64) ^ ((product >> 64) as u64));
        }
    });
}
