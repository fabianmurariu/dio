//! **Group-by aggregation: a generated `ChainedTable` vs a host hashbrown index.**
//!
//! Every variant computes `SELECT key, count(*), sum(value) GROUP BY key` over the same
//! rows, then folds its groups into one checksum (so each variant also reads every group
//! back once, as a real query would). A fresh table per run: allocation and growth are
//! part of the cost.
//!
//! Two keys:
//!
//! - `u64`: one random `u64` per group;
//! - `u64_str`: a `u64` and a string per group, half of them inline (≤ 12 bytes) and half
//!   long. Each row's string is its own copy in the input's pool, as in a real block, so
//!   a long string's view differs from row to row and equality reads bytes.
//!
//! | variant     | what runs                                                                 |
//! |-------------|---------------------------------------------------------------------------|
//! | `native`    | plain Rust over a `hashbrown::HashMap` (the ceiling, not staged)          |
//! | `chained`   | a kernel probing and filling a [`ChainedTable`]: no call per row          |
//! | `ffi_index` | a kernel calling one extern per row: a host `HashTable<u32>` maps the key |
//! |             | to an index into a vector of fixed-size records, the extern returns the   |
//! |             | record's address, and the kernel folds count and sum into it               |
//!
//! Keys are fixed size: a string key is its `u128` view, and only a new group copies a
//! long string's bytes into the table's pool, so no row allocates. Both tables hash the
//! same words (the `u64` and the view's length and prefix) and compare long strings by
//! content.
//!
//! Run:
//!   cargo bench -p rust-lms-std --bench group_count                  # Cranelift
//!   cargo bench -p rust-lms-std --bench group_count --features llvm  # + LLVM

use std::{hash::BuildHasher, hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use hashbrown::{DefaultHashBuilder, HashTable};
use rand::{Rng, SeedableRng, rngs::StdRng};
use rust_lms::{byte_view::MAX_INLINE_LEN, prelude::*};
use rust_lms_std::{
    ChainedTable, ChainedTableInsertExtern, EntryLayout, StagedTable, TableRef,
    hash::{SEED, hash_word},
};

const ROWS: usize = 16 << 20;

// =============================================================================
// Rows
// =============================================================================

/// `ROWS` rows over `distinct` groups: each row's group key (a random `u64`), its
/// group's string copied into `pool`, and a random value.
struct Rows {
    keys: Vec<u64>,
    views: Vec<u128>,
    values: Vec<u64>,
    pool: BytesPool,
}

impl Rows {
    fn new(distinct: usize, with_strings: bool, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let domain: Vec<u64> = (0..distinct).map(|_| rng.random()).collect();
        let mut pool = BytesPool::new();
        let (mut keys, mut views, mut values) = (vec![], vec![], vec![]);
        for _ in 0..ROWS {
            let g = rng.random_range(0..distinct);
            keys.push(domain[g]);
            if with_strings {
                views.push(pool.append(group_string(g, domain[g]).as_bytes()));
            }
            values.push(rng.random_range(0..1000));
        }
        Rows {
            keys,
            views,
            values,
            pool,
        }
    }

    /// Row `i`'s string, as a hashable key: an inline view stands for its bytes.
    fn str(&self, i: usize) -> Str<'_> {
        let view = self.views[i];
        match view as u32 as usize > MAX_INLINE_LEN {
            true => Str::Long(long_bytes(&self.pool, view)),
            false => Str::Inline(view),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
enum Str<'a> {
    Inline(u128),
    Long(&'a [u8]),
}

/// The bytes of a long `view` of `pool`.
fn long_bytes(pool: &BytesPool, view: u128) -> &[u8] {
    let (len, buf, offset) = (
        view as u32 as usize,
        (view >> 64) as u32 as usize,
        (view >> 96) as u32 as usize,
    );
    // SAFETY: a long view `pool` returned, its buffer alive while `pool` is.
    unsafe { &pool.buffers()[buf].as_slice()[offset..offset + len] }
}

/// Group `g`'s string: inline for even groups, long (28 bytes) for odd ones.
fn group_string(g: usize, key: u64) -> String {
    match g % 2 {
        0 => format!("g{:x}", key >> 24),
        _ => format!("{key:016x}-a-long-key"),
    }
}

/// One group folded into the checksum; every variant folds the same way.
fn fold(acc: u64, key: u64, count: u64, sum: u64) -> u64 {
    acc.wrapping_add(key ^ count).wrapping_add(sum)
}

// =============================================================================
// native
// =============================================================================

fn native_u64(rows: &Rows) -> u64 {
    let mut map: hashbrown::HashMap<u64, (u64, u64)> = hashbrown::HashMap::new();
    for (&k, &v) in rows.keys.iter().zip(&rows.values) {
        let e = map.entry(k).or_insert((0, 0));
        e.0 += 1;
        e.1 += v;
    }
    map.iter().fold(0, |acc, (&k, &(c, s))| fold(acc, k, c, s))
}

fn native_u64_str(rows: &Rows) -> u64 {
    let mut map: hashbrown::HashMap<(u64, Str), (u64, u64)> = hashbrown::HashMap::new();
    for i in 0..ROWS {
        let e = map.entry((rows.keys[i], rows.str(i))).or_insert((0, 0));
        e.0 += 1;
        e.1 += rows.values[i];
    }
    map.iter()
        .fold(0, |acc, (&(k, _), &(c, s))| fold(acc, k, c, s))
}

// =============================================================================
// ffi_index: a host HashTable<u32> of indices into fixed-size records
// =============================================================================

/// Records `[key, count, sum]`; the table holds indices into them.
pub struct U64Index {
    hasher: DefaultHashBuilder,
    table: HashTable<u32>,
    records: Vec<[u64; 3]>,
}

/// The address of `key`'s count (then its sum), its record appended when new.
#[extern_fn]
pub extern "C" fn u64_index_find(index: &mut U64Index, key: u64) -> *mut u64 {
    let U64Index {
        hasher,
        table,
        records,
    } = index;
    let hash = hasher.hash_one(key);
    let at = match table.find(hash, |&i| records[i as usize][0] == key) {
        Some(&i) => i as usize,
        None => {
            let i = records.len();
            records.push([key, 0, 0]);
            table.insert_unique(hash, i as u32, |&i| hasher.hash_one(records[i as usize][0]));
            i
        }
    };
    records[at][1..].as_mut_ptr()
}

/// Records `[key, view lo, view hi, count, sum]` (the view of the table's own copy of a
/// long string), with each record's hash beside it for regrowth; the table holds indices.
pub struct StrIndex {
    hasher: DefaultHashBuilder,
    table: HashTable<u32>,
    records: Vec<[u64; 5]>,
    hashes: Vec<u64>,
    pool: BytesPool,
    /// The input's pool, which probe keys' long views index.
    input: *const BytesPool,
}

/// The words a `(key, string)` is hashed by: the key, the view's length and prefix.
fn str_key_hash(hasher: &DefaultHashBuilder, key: u64, view: u128) -> u64 {
    hasher.hash_one((key, view as u64))
}

/// The address of `(key, view)`'s count (then its sum), its record appended when new.
///
/// # Safety
///
/// A long `view` is one of the input pool's, which is alive and no longer appended to.
#[extern_fn]
pub unsafe extern "C" fn str_index_find(
    index: &mut StrIndex,
    key: u64,
    lo: u64,
    hi: u64,
) -> *mut u64 {
    let view = (u128::from(hi) << 64) | u128::from(lo);
    let StrIndex {
        hasher,
        table,
        records,
        hashes,
        pool,
        input,
    } = index;
    let hash = str_key_hash(hasher, key, view);
    let long = view as u32 as usize > MAX_INLINE_LEN;
    // SAFETY: the run's input pool, alive and no longer appended to; the view is one of
    // its own.
    let probe = || long_bytes(unsafe { &**input }, view);
    let found = table.find(hash, |&i| {
        let r = &records[i as usize];
        if r[0] != key || r[1] != lo {
            return false;
        }
        if !long {
            return r[2] == hi;
        }
        let stored = (u128::from(r[2]) << 64) | u128::from(r[1]);
        long_bytes(pool, stored) == probe()
    });
    let at = match found {
        Some(&i) => i as usize,
        None => {
            let stored = if long { pool.append(probe()) } else { view };
            let i = records.len();
            records.push([key, stored as u64, (stored >> 64) as u64, 0, 0]);
            hashes.push(hash);
            table.insert_unique(hash, i as u32, |&i| hashes[i as usize]);
            i
        }
    };
    records[at][3..].as_mut_ptr()
}

// =============================================================================
// Kernels
// =============================================================================

type U64s = Var<SRef<Slice<u64>>>;
type Views = Var<SRef<Slice<u128>>>;
type Kernel = Box<dyn Fn(&Rows) -> u64>;

/// Folds a row's value into the `[count, sum]` at `p`.
fn fold_into(ctx: &Ctx, p: Var<SMutPtr<u64>>, v: Var<u64>) {
    // SAFETY: `p` is a live record's count, its sum next to it.
    unsafe {
        let sum = ctx.bind(ptr_offset_mut(p, ctx.var(1i64)));
        ctx.emit(store(p, add(load_mut(p), 1u64)));
        ctx.emit(store(sum, add(load_mut(sum), v)));
    }
}

fn chained_u64(backend: JitBackend) -> Kernel {
    let mut layout = EntryLayout::new();
    let key = layout.field::<u64>();
    let count = layout.field::<u64>();
    let sum = layout.field::<u64>();
    let mut compiler = Compiler::new().with_backend(backend);
    let insert = compiler.extern_fn::<ChainedTableInsertExtern>();
    let f = compiler.fun3(
        "chained_u64",
        move |ctx, keys: U64s, values: U64s, table: Var<SRef<TableRef>>| {
            // SAFETY: the run's own table, built from `layout`.
            let table = unsafe { StagedTable::new(ctx, &layout, table, insert) };
            keys.staged_iter().zip(values).for_each(ctx, |ctx, row| {
                let (k, v) = (ctx.bind(row.first()), ctx.bind(row.second()));
                let e = table.find_or_insert(
                    ctx,
                    hash_word(ctx, SEED, k),
                    |ctx, e| ctx.bind(eq(e.get(ctx, key), k)),
                    |ctx, e| e.set(ctx, key, k),
                );
                e.set(ctx, count, ctx.bind(add(e.get(ctx, count), 1u64)));
                e.set(ctx, sum, ctx.bind(add(e.get(ctx, sum), v)));
            });
            let acc = ctx.var(0u64);
            table.for_each(ctx, |ctx, e| {
                let mixed = ctx.bind(bitxor(e.get(ctx, key), e.get(ctx, count)));
                ctx.store(acc, add(add(acc, mixed), e.get(ctx, sum)));
            });
            acc
        },
    );
    let k = compiler.compile(f).expect("compile");
    Box::new(move |rows| {
        let mut table = ChainedTable::new(&layout);
        k.as_fn().call(&rows.keys, &rows.values, &table.table_ref())
    })
}

fn chained_u64_str(backend: JitBackend) -> Kernel {
    let mut layout = EntryLayout::new();
    let key = layout.field::<u64>();
    let text = layout.field::<u128>();
    let count = layout.field::<u64>();
    let sum = layout.field::<u64>();
    let mut compiler = Compiler::new().with_backend(backend);
    let insert = compiler.extern_fn::<ChainedTableInsertExtern>();
    let new_block = compiler.extern_fn::<BytesPoolNewBlockExtern>();
    let f = compiler.fun6(
        "chained_u64_str",
        move |ctx,
              keys: U64s,
              views: Views,
              values: U64s,
              input: Var<SPtr<ByteBuffer>>,
              table: Var<SRef<TableRef>>,
              pool: Var<SRef<PoolRef>>| {
            // SAFETY: the run's own table and pool.
            let (table, pool) = unsafe {
                (
                    StagedTable::new(ctx, &layout, table, insert),
                    StagedPool::new(ctx, pool, new_block),
                )
            };
            let n = ctx.bind(keys.len());
            range(0u64, n).for_each(ctx, |ctx, i| {
                // SAFETY: `i < n`, and every column has `n` rows; views index `input`.
                let (k, view, v, s) = unsafe {
                    let view = ctx.bind(views.get_unchecked(i));
                    (
                        ctx.bind(keys.get_unchecked(i)),
                        view,
                        ctx.bind(values.get_unchecked(i)),
                        ctx.bind(ByteView::<Utf8>::from_parts(view, input)),
                    )
                };
                let low = ctx.bind(int_cast::<u64, u128, _>(view));
                let hash = hash_word(ctx, hash_word(ctx, SEED, k), low);
                let e = table.find_or_insert(
                    ctx,
                    hash,
                    |ctx, e| {
                        let same = ctx.var(false);
                        ctx.if_then(eq(e.get(ctx, key), k), |ctx| {
                            // SAFETY: an entry's view was copied into `pool`.
                            let stored = unsafe { pool.value::<Utf8>(ctx, e.get(ctx, text)) };
                            ctx.store(same, stored.bytes_eq(ctx, s));
                        });
                        same
                    },
                    |ctx, e| {
                        e.set(ctx, key, k);
                        e.set(ctx, text, pool.copy_in(ctx, s));
                    },
                );
                e.set(ctx, count, ctx.bind(add(e.get(ctx, count), 1u64)));
                e.set(ctx, sum, ctx.bind(add(e.get(ctx, sum), v)));
            });
            let acc = ctx.var(0u64);
            table.for_each(ctx, |ctx, e| {
                let mixed = ctx.bind(bitxor(e.get(ctx, key), e.get(ctx, count)));
                ctx.store(acc, add(add(acc, mixed), e.get(ctx, sum)));
            });
            acc
        },
    );
    let k = compiler.compile(f).expect("compile");
    Box::new(move |rows| {
        let mut table = ChainedTable::new(&layout);
        let mut pool = BytesPool::new();
        let pool_ref = pool.pool_ref();
        k.as_fn().call(
            &rows.keys,
            &rows.views,
            &rows.values,
            rows.pool.buffers().as_ptr(),
            &table.table_ref(),
            &pool_ref,
        )
    })
}

fn ffi_index_u64(backend: JitBackend) -> Kernel {
    let mut compiler = Compiler::new().with_backend(backend);
    let find = compiler.extern_fn::<U64IndexFindExtern>();
    let f = compiler.fun3(
        "ffi_index_u64",
        move |ctx, keys: U64s, values: U64s, mut index: Var<SRefMut<Opaque<U64Index>>>| {
            keys.staged_iter().zip(values).for_each(ctx, |ctx, row| {
                let (k, v) = (ctx.bind(row.first()), ctx.bind(row.second()));
                let p = ctx.bind(call_extern2(find, index.reborrow(), k));
                fold_into(ctx, p, v);
            });
            ctx.var(0u64)
        },
    );
    let k = compiler.compile(f).expect("compile");
    Box::new(move |rows| {
        let mut index = U64Index {
            hasher: DefaultHashBuilder::default(),
            table: HashTable::new(),
            records: Vec::new(),
        };
        k.as_fn().call(&rows.keys, &rows.values, &mut index);
        index
            .records
            .iter()
            .fold(0, |acc, r| fold(acc, r[0], r[1], r[2]))
    })
}

fn ffi_index_u64_str(backend: JitBackend) -> Kernel {
    let mut compiler = Compiler::new().with_backend(backend);
    let find = compiler.extern_fn::<StrIndexFindExtern>();
    let f = compiler.fun4(
        "ffi_index_u64_str",
        move |ctx,
              keys: U64s,
              views: Views,
              values: U64s,
              mut index: Var<SRefMut<Opaque<StrIndex>>>| {
            let n = ctx.bind(keys.len());
            range(0u64, n).for_each(ctx, |ctx, i| {
                // SAFETY: `i < n`, and every column has `n` rows.
                let (k, view, v) = unsafe {
                    (
                        ctx.bind(keys.get_unchecked(i)),
                        ctx.bind(views.get_unchecked(i)),
                        ctx.bind(values.get_unchecked(i)),
                    )
                };
                let lo = ctx.bind(int_cast::<u64, u128, _>(view));
                let hi = ctx.bind(int_cast::<u64, u128, _>(shr(view, 64u128)));
                // SAFETY: the views index the input pool the index was built with.
                let p =
                    ctx.bind(unsafe { call_extern4_unchecked(find, index.reborrow(), k, lo, hi) });
                fold_into(ctx, p, v);
            });
            ctx.var(0u64)
        },
    );
    let k = compiler.compile(f).expect("compile");
    Box::new(move |rows| {
        let mut index = StrIndex {
            hasher: DefaultHashBuilder::default(),
            table: HashTable::new(),
            records: Vec::new(),
            hashes: Vec::new(),
            pool: BytesPool::new(),
            input: &rows.pool,
        };
        k.as_fn()
            .call(&rows.keys, &rows.views, &rows.values, &mut index);
        index
            .records
            .iter()
            .fold(0, |acc, r| fold(acc, r[0], r[3], r[4]))
    })
}

fn backends() -> Vec<(&'static str, JitBackend)> {
    #[allow(unused_mut)]
    let mut v = vec![("cranelift", JitBackend::Cranelift)];
    #[cfg(feature = "llvm")]
    v.push(("llvm", JitBackend::Llvm));
    v
}

// =============================================================================
// Benchmark
// =============================================================================

/// From L1-sized (1k groups) to far out of cache (4M groups).
const DISTINCT: &[usize] = &[1 << 10, 1 << 16, 1 << 20, 1 << 22];

type Variants = fn(JitBackend) -> [(&'static str, Kernel); 2];

fn bench_key(
    c: &mut Criterion,
    name: &str,
    with_strings: bool,
    native: fn(&Rows) -> u64,
    variants: Variants,
) {
    for &distinct in DISTINCT {
        let rows = Rows::new(distinct, with_strings, 7);
        let expected = native(&rows);
        let mut group = c.benchmark_group(format!("group_count/{name}/{distinct}"));
        group.throughput(Throughput::Elements(ROWS as u64));
        group.sample_size(10);
        group.measurement_time(Duration::from_secs(5));

        group.bench_function("native", |b| b.iter(|| black_box(native(black_box(&rows)))));
        for (backend_name, backend) in backends() {
            for (variant, k) in variants(backend) {
                assert_eq!(
                    k(&rows),
                    expected,
                    "{variant}/{backend_name}, {distinct} groups"
                );
                group.bench_function(BenchmarkId::new(variant, backend_name), |b| {
                    b.iter(|| black_box(k(black_box(&rows))))
                });
            }
        }
        group.finish();
    }
}

fn bench_group_count(c: &mut Criterion) {
    bench_key(c, "u64", false, native_u64, |b| {
        [("chained", chained_u64(b)), ("ffi_index", ffi_index_u64(b))]
    });
    bench_key(c, "u64_str", true, native_u64_str, |b| {
        [
            ("chained", chained_u64_str(b)),
            ("ffi_index", ffi_index_u64_str(b)),
        ]
    });
}

criterion_group!(benches, bench_group_count);
criterion_main!(benches);
