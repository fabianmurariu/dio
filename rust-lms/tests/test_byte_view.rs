//! `ByteView<K>`: Arrow byte views as staged `&str` / `&[u8]`. Every operation is
//! checked against Rust's on the bytes, over values of every length class (empty,
//! inside the prefix, inline, just past inline, long), in two pools: the same long
//! string has a different `(buf_idx, offset)` and a different table in each, so equality
//! and order cannot lean on where a value is stored.

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

/// Byte strings of every length class, and the pairs that trip a careless compare: equal
/// prefixes of different lengths, a NUL after a prefix, a difference only in the last
/// byte, multi-byte UTF-8 and bytes `>= 0x80`.
const STRINGS: &[&str] = &[
    "",
    "a",
    "ab",
    "abc",
    "abcd",
    "abcde",
    "bar",
    "bar\0",
    "b",
    "z",
    "é",
    "zé",
    "twelve bytes",
    "twelve bytez",
    "thirteen byte",
    "thirteen bytes",
    "a string long enough to span words",
    "a string long enough to span wordz",
    "a string long enough to span words and more",
    "a string long enough",
    "日本語のテキスト",
    "日本語のテキストと",
    "\u{80}\u{7ff}\u{800}\u{ffff}\u{10000}",
    "abcdefghijklmnopqrstuvwxyz",
    "abcdefghijklmnopqrstuvwxyA",
    "abcdéfghijklmnop",
    "abcdzfghijklmnop",
];

/// Raw bytes, not UTF-8: high bits set, zeros, `0xff`.
const BYTES: &[&[u8]] = &[
    b"",
    b"\x00",
    b"\x7f",
    b"\x80",
    b"\xff",
    b"\xff\xff\xff\xff\x00",
    b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c",
    b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d",
    b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x80",
    b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\xff",
    b"\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11\x12\x13\x00",
    // Long, one prefix, differing in the first byte of the first word after it.
    b"\x01\x02\x03\x04\x80\x06\x07\x08\x09\x0a\x0b\x0c\x0d",
    b"\x01\x02\x03\x04\x7f\x06\x07\x08\x09\x0a\x0b\x0c\x0d",
];

/// A pool holding `values`, small blocks so that later values land in later blocks, and
/// each value's view. The first pool starts with filler so offsets differ between pools.
fn pool_of(values: &[&[u8]], filler: usize) -> (BytesPool, Vec<u128>) {
    let mut pool = BytesPool::with_block_size(48);
    if filler > 0 {
        pool.append(&vec![b'#'; filler]);
    }
    let views = values.iter().map(|v| pool.append(v)).collect();
    (pool, views)
}

fn strings() -> Vec<&'static [u8]> {
    STRINGS.iter().map(|s| s.as_bytes()).collect()
}

/// `-1`, `0` or `1`, as `compare` returns.
fn sign(o: std::cmp::Ordering) -> i8 {
    o as i8
}

/// Every pair of `values`, the left from one pool and the right from another, through a
/// compiled `f(a, b)`, against `host(a, b)`.
fn each_pair<K, R, V>(
    values: &[&[u8]],
    f: impl Fn(&Ctx, Var<ByteView<K>>, Var<ByteView<K>>) -> Var<R> + Copy + 'static,
    host: impl Fn(&[u8], &[u8]) -> V,
) where
    K: ByteKind,
    R: StagedType + for<'c> RuntimeResult<Output<'c> = V> + 'static,
    V: PartialEq + std::fmt::Debug,
{
    let (left, left_views) = pool_of(values, 0);
    let (right, right_views) = pool_of(values, 17);
    // The long values are stored at different places in the two pools: check it.
    assert!(left_views.iter().zip(&right_views).any(|(l, r)| l != r));
    for_each_backend(|mut compiler| {
        let fun = compiler.fun2(
            "pair",
            move |ctx, a: Var<ByteView<K>>, b: Var<ByteView<K>>| f(ctx, a, b),
        );
        let compiled = compiler.compile(fun).expect("compile");
        let fun = compiled.as_fn();
        for (a, &va) in values.iter().zip(&left_views) {
            for (b, &vb) in values.iter().zip(&right_views) {
                let got = fun.call(left.value(va), right.value(vb));
                assert_eq!(got, host(a, b), "{:?} vs {:?}", show(a), show(b));
            }
        }
    });
}

fn show(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes).into_owned()
}

#[test]
fn equality_is_byte_equality() {
    each_pair::<Utf8, bool, _>(&strings(), |ctx, a, b| a.bytes_eq(ctx, b), |a, b| a == b);
    each_pair::<Binary, bool, _>(BYTES, |ctx, a, b| a.bytes_eq(ctx, b), |a, b| a == b);
}

#[test]
fn order_is_byte_order() {
    each_pair::<Utf8, i8, _>(
        &strings(),
        |ctx, a, b| a.compare(ctx, b),
        |a, b| sign(a.cmp(b)),
    );
    each_pair::<Binary, i8, _>(BYTES, |ctx, a, b| a.compare(ctx, b), |a, b| sign(a.cmp(b)));
}

#[test]
fn starts_with_ends_with_and_contains_match_rust() {
    let strings = strings();
    each_pair::<Utf8, bool, _>(
        &strings,
        |ctx, a, b| a.starts_with(ctx, b),
        |a, b| a.starts_with(b),
    );
    each_pair::<Utf8, bool, _>(
        &strings,
        |ctx, a, b| a.ends_with(ctx, b),
        |a, b| a.ends_with(b),
    );
    each_pair::<Utf8, bool, _>(
        &strings,
        |ctx, a, b| a.contains(ctx, b),
        |a, b| b.is_empty() || a.windows(b.len()).any(|w| w == b),
    );
    each_pair::<Binary, bool, _>(
        BYTES,
        |ctx, a, b| a.starts_with(ctx, b),
        |a, b| a.starts_with(b),
    );
}

/// Needles found at each position of a long haystack, and ones that only nearly occur.
#[test]
fn contains_finds_needles_anywhere() {
    let hay: &[u8] = b"the quick brown fox jumps over the lazy dog";
    let mut values = vec![hay];
    for (start, len) in [
        (0, 3),
        (4, 5),
        (10, 13),
        (30, 13),
        (40, 3),
        (0, 43),
        (35, 8),
    ] {
        values.push(&hay[start..start + len]);
    }
    values.extend([
        &b"lazy cat"[..],
        b"the quick brown fox jumps over the lazy dogs",
        b"dog ",
    ]);
    each_pair::<Utf8, bool, _>(
        &values,
        |ctx, a, b| a.contains(ctx, b),
        |a, b| b.is_empty() || a.windows(b.len()).any(|w| w == b),
    );
}

#[test]
fn lengths_count_bytes_and_characters() {
    let (pool, views) = pool_of(&strings(), 5);
    for_each_backend(|mut compiler| {
        let utf8_len = compiler.extern_fn::<Utf8LenExtern>();
        let fun = compiler.fun1("lens", move |ctx, s: Var<Str>| {
            // Characters in the high half, bytes in the low.
            let chars = s.len(ctx, utf8_len);
            (chars << 32u64) | s.byte_len()
        });
        let compiled = compiler.compile(fun).expect("compile");
        let fun = compiled.as_fn();
        for (s, &view) in STRINGS.iter().zip(&views) {
            let want = ((s.chars().count() as u64) << 32) | s.len() as u64;
            assert_eq!(fun.call(pool.value(view)), want, "{s:?}");
        }
    });
}

/// A pool's value baked in as a constant (a plan's literal) meets a parameter.
#[test]
fn a_literal_compares_with_a_value() {
    let mut literals = BytesPool::new();
    // Append first: the table may move while the pool grows, so take its address after.
    let short = literals.append(b"short");
    let long = literals.append(b"a literal longer than twelve");
    let (short, long) = (literals.value(short), literals.value(long));
    let (pool, views) = pool_of(&strings(), 0);
    for literal in [short, long] {
        for_each_backend(|mut compiler| {
            let fun = compiler.fun1("vs_literal", move |ctx, s: Var<Str>| {
                let literal = ctx.bind(Const::<Str>::new(literal));
                s.compare(ctx, literal)
            });
            let compiled = compiler.compile(fun).expect("compile");
            let fun = compiled.as_fn();
            // SAFETY: the literal pool is alive and unchanged.
            let text = unsafe { literal.bytes() };
            for (s, &view) in strings().iter().zip(&views) {
                assert_eq!(
                    fun.call(pool.value(view)),
                    sign(s.cmp(&text)),
                    "{:?}",
                    show(s)
                );
            }
        });
    }
}

/// A view and its table read from memory, as a kernel reads an Arrow array's row: the
/// view from a slice of views, the table from a pointer.
#[test]
fn a_view_is_read_from_a_column() {
    let (pool, views) = pool_of(&strings(), 3);
    let table = pool.buffers().as_ptr();
    for_each_backend(|mut compiler| {
        // How many rows equal row 0 or start with row 1.
        let fun = compiler.fun1("rows", move |ctx, rows: Var<SRef<Slice<u128>>>| {
            let table = ctx.bind(const_ptr::<ByteBuffer>(table));
            let first = row_at(ctx, rows, ctx.var(0u64), table);
            let second = row_at(ctx, rows, ctx.var(1u64), table);
            let n = ctx.var(0u64);
            let i = ctx.var(0u64);
            ctx.while_loop(lt(i, rows.len()), |ctx| {
                let s = row_at(ctx, rows, i, table);
                let hit = s.bytes_eq(ctx, first) | s.starts_with(ctx, second);
                ctx.if_then(hit, |ctx| ctx.store(n, n + 1u64));
                ctx.store(i, i + 1u64);
            });
            n
        });
        let compiled = compiler.compile(fun).expect("compile");
        let strings = strings();
        let want = strings
            .iter()
            .filter(|s| **s == strings[0] || s.starts_with(strings[1]))
            .count() as u64;
        assert_eq!(compiled.as_fn().call(&views), want);
    });
}

/// Row `i` of `rows` against `table`.
fn row_at(
    ctx: &Ctx,
    rows: Var<SRef<Slice<u128>>>,
    i: Var<u64>,
    table: Var<SPtr<ByteBuffer>>,
) -> Var<Str> {
    // SAFETY: `i` is a row, and the rows are views into `table`'s blocks, alive for
    // the call.
    ctx.bind(unsafe { ByteView::<Utf8>::from_parts(rows.get_unchecked(i), table) })
}

/// A string is an item like any `Copy` value: the greatest of a column, by byte order.
#[test]
fn a_view_is_an_iterator_item() {
    let (pool, views) = pool_of(&strings(), 0);
    let values: Vec<RawByteView> = views.iter().map(|&v| pool.value(v)).collect();
    for_each_backend(|mut compiler| {
        let fun = compiler.fun1("greatest", |ctx, xs: Var<SRef<Slice<Str>>>| {
            // SAFETY: the column is not empty.
            let best = ctx.var(unsafe { xs.get_unchecked(0u64) });
            xs.staged_iter().for_each(ctx, move |ctx, x| {
                let greater = gt(x.compare(ctx, best), 0i8);
                ctx.if_then(greater, |ctx| ctx.store(best, x));
            });
            best
        });
        let compiled = compiler.compile(fun).expect("compile");
        let got = compiled.as_fn().call(&values);
        // SAFETY: the pool is alive and unchanged.
        let got = unsafe { got.bytes() };
        assert_eq!(got, strings().into_iter().max().unwrap());
    });
}
