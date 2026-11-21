//! A kernel reads a [`BytesPool`]'s values as it reads an Arrow array's rows, through the
//! pool's reader ([`BytesPool::reader`]), whichever block a value landed in.

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

#[test]
fn a_kernel_reads_values_from_every_block() {
    // Small blocks: the long values spread over several.
    let mut pool = BytesPool::with_block_size(32);
    let values: Vec<String> = (0..40)
        .map(|i| match i % 3 {
            0 => format!("v{i}"),
            1 => format!("a value of block-sized length {i}"),
            _ => format!("medium value {i:03}"),
        })
        .collect();
    let views: Vec<u128> = values.iter().map(|v| pool.append(v.as_bytes())).collect();
    assert!(pool.buffers().len() > 3);

    for_each_backend(|mut compiler| {
        // How many values end with "7", and their total length.
        let fun = compiler.fun2(
            "scan",
            move |ctx, rows: Var<SRef<Slice<u128>>>, reader: Var<SRef<PoolReader>>| {
                let pool = StagedPoolReader::new(ctx, ctx.bind(load_ref(reader)));
                let seven = ctx.bind(Const::<Str>::new(RawByteView::inline(b"7").unwrap()));
                let (hits, bytes) = (ctx.var(0u64), ctx.var(0u64));
                let i = ctx.var(0u64);
                ctx.while_loop(lt(i, rows.len()), |ctx| {
                    // SAFETY: `i` is a row; the views are the pool's, which nothing appends to
                    // during the call.
                    let s = unsafe { pool.value::<Utf8>(ctx, ctx.bind(rows.get_unchecked(i))) };
                    ctx.if_then(s.ends_with(ctx, seven), |ctx| ctx.store(hits, hits + 1u64));
                    ctx.store(bytes, bytes + s.byte_len());
                    ctx.store(i, i + 1u64);
                });
                (hits << 32u64) | bytes
            },
        );
        let compiled = compiler.compile(fun).expect("compile");
        let hits = values.iter().filter(|v| v.ends_with('7')).count() as u64;
        let bytes = values.iter().map(|v| v.len() as u64).sum::<u64>();
        assert_eq!(
            compiled.as_fn().call(&views, &pool.reader()),
            (hits << 32) | bytes
        );
    });
}

/// A kernel copies values into a pool: short ones stay their views, long ones land in its
/// blocks after what the host appended, a full block starts the next (through the extern),
/// and a value larger than a block gets one of its own.
#[test]
fn a_kernel_copies_values_into_a_pool() {
    let mut source = BytesPool::new();
    let values: Vec<Vec<u8>> = (0..30)
        .map(|i| match i % 3 {
            0 => format!("s{i}").into_bytes(),
            1 => format!("a long value, number {i:02}").into_bytes(),
            _ => vec![b'x' + (i % 3) as u8; 70 + i],
        })
        .collect();
    let views: Vec<u128> = values.iter().map(|v| source.append(v)).collect();
    let table = source.buffers().as_ptr();

    for_each_backend(|mut compiler| {
        let mut pool = BytesPool::with_block_size(64);
        let host = pool.append(b"appended by the host first");
        let new_block = compiler.extern_fn::<BytesPoolNewBlockExtern>();
        let fun = compiler.fun3(
            "copy_in",
            move |ctx,
                  rows: Var<SRef<Slice<u128>>>,
                  mut out: Var<SRefMut<Slice<u128>>>,
                  pool: Var<SRef<PoolRef>>| {
                // SAFETY: the test's pool, in place and appended to by this call only.
                let pool = unsafe { StagedPool::new(ctx, pool, new_block) };
                let table = ctx.bind(const_ptr::<ByteBuffer>(table));
                range(0u64, rows.len()).for_each(ctx, move |ctx, i| {
                    // SAFETY: `i` is a row of both slices; the views are `source`'s.
                    unsafe {
                        let s =
                            ctx.bind(ByteView::<Binary>::from_parts(rows.get_unchecked(i), table));
                        let view = pool.copy_in(ctx, s);
                        ctx.emit(out.reborrow().set_unchecked(i, view));
                    }
                });
                Const::<()>::new(())
            },
        );
        let compiled = compiler.compile(fun).expect("compile");
        let mut copied = vec![0u128; views.len()];
        let pool_ref = pool.pool_ref();
        compiled.as_fn().call(&views, &mut copied, &pool_ref);
        pool.settle();

        // SAFETY (each read): views of `pool`, alive and not appended to while read.
        let read = |view: u128| unsafe { pool.value(view).bytes().to_vec() };
        assert_eq!(read(host), b"appended by the host first");
        for (value, &view) in values.iter().zip(&copied) {
            assert_eq!(&read(view), value);
        }
        let short = values.iter().zip(&copied).filter(|(v, _)| v.len() <= 12);
        for ((_, &view), &original) in short.zip(views.iter().step_by(3)) {
            assert_eq!(view, original, "a short value is its own view");
        }
        // SAFETY: the pool is settled and alive.
        let sizes: Vec<usize> = pool
            .buffers()
            .iter()
            .map(|b| unsafe { b.as_slice() }.len())
            .collect();
        assert!(sizes.len() > 10, "full blocks started new ones: {sizes:?}");
        assert!(
            sizes.iter().any(|&n| n > 64),
            "an oversized value got its own block"
        );
    });
}

/// A value read through the pool's current table is already in the pool: copying it in
/// keeps its view and appends nothing. A value read before the table moved (the pool
/// outgrew it, starting blocks) is still readable through the old one, and is copied.
#[test]
fn a_value_already_in_the_pool_keeps_its_view() {
    let mut source = BytesPool::new();
    let values: Vec<Vec<u8>> = (0..20)
        .map(|i| format!("a value longer than a view, {i:02}").into_bytes())
        .collect();
    let views: Vec<u128> = values.iter().map(|v| source.append(v)).collect();
    let table = source.buffers().as_ptr();

    for_each_backend(|mut compiler| {
        // A block per value: the table outgrows its first allocation.
        let mut pool = BytesPool::with_block_size(32);
        let new_block = compiler.extern_fn::<BytesPoolNewBlockExtern>();
        let fun = compiler.fun4(
            "copy_twice",
            move |ctx,
                  rows: Var<SRef<Slice<u128>>>,
                  mut copied: Var<SRefMut<Slice<u128>>>,
                  mut again: Var<SRefMut<Slice<u128>>>,
                  pool: Var<SRef<PoolRef>>| {
                // SAFETY: the test's pool, in place and appended to by this call only.
                let pool = unsafe { StagedPool::new(ctx, pool, new_block) };
                let table = ctx.bind(const_ptr::<ByteBuffer>(table));
                // SAFETY (every read): rows of the slices; `source`'s views and the pool's.
                let source = |ctx: &Ctx, i: Var<u64>| unsafe {
                    ctx.bind(ByteView::<Binary>::from_parts(rows.get_unchecked(i), table))
                };
                let first = pool.copy_in(ctx, source(ctx, ctx.var(0u64)));
                let early = unsafe { pool.value::<Binary>(ctx, first) };
                ctx.emit(unsafe { copied.reborrow().set_unchecked(0u64, first) });
                range(1u64, rows.len()).for_each(ctx, |ctx, i| unsafe {
                    let view = pool.copy_in(ctx, source(ctx, i));
                    ctx.emit(copied.reborrow().set_unchecked(i, view));
                    let read = pool.value::<Binary>(ctx, view);
                    ctx.emit(again.reborrow().set_unchecked(i, pool.copy_in(ctx, read)));
                });
                ctx.emit(unsafe {
                    again
                        .reborrow()
                        .set_unchecked(0u64, pool.copy_in(ctx, early))
                });
                Const::<()>::new(())
            },
        );
        let compiled = compiler.compile(fun).expect("compile");
        let (mut copied, mut again) = (vec![0u128; views.len()], vec![0u128; views.len()]);
        let pool_ref = pool.pool_ref();
        compiled
            .as_fn()
            .call(&views, &mut copied, &mut again, &pool_ref);
        pool.settle();

        assert_eq!(
            again[1..],
            copied[1..],
            "a value already in the pool keeps its view"
        );
        assert_ne!(
            again[0], copied[0],
            "a value read through a moved table is copied"
        );
        // SAFETY (each read): views of `pool`, alive and not appended to while read.
        let read = |view: u128| unsafe { pool.value(view).bytes().to_vec() };
        assert_eq!(read(again[0]), values[0]);
        for (value, &view) in values.iter().zip(&copied) {
            assert_eq!(&read(view), value);
        }
        assert_eq!(
            pool.buffers().len(),
            values.len() + 1,
            "one block per value, and the copy"
        );
    });
}
