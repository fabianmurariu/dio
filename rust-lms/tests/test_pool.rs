//! A kernel reads a [`BytesPool`]'s values as it reads an Arrow array's rows: each view
//! against the pool's table, whichever block a value landed in.

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
    assert!(pool.blocks().len() > 3);
    let table = pool.buffers().as_ptr();

    for_each_backend(|mut compiler| {
        // How many values end with "7", and their total length.
        let fun = compiler.fun1("scan", move |ctx, rows: Var<SRef<Slice<u128>>>| {
            let table = ctx.bind(const_ptr::<ByteBuffer>(table));
            let seven = ctx.bind(Const::<Str>::new(RawByteView::inline(b"7").unwrap()));
            let (hits, bytes) = (ctx.var(0u64), ctx.var(0u64));
            let i = ctx.var(0u64);
            ctx.while_loop(lt(i, rows.len()), |ctx| {
                // SAFETY: `i` is a row; the views are the pool's, alive for the call.
                let s =
                    ctx.bind(unsafe { ByteView::<Utf8>::from_parts(rows.get_unchecked(i), table) });
                ctx.if_then(s.ends_with(ctx, seven), |ctx| ctx.store(hits, hits + 1u64));
                ctx.store(bytes, bytes + s.byte_len());
                ctx.store(i, i + 1u64);
            });
            (hits << 32u64) | bytes
        });
        let compiled = compiler.compile(fun).expect("compile");
        let hits = values.iter().filter(|v| v.ends_with('7')).count() as u64;
        let bytes = values.iter().map(|v| v.len() as u64).sum::<u64>();
        assert_eq!(compiled.as_fn().call(&views), (hits << 32) | bytes);
    });
}
