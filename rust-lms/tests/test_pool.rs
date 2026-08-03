//! A JIT kernel appending bytes into a host-owned [`BytesPool`] via the
//! `pool_append` extern: proves the FFI plumbing (opaque `&mut` handle in,
//! stable pointer out) and that the host recovers the bytes zero-copy.

use rust_lms::pool::BytesPool;
use rust_lms::prelude::*;

type PoolRef = SRefMut<'static, Opaque<BytesPool>>;

/// Build a `FatSlice<u8>` of literal bytes baked into the kernel frame.
fn lit(ctx: &mut Ctx, bytes: &[u8]) -> impl Staged<Out = FatSliceType<u8>> {
    let ptr = ctx.bind(stack_bytes(bytes));
    slice_from_raw_parts::<u8, _, _>(ptr, bytes.len() as u64)
}

#[test]
fn kernel_appends_into_pool() {
    let mut compiler = Compiler::new();
    let append = compiler.extern_fn::<PoolAppendExtern>();

    // fn(&mut BytesPool) -> u64 : append "hi" then "world", return the 2nd ptr.
    // The return type is derived from `pool_append`'s own signature — no turbofish.
    let f = compiler.fun1("append_two", move |ctx, pool: Var<PoolRef>| {
        let hi = lit(ctx, b"hi");
        let _a = ctx.bind(call_extern2(append, pool, hi));
        let world = lit(ctx, b"world");
        ctx.bind(call_extern2(append, pool, world))
    });
    let compiled = compiler.compile(f).expect("compile");

    let mut pool = BytesPool::new();
    let world_addr = compiled.as_fn()(&mut pool);

    // Both appends landed contiguously in one chunk, recoverable by the host.
    assert_eq!(pool.chunks().len(), 1);
    assert_eq!(pool.chunks()[0], b"hiworld");

    // The returned pointer is stable and points at "world" within that chunk.
    let base = pool.chunks()[0].as_ptr() as u64;
    assert_eq!(world_addr, base + 2);
}
