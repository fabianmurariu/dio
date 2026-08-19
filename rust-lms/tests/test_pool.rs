//! A JIT kernel appending bytes into a host-owned [`BytesPool`] via the
//! `pool_append` extern: proves the FFI plumbing (opaque `&mut` handle in,
//! stable pointer out) and that the host recovers the bytes zero-copy.

use rust_lms::pool::BytesPool;
use rust_lms::prelude::*;

type PoolRef = SRefMut<'static, Opaque<BytesPool>>;

/// Build a `FatSlice<u8>` of literal bytes baked into the kernel frame.
fn lit(ctx: &mut Ctx, bytes: &[u8]) -> impl Staged<Out = FatSliceType<u8>> {
    let ptr = ctx.bind(stack_bytes(bytes));
    // SAFETY: `stack_bytes` materializes exactly this many initialized bytes in
    // the kernel frame, which remains live for the extern call.
    unsafe { slice_from_raw_parts::<u8, _, _>(ptr, bytes.len() as u64) }
}

#[test]
fn kernel_appends_into_pool() {
    let mut compiler = Compiler::new();
    let append = compiler.extern_fn::<PoolAppendExtern>();

    // fn(&mut BytesPool) -> u64 : append "hi" then "world", return the 2nd ptr.
    // The return type is derived from `pool_append`'s own signature — no turbofish.
    let f = compiler.fun1("append_two", move |ctx, pool: Var<PoolRef>| {
        let pool_ptr = ref_mut_as_ptr(pool);
        let hi = lit(ctx, b"hi");
        // SAFETY: the kernel receives exclusive access to `pool`; calls are
        // sequenced and each `FatSlice` points into the live kernel frame.
        let _a = ctx.bind(unsafe { call_extern2_unchecked(append, pool_ptr, hi) });
        let world = lit(ctx, b"world");
        // SAFETY: same exclusive pool borrow; `world` remains live for this call.
        ctx.bind(unsafe { call_extern2_unchecked(append, pool_ptr, world) })
    });
    let compiled = compiler.compile(f).expect("compile");

    let mut pool = BytesPool::new();
    let world_addr = compiled.call(&mut pool);

    // Both appends landed contiguously in one chunk, recoverable by the host.
    assert_eq!(pool.chunks().len(), 1);
    assert_eq!(pool.chunks()[0], b"hiworld");

    // The returned pointer is stable and points at "world" within that chunk.
    let base = pool.chunks()[0].as_ptr() as u64;
    assert_eq!(world_addr, base + 2);
}
