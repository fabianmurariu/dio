//! `stack_alloc`: a runtime-writable stack scratch slot — write typed values into
//! it and read them back within the kernel.

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

#[test]
fn stack_alloc_write_read_roundtrip() {
    for_each_backend(|mut compiler| {
        // Reserve 16 bytes (two i64 cells), write a and b, read them back, return a+b.
        let f = compiler.fun2("scratch", |ctx, a: Var<i64>, b: Var<i64>| {
            let buf = ctx.bind(stack_alloc(16));
            // buf[0] = a ; buf[1] = b  (via typed i64 pointer indexing)
            let p = ptr_cast_mut::<i64, u8, _>(buf);
            // SAFETY: `stack_alloc(16)` provides two aligned i64 slots that remain
            // live for the generated function; both offsets are in bounds.
            let (write_a, write_b, read_a, read_b) = unsafe {
                (
                    store(ptr_offset_mut(p, Const::<i64>::new(0)), a),
                    store(ptr_offset_mut(p, Const::<i64>::new(1)), b),
                    load_mut(ptr_offset_mut(p, Const::<i64>::new(0))),
                    load_mut(ptr_offset_mut(p, Const::<i64>::new(1))),
                )
            };
            ctx.emit(write_a);
            ctx.emit(write_b);
            let ra = ctx.bind(read_a);
            let rb = ctx.bind(read_b);
            add(ra, rb)
        });
        let compiled = compiler.compile(f).expect("compile");
        assert_eq!(compiled.call(7, 35), 42);
    });
}

#[test]
fn stack_array_holds_i128_keys() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("pairs", |ctx, a: Var<i64>, b: Var<u64>| {
            let keys = ctx.bind(stack_array::<i128>(3));
            let neg = ctx.bind(sub(0i64, a));
            // SAFETY: three slots, each written before it is read.
            unsafe {
                for (slot, key) in [(0i64, pack_i128(a, b)), (2, pack_i128(neg, b))] {
                    ctx.emit(store(ptr_offset_mut(keys, Const::new(slot)), key));
                }
                let first = ctx.bind(load_mut(ptr_offset_mut(keys, Const::new(0i64))));
                let last = ctx.bind(load_mut(ptr_offset_mut(keys, Const::new(2i64))));
                select(lt(first, last), 1u64, 0u64)
            }
        });
        let compiled = compiler.compile(f).expect("compile");
        assert_eq!(compiled.call(-3, 9), 1);
        assert_eq!(compiled.call(3, 9), 0);
    });
}
