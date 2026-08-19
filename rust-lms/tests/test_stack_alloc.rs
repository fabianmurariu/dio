//! `stack_alloc`: a runtime-writable stack scratch slot — write typed values into
//! it and read them back within the kernel.

use rust_lms::prelude::*;

#[test]
fn stack_alloc_write_read_roundtrip() {
    let mut compiler = Compiler::new();
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
}
