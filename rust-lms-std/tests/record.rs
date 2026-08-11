//! `RecordLayout` / `FieldId` / `DynamicRecord`: the offset math is a plain unit
//! check; a JIT kernel then writes typed fields into two packed records (through
//! tokens, never a raw pointer) and the host reads the bytes back at their offsets.

use rust_lms::prelude::*;
use rust_lms_std::{DynamicRecord, RecordLayout};

#[test]
fn layout_offsets_and_stride() {
    let mut l = RecordLayout::new();
    let a = l.field::<i64>(); // offset 0
    let b = l.field::<f64>(); // offset 8
    let c = l.field::<i64>(); // offset 16
    assert_eq!(a.offset(), 0);
    assert_eq!(b.offset(), 8);
    assert_eq!(c.offset(), 16);
    assert_eq!((a.index(), b.index(), c.index()), (0, 1, 2));
    assert_eq!(l.align(), 8);
    assert_eq!(l.stride(), 24);
    assert_eq!(l.num_fields(), 3);
}

/// A records buffer of `[i64 key, f64 val, i64 count]` per record. A kernel writes
/// record `i`'s three fields through typed tokens (the `*mut u8` never appears),
/// for `i in 0..n`; the host reads the raw buffer back and checks every field.
#[test]
fn jit_writes_typed_fields_into_packed_records() {
    let mut layout = RecordLayout::new();
    let key = layout.field::<i64>();
    let val = layout.field::<f64>();
    let cnt = layout.field::<i64>();
    let stride = layout.stride();
    assert_eq!(stride, 24);

    const N: usize = 5;
    // 8-byte-aligned backing store (Vec<u64> guarantees alignment); zero-filled.
    let words = N * stride / 8;
    let mut buf = vec![0u64; words];
    let base: *mut u8 = buf.as_mut_ptr() as *mut u8;

    let mut compiler = Compiler::new();
    let fill = compiler.fun1("fill", move |ctx, n: Var<u64>| {
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, n), move |ctx| {
            let rec: DynamicRecord = layout.record(ctx, const_mut_ptr::<u8>(base), i);
            // key = i*100 ; val = i*100 + 0.5 ; count = i
            let k = ctx.bind(mul(int_cast::<i64, u64, _>(i), 100i64));
            rec.set(ctx, key, k);
            let vf = ctx.bind(add(int_to_float::<f64, i64, _>(k), 0.5f64));
            rec.set(ctx, val, vf);
            let ci = ctx.bind(int_cast::<i64, u64, _>(i));
            rec.set(ctx, cnt, ci);
            ctx.store(i, add(i, 1u64));
        });
        i
    });
    let compiled = compiler.compile(fill).expect("compile");
    compiled.as_fn()(N as u64);

    for i in 0..N {
        let rec = unsafe { (base as *const u8).add(i * stride) };
        let k = unsafe { *(rec.add(key.offset()) as *const i64) };
        let v = unsafe { *(rec.add(val.offset()) as *const f64) };
        let c = unsafe { *(rec.add(cnt.offset()) as *const i64) };
        assert_eq!(k, i as i64 * 100, "key at record {i}");
        assert_eq!(v, i as f64 * 100.0 + 0.5, "val at record {i}");
        assert_eq!(c, i as i64, "count at record {i}");
    }
}

/// A token from one layout used on a record from another panics at stage 0 (during
/// codegen) — the brand guards against a mis-addressed field reaching an unchecked
/// pointer op.
#[test]
#[should_panic(expected = "different RecordLayout")]
fn cross_layout_token_panics() {
    let mut a = RecordLayout::new();
    let a_field = a.field::<i64>();

    let mut b = RecordLayout::new();
    let _ = b.field::<i64>();

    let mut buf = [0u64; 1];
    let base = buf.as_mut_ptr() as *mut u8;

    let mut compiler = Compiler::new();
    // The closure runs at codegen time; `get` asserts the brand and panics there.
    let _ = compiler.fun0("bad", move |ctx| {
        let ptr = ctx.bind(const_mut_ptr::<u8>(base));
        let rec = b.wrap(ptr); // record branded `b`
        rec.get(ctx, a_field) // token from `a` → panic
    });
}
