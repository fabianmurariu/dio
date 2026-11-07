//! `Ctx::invariant`: loads of memory nothing writes may be removed when unused, merged
//! when repeated, and hoisted out of loops. Cranelift only: `Compiled::sizes` is empty
//! for LLVM, which keeps such loads plain.

use rust_lms::func::JitBackend;
use rust_lms::prelude::*;

/// The optimised instruction count of function `name`.
fn size<T: StagedType>(compiled: &rust_lms::func::Compiled<T>, name: &str) -> usize {
    compiled
        .sizes()
        .iter()
        .find(|(n, _)| n == name)
        .map(|(_, size)| *size)
        .expect("the function was compiled")
}

/// `arr[0] + arr[0]`, plus a read of `arr[1]` nothing uses, with every read invariant
/// or not; and its size.
fn reads(invariant: bool) -> (i64, usize) {
    let mut compiler = Compiler::new().with_backend(JitBackend::Cranelift);
    let f = compiler.fun1("reads", move |ctx, arr: Var<SRef<Slice<i64>>>| {
        let read = |ctx: &Ctx| {
            // SAFETY: the test passes a slice of length 2.
            let a = ctx.bind(unsafe { arr.get_unchecked(0u64) });
            let b = ctx.bind(unsafe { arr.get_unchecked(0u64) });
            let _unused = ctx.bind(unsafe { arr.get_unchecked(1u64) });
            ctx.bind(a + b)
        };
        if invariant {
            // SAFETY: `arr` is never written, and both indices are inside it.
            unsafe { ctx.invariant(read) }
        } else {
            read(ctx)
        }
    });
    let compiled = compiler.compile(f).expect("compilation failed");
    let data = [21i64, 5];
    (compiled.as_fn().call(&data[..]), size(&compiled, "reads"))
}

#[test]
fn invariant_reads_are_merged_and_unused_ones_removed() {
    let (plain, plain_size) = reads(false);
    let (invariant, invariant_size) = reads(true);
    assert_eq!((plain, invariant), (42, 42));
    // Three loads become one.
    assert!(
        invariant_size + 2 <= plain_size,
        "invariant {invariant_size} vs plain {plain_size} instructions"
    );
}

/// The sum of `arr[0]` over `0..n`, the read inside the loop; and the size of the loop
/// body, as the number of loads left inside it.
fn hoisted(invariant: bool) -> i64 {
    let mut compiler = Compiler::new().with_backend(JitBackend::Cranelift);
    let f = compiler.fun2(
        "sum",
        move |ctx, arr: Var<SRef<Slice<i64>>>, n: Var<u64>| {
            range(0u64, n).fold(ctx, 0i64, move |ctx, acc, _| {
                let read = |ctx: &Ctx| ctx.bind(unsafe { arr.get_unchecked(0u64) });
                // SAFETY: `arr` is never written and has an element 0.
                let x = if invariant {
                    unsafe { ctx.invariant(read) }
                } else {
                    read(ctx)
                };
                acc + x
            })
        },
    );
    let compiled = compiler.compile(f).expect("compilation failed");
    compiled.as_fn().call(&[3i64][..], 4u64)
}

#[test]
fn invariant_reads_in_loops_compute_the_same() {
    assert_eq!(hoisted(false), 12);
    assert_eq!(hoisted(true), 12);
}
