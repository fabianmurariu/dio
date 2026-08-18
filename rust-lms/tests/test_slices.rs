//! Integration tests for slice support.

use rust_lms::prelude::*;

#[test]
fn test_slice_len() {
    let mut compiler = Compiler::new();
    let get_len = compiler.fun1("get_len", |_ctx, arr: Var<SRef<Slice<i64>>>| arr.len());
    let compiled = compiler.compile(get_len).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 5] = [10, 20, 30, 40, 50];
    assert_eq!(f.call(&data[..]), 5);
}

#[test]
fn test_slice_len_empty() {
    let mut compiler = Compiler::new();
    let get_len = compiler.fun1("get_len", |_ctx, arr: Var<SRef<Slice<i64>>>| arr.len());
    let compiled = compiler.compile(get_len).expect("compilation failed");
    let f = compiled.as_fn();
    assert_eq!(f.call(&[][..]), 0);
}

#[test]
fn test_slice_get_unchecked() {
    let mut compiler = Compiler::new();
    let get_second = compiler.fun1("get_second", |_ctx, arr: Var<SRef<Slice<i64>>>| {
        arr.get_unchecked(1u64)
    });
    let compiled = compiler.compile(get_second).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 5] = [10, 20, 30, 40, 50];
    assert_eq!(f.call(&data[..]), 20);
}

#[test]
fn test_slice_sum() {
    let mut compiler = Compiler::new();
    let sum = compiler.fun1("sum", |ctx, arr: Var<SRef<Slice<i64>>>| {
        let i = ctx.var(0u64);
        let total = ctx.var(0i64);
        ctx.while_loop(lt(i, arr.len()), move |ctx| {
            ctx.store(total, total + arr.get_unchecked(i));
            ctx.store(i, i + 1u64);
        });
        total
    });
    let compiled = compiler.compile(sum).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 5] = [10, 20, 30, 40, 50];
    assert_eq!(f.call(&data[..]), 150);
}

#[test]
fn test_slice_mutable_set() {
    let mut compiler = Compiler::new();
    let set_first = compiler.fun1("set_first", |_ctx, arr: Var<SRefMut<Slice<i64>>>| {
        arr.set_unchecked(0u64, 999i64)
    });
    let compiled = compiler.compile(set_first).expect("compilation failed");
    let f = compiled.as_fn();
    let mut data: [i64; 3] = [10, 20, 30];
    f.call(&mut data[..]);
    assert_eq!(data[0], 999);
    assert_eq!(data[1], 20);
}

#[test]
fn test_slice_mutable_fill() {
    let mut compiler = Compiler::new();
    let fill = compiler.fun1("fill", |ctx, arr: Var<SRefMut<Slice<i64>>>| {
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, arr.len()), move |ctx| {
            ctx.emit(arr.set_unchecked(i, 42i64));
            ctx.store(i, i + 1u64);
        });
        Const::<()>::new(())
    });
    let compiled = compiler.compile(fill).expect("compilation failed");
    let f = compiled.as_fn();
    let mut data: [i64; 4] = [0, 0, 0, 0];
    f.call(&mut data[..]);
    assert_eq!(data, [42, 42, 42, 42]);
}

#[test]
fn test_slice_subslice() {
    let mut compiler = Compiler::new();
    let sum_middle = compiler.fun1("sum_middle", |ctx, arr: Var<SRef<Slice<i64>>>| {
        let i = ctx.var(0u64);
        let total = ctx.var(0i64);
        let sub = arr.slice_unchecked(1u64, 4u64);
        ctx.while_loop(lt(i, sub.len()), move |ctx| {
            ctx.store(total, total + sub.get_unchecked(i));
            ctx.store(i, i + 1u64);
        });
        total
    });
    let compiled = compiler.compile(sum_middle).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 6] = [100, 10, 20, 30, 200, 300];
    assert_eq!(f.call(&data[..]), 60); // arr[1..4] = [10,20,30]
}

#[test]
fn test_slice_swap() {
    let mut compiler = Compiler::new();
    let swap = compiler.fun1("swap_ends", |_ctx, arr: Var<SRefMut<Slice<i64>>>| {
        // swap arr[0] and arr[last]
        let last = arr.len() - 1u64;
        arr.swap_unchecked(0u64, last)
    });
    let compiled = compiler.compile(swap).expect("compilation failed");
    let f = compiled.as_fn();
    let mut data: [i64; 4] = [1, 2, 3, 4];
    f.call(&mut data[..]);
    assert_eq!(data, [4, 2, 3, 1]);
}

#[test]
fn test_slice_of_slice() {
    // Slicing is closed: a sub-slice supports `slice_unchecked` again.
    let mut compiler = Compiler::new();
    let sum = compiler.fun1("sub_of_sub", |ctx, arr: Var<SRef<Slice<i64>>>| {
        let i = ctx.var(0u64);
        let total = ctx.var(0i64);
        // arr[1..5] then [1..3] of that == arr[2..4]
        let sub = arr.slice_unchecked(1u64, 5u64).slice_unchecked(1u64, 3u64);
        ctx.while_loop(lt(i, sub.len()), move |ctx| {
            ctx.store(total, total + sub.get_unchecked(i));
            ctx.store(i, i + 1u64);
        });
        total
    });
    let compiled = compiler.compile(sum).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 6] = [0, 1, 2, 3, 4, 5];
    assert_eq!(f.call(&data[..]), 5); // arr[2..4] = [2, 3]
}

#[test]
fn test_mut_subslice_stays_mutable() {
    // A sub-slice of `&mut [T]` is itself `&mut [T]`, so `set_unchecked` works.
    let mut compiler = Compiler::new();
    let set = compiler.fun1("mut_sub_set", |_ctx, arr: Var<SRefMut<Slice<i64>>>| {
        // sub = &mut arr[1..3]; sub[1] = 777  =>  writes arr[2]
        arr.slice_mut_unchecked(1u64, 3u64)
            .set_unchecked(1u64, 777i64)
    });
    let compiled = compiler.compile(set).expect("compilation failed");
    let f = compiled.as_fn();
    let mut data: [i64; 4] = [10, 20, 30, 40];
    f.call(&mut data[..]);
    assert_eq!(data, [10, 20, 777, 40]);
}

#[test]
fn test_subslice_bind_reuse() {
    // `ctx.bind` materializes the sub-slice once; the resulting `Var` is `Copy`,
    // so it can be reused (len + element reads) with no `.clone()`.
    let mut compiler = Compiler::new();
    let sum = compiler.fun1("bind_sub", |ctx, arr: Var<SRef<Slice<i64>>>| {
        let sub: Var<SRef<Slice<i64>>> = ctx.bind(arr.slice_unchecked(1u64, 4u64));
        let i = ctx.var(0u64);
        let total = ctx.var(0i64);
        ctx.while_loop(lt(i, sub.len()), move |ctx| {
            ctx.store(total, total + sub.get_unchecked(i));
            ctx.store(i, i + 1u64);
        });
        total
    });
    let compiled = compiler.compile(sum).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 6] = [100, 10, 20, 30, 200, 300];
    assert_eq!(f.call(&data[..]), 60); // arr[1..4] = [10, 20, 30]
}

#[test]
fn test_slice_count_all_larger_than_3() {
    let mut compiler = Compiler::new();
    let count_greater_than_3 =
        compiler.fun1("count_greater_than_3", |ctx, arr: Var<SRef<Slice<i64>>>| {
            let i = ctx.var(0u64);
            let count = ctx.var(0u64);
            ctx.while_loop(lt(i, arr.len()), move |ctx| {
                ctx.if_then(gt(arr.get_unchecked(i), 3i64), move |ctx| {
                    ctx.store(count, count + 1u64);
                });
                ctx.store(i, i + 1u64);
            });
            count
        });
    let compiled = compiler
        .compile(count_greater_than_3)
        .expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 5] = [1, 4, 5, 2, 6];
    assert_eq!(f.call(&data[..]), 3u64); // 4, 5, 6 > 3
}

#[test]
fn test_slice_f64() {
    let mut compiler = Compiler::new();
    let sum_f64 = compiler.fun1("sum_f64", |ctx, arr: Var<SRef<Slice<f64>>>| {
        let i = ctx.var(0u64);
        let total = ctx.var(0.0f64);
        ctx.while_loop(lt(i, arr.len()), move |ctx| {
            ctx.store(total, total + arr.get_unchecked(i));
            ctx.store(i, i + 1u64);
        });
        total
    });
    let compiled = compiler.compile(sum_f64).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [f64; 4] = [1.5, 2.5, 3.0, 4.0];
    assert!((f.call(&data[..]) - 11.0).abs() < 0.0001);
}

#[test]
fn test_slice_return_subslice_len() {
    let mut compiler = Compiler::new();
    let get_half_len = compiler.fun1("get_half_len", |_ctx, arr: Var<SRef<Slice<i64>>>| {
        let half = arr.len() / 2u64;
        let sub = arr.slice_unchecked(0u64, half);
        sub.len()
    });
    let compiled = compiler.compile(get_half_len).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 10] = [0; 10];
    assert_eq!(f.call(&data[..]), 5);
}
