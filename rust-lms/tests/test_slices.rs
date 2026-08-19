//! Integration tests for slice support.

use rust_lms::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

static CHECKED_GET_SLICE_CODEGENS: AtomicUsize = AtomicUsize::new(0);
static CHECKED_SET_SLICE_CODEGENS: AtomicUsize = AtomicUsize::new(0);

#[derive(Clone)]
struct CountedSharedSlice<S> {
    inner: S,
}

unsafe impl<'a, S> Staged for CountedSharedSlice<S>
where
    S: Staged<Out = SRef<'a, Slice<i64>>>,
{
    type Out = SRef<'a, Slice<i64>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        CHECKED_GET_SLICE_CODEGENS.fetch_add(1, Ordering::SeqCst);
        self.inner.codegen(ctx)
    }
}

struct CountedMutSlice<S> {
    inner: S,
}

unsafe impl<'a, S> Staged for CountedMutSlice<S>
where
    S: Staged<Out = SRefMut<'a, Slice<i64>>>,
{
    type Out = SRefMut<'a, Slice<i64>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        CHECKED_SET_SLICE_CODEGENS.fetch_add(1, Ordering::SeqCst);
        self.inner.codegen(ctx)
    }
}

impl<'a, S> SliceMutOps<'a, i64> for CountedMutSlice<S> where
    S: Staged<Out = SRefMut<'a, Slice<i64>>>
{
}

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
        // SAFETY: this test calls the kernel only with slices of length >= 2.
        unsafe { arr.get_unchecked(1u64) }
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
        ctx.while_loop(lt(i, arr.len()), |ctx| {
            // SAFETY: the loop condition proves `i < arr.len()`.
            ctx.store(total, total + unsafe { arr.get_unchecked(i) });
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
    let set_first = compiler.fun1("set_first", |_ctx, mut arr: Var<SRefMut<Slice<i64>>>| {
        // SAFETY: this test calls the kernel only with non-empty slices.
        unsafe { arr.set_unchecked(0u64, 999i64) }
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
    let fill = compiler.fun1("fill", |ctx, mut arr: Var<SRefMut<Slice<i64>>>| {
        let i = ctx.var(0u64);
        ctx.while_loop(lt(i, arr.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < arr.len()`.
            ctx.emit(unsafe { arr.set_unchecked(i, 42i64) });
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
        // SAFETY: this test calls the kernel only with slices of length >= 4.
        let sub = unsafe { arr.slice_unchecked(1u64, 4u64) };
        ctx.while_loop(lt(i, sub.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < sub.len()`.
            ctx.store(total, total + unsafe { sub.get_unchecked(i) });
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
    let swap = compiler.fun1("swap_ends", |_ctx, mut arr: Var<SRefMut<Slice<i64>>>| {
        // swap arr[0] and arr[last]
        let last = arr.len() - 1u64;
        // SAFETY: this test calls the kernel only with non-empty slices.
        unsafe { arr.swap_unchecked(0u64, last) }
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
        // SAFETY: this test uses slices of length >= 5, and both ranges are
        // ordered and within their respective source slices.
        let sub = unsafe { arr.slice_unchecked(1u64, 5u64).slice_unchecked(1u64, 3u64) };
        ctx.while_loop(lt(i, sub.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < sub.len()`.
            ctx.store(total, total + unsafe { sub.get_unchecked(i) });
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
        // SAFETY: this test uses slices of length >= 3; both the range and
        // element index are within bounds and no overlapping view is used.
        unsafe {
            arr.slice_mut_unchecked(1u64, 3u64)
                .set_unchecked(1u64, 777i64)
        }
    });
    let compiled = compiler.compile(set).expect("compilation failed");
    let f = compiled.as_fn();
    let mut data: [i64; 4] = [10, 20, 30, 40];
    f.call(&mut data[..]);
    assert_eq!(data, [10, 20, 777, 40]);
}

#[test]
fn test_checked_slice_get_or() {
    let mut compiler = Compiler::new();
    let get = compiler.fun2(
        "checked_get",
        |_ctx, arr: Var<SRef<Slice<i64>>>, index: Var<u64>| arr.get_or(index, -1i64),
    );
    let compiled = compiler.compile(get).expect("compilation failed");
    let data = [10i64, 20, 30];
    assert_eq!(compiled.call(&data, 1), 20);
    assert_eq!(compiled.call(&data, 3), -1);
    assert_eq!(compiled.call(&data, u64::MAX), -1);
}

#[test]
fn test_checked_slice_operations_evaluate_the_slice_once() {
    CHECKED_GET_SLICE_CODEGENS.store(0, Ordering::SeqCst);
    let mut compiler = Compiler::new();
    let get = compiler.fun1("checked_get_once", |_ctx, arr: Var<SRef<Slice<i64>>>| {
        // SAFETY: the full range is ordered and bounded by `arr.len()`.
        let full = unsafe { arr.slice_unchecked(0u64, arr.len()) };
        CountedSharedSlice { inner: full }.get_or(0u64, -1i64)
    });
    let compiled = compiler.compile(get).expect("compilation failed");
    assert_eq!(CHECKED_GET_SLICE_CODEGENS.load(Ordering::SeqCst), 1);
    assert_eq!(compiled.call(&[12i64, 13]), 12);

    CHECKED_SET_SLICE_CODEGENS.store(0, Ordering::SeqCst);
    let mut compiler = Compiler::new();
    let set = compiler.fun1("checked_set_once", |_ctx, arr: Var<SRefMut<Slice<i64>>>| {
        let len = arr.len();
        // SAFETY: the full range is ordered and bounded by `arr.len()`.
        let full = unsafe { arr.slice_mut_unchecked(0u64, len) };
        CountedMutSlice { inner: full }.set(0u64, 14i64)
    });
    let compiled = compiler.compile(set).expect("compilation failed");
    assert_eq!(CHECKED_SET_SLICE_CODEGENS.load(Ordering::SeqCst), 1);
    let mut data = [12i64, 13];
    assert!(compiled.call(&mut data));
    assert_eq!(data, [14, 13]);
}

#[test]
fn test_slice_as_ptr_is_raw_and_accepts_empty_slices() {
    fn assert_raw_pointer<E: Staged<Out = SPtr<i64>>>(_expr: E) {}

    let mut compiler = Compiler::new();
    let data_ptr = compiler.fun1("data_ptr", |_ctx, arr: Var<SRef<Slice<i64>>>| {
        let pointer = arr.as_ptr();
        assert_raw_pointer(pointer);
        arr.as_ptr()
    });
    let compiled = compiler.compile(data_ptr).expect("compilation failed");

    let data = [4i64, 5, 6];
    assert_eq!(compiled.call(&data), data.as_ptr());
    let empty: [i64; 0] = [];
    assert_eq!(compiled.call(&empty), empty.as_ptr());
}

#[test]
fn test_checked_mutable_slice_set() {
    let mut compiler = Compiler::new();
    let set = compiler.fun3(
        "checked_set",
        |_ctx, mut arr: Var<SRefMut<Slice<i64>>>, index: Var<u64>, value: Var<i64>| {
            arr.set(index, value)
        },
    );
    let compiled = compiler.compile(set).expect("compilation failed");
    let mut data = [10i64, 20, 30];
    assert!(compiled.call(&mut data, 1, 99));
    assert_eq!(data, [10, 99, 30]);
    assert!(!compiled.call(&mut data, 3, 123));
    assert_eq!(data, [10, 99, 30]);
}

#[test]
fn test_consuming_mutable_element_projection() {
    let mut compiler = Compiler::new();
    let set = compiler.fun1("element_ref", |_ctx, arr: Var<SRefMut<Slice<i64>>>| {
        // SAFETY: the test invokes this kernel with a slice of length 3.
        let element = unsafe { arr.get_mut_unchecked(1u64) };
        store_ref(element, Const::<i64>::new(77))
    });
    let compiled = compiler.compile(set).expect("compilation failed");
    let mut data = [1i64, 2, 3];
    compiled.call(&mut data);
    assert_eq!(data, [1, 77, 3]);
}

#[test]
fn test_subslice_bind_reuse() {
    // `ctx.bind` materializes the sub-slice once; the resulting `Var` is `Copy`,
    // so it can be reused (len + element reads) with no `.clone()`.
    let mut compiler = Compiler::new();
    let sum = compiler.fun1("bind_sub", |ctx, arr: Var<SRef<Slice<i64>>>| {
        // SAFETY: this test calls the kernel only with slices of length >= 4.
        let sub: Var<SRef<Slice<i64>>> = ctx.bind(unsafe { arr.slice_unchecked(1u64, 4u64) });
        let i = ctx.var(0u64);
        let total = ctx.var(0i64);
        ctx.while_loop(lt(i, sub.len()), move |ctx| {
            // SAFETY: the loop condition proves `i < sub.len()`.
            ctx.store(total, total + unsafe { sub.get_unchecked(i) });
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
                // SAFETY: the loop condition proves `i < arr.len()`.
                ctx.if_then(gt(unsafe { arr.get_unchecked(i) }, 3i64), move |ctx| {
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
            // SAFETY: the loop condition proves `i < arr.len()`.
            ctx.store(total, total + unsafe { arr.get_unchecked(i) });
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
        // SAFETY: `half = len / 2`, so `0 <= half <= len`.
        let sub = unsafe { arr.slice_unchecked(0u64, half) };
        sub.len()
    });
    let compiled = compiler.compile(get_half_len).expect("compilation failed");
    let f = compiled.as_fn();
    let data: [i64; 10] = [0; 10];
    assert_eq!(f.call(&data[..]), 5);
}
