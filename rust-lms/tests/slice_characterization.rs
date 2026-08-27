//! # Slice API characterization matrix — refactor plan row 0
//!
//! Locks in the **current** behaviour of every slice origin *before* the
//! refactor (`docs/refactor_slice_api.md`) renames or consolidates anything, so
//! the redesign cannot silently drop ABI or typed-leaf behaviour that already
//! works. Every runtime cell runs on both backends via the shared harness.
//!
//! | Origin | shared read | mut write | sub-slice | nested sub-slice | internal call/ret | FFI call/ret | iterator |
//! |---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
//! | shared parameter `SRef<Slice<T>>`    | OK | n/a | OK | OK | OK | OK / **G6** | OK |
//! | mutable parameter `SRefMut<Slice<T>>`| OK | OK  | OK | OK | OK | **G6**      | **G10** |
//! | raw descriptor `FatSliceType<T>`     | OK | **G6** | OK | OK | OK | OK       | **G10** |
//! | descriptor field (`SliceRepr`)       | OK | OK  | OK | OK | OK | OK          | OK |
//!
//! Legend — `OK`: characterized by a test below. `n/a`: not meaningful for that
//! origin. `Gn`: **not expressible today**, closed by plan row *n*. The `OK`
//! cells are behaviour the refactor must preserve; the `Gn` cells are the
//! checklist of what it must make possible.
//!
//! ## Gaps found while building this matrix
//!
//! * **G6a** — a slice *parameter* cannot be handed to an extern declared with
//!   `FatSlice<T>`/`FatSliceMut<T>`, in either direction, even though both lower
//!   to the same `(ptr, len)` argument pair. `UncheckedExternArg` witnesses
//!   `FatSliceType<T> -> SRef<Slice<T>>` but not the reverse, so today a kernel
//!   must declare its parameter as `Var<FatSliceType<T>>` to reach such an
//!   extern. (`ext_double_slice` in `test_extern_fn.rs` is declared but never
//!   called by any test — this is why.)
//! * **G6b** — `FatSliceMutType<T>` implements no `SliceType`, so a mutable raw
//!   descriptor supports no slice operation at all, not even `len`.
//! * **G10** — `SliceIter` is bound to `SRef<Slice<T>>`, so neither a mutable
//!   parameter nor a raw descriptor can be iterated.

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

// =============================================================================
// Origin 1: shared function parameter — `Var<SRef<Slice<T>>>`
// =============================================================================

#[test]
fn shared_param_read() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("r", |ctx, a: Var<SRef<Slice<i64>>>| {
            let first = ctx.bind(unsafe { a.get_unchecked(0u64) });
            add(int_cast::<i64, u64, _>(mul(a.count(), 100u64)), first)
        });
        let c = compiler.compile(f).unwrap();
        assert_eq!(c.call(&[7i64, 8, 9]), 307);
    });
}

#[test]
fn shared_param_get_or_is_bounds_checked() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("g", |_c, a: Var<SRef<Slice<i64>>>, i: Var<u64>| {
            a.get_or(i, -1i64)
        });
        let c = compiler.compile(f).unwrap();
        assert_eq!(c.call(&[5i64, 6], 1), 6);
        assert_eq!(c.call(&[5i64, 6], 2), -1);
    });
}

#[test]
fn shared_param_subslice_and_nested_subslice() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("s", |ctx, a: Var<SRef<Slice<i64>>>| {
            let outer = ctx.bind(unsafe { a.slice_unchecked(1u64, 6u64) }); // [1..6)
            let inner = ctx.bind(unsafe { outer.slice_unchecked(1u64, 3u64) }); // [2..4)
            add(
                int_cast::<i64, u64, _>(mul(inner.count(), 1000u64)),
                unsafe { inner.get_unchecked(0u64) },
            )
        });
        let c = compiler.compile(f).unwrap();
        // inner = [20, 30], len 2 -> 2000 + 20
        assert_eq!(c.call(&[0i64, 10, 20, 30, 40, 50, 60]), 2020);
    });
}

#[test]
fn shared_param_internal_call_and_fat_return() {
    for_each_backend(|mut compiler| {
        // A sub-slice crosses the private function ABI as a fat value and back.
        let tail = compiler.fun1("tail", |_c, a: Var<SRef<Slice<i64>>>| unsafe {
            a.slice_unchecked(1u64, a.count())
        });
        let f = compiler.fun1("f", |ctx, a: Var<SRef<Slice<i64>>>| {
            let t = ctx.bind(call1(tail, a));
            add(int_cast::<i64, u64, _>(mul(t.count(), 10u64)), unsafe {
                t.get_unchecked(0u64)
            })
        });
        let c = compiler.compile(f).unwrap();
        // tail = [2,3,4], len 3 -> 30 + 2
        assert_eq!(c.call(&[1i64, 2, 3, 4]), 32);
    });
}

/// The half of the shared-parameter FFI cell that *does* work today: an extern
/// declared with `&[T]`. Unchecked only because Rust slice references have no
/// stable C ABI. The `FatSlice<T>`-declared half is **G6a**.
#[allow(improper_ctypes_definitions)]
#[extern_fn]
#[no_mangle]
pub extern "C" fn charz_ref_len(data: &[i64]) -> usize {
    data.len()
}

#[test]
fn shared_param_to_reference_extern() {
    for_each_backend(|mut compiler| {
        let ext = compiler.extern_fn::<CharzRefLenExtern>();
        let f = compiler.fun1("rl", |_c, a: Var<SRef<Slice<i64>>>| {
            // SAFETY: `a` is a live shared slice reference for the call.
            unsafe { call_extern1_unchecked(ext, a) }
        });
        let c = compiler.compile(f).unwrap();
        assert_eq!(c.call(&[3i64, 5, 8, 13]), 4);
    });
}

#[test]
fn shared_param_iterator() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("i", |ctx, a: Var<SRef<Slice<i64>>>| {
            a.staged_iter().sum(ctx)
        });
        let c = compiler.compile(f).unwrap();
        assert_eq!(c.call(&[1i64, 2, 3, 4]), 10);
    });
}

// =============================================================================
// Origin 2: mutable function parameter — `Var<SRefMut<Slice<T>>>`
// =============================================================================

#[test]
fn mut_param_read_and_write() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("w", |ctx, mut a: Var<SRefMut<Slice<i64>>>| {
            let v = ctx.bind(unsafe { a.get_unchecked(0u64) });
            ctx.emit(unsafe { a.set_unchecked(1u64, add(v, 5i64)) });
            ctx.emit(unsafe { a.swap_unchecked(0u64, 2u64) });
            a.len()
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [1i64, 2, 3];
        assert_eq!(c.call(&mut data), 3);
        assert_eq!(data, [3i64, 6, 1]);
    });
}

#[test]
fn mut_param_set_is_bounds_checked() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2("cs", |_c, mut a: Var<SRefMut<Slice<i64>>>, i: Var<u64>| {
            a.set(i, 42i64)
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [1i64, 2];
        assert!(c.call(&mut data, 1));
        assert_eq!(data, [1i64, 42]);
        let mut data = [1i64, 2];
        assert!(!c.call(&mut data, 9));
        assert_eq!(data, [1i64, 2], "out-of-bounds set must not write");
    });
}

#[test]
fn mut_param_subslice_stays_mutable_and_nests() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("ms", |ctx, a: Var<SRefMut<Slice<i64>>>| {
            let outer = unsafe { a.slice_mut_unchecked(1u64, 5u64) }; // [1..5)
            let inner = unsafe { outer.slice_mut_unchecked(1u64, 3u64) }; // [2..4)
            ctx.emit(unsafe { inner.set_unchecked(0u64, 99i64) }); // index 2
            Const::<i64>::new(0)
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [0i64, 1, 2, 3, 4, 5];
        c.call(&mut data);
        assert_eq!(data, [0i64, 1, 99, 3, 4, 5]);
    });
}

#[test]
fn mut_param_crosses_an_internal_call() {
    for_each_backend(|mut compiler| {
        let bump = compiler.fun1("bump", |ctx, mut a: Var<SRefMut<Slice<i64>>>| {
            ctx.emit(unsafe { a.set_unchecked(0u64, 77i64) });
            a.len()
        });
        let f = compiler.fun1("f", |_c, a: Var<SRefMut<Slice<i64>>>| call1(bump, a));
        let c = compiler.compile(f).unwrap();
        let mut data = [1i64, 2];
        assert_eq!(c.call(&mut data), 2);
        assert_eq!(data, [77i64, 2]);
    });
}

// =============================================================================
// Origin 3: raw descriptor — `FatSliceType<T>`
// =============================================================================

#[extern_fn]
#[no_mangle]
pub extern "C" fn charz_sum(data: FatSlice<i64>) -> i64 {
    // SAFETY: staged code passes a live `(ptr, len)` for the duration of the call.
    unsafe { data.as_slice().iter().sum() }
}

#[test]
fn raw_descriptor_read_subslice_and_nested() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("raw", |ctx, d: Var<FatSliceType<i64>>| {
            let outer = ctx.bind(unsafe { d.slice_unchecked(1u64, 6u64) });
            let inner = ctx.bind(unsafe { outer.slice_unchecked(1u64, 3u64) });
            add(int_cast::<i64, u64, _>(mul(inner.len(), 1000u64)), unsafe {
                inner.get_unchecked(0u64)
            })
        });
        let c = compiler.compile(f).unwrap();
        let data = [0i64, 10, 20, 30, 40, 50, 60];
        assert_eq!(c.as_fn().call(FatSlice::from_slice(&data)), 2020);
    });
}

#[test]
fn raw_descriptor_ffi_round_trip() {
    for_each_backend(|mut compiler| {
        let ext = compiler.extern_fn::<CharzSumExtern>();
        let f = compiler.fun1("fx", |_c, d: Var<FatSliceType<i64>>| call_extern1(ext, d));
        let c = compiler.compile(f).unwrap();
        let data = [1i64, 2, 3, 4];
        assert_eq!(c.as_fn().call(FatSlice::from_slice(&data)), 10);
    });
}

#[test]
fn raw_descriptor_crosses_an_internal_call() {
    for_each_backend(|mut compiler| {
        let inner = compiler.fun1("il", |_c, d: Var<FatSliceType<i64>>| d.len());
        let f = compiler.fun1("f", |_c, d: Var<FatSliceType<i64>>| call1(inner, d));
        let c = compiler.compile(f).unwrap();
        let data = [1i64, 2, 3];
        assert_eq!(c.as_fn().call(FatSlice::from_slice(&data)), 3);
    });
}

// =============================================================================
// Origin 4: descriptor field behind a `SliceRepr` witness
// (the `arrow-lms` `FfiBuffer` pattern, reproduced in-tree)
// =============================================================================

/// A `#[repr(C)]` descriptor whose first two fields are `ptr`/`len` — the shape
/// a data-layer crate witnesses with `SliceRepr`.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct Desc {
    #[staged(SPtr<i64>)]
    ptr: *const i64,
    #[staged(u64)]
    len: usize,
}

const _: () = {
    assert!(std::mem::offset_of!(Desc, ptr) == 0);
    assert!(std::mem::offset_of!(Desc, len) == 8);
};

// SAFETY: repr(C), pointer at offset 0 and a u64 element count at offset 8.
unsafe impl SliceRepr<i64> for Desc {}

/// The mutable twin of [`Desc`], witnessing a writable buffer.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct DescMut {
    #[staged(SMutPtr<i64>)]
    ptr: *mut i64,
    #[staged(u64)]
    len: usize,
}

const _: () = {
    assert!(std::mem::offset_of!(DescMut, ptr) == 0);
    assert!(std::mem::offset_of!(DescMut, len) == 8);
};

// SAFETY: repr(C) `(ptr, len)` as above, and the pointer permits writes.
unsafe impl SliceRepr<i64> for DescMut {}
unsafe impl MutSliceRepr<i64> for DescMut {}

#[test]
fn descriptor_field_read_subslice_and_iterate() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("d", |ctx, d: Var<SRef<Desc>>| {
            // SAFETY: the test keeps `data` alive across the call.
            let s = ctx.bind(unsafe { d.into_slice::<i64>() });
            let sub = ctx.bind(unsafe { s.slice_unchecked(1u64, 4u64) });
            add(
                sub.staged_iter().sum(ctx),
                int_cast::<i64, u64, _>(s.count()),
            )
        });
        let c = compiler.compile(f).unwrap();
        let data = [1i64, 2, 3, 4, 5];
        let desc = Desc {
            ptr: data.as_ptr(),
            len: data.len(),
        };
        // sub = [2,3,4] -> 9, plus len 5 -> 14
        assert_eq!(c.call(&desc), 14);
    });
}

#[test]
fn descriptor_field_mutable_write() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("dm", |ctx, d: Var<SRefMut<DescMut>>| {
            // SAFETY: the test keeps `data` alive and exclusive across the call.
            let s = unsafe { d.into_mut_slice::<i64>() };
            ctx.emit(unsafe { s.set_unchecked(1u64, 42i64) });
            Const::<i64>::new(0)
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [1i64, 2, 3];
        let mut desc = DescMut {
            ptr: data.as_mut_ptr(),
            len: data.len(),
        };
        c.call(&mut desc);
        assert_eq!(data, [1i64, 42, 3]);
    });
}
