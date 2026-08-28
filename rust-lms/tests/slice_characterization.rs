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
//! * **G6b — CLOSED (rows 3 + 4).** A mutable raw descriptor supported no slice
//!   operation at all, not even `len`. Row 3 classified `FatSliceMutType<T>` as
//!   `SliceType + RawSliceType`; row 4 put the representation-only ops on
//!   `SliceOps`, so it now reaches them. `mut write` reads `n/a` for both raw
//!   rows because writing through a raw descriptor requires an explicit
//!   promotion to a trusted slice rather than the slice write API (row 6).
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
            add(int_cast::<i64, u64, _>(mul(a.len(), 100u64)), first)
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
            let outer = ctx.bind(unsafe { a.subslice_unchecked(1u64, 6u64) }); // [1..6)
            let inner = ctx.bind(unsafe { outer.subslice_unchecked(1u64, 3u64) }); // [2..4)
            add(int_cast::<i64, u64, _>(mul(inner.len(), 1000u64)), unsafe {
                inner.get_unchecked(0u64)
            })
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
            a.subslice_unchecked(1u64, a.len())
        });
        let f = compiler.fun1("f", |ctx, a: Var<SRef<Slice<i64>>>| {
            let t = ctx.bind(call1(tail, a));
            add(int_cast::<i64, u64, _>(mul(t.len(), 10u64)), unsafe {
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
            let v = ctx.bind(unsafe { a.reborrow().get_unchecked(0u64) });
            ctx.emit(unsafe { a.reborrow().set_unchecked(1u64, add(v, 5i64)) });
            ctx.emit(unsafe { a.reborrow().swap_unchecked(0u64, 2u64) });
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
        let f = compiler.fun2("cs", |_c, a: Var<SRefMut<Slice<i64>>>, i: Var<u64>| {
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
            let outer = unsafe { a.subslice_unchecked(1u64, 5u64) }; // [1..5)
            let inner = unsafe { outer.subslice_unchecked(1u64, 3u64) }; // [2..4)
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
            ctx.emit(unsafe { a.reborrow().set_unchecked(0u64, 77i64) });
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
            let outer = ctx.bind(unsafe { d.subslice_unchecked(1u64, 6u64) });
            let inner = ctx.bind(unsafe { outer.subslice_unchecked(1u64, 3u64) });
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
            let sub = ctx.bind(unsafe { s.subslice_unchecked(1u64, 4u64) });
            add(sub.staged_iter().sum(ctx), int_cast::<i64, u64, _>(s.len()))
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

/// **G6b closed (rows 3 + 4).** Before the taxonomy, `FatSliceMutType<T>`
/// implemented nothing and supported no slice operation — not even `len`. Row 3
/// classified it `SliceType + RawSliceType`; row 4 moved the representation-only
/// ops onto `SliceOps`, so a mutable raw descriptor now reaches them.
///
/// It stays `RawSliceType`, so the *safe* accessors (`get_or`, `set`) remain out
/// of reach until an explicit promotion — see `slice_taxonomy.rs`.
#[test]
fn raw_mut_descriptor_has_representation_ops() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("rm", |ctx, d: Var<FatSliceMutType<i64>>| {
            let mut sub = ctx.bind(unsafe { d.subslice_unchecked(1u64, 3u64) });
            let n = ctx.bind(sub.reborrow().len());
            // SAFETY: the descriptor covers [1, 3), so index 0 of the sub-slice
            // is in bounds, and the test passes a live buffer.
            let first = ctx.bind(unsafe { sub.get_unchecked(0u64) });
            add(int_cast::<i64, u64, _>(mul(n, 100u64)), first)
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [0i64, 10, 20, 30];
        let desc = FatSliceMut::from_slice(&mut data);
        // sub = [10, 20] -> len 2 => 200 + 10
        assert_eq!(c.as_fn().call(desc), 210);
    });
}

// =============================================================================
// Origin independence — refactor plan row 5
// =============================================================================
//
// The point of the consolidation: once a value is a trusted slice, its storage
// origin is irrelevant. These helpers are written *once*, against the capability
// traits, and every origin below feeds the same code.

// A generic slice helper must **bind once and reborrow**, not clone per use.
// Unique slice expressions are deliberately not `Clone` — that is the
// uniqueness guarantee — so a `S: Clone` bound would silently restrict the
// helper to shared origins. Binding the expression into a variable costs one
// use, and every later use is a reborrow, which works for both capabilities.

/// Sum a trusted `i64` slice, whatever it came from — shared *or* unique.
fn total<S>(ctx: &mut Ctx, s: S) -> Var<i64>
where
    S: Staged + 'static,
    S::Out: TrustedSliceType<Elem = i64>,
{
    let mut v = ctx.bind(s);
    let n = ctx.bind(v.reborrow().len());
    let acc = ctx.var(0i64);
    let i = ctx.var(0u64);
    ctx.while_loop(lt(i, n), move |ctx| {
        // SAFETY: the loop condition proves `i < len`.
        ctx.store(acc, add(acc, unsafe { v.reborrow().get_unchecked(i) }));
        ctx.store(i, add(i, 1u64));
    });
    acc
}

/// Fill a *unique* `i64` slice, whatever it came from. Accepts only unique
/// origins — `S::Out: MutSliceType`.
fn fill<S>(ctx: &mut Ctx, s: S, value: i64)
where
    S: Staged + 'static,
    S::Out: MutSliceType<Elem = i64>,
{
    let mut v = ctx.bind(s);
    let n = ctx.bind(v.reborrow().len());
    let i = ctx.var(0u64);
    ctx.while_loop(lt(i, n), move |ctx| {
        // SAFETY: the loop condition proves `i < len`.
        ctx.emit(unsafe { v.reborrow().set_unchecked(i, value) });
        ctx.store(i, add(i, 1u64));
    });
}

/// One helper, three trusted shared origins: a function parameter, a sub-slice
/// of it, and a witnessed descriptor field.
#[test]
fn one_helper_serves_every_trusted_shared_origin() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "origins",
            |ctx, a: Var<SRef<Slice<i64>>>, d: Var<SRef<Desc>>| {
                let from_param = total(ctx, a);
                // SAFETY: the test calls this with at least 3 elements.
                let sub = ctx.bind(unsafe { a.subslice_unchecked(1u64, 3u64) });
                let from_subslice = total(ctx, sub);
                // SAFETY: the test keeps the descriptor's buffer alive.
                let view = ctx.bind(unsafe { d.into_slice::<i64>() });
                let from_descriptor = total(ctx, view);
                add(add(from_param, from_subslice), from_descriptor)
            },
        );
        let c = compiler.compile(f).unwrap();
        let data = [1i64, 2, 3, 4];
        let other = [10i64, 20];
        let desc = Desc {
            ptr: other.as_ptr(),
            len: other.len(),
        };
        // 10 (param) + 5 (sub [2,3]) + 30 (descriptor) = 45
        assert_eq!(c.call(&data, &desc), 45);
    });
}

/// The same for the unique side: a mutable parameter and a mutable sub-slice
/// both drive one `fill`.
#[test]
fn one_helper_serves_every_unique_origin() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("fill_both", |ctx, mut a: Var<SRefMut<Slice<i64>>>| {
            // SAFETY: the test calls this with 4 elements.
            let sub = ctx.bind(unsafe { a.reborrow().subslice_unchecked(2u64, 4u64) });
            fill(ctx, sub, 9i64);
            fill(ctx, a.reborrow(), 7i64);
            // `total` is capability-independent: the same helper that served the
            // three shared origins above also takes a unique one.
            let _ = total(ctx, a.reborrow());
            a.len()
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [0i64; 4];
        assert_eq!(c.call(&mut data), 4);
        // the whole-slice fill runs second, so 7 wins everywhere
        assert_eq!(data, [7i64, 7, 7, 7]);
    });
}
