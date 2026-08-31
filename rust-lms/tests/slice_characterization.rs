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
//! | raw descriptor `RawSlice<T>`     | OK | **G6** | OK | OK | OK | OK       | **G10** |
//! | descriptor field (`SliceRepr`)       | OK | OK  | OK | OK | OK | OK          | OK |
//!
//! Legend — `OK`: characterized by a test below. `n/a`: not meaningful for that
//! origin (a raw descriptor does not iterate by design: nothing has vouched for
//! the memory, so it must be promoted first).
//!
//! **All three gaps the matrix originally surfaced are now closed** — G6a and
//! G6b in row 6, G10 in row 10. They are recorded below because the reason each
//! existed is worth keeping.
//!
//! ## Gaps found while building this matrix
//!
//! * **G6a** — a slice *parameter* cannot be handed to an extern declared with
//!   `FatSlice<T>`/`FatSliceMut<T>`, in either direction, even though both lower
//!   to the same `(ptr, len)` argument pair. `UncheckedExternArg` witnesses
//!   `RawSlice<T> -> SRef<Slice<T>>` but not the reverse, so today a kernel
//!   must declare its parameter as `Var<RawSlice<T>>` to reach such an
//!   extern. (`ext_double_slice` in `test_extern_fn.rs` is declared but never
//!   called by any test — this is why.)
//! * **G6b — CLOSED (rows 3 + 4).** A mutable raw descriptor supported no slice
//!   operation at all, not even `len`. Row 3 classified `RawSliceMut<T>` as
//!   `SliceType + RawSliceType`; row 4 put the representation-only ops on
//!   `SliceOps`, so it now reaches them. `mut write` reads `n/a` for both raw
//!   rows because writing through a raw descriptor requires an explicit
//!   promotion to a trusted slice rather than the slice write API (row 6).
//! * **G10 — CLOSED (row 10).** `SliceIter` was bound to `SRef<Slice<T>>`, so a
//!   mutable parameter could not be iterated. Row 10 keyed it on
//!   `TrustedSliceType` instead — but the bound alone was not enough: `for_each`
//!   required `S: Clone`, and a unique slice expression deliberately is not
//!   `Clone`. It now binds once and reborrows, which also hoists the length out
//!   of the loop. Raw descriptors still do not iterate, by design.

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
#[unsafe(no_mangle)]
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
// Origin 3: raw descriptor — `RawSlice<T>`
// =============================================================================

#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn charz_sum(data: FatSlice<i64>) -> i64 {
    // SAFETY: staged code passes a live `(ptr, len)` for the duration of the call.
    unsafe { data.as_slice().iter().sum() }
}

#[test]
fn raw_descriptor_read_subslice_and_nested() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("raw", |ctx, d: Var<RawSlice<i64>>| {
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
        let f = compiler.fun1("fx", |_c, d: Var<RawSlice<i64>>| call_extern1(ext, d));
        let c = compiler.compile(f).unwrap();
        let data = [1i64, 2, 3, 4];
        assert_eq!(c.as_fn().call(FatSlice::from_slice(&data)), 10);
    });
}

#[test]
fn raw_descriptor_crosses_an_internal_call() {
    for_each_backend(|mut compiler| {
        let inner = compiler.fun1("il", |_c, d: Var<RawSlice<i64>>| d.len());
        let f = compiler.fun1("f", |_c, d: Var<RawSlice<i64>>| call1(inner, d));
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
            // SAFETY: the test keeps `data` alive and unmutated across the call.
            let s = ctx.bind(unsafe { d.into_raw_slice::<i64>().assume_shared() });
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
            let s = unsafe { d.into_raw_slice_mut::<i64>().assume_unique() };
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

/// **G6b closed (rows 3 + 4).** Before the taxonomy, `RawSliceMut<T>`
/// implemented nothing and supported no slice operation — not even `len`. Row 3
/// classified it `SliceType + RawSliceType`; row 4 moved the representation-only
/// ops onto `SliceOps`, so a mutable raw descriptor now reaches them.
///
/// It stays `RawSliceType`, so the *safe* accessors (`get_or`, `set`) remain out
/// of reach until an explicit promotion — see `slice_taxonomy.rs`.
#[test]
fn raw_mut_descriptor_has_representation_ops() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("rm", |ctx, d: Var<RawSliceMut<i64>>| {
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
                let view = ctx.bind(unsafe { d.into_raw_slice::<i64>().assume_shared() });
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
/// both drive `SliceMutOps::fill`.
#[test]
fn one_helper_serves_every_unique_origin() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("fill_both", |ctx, mut a: Var<SRefMut<Slice<i64>>>| {
            // SAFETY: the test calls this with 4 elements.
            let sub = ctx.bind(unsafe { a.reborrow().subslice_unchecked(2u64, 4u64) });
            sub.fill(ctx, 9i64);
            a.reborrow().fill(ctx, 7i64);
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

// =============================================================================
// Raw/FFI descriptor boundary — refactor plan row 6
// =============================================================================

#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn charz_g6a_sum(data: FatSlice<i64>) -> i64 {
    // SAFETY: staged code passes a live `(ptr, len)` for the duration of the call.
    unsafe { data.as_slice().iter().sum() }
}

#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn charz_g6a_double(mut data: FatSliceMut<i64>) {
    // SAFETY: staged code passes a live, exclusively owned `(ptr, len)`.
    unsafe {
        for x in data.as_slice_mut() {
            *x *= 2;
        }
    }
}

/// **G6a closed (row 6).** A shared slice *parameter* can now reach an extern
/// declared with `FatSlice<T>`. Before row 6 the `UncheckedExternArg` table
/// witnessed only `Raw* -> SRef/SRefMut`, never the reverse, so a kernel had to
/// declare its parameter as `Var<RawSlice<T>>` to call such an extern.
#[test]
fn shared_param_reaches_a_fat_slice_extern() {
    for_each_backend(|mut compiler| {
        let ext = compiler.extern_fn::<CharzG6aSumExtern>();
        let f = compiler.fun1("s", |_c, a: Var<SRef<Slice<i64>>>| {
            // SAFETY: `a` is a live shared slice for the call; unchecked only
            // because Rust slice references have no stable C ABI.
            unsafe { call_extern1_unchecked(ext, a) }
        });
        let c = compiler.compile(f).unwrap();
        assert_eq!(c.call(&[1i64, 2, 3]), 6);
    });
}

/// **G6a closed, mutable half.** This is the case `ext_double_slice` in
/// `test_extern_fn.rs` was written for and that no test could express — it was
/// declared but callable by nothing.
#[test]
fn mut_param_reaches_a_fat_slice_mut_extern() {
    for_each_backend(|mut compiler| {
        let ext = compiler.extern_fn::<CharzG6aDoubleExtern>();
        let f = compiler.fun1("m", |_c, a: Var<SRefMut<Slice<i64>>>| {
            // SAFETY: `a` is a live, exclusively owned slice for the call.
            unsafe { call_extern1_unchecked(ext, a) }
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [1i64, 2, 3];
        c.call(&mut data);
        assert_eq!(data, [2i64, 4, 6]);
    });
}

/// The provenance boundary. A raw descriptor reaches the *safe* accessors only
/// after an explicit promotion; `get_or` is unreachable before it, and a shared
/// raw cannot promote to a unique view — both proven by `compile_fail` doctests
/// on `RawSliceOps`.
#[test]
fn promotion_unlocks_the_safe_surface() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("p", |ctx, d: Var<RawSlice<i64>>| {
            // SAFETY: the test owns `data`, keeps it alive and unmutated for the
            // call, and never reallocates it.
            let trusted = ctx.bind(unsafe { d.assume_shared() });
            trusted.get_or(1u64, -1i64)
        });
        let c = compiler.compile(f).unwrap();
        let data = [7i64, 8, 9];
        assert_eq!(c.as_fn().call(FatSlice::from_slice(&data)), 8);
    });
}

/// A *mutable* raw descriptor promotes to a unique slice and then writes
/// through the ordinary `SliceMutOps` surface — the path that makes a raw
/// mutable descriptor useful at all.
#[test]
fn mutable_promotion_reaches_the_write_surface() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("pm", |ctx, d: Var<RawSliceMut<i64>>| {
            // SAFETY: the test owns the buffer and hands out no other reference.
            let mut trusted = ctx.bind(unsafe { d.assume_unique() });
            ctx.emit(unsafe { trusted.reborrow().set_unchecked(0u64, 99i64) });
            trusted.len()
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [1i64, 2, 3];
        let desc = FatSliceMut::from_slice(&mut data);
        assert_eq!(c.as_fn().call(desc), 3);
        assert_eq!(data, [99i64, 2, 3]);
    });
}

// =============================================================================
// Checked sub-slicing — refactor plan row 9
// =============================================================================
//
// `get_range` is the safe counterpart to `subslice_unchecked`: it yields a
// `StagedOpt`, `Some` when `start <= end && end <= len`. Both take `(start,
// end)` rather than a `start..end` range — `Range<Idx>` has a single index
// type, so the common mixed form `0u64 .. len` (literal start, staged end) is a
// type error at the `..`, and that is the dynamic third of the call sites.
//
// The result preserves its source's capability *and* provenance, because
// `SliceGetRange::Item = S::Out` — asserted in `slice_taxonomy.rs`.

/// Report the length of a checked sub-slice, or `-1` when the range is invalid.
/// Written once and reused by the per-origin tests below.
macro_rules! range_probe {
    ($compiler:expr, $name:literal, $param:ty) => {
        $compiler.fun3($name, |ctx, a: Var<$param>, s: Var<u64>, e: Var<u64>| {
            let out = ctx.var(-1i64);
            a.get_range(s, e).eliminate(
                ctx,
                move |ctx, sub| {
                    ctx.store(out, int_cast::<i64, u64, _>(sub.len()));
                },
                move |ctx| ctx.store(out, -1i64),
            );
            out
        })
    };
}

/// Trusted shared origin: in range, empty range, and both ways of being out of
/// range.
#[test]
fn get_range_on_a_shared_parameter() {
    for_each_backend(|mut compiler| {
        let f = range_probe!(compiler, "shared", SRef<Slice<i64>>);
        let c = compiler.compile(f).unwrap();
        let data = [10i64, 20, 30, 40];
        assert_eq!(c.call(&data, 1, 3), 2, "in range");
        assert_eq!(c.call(&data, 2, 2), 0, "empty range is valid");
        assert_eq!(c.call(&data, 0, 4), 4, "full range is valid");
        assert_eq!(c.call(&data, 3, 1), -1, "start > end");
        assert_eq!(c.call(&data, 0, 5), -1, "end > len");
        assert_eq!(c.call(&data, 5, 9), -1, "wholly past the end");
    });
}

/// Trusted unique origin. The slice expression is needed twice (`len`, then the
/// sub-slice) and a unique origin cannot clone, so this only works because
/// `get_range` binds once and reborrows.
#[test]
fn get_range_on_a_mutable_parameter() {
    for_each_backend(|mut compiler| {
        let f = range_probe!(compiler, "unique", SRefMut<Slice<i64>>);
        let c = compiler.compile(f).unwrap();
        let mut data = [10i64, 20, 30, 40];
        assert_eq!(c.call(&mut data, 1, 3), 2);
        assert_eq!(c.call(&mut data, 3, 1), -1);
        assert_eq!(c.call(&mut data, 0, 5), -1);
    });
}

/// Raw origin: checked sub-slicing needs no provenance, so it is available
/// before promotion — and the result is still raw.
#[test]
fn get_range_on_a_raw_descriptor() {
    for_each_backend(|mut compiler| {
        let f = range_probe!(compiler, "raw", RawSlice<i64>);
        let c = compiler.compile(f).unwrap();
        let data = [10i64, 20, 30, 40];
        let fat = FatSlice::from_slice(&data);
        assert_eq!(c.as_fn().call(fat, 1, 3), 2);
        assert_eq!(c.as_fn().call(fat, 3, 1), -1);
        assert_eq!(c.as_fn().call(fat, 0, 5), -1);
    });
}

/// A checked sub-slice of a *unique* origin is still unique, so it reaches the
/// writing ops — capability survives the round trip.
#[test]
fn checked_subslice_of_a_unique_origin_can_be_written() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("w", |ctx, a: Var<SRefMut<Slice<i64>>>| {
            let wrote = ctx.var(0i64);
            a.get_range(1u64, 3u64).eliminate(
                ctx,
                move |ctx, mut sub| {
                    // SAFETY: `get_range` proved the sub-slice has two elements.
                    ctx.emit(unsafe { sub.reborrow().set_unchecked(0u64, 99i64) });
                    ctx.store(wrote, 1i64);
                },
                move |ctx| ctx.store(wrote, 0i64),
            );
            wrote
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [1i64, 2, 3, 4];
        assert_eq!(c.call(&mut data), 1);
        assert_eq!(
            data,
            [1i64, 99, 3, 4],
            "wrote through the checked sub-slice"
        );
    });
}

/// Nested: a checked sub-slice of a checked sub-slice, still closed.
#[test]
fn checked_subslices_nest() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("n", |ctx, a: Var<SRef<Slice<i64>>>| {
            let out = ctx.var(-1i64);
            a.get_range(1u64, 6u64).eliminate(
                ctx,
                move |ctx, outer| {
                    outer.get_range(1u64, 3u64).eliminate(
                        ctx,
                        move |ctx, inner| {
                            // SAFETY: the inner range proved two elements.
                            ctx.store(out, unsafe { inner.get_unchecked(0u64) });
                        },
                        move |ctx| ctx.store(out, -2i64),
                    );
                },
                move |ctx| ctx.store(out, -1i64),
            );
            out
        });
        let c = compiler.compile(f).unwrap();
        // a[1..6) = [10,20,30,40,50]; its [1..3) = [20,30]; first is 20
        assert_eq!(c.call(&[0i64, 10, 20, 30, 40, 50, 60]), 20);
    });
}

// =============================================================================
// Iteration is an operation of a slice — refactor plan row 10
// =============================================================================
//
// `SliceIter` is keyed on `TrustedSliceType`, so every trusted origin drives the
// same iterator. A *raw* descriptor deliberately does not iterate — it is not
// trusted, and nothing has vouched for the memory the loop would read; that
// negative is a `compile_fail` doctest on `SliceIter`.

/// **G10 closed.** A mutable parameter iterates. This needs more than a relaxed
/// bound: a unique slice expression is not `Clone`, so `for_each` had to bind
/// once and reborrow rather than clone per use.
#[test]
fn mutable_parameter_iterates() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("mi", |ctx, a: Var<SRefMut<Slice<i64>>>| {
            a.staged_iter().sum(ctx)
        });
        let c = compiler.compile(f).unwrap();
        let mut data = [1i64, 2, 3, 4];
        assert_eq!(c.call(&mut data), 10);
    });
}

/// **G10 closed.** A promoted FFI descriptor iterates — after `assume_shared`,
/// it is an ordinary trusted slice and nothing else changes.
#[test]
fn promoted_raw_descriptor_iterates() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("pi", |ctx, d: Var<RawSlice<i64>>| {
            // SAFETY: the test owns the buffer and keeps it alive and unmutated.
            unsafe { d.assume_shared() }.staged_iter().sum(ctx)
        });
        let c = compiler.compile(f).unwrap();
        let data = [5i64, 6, 7];
        assert_eq!(c.as_fn().call(FatSlice::from_slice(&data)), 18);
    });
}

/// A sub-slice iterates, and so does a sub-slice of a *mutable* parent — the
/// capability survives both the sub-slicing and the iteration.
#[test]
fn sub_slices_iterate() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "si",
            |ctx, a: Var<SRef<Slice<i64>>>, m: Var<SRefMut<Slice<i64>>>| {
                // SAFETY: the test passes at least four elements to each.
                let shared_sub = ctx.bind(unsafe { a.subslice_unchecked(1u64, 3u64) });
                let unique_sub = ctx.bind(unsafe { m.subslice_unchecked(0u64, 2u64) });
                let x = shared_sub.staged_iter().sum(ctx);
                let y = unique_sub.staged_iter().sum(ctx);
                add(x, y)
            },
        );
        let c = compiler.compile(f).unwrap();
        let mut mdata = [100i64, 200, 300, 400];
        // shared [20,30] = 50, unique [100,200] = 300
        assert_eq!(c.call(&[10i64, 20, 30, 40], &mut mdata), 350);
    });
}

/// A checked sub-slice iterates too: `get_range` hands back the same trusted
/// capability, so the `Some` arm is an ordinary iterable slice.
#[test]
fn checked_sub_slices_iterate() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("ci", |ctx, a: Var<SRef<Slice<i64>>>| {
            let out = ctx.var(-1i64);
            a.get_range(1u64, 4u64).eliminate(
                ctx,
                move |ctx, sub| {
                    let s = sub.staged_iter().sum(ctx);
                    ctx.store(out, s);
                },
                move |ctx| ctx.store(out, -1i64),
            );
            out
        });
        let c = compiler.compile(f).unwrap();
        assert_eq!(c.call(&[10i64, 20, 30, 40, 50]), 90); // 20+30+40
    });
}
