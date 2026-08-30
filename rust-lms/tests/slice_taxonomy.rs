//! # Slice capability taxonomy assertions — refactor plan row 3
//!
//! Row 3's exit criterion: *compile-time trait assertions cover every staged
//! slice marker*. Everything below is checked by `rustc` at compile time; the
//! `#[test]` exists only so the file reports in the suite.
//!
//! | Marker | `SliceType` | `TrustedSliceType` | `MutSliceType` | `RawSliceType` |
//! |---|:---:|:---:|:---:|:---:|
//! | `SRef<Slice<T>>`      | yes | yes | no  | no  |
//! | `SRefMut<Slice<T>>`   | yes | yes | yes | no  |
//! | `RawSlice<T>`     | yes | no  | no  | yes |
//! | `RawSliceMut<T>`  | yes | no  | no  | yes |
//!
//! The `no` cells are the load-bearing half — they are proven by `compile_fail`
//! doctests on the traits themselves in `slice.rs`, since a negative trait bound
//! cannot be asserted positively here.

// Every item here is a *compile-time* assertion: rustc checking the bound is
// the whole test, so nothing is ever called at runtime.
#![allow(dead_code)]

use rust_lms::prelude::*;

fn assert_slice<S: SliceType>() {}
fn assert_trusted<S: TrustedSliceType>() {}
fn assert_writable<S: MutSliceType>() {}
fn assert_raw<S: RawSliceType>() {}

/// Every marker has a representation.
fn _representation() {
    assert_slice::<SRef<Slice<i64>>>();
    assert_slice::<SRefMut<Slice<i64>>>();
    assert_slice::<RawSlice<i64>>();
    assert_slice::<RawSliceMut<i64>>();
}

/// Only the reference forms carry established validity.
fn _trusted() {
    assert_trusted::<SRef<Slice<i64>>>();
    assert_trusted::<SRefMut<Slice<i64>>>();
}

/// Only the unique reference form may write through the slice API.
fn _writable() {
    assert_writable::<SRefMut<Slice<i64>>>();
}

/// Both descriptor forms are raw — including the mutable one, which before the
/// taxonomy implemented nothing at all (gap G6b from the row-0 matrix).
fn _raw() {
    assert_raw::<RawSlice<i64>>();
    assert_raw::<RawSliceMut<i64>>();
}

// -----------------------------------------------------------------------------
// Associated types: mutability is visible in the projections, not just the bound
// -----------------------------------------------------------------------------

fn assert_elem<S: SliceType<Elem = E>, E>() {}
fn assert_data_ptr<S: SliceType<DataPtr = P>, P>() {}
fn assert_elem_ref<S: TrustedSliceType<ElemRef = R>, R>() {}

fn _projections() {
    assert_elem::<SRef<Slice<i64>>, i64>();
    assert_elem::<SRefMut<Slice<f64>>, f64>();
    assert_elem::<RawSlice<u8>, u8>();
    assert_elem::<RawSliceMut<u8>, u8>();

    // A shared origin projects a const pointer, a unique one a mut pointer —
    // and that holds for raw descriptors too.
    assert_data_ptr::<SRef<Slice<i64>>, SPtr<i64>>();
    assert_data_ptr::<SRefMut<Slice<i64>>, SMutPtr<i64>>();
    assert_data_ptr::<RawSlice<i64>, SPtr<i64>>();
    assert_data_ptr::<RawSliceMut<i64>, SMutPtr<i64>>();

    // Only trusted slices yield a *reference* to an element.
    assert_elem_ref::<SRef<Slice<i64>>, SRef<i64>>();
    assert_elem_ref::<SRefMut<Slice<i64>>, SRefMut<i64>>();
}

// -----------------------------------------------------------------------------
// Closure: a sub-slice keeps its origin's capability and provenance
// -----------------------------------------------------------------------------

/// `SliceSliceUnchecked` reports `Out = S::Out`. That single equality is what
/// makes "a slice of a slice is a slice" true *without laundering provenance*:
/// a sub-slice of a raw descriptor stays raw, and a sub-slice of a `&mut [T]`
/// stays writable. Asserted directly rather than by example.
fn _sub_slicing_is_closed<S, START, END>()
where
    S: Staged,
    S::Out: SliceType,
    START: Staged<Out = u64>,
    END: Staged<Out = u64>,
{
    fn out_is<A: Staged<Out = O>, O>() {}
    out_is::<SliceSliceUnchecked<S, START, END>, S::Out>();
}

/// The checked variant is closed the same way: `SliceGetRange::Item = S::Out`,
/// so `get_range` on a raw descriptor yields a raw sub-slice and on a `&mut [T]`
/// yields a writable one. Capability *and* provenance survive the bounds check.
fn _checked_sub_slicing_is_closed<S, START, END>()
where
    S: Staged + 'static,
    S::Out: SliceType + 'static,
    START: Staged<Out = u64> + 'static,
    END: Staged<Out = u64> + 'static,
{
    fn item_is<A: StagedOpt<Item = I>, I>() {}
    item_is::<SliceGetRange<S, START, END>, S::Out>();
}

#[test]
fn taxonomy_assertions_hold() {
    // The assertions above are compile-time; reaching this line means rustc
    // accepted every positive classification.
}
