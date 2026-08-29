//! Slice support for staged computations.
//!
//! This module provides:
//! - `Slice<T>`: Marker type for dynamically-sized slices (DST)
//! - `SRef<Slice<T>>`: Immutable slice reference (`&[T]`) - 16-byte fat pointer
//! - `SRefMut<Slice<T>>`: Mutable slice reference (`&mut [T]`) - 16-byte fat pointer
//!
//! # Fat Pointer Layout
//!
//! A slice reference is a fat pointer with layout:
//! ```text
//! offset 0: ptr (*const T / *mut T) - pointer to first element
//! offset 8: len (usize)             - number of elements
//! ```
//!
//! At generated-function boundaries, a pointer to this descriptor is passed.
//!
//! # Canonical Staged representation
//!
//! Within the staged graph, a slice's `codegen` value is `Value::Fat { ptr, len }`.
//! Both leaves remain in SSA registers through variables, sub-slicing, and slice ops.
//! The pair is materialized as a descriptor only when it crosses the private function or
//! extern ABI, and loaded back into a fat value on entry.
//!
//! # Example
//!
//! ```ignore
//! // Sum all elements in a slice
//! let sum = compiler.fun1("sum", |arr: Var<SRef<Slice<i64>>>| {
//!     let i = compiler.let_var(0u64);
//!     let total = compiler.let_var(0i64);
//!
//!     (
//!         i,
//!         total,
//!         while_loop(
//!             lt(i, arr.len()),
//!             (
//!                 assign(total, add(total, arr.get_unchecked(i))),
//!                 assign(i, add(i, 1u64)),
//!             )
//!         ),
//!         total
//!     )
//! });
//! ```

use crate::ffi::{RawSlice, RawSliceMut};
use crate::func::Ctx;
use crate::r#struct::{Field, FieldAddr, MutField};
use crate::refer::{SMutPtr, SPtr, SRef, SRefMut};
use crate::staged::{CompilationContext, IntoStaged, Staged, Value, ValueId, Var, VarUse};
use crate::staged_opt::StagedOpt;
use crate::types::{
    CopyType, DirectValue, IntCmp, RuntimeParam, RuntimeResult, ScalarType, StagedType,
};
use std::marker::PhantomData;

// =============================================================================
// Slice<T>: DST Marker Type
// =============================================================================

/// Marker type for dynamically-sized slices.
///
/// `Slice<T>` is a DST (dynamically sized type) and cannot exist by itself.
/// It must always be behind a reference:
/// - `SRef<Slice<T>>` = `&[T]`
/// - `SRefMut<Slice<T>>` = `&mut [T]`
///
/// Note: `Slice<T>` intentionally does NOT implement `StagedType`.
/// This ensures that `Var<Slice<T>>` is not valid (just like `[T]` in Rust).
/// Only `Var<SRef<Slice<T>>>` and `Var<SRefMut<Slice<T>>>` are valid.
#[derive(Clone, Copy, Debug)]
pub struct Slice<T: StagedType> {
    _phantom: PhantomData<T>,
}

// =============================================================================
// AsSlice: view any repr-compatible `(ptr, len)` value as a staged slice
// =============================================================================

/// Unsafe layout witness for staged types that can be decoded as a slice
/// descriptor for `T`.
///
/// This trait witnesses only the representation layout. A particular
/// descriptor's pointer validity, alignment, element count, and lifetime are
/// checked by the caller of [`ReprSliceOps::as_slice`].
///
/// # Safety
///
/// The staged representation of `Self` must store a pointer at byte offset 0
/// and a `u64` element count at byte offset 8. The pointer must have the same
/// representation as a pointer to `T`.
///
/// # Why this is an open `unsafe trait`, not sealed
///
/// This is deliberately implementable by downstream crates, and it must stay
/// that way: the only implementors today are `arrow-lms`'s `FfiBuffer` /
/// `FfiBufferMut`, and `arrow-lms` is a *separate* crate. Sealing (a private
/// supertrait) would confine impls to `rust-lms` itself and break the intended
/// pattern where a data-layer crate defines its own `#[repr(C)]` descriptor and
/// witnesses its layout. `unsafe` — plus the `unsafe fn as_slice` on the read
/// side — is the whole safety boundary: an implementor must uphold the offset/
/// representation contract above under `unsafe impl`, and no *safe* code can
/// reinterpret an arbitrary `SRef<R>` as a slice (proven by the `compile_fail`
/// doctest on `ReprSliceOps::as_slice`). Do not seal this without first moving
/// every descriptor type into `rust-lms`.
pub unsafe trait SliceRepr<T: StagedType>: StagedType {}

/// Unsafe layout witness for staged types that can also be decoded as a
/// mutable slice descriptor for `T`.
///
/// # Safety
///
/// In addition to the requirements of [`SliceRepr`], the pointer field must
/// have a representation that permits writes. Whether a particular descriptor
/// is exclusively writable remains the caller's responsibility.
pub unsafe trait MutSliceRepr<T: StagedType>: SliceRepr<T> {}

/// Re-types a reference to a repr-compatible `(ptr, len)` value as a staged
/// `Slice<T>`.
///
/// This emits no code of its own: it forwards the address of the representation
/// unchanged and lets the normal slice operations load `ptr` and `len` from
/// offsets 0 and 8. Use it for `#[repr(C)]` FFI descriptors whose first two
/// fields are pointer-sized `ptr` and `len` values.
pub struct AsSlice<P, T> {
    repr: P,
    _elem: PhantomData<T>,
}

/// Re-types a raw pointer to a repr-compatible `(ptr, len)` descriptor as a
/// lifetime-free staged [`RawSlice<T>`].
pub struct AsRawSlice<P, T> {
    repr: P,
    _elem: PhantomData<T>,
}

impl<P: Clone, T> Clone for AsRawSlice<P, T> {
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, T> Copy for AsRawSlice<P, T> {}

unsafe impl<P, R, T> Staged for AsRawSlice<P, T>
where
    P: Staged<Out = SPtr<R>>,
    R: SliceRepr<T>,
    T: StagedType,
{
    type Out = RawSlice<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Reinterpret the pointed-to {ptr,len} descriptor as a slice: load it into a fat value.
        let base = self.repr.codegen(ctx).leaf();
        ctx.load_fat(base)
    }
}

/// Extension trait for raw pointers to repr-compatible slice descriptors.
pub trait ReprRawSliceOps<R>: Staged<Out = SPtr<R>> + Sized
where
    R: StagedType,
{
    /// Interpret the pointed-to descriptor as a lifetime-free raw slice.
    ///
    /// # Safety
    ///
    /// The descriptor must contain a pointer that is live and aligned for
    /// reads of `len` initialized values of `T` for every generated-code use.
    unsafe fn into_raw_slice<T>(self) -> AsRawSlice<Self, T>
    where
        T: StagedType,
        R: SliceRepr<T>,
    {
        AsRawSlice {
            repr: self,
            _elem: PhantomData,
        }
    }
}

impl<R, S> ReprRawSliceOps<R> for S
where
    R: StagedType,
    S: Staged<Out = SPtr<R>> + Sized,
{
}

impl<P: Clone, T> Clone for AsSlice<P, T> {
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, T> Copy for AsSlice<P, T> {}

unsafe impl<P, R, T> Staged for AsSlice<P, T>
where
    P: Staged<Out = SRef<R>>,
    R: SliceRepr<T>,
    T: StagedType,
{
    /// **Raw**, not trusted. Reading a `(ptr, len)` out of a valid `&R` is
    /// sound, but says nothing about the buffer that pointer addresses — that
    /// claim belongs to [`RawSliceOps::assume_shared`].
    type Out = RawSlice<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Reinterpret the pointed-to {ptr,len} descriptor as a slice: load it into a fat value.
        let base = self.repr.codegen(ctx).leaf();
        ctx.load_fat(base)
    }
}

/// Extension trait for values that point at a witnessed `(ptr, len)`
/// representation.
///
/// Ordinary staged references cannot be reinterpreted as slices:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn arbitrary_value_is_not_a_slice(value: Var<SRef<i64>>) {
///     let _ = unsafe { value.as_slice::<u8>() };
/// }
/// ```
pub trait ReprSliceOps<R>: Staged<Out = SRef<R>> + Sized
where
    R: StagedType,
{
    /// Read the descriptor as a **raw** slice of `T`.
    ///
    /// Safe: the receiver is a valid reference, so loading its `(ptr, len)`
    /// dereferences nothing unproven, and the result makes no claim about the
    /// buffer. Crossing into a trusted slice is
    /// [`RawSliceOps::assume_shared`], which is where the contract lives.
    fn into_raw_slice<T>(self) -> AsSlice<Self, T>
    where
        T: StagedType,
        R: SliceRepr<T>,
    {
        AsSlice {
            repr: self,
            _elem: PhantomData,
        }
    }
}

impl<R, S> ReprSliceOps<R> for S
where
    R: StagedType,
    S: Staged<Out = SRef<R>> + Sized,
{
}

/// Extension trait for *mutable* references to a repr-compatible `(ptr, len)`.
///
/// The mutable twin of [`ReprSliceOps`]: given a `&mut` to an FFI descriptor
/// whose first two fields are `ptr`/`len`, reinterpret it as a `&mut [T]`.
pub trait ReprSliceMutOps<R>: Staged<Out = SRefMut<R>> + Sized
where
    R: StagedType,
{
    /// Read the descriptor as a **raw mutable** slice of `T`. Safe for the
    /// same reason as [`ReprSliceOps::into_raw_slice`]; promote with
    /// [`RawSliceOps::assume_unique`].
    fn into_raw_slice_mut<T>(self) -> AsMutSlice<Self, T>
    where
        T: StagedType,
        R: MutSliceRepr<T>,
    {
        AsMutSlice {
            repr: self,
            _elem: PhantomData,
        }
    }
}

impl<R, S> ReprSliceMutOps<R> for S
where
    R: StagedType,
    S: Staged<Out = SRefMut<R>> + Sized,
{
}

/// Re-types a *mutable* reference to a repr-compatible `(ptr, len)` value as a
/// staged `&mut [T]`. Like [`AsSlice`] it emits no code — it forwards the
/// address and lets the slice ops load `ptr`/`len` from offsets 0/8.
pub struct AsMutSlice<P, T> {
    repr: P,
    _elem: PhantomData<T>,
}

impl<P: Clone, T> Clone for AsMutSlice<P, T> {
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, T> Copy for AsMutSlice<P, T> {}

unsafe impl<P, R, T> Staged for AsMutSlice<P, T>
where
    P: Staged<Out = SRefMut<R>>,
    R: MutSliceRepr<T>,
    T: StagedType,
{
    /// **Raw**, not trusted — see [`AsSlice`]. Promote with
    /// [`RawSliceOps::assume_unique`].
    type Out = RawSliceMut<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Reinterpret the pointed-to {ptr,len} descriptor as a slice: load it into a fat value.
        let base = self.repr.codegen(ctx).leaf();
        ctx.load_fat(base)
    }
}

// =============================================================================
// StagedType for SRef<Slice<T>> - Immutable Fat Pointer
// =============================================================================

unsafe impl<T: StagedType> StagedType for SRef<Slice<T>> {
    /// A staged slice reference is a `(ptr, len)` descriptor at runtime; the
    /// safe, lifetime-bounded view is `RuntimeResult::Output<'call>`.
    type RuntimeValue = *const [T::RuntimeValue];

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }

    fn size_of() -> usize {
        16 // ptr (8) + len (8)
    }

    fn align_of() -> usize {
        8
    }

    fn is_copy_struct() -> bool {
        true // Fat pointer is Copy
    }

    fn is_fat_pointer() -> bool {
        true // Slice references are fat pointers
    }
}

unsafe impl<T: StagedType> CopyType for SRef<Slice<T>> {}

unsafe impl<T> RuntimeParam for SRef<Slice<T>>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = &'call [T::RuntimeValue];
}

unsafe impl<T> RuntimeResult for SRef<Slice<T>>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Output<'call> = &'call [T::RuntimeValue];
}

// =============================================================================
// StagedType for SRefMut<Slice<T>> - Mutable Fat Pointer
// =============================================================================

unsafe impl<T: StagedType> StagedType for SRefMut<Slice<T>> {
    type RuntimeValue = *mut [T::RuntimeValue];

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }

    fn size_of() -> usize {
        16
    }

    fn align_of() -> usize {
        8
    }

    fn is_copy_struct() -> bool {
        true // ABI-classified as a two-register aggregate; not semantically Copy.
    }

    fn is_fat_pointer() -> bool {
        true // Mutable slice references are fat pointers
    }
}

unsafe impl<T> RuntimeParam for SRefMut<Slice<T>>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = &'call mut [T::RuntimeValue];
}

unsafe impl<T> RuntimeResult for SRefMut<Slice<T>>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Output<'call> = &'call mut [T::RuntimeValue];
}

// =============================================================================
// Capability taxonomy: representation, trusted provenance, writability
// =============================================================================
//
// Four *different facts* about a staged slice, each its own trait, because a
// `(ptr, len)` pair alone proves none of the others:
//
// ```text
//   SliceType            representation: two words, an element type, a data pointer
//   |
//   +-- TrustedSliceType validity + provenance established -> element access is safe
//   |     |
//   |     +-- MutSliceType   ... and writes are permitted and exclusive
//   |
//   +-- RawSliceType     provenance unknown -> element access stays `unsafe`
// ```
//
// `len` and the data pointer are available on any `SliceType`: reading the
// descriptor you already hold dereferences nothing. Everything that *touches
// memory* requires `TrustedSliceType`, and writing additionally requires
// `MutSliceType`. A raw descriptor reaches the trusted half only through an
// explicit unsafe promotion — never implicitly.

mod slice_type_sealed {
    pub trait Sealed {}
    pub trait TrustedSealed: Sealed {}
    pub trait MutableSealed: TrustedSealed {}
    pub trait RawSealed: Sealed {}
}

/// A staged slice's **representation**: a `(ptr, len)` pair with an element
/// type. Says nothing about whether the pointer is valid or writable — see
/// [`TrustedSliceType`] and [`MutSliceType`].
///
/// This is what keeps slices *closed under sub-slicing*: the sub-slice node
/// reports `Out = S::Out`, so slicing a `&mut [T]` yields a `&mut [T]` and
/// slicing a raw descriptor yields a raw descriptor — capability and provenance
/// are preserved rather than laundered.
///
/// The trait is sealed; only representations supplied by this crate participate
/// in slice lowering.
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// #[repr(C)]
/// #[derive(Clone, Copy, StagedType)]
/// struct FabricatedSlice {
///     ptr: u64,
///     len: u64,
/// }
///
/// impl SliceType for FabricatedSlice {
///     type Elem = u8;
///     type DataPtr = SPtr<u8>;
/// }
/// ```
pub trait SliceType: StagedType + slice_type_sealed::Sealed {
    /// Element type (`T`).
    type Elem: StagedType;
    /// Raw pointer produced by `as_ptr` / `as_mut_ptr`.
    type DataPtr: StagedType;
}

/// A slice whose pointer is **known valid** for its element count, so element
/// access is an ordinary safe operation rather than an unsafe one.
///
/// Carries [`ElemRef`](TrustedSliceType::ElemRef) because only a trusted slice
/// can yield a *reference* to an element; a raw descriptor can offer no more
/// than a pointer.
///
/// A raw descriptor cannot be treated as trusted:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn trusted_only<S: TrustedSliceType>() {}
///
/// fn raw_is_not_trusted() {
///     trusted_only::<RawSlice<i64>>();
/// }
/// ```
pub trait TrustedSliceType: SliceType + slice_type_sealed::TrustedSealed {
    /// Reference-to-element produced by `get_ref_unchecked`:
    /// `SRef<T>` for a shared slice, `SRefMut<T>` for a unique one.
    type ElemRef: StagedType;
}

/// A trusted slice that is additionally **writable and exclusive**. Gates every
/// mutating op (`set`, `set_unchecked`, `swap_unchecked`) so they cannot be
/// reached from a shared slice.
///
/// Writability alone is not enough — a raw *mutable* descriptor is writable but
/// not trusted, so it deliberately does not implement this:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn writable_only<S: MutSliceType>() {}
///
/// fn raw_mut_is_not_writable_through_the_slice_api() {
///     writable_only::<RawSliceMut<i64>>();
/// }
/// ```
pub trait MutSliceType: TrustedSliceType + slice_type_sealed::MutableSealed {}

/// A slice descriptor of **unknown provenance** — an FFI return, or a
/// `(ptr, len)` read out of a foreign struct. It has a representation, so `len`,
/// the data pointer, and sub-slicing all work, but nothing may dereference it
/// without an explicit unsafe step.
///
/// Disjoint from [`TrustedSliceType`] in practice: both sit under private
/// sealed super-traits, so only this crate could implement them, and no marker
/// implements both. `tests/slice_taxonomy.rs` asserts the full classification.
pub trait RawSliceType: SliceType + slice_type_sealed::RawSealed {}

// --- `&[T]` — trusted, shared -------------------------------------------------

impl<T: StagedType> SliceType for SRef<Slice<T>> {
    type Elem = T;
    type DataPtr = SPtr<T>;
}
impl<T: StagedType> TrustedSliceType for SRef<Slice<T>> {
    type ElemRef = SRef<T>;
}
impl<T: StagedType> slice_type_sealed::Sealed for SRef<Slice<T>> {}
impl<T: StagedType> slice_type_sealed::TrustedSealed for SRef<Slice<T>> {}

// --- `&mut [T]` — trusted, unique, writable -----------------------------------

impl<T: StagedType> SliceType for SRefMut<Slice<T>> {
    type Elem = T;
    type DataPtr = SMutPtr<T>;
}
impl<T: StagedType> TrustedSliceType for SRefMut<Slice<T>> {
    type ElemRef = SRefMut<T>;
}
impl<T: StagedType> MutSliceType for SRefMut<Slice<T>> {}
impl<T: StagedType> slice_type_sealed::Sealed for SRefMut<Slice<T>> {}
impl<T: StagedType> slice_type_sealed::TrustedSealed for SRefMut<Slice<T>> {}
impl<T: StagedType> slice_type_sealed::MutableSealed for SRefMut<Slice<T>> {}

// --- raw shared descriptor ----------------------------------------------------

impl<T: StagedType> SliceType for RawSlice<T> {
    type Elem = T;
    type DataPtr = SPtr<T>;
}
impl<T: StagedType> RawSliceType for RawSlice<T> {}
impl<T: StagedType> slice_type_sealed::Sealed for RawSlice<T> {}
impl<T: StagedType> slice_type_sealed::RawSealed for RawSlice<T> {}

// --- raw mutable descriptor ---------------------------------------------------
//
// Half of gap G6b from the row-0 characterization matrix: before the taxonomy
// this marker implemented nothing, so a mutable raw descriptor supported no
// slice operation at all — not even `len`. Classifying it here makes every op
// *node* accept it; the other half is row 4, where `RawSliceOps` stops being
// hard-bound to `RawSlice<T>` so a call site can actually reach them.
//
// It is `RawSliceType`, not `MutSliceType`: the pointer permits writes, but
// provenance is unproven, so writing goes through an explicit promotion to a
// trusted slice rather than through the slice write API.

impl<T: StagedType> SliceType for RawSliceMut<T> {
    type Elem = T;
    type DataPtr = SMutPtr<T>;
}
impl<T: StagedType> RawSliceType for RawSliceMut<T> {}
impl<T: StagedType> slice_type_sealed::Sealed for RawSliceMut<T> {}
impl<T: StagedType> slice_type_sealed::RawSealed for RawSliceMut<T> {}

/// Convenience accessor for `S`'s element type inside generic op impls.
type ElemOf<S> = <<S as Staged>::Out as SliceType>::Elem;

/// Emit `data_ptr + index * sizeof(Elem)`, the address of element `index`.
fn element_addr<S>(ctx: &mut CompilationContext, data_ptr: ValueId, index: ValueId) -> ValueId
where
    S: Staged,
    S::Out: SliceType,
{
    let element_size = ElemOf::<S>::size_of() as i64;
    let scale = ctx.iconst(ScalarType::I64, element_size);
    let byte_offset = ctx.imul(index, scale);
    ctx.ptr_offset_bytes(data_ptr, byte_offset)
}

// =============================================================================
// SliceLen: Get length of a slice
// =============================================================================

/// Get the length of a slice (immutable or mutable).
#[derive(Clone, Copy)]
pub struct SliceLen<S> {
    slice: S,
}

unsafe impl<S> Staged for SliceLen<S>
where
    S: Staged,
    S::Out: SliceType,
{
    type Out = u64;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        Value::scalar(ctx.slice_len(&self.slice))
    }
}

// =============================================================================
// SliceAsPtr: Get raw pointer to slice data
// =============================================================================

/// Get the raw data pointer of a slice. Empty slices are valid inputs, so this
/// deliberately does not claim to produce a Rust reference marker.
#[derive(Clone, Copy)]
pub struct SliceAsPtr<S> {
    slice: S,
}

unsafe impl<S> Staged for SliceAsPtr<S>
where
    S: Staged,
    S::Out: SliceType,
{
    type Out = <S::Out as SliceType>::DataPtr;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        Value::scalar(ctx.slice_data_ptr(&self.slice))
    }
}

// =============================================================================
// SliceGetRefUnchecked: Get reference to element (no bounds check)
// =============================================================================

/// Get a reference to an element without bounds checking. Mutability follows
/// the slice: `SRef<T>` for `&[T]`, `SRefMut<T>` for `&mut [T]`.
#[derive(Clone, Copy)]
pub struct SliceGetRefUnchecked<S, I> {
    slice: S,
    index: I,
}

unsafe impl<S, I> Staged for SliceGetRefUnchecked<S, I>
where
    S: Staged,
    // A *reference* to an element requires trusted provenance; a raw descriptor
    // can only ever yield a pointer (`SliceGetPtrUnchecked`).
    S::Out: TrustedSliceType,
    I: Staged<Out = u64>,
{
    type Out = <S::Out as TrustedSliceType>::ElemRef;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let index = self.index.codegen(ctx);
        let data_ptr = ctx.slice_data_ptr(&self.slice);
        Value::scalar(element_addr::<S>(ctx, data_ptr, index.leaf()))
    }
}

/// Get a lifetime-free raw pointer to an element without bounds checking.
#[derive(Clone, Copy)]
pub struct SliceGetPtrUnchecked<S, I> {
    slice: S,
    index: I,
}

unsafe impl<S, I> Staged for SliceGetPtrUnchecked<S, I>
where
    S: Staged,
    S::Out: SliceType,
    I: Staged<Out = u64>,
{
    type Out = SPtr<ElemOf<S>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let index = self.index.codegen(ctx);
        let data_ptr = ctx.slice_data_ptr(&self.slice);
        Value::scalar(element_addr::<S>(ctx, data_ptr, index.leaf()))
    }
}

/// Get a raw element pointer from any staged slice representation.
///
/// # Safety
///
/// At execution, `index` must be less than `slice`'s element count. Any later
/// dereference must also satisfy the source storage's lifetime and aliasing
/// requirements.
pub unsafe fn slice_get_ptr_unchecked<S, I>(
    slice: S,
    index: I,
) -> SliceGetPtrUnchecked<S, I::Staged>
where
    S: Staged,
    S::Out: SliceType,
    I: IntoStaged<u64>,
{
    SliceGetPtrUnchecked {
        slice,
        index: index.into_staged(),
    }
}

// =============================================================================
// SliceGetUnchecked: Get element by value (no bounds check, CopyType only)
// =============================================================================

/// Get an element by value without bounds checking (`CopyType` elements only).
#[derive(Clone, Copy)]
pub struct SliceGetUnchecked<S, I> {
    slice: S,
    index: I,
}

// =============================================================================
// Bounds-checked scalar element operations
// =============================================================================

/// Read a scalar element when `index < len`, otherwise evaluate `default`.
pub struct SliceGetOr<S, I, D> {
    slice: S,
    index: I,
    default: D,
}

unsafe impl<S, I, D> Staged for SliceGetOr<S, I, D>
where
    S: Staged,
    S::Out: SliceType,
    ElemOf<S>: DirectValue,
    I: Staged<Out = u64>,
    D: Staged<Out = ElemOf<S>>,
{
    type Out = ElemOf<S>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let index = self.index.codegen(ctx);
        let (data_ptr, len) = ctx.slice_parts(&self.slice);
        let in_bounds = ctx.icmp(IntCmp::Ult, index.leaf(), len);

        let get_block = ctx.create_block();
        let default_block = ctx.create_block();
        let merge_block = ctx.create_block();
        ctx.append_block_param(merge_block, ElemOf::<S>::scalar_type());
        ctx.brif(in_bounds, get_block, &[], default_block, &[]);

        ctx.switch_to_block(get_block);
        ctx.seal_block(get_block);
        let element_ptr = element_addr::<S>(ctx, data_ptr, index.leaf());
        let value = ctx.load(ElemOf::<S>::scalar_type(), element_ptr, 0);
        ctx.jump(merge_block, &[value]);

        ctx.switch_to_block(default_block);
        ctx.seal_block(default_block);
        let default = self.default.codegen(ctx);
        ctx.jump(merge_block, &[default.leaf()]);

        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);
        Value::scalar(ctx.block_param(merge_block, 0, ElemOf::<S>::scalar_type()))
    }
}

/// Write a scalar element when `index < len`, returning whether the write ran.
pub struct SliceSet<S, I, V> {
    slice: S,
    index: I,
    value: V,
}

unsafe impl<S, I, V> Staged for SliceSet<S, I, V>
where
    S: Staged,
    S::Out: MutSliceType,
    ElemOf<S>: DirectValue,
    I: Staged<Out = u64>,
    V: Staged<Out = ElemOf<S>>,
{
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let index = self.index.codegen(ctx);
        let (data_ptr, len) = ctx.slice_parts(&self.slice);
        let in_bounds = ctx.icmp(IntCmp::Ult, index.leaf(), len);

        let set_block = ctx.create_block();
        let out_of_bounds_block = ctx.create_block();
        let merge_block = ctx.create_block();
        ctx.append_block_param(merge_block, ScalarType::Bool);
        ctx.brif(in_bounds, set_block, &[], out_of_bounds_block, &[]);

        ctx.switch_to_block(set_block);
        ctx.seal_block(set_block);
        let value = self.value.codegen(ctx);
        let element_ptr = element_addr::<S>(ctx, data_ptr, index.leaf());
        ctx.store(value.leaf(), element_ptr, 0);
        let written = ctx.iconst(ScalarType::Bool, 1);
        ctx.jump(merge_block, &[written]);

        ctx.switch_to_block(out_of_bounds_block);
        ctx.seal_block(out_of_bounds_block);
        let not_written = ctx.iconst(ScalarType::Bool, 0);
        ctx.jump(merge_block, &[not_written]);

        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);
        Value::scalar(ctx.block_param(merge_block, 0, ScalarType::Bool))
    }
}

unsafe impl<S, I> Staged for SliceGetUnchecked<S, I>
where
    S: Staged,
    S::Out: SliceType,
    ElemOf<S>: CopyType,
    I: Staged<Out = u64>,
{
    type Out = ElemOf<S>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let index = self.index.codegen(ctx);
        let data_ptr = ctx.slice_data_ptr(&self.slice);
        let element_ptr = element_addr::<S>(ctx, data_ptr, index.leaf());
        ctx.load_value::<ElemOf<S>>(element_ptr)
    }
}

// =============================================================================
// SliceSetUnchecked: Set element (no bounds check, mutable slices only)
// =============================================================================

/// Set an element without bounds checking. Only valid on a mutable slice
/// (`S::Out: MutSliceType`).
#[derive(Clone, Copy)]
pub struct SliceSetUnchecked<S, I, V> {
    slice: S,
    index: I,
    value: V,
}

unsafe impl<S, I, V> Staged for SliceSetUnchecked<S, I, V>
where
    S: Staged,
    S::Out: MutSliceType,
    I: Staged<Out = u64>,
    V: Staged<Out = ElemOf<S>>,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let index = self.index.codegen(ctx);
        let value = self.value.codegen(ctx);
        let data_ptr = ctx.slice_data_ptr(&self.slice);
        let element_ptr = element_addr::<S>(ctx, data_ptr, index.leaf());
        ctx.store_value::<ElemOf<S>>(element_ptr, value);
        Value::scalar(ctx.get_unit_value())
    }
}

// =============================================================================
// SliceSwapUnchecked: Swap two elements (no bounds check, mutable slices only)
// =============================================================================

/// Swap the elements at indices `i` and `j` without bounds checking. Only valid
/// on a mutable `CopyType` slice. Emits two loads then two stores.
#[derive(Clone, Copy)]
pub struct SliceSwapUnchecked<S, I, J> {
    slice: S,
    i: I,
    j: J,
}

unsafe impl<S, I, J> Staged for SliceSwapUnchecked<S, I, J>
where
    S: Staged,
    S::Out: MutSliceType,
    ElemOf<S>: CopyType,
    I: Staged<Out = u64>,
    J: Staged<Out = u64>,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let i = self.i.codegen(ctx);
        let j = self.j.codegen(ctx);
        let data_ptr = ctx.slice_data_ptr(&self.slice);
        let addr_i = element_addr::<S>(ctx, data_ptr, i.leaf());
        let addr_j = element_addr::<S>(ctx, data_ptr, j.leaf());

        let vi = ctx.load_value::<ElemOf<S>>(addr_i);
        let vj = ctx.load_value::<ElemOf<S>>(addr_j);
        ctx.store_value::<ElemOf<S>>(addr_i, vj);
        ctx.store_value::<ElemOf<S>>(addr_j, vi);
        Value::scalar(ctx.get_unit_value())
    }
}

// =============================================================================
// SliceSliceUnchecked: Get sub-slice (no bounds check)
// =============================================================================

/// Get a sub-slice without bounds checking, for elements `[start..end]`.
///
/// Reports `Out = S::Out`, so sub-slicing a `&[T]` yields a `&[T]` and
/// sub-slicing a `&mut [T]` yields a `&mut [T]` — slices stay closed under
/// this operation. The result stays as a `(ptr, len)` SSA pair.
#[derive(Clone, Copy)]
pub struct SliceSliceUnchecked<S, START, END> {
    slice: S,
    start: START,
    end: END,
}

unsafe impl<S, START, END> Staged for SliceSliceUnchecked<S, START, END>
where
    S: Staged,
    S::Out: SliceType,
    START: Staged<Out = u64>,
    END: Staged<Out = u64>,
{
    type Out = S::Out;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let start = self.start.codegen(ctx);
        let end = self.end.codegen(ctx);
        let data_ptr = ctx.slice_data_ptr(&self.slice);

        // New base pointer: data_ptr + start * sizeof(Elem); new len: end - start.
        // The sub-slice is a fat value (two SSA leaves) — no stack slot, unlike the old
        // memory-resolved encoding.
        let new_ptr = element_addr::<S>(ctx, data_ptr, start.leaf());
        let new_len = ctx.isub(end.leaf(), start.leaf());
        Value::fat(new_ptr, new_len)
    }
}

// =============================================================================
// Operation umbrella: one op surface per capability
// =============================================================================
//
// The op traits mirror the capability traits above one-for-one, and each adds
// exactly what its capability licenses:
//
// ```text
//   SliceOps         : SliceType         len, as_ptr, subslice_unchecked, get_unchecked
//   TrustedSliceOps  : TrustedSliceType  get_or (SAFE), get_ref_unchecked
//   SliceMutOps      : MutSliceType      set (SAFE), set_unchecked, swap_unchecked
// ```
//
// The dividing line is *safety*, not merely which type you hold: the two safe,
// bounds-checked operations (`get_or`, `set`) are precisely the ones that need
// established provenance, so they sit above `TrustedSliceType`. Everything on
// `SliceOps` either touches no memory (`len`, `as_ptr`, `subslice_unchecked`) or
// is already `unsafe`, which is why a raw descriptor can have it without a
// separate trait — there is no `RawSliceOps`; raw types simply stop at
// `SliceOps`.
//
// Every method takes `self` by value: a staged expression *is* a value, and
// consuming it is what stops a unique `Var<SRefMut<_>>` being reused after it
// has been projected. Where reborrowing a non-`Copy` variable is wanted instead,
// `Var<SRefMut<Slice<T>>>` carries inherent `&self`/`&mut self` methods that a
// by-value blanket trait cannot express; inherent methods win method resolution,
// so `var.len()` reborrows while `expr.len()` consumes.

/// Operations available from a slice's **representation** alone.
///
/// `len`, `as_ptr` and `subslice_unchecked` dereference nothing — they read or
/// arithmetically adjust a `(ptr, len)` pair you already hold — so they are
/// available on every [`SliceType`], raw descriptors included.
// `len` is staged; host-side emptiness is unknowable, so there is no `is_empty`.
#[allow(clippy::len_without_is_empty)]
pub trait SliceOps: Staged + Sized
where
    Self::Out: SliceType,
{
    /// Number of elements.
    fn len(self) -> SliceLen<Self> {
        SliceLen { slice: self }
    }

    /// The data pointer. Mutability follows the slice's capability: `SPtr<T>`
    /// from a shared slice, `SMutPtr<T>` from a unique one — the associated
    /// [`SliceType::DataPtr`] carries it, so one method replaces the old
    /// `into_ptr`/`into_mut_ptr` pair.
    ///
    /// Named `into_` rather than `as_`: every op here consumes the expression,
    /// and `as_*` conventionally borrows.
    fn into_ptr(self) -> SliceAsPtr<Self> {
        SliceAsPtr { slice: self }
    }

    /// Read an element by value without bounds checking (`CopyType` elements).
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length. On a raw
    /// descriptor the caller must *also* establish that the pointer is live and
    /// aligned for `T` — that is the provenance a [`TrustedSliceType`] would
    /// have proven.
    unsafe fn get_unchecked<I>(self, index: I) -> SliceGetUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
        ElemOf<Self>: CopyType,
    {
        SliceGetUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Derive a sub-slice over `[start, end)` **with bounds checking**,
    /// yielding a [`StagedOpt`]: `Some` when `start <= end && end <= len`,
    /// otherwise `None`.
    ///
    /// Safe, and available on every origin — checking bounds and adjusting a
    /// `(ptr, len)` pair dereferences nothing, so a raw descriptor gets a
    /// checked *raw* sub-slice just as a trusted slice gets a trusted one.
    ///
    /// Takes two arguments rather than a `start..end` range: `Range<Idx>` has a
    /// single index type, so the common mixed form — a literal start with a
    /// staged end, `0u64 .. len` — is a type error at the `..` itself. That is
    /// 5 of the 25 sub-slice call sites in this workspace, and they are the
    /// dynamic ones.
    fn get_range<START, END>(
        self,
        start: START,
        end: END,
    ) -> SliceGetRange<Self, START::Staged, END::Staged>
    where
        START: IntoStaged<u64>,
        END: IntoStaged<u64>,
    {
        SliceGetRange {
            slice: self,
            start: start.into_staged(),
            end: end.into_staged(),
        }
    }

    /// Derive a sub-slice over `[start, end)`.
    ///
    /// Closed under capability *and* provenance: the result reports
    /// `Out = Self::Out`, so a sub-slice of a `&mut [T]` stays writable and a
    /// sub-slice of a raw descriptor stays raw.
    ///
    /// Consuming the receiver is what prevents a unique parent from being used
    /// alongside its own sub-slice:
    ///
    /// ```compile_fail
    /// use rust_lms::prelude::*;
    ///
    /// fn overlapping(mut slice: Var<SRefMut<Slice<i64>>>) {
    ///     let sub = unsafe { slice.subslice_unchecked(0u64, 1u64) };
    ///     let _parent = slice.reborrow().len(); // `slice` was consumed above
    ///     let _ = sub;
    /// }
    /// ```
    ///
    /// # Safety
    ///
    /// At execution, `start <= end` and `end <= self.len()` must hold.
    unsafe fn subslice_unchecked<START, END>(
        self,
        start: START,
        end: END,
    ) -> SliceSliceUnchecked<Self, START::Staged, END::Staged>
    where
        START: IntoStaged<u64>,
        END: IntoStaged<u64>,
    {
        SliceSliceUnchecked {
            slice: self,
            start: start.into_staged(),
            end: end.into_staged(),
        }
    }
}

impl<S> SliceOps for S
where
    S: Staged + Sized,
    S::Out: SliceType,
{
}

/// A **checked** sub-slice: `Some(slice[start..end])` when
/// `start <= end && end <= len`, otherwise `None`.
///
/// A [`StagedOpt`], not a [`Staged`] value — it never materializes a
/// discriminant. The bounds test lowers to one branchless `select`, and the
/// sub-slice is only built on the taken arm.
///
/// The slice expression is used *twice* (once for `len`, once for the
/// sub-slice), which a unique origin cannot supply by cloning. It is therefore
/// bound to a variable once and reborrowed — the same bind-once-and-reborrow
/// idiom a generic slice helper needs, which is what lets this serve every
/// capability rather than only shared ones.
pub struct SliceGetRange<S, START, END> {
    slice: S,
    start: START,
    end: END,
}

impl<S, START, END> StagedOpt for SliceGetRange<S, START, END>
where
    S: Staged + 'static,
    S::Out: SliceType + 'static,
    START: Staged<Out = u64> + 'static,
    END: Staged<Out = u64> + 'static,
{
    type Item = S::Out;

    fn eliminate<F, N>(self, ctx: &mut Ctx, on_some: F, on_none: N)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
        N: FnOnce(&mut Ctx) + 'static,
    {
        let mut slice = ctx.bind(self.slice);
        let start = ctx.bind(self.start);
        let end = ctx.bind(self.end);
        let len = ctx.bind(slice.reborrow().len());

        // `start <= end && end <= len`, branchless: there is no `le`/`and`, so
        // read it as "if start > end then false, else end <= len".
        let in_range = ctx.bind(crate::num::select(
            crate::num::gt(start, end),
            false,
            crate::control::not(crate::num::gt(end, len)),
        ));

        // SAFETY: only codegen'd on the arm where `in_range` proved
        // `start <= end <= len`.
        let sub = unsafe { slice.subslice_unchecked(start, end) };
        ctx.if_then_else(
            in_range,
            move |ctx| {
                let bound = ctx.bind(sub);
                on_some(ctx, bound);
            },
            on_none,
        );
    }
}

/// Operations licensed by **established provenance** — the two safe,
/// bounds-checked accessors, plus element references.
///
/// A raw descriptor deliberately cannot reach these: a safe `get_or` on an
/// unvalidated pointer would be a safe dereference of something nothing has
/// vouched for. Raw values join this trait only by crossing an explicit unsafe
/// promotion into a trusted slice.
pub trait TrustedSliceOps: SliceOps
where
    Self::Out: TrustedSliceType,
{
    /// Read `index` when it is in bounds, otherwise evaluate `default`.
    fn get_or<I, D>(self, index: I, default: D) -> SliceGetOr<Self, I::Staged, D::Staged>
    where
        I: IntoStaged<u64>,
        D: IntoStaged<ElemOf<Self>>,
        ElemOf<Self>: DirectValue,
    {
        SliceGetOr {
            slice: self,
            index: index.into_staged(),
            default: default.into_staged(),
        }
    }

    /// Project a reference to an element. Mutability follows the slice:
    /// `SRef<T>` from `&[T]`, `SRefMut<T>` from `&mut [T]`, carried by
    /// [`TrustedSliceType::ElemRef`].
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length, and the
    /// result must obey the source slice's aliasing contract.
    unsafe fn get_ref_unchecked<I>(self, index: I) -> SliceGetRefUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
    {
        SliceGetRefUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }
}

impl<S> TrustedSliceOps for S
where
    S: Staged + Sized,
    S::Out: TrustedSliceType,
{
}

/// Writing operations, licensed by trusted **and exclusive** access.
///
/// A shared slice cannot reach them, so a generic helper bounded on
/// `MutSliceType` accepts only unique origins:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn write_to_shared(slice: Var<SRef<Slice<i64>>>) {
///     let _ = slice.set(0u64, 1i64);
/// }
/// ```
pub trait SliceMutOps: TrustedSliceOps
where
    Self::Out: MutSliceType,
{
    /// Write `index` when it is in bounds, returning whether the write ran.
    fn set<I, V>(self, index: I, value: V) -> SliceSet<Self, I::Staged, V::Staged>
    where
        I: IntoStaged<u64>,
        V: IntoStaged<ElemOf<Self>>,
        ElemOf<Self>: DirectValue,
    {
        SliceSet {
            slice: self,
            index: index.into_staged(),
            value: value.into_staged(),
        }
    }

    /// Write an element without bounds checking.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length and the
    /// slice must remain exclusively accessible for the write.
    unsafe fn set_unchecked<I, V>(
        self,
        index: I,
        value: V,
    ) -> SliceSetUnchecked<Self, I::Staged, V::Staged>
    where
        I: IntoStaged<u64>,
        V: IntoStaged<ElemOf<Self>>,
    {
        SliceSetUnchecked {
            slice: self,
            index: index.into_staged(),
            value: value.into_staged(),
        }
    }

    /// Swap two elements (`CopyType` elements only).
    ///
    /// # Safety
    ///
    /// At execution, both indices must be less than this slice's length.
    unsafe fn swap_unchecked<I, J>(
        self,
        i: I,
        j: J,
    ) -> SliceSwapUnchecked<Self, I::Staged, J::Staged>
    where
        I: IntoStaged<u64>,
        J: IntoStaged<u64>,
        ElemOf<Self>: CopyType,
    {
        SliceSwapUnchecked {
            slice: self,
            i: i.into_staged(),
            j: j.into_staged(),
        }
    }
}

impl<S> SliceMutOps for S
where
    S: Staged + Sized,
    S::Out: MutSliceType,
{
}

// =============================================================================
// The provenance boundary: raw -> trusted
// =============================================================================

/// Reinterprets a raw `(ptr, len)` descriptor as a *trusted* slice.
///
/// Emits nothing: the value is already a `Value::Fat` pair, and promotion is a
/// statement about provenance, not about representation. All of the work is in
/// the safety contract on the constructing method.
pub struct AssumeTrusted<S, OUT> {
    raw: S,
    _out: PhantomData<OUT>,
}

impl<S: Clone, OUT> Clone for AssumeTrusted<S, OUT> {
    fn clone(&self) -> Self {
        Self {
            raw: self.raw.clone(),
            _out: PhantomData,
        }
    }
}

impl<S: Copy, OUT> Copy for AssumeTrusted<S, OUT> {}

unsafe impl<S, OUT> Staged for AssumeTrusted<S, OUT>
where
    S: Staged,
    S::Out: RawSliceType,
    OUT: TrustedSliceType<Elem = ElemOf<S>>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.raw.codegen(ctx)
    }
}

/// The one way a raw descriptor becomes a trusted slice.
///
/// An `extern "C"` function can return a dangling `FatSlice<T>` without writing
/// a line of `unsafe` — the struct is two public fields and carries no lifetime.
/// So an FFI result can never *become* `SRef<Slice<T>>` implicitly. The pipeline
/// is deliberately three steps:
///
/// ```text
/// extern result -> raw staged slice -> explicit unsafe promotion -> ordinary SliceOps
/// ```
///
/// Before promotion a raw value stops at [`SliceOps`], where every accessor is
/// `unsafe`; after it, the result is an ordinary trusted slice with the full
/// safe surface. Nothing safe dereferences in between:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn safe_read_before_promotion(d: Var<RawSlice<i64>>) {
///     let _ = d.get_or(0u64, 0i64);
/// }
/// ```
///
/// And a *shared* raw descriptor cannot launder itself into a writable view —
/// `assume_unique` requires `DataPtr = SMutPtr<T>`:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn launder(d: Var<RawSlice<i64>>) {
///     let _ = unsafe { d.assume_unique() };
/// }
/// ```
pub trait RawSliceOps: SliceOps
where
    Self::Out: RawSliceType,
{
    /// Promote to a trusted **shared** slice (`&[T]`).
    ///
    /// # Safety
    ///
    /// The caller states, for the whole of generated execution:
    /// * which owner keeps the allocation alive, and that it outlives every use;
    /// * that the pointer is aligned and points at `len` initialized `T`;
    /// * that the storage is not mutated while this shared view is live;
    /// * that the producer cannot reallocate the storage.
    unsafe fn assume_shared(self) -> AssumeTrusted<Self, SRef<Slice<ElemOf<Self>>>> {
        AssumeTrusted {
            raw: self,
            _out: PhantomData,
        }
    }

    /// Promote to a trusted **unique** slice (`&mut [T]`).
    ///
    /// Available only from a *mutable* raw descriptor: the method's bound
    /// requires `DataPtr = SMutPtr<T>`, so a shared [`RawSlice`] cannot launder
    /// itself into a writable view.
    ///
    /// # Safety
    ///
    /// As [`assume_shared`](Self::assume_shared), and additionally that access
    /// through this view is *exclusive* — no other staged or host reference to
    /// the storage is used while it is live.
    unsafe fn assume_unique(self) -> AssumeTrusted<Self, SRefMut<Slice<ElemOf<Self>>>>
    where
        Self::Out: SliceType<DataPtr = SMutPtr<ElemOf<Self>>>,
    {
        AssumeTrusted {
            raw: self,
            _out: PhantomData,
        }
    }
}

impl<S> RawSliceOps for S
where
    S: Staged + Sized,
    S::Out: RawSliceType,
{
}

type MutFieldSlice<T, F, E> = AsMutSlice<FieldAddr<VarUse<SRefMut<T>>, F>, E>;

impl<'borrow, T, F> MutField<'borrow, T, F>
where
    T: StagedType,
    F: Field<Parent = T>,
{
    /// Read this `(ptr, len)` descriptor field as a **raw mutable** slice,
    /// reborrowing the parent for one use.
    ///
    /// Safe, and raw: the result carries the representation-only op surface
    /// ([`SliceOps`]) and no claim about the buffer. Promote it with
    /// [`RawSliceOps::assume_unique`] to reach the writing ops — that is where
    /// the validity contract is stated, once, instead of on every accessor.
    pub fn as_mut_slice<E>(&mut self) -> MutFieldSlice<T, F, E>
    where
        E: StagedType,
        F::Out: MutSliceRepr<E>,
    {
        AsMutSlice {
            repr: self.use_once(),
            _elem: PhantomData,
        }
    }
}
