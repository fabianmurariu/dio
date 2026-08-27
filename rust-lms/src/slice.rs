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

use crate::ffi::{FatSliceMutType, FatSliceType};
use crate::r#struct::{Field, FieldAddr, MutField};
use crate::refer::{SMutPtr, SPtr, SRef, SRefMut};
use crate::staged::{CompilationContext, IntoStaged, Staged, Value, ValueId, Var, VarUse};
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
/// lifetime-free staged [`FatSliceType<T>`].
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
    type Out = FatSliceType<T>;

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
    type Out = SRef<Slice<T>>;

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
    /// Reinterpret the pointed-to representation as a staged slice of `T`.
    ///
    /// # Safety
    ///
    /// The descriptor must contain a pointer that is live and aligned for
    /// reads of `len` values of `T` for the duration of generated execution.
    unsafe fn into_slice<T>(self) -> AsSlice<Self, T>
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
    /// Reinterpret the pointed-to representation as a staged *mutable* slice.
    ///
    /// # Safety
    ///
    /// The descriptor must contain a pointer that is live, aligned, and
    /// exclusively writable for `len` values of `T` for the duration of
    /// generated execution.
    unsafe fn into_mut_slice<T>(self) -> AsMutSlice<Self, T>
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
    type Out = SRefMut<Slice<T>>;

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
///     trusted_only::<FatSliceType<i64>>();
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
///     writable_only::<FatSliceMutType<i64>>();
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

impl<T: StagedType> SliceType for FatSliceType<T> {
    type Elem = T;
    type DataPtr = SPtr<T>;
}
impl<T: StagedType> RawSliceType for FatSliceType<T> {}
impl<T: StagedType> slice_type_sealed::Sealed for FatSliceType<T> {}
impl<T: StagedType> slice_type_sealed::RawSealed for FatSliceType<T> {}

// --- raw mutable descriptor ---------------------------------------------------
//
// Half of gap G6b from the row-0 characterization matrix: before the taxonomy
// this marker implemented nothing, so a mutable raw descriptor supported no
// slice operation at all — not even `len`. Classifying it here makes every op
// *node* accept it; the other half is row 4, where `RawSliceOps` stops being
// hard-bound to `FatSliceType<T>` so a call site can actually reach them.
//
// It is `RawSliceType`, not `MutSliceType`: the pointer permits writes, but
// provenance is unproven, so writing goes through an explicit promotion to a
// trusted slice rather than through the slice write API.

impl<T: StagedType> SliceType for FatSliceMutType<T> {
    type Elem = T;
    type DataPtr = SMutPtr<T>;
}
impl<T: StagedType> RawSliceType for FatSliceMutType<T> {}
impl<T: StagedType> slice_type_sealed::Sealed for FatSliceMutType<T> {}
impl<T: StagedType> slice_type_sealed::RawSealed for FatSliceMutType<T> {}

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
// Extension trait for Var<SRef<Slice<T>>> - Immutable slice operations
// =============================================================================

/// Extension trait for immutable slice operations.
pub trait SliceRefOps<T: StagedType>: Staged<Out = SRef<Slice<T>>> + Sized + Clone {
    /// Get the length of the slice.
    fn count(self) -> SliceLen<Self> {
        SliceLen { slice: self }
    }

    /// Get the raw data pointer.
    fn into_ptr(self) -> SliceAsPtr<Self> {
        SliceAsPtr { slice: self }
    }

    /// Read `index` when it is in bounds, otherwise return `default`.
    fn get_or<I, D>(self, index: I, default: D) -> SliceGetOr<Self, I::Staged, D::Staged>
    where
        I: IntoStaged<u64>,
        D: IntoStaged<T>,
        T: DirectValue,
    {
        SliceGetOr {
            slice: self,
            index: index.into_staged(),
            default: default.into_staged(),
        }
    }

    /// Get a reference to an element without bounds checking.
    ///
    /// Accepts any value that can be converted into a u64 staged expression for the index.
    /// This allows ergonomic usage like `arr.get_ref_unchecked(5u64)` instead of
    /// `arr.get_ref_unchecked(Const::<u64>::new(5))`.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length and the
    /// resulting reference must obey the source slice's aliasing contract.
    unsafe fn get_ref_unchecked<I>(self, index: I) -> SliceGetRefUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
    {
        SliceGetRefUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Get an element by value without bounds checking.
    ///
    /// Only available for `CopyType` elements.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length.
    unsafe fn get_unchecked<I>(self, index: I) -> SliceGetUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
        T: CopyType,
    {
        SliceGetUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Get a sub-slice without bounds checking.
    ///
    /// # Safety
    ///
    /// At execution, `start <= end` and `end <= self.len()` must hold.
    unsafe fn slice_unchecked<START, END>(
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

impl<T: StagedType, S> SliceRefOps<T> for S where S: Staged<Out = SRef<Slice<T>>> + Clone {}

// =============================================================================
// Extension trait for lifetime-free FatSliceType<T> operations
// =============================================================================

/// Read operations on a lifetime-free raw `(ptr, len)` slice descriptor.
#[allow(clippy::len_without_is_empty)] // `len` is staged; host-side emptiness is unknowable.
pub trait RawSliceOps<T: StagedType>: Staged<Out = FatSliceType<T>> + Sized + Clone {
    fn len(self) -> SliceLen<Self> {
        SliceLen { slice: self }
    }

    /// Read an element without bounds checking.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this descriptor's element count,
    /// and its pointer must remain live and aligned for `T`.
    unsafe fn get_unchecked<I>(self, index: I) -> SliceGetUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
        T: CopyType,
    {
        SliceGetUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Create a raw sub-slice without bounds checking.
    ///
    /// # Safety
    ///
    /// At execution, `start <= end` and `end <= self.len()` must hold, and the
    /// source pointer must remain live for every use of the result.
    unsafe fn slice_unchecked<START, END>(
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

impl<T: StagedType, S> RawSliceOps<T> for S where S: Staged<Out = FatSliceType<T>> + Sized + Clone {}

type MutFieldSlice<T, F, E> = AsMutSlice<FieldAddr<VarUse<SRefMut<T>>, F>, E>;

impl<'borrow, T, F> MutField<'borrow, T, F>
where
    T: StagedType,
    F: Field<Parent = T>,
{
    fn as_mut_slice_once<E>(&mut self) -> MutFieldSlice<T, F, E>
    where
        E: StagedType,
        F::Out: MutSliceRepr<E>,
    {
        AsMutSlice {
            repr: self.use_once(),
            _elem: PhantomData,
        }
    }

    /// Read the length stored in a mutable slice-descriptor field.
    ///
    /// # Safety
    ///
    /// The descriptor must contain a live, aligned, exclusively owned buffer
    /// for its recorded element count.
    pub unsafe fn slice_len<E>(&mut self) -> SliceLen<MutFieldSlice<T, F, E>>
    where
        E: StagedType,
        F::Out: MutSliceRepr<E>,
    {
        SliceLen {
            slice: self.as_mut_slice_once(),
        }
    }

    /// Read one element from a mutable slice-descriptor field.
    ///
    /// # Safety
    ///
    /// The descriptor must contain a live, aligned, exclusively owned buffer,
    /// and `index` must be less than its recorded element count at execution.
    pub unsafe fn slice_get_unchecked<E, I>(
        &mut self,
        index: I,
    ) -> SliceGetUnchecked<MutFieldSlice<T, F, E>, I::Staged>
    where
        E: CopyType,
        I: IntoStaged<u64>,
        F::Out: MutSliceRepr<E>,
    {
        SliceGetUnchecked {
            slice: self.as_mut_slice_once(),
            index: index.into_staged(),
        }
    }

    /// Write one element through a mutable slice-descriptor field.
    ///
    /// # Safety
    ///
    /// The descriptor must contain a live, aligned, exclusively owned buffer,
    /// and `index` must be less than its recorded element count at execution.
    pub unsafe fn slice_set_unchecked<E, I, V>(
        &mut self,
        index: I,
        value: V,
    ) -> SliceSetUnchecked<MutFieldSlice<T, F, E>, I::Staged, V::Staged>
    where
        E: StagedType,
        I: IntoStaged<u64>,
        V: IntoStaged<E>,
        F::Out: MutSliceRepr<E>,
    {
        SliceSetUnchecked {
            slice: self.as_mut_slice_once(),
            index: index.into_staged(),
            value: value.into_staged(),
        }
    }
}

// =============================================================================
// Extension trait for Var<SRefMut<Slice<T>>> - Mutable slice operations
// =============================================================================

impl<T: StagedType> Var<SRefMut<Slice<T>>> {
    /// Reborrow this unique slice handle to read its length.
    pub fn len(&self) -> SliceLen<VarUse<SRefMut<Slice<T>>>> {
        SliceLen {
            slice: self.use_once(),
        }
    }

    /// Read `index` when it is in bounds, otherwise return `default`.
    pub fn get_or<I, D>(
        &self,
        index: I,
        default: D,
    ) -> SliceGetOr<VarUse<SRefMut<Slice<T>>>, I::Staged, D::Staged>
    where
        I: IntoStaged<u64>,
        D: IntoStaged<T>,
        T: DirectValue,
    {
        SliceGetOr {
            slice: self.use_once(),
            index: index.into_staged(),
            default: default.into_staged(),
        }
    }

    /// Write `index` when it is in bounds, returning whether the write ran.
    pub fn set<I, V>(
        &mut self,
        index: I,
        value: V,
    ) -> SliceSet<VarUse<SRefMut<Slice<T>>>, I::Staged, V::Staged>
    where
        I: IntoStaged<u64>,
        V: IntoStaged<T>,
        T: DirectValue,
    {
        SliceSet {
            slice: self.use_once(),
            index: index.into_staged(),
            value: value.into_staged(),
        }
    }

    /// Consume this unique slice and project one mutable element.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length.
    pub unsafe fn get_mut_unchecked<I>(self, index: I) -> SliceGetRefUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
    {
        SliceGetRefUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Reborrow this unique slice handle to read one element.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length.
    pub unsafe fn get_unchecked<I>(
        &self,
        index: I,
    ) -> SliceGetUnchecked<VarUse<SRefMut<Slice<T>>>, I::Staged>
    where
        I: IntoStaged<u64>,
        T: CopyType,
    {
        SliceGetUnchecked {
            slice: self.use_once(),
            index: index.into_staged(),
        }
    }

    /// Reborrow this unique slice handle for one element write.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length.
    pub unsafe fn set_unchecked<I, V>(
        &mut self,
        index: I,
        value: V,
    ) -> SliceSetUnchecked<VarUse<SRefMut<Slice<T>>>, I::Staged, V::Staged>
    where
        I: IntoStaged<u64>,
        V: IntoStaged<T>,
    {
        SliceSetUnchecked {
            slice: self.use_once(),
            index: index.into_staged(),
            value: value.into_staged(),
        }
    }

    /// Reborrow this unique slice handle to swap two elements.
    ///
    /// # Safety
    ///
    /// At execution, both indices must be less than this slice's length.
    pub unsafe fn swap_unchecked<I, J>(
        &mut self,
        i: I,
        j: J,
    ) -> SliceSwapUnchecked<VarUse<SRefMut<Slice<T>>>, I::Staged, J::Staged>
    where
        I: IntoStaged<u64>,
        J: IntoStaged<u64>,
        T: CopyType,
    {
        SliceSwapUnchecked {
            slice: self.use_once(),
            i: i.into_staged(),
            j: j.into_staged(),
        }
    }

    /// Consume this unique slice handle to construct a mutable sub-slice.
    ///
    /// # Safety
    ///
    /// At execution, `start <= end <= self.len()` must hold. The consuming
    /// receiver prevents the parent handle from being reused.
    ///
    /// ```compile_fail
    /// use rust_lms::prelude::*;
    ///
    /// fn overlapping(slice: Var<SRefMut<Slice<i64>>>) {
    ///     let sub = unsafe { slice.slice_mut_unchecked(0u64, 1u64) };
    ///     let _parent_len = slice.len();
    ///     let _ = sub;
    /// }
    /// ```
    pub unsafe fn slice_mut_unchecked<START, END>(
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

/// Extension trait for mutable slice operations.
///
/// The read-only ops (`len`, `as_mut_ptr`, `get_*`, `slice_mut_unchecked`)
/// build the same unified op structs as [`SliceRefOps`]; the associated
/// `ElemRef`/`Out` keep their mutable flavor automatically. `set_unchecked` is
/// gated on `MutSliceType`, so it only exists here.
pub trait SliceMutOps<T: StagedType>: Staged<Out = SRefMut<Slice<T>>> + Sized {
    /// Get the length of the slice.
    fn count(self) -> SliceLen<Self> {
        SliceLen { slice: self }
    }

    /// Get the raw mutable data pointer.
    fn into_mut_ptr(self) -> SliceAsPtr<Self> {
        SliceAsPtr { slice: self }
    }

    /// Read `index` when it is in bounds, otherwise return `default`.
    fn get_or<I, D>(self, index: I, default: D) -> SliceGetOr<Self, I::Staged, D::Staged>
    where
        I: IntoStaged<u64>,
        D: IntoStaged<T>,
        T: DirectValue,
    {
        SliceGetOr {
            slice: self,
            index: index.into_staged(),
            default: default.into_staged(),
        }
    }

    /// Write `index` when it is in bounds, returning whether the write ran.
    fn set<I, V>(self, index: I, value: V) -> SliceSet<Self, I::Staged, V::Staged>
    where
        I: IntoStaged<u64>,
        V: IntoStaged<T>,
        T: DirectValue,
    {
        SliceSet {
            slice: self,
            index: index.into_staged(),
            value: value.into_staged(),
        }
    }

    /// Get a mutable reference to an element without bounds checking.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length, and no
    /// overlapping staged reference may be used while the result is live.
    unsafe fn get_mut_unchecked<I>(self, index: I) -> SliceGetRefUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
    {
        SliceGetRefUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Get an element by value without bounds checking.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length.
    unsafe fn get_unchecked<I>(self, index: I) -> SliceGetUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
        T: CopyType,
    {
        SliceGetUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Set an element without bounds checking.
    ///
    /// Accepts any value that can be converted into staged expressions.
    /// This allows ergonomic usage like `arr.set_unchecked(0u64, 42i64)`.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this slice's length and the
    /// mutable slice must remain exclusively accessible for the write.
    unsafe fn set_unchecked<I, V>(
        self,
        index: I,
        value: V,
    ) -> SliceSetUnchecked<Self, I::Staged, V::Staged>
    where
        I: IntoStaged<u64>,
        V: IntoStaged<T>,
    {
        SliceSetUnchecked {
            slice: self,
            index: index.into_staged(),
            value: value.into_staged(),
        }
    }

    /// Get a mutable sub-slice without bounds checking.
    ///
    /// # Safety
    ///
    /// At execution, `start <= end` and `end <= self.len()` must hold. No
    /// overlapping staged reference may be used while the result is live.
    unsafe fn slice_mut_unchecked<START, END>(
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

    /// Swap the elements at indices `i` and `j` without bounds checking.
    ///
    /// Only available for `CopyType` elements. Ergonomic like the other ops:
    /// `arr.swap_unchecked(0u64, lo + 1u64)`.
    ///
    /// # Safety
    ///
    /// At execution, both `i` and `j` must be less than this slice's length.
    unsafe fn swap_unchecked<I, J>(
        self,
        i: I,
        j: J,
    ) -> SliceSwapUnchecked<Self, I::Staged, J::Staged>
    where
        I: IntoStaged<u64>,
        J: IntoStaged<u64>,
        T: CopyType,
    {
        SliceSwapUnchecked {
            slice: self,
            i: i.into_staged(),
            j: j.into_staged(),
        }
    }
}

impl<P, R, T> SliceMutOps<T> for AsMutSlice<P, T>
where
    P: Staged<Out = SRefMut<R>>,
    R: MutSliceRepr<T>,
    T: StagedType,
{
}

impl<S, START, END, T> SliceMutOps<T> for SliceSliceUnchecked<S, START, END>
where
    S: Staged<Out = SRefMut<Slice<T>>>,
    START: Staged<Out = u64>,
    END: Staged<Out = u64>,
    T: StagedType,
{
}
