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
//! Within the staged graph a slice's `codegen` value is always a single `i64`,
//! resolved one of two ways (see [`CompilationContext::slice_data_ptr`] /
//! [`CompilationContext::slice_len`], which are the only code that knows this):
//!
//! - **register-resolved** — slice *parameters* load the descriptor into two
//!   Cranelift variables (`ptr_var`, `len_var`) kept in
//!   `ctx.slice_vars` keyed by `var_id`. Slice ops read those registers
//!   directly, with no memory access (the fast path for tight loops).
//! - **memory-resolved** — subslices (and any operand without a `var_id`) have
//!   a `codegen` value that is a *pointer to* a `(ptr, len)` pair on a stack
//!   slot: `ptr` at offset 0, `len` at offset 8.
//!
//! So `Slice<T>` really is just "ptr + len"; the indirection only exists
//! because `Staged::codegen` returns a single `Value`, so an anonymous slice
//! needs somewhere (the stack slot) to hold its two halves.
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

use crate::ffi::FatSliceType;
use crate::r#struct::{Field, FieldAddr, MutField};
use crate::refer::{SMutPtr, SPtr, SRef, SRefMut};
use crate::staged::{CompilationContext, IntoStaged, Staged, Var, VarUse};
use crate::types::{ScalarType, CopyType, DirectValue, RuntimeParam, RuntimeResult, StagedType};
use cranelift_codegen::ir::{
    condcodes::IntCC, types, StackSlotData, StackSlotKind, Value,
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
        self.repr.codegen(ctx)
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

unsafe impl<'a, P, R, T> Staged for AsSlice<P, T>
where
    P: Staged<Out = SRef<'a, R>>,
    R: SliceRepr<T> + 'a,
    T: StagedType + 'a,
{
    type Out = SRef<'a, Slice<T>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.repr.codegen(ctx)
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
/// fn arbitrary_value_is_not_a_slice(value: Var<SRef<'_, i64>>) {
///     let _ = unsafe { value.as_slice::<u8>() };
/// }
/// ```
pub trait ReprSliceOps<'a, R>: Staged<Out = SRef<'a, R>> + Sized
where
    R: StagedType + 'a,
{
    /// Reinterpret the pointed-to representation as a staged slice of `T`.
    ///
    /// # Safety
    ///
    /// The descriptor must contain a pointer that is live and aligned for
    /// reads of `len` values of `T` for the duration of generated execution.
    unsafe fn as_slice<T>(self) -> AsSlice<Self, T>
    where
        T: StagedType + 'a,
        R: SliceRepr<T>,
    {
        AsSlice {
            repr: self,
            _elem: PhantomData,
        }
    }
}

impl<'a, R, S> ReprSliceOps<'a, R> for S
where
    R: StagedType + 'a,
    S: Staged<Out = SRef<'a, R>> + Sized,
{
}

/// Extension trait for *mutable* references to a repr-compatible `(ptr, len)`.
///
/// The mutable twin of [`ReprSliceOps`]: given a `&mut` to an FFI descriptor
/// whose first two fields are `ptr`/`len`, reinterpret it as a `&mut [T]`.
pub trait ReprSliceMutOps<'a, R>: Staged<Out = SRefMut<'a, R>> + Sized
where
    R: StagedType + 'a,
{
    /// Reinterpret the pointed-to representation as a staged *mutable* slice.
    ///
    /// # Safety
    ///
    /// The descriptor must contain a pointer that is live, aligned, and
    /// exclusively writable for `len` values of `T` for the duration of
    /// generated execution.
    unsafe fn as_mut_slice<T>(self) -> AsMutSlice<Self, T>
    where
        T: StagedType + 'a,
        R: MutSliceRepr<T>,
    {
        AsMutSlice {
            repr: self,
            _elem: PhantomData,
        }
    }
}

impl<'a, R, S> ReprSliceMutOps<'a, R> for S
where
    R: StagedType + 'a,
    S: Staged<Out = SRefMut<'a, R>> + Sized,
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

unsafe impl<'a, P, R, T> Staged for AsMutSlice<P, T>
where
    P: Staged<Out = SRefMut<'a, R>>,
    R: MutSliceRepr<T> + 'a,
    T: StagedType + 'a,
{
    type Out = SRefMut<'a, Slice<T>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.repr.codegen(ctx)
    }
}

// =============================================================================
// StagedType for SRef<Slice<T>> - Immutable Fat Pointer
// =============================================================================

unsafe impl<'a, T: StagedType> StagedType for SRef<'a, Slice<T>> {
    /// Runtime type is `&[T::RuntimeValue]`
    type RuntimeValue = &'a [T::RuntimeValue];

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        // Internally represented as pointer to (ptr, len) pair
        types::I64
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

unsafe impl<'a, T: StagedType> CopyType for SRef<'a, Slice<T>> {}

unsafe impl<'stage, T> RuntimeParam for SRef<'stage, Slice<T>>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = &'call [T::RuntimeValue];
}

unsafe impl<'stage, T> RuntimeResult for SRef<'stage, Slice<T>>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Output<'call> = &'call [T::RuntimeValue];
}

// =============================================================================
// StagedType for SRefMut<Slice<T>> - Mutable Fat Pointer
// =============================================================================

unsafe impl<'a, T: StagedType> StagedType for SRefMut<'a, Slice<T>> {
    /// Runtime type is `&mut [T::RuntimeValue]`
    type RuntimeValue = &'a mut [T::RuntimeValue];

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
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

unsafe impl<'stage, T> RuntimeParam for SRefMut<'stage, Slice<T>>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = &'call mut [T::RuntimeValue];
}

unsafe impl<'stage, T> RuntimeResult for SRefMut<'stage, Slice<T>>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Output<'call> = &'call mut [T::RuntimeValue];
}

// =============================================================================
// SliceType: unify immutable and mutable slice fat pointers
// =============================================================================

mod slice_type_sealed {
    pub trait Sealed {}
    pub trait MutableSealed: Sealed {}
}

/// A staged slice fat-pointer type. Implemented by both `SRef<Slice<T>>`
/// (`&[T]`) and `SRefMut<Slice<T>>` (`&mut [T]`), so a *single* op impl can
/// serve both — the mutability lives entirely in the associated `ElemRef`.
///
/// This is what keeps slices *closed under sub-slicing*: every op below is
/// generic over `S::Out: SliceType`, and `SliceSliceUnchecked` reports
/// `Out = S::Out`, so the result of `slice_unchecked` is itself a slice that
/// supports `len`/`get`/`slice_unchecked`/… all over again — and a sub-slice
/// of a `&mut [T]` stays mutable.
///
/// The trait is sealed; only the slice representations supplied by this crate
/// may participate in slice lowering.
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
///     type ElemRef = SPtr<u8>;
///     type DataPtr = SPtr<u8>;
/// }
/// ```
pub trait SliceType: StagedType + slice_type_sealed::Sealed {
    /// Element type (`T`).
    type Elem: StagedType;
    /// Reference-to-element produced by `get_ref_unchecked`:
    /// `SRef<T>` for an immutable slice, `SRefMut<T>` for a mutable one.
    type ElemRef: StagedType;
    /// Raw pointer produced by `as_ptr` / `as_mut_ptr`.
    type DataPtr: StagedType;
}

impl<'a, T: StagedType> SliceType for SRef<'a, Slice<T>> {
    type Elem = T;
    type ElemRef = SRef<'a, T>;
    type DataPtr = SPtr<T>;
}

impl<'a, T: StagedType> slice_type_sealed::Sealed for SRef<'a, Slice<T>> {}

impl<'a, T: StagedType> SliceType for SRefMut<'a, Slice<T>> {
    type Elem = T;
    type ElemRef = SRefMut<'a, T>;
    type DataPtr = SMutPtr<T>;
}

impl<'a, T: StagedType> slice_type_sealed::Sealed for SRefMut<'a, Slice<T>> {}

impl<T: StagedType> SliceType for FatSliceType<T> {
    type Elem = T;
    type ElemRef = SPtr<T>;
    type DataPtr = SPtr<T>;
}

impl<T: StagedType> slice_type_sealed::Sealed for FatSliceType<T> {}

/// Marker for *mutable* slices (`SRefMut<Slice<T>>`). Gates the writing ops
/// (`set_unchecked`) so they cannot be called on an immutable slice.
pub trait MutSliceType: SliceType + slice_type_sealed::MutableSealed {}

impl<'a, T: StagedType> MutSliceType for SRefMut<'a, Slice<T>> {}
impl<'a, T: StagedType> slice_type_sealed::MutableSealed for SRefMut<'a, Slice<T>> {}

/// Convenience accessor for `S`'s element type inside generic op impls.
type ElemOf<S> = <<S as Staged>::Out as SliceType>::Elem;

/// Emit `data_ptr + index * sizeof(Elem)`, the address of element `index`.
fn element_addr<S>(ctx: &mut CompilationContext, data_ptr: Value, index: Value) -> Value
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
        ctx.slice_len(&self.slice)
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
        ctx.slice_data_ptr(&self.slice)
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
    S::Out: SliceType,
    I: Staged<Out = u64>,
{
    type Out = <S::Out as SliceType>::ElemRef;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let index = self.index.codegen(ctx);
        let data_ptr = ctx.slice_data_ptr(&self.slice);
        element_addr::<S>(ctx, data_ptr, index)
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
        element_addr::<S>(ctx, data_ptr, index)
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
        let in_bounds = ctx.icmp(IntCC::UnsignedLessThan, index, len);

        let get_block = ctx.create_block();
        let default_block = ctx.create_block();
        let merge_block = ctx.create_block();
        ctx.append_block_param(merge_block, ElemOf::<S>::scalar_type());
        ctx.brif(in_bounds, get_block, &[], default_block, &[]);

        ctx.switch_to_block(get_block);
        ctx.seal_block(get_block);
        let element_ptr = element_addr::<S>(ctx, data_ptr, index);
        let value = ctx.load(ElemOf::<S>::scalar_type(), element_ptr, 0,);
        ctx.jump(merge_block, &[value]);

        ctx.switch_to_block(default_block);
        ctx.seal_block(default_block);
        let default = self.default.codegen(ctx);
        ctx.jump(merge_block, &[default]);

        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);
        ctx.block_param(merge_block, 0)
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
        let in_bounds = ctx.icmp(IntCC::UnsignedLessThan, index, len);

        let set_block = ctx.create_block();
        let out_of_bounds_block = ctx.create_block();
        let merge_block = ctx.create_block();
        ctx.append_block_param(merge_block, ScalarType::I8);
        ctx.brif(in_bounds, set_block, &[], out_of_bounds_block, &[]);

        ctx.switch_to_block(set_block);
        ctx.seal_block(set_block);
        let value = self.value.codegen(ctx);
        let element_ptr = element_addr::<S>(ctx, data_ptr, index);
        ctx.store(value, element_ptr, 0);
        let written = ctx.iconst(ScalarType::I8, 1);
        ctx.jump(merge_block, &[written]);

        ctx.switch_to_block(out_of_bounds_block);
        ctx.seal_block(out_of_bounds_block);
        let not_written = ctx.iconst(ScalarType::I8, 0);
        ctx.jump(merge_block, &[not_written]);

        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);
        ctx.block_param(merge_block, 0)
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
        let element_ptr = element_addr::<S>(ctx, data_ptr, index);
        ctx.load(ElemOf::<S>::scalar_type(), element_ptr, 0,)
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
        let element_ptr = element_addr::<S>(ctx, data_ptr, index);
        ctx.store(value, element_ptr, 0);
        ctx.get_unit_value()
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
        let addr_i = element_addr::<S>(ctx, data_ptr, i);
        let addr_j = element_addr::<S>(ctx, data_ptr, j);

        let ty = ElemOf::<S>::scalar_type();
        let vi = ctx.load(ty, addr_i, 0);
        let vj = ctx.load(ty, addr_j, 0);
        ctx.store(vj, addr_i, 0);
        ctx.store(vi, addr_j, 0);
        ctx.get_unit_value()
    }
}

// =============================================================================
// SliceSliceUnchecked: Get sub-slice (no bounds check)
// =============================================================================

/// Get a sub-slice without bounds checking, for elements `[start..end]`.
///
/// Reports `Out = S::Out`, so sub-slicing a `&[T]` yields a `&[T]` and
/// sub-slicing a `&mut [T]` yields a `&mut [T]` — slices stay closed under
/// this operation. Materializes a fresh `(ptr, len)` pair on a stack slot
/// (the memory-resolved encoding; see the module docs).
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
        let new_ptr = element_addr::<S>(ctx, data_ptr, start);
        let new_len = ctx.isub(end, start);

        // Materialize the new (ptr, len) pair on a 16-byte stack slot.
        let slot = ctx.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16, // size
            3,  // align_shift = log2(8) = 3
        ));
        let slot_ptr = ctx.stack_addr(slot, 0);
        ctx.store(new_ptr, slot_ptr, 0);
        ctx.store(new_len, slot_ptr, 8);
        slot_ptr
    }
}

// =============================================================================
// Extension trait for Var<SRef<Slice<T>>> - Immutable slice operations
// =============================================================================

/// Extension trait for immutable slice operations.
pub trait SliceRefOps<'a, T: StagedType + 'a>:
    Staged<Out = SRef<'a, Slice<T>>> + Sized + Clone
{
    /// Get the length of the slice.
    fn len(self) -> SliceLen<Self> {
        SliceLen { slice: self }
    }

    /// Get the raw data pointer.
    fn as_ptr(self) -> SliceAsPtr<Self> {
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

impl<'a, T: StagedType + 'a, S> SliceRefOps<'a, T> for S where
    S: Staged<Out = SRef<'a, Slice<T>>> + Clone
{
}

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

type MutFieldSlice<'stage, T, F, E> = AsMutSlice<FieldAddr<VarUse<SRefMut<'stage, T>>, F>, E>;

impl<'borrow, 'stage, T, F> MutField<'borrow, 'stage, T, F>
where
    T: StagedType + 'stage,
    F: Field<Parent = T>,
    F::Out: 'stage,
{
    fn as_mut_slice_once<E>(&mut self) -> MutFieldSlice<'stage, T, F, E>
    where
        E: StagedType + 'stage,
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
    pub unsafe fn slice_len<E>(&mut self) -> SliceLen<MutFieldSlice<'stage, T, F, E>>
    where
        E: StagedType + 'stage,
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
    ) -> SliceGetUnchecked<MutFieldSlice<'stage, T, F, E>, I::Staged>
    where
        E: CopyType + 'stage,
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
    ) -> SliceSetUnchecked<MutFieldSlice<'stage, T, F, E>, I::Staged, V::Staged>
    where
        E: StagedType + 'stage,
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

impl<'a, T: StagedType + 'a> Var<SRefMut<'a, Slice<T>>> {
    /// Reborrow this unique slice handle to read its length.
    pub fn len(&self) -> SliceLen<VarUse<SRefMut<'a, Slice<T>>>> {
        SliceLen {
            slice: self.use_once(),
        }
    }

    /// Read `index` when it is in bounds, otherwise return `default`.
    pub fn get_or<I, D>(
        &self,
        index: I,
        default: D,
    ) -> SliceGetOr<VarUse<SRefMut<'a, Slice<T>>>, I::Staged, D::Staged>
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
    ) -> SliceSet<VarUse<SRefMut<'a, Slice<T>>>, I::Staged, V::Staged>
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
    ) -> SliceGetUnchecked<VarUse<SRefMut<'a, Slice<T>>>, I::Staged>
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
    ) -> SliceSetUnchecked<VarUse<SRefMut<'a, Slice<T>>>, I::Staged, V::Staged>
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
    ) -> SliceSwapUnchecked<VarUse<SRefMut<'a, Slice<T>>>, I::Staged, J::Staged>
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
    /// fn overlapping(slice: Var<SRefMut<'static, Slice<i64>>>) {
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
pub trait SliceMutOps<'a, T: StagedType + 'a>: Staged<Out = SRefMut<'a, Slice<T>>> + Sized {
    /// Get the length of the slice.
    fn len(self) -> SliceLen<Self> {
        SliceLen { slice: self }
    }

    /// Get the raw mutable data pointer.
    fn as_mut_ptr(self) -> SliceAsPtr<Self> {
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

impl<'a, P, R, T> SliceMutOps<'a, T> for AsMutSlice<P, T>
where
    P: Staged<Out = SRefMut<'a, R>>,
    R: MutSliceRepr<T> + 'a,
    T: StagedType + 'a,
{
}

impl<'a, S, START, END, T> SliceMutOps<'a, T> for SliceSliceUnchecked<S, START, END>
where
    S: Staged<Out = SRefMut<'a, Slice<T>>>,
    START: Staged<Out = u64>,
    END: Staged<Out = u64>,
    T: StagedType + 'a,
{
}

#[cfg(test)]
mod tests {
    // Tests will be added in a separate file
}
