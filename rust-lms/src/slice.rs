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
//! At the ABI boundary, slice references are passed as 2 x i64 (ptr, len).
//!
//! # Canonical Staged representation
//!
//! Within the staged graph a slice's `codegen` value is always a single `i64`,
//! resolved one of two ways (see [`CompilationContext::slice_data_ptr`] /
//! [`CompilationContext::slice_len`], which are the only code that knows this):
//!
//! - **register-resolved** — slice *parameters* are split at the ABI boundary
//!   into two Cranelift variables (`ptr_var`, `len_var`) kept in
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

use crate::refer::{SRef, SRefMut};
use crate::staged::{CompilationContext, IntoStaged, Staged};
use crate::types::{CopyType, StagedType};
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value};
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

impl<P: Clone, T> Clone for AsSlice<P, T> {
    fn clone(&self) -> Self {
        Self {
            repr: self.repr.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, T> Copy for AsSlice<P, T> {}

impl<'a, P, R, T> Staged for AsSlice<P, T>
where
    P: Staged<Out = SRef<'a, R>>,
    R: StagedType + 'a,
    T: StagedType + 'a,
{
    type Out = SRef<'a, Slice<T>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.repr.codegen(ctx)
    }
}

/// Extension trait for values that point at a repr-compatible `(ptr, len)`.
pub trait ReprSliceOps<'a, R>: Staged<Out = SRef<'a, R>> + Sized
where
    R: StagedType + 'a,
{
    /// Reinterpret the pointed-to representation as a staged slice of `T`.
    fn as_slice<T>(self) -> AsSlice<Self, T>
    where
        T: StagedType + 'a,
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
    fn as_mut_slice<T>(self) -> AsMutSlice<Self, T>
    where
        T: StagedType + 'a,
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

impl<'a, P, R, T> Staged for AsMutSlice<P, T>
where
    P: Staged<Out = SRefMut<'a, R>>,
    R: StagedType + 'a,
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

impl<'a, T: StagedType> StagedType for SRef<'a, Slice<T>> {
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

    fn num_abi_values() -> usize {
        2 // ptr, len
    }

    fn abi_types() -> Vec<cranelift_codegen::ir::Type> {
        vec![types::I64, types::I64]
    }

    fn is_fat_pointer() -> bool {
        true // Slice references are fat pointers
    }
}

// =============================================================================
// StagedType for SRefMut<Slice<T>> - Mutable Fat Pointer
// =============================================================================

impl<'a, T: StagedType> StagedType for SRefMut<'a, Slice<T>> {
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
        true // Fat pointer is Copy (the pointer itself, not what it points to)
    }

    fn num_abi_values() -> usize {
        2
    }

    fn abi_types() -> Vec<cranelift_codegen::ir::Type> {
        vec![types::I64, types::I64]
    }

    fn is_fat_pointer() -> bool {
        true // Mutable slice references are fat pointers
    }
}

// =============================================================================
// SliceType: unify immutable and mutable slice fat pointers
// =============================================================================

/// A staged slice fat-pointer type. Implemented by both `SRef<Slice<T>>`
/// (`&[T]`) and `SRefMut<Slice<T>>` (`&mut [T]`), so a *single* op impl can
/// serve both — the mutability lives entirely in the associated `ElemRef`.
///
/// This is what keeps slices *closed under sub-slicing*: every op below is
/// generic over `S::Out: SliceType`, and `SliceSliceUnchecked` reports
/// `Out = S::Out`, so the result of `slice_unchecked` is itself a slice that
/// supports `len`/`get`/`slice_unchecked`/… all over again — and a sub-slice
/// of a `&mut [T]` stays mutable.
pub trait SliceType: StagedType {
    /// Element type (`T`).
    type Elem: StagedType;
    /// Reference-to-element produced by `as_ptr`/`get_ref_unchecked`:
    /// `SRef<T>` for an immutable slice, `SRefMut<T>` for a mutable one.
    type ElemRef: StagedType;
}

impl<'a, T: StagedType> SliceType for SRef<'a, Slice<T>> {
    type Elem = T;
    type ElemRef = SRef<'a, T>;
}

impl<'a, T: StagedType> SliceType for SRefMut<'a, Slice<T>> {
    type Elem = T;
    type ElemRef = SRefMut<'a, T>;
}

/// Marker for *mutable* slices (`SRefMut<Slice<T>>`). Gates the writing ops
/// (`set_unchecked`) so they cannot be called on an immutable slice.
pub trait MutSliceType: SliceType {}

impl<'a, T: StagedType> MutSliceType for SRefMut<'a, Slice<T>> {}

/// Convenience accessor for `S`'s element type inside generic op impls.
type ElemOf<S> = <<S as Staged>::Out as SliceType>::Elem;

/// Emit `data_ptr + index * sizeof(Elem)`, the address of element `index`.
fn element_addr<S>(ctx: &mut CompilationContext, data_ptr: Value, index: Value) -> Value
where
    S: Staged,
    S::Out: SliceType,
{
    let element_size = ElemOf::<S>::size_of() as i64;
    let scale = ctx.builder.ins().iconst(types::I64, element_size);
    let byte_offset = ctx.builder.ins().imul(index, scale);
    ctx.builder.ins().iadd(data_ptr, byte_offset)
}

// =============================================================================
// SliceLen: Get length of a slice
// =============================================================================

/// Get the length of a slice (immutable or mutable).
#[derive(Clone, Copy)]
pub struct SliceLen<S> {
    slice: S,
}

impl<S> Staged for SliceLen<S>
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

/// Get the data pointer of a slice. Yields `SRef<T>` for an immutable slice and
/// `SRefMut<T>` for a mutable one (via `SliceType::ElemRef`).
#[derive(Clone, Copy)]
pub struct SliceAsPtr<S> {
    slice: S,
}

impl<S> Staged for SliceAsPtr<S>
where
    S: Staged,
    S::Out: SliceType,
{
    type Out = <S::Out as SliceType>::ElemRef;

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

impl<S, I> Staged for SliceGetRefUnchecked<S, I>
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

// =============================================================================
// SliceGetUnchecked: Get element by value (no bounds check, CopyType only)
// =============================================================================

/// Get an element by value without bounds checking (`CopyType` elements only).
#[derive(Clone, Copy)]
pub struct SliceGetUnchecked<S, I> {
    slice: S,
    index: I,
}

impl<S, I> Staged for SliceGetUnchecked<S, I>
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
        ctx.builder.ins().load(
            ElemOf::<S>::cranelift_type(),
            MemFlags::trusted(),
            element_ptr,
            0,
        )
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

impl<S, I, V> Staged for SliceSetUnchecked<S, I, V>
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
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), value, element_ptr, 0);
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

impl<S, I, J> Staged for SliceSwapUnchecked<S, I, J>
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

        let ty = ElemOf::<S>::cranelift_type();
        let vi = ctx.builder.ins().load(ty, MemFlags::trusted(), addr_i, 0);
        let vj = ctx.builder.ins().load(ty, MemFlags::trusted(), addr_j, 0);
        ctx.builder.ins().store(MemFlags::trusted(), vj, addr_i, 0);
        ctx.builder.ins().store(MemFlags::trusted(), vi, addr_j, 0);
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

impl<S, START, END> Staged for SliceSliceUnchecked<S, START, END>
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
        let new_len = ctx.builder.ins().isub(end, start);

        // Materialize the new (ptr, len) pair on a 16-byte stack slot.
        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16, // size
            3,  // align_shift = log2(8) = 3
        ));
        let slot_ptr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), new_ptr, slot_ptr, 0);
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), new_len, slot_ptr, 8);
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

    /// Get a reference to an element without bounds checking.
    ///
    /// Accepts any value that can be converted into a u64 staged expression for the index.
    /// This allows ergonomic usage like `arr.get_ref_unchecked(5u64)` instead of
    /// `arr.get_ref_unchecked(Const::<u64>::new(5))`.
    fn get_ref_unchecked<I>(self, index: I) -> SliceGetRefUnchecked<Self, I::Staged>
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
    fn get_unchecked<I>(self, index: I) -> SliceGetUnchecked<Self, I::Staged>
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
    fn slice_unchecked<START, END>(
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
// Extension trait for Var<SRefMut<Slice<T>>> - Mutable slice operations
// =============================================================================

/// Extension trait for mutable slice operations.
///
/// The read-only ops (`len`, `as_mut_ptr`, `get_*`, `slice_mut_unchecked`)
/// build the same unified op structs as [`SliceRefOps`]; the associated
/// `ElemRef`/`Out` keep their mutable flavor automatically. `set_unchecked` is
/// gated on `MutSliceType`, so it only exists here.
pub trait SliceMutOps<'a, T: StagedType + 'a>:
    Staged<Out = SRefMut<'a, Slice<T>>> + Sized + Clone
{
    /// Get the length of the slice.
    fn len(self) -> SliceLen<Self> {
        SliceLen { slice: self }
    }

    /// Get the raw mutable data pointer.
    fn as_mut_ptr(self) -> SliceAsPtr<Self> {
        SliceAsPtr { slice: self }
    }

    /// Get a mutable reference to an element without bounds checking.
    fn get_mut_unchecked<I>(self, index: I) -> SliceGetRefUnchecked<Self, I::Staged>
    where
        I: IntoStaged<u64>,
    {
        SliceGetRefUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Get an element by value without bounds checking.
    fn get_unchecked<I>(self, index: I) -> SliceGetUnchecked<Self, I::Staged>
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
    fn set_unchecked<I, V>(
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
    fn slice_mut_unchecked<START, END>(
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
    fn swap_unchecked<I, J>(self, i: I, j: J) -> SliceSwapUnchecked<Self, I::Staged, J::Staged>
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

impl<'a, T: StagedType + 'a, S> SliceMutOps<'a, T> for S where
    S: Staged<Out = SRefMut<'a, Slice<T>>> + Clone
{
}

#[cfg(test)]
mod tests {
    // Tests will be added in a separate file
}
