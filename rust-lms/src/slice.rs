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
//! Internally, they're stored in a stack slot and accessed via pointer.
//!
//! # Example
//!
//! ```ignore
//! // Sum all elements in a slice
//! let sum = compiler.fun1("sum", |arr: Var<SRef<Slice<I64Type>>>| {
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
use crate::types::{CopyType, StagedType, U64Type, UnitType};
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, StackSlotData, StackSlotKind};
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
}

// =============================================================================
// SliceLen: Get length of a slice
// =============================================================================

/// Get the length of an immutable slice.
#[derive(Clone, Copy)]
pub struct SliceLen<S> {
    slice: S,
}

impl<S, T> Staged for SliceLen<S>
where
    S: Staged<Out = SRef<Slice<T>>>,
    T: StagedType,
{
    type Out = U64Type;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        // slice is a pointer to (ptr, len) pair
        let slice_ptr = self.slice.codegen(ctx);
        // Load len from offset 8
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 8)
    }
}

/// Get the length of a mutable slice.
#[derive(Clone, Copy)]
pub struct SliceLenMut<S> {
    slice: S,
}

impl<S, T> Staged for SliceLenMut<S>
where
    S: Staged<Out = SRefMut<Slice<T>>>,
    T: StagedType,
{
    type Out = U64Type;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 8)
    }
}

// =============================================================================
// SliceAsPtr: Get raw pointer to slice data
// =============================================================================

/// Get the raw data pointer from an immutable slice.
#[derive(Clone, Copy)]
pub struct SliceAsPtr<'a, S> {
    slice: S,
    _phantom: PhantomData<&'a ()>,
}

impl<'a, S, T> Staged for SliceAsPtr<'a, S>
where
    S: Staged<Out = SRef<'a, Slice<T>>>,
    T: StagedType,
    T: 'a
{
    type Out = SRef<'a, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        // Load ptr from offset 0
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0)
    }
}

/// Get the raw mutable data pointer from a mutable slice.
#[derive(Clone, Copy)]
pub struct SliceAsMutPtr<S> {
    slice: S,
}

impl<S, T> Staged for SliceAsMutPtr<S>
where
    S: Staged<Out = SRefMut<Slice<T>>>,
    T: StagedType,
{
    type Out = SRefMut<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        ctx.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0)
    }
}

// =============================================================================
// SliceGetRefUnchecked: Get reference to element (no bounds check)
// =============================================================================

/// Get an immutable reference to an element without bounds checking.
#[derive(Clone, Copy)]
pub struct SliceGetRefUnchecked<S, I> {
    slice: S,
    index: I,
}

impl<S, I, T> Staged for SliceGetRefUnchecked<S, I>
where
    S: Staged<Out = SRef<Slice<T>>>,
    I: Staged<Out = U64Type>,
    T: StagedType,
{
    type Out = SRef<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        let index = self.index.codegen(ctx);

        // Load data pointer from slice
        let data_ptr = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0);

        // Compute element address: data_ptr + index * sizeof(T)
        let element_size = T::size_of() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(index, scale);
        ctx.builder.ins().iadd(data_ptr, byte_offset)
    }
}

/// Get a mutable reference to an element without bounds checking.
#[derive(Clone, Copy)]
pub struct SliceGetMutUnchecked<S, I> {
    slice: S,
    index: I,
}

impl<S, I, T> Staged for SliceGetMutUnchecked<S, I>
where
    S: Staged<Out = SRefMut<Slice<T>>>,
    I: Staged<Out = U64Type>,
    T: StagedType,
{
    type Out = SRefMut<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        let index = self.index.codegen(ctx);

        let data_ptr = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0);

        let element_size = T::size_of() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(index, scale);
        ctx.builder.ins().iadd(data_ptr, byte_offset)
    }
}

// =============================================================================
// SliceGetUnchecked: Get element by value (no bounds check, CopyType only)
// =============================================================================

/// Get an element by value from an immutable slice without bounds checking.
#[derive(Clone, Copy)]
pub struct SliceGetUnchecked<S, I> {
    slice: S,
    index: I,
}

impl<S, I, T> Staged for SliceGetUnchecked<S, I>
where
    S: Staged<Out = SRef<Slice<T>>>,
    I: Staged<Out = U64Type>,
    T: StagedType + CopyType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        let index = self.index.codegen(ctx);

        // Load data pointer
        let data_ptr = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0);

        // Compute element address
        let element_size = T::size_of() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(index, scale);
        let element_ptr = ctx.builder.ins().iadd(data_ptr, byte_offset);

        // Load the element value
        ctx.builder
            .ins()
            .load(T::cranelift_type(), MemFlags::trusted(), element_ptr, 0)
    }
}

/// Get an element by value from a mutable slice without bounds checking.
#[derive(Clone, Copy)]
pub struct SliceGetUncheckedMut<S, I> {
    slice: S,
    index: I,
}

impl<S, I, T> Staged for SliceGetUncheckedMut<S, I>
where
    S: Staged<Out = SRefMut<Slice<T>>>,
    I: Staged<Out = U64Type>,
    T: StagedType + CopyType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        let index = self.index.codegen(ctx);

        let data_ptr = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0);

        let element_size = T::size_of() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(index, scale);
        let element_ptr = ctx.builder.ins().iadd(data_ptr, byte_offset);

        ctx.builder
            .ins()
            .load(T::cranelift_type(), MemFlags::trusted(), element_ptr, 0)
    }
}

// =============================================================================
// SliceSetUnchecked: Set element (no bounds check)
// =============================================================================

/// Set an element without bounds checking.
#[derive(Clone, Copy)]
pub struct SliceSetUnchecked<S, I, V> {
    slice: S,
    index: I,
    value: V,
}

impl<S, I, V, T> Staged for SliceSetUnchecked<S, I, V>
where
    S: Staged<Out = SRefMut<Slice<T>>>,
    I: Staged<Out = U64Type>,
    V: Staged<Out = T>,
    T: StagedType,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        let index = self.index.codegen(ctx);
        let value = self.value.codegen(ctx);

        // Load data pointer
        let data_ptr = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0);

        // Compute element address
        let element_size = T::size_of() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(index, scale);
        let element_ptr = ctx.builder.ins().iadd(data_ptr, byte_offset);

        // Store the value
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), value, element_ptr, 0);

        // Return unit
        ctx.builder.ins().iconst(types::I8, 0)
    }
}

// =============================================================================
// SliceSliceUnchecked: Get sub-slice (no bounds check)
// =============================================================================

/// Get a sub-slice without bounds checking.
///
/// Creates a new (ptr, len) pair pointing to elements [start..end].
#[derive(Clone, Copy)]
pub struct SliceSliceUnchecked<S, START, END> {
    slice: S,
    start: START,
    end: END,
}

impl<S, START, END, T> Staged for SliceSliceUnchecked<S, START, END>
where
    S: Staged<Out = SRef<Slice<T>>>,
    START: Staged<Out = U64Type>,
    END: Staged<Out = U64Type>,
    T: StagedType,
{
    type Out = SRef<Slice<T>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        let start = self.start.codegen(ctx);
        let end = self.end.codegen(ctx);

        // Load original data pointer
        let data_ptr = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0);

        // Compute new pointer: data_ptr + start * sizeof(T)
        let element_size = T::size_of() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let start_offset = ctx.builder.ins().imul(start, scale);
        let new_ptr = ctx.builder.ins().iadd(data_ptr, start_offset);

        // Compute new length: end - start
        let new_len = ctx.builder.ins().isub(end, start);

        // Create stack slot for the new (ptr, len) pair
        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16, // size
            3,  // align_shift = log2(8) = 3
        ));
        let slot_ptr = ctx.builder.ins().stack_addr(types::I64, slot, 0);

        // Store ptr at offset 0, len at offset 8
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), new_ptr, slot_ptr, 0);
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), new_len, slot_ptr, 8);

        // Return pointer to the stack slot
        slot_ptr
    }
}

/// Get a mutable sub-slice without bounds checking.
#[derive(Clone, Copy)]
pub struct SliceSliceMutUnchecked<S, START, END> {
    slice: S,
    start: START,
    end: END,
}

impl<S, START, END, T> Staged for SliceSliceMutUnchecked<S, START, END>
where
    S: Staged<Out = SRefMut<Slice<T>>>,
    START: Staged<Out = U64Type>,
    END: Staged<Out = U64Type>,
    T: StagedType,
{
    type Out = SRefMut<Slice<T>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let slice_ptr = self.slice.codegen(ctx);
        let start = self.start.codegen(ctx);
        let end = self.end.codegen(ctx);

        let data_ptr = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0);

        let element_size = T::size_of() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let start_offset = ctx.builder.ins().imul(start, scale);
        let new_ptr = ctx.builder.ins().iadd(data_ptr, start_offset);

        let new_len = ctx.builder.ins().isub(end, start);

        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16,
            3,
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
pub trait SliceRefOps<T: StagedType>: Staged<Out = SRef<Slice<T>>> + Sized + Clone {
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
    /// `arr.get_ref_unchecked(Const::<U64Type>::new(5))`.
    fn get_ref_unchecked<I>(self, index: I) -> SliceGetRefUnchecked<Self, I::Staged>
    where
        I: IntoStaged<U64Type>,
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
        I: IntoStaged<U64Type>,
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
        START: IntoStaged<U64Type>,
        END: IntoStaged<U64Type>,
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
// Extension trait for Var<SRefMut<Slice<T>>> - Mutable slice operations
// =============================================================================

/// Extension trait for mutable slice operations.
pub trait SliceMutOps<T: StagedType>: Staged<Out = SRefMut<Slice<T>>> + Sized + Clone {
    /// Get the length of the slice.
    fn len(self) -> SliceLenMut<Self> {
        SliceLenMut { slice: self }
    }

    /// Get the raw mutable data pointer.
    fn as_mut_ptr(self) -> SliceAsMutPtr<Self> {
        SliceAsMutPtr { slice: self }
    }

    /// Get a mutable reference to an element without bounds checking.
    fn get_mut_unchecked<I>(self, index: I) -> SliceGetMutUnchecked<Self, I::Staged>
    where
        I: IntoStaged<U64Type>,
    {
        SliceGetMutUnchecked {
            slice: self,
            index: index.into_staged(),
        }
    }

    /// Get an element by value without bounds checking.
    fn get_unchecked<I>(self, index: I) -> SliceGetUncheckedMut<Self, I::Staged>
    where
        I: IntoStaged<U64Type>,
        T: CopyType,
    {
        SliceGetUncheckedMut {
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
        I: IntoStaged<U64Type>,
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
    ) -> SliceSliceMutUnchecked<Self, START::Staged, END::Staged>
    where
        START: IntoStaged<U64Type>,
        END: IntoStaged<U64Type>,
    {
        SliceSliceMutUnchecked {
            slice: self,
            start: start.into_staged(),
            end: end.into_staged(),
        }
    }
}

impl<T: StagedType, S> SliceMutOps<T> for S where S: Staged<Out = SRefMut<Slice<T>>> + Clone {}

#[cfg(test)]
mod tests {
    // Tests will be added in a separate file
}
