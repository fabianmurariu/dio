//! Pointer types and memory operations for staged computations.
//!
//! This module provides:
//! - `SPtr<T>`: Immutable reference type (surfaces as &T in runtime)
//! - `SMutPtr<T>`: Mutable reference type (surfaces as &mut T in runtime)
//! - Memory operations: Load, Store, PtrOffset, ArrayIndex

use crate::staged::{CompilationContext, Staged};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, InstBuilder, MemFlags};
use std::marker::PhantomData;

// =============================================================================
// reference Types
// =============================================================================

/// Immutable reference type - the base for read-only memory access.
///
/// In Cranelift IR, this is represented as an i64 (pointer-sized value).
/// At runtime, this surfaces as `*const T::RuntimeValue` or `&T::RuntimeValue`
/// when used in function signatures.
#[derive(Clone, Copy, Debug)]
pub struct SRef<T: StagedType> {
    _phantom: PhantomData<T>,
}

impl<T: StagedType> StagedType for SRef<T> {
    type RuntimeValue<'a> = &'a T::RuntimeValue<'a>;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

/// Mutable reference type - the base for read-write memory access.
///
/// In Cranelift IR, this is represented as an i64 (reference-sized value).
/// At runtime, this surfaces as `*mut T::RuntimeValue` or `&mut T::RuntimeValue`
/// when used in function signatures.
#[derive(Clone, Copy, Debug)]
pub struct SMutRef<T: StagedType> {
    _phantom: PhantomData<T>,
}

impl<T: StagedType> StagedType for SMutRef<T> {
    type RuntimeValue<'a> = &'a mut T::RuntimeValue<'a>;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

// =============================================================================
// Load: Dereference a pointer to read a value
// =============================================================================

/// Load value from pointer: `*ptr`
///
/// Dereferences a pointer to read the value it points to.
///
/// # Example
/// ```ignore
/// let ptr: VarRef<SPtr<I64Type>> = ...;
/// let value = load(ptr); // value: impl Staged<Out = I64Type>
/// ```
pub struct LoadRef<P> {
    ptr: P,
}

impl<P, T> Staged for LoadRef<P>
where
    P: Staged<Out = SRef<T>>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr_val = self.ptr.codegen(ctx);
        // Load T from memory at ptr with trusted flags (no null check, aligned)
        ctx.builder.ins().load(
            T::cranelift_type(),
            MemFlags::trusted(),
            ptr_val,
            0, // offset
        )
    }
}

/// Create a load operation from an immutable pointer
pub fn load_ref<P, T>(ptr: P) -> LoadRef<P>
where
    P: Staged<Out = SRef<T>>,
    T: StagedType,
{
    LoadRef { ptr }
}

/// Load from mutable pointer (same as load from immutable pointer)
pub struct LoadMutRef<P> {
    ptr: P,
}

impl<P, T> Staged for LoadMutRef<P>
where
    P: Staged<Out = SMutRef<T>>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr_val = self.ptr.codegen(ctx);
        ctx.builder
            .ins()
            .load(T::cranelift_type(), MemFlags::trusted(), ptr_val, 0)
    }
}

/// Create a load operation from a mutable pointer
pub fn load_ref_mut<P, T>(ptr: P) -> LoadMutRef<P>
where
    P: Staged<Out = SMutRef<T>>,
    T: StagedType,
{
    LoadMutRef { ptr }
}

// =============================================================================
// Store: Write a value to a pointer
// =============================================================================

/// Store value to pointer: `*ptr = val`
///
/// Writes a value to the memory location pointed to by a mutable pointer.
///
/// # Example
/// ```ignore
/// let ptr: VarRef<SMutPtr<I64Type>> = ...;
/// let val = Const::<I64Type>::new(42);
/// store(ptr, val) // Writes 42 to *ptr
/// ```
pub struct Store<P, V> {
    ptr: P,
    val: V,
}

impl<P, V, T> Staged for Store<P, V>
where
    P: Staged<Out = SMutRef<T>>,
    V: Staged<Out = T>,
    T: StagedType,
{
    type Out = crate::types::UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr_val = self.ptr.codegen(ctx);
        let value = self.val.codegen(ctx);

        // Store value to memory at ptr
        ctx.builder.ins().store(
            MemFlags::trusted(),
            value,
            ptr_val,
            0, // offset
        );

        // Return unit value (store is used for side effects)
        ctx.builder.ins().iconst(types::I8, 0)
    }
}

/// Create a store operation
pub fn store_ref<P, V, T>(ptr: P, val: V) -> Store<P, V>
where
    P: Staged<Out = SMutRef<T>>,
    V: Staged<Out = T>,
    T: StagedType,
{
    Store { ptr, val }
}

// =============================================================================
// PtrOffset: Pointer arithmetic with element scaling
// =============================================================================

/// Pointer offset: `ptr + offset` (with scaling by element size)
///
/// Adds an offset to a pointer, automatically scaling by the size of the
/// pointed-to type. This is equivalent to C's `ptr[offset]` pointer arithmetic.
///
/// # Example
/// ```ignore
/// let ptr: VarRef<SPtr<I64Type>> = ...;
/// let idx = Const::<I64Type>::new(3);
/// let offset_ptr = ptr_offset(ptr, idx); // ptr + 3 * sizeof(i64)
/// ```
#[derive(Clone)]
pub struct PtrOffset<P, I> {
    ptr: P,
    index: I,
}

impl<P, I, T> Staged for PtrOffset<P, I>
where
    P: Staged<Out = SRef<T>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
{
    type Out = SRef<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        // Calculate byte offset = index * sizeof(T)
        let element_size = std::mem::size_of::<T::RuntimeValue<'static>>() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(idx, scale);

        // Add to base pointer
        ctx.builder.ins().iadd(ptr, byte_offset)
    }
}

/// Create a pointer offset operation for immutable pointers
pub fn ptr_offset<P, I, T>(ptr: P, index: I) -> PtrOffset<P, I>
where
    P: Staged<Out = SRef<T>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
{
    PtrOffset { ptr, index }
}

/// Pointer offset for mutable pointers
#[derive(Clone)]
pub struct PtrOffsetMut<P, I> {
    ptr: P,
    index: I,
}

impl<P, I, T> Staged for PtrOffsetMut<P, I>
where
    P: Staged<Out = SMutRef<T>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
{
    type Out = SMutRef<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        let element_size = std::mem::size_of::<T::RuntimeValue<'static>>() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(idx, scale);

        ctx.builder.ins().iadd(ptr, byte_offset)
    }
}

/// Create a pointer offset operation for mutable pointers
pub fn ptr_offset_mut<P, I, T>(ptr: P, index: I) -> PtrOffsetMut<P, I>
where
    P: Staged<Out = SMutRef<T>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
{
    PtrOffsetMut { ptr, index }
}

// =============================================================================
// ArrayIndex: Combined pointer offset + load
// =============================================================================

/// Array indexing: `ptr[index]`
///
/// Combines pointer offset and load in one operation. Equivalent to
/// `load(ptr_offset(ptr, index))`.
///
/// # Example
/// ```ignore
/// let array: VarRef<SPtr<I64Type>> = ...;
/// let idx = Const::<I64Type>::new(5);
/// let elem = array_index(array, idx); // array[5]
/// ```
pub struct ArrayIndex<P, I> {
    ptr: P,
    index: I,
}

impl<P, I, T> Staged for ArrayIndex<P, I>
where
    P: Staged<Out = SRef<T>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        // Combine offset + load for efficiency
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        // Calculate byte offset
        let element_size = std::mem::size_of::<T::RuntimeValue<'static>>() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(idx, scale);

        // Add to base pointer
        let offset_ptr = ctx.builder.ins().iadd(ptr, byte_offset);

        // Load from offset pointer
        ctx.builder
            .ins()
            .load(T::cranelift_type(), MemFlags::trusted(), offset_ptr, 0)
    }
}

/// Create an array indexing operation
pub fn array_index<P, I, T>(ptr: P, index: I) -> ArrayIndex<P, I>
where
    P: Staged<Out = SRef<T>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
{
    ArrayIndex { ptr, index }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_load_store_i64() {
        let mut compiler = Compiler::new();

        // Create a function that takes a mutable pointer and writes to it
        // fn write_42(ptr: &mut i64) -> i64
        let write_fn = compiler.fun1("write_42", |ptr: VarRef<SMutRef<I64Type>>| {
            seq(store_ref(ptr, Const::<I64Type>::new(42)), load_ref_mut(ptr))
        });

        let compiled = compiler.compile(write_fn).expect("compilation failed");
        let f = compiled.as_fn();

        // Test with actual memory
        let mut value = 0i64;
        let result = f(&mut value);

        assert_eq!(value, 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_load_f64() {
        let mut compiler = Compiler::new();

        // fn read(ptr: &f64) -> f64
        let read_fn = compiler.fun1("read", |ptr: VarRef<SRef<F64Type>>| load_ref(ptr));

        let compiled = compiler.compile(read_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let value = 3.14159;
        let result = f(&value);

        assert_eq!(result, 3.14159);
    }

    #[test]
    fn store_into_array_f64() {
        let mut compiler = Compiler::new();

        // fn store_value(ptr: &mut f64) -> ()
        let store_fn = compiler.fun1("store_value", |ptr: VarRef<SMutRef<F64Type>>| {
            store_ref(ptr, Const::<F64Type>::new(2.71828))
        });

        let compiled = compiler.compile(store_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let mut array = [0.0f64; 3];
        f(&mut array[1]); // Store into array[1]

        assert_eq!(array, [0.0, 2.71828, 0.0]);
    }
}
