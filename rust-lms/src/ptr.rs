//! Pointer types and memory operations for staged computations.
//!
//! This module provides:
//! - `SPtr<T>`: Immutable pointer type (surfaces as &T in runtime)
//! - `SMutPtr<T>`: Mutable pointer type (surfaces as &mut T in runtime)
//! - Memory operations: Load, Store, PtrOffset, ArrayIndex

use crate::staged::{CompilationContext, Staged};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, InstBuilder, MemFlags};
use std::marker::PhantomData;

// =============================================================================
// Pointer Types
// =============================================================================

/// Immutable pointer type - the base for read-only memory access.
///
/// In Cranelift IR, this is represented as an i64 (pointer-sized value).
/// At runtime, this surfaces as `*const T::RuntimeValue` or `&T::RuntimeValue`
/// when used in function signatures.
#[derive(Clone, Copy, Debug)]
pub struct SPtr<T: StagedType> {
    _phantom: PhantomData<T>,
}

impl<T: StagedType> StagedType for SPtr<T> {
    type RuntimeValue<'a> = *const T::RuntimeValue<'a>;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

/// Mutable pointer type - the base for read-write memory access.
///
/// In Cranelift IR, this is represented as an i64 (pointer-sized value).
/// At runtime, this surfaces as `*mut T::RuntimeValue` or `&mut T::RuntimeValue`
/// when used in function signatures.
#[derive(Clone, Copy, Debug)]
pub struct SMutPtr<T: StagedType> {
    _phantom: PhantomData<T>,
}

impl<T: StagedType> StagedType for SMutPtr<T> {
    type RuntimeValue<'a> = *mut T::RuntimeValue<'a>;

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
pub struct Load<P> {
    ptr: P,
}

impl<P, T> Staged for Load<P>
where
    P: Staged<Out = SPtr<T>>,
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
pub fn load<P, T>(ptr: P) -> Load<P>
where
    P: Staged<Out = SPtr<T>>,
    T: StagedType,
{
    Load { ptr }
}

/// Load from mutable pointer (same as load from immutable pointer)
pub struct LoadMut<P> {
    ptr: P,
}

impl<P, T> Staged for LoadMut<P>
where
    P: Staged<Out = SMutPtr<T>>,
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
pub fn load_mut<P, T>(ptr: P) -> LoadMut<P>
where
    P: Staged<Out = SMutPtr<T>>,
    T: StagedType,
{
    LoadMut { ptr }
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
    P: Staged<Out = SMutPtr<T>>,
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
pub fn store<P, V, T>(ptr: P, val: V) -> Store<P, V>
where
    P: Staged<Out = SMutPtr<T>>,
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
    P: Staged<Out = SPtr<T>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
{
    type Out = SPtr<T>;

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
    P: Staged<Out = SPtr<T>>,
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
    P: Staged<Out = SMutPtr<T>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
{
    type Out = SMutPtr<T>;

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
    P: Staged<Out = SMutPtr<T>>,
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
    P: Staged<Out = SPtr<T>>,
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
    P: Staged<Out = SPtr<T>>,
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
        let write_fn = compiler.fun1("write_42", |ptr: Var<SMutPtr<I64Type>>| {
            (store(ptr, Const::<I64Type>::new(42)), load_mut(ptr))
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
        let read_fn = compiler.fun1("read", |ptr: Var<SPtr<F64Type>>| load(ptr));

        let compiled = compiler.compile(read_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let value = 3.14159;
        let result = f(&value);

        assert_eq!(result, 3.14159);
    }

    #[test]
    fn test_ptr_offset() {
        let mut compiler = Compiler::new();

        // fn test_offset(ptr: &i64) -> i64
        // Reads from ptr[2]
        let test_fn = compiler.fun1("test_offset", |ptr: Var<SPtr<I64Type>>| {
            // Read from ptr[2]
            load(ptr_offset(ptr, Const::<I64Type>::new(2)))
        });

        let compiled = compiler.compile(test_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let array = [10i64, 20, 30, 40, 50];
        let result = f(array.as_ptr());

        assert_eq!(result, 30); // array[2]
    }

    #[test]
    fn test_array_index() {
        let mut compiler = Compiler::new();

        // fn get_third(ptr: &i64) -> i64
        // Returns ptr[3]
        let get_third = compiler.fun1("get_third", |ptr: Var<SPtr<I64Type>>| {
            array_index(ptr, Const::<I64Type>::new(3))
        });

        let compiled = compiler.compile(get_third).expect("compilation failed");
        let f = compiled.as_fn();

        let array = [100i64, 200, 300, 400, 500];
        let result = f(array.as_ptr());

        assert_eq!(result, 400); // array[3]
    }

    #[test]
    fn test_store_to_array() {
        let mut compiler = Compiler::new();

        // fn write_to_index(ptr: &mut i64) -> ()
        // Writes 999 to ptr[1]
        let write_fn = compiler.fun1("write_to_index", |ptr: Var<SMutPtr<I64Type>>| {
            store(
                ptr_offset_mut(ptr, Const::<I64Type>::new(1)),
                Const::<I64Type>::new(999),
            )
        });

        let compiled = compiler.compile(write_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let mut array = [1i64, 2, 3, 4, 5];
        f(array.as_mut_ptr());

        assert_eq!(array, [1, 999, 3, 4, 5]); // array[1] was modified
    }

    #[test]
    fn test_swap_values() {
        let mut compiler = Compiler::new();

        // Create variables for temp storage before defining function
        let temp = compiler.var_unchecked::<I64Type>();

        // fn swap(ptr: &mut i64) -> ()
        // Swaps ptr[0] and ptr[1]
        let swap_fn = compiler.fun1("swap", |ptr: Var<SMutPtr<I64Type>>| {
            let ptr0 = ptr;
            let ptr1 = ptr_offset_mut(ptr, Const::<I64Type>::new(1));

            (
                assign(temp, load_mut(ptr0)),
                store(ptr0, load_mut(ptr1.clone())),
                store(ptr1, temp),
            )
        });

        let compiled = compiler.compile(swap_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let mut array = [10i64, 20, 30];
        f(array.as_mut_ptr());

        assert_eq!(array, [20, 10, 30]); // First two elements swapped
    }
}
