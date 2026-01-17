//! Reference and pointer types for staged computations.
//!
//! This module provides:
//! - `SRef<T>`: Immutable reference type (`&T` at runtime)
//! - `SRefMut<T>`: Mutable reference type (`&mut T` at runtime)
//! - `SPtr<T>`: Immutable raw pointer type (`*const T` at runtime)
//! - `SMutPtr<T>`: Mutable raw pointer type (`*mut T` at runtime)
//!
//! All four types are implemented via type tagging on `SRef` and `SRefMut`:
//! - `SRef<T>` = `SRef<T, RustRef>` (default)
//! - `SPtr<T>` = `SRef<T, RustPtr>`
//! - `SRefMut<T>` = `SRefMut<T, RustRef>` (default)
//! - `SMutPtr<T>` = `SRefMut<T, RustPtr>`

use crate::staged::{CompilationContext, Staged};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, InstBuilder, MemFlags};
use std::marker::PhantomData;

// =============================================================================
// Tag Types for Reference vs Pointer
// =============================================================================

/// Tag type indicating a Rust reference (`&T` or `&mut T`)
#[derive(Clone, Copy, Debug)]
pub struct RustRef;

/// Tag type indicating a raw pointer (`*const T` or `*mut T`)
#[derive(Clone, Copy, Debug)]
pub struct RustPtr;

// =============================================================================
// SRef<T, Tag> - Immutable reference/pointer type
// =============================================================================

/// Immutable reference or pointer type.
///
/// In Cranelift IR, this is represented as an i64 (pointer-sized value).
/// The `Tag` parameter determines the runtime type:
/// - `RustRef` (default): surfaces as `&T::RuntimeValue`
/// - `RustPtr`: surfaces as `*const T::RuntimeValue`
///
/// Note: The `T: StagedType` bound is only required on the `StagedType` impl,
/// not on the struct itself. This allows `SRef<Slice<T>>` to work even though
/// `Slice<T>` doesn't implement `StagedType` (since it's a DST marker).
#[derive(Clone, Copy, Debug)]
pub struct SRef<'a, T, Tag = RustRef> {
    _phantom: PhantomData<&'a (T, Tag)>,
}

impl<'a, T: StagedType> StagedType for SRef<'a, T, RustRef> {
    type RuntimeValue = &'a T::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

impl<'a, T: StagedType> StagedType for SRef<'a, T, RustPtr> {
    type RuntimeValue = *const T::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

// =============================================================================
// SRefMut<T, Tag> - Mutable reference/pointer type
// =============================================================================

/// Mutable reference or pointer type.
///
/// In Cranelift IR, this is represented as an i64 (pointer-sized value).
/// The `Tag` parameter determines the runtime type:
/// - `RustRef` (default): surfaces as `&mut T::RuntimeValue`
/// - `RustPtr`: surfaces as `*mut T::RuntimeValue`
#[derive(Clone, Copy, Debug)]
pub struct SRefMut<'a, T, Tag = RustRef> {
    _phantom: PhantomData<&'a mut (T, Tag)>,
}

impl<'a, T: StagedType> StagedType for SRefMut<'a, T, RustRef> {
    type RuntimeValue = &'a mut T::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

impl<'a, T: StagedType> StagedType for SRefMut<'a, T, RustPtr> {
    type RuntimeValue = *mut T::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

// =============================================================================
// Type Aliases for Convenience
// =============================================================================

/// Immutable raw pointer type (`*const T` at runtime).
/// Alias for `SRef<T, RustPtr>`.
pub type SPtr<T> = SRef<'static, T, RustPtr>;

/// Mutable raw pointer type (`*mut T` at runtime).
/// Alias for `SRefMut<T, RustPtr>`.
pub type SMutPtr<T> = SRefMut<'static, T, RustPtr>;

// =============================================================================
// Load: Dereference to read a value
// =============================================================================

/// Load value from immutable reference/pointer: `*ptr`
pub struct LoadRef<'a, P> {
    ptr: P,
    _marker: PhantomData<&'a ()>,
}

impl<'a, P, T, Tag> Staged for LoadRef<'a, P>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    T: StagedType,
    Tag: 'a,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr_val = self.ptr.codegen(ctx);
        ctx.builder
            .ins()
            .load(T::cranelift_type(), MemFlags::trusted(), ptr_val, 0)
    }
}

/// Create a load operation from an immutable reference/pointer
pub fn load_ref<'a, P, T, Tag>(ptr: P) -> LoadRef<'a, P>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    T: StagedType,
    Tag: 'a,
{
    LoadRef {
        ptr,
        _marker: PhantomData,
    }
}

/// Alias for `load_ref` for raw pointer semantics
pub fn load<'a, P, T>(ptr: P) -> LoadRef<'a, P>
where
    P: Staged<Out = SPtr<T>>,
    T: StagedType,
{
    LoadRef {
        ptr,
        _marker: PhantomData,
    }
}

/// Load from mutable reference/pointer
pub struct LoadMutRef<'a, P> {
    ptr: P,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a, P, T, Tag> Staged for LoadMutRef<'a, P>
where
    P: Staged<Out = SRefMut<'a, T, Tag>>,
    T: StagedType,
    Tag: 'a,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr_val = self.ptr.codegen(ctx);
        ctx.builder
            .ins()
            .load(T::cranelift_type(), MemFlags::trusted(), ptr_val, 0)
    }
}

/// Create a load operation from a mutable reference/pointer
pub fn load_ref_mut<'a, P, T, Tag>(ptr: P) -> LoadMutRef<'a, P>
where
    P: Staged<Out = SRefMut<'a, T, Tag>>,
    T: StagedType,
    Tag: 'a,
{
    LoadMutRef {
        ptr,
        _marker: PhantomData,
    }
}

/// Alias for `load_ref_mut` for raw pointer semantics
pub fn load_mut<'a, P, T>(ptr: P) -> LoadMutRef<'a, P>
where
    P: Staged<Out = SMutPtr<T>>,
    T: StagedType,
{
    LoadMutRef {
        ptr,
        _marker: PhantomData,
    }
}

// =============================================================================
// Store: Write a value
// =============================================================================

/// Store value to mutable reference/pointer: `*ptr = val`
pub struct Store<'a, P, V> {
    ptr: P,
    val: V,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a, P, V, T, Tag> Staged for Store<'a, P, V>
where
    P: Staged<Out = SRefMut<'a, T, Tag>>,
    V: Staged<Out = T>,
    T: StagedType,
    Tag: 'a,
{
    type Out = crate::types::UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr_val = self.ptr.codegen(ctx);
        let value = self.val.codegen(ctx);

        ctx.builder
            .ins()
            .store(MemFlags::trusted(), value, ptr_val, 0);

        ctx.builder.ins().iconst(types::I8, 0)
    }
}

/// Create a store operation (for references)
pub fn store_ref<'a, P, V, T, Tag>(ptr: P, val: V) -> Store<'a, P, V>
where
    P: Staged<Out = SRefMut<'a, T, Tag>>,
    V: Staged<Out = T>,
    T: StagedType,
    Tag: 'a,
{
    Store {
        ptr,
        val,
        _marker: PhantomData,
    }
}

/// Create a store operation (for raw pointers)
pub fn store<'a, P, V, T>(ptr: P, val: V) -> Store<'a, P, V>
where
    P: Staged<Out = SMutPtr<T>>,
    V: Staged<Out = T>,
    T: StagedType,
{
    Store {
        ptr,
        val,
        _marker: PhantomData,
    }
}

// =============================================================================
// PtrOffset: Pointer arithmetic with element scaling
// =============================================================================

/// Pointer offset for immutable reference/pointer
#[derive(Clone)]
pub struct PtrOffset<'a, P, I> {
    ptr: P,
    index: I,
    _marker: PhantomData<&'a ()>,
}

impl<'a, P, I, T, Tag> Staged for PtrOffset<'a, P, I>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
    SRef<'a, T, Tag>: StagedType,
{
    type Out = SRef<'a, T, Tag>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        let element_size = std::mem::size_of::<T::RuntimeValue>() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(idx, scale);

        ctx.builder.ins().iadd(ptr, byte_offset)
    }
}

/// Create a pointer offset operation for immutable reference/pointer
pub fn ptr_offset<'a, P, I, T, Tag>(ptr: P, index: I) -> PtrOffset<'a, P, I>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
    Tag: 'a,
{
    PtrOffset { ptr, index , _marker: PhantomData }
}

/// Pointer offset for mutable reference/pointer
#[derive(Clone)]
pub struct PtrOffsetMut<'a, P, I> {
    ptr: P,
    index: I,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a, P, I, T, Tag> Staged for PtrOffsetMut<'a, P, I>
where
    P: Staged<Out = SRefMut<'a, T, Tag>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
    SRefMut<'a, T, Tag>: StagedType,
{
    type Out = SRefMut<'a, T, Tag>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        let element_size = std::mem::size_of::<T::RuntimeValue>() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(idx, scale);

        ctx.builder.ins().iadd(ptr, byte_offset)
    }
}

/// Create a pointer offset operation for mutable reference/pointer
pub fn ptr_offset_mut<'a, P, I, T, Tag>(ptr: P, index: I) -> PtrOffsetMut<'a, P, I>
where
    P: Staged<Out = SRefMut<'a, T, Tag>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
    Tag: 'a,
{
    PtrOffsetMut { ptr, index, _marker: PhantomData }
}

// =============================================================================
// ArrayIndex: Combined pointer offset + load
// =============================================================================

/// Array indexing: `ptr[index]`
pub struct ArrayIndex<'a, P, I> {
    ptr: P,
    index: I,
    _marker: PhantomData<&'a ()>,
}

impl<'a, P, I, T, Tag> Staged for ArrayIndex<'a, P, I>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
    Tag: 'a,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        let element_size = std::mem::size_of::<T::RuntimeValue>() as i64;
        let scale = ctx.builder.ins().iconst(types::I64, element_size);
        let byte_offset = ctx.builder.ins().imul(idx, scale);

        let offset_ptr = ctx.builder.ins().iadd(ptr, byte_offset);

        ctx.builder
            .ins()
            .load(T::cranelift_type(), MemFlags::trusted(), offset_ptr, 0)
    }
}

/// Create an array indexing operation
pub fn array_index<'a, P, I, T, Tag>(ptr: P, index: I) -> ArrayIndex<'a, P, I>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    I: Staged<Out = crate::types::I64Type>,
    T: StagedType,
    Tag: 'a,
{
    ArrayIndex { ptr, index, _marker: PhantomData }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_load_store_i64() {
        let mut compiler = Compiler::new();

        let write_fn = compiler.fun1("write_42", |_ctx, ptr: Var<SRefMut<I64Type>>| {
            (store_ref(ptr, Const::<I64Type>::new(42)), load_ref_mut(ptr))
        });

        let compiled = compiler.compile(write_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let mut value = 0i64;
        let result = f(&mut value);

        assert_eq!(value, 42);
        assert_eq!(result, 42);
    }

    #[test]
    fn test_load_f64() {
        let mut compiler = Compiler::new();

        let read_fn = compiler.fun1("read", |_ctx, ptr: Var<SRef<F64Type>>| load_ref(ptr));

        let compiled = compiler.compile(read_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let value = 3.14159;
        let result = f(&value);

        assert_eq!(result, 3.14159);
    }

    #[test]
    fn test_raw_ptr_load_store() {
        let mut compiler = Compiler::new();

        // Using raw pointer types (SPtr/SMutPtr)
        let write_fn = compiler.fun1("write_ptr", |_ctx, ptr: Var<SMutPtr<I64Type>>| {
            (store(ptr, Const::<I64Type>::new(99)), load_mut(ptr))
        });

        let compiled = compiler.compile(write_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let mut value = 0i64;
        let result = f(&mut value);

        assert_eq!(value, 99);
        assert_eq!(result, 99);
    }

    #[test]
    fn test_ptr_offset() {
        let mut compiler = Compiler::new();

        let test_fn = compiler.fun1("test_offset", |_ctx, ptr: Var<SPtr<I64Type>>| {
            load(ptr_offset(ptr, Const::<I64Type>::new(2)))
        });

        let compiled = compiler.compile(test_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let array = [10i64, 20, 30, 40, 50];
        let result = f(array.as_ptr());

        assert_eq!(result, 30);
    }

    #[test]
    fn test_array_index() {
        let mut compiler = Compiler::new();

        let get_third = compiler.fun1("get_third", |_ctx, ptr: Var<SPtr<I64Type>>| {
            array_index(ptr, Const::<I64Type>::new(3))
        });

        let compiled = compiler.compile(get_third).expect("compilation failed");
        let f = compiled.as_fn();

        let array = [100i64, 200, 300, 400, 500];
        let result = f(array.as_ptr());

        assert_eq!(result, 400);
    }
}
