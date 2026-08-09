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
use crate::types::{CopyType, StagedType};
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
#[derive(Debug)]
pub struct SRef<'a, T, Tag = RustRef> {
    _phantom: PhantomData<&'a (T, Tag)>,
}

// A reference handle is just a phantom, so it is always Copy regardless of `T`
// (a `T: Copy` bound from `#[derive]` would leak into every holder).
impl<'a, T, Tag> Clone for SRef<'a, T, Tag> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T, Tag> Copy for SRef<'a, T, Tag> {}

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
#[derive(Debug)]
pub struct SRefMut<'a, T, Tag = RustRef> {
    _phantom: PhantomData<&'a mut (T, Tag)>,
}

// As with `SRef`, a mutable-reference handle is always Copy regardless of `T`.
impl<'a, T, Tag> Clone for SRefMut<'a, T, Tag> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T, Tag> Copy for SRefMut<'a, T, Tag> {}

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

// Raw pointers are register-sized, `Copy` values — so they are `CopyType` and can
// be loaded from a field / bound to a `Var` (e.g. a `SVec`'s buffer pointer read
// out of its control block). Only the raw-pointer (`RustPtr`) flavors: a `&T`
// (`RustRef`) stays off the copy path.
impl<'a, T: StagedType> CopyType for SRef<'a, T, RustPtr> {}
impl<'a, T: StagedType> CopyType for SRefMut<'a, T, RustPtr> {}

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
    T: StagedType + 'a,
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
    T: StagedType + 'a,
    Tag: 'a,
{
    LoadRef {
        ptr,
        _marker: PhantomData,
    }
}

/// Alias for `load_ref` for raw pointer semantics
pub fn load<P, T>(ptr: P) -> LoadRef<'static, P>
where
    P: Staged<Out = SPtr<T>>,
    T: StagedType + 'static,
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
    T: StagedType + 'a,
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
    T: StagedType + 'a,
    Tag: 'a,
{
    LoadMutRef {
        ptr,
        _marker: PhantomData,
    }
}

/// Alias for `load_ref_mut` for raw pointer semantics
pub fn load_mut<P, T>(ptr: P) -> LoadMutRef<'static, P>
where
    P: Staged<Out = SMutPtr<T>>,
    T: StagedType + 'static,
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
    T: StagedType + 'a,
    Tag: 'a,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let ptr_val = self.ptr.codegen(ctx);
        let value = self.val.codegen(ctx);

        ctx.builder
            .ins()
            .store(MemFlags::trusted(), value, ptr_val, 0);

        ctx.get_unit_value()
    }
}

/// Create a store operation (for references)
pub fn store_ref<'a, P, V, T, Tag>(ptr: P, val: V) -> Store<'a, P, V>
where
    P: Staged<Out = SRefMut<'a, T, Tag>>,
    V: Staged<Out = T>,
    T: StagedType + 'a,
    Tag: 'a,
{
    Store {
        ptr,
        val,
        _marker: PhantomData,
    }
}

/// Create a store operation (for raw pointers)
pub fn store<P, V, T>(ptr: P, val: V) -> Store<'static, P, V>
where
    P: Staged<Out = SMutPtr<T>>,
    V: Staged<Out = T>,
    T: StagedType + 'static,
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

impl<'a, P, I, T, Tag: 'a> Staged for PtrOffset<'a, P, I>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    I: Staged<Out = i64>,
    T: StagedType + 'a,
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
    I: Staged<Out = i64>,
    T: StagedType + 'a,
    Tag: 'a,
{
    PtrOffset {
        ptr,
        index,
        _marker: PhantomData,
    }
}

/// Pointer offset for mutable reference/pointer
#[derive(Clone)]
pub struct PtrOffsetMut<'a, P, I> {
    ptr: P,
    index: I,
    _marker: PhantomData<&'a mut ()>,
}

impl<'a, P, I, T, Tag: 'a> Staged for PtrOffsetMut<'a, P, I>
where
    P: Staged<Out = SRefMut<'a, T, Tag>>,
    I: Staged<Out = i64>,
    T: StagedType + 'a,
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
    I: Staged<Out = i64>,
    T: StagedType + 'a,
    Tag: 'a,
{
    PtrOffsetMut {
        ptr,
        index,
        _marker: PhantomData,
    }
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

impl<'a, P, I, T, Tag: 'a> Staged for ArrayIndex<'a, P, I>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    I: Staged<Out = i64>,
    T: StagedType + 'a,
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
pub fn array_index<'a, P, I, T, Tag: 'a>(ptr: P, index: I) -> ArrayIndex<'a, P, I>
where
    P: Staged<Out = SRef<'a, T, Tag>>,
    I: Staged<Out = i64>,
    T: StagedType + 'a,
{
    ArrayIndex {
        ptr,
        index,
        _marker: PhantomData,
    }
}

// =============================================================================
// ConstPtr: bake a typed host pointer as a staged pointer/reference
// =============================================================================

/// A staged pointer/reference whose value is a **baked host address** — a
/// compile-time constant known at stage 0. The pointer's Rust type (`*const T` /
/// `*mut T`) is checked when you construct it; only the address reaches Cranelift.
///
/// Use it to hand a kernel a host buffer that outlives the run (a GROUP BY's
/// accumulator arrays, the string pool). This is how a compiler author keeps
/// pointers *typed* instead of smuggling `u64`s around — see [`const_ptr`],
/// [`const_mut_ptr`], and (for opaque host structs) `opaque` constructors.
pub struct ConstPtr<S> {
    addr: usize,
    _s: PhantomData<S>,
}

impl<S> Clone for ConstPtr<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S> Copy for ConstPtr<S> {}

impl<S: StagedType> Staged for ConstPtr<S> {
    type Out = S;
    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        ctx.builder.ins().iconst(types::I64, self.addr as i64)
    }
}

impl<S> ConstPtr<S> {
    /// Bake `addr` (a host pointer's address) as this staged pointer type. Prefer
    /// the typed [`const_ptr`]/[`const_mut_ptr`] constructors, which take a real
    /// `*const T`/`*mut T` and so check the pointee type.
    pub fn from_addr(addr: usize) -> Self {
        ConstPtr {
            addr,
            _s: PhantomData,
        }
    }
}

/// Bake a host `*const T::RuntimeValue` as a staged `SPtr<T>` (`*const T`).
pub fn const_ptr<T: StagedType>(p: *const T::RuntimeValue) -> ConstPtr<SPtr<T>> {
    ConstPtr::from_addr(p as usize)
}

/// Bake a host `*mut T::RuntimeValue` as a staged `SMutPtr<T>` (`*mut T`).
pub fn const_mut_ptr<T: StagedType>(p: *mut T::RuntimeValue) -> ConstPtr<SMutPtr<T>> {
    ConstPtr::from_addr(p as usize)
}

/// Reinterpret a staged pointer as pointing to a different element type — same
/// address, emits no code. The typed-pointer analogue of [`opaque_ref`](crate::opaque::opaque_ref):
/// a pointer *is* its address, so the pointee type is the staged author's contract.
/// Use it to turn a raw byte buffer (`SMutPtr<u8>` loaded from a control block) into
/// a typed `SMutPtr<T>` for element-strided indexing. Prefer the [`ptr_cast`] /
/// [`ptr_cast_mut`] constructors, which fix the input to a real pointer type.
pub struct PtrCast<P, S> {
    ptr: P,
    _s: PhantomData<S>,
}

impl<P: Clone, S> Clone for PtrCast<P, S> {
    fn clone(&self) -> Self {
        PtrCast {
            ptr: self.ptr.clone(),
            _s: PhantomData,
        }
    }
}
impl<P: Copy, S> Copy for PtrCast<P, S> {}

impl<P: Staged, S: StagedType> Staged for PtrCast<P, S> {
    type Out = S;
    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        // A cast is a no-op on the address value; only the static type changes.
        self.ptr.codegen(ctx)
    }
}

/// Reinterpret `*const U` as `*const T` at the same address (no code emitted).
pub fn ptr_cast<T, U, P>(ptr: P) -> PtrCast<P, SPtr<T>>
where
    T: StagedType + 'static,
    U: StagedType + 'static,
    P: Staged<Out = SPtr<U>>,
{
    PtrCast {
        ptr,
        _s: PhantomData,
    }
}

/// Reinterpret `*mut U` as `*mut T` at the same address (no code emitted).
pub fn ptr_cast_mut<T, U, P>(ptr: P) -> PtrCast<P, SMutPtr<T>>
where
    T: StagedType + 'static,
    U: StagedType + 'static,
    P: Staged<Out = SMutPtr<U>>,
{
    PtrCast {
        ptr,
        _s: PhantomData,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_load_store_i64() {
        let mut compiler = Compiler::new();

        let write_fn = compiler.fun1("write_42", |_ctx, ptr: Var<SRefMut<i64>>| {
            (store_ref(ptr, Const::<i64>::new(42)), load_ref_mut(ptr))
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

        let read_fn = compiler.fun1("read", |_ctx, ptr: Var<SRef<f64>>| load_ref(ptr));

        let compiled = compiler.compile(read_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let value = 3.15159;
        let result = f(&value);

        assert_eq!(result, 3.15159);
    }

    #[test]
    fn test_raw_ptr_load_store() {
        let mut compiler = Compiler::new();

        // Using raw pointer types (SPtr/SMutPtr)
        let write_fn = compiler.fun1("write_ptr", |_ctx, ptr: Var<SMutPtr<i64>>| {
            (store(ptr, Const::<i64>::new(99)), load_mut(ptr))
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

        let test_fn = compiler.fun1("test_offset", |_ctx, ptr: Var<SPtr<i64>>| {
            load(ptr_offset(ptr, Const::<i64>::new(2)))
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

        let get_third = compiler.fun1("get_third", |_ctx, ptr: Var<SPtr<i64>>| {
            array_index(ptr, Const::<i64>::new(3))
        });

        let compiled = compiler.compile(get_third).expect("compilation failed");
        let f = compiled.as_fn();

        let array = [100i64, 200, 300, 400, 500];
        let result = f(array.as_ptr());

        assert_eq!(result, 400);
    }
}
