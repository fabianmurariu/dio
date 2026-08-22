//! Reference and pointer types for staged computations.
//!
//! This module provides:
//! - `SRef<T>`: Immutable reference type (`&T` at runtime)
//! - `SRefMut<T>`: Mutable reference type (`&mut T` at runtime)
//! - `SPtr<T>`: Immutable raw pointer type (`*const T` at runtime)
//! - `SMutPtr<T>`: Mutable raw pointer type (`*mut T` at runtime)
//!
//! References and raw pointers are distinct staged types even though they have
//! the same runtime representation. References carry Rust validity, lifetime,
//! and aliasing guarantees; raw pointers do not.

use crate::staged::{ValueId, CompilationContext, Staged, Var, VarUse};
use crate::types::{ScalarType, CopyType, RuntimeParam, RuntimeResult, StagedType};
use cranelift_codegen::ir::types;
use std::marker::PhantomData;

// =============================================================================
// SRef<T> - Immutable reference type
// =============================================================================

/// Immutable reference type (`&T` at runtime).
///
/// Note: The `T: StagedType` bound is only required on the `StagedType` impl,
/// not on the struct itself. This allows `SRef<Slice<T>>` to work even though
/// `Slice<T>` doesn't implement `StagedType` (since it's a DST marker).
#[derive(Debug)]
pub struct SRef<'a, T> {
    _phantom: PhantomData<&'a T>,
}

// A reference handle is just a phantom, so it is always Copy regardless of `T`
// (a `T: Copy` bound from `#[derive]` would leak into every holder).
impl<'a, T> Clone for SRef<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Copy for SRef<'a, T> {}

unsafe impl<'a, T: StagedType> StagedType for SRef<'a, T> {
    type RuntimeValue = &'a T::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

unsafe impl<'a, T: StagedType> CopyType for SRef<'a, T> {}

unsafe impl<'stage, T> RuntimeParam for SRef<'stage, T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = &'call T::RuntimeValue;
}

unsafe impl<'stage, T> RuntimeResult for SRef<'stage, T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Output<'call> = &'call T::RuntimeValue;
}

// =============================================================================
// SRefMut<T> - Mutable reference type
// =============================================================================

/// Mutable reference type (`&mut T` at runtime).
#[derive(Debug)]
pub struct SRefMut<'a, T> {
    _phantom: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: StagedType> StagedType for SRefMut<'a, T> {
    type RuntimeValue = &'a mut T::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer-sized
    }
}

unsafe impl<'stage, T> RuntimeParam for SRefMut<'stage, T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = &'call mut T::RuntimeValue;
}

unsafe impl<'stage, T> RuntimeResult for SRefMut<'stage, T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Output<'call> = &'call mut T::RuntimeValue;
}

// =============================================================================
// Raw pointer types
// =============================================================================

/// Immutable raw pointer type (`*const T` at runtime).
#[derive(Debug)]
pub struct SPtr<T> {
    _phantom: PhantomData<*const T>,
}

impl<T> Clone for SPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SPtr<T> {}

unsafe impl<T: StagedType> StagedType for SPtr<T> {
    type RuntimeValue = *const T::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }
}

/// Mutable raw pointer type (`*mut T` at runtime).
#[derive(Debug)]
pub struct SMutPtr<T> {
    _phantom: PhantomData<*mut T>,
}

impl<T> Clone for SMutPtr<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for SMutPtr<T> {}

unsafe impl<T: StagedType> StagedType for SMutPtr<T> {
    type RuntimeValue = *mut T::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }
}

unsafe impl<T: StagedType> RuntimeParam for SPtr<T> {
    type Arg<'call> = *const T::RuntimeValue;
}

unsafe impl<T: StagedType> RuntimeResult for SPtr<T> {
    type Output<'call> = *const T::RuntimeValue;
}

unsafe impl<T: StagedType> RuntimeParam for SMutPtr<T> {
    type Arg<'call> = *mut T::RuntimeValue;
}

unsafe impl<T: StagedType> RuntimeResult for SMutPtr<T> {
    type Output<'call> = *mut T::RuntimeValue;
}

unsafe impl<T: StagedType> CopyType for SPtr<T> {}
unsafe impl<T: StagedType> CopyType for SMutPtr<T> {}

// =============================================================================
// Load: Dereference to read a value
// =============================================================================

/// Load value from immutable reference/pointer: `*ptr`
pub struct LoadRef<'a, P> {
    ptr: P,
    _marker: PhantomData<&'a ()>,
}

unsafe impl<'a, P, T> Staged for LoadRef<'a, P>
where
    P: Staged<Out = SRef<'a, T>>,
    T: StagedType + 'a,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr_val = self.ptr.codegen(ctx);
        ctx.load(T::scalar_type(), ptr_val, 0)
    }
}

/// Create a load operation from an immutable reference.
///
/// Raw pointers cannot use this safe reference operation:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn read_raw(ptr: Var<SPtr<i64>>) {
///     let _ = load_ref(ptr);
/// }
/// ```
pub fn load_ref<'a, P, T>(ptr: P) -> LoadRef<'a, P>
where
    P: Staged<Out = SRef<'a, T>>,
    T: StagedType + 'a,
{
    LoadRef {
        ptr,
        _marker: PhantomData,
    }
}

/// Load a value through an immutable raw pointer.
pub struct LoadPtr<P> {
    ptr: P,
}

unsafe impl<P, T> Staged for LoadPtr<P>
where
    P: Staged<Out = SPtr<T>>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr_val = self.ptr.codegen(ctx);
        ctx.load(T::scalar_type(), ptr_val, 0)
    }
}

/// Create a load operation for an immutable raw pointer.
///
/// # Safety
///
/// At execution, the pointer must be aligned, dereferenceable, and point to an
/// initialized valid `T::RuntimeValue`.
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn read_raw(ptr: Var<SPtr<i64>>) {
///     let _ = load(ptr);
/// }
/// ```
pub unsafe fn load<P, T>(ptr: P) -> LoadPtr<P>
where
    P: Staged<Out = SPtr<T>>,
    T: StagedType,
{
    LoadPtr { ptr }
}

/// Load from mutable reference/pointer
pub struct LoadMutRef<'a, P> {
    ptr: P,
    _marker: PhantomData<&'a mut ()>,
}

unsafe impl<'a, P, T> Staged for LoadMutRef<'a, P>
where
    P: Staged<Out = SRefMut<'a, T>>,
    T: StagedType + 'a,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr_val = self.ptr.codegen(ctx);
        ctx.load(T::scalar_type(), ptr_val, 0)
    }
}

/// Convert a unique staged reference into one owned occurrence for an
/// operation.
///
/// A mutable-reference expression is consumed. A mutable-reference variable is
/// instead reborrowed into a crate-controlled [`VarUse`], allowing a later
/// sequential operation to reborrow the same root again.
pub trait IntoMutRef<'a, T: StagedType + 'a> {
    type Staged: Staged<Out = SRefMut<'a, T>>;

    fn into_mut_ref(self) -> Self::Staged;
}

impl<'a, P, T> IntoMutRef<'a, T> for P
where
    P: Staged<Out = SRefMut<'a, T>>,
    T: StagedType + 'a,
{
    type Staged = P;

    fn into_mut_ref(self) -> Self::Staged {
        self
    }
}

impl<'a, T> IntoMutRef<'a, T> for &mut Var<SRefMut<'a, T>>
where
    T: StagedType + 'a,
{
    type Staged = VarUse<SRefMut<'a, T>>;

    fn into_mut_ref(self) -> Self::Staged {
        self.use_once()
    }
}

/// Create a load operation from a unique mutable reference.
pub fn load_ref_mut<'a, P, T>(ptr: P) -> LoadMutRef<'a, P::Staged>
where
    P: IntoMutRef<'a, T>,
    T: StagedType + 'a,
{
    LoadMutRef {
        ptr: ptr.into_mut_ref(),
        _marker: PhantomData,
    }
}

/// Load a value through a mutable raw pointer.
pub struct LoadMutPtr<P> {
    ptr: P,
}

unsafe impl<P, T> Staged for LoadMutPtr<P>
where
    P: Staged<Out = SMutPtr<T>>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr_val = self.ptr.codegen(ctx);
        ctx.load(T::scalar_type(), ptr_val, 0)
    }
}

/// Create a load operation for a mutable raw pointer.
///
/// # Safety
///
/// At execution, the pointer must be aligned, dereferenceable, and point to an
/// initialized valid `T::RuntimeValue`.
pub unsafe fn load_mut<P, T>(ptr: P) -> LoadMutPtr<P>
where
    P: Staged<Out = SMutPtr<T>>,
    T: StagedType,
{
    LoadMutPtr { ptr }
}

// =============================================================================
// Store: Write a value
// =============================================================================

/// Store value to mutable reference/pointer: `*ptr = val`
pub struct StoreRef<'a, P, V> {
    ptr: P,
    val: V,
    _marker: PhantomData<&'a mut ()>,
}

unsafe impl<'a, P, V, T> Staged for StoreRef<'a, P, V>
where
    P: Staged<Out = SRefMut<'a, T>>,
    V: Staged<Out = T>,
    T: StagedType + 'a,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr_val = self.ptr.codegen(ctx);
        let value = self.val.codegen(ctx);

        ctx.store(value, ptr_val, 0);

        ctx.get_unit_value()
    }
}

/// Create a store operation through a unique mutable reference.
pub fn store_ref<'a, P, V, T>(ptr: P, val: V) -> StoreRef<'a, P::Staged, V>
where
    P: IntoMutRef<'a, T>,
    V: Staged<Out = T>,
    T: StagedType + 'a,
{
    StoreRef {
        ptr: ptr.into_mut_ref(),
        val,
        _marker: PhantomData,
    }
}

/// Store a value through a mutable raw pointer.
pub struct Store<P, V> {
    ptr: P,
    val: V,
}

unsafe impl<P, V, T> Staged for Store<P, V>
where
    P: Staged<Out = SMutPtr<T>>,
    V: Staged<Out = T>,
    T: StagedType,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr_val = self.ptr.codegen(ctx);
        let value = self.val.codegen(ctx);
        ctx.store(value, ptr_val, 0);
        ctx.get_unit_value()
    }
}

/// Create a store operation for a mutable raw pointer.
///
/// # Safety
///
/// At execution, the pointer must be aligned and valid for writing a
/// `T::RuntimeValue`, with no conflicting live references.
pub unsafe fn store<P, V, T>(ptr: P, val: V) -> Store<P, V>
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

/// Pointer offset for immutable reference/pointer
#[derive(Clone)]
pub struct PtrOffset<P, I> {
    ptr: P,
    index: I,
}

unsafe impl<P, I, T> Staged for PtrOffset<P, I>
where
    P: Staged<Out = SPtr<T>>,
    I: Staged<Out = i64>,
    T: StagedType,
{
    type Out = SPtr<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        let element_size = std::mem::size_of::<T::RuntimeValue>() as i64;
        let scale = ctx.iconst(ScalarType::I64, element_size);
        let byte_offset = ctx.imul(idx, scale);

        ctx.ptr_offset_bytes(ptr, byte_offset)
    }
}

/// Create an in-allocation element offset for an immutable raw pointer.
///
/// # Safety
///
/// At execution, both the base and resulting pointer must remain within, or one
/// byte past, the same allocated object as required by pointer offset semantics.
pub unsafe fn ptr_offset<P, I, T>(ptr: P, index: I) -> PtrOffset<P, I>
where
    P: Staged<Out = SPtr<T>>,
    I: Staged<Out = i64>,
    T: StagedType,
{
    PtrOffset { ptr, index }
}

/// Pointer offset for mutable reference/pointer
#[derive(Clone)]
pub struct PtrOffsetMut<P, I> {
    ptr: P,
    index: I,
}

unsafe impl<P, I, T> Staged for PtrOffsetMut<P, I>
where
    P: Staged<Out = SMutPtr<T>>,
    I: Staged<Out = i64>,
    T: StagedType,
{
    type Out = SMutPtr<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        let element_size = std::mem::size_of::<T::RuntimeValue>() as i64;
        let scale = ctx.iconst(ScalarType::I64, element_size);
        let byte_offset = ctx.imul(idx, scale);

        ctx.ptr_offset_bytes(ptr, byte_offset)
    }
}

/// Create an in-allocation element offset for a mutable raw pointer.
///
/// # Safety
///
/// At execution, both the base and resulting pointer must remain within, or one
/// byte past, the same allocated object as required by pointer offset semantics.
pub unsafe fn ptr_offset_mut<P, I, T>(ptr: P, index: I) -> PtrOffsetMut<P, I>
where
    P: Staged<Out = SMutPtr<T>>,
    I: Staged<Out = i64>,
    T: StagedType,
{
    PtrOffsetMut { ptr, index }
}

// =============================================================================
// ArrayIndex: Combined pointer offset + load
// =============================================================================

/// Array indexing: `ptr[index]`
pub struct ArrayIndex<P, I> {
    ptr: P,
    index: I,
}

unsafe impl<P, I, T> Staged for ArrayIndex<P, I>
where
    P: Staged<Out = SPtr<T>>,
    I: Staged<Out = i64>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.ptr.codegen(ctx);
        let idx = self.index.codegen(ctx);

        let element_size = std::mem::size_of::<T::RuntimeValue>() as i64;
        let scale = ctx.iconst(ScalarType::I64, element_size);
        let byte_offset = ctx.imul(idx, scale);

        let offset_ptr = ctx.ptr_offset_bytes(ptr, byte_offset);

        ctx.load(T::scalar_type(), offset_ptr, 0)
    }
}

/// Create an unchecked raw-pointer indexing operation.
///
/// # Safety
///
/// At execution, `index` must identify an aligned, initialized `T` within the
/// allocation addressed by `ptr`.
pub unsafe fn array_index<P, I, T>(ptr: P, index: I) -> ArrayIndex<P, I>
where
    P: Staged<Out = SPtr<T>>,
    I: Staged<Out = i64>,
    T: StagedType,
{
    ArrayIndex { ptr, index }
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

unsafe impl<S: StagedType> Staged for ConstPtr<S> {
    type Out = S;
    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        // A baked host address becomes a staged pointer (Cranelift: the i64 itself;
        // MLIR: inttoptr).
        let addr = ctx.iconst(ScalarType::I64, self.addr as i64);
        ctx.addr_to_ptr(addr)
    }
}

impl<S> ConstPtr<S> {
    /// Bake an exposed host address as an arbitrary staged pointer/reference type.
    ///
    /// Prefer [`const_ptr`] or [`const_mut_ptr`], which preserve the host pointer's
    /// pointee type and do not manufacture a Rust reference.
    ///
    /// # Safety
    ///
    /// If `S::RuntimeValue` is a Rust reference, the address must satisfy that
    /// reference's validity, lifetime, alignment, and aliasing requirements for
    /// every generated-code use. For raw-pointer `S`, dereferencing remains a
    /// separate unsafe operation.
    pub unsafe fn from_addr_unchecked(addr: usize) -> Self {
        ConstPtr {
            addr,
            _s: PhantomData,
        }
    }
}

/// Bake a host `*const T::RuntimeValue` as a staged `SPtr<T>` (`*const T`).
pub fn const_ptr<T: StagedType>(p: *const T::RuntimeValue) -> ConstPtr<SPtr<T>> {
    // SAFETY: the staged result is a raw pointer with the same pointee type;
    // constructing it does not claim that the pointer is dereferenceable.
    unsafe { ConstPtr::from_addr_unchecked(p as usize) }
}

/// Bake a host `*mut T::RuntimeValue` as a staged `SMutPtr<T>` (`*mut T`).
pub fn const_mut_ptr<T: StagedType>(p: *mut T::RuntimeValue) -> ConstPtr<SMutPtr<T>> {
    // SAFETY: the staged result is a raw pointer with the same pointee type;
    // constructing it does not claim that the pointer is dereferenceable.
    unsafe { ConstPtr::from_addr_unchecked(p as usize) }
}

/// Reinterpret a staged pointer as pointing to a different element type — same
/// address, emits no code. Use this instead of encoding an address as an integer:
/// a pointer *is* its address, so the pointee type is the staged author's contract.
/// Use it to turn a raw byte buffer (`SMutPtr<u8>` loaded from a control block) into
/// a typed `SMutPtr<T>` for element-strided indexing. Prefer the [`ptr_cast`] /
/// [`ptr_cast_mut`] constructors, which fix the input to a real pointer type.
///
/// # Why this is safe despite reinterpreting the pointee
///
/// The cast itself cannot cause undefined behavior: it produces no instructions
/// (the address value is unchanged) and only rewrites the *stage-0* type. The
/// "the new pointee type is correct" claim is therefore an obligation that is
/// **redeemed at the point of use, not here** — every way to actually touch the
/// pointee (`load`, `store`, `ptr_offset`, `array_index`) is an `unsafe` staging
/// operation whose contract already requires a valid, correctly-typed, aligned
/// address. So a wrong `ptr_cast` is not itself unsound; it only becomes unsound
/// through a later `unsafe` deref, which is where the audit belongs. Read a safe
/// `ptr_cast` as "relabel this address", never as a proof that the bytes match `T`.
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

unsafe impl<P: Staged, S: StagedType> Staged for PtrCast<P, S> {
    type Out = S;
    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
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

/// Demote `*mut T` to `*const T` at the same address (no code emitted) — e.g. to
/// hand a runtime scratch buffer to an extern taking `*const T`.
pub fn ptr_as_const<T, P>(ptr: P) -> PtrCast<P, SPtr<T>>
where
    T: StagedType + 'static,
    P: Staged<Out = SMutPtr<T>>,
{
    PtrCast {
        ptr,
        _s: PhantomData,
    }
}

/// Expose a staged shared reference as a raw pointer at the same address.
///
/// The result retains no reference lifetime claim. Dereferencing it or passing
/// it to an extern that expects `&T` still requires an unsafe operation.
pub fn ref_as_ptr<'a, T, P>(reference: P) -> PtrCast<P, SPtr<T>>
where
    T: StagedType + 'a,
    P: Staged<Out = SRef<'a, T>>,
{
    PtrCast {
        ptr: reference,
        _s: PhantomData,
    }
}

/// Expose a staged mutable reference as a mutable raw pointer at the same
/// address.
///
/// The result retains no reference lifetime or exclusivity claim.
pub fn ref_mut_as_ptr<'a, T, P>(reference: P) -> PtrCast<P, SMutPtr<T>>
where
    T: StagedType + 'a,
    P: Staged<Out = SRefMut<'a, T>>,
{
    PtrCast {
        ptr: reference,
        _s: PhantomData,
    }
}

/// Demote a staged `&mut T` to `&T` at the same address (no code emitted).
///
/// This is the staged equivalent of reborrowing a mutable Rust reference as an
/// immutable reference, primarily for passing an `SRefMut` to an external
/// function whose typed signature expects `SRef`.
pub fn ref_as_const<'a, T, P>(reference: P) -> PtrCast<P, SRef<'a, T>>
where
    T: StagedType + 'a,
    P: Staged<Out = SRefMut<'a, T>>,
{
    PtrCast {
        ptr: reference,
        _s: PhantomData,
    }
}

/// True iff the pointer is null (`ptr == 0`) — the staged twin of `p.is_null()`.
/// The sentinel test for an extern returning a nullable `*const T` / `*mut T`
/// (e.g. a scan stream's "no more batches"). Emits a single `icmp eq ptr, 0`.
pub struct PtrIsNull<P> {
    ptr: P,
}

mod raw_pointer_sealed {
    pub trait Sealed {}
}

/// A staged raw-pointer marker accepted by address-only operations.
///
/// This trait is sealed so integer-like staged values cannot opt into pointer
/// operations merely because they share the same Cranelift representation.
pub trait RawPointer: StagedType + raw_pointer_sealed::Sealed {}

impl<T: StagedType> raw_pointer_sealed::Sealed for SPtr<T> {}
impl<T: StagedType> RawPointer for SPtr<T> {}
impl<T: StagedType> raw_pointer_sealed::Sealed for SMutPtr<T> {}
impl<T: StagedType> RawPointer for SMutPtr<T> {}

impl<P: Clone> Clone for PtrIsNull<P> {
    fn clone(&self) -> Self {
        PtrIsNull {
            ptr: self.ptr.clone(),
        }
    }
}
impl<P: Copy> Copy for PtrIsNull<P> {}

unsafe impl<P> Staged for PtrIsNull<P>
where
    P: Staged,
    P::Out: RawPointer,
{
    type Out = bool;
    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let p = self.ptr.codegen(ctx);
        ctx.icmp_imm(cranelift_codegen::ir::condcodes::IntCC::Equal, p, 0)
    }
}

/// Build a staged null-check over any pointer (`SPtr<T>` / `SMutPtr<T>`).
pub fn ptr_is_null<P>(ptr: P) -> PtrIsNull<P>
where
    P: Staged,
    P::Out: RawPointer,
{
    PtrIsNull { ptr }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_load_store_i64() {
        let mut compiler = Compiler::new();

        let write_fn = compiler.fun1("write_42", |_ctx, mut ptr: Var<SRefMut<i64>>| {
            (
                store_ref(&mut ptr, Const::<i64>::new(42)),
                load_ref_mut(&mut ptr),
            )
        });

        let compiled = compiler.compile(write_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let mut value = 0i64;
        let result = f.call(&mut value);

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
        let result = f.call(&value);

        assert_eq!(result, 3.15159);
    }

    #[test]
    fn test_raw_ptr_load_store() {
        let mut compiler = Compiler::new();

        // Using raw pointer types (SPtr/SMutPtr)
        let write_fn = compiler.fun1("write_ptr", |_ctx, ptr: Var<SMutPtr<i64>>| {
            // SAFETY: the generated function is called with a valid, aligned
            // pointer to the live `value` below.
            unsafe { (store(ptr, Const::<i64>::new(99)), load_mut(ptr)) }
        });

        let compiled = compiler.compile(write_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let mut value = 0i64;
        let result = f.call(&mut value);

        assert_eq!(value, 99);
        assert_eq!(result, 99);
    }

    #[test]
    fn test_ptr_offset() {
        let mut compiler = Compiler::new();

        let test_fn = compiler.fun1("test_offset", |_ctx, ptr: Var<SPtr<i64>>| {
            // SAFETY: the generated function is called with the five-element
            // `array` below, and index 2 is initialized and in bounds.
            unsafe { load(ptr_offset(ptr, Const::<i64>::new(2))) }
        });

        let compiled = compiler.compile(test_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let array = [10i64, 20, 30, 40, 50];
        let result = f.call(array.as_ptr());

        assert_eq!(result, 30);
    }

    #[test]
    fn test_array_index() {
        let mut compiler = Compiler::new();

        let get_third = compiler.fun1("get_third", |_ctx, ptr: Var<SPtr<i64>>| {
            // SAFETY: the generated function is called with the five-element
            // `array` below, and index 3 is initialized and in bounds.
            unsafe { array_index(ptr, Const::<i64>::new(3)) }
        });

        let compiled = compiler.compile(get_third).expect("compilation failed");
        let f = compiled.as_fn();

        let array = [100i64, 200, 300, 400, 500];
        let result = f.call(array.as_ptr());

        assert_eq!(result, 400);
    }
}
