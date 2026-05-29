//! Struct support for staged computations.
//!
//! This module provides:
//! - `Field` trait for describing struct fields
//! - Field accessor operations (LoadField, FieldRef, FieldPath, FieldPtr, etc.)
//! - Traits for field access with proper lifetime tracking:
//!   - `CopyFieldAccess` - load Copy fields (all types)
//!   - `RefFieldAccess` - get immutable references (pointer types only)
//!   - `MutRefFieldAccess` - get mutable references (mutable pointer types only)
//!   - `OwnedFieldAccess` - access owned struct fields (no lifetime)

use crate::refer::{SMutPtr, SPtr, SRef, SRefMut};
use crate::staged::{CompilationContext, Staged, Var};
use crate::types::{CopyType, Owned, StagedType};
use cranelift_codegen::ir::{types, InstBuilder, MemFlags};
use std::marker::PhantomData;

// =============================================================================
// Field Trait
// =============================================================================

/// Trait for struct field descriptors.
///
/// This trait is implemented by the proc macro for each field of a struct.
/// It provides type-level information about the field's parent struct,
/// output type, and memory offset.
pub trait Field: Copy {
    /// The parent struct type
    type Parent: StagedType;

    /// The field's type
    type Out: StagedType;

    /// Byte offset of this field from the struct base
    const OFFSET: usize;

    /// Field index (for debugging)
    const INDEX: usize;
}

// =============================================================================
// LoadField: Load a Copy field's value
// =============================================================================

/// Load a field's value from a struct.
///
/// This operation is only valid for Copy fields. It performs:
/// 1. Compute field pointer (base + offset)
/// 2. Load value from memory
///
/// # Example
/// ```ignore
/// let pt: VarRef<Point> = ...;
/// let x = pt.get(PointType::x);  // Uses LoadField internally
/// ```
pub struct LoadField<P, F> {
    ptr: P,
    _field: PhantomData<F>,
}

impl<P, F> Staged for LoadField<P, F>
where
    P: Staged,
    P::Out: StagedType,
    F: Field,
    F::Out: CopyType,
{
    type Out = F::Out;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        // Generate code for the base pointer
        let base_ptr = self.ptr.codegen(ctx);

        // Load from base_ptr + offset
        let offset = F::OFFSET as i32;
        ctx.builder.ins().load(
            F::Out::cranelift_type(),
            MemFlags::trusted(),
            base_ptr,
            offset,
        )
    }
}

/// Helper to create a LoadField operation
pub fn load_field<P, F>(ptr: P, _field: F) -> LoadField<P, F>
where
    P: Staged,
    F: Field,
    F::Out: CopyType,
{
    LoadField {
        ptr,
        _field: PhantomData,
    }
}

// =============================================================================
// FieldRef: Get an immutable reference to a field (for pointer types)
// =============================================================================

/// Get an immutable reference to a field.
///
/// This computes the field's address but doesn't load its value.
/// The lifetime `'a` flows through from the parent pointer type.
///
/// Used for pointer-based access where lifetime must be preserved:
/// - `Var<SRef<'a, T>>` -> `FieldRef<'a, ..>` -> can be returned
/// - Chained: `FieldRef<'a, ..>` -> `FieldRef<'a, ..>` -> lifetime preserved
pub struct FieldRef<'a, P, F> {
    ptr: P, // Stored by value, not &'a P
    _lifetime: PhantomData<&'a ()>,
    _field: PhantomData<F>,
}

impl<'a, P, F> Clone for FieldRef<'a, P, F>
where
    P: Clone,
{
    fn clone(&self) -> Self {
        FieldRef {
            ptr: self.ptr.clone(),
            _lifetime: PhantomData,
            _field: PhantomData,
        }
    }
}

impl<'a, P, F> Copy for FieldRef<'a, P, F> where P: Copy {}

impl<'a, P, F, ParentT> Staged for FieldRef<'a, P, F>
where
    P: Staged<Out = SRef<'a, ParentT>>,
    ParentT: StagedType + 'a,
    F: Field<Parent = ParentT>,
    F::Out: StagedType + 'a,
{
    type Out = SRef<'a, F::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let base_ptr = self.ptr.codegen(ctx);

        // Compute field pointer: base + offset
        let offset_const = ctx.builder.ins().iconst(types::I64, F::OFFSET as i64);
        ctx.builder.ins().iadd(base_ptr, offset_const)
    }
}

/// Helper to create a FieldRef operation
pub fn field_ref<'a, P, F>(ptr: P, _field: F) -> FieldRef<'a, P, F>
where
    P: Staged,
    F: Field,
{
    FieldRef {
        ptr,
        _lifetime: PhantomData,
        _field: PhantomData,
    }
}

// =============================================================================
// FieldMutRef: Get a mutable reference to a field (for mutable pointer types)
// =============================================================================

/// Get a mutable reference to a field.
///
/// Similar to `FieldRef`, but produces a mutable reference.
/// Only available on mutable pointer types (`Var<SRefMut<'a, T>>`).
pub struct FieldMutRef<'a, P, F> {
    ptr: P,
    _lifetime: PhantomData<&'a mut ()>,
    _field: PhantomData<F>,
}

impl<'a, P, F> Clone for FieldMutRef<'a, P, F>
where
    P: Clone,
{
    fn clone(&self) -> Self {
        FieldMutRef {
            ptr: self.ptr.clone(),
            _lifetime: PhantomData,
            _field: PhantomData,
        }
    }
}

impl<'a, P, F> Copy for FieldMutRef<'a, P, F> where P: Copy {}

impl<'a, P, F, ParentT> Staged for FieldMutRef<'a, P, F>
where
    P: Staged<Out = SRefMut<'a, ParentT>>,
    ParentT: StagedType + 'a,
    F: Field<Parent = ParentT>,
    F::Out: StagedType + 'a,
{
    type Out = SRefMut<'a, F::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let base_ptr = self.ptr.codegen(ctx);

        let offset_const = ctx.builder.ins().iconst(types::I64, F::OFFSET as i64);
        ctx.builder.ins().iadd(base_ptr, offset_const)
    }
}

/// Helper to create a FieldMutRef operation
pub fn field_mut_ref<'a, P, F>(ptr: P, _field: F) -> FieldMutRef<'a, P, F>
where
    P: Staged,
    F: Field,
{
    FieldMutRef {
        ptr,
        _lifetime: PhantomData,
        _field: PhantomData,
    }
}

// =============================================================================
// FieldPtr: Get a raw immutable pointer to a field
// =============================================================================

/// Get a raw immutable pointer to a field.
pub struct FieldPtr<P, F> {
    ptr: P,
    _field: PhantomData<F>,
}

impl<P, F> Staged for FieldPtr<P, F>
where
    P: Staged,
    P::Out: StagedType,
    F: Field,
    <F as Field>::Out: 'static,
{
    type Out = SPtr<F::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let base_ptr = self.ptr.codegen(ctx);

        let offset_const = ctx.builder.ins().iconst(types::I64, F::OFFSET as i64);
        ctx.builder.ins().iadd(base_ptr, offset_const)
    }
}

/// Helper to create a FieldPtr operation
pub fn field_ptr<P, F>(ptr: P, _field: F) -> FieldPtr<P, F>
where
    P: Staged,
    F: Field,
{
    FieldPtr {
        ptr,
        _field: PhantomData,
    }
}

// =============================================================================
// FieldMutPtr: Get a raw mutable pointer to a field
// =============================================================================

/// Get a raw mutable pointer to a field.
pub struct FieldMutPtr<P, F> {
    ptr: P,
    _field: PhantomData<F>,
}

impl<P, F> Staged for FieldMutPtr<P, F>
where
    P: Staged,
    P::Out: StagedType,
    F: Field,
    <F as Field>::Out: 'static,
{
    type Out = SMutPtr<F::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let base_ptr = self.ptr.codegen(ctx);

        let offset_const = ctx.builder.ins().iconst(types::I64, F::OFFSET as i64);
        ctx.builder.ins().iadd(base_ptr, offset_const)
    }
}

/// Helper to create a FieldMutPtr operation
pub fn field_mut_ptr<P, F>(ptr: P, _field: F) -> FieldMutPtr<P, F>
where
    P: Staged,
    F: Field,
{
    FieldMutPtr {
        ptr,
        _field: PhantomData,
    }
}

// =============================================================================
// FieldPath: Lifetime-free access chain for owned structs
// =============================================================================

/// Lifetime-free field path for accessing nested fields in owned structs.
///
/// `FieldPath` is used when accessing fields of structs passed by value
/// (`Var<Owned<T>>`). It has no lifetime parameter, which means:
/// - References cannot be returned (use `get_ptr` for raw pointers)
/// - Can chain to nested structs with `.field()`
/// - Can load Copy values with `.get()`
///
/// # Example
/// ```ignore
/// // Access nested Copy field from owned struct
/// compiler.fun1("sum", |_ctx, outer: Var<Owned<Outer>>| {
///     outer.field(OuterType::inner).get(InnerType::value)
/// });
/// ```
pub struct FieldPath<P, F> {
    ptr: P,
    _field: PhantomData<F>,
}

impl<P, F> Clone for FieldPath<P, F>
where
    P: Clone,
{
    fn clone(&self) -> Self {
        FieldPath {
            ptr: self.ptr.clone(),
            _field: PhantomData,
        }
    }
}

impl<P, F> Copy for FieldPath<P, F> where P: Copy {}

impl<P, F> Staged for FieldPath<P, F>
where
    P: Staged,
    P::Out: StagedType,
    F: Field,
    F::Out: StagedType + 'static,
{
    // FieldPath produces a raw pointer (no lifetime)
    type Out = SPtr<F::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let base_ptr = self.ptr.codegen(ctx);

        // Compute field pointer: base + offset
        let offset_const = ctx.builder.ins().iconst(types::I64, F::OFFSET as i64);
        ctx.builder.ins().iadd(base_ptr, offset_const)
    }
}

/// Helper to create a FieldPath operation
pub fn field_path<P, F>(ptr: P, _field: F) -> FieldPath<P, F>
where
    P: Staged,
    F: Field,
{
    FieldPath {
        ptr,
        _field: PhantomData,
    }
}

// =============================================================================
// Field Access Traits
// =============================================================================

/// Trait for loading Copy field values.
///
/// This is the most basic field access - available on all struct-like types.
/// Loads the field value by copy.
pub trait CopyFieldAccess<T: StagedType>: Sized + Staged {
    /// Load a Copy field's value.
    ///
    /// # Example
    /// ```ignore
    /// let pt: Var<Owned<Point>> = ...;
    /// let x = pt.get(PointType::x);  // Returns impl Staged<Out = i64>
    /// ```
    fn get<F>(self, field: F) -> LoadField<Self, F>
    where
        F: Field<Parent = T>,
        F::Out: CopyType,
    {
        load_field(self, field)
    }
}

/// Trait for getting immutable references to fields (pointer types only).
///
/// Only implemented for types that represent pointers with a lifetime:
/// - `Var<SRef<'a, T>>` - immutable reference input
/// - `Var<SRefMut<'a, T>>` - mutable reference input (can get immutable ref)
/// - `FieldRef<'a, P, F>` - chained immutable reference
/// - `FieldMutRef<'a, P, F>` - chained mutable reference (can get immutable ref)
///
/// NOT implemented for `Var<Owned<T>>` to prevent returning references to stack data.
pub trait RefFieldAccess<'a, T: StagedType>: Sized + Staged {
    /// Get an immutable reference to a field.
    ///
    /// The returned `FieldRef` preserves the lifetime `'a` from the input,
    /// allowing it to be returned from functions.
    fn get_ref<F>(self, field: F) -> FieldRef<'a, Self, F>
    where
        F: Field<Parent = T>,
    {
        field_ref(self, field)
    }
}

/// Trait for getting mutable references to fields (mutable pointer types only).
///
/// Only implemented for types that represent mutable pointers:
/// - `Var<SRefMut<'a, T>>` - mutable reference input
/// - `FieldMutRef<'a, P, F>` - chained mutable reference
///
/// NOT implemented for `Var<SRef<'a, T>>` (can't get mutable from immutable).
pub trait MutRefFieldAccess<'a, T: StagedType>: Sized + Staged {
    /// Get a mutable reference to a field.
    fn get_ref_mut<F>(self, field: F) -> FieldMutRef<'a, Self, F>
    where
        F: Field<Parent = T>,
    {
        field_mut_ref(self, field)
    }
}

/// Trait for accessing fields of owned (by-value) structs.
///
/// Implemented for `Var<Owned<T>>` and `FieldPath<P, F>`.
/// Provides lifetime-free access that cannot return references.
pub trait OwnedFieldAccess<T: StagedType>: Sized + Staged {
    /// Navigate to a nested struct field.
    ///
    /// Returns a `FieldPath` that can be chained further or terminated
    /// with `.get()` for Copy values or `.get_ptr()` for raw pointers.
    fn field<F>(self, field: F) -> FieldPath<Self, F>
    where
        F: Field<Parent = T>,
    {
        field_path(self, field)
    }

    /// Get a raw pointer to a field.
    ///
    /// Returns `SPtr<F::Out>` (no lifetime) - useful for passing to external functions.
    fn get_ptr<F>(self, field: F) -> FieldPtr<Self, F>
    where
        F: Field<Parent = T>,
    {
        field_ptr(self, field)
    }
}

// =============================================================================
// Trait Implementations
// =============================================================================

// -----------------------------------------------------------------------------
// CopyFieldAccess - available on all struct-like types
// -----------------------------------------------------------------------------

// For Var<T> - plain struct (by-value, no wrapper)
// This allows loading Copy fields from any struct
impl<T: StagedType> CopyFieldAccess<T> for Var<T> where Var<T>: Staged {}

// For Var<Owned<T>> - owned structs (explicit marker)
impl<T: StagedType> CopyFieldAccess<T> for Var<Owned<T>> where Var<Owned<T>>: Staged {}

// For Var<SRef<'a, T>> - immutable reference
impl<'a, T: StagedType> CopyFieldAccess<T> for Var<SRef<'a, T>> where Var<SRef<'a, T>>: Staged {}

// For Var<SRefMut<'a, T>> - mutable reference
impl<'a, T: StagedType> CopyFieldAccess<T> for Var<SRefMut<'a, T>> where Var<SRefMut<'a, T>>: Staged {}

// For FieldRef - chained immutable reference
impl<'a, P, F, T, ParentT> CopyFieldAccess<T> for FieldRef<'a, P, F>
where
    P: Staged<Out = SRef<'a, ParentT>>,
    ParentT: StagedType + 'a,
    F: Field<Parent = ParentT, Out = T>,
    T: StagedType + 'a,
{
}

// For FieldMutRef - chained mutable reference
impl<'a, P, F, T, ParentT> CopyFieldAccess<T> for FieldMutRef<'a, P, F>
where
    P: Staged<Out = SRefMut<'a, ParentT>>,
    ParentT: StagedType + 'a,
    F: Field<Parent = ParentT, Out = T>,
    T: StagedType + 'a,
{
}

// For FieldPath - owned access chain
impl<P, F, T> CopyFieldAccess<T> for FieldPath<P, F>
where
    P: Staged,
    P::Out: StagedType,
    F: Field<Out = T>,
    T: StagedType,
    FieldPath<P, F>: Staged,
{
}

// -----------------------------------------------------------------------------
// RefFieldAccess - for pointer types only (not Owned)
// -----------------------------------------------------------------------------

// For Var<SRef<'a, T>> - can get immutable references
impl<'a, T: StagedType> RefFieldAccess<'a, T> for Var<SRef<'a, T>> where Var<SRef<'a, T>>: Staged {}

// For Var<SRefMut<'a, T>> - can also get immutable references
impl<'a, T: StagedType> RefFieldAccess<'a, T> for Var<SRefMut<'a, T>> where
    Var<SRefMut<'a, T>>: Staged
{
}

// For FieldRef - chaining
impl<'a, P, F, T, ParentT> RefFieldAccess<'a, T> for FieldRef<'a, P, F>
where
    P: Staged<Out = SRef<'a, ParentT>>,
    ParentT: StagedType + 'a,
    F: Field<Parent = ParentT, Out = T>,
    T: StagedType + 'a,
{
}

// For FieldMutRef - can get immutable reference from mutable
impl<'a, P, F, T, ParentT> RefFieldAccess<'a, T> for FieldMutRef<'a, P, F>
where
    P: Staged<Out = SRefMut<'a, ParentT>>,
    ParentT: StagedType + 'a,
    F: Field<Parent = ParentT, Out = T>,
    T: StagedType + 'a,
{
}

// -----------------------------------------------------------------------------
// MutRefFieldAccess - for mutable pointer types only
// -----------------------------------------------------------------------------

// For Var<SRefMut<'a, T>> - can get mutable references
impl<'a, T: StagedType> MutRefFieldAccess<'a, T> for Var<SRefMut<'a, T>> where
    Var<SRefMut<'a, T>>: Staged
{
}

// For FieldMutRef - chaining mutable references
impl<'a, P, F, T, ParentT> MutRefFieldAccess<'a, T> for FieldMutRef<'a, P, F>
where
    P: Staged<Out = SRefMut<'a, ParentT>>,
    ParentT: StagedType + 'a,
    F: Field<Parent = ParentT, Out = T>,
    T: StagedType + 'a,
{
}

// -----------------------------------------------------------------------------
// OwnedFieldAccess - for Owned types (no lifetime, no references)
// -----------------------------------------------------------------------------

// For Var<T> - plain struct (by-value)
// Allows .field() and .get_ptr() but NOT .get_ref() (would return dangling)
impl<T: StagedType> OwnedFieldAccess<T> for Var<T> where Var<T>: Staged {}

// For Var<Owned<T>> - explicit owned marker
impl<T: StagedType> OwnedFieldAccess<T> for Var<Owned<T>> where Var<Owned<T>>: Staged {}

// For FieldPath - chaining
impl<P, F, T> OwnedFieldAccess<T> for FieldPath<P, F>
where
    P: Staged,
    P::Out: StagedType,
    F: Field<Out = T>,
    T: StagedType,
    FieldPath<P, F>: Staged,
{
}
