//! Struct support for staged computations.
//!
//! This module provides:
//! - `Field` trait for describing struct fields
//! - Field accessor operations (LoadField, FieldRef, FieldPtr, etc.)
//! - Extension methods on VarRef for field access

use crate::ptr::{SPtr, SMutPtr};
use crate::refer::{SRef, SRefMut};
use crate::staged::{CompilationContext, Staged, Var};
use crate::types::{CopyType, StagedType};
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
pub trait Field: Copy + 'static {
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
// FieldRef: Get an immutable reference to a field
// =============================================================================

/// Get an immutable reference to a field.
///
/// This computes the field's address but doesn't load its value.
/// Returns a pointer/reference to the field.
pub struct FieldRef<P, F> {
    ptr: P,
    _field: PhantomData<F>,
}

impl<P, F> Staged for FieldRef<P, F>
where
    P: Staged,
    P::Out: StagedType,
    F: Field,
{
    type Out = SRef<F::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let base_ptr = self.ptr.codegen(ctx);

        // Compute field pointer: base + offset
        let offset_const = ctx.builder.ins().iconst(types::I64, F::OFFSET as i64);
        ctx.builder.ins().iadd(base_ptr, offset_const)
    }
}

/// Helper to create a FieldRef operation
pub fn field_ref<P, F>(ptr: P, _field: F) -> FieldRef<P, F>
where
    P: Staged,
    F: Field,
{
    FieldRef {
        ptr,
        _field: PhantomData,
    }
}

// =============================================================================
// FieldMutRef: Get a mutable reference to a field
// =============================================================================

/// Get a mutable reference to a field.
pub struct FieldMutRef<P, F> {
    ptr: P,
    _field: PhantomData<F>,
}

impl<P, F> Staged for FieldMutRef<P, F>
where
    P: Staged,
    P::Out: StagedType,
    F: Field,
{
    type Out = SRefMut<F::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let base_ptr = self.ptr.codegen(ctx);

        let offset_const = ctx.builder.ins().iconst(types::I64, F::OFFSET as i64);
        ctx.builder.ins().iadd(base_ptr, offset_const)
    }
}

/// Helper to create a FieldMutRef operation
pub fn field_mut_ref<P, F>(ptr: P, _field: F) -> FieldMutRef<P, F>
where
    P: Staged,
    F: Field,
{
    FieldMutRef {
        ptr,
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
// Extension methods for VarRef
// =============================================================================

/// Extension trait for field access on VarRef<T> where T is a struct.
///
/// This trait is automatically implemented for all struct types.
pub trait StructFieldAccess<T: StagedType>: Sized + Staged {
    /// Load a Copy field's value.
    ///
    /// # Example
    /// ```ignore
    /// let pt: VarRef<Point> = ...;
    /// let x = pt.get(PointType::x);  // Returns impl Staged<Out = I64Type>
    /// ```
    fn get<F>(self, field: F) -> LoadField<Self, F>
    where
        F: Field<Parent = T, Out: CopyType>,
    {
        load_field(self, field)
    }

    /// Get an immutable reference to a field.
    ///
    /// # Example
    /// ```ignore
    /// let outer: VarRef<Outer> = ...;
    /// let inner_ref = outer.get_ref(OuterType::inner);
    /// ```
    fn get_ref<F>(self, field: F) -> FieldRef<Self, F>
    where
        F: Field<Parent = T>,
    {
        field_ref(self, field)
    }

    /// Get a mutable reference to a field.
    fn get_ref_mut<F>(self, field: F) -> FieldMutRef<Self, F>
    where
        F: Field<Parent = T>,
    {
        field_mut_ref(self, field)
    }

    /// Get a raw immutable pointer to a field.
    fn get_ptr<F>(self, field: F) -> FieldPtr<Self, F>
    where
        F: Field<Parent = T>,
    {
        field_ptr(self, field)
    }

    /// Get a raw mutable pointer to a field.
    fn get_mut_ptr<F>(self, field: F) -> FieldMutPtr<Self, F>
    where
        F: Field<Parent = T>,
    {
        field_mut_ptr(self, field)
    }
}

// Implement for VarRef<T> where T is any struct (StagedType)
impl<T: StagedType> StructFieldAccess<T> for Var<T> where Var<T>: Staged {}

// Implement for VarRef<SRef<T>> for composability
impl<T: StagedType> StructFieldAccess<T> for Var<SRef<T>> where Var<SRef<T>>: Staged {}

// Implement for VarRef<SMutRef<T>> for composability
impl<T: StagedType> StructFieldAccess<T> for Var<SRefMut<T>> where Var<SRefMut<T>>: Staged {}

// Implement for FieldRef to enable chaining (e.g., outer.get_ref(field1).get(field2))
// FieldRef<P, F> implements Staged<Out = SRef<T>> when F: Field<Out = T>
impl<P, F, T> StructFieldAccess<T> for FieldRef<P, F>
where
    P: Staged,
    F: Field<Out = T>,
    T: StagedType,
    FieldRef<P, F>: Staged<Out = SRef<T>>,
{}

// Implement for FieldMutRef to enable chaining
impl<P, F, T> StructFieldAccess<T> for FieldMutRef<P, F>
where
    P: Staged,
    F: Field<Out = T>,
    T: StagedType,
    FieldMutRef<P, F>: Staged<Out = SRefMut<T>>,
{}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    // Simple test struct (will use derive macro once integrated)
    #[repr(C)]
    pub struct TestPoint {
        x: i64,
        y: f64,
    }

    // Manually implement StagedType for testing
    impl StagedType for TestPoint {
        type RuntimeValue<'a> = &'a TestPoint;
        fn cranelift_type() -> cranelift_codegen::ir::Type {
            cranelift_codegen::ir::types::I64
        }
    }

    impl CopyType for TestPoint {}

    // Manual field descriptors for testing
    mod TestPointType {
        use super::*;

        #[derive(Copy, Clone)]
        pub struct x;

        impl Field for x {
            type Parent = TestPoint;
            type Out = I64Type;
            const OFFSET: usize = 0;
            const INDEX: usize = 0;
        }

        #[derive(Copy, Clone)]
        pub struct y;

        impl Field for y {
            type Parent = TestPoint;
            type Out = F64Type;
            const OFFSET: usize = 8;
            const INDEX: usize = 1;
        }
    }

    #[test]
    fn test_field_offset_calculation() {
        // Test that field offsets are correct
        assert_eq!(TestPointType::x::OFFSET, 0);
        assert_eq!(TestPointType::y::OFFSET, 8);
    }
}
