//! Core staged computation traits and abstractions.
//!
//! This module defines the foundation for type-safe staged computations:
//! - `Staged`: Trait for anything that can generate runtime code
//! - `Var<T>`: Typed variable references (Copy-able)
//! - `Const<T>`: Typed constants (Copy-able)
//! - Boxing support for dynamic dispatch

use cranelift_codegen::ir::Value;
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_jit::JITModule;

use crate::types::{ConstantType, StagedType};

// =============================================================================
// Compilation Context
// =============================================================================

/// Context provided during code generation.
///
/// This gives access to both the function builder (for generating IR within
/// the current function) and the JIT module (for creating new functions).
pub struct CompilationContext<'a, 'b> {
    /// The function builder for the current function
    pub builder: &'b mut FunctionBuilder<'a>,
    /// The JIT module for creating new functions
    pub module: &'b mut JITModule,
}

// =============================================================================
// Core Trait: Staged
// =============================================================================

/// Anything that represents a staged computation.
///
/// Types implementing this trait can generate Cranelift IR code that produces
/// a value of type `Self::Out` at runtime.
pub trait Staged {
    /// The output type this staged computation produces
    type Out: StagedType;

    /// Generate Cranelift IR code for this computation
    fn codegen(&self, ctx: &mut CompilationContext) -> Value;
}

// =============================================================================
// Var<T> - Lightweight, Copy-able variable reference
// =============================================================================

/// A lightweight reference to a staged variable.
///
/// `Var<T>` is Copy when `T::RuntimeValue` is Copy, enabling easy reuse
/// in expressions without explicit cloning.
///
/// # Example
/// ```ignore
/// let x = Var::<I64Type>::new(Variable::from_u32(0));
/// let expr1 = add(x, y);  // x used here
/// let expr2 = add(x, x);  // x used again - no clone needed!
/// ```
#[derive(Clone)]
pub struct Var<T: StagedType> {
    var: Variable,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: StagedType> Var<T> {
    /// Create a new typed variable reference
    pub fn new(var: Variable) -> Self {
        Var {
            var,
            _phantom: std::marker::PhantomData,
        }
    }
}

// Conditionally implement Copy when T::RuntimeValue is Copy
// Note: We need T: Copy because PhantomData<T> requires it for Copy
impl<T: StagedType + Copy> Copy for Var<T> where T::RuntimeValue: Copy {}

impl<T: StagedType> Staged for Var<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        ctx.builder.use_var(self.var)
    }
}

// =============================================================================
// Const<T> - Typed constants
// =============================================================================

/// A compile-time constant that will be embedded in generated code.
///
/// `Const<T>` is Copy when `T::RuntimeValue` is Copy.
///
/// # Example
/// ```ignore
/// let five = Const::<I64Type>::new(5);
/// let ten = Const::<I64Type>::new(10);
/// ```
#[derive(Clone)]
pub struct Const<T: ConstantType> {
    value: T::RuntimeValue,
}

impl<T: ConstantType> Const<T> {
    /// Create a new constant value
    pub fn new(value: T::RuntimeValue) -> Self {
        Const { value }
    }
}

// Conditionally implement Copy when T and T::RuntimeValue are Copy
impl<T: ConstantType + Copy> Copy for Const<T> where T::RuntimeValue: Copy {}

impl<T: ConstantType> Staged for Const<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        T::codegen_constant(&self.value, ctx.builder)
    }
}

// =============================================================================
// Boxing support: Enable dynamic dispatch when needed
// =============================================================================

/// Extension trait to enable boxing any Staged value for dynamic dispatch.
///
/// # Example
/// ```ignore
/// let x = Var::<I64Type>::new(Variable::from_u32(0));
/// let boxed: Box<dyn Staged<Out = I64Type>> = x.boxed();
/// ```
pub trait BoxableStaged: Staged {
    /// Box this staged value for dynamic dispatch
    fn boxed(&self) -> Box<dyn Staged<Out = Self::Out>>
    where
        Self: Clone + 'static,
        Self::Out: 'static,
    {
        Box::new(self.clone())
    }
}

// Blanket implementation: all Staged types can be boxed
impl<T: Staged> BoxableStaged for T {}