//! Core staged computation traits and abstractions.
//!
//! This module defines the foundation for type-safe staged computations:
//! - `Staged`: Trait for anything that can generate runtime code
//! - `VarRef<T>`: Typed variable references (just indices, Copy-able)
//! - `Const<T>`: Typed constants (Copy-able)

use cranelift_codegen::ir::Value;
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_jit::JITModule;
use std::collections::HashMap;

use crate::types::{ConstantType, StagedType};

// =============================================================================
// Compilation Context
// =============================================================================

/// Context provided during code generation.
///
/// This gives access to the function builder, JIT module, and mappings from
/// our internal IDs to Cranelift entities.
pub struct CompilationContext<'a, 'b> {
    /// The function builder for the current function
    pub builder: &'b mut FunctionBuilder<'a>,
    /// The JIT module for creating new functions
    pub module: &'b mut JITModule,
    /// Mapping from our variable IDs to Cranelift Variables
    pub var_map: &'b mut HashMap<usize, Variable>,
    /// Mapping from our function IDs to Cranelift FuncIds
    pub func_map: &'b HashMap<usize, cranelift_module::FuncId>,
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
// VarRef<T> - Lightweight, Copy-able variable reference
// =============================================================================

/// A lightweight reference to a staged variable.
///
/// `VarRef<T>` is just an index into the Compiler's variable tracking.
/// It's always Copy, enabling easy reuse in expressions.
///
/// # Example
/// ```ignore
/// let x: VarRef<I64Type> = compiler.var();
/// let expr = add(x, x);  // x used twice - no problem, it's Copy!
/// ```
#[derive(Clone, Copy)]
pub struct VarRef<T: StagedType> {
    pub(crate) id: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: StagedType> VarRef<T> {
    /// Create a new variable reference with the given ID
    pub(crate) fn new(id: usize) -> Self {
        VarRef {
            id,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: StagedType> Staged for VarRef<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Look up our ID in the var_map to get the Cranelift Variable
        let var = ctx.var_map.get(&self.id)
            .expect(&format!("Variable {} not found in var_map", self.id));
        ctx.builder.use_var(*var)
    }
}

impl<T: StagedType> std::fmt::Debug for VarRef<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VarRef({})", self.id)
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
