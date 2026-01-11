//! Core staged computation traits and abstractions.
//!
//! This module defines the foundation for type-safe staged computations:
//! - `Staged`: Trait for anything that can generate runtime code
//! - `VarRef<T>`: Typed variable references (just indices, Copy-able)
//! - `Const<T>`: Typed constants (Copy-able)

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_jit::JITModule;
use std::collections::HashMap;

use crate::types::{ConstantType, StagedType, UnitType};
use cranelift_codegen::ir::types;

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
pub struct Var<T: StagedType> {
    pub(crate) id: usize,
    _phantom: std::marker::PhantomData<T>,
}

// Manually implement Clone and Copy to avoid requiring T: Clone
impl<T: StagedType> Clone for Var<T> {
    fn clone(&self) -> Self {
        Var {
            id: self.id,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: StagedType> Copy for Var<T> {}

impl<T: StagedType> Var<T> {
    /// Create a new variable reference with the given ID
    pub(crate) fn new(id: usize) -> Self {
        Var {
            id,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<T: StagedType> Staged for Var<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Look up our ID in the var_map to get the Cranelift Variable
        let var = ctx.var_map.get(&self.id)
            .expect(&format!("Variable {} not found in var_map", self.id));
        ctx.builder.use_var(*var)
    }
}

impl<T: StagedType> std::fmt::Debug for Var<T> {
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
    value: T::RuntimeValue<'static>,
}

impl<T: ConstantType> Const<T> {
    /// Create a new constant value
    pub fn new(value: T::RuntimeValue<'static>) -> Self {
        Const { value }
    }
}

// Conditionally implement Copy when T and T::RuntimeValue are Copy
impl<T: ConstantType + Copy> Copy for Const<T> where T::RuntimeValue<'static>: Copy {}

impl<T: ConstantType> Staged for Const<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        T::codegen_constant(&self.value, ctx.builder)
    }
}

// =============================================================================
// From implementations for ergonomic constant creation
// =============================================================================

impl From<i64> for Const<crate::types::I64Type> {
    fn from(value: i64) -> Self {
        Const::new(value)
    }
}

impl From<u64> for Const<crate::types::U64Type> {
    fn from(value: u64) -> Self {
        Const::new(value)
    }
}

impl From<f64> for Const<crate::types::F64Type> {
    fn from(value: f64) -> Self {
        Const::new(value)
    }
}

impl From<bool> for Const<crate::types::BoolType> {
    fn from(value: bool) -> Self {
        Const::new(value)
    }
}

impl From<()> for Const<crate::types::UnitType> {
    fn from(value: ()) -> Self {
        Const::new(value)
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

// =============================================================================
// IntoStaged trait for ergonomic constant creation
// =============================================================================

/// Trait for values that can be converted into staged expressions.
///
/// This trait enables ergonomic APIs like `assign(var, 42i64)` instead of
/// `assign(var, Const::<I64Type>::new(42))`.
pub trait IntoStaged<T: StagedType> {
    /// The staged type this converts to
    type Staged: Staged<Out = T>;

    /// Convert into a staged expression
    fn into_staged(self) -> Self::Staged;
}

// Implement IntoStaged for primitives
impl IntoStaged<crate::types::I64Type> for i64 {
    type Staged = Const<crate::types::I64Type>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<crate::types::U64Type> for u64 {
    type Staged = Const<crate::types::U64Type>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<crate::types::F64Type> for f64 {
    type Staged = Const<crate::types::F64Type>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<crate::types::BoolType> for bool {
    type Staged = Const<crate::types::BoolType>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<crate::types::UnitType> for () {
    type Staged = Const<crate::types::UnitType>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

// Blanket impl for anything that's already Staged
impl<T, S> IntoStaged<T> for S
where
    T: StagedType,
    S: Staged<Out = T>,
{
    type Staged = S;
    fn into_staged(self) -> Self::Staged {
        self
    }
}

// =============================================================================
// Assign<V, EXPR> - Variable assignment (side effect, returns unit)
// =============================================================================

/// Assignment expression: assigns a value to a variable.
///
/// This is a side-effecting operation that returns `UnitType`.
/// Use with tuples to chain multiple assignments or continue with other expressions.
///
/// # Example
/// ```ignore
/// let x = compiler.var::<I64Type>();
/// let expr = (assign(x, 5i64), x);  // assigns 5 to x, returns x
/// ```
#[derive(Clone)]
pub struct Assign<V, EXPR> {
    var: V,
    expr: EXPR,
}

impl<T, EXPR> Staged for Assign<Var<T>, EXPR>
where
    T: StagedType,
    EXPR: Staged<Out = T>,
{
    type Out = UnitType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate code for the value expression
        let value = self.expr.codegen(ctx);

        // Get or declare the Cranelift Variable
        let var = if let Some(&var) = ctx.var_map.get(&self.var.id) {
            var
        } else {
            // First assignment to this variable - declare it
            let var = ctx.builder.declare_var(T::cranelift_type());
            ctx.var_map.insert(self.var.id, var);
            var
        };

        ctx.builder.def_var(var, value);

        // Return unit value
        ctx.builder.ins().iconst(types::I8, 0)
    }
}

/// Create an assignment expression
///
/// Accepts any value that implements `IntoStaged<T>`.
/// This allows ergonomic usage like `assign(var, 42i64)` instead of
/// `assign(var, Const::<I64Type>::new(42))`.
pub fn assign<T, E>(var: Var<T>, expr: E) -> Assign<Var<T>, E::Staged>
where
    T: StagedType,
    E: IntoStaged<T>,
{
    Assign { var, expr: expr.into_staged() }
}

/// Create a unit constant
pub fn unit() -> Const<UnitType> {
    Const::new(())
}
