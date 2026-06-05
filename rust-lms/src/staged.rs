//! Core staged computation traits and abstractions.
//!
//! This module defines the foundation for type-safe staged computations:
//! - `Staged`: Trait for anything that can generate runtime code
//! - `VarRef<T>`: Typed variable references (just indices, Copy-able)
//! - `Const<T>`: Typed constants (Copy-able)

use cranelift_codegen::ir::{Block, InstBuilder, MemFlags, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_jit::JITModule;
use cranelift_module::Module;
use std::collections::HashMap;

use crate::types::{ConstantType, StagedType};
use cranelift_codegen::ir::types;

// =============================================================================
// Compilation Context
// =============================================================================

/// Optimized storage for slice parameters (ptr and len as separate variables).
/// This avoids the need for stack slot loads in tight loops.
#[derive(Clone, Copy)]
pub struct SliceVars {
    pub ptr_var: Variable,
    pub len_var: Variable,
}

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
    /// Mapping from extern function IDs to Cranelift FuncRefs (per-function)
    pub extern_func_refs: &'b mut HashMap<usize, cranelift_codegen::ir::FuncRef>,
    /// Mapping from extern function IDs to module FuncIds
    pub extern_func_ids: &'b HashMap<usize, cranelift_module::FuncId>,
    /// Optimized slice variable storage: var_id -> (ptr_var, len_var)
    /// For slice parameters, this allows direct register access instead of stack loads
    pub slice_vars: &'b mut HashMap<usize, SliceVars>,
    /// Cached unit value (iconst.i8 0) - avoids creating duplicate dead values
    pub unit_value: Option<Value>,
    /// Stack of enclosing loops' exit blocks. The innermost loop's exit is on
    /// top; `break_loop` jumps to it. Pushed/popped by the loop codegen.
    pub loop_exit_stack: Vec<Block>,
}

impl<'a, 'b> CompilationContext<'a, 'b> {
    /// Get or create a FuncRef for an external function.
    ///
    /// FuncRefs are per-function, so we cache them in extern_func_refs.
    pub fn get_extern_func_ref(&mut self, extern_id: usize) -> cranelift_codegen::ir::FuncRef {
        if let Some(&func_ref) = self.extern_func_refs.get(&extern_id) {
            return func_ref;
        }

        let func_id = self
            .extern_func_ids
            .get(&extern_id)
            .expect(&format!("Extern function {} not found", extern_id));

        let func_ref = self
            .module
            .declare_func_in_func(*func_id, self.builder.func);
        self.extern_func_refs.insert(extern_id, func_ref);
        func_ref
    }

    /// Get or create the cached unit value (iconst.i8 0).
    ///
    /// This avoids creating duplicate dead values when sequencing side-effecting
    /// operations like `Assign` and `InitVar`.
    pub fn get_unit_value(&mut self) -> Value {
        if let Some(val) = self.unit_value {
            val
        } else {
            let val = self.builder.ins().iconst(types::I8, 0);
            self.unit_value = Some(val);
            val
        }
    }

    /// Resolve the data pointer (`*T`) of a slice operand.
    ///
    /// A slice's Staged value is a single `i64` with one of two encodings:
    /// - **register-resolved** (slice parameters): the `(ptr, len)` pair lives
    ///   in two Cranelift variables recorded in [`Self::slice_vars`], keyed by
    ///   the operand's `var_id`. We read `ptr` directly — no memory access.
    /// - **memory-resolved** (subslices and anything without a `var_id`): the
    ///   operand's `codegen` value is a pointer to a `(ptr, len)` pair on a
    ///   stack slot, with `ptr` at offset 0 and `len` at offset 8.
    ///
    /// This pair of helpers ([`Self::slice_data_ptr`] / [`Self::slice_len`]) is
    /// the single place that knows about slice layout; slice ops call into it
    /// rather than re-deriving the pointer themselves.
    pub fn slice_data_ptr(&mut self, slice: &impl Staged) -> Value {
        if let Some(var_id) = slice.var_id() {
            if let Some(sv) = self.slice_vars.get(&var_id).copied() {
                return self.builder.use_var(sv.ptr_var);
            }
        }
        // Memory-resolved: load ptr from offset 0 of the (ptr, len) pair.
        let slice_ptr = slice.codegen(self);
        self.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 0)
    }

    /// Resolve the length (`usize`) of a slice operand.
    ///
    /// See [`Self::slice_data_ptr`] for the two encodings; `len` is the second
    /// register variable, or offset 8 of the `(ptr, len)` pair.
    pub fn slice_len(&mut self, slice: &impl Staged) -> Value {
        if let Some(var_id) = slice.var_id() {
            if let Some(sv) = self.slice_vars.get(&var_id).copied() {
                return self.builder.use_var(sv.len_var);
            }
        }
        let slice_ptr = slice.codegen(self);
        self.builder
            .ins()
            .load(types::I64, MemFlags::trusted(), slice_ptr, 8)
    }
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

    /// Return the variable ID if this is a direct Var reference.
    /// Used for optimized slice access to bypass stack loads.
    fn var_id(&self) -> Option<usize> {
        None
    }
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
/// let x: VarRef<i64> = compiler.var();
/// let expr = add(x, x);  // x used twice - no problem, it's Copy!
/// ```
pub struct Var<T: StagedType> {
    pub(crate) id: usize,
    _phantom: std::marker::PhantomData<T>,
}

// Manually implement Clone and Copy: a `Var` is just an id, so it is always
// Copy regardless of `T` (no `T: Copy` bound — that would leak into every
// generic that holds a `Var`).
impl<T: StagedType> Clone for Var<T> {
    fn clone(&self) -> Self {
        *self
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
        let var = ctx
            .var_map
            .get(&self.id)
            .expect(&format!("Variable {} not found in var_map", self.id));
        ctx.builder.use_var(*var)
    }

    fn var_id(&self) -> Option<usize> {
        Some(self.id)
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
/// let five = Const::<i64>::new(5);
/// let ten = Const::<i64>::new(10);
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
// From implementations for ergonomic constant creation
// =============================================================================

impl From<i64> for Const<i64> {
    fn from(value: i64) -> Self {
        Const::new(value)
    }
}

impl From<u64> for Const<u64> {
    fn from(value: u64) -> Self {
        Const::new(value)
    }
}

impl From<f64> for Const<f64> {
    fn from(value: f64) -> Self {
        Const::new(value)
    }
}

impl From<bool> for Const<bool> {
    fn from(value: bool) -> Self {
        Const::new(value)
    }
}

impl From<()> for Const<()> {
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
/// `assign(var, Const::<i64>::new(42))`.
pub trait IntoStaged<T: StagedType> {
    /// The staged type this converts to
    type Staged: Staged<Out = T>;

    /// Convert into a staged expression
    fn into_staged(self) -> Self::Staged;
}

// Implement IntoStaged for primitives
impl IntoStaged<i64> for i64 {
    type Staged = Const<i64>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<u64> for u64 {
    type Staged = Const<u64>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<f64> for f64 {
    type Staged = Const<f64>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<bool> for bool {
    type Staged = Const<bool>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<()> for () {
    type Staged = Const<()>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<i32> for i32 {
    type Staged = Const<i32>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<u32> for u32 {
    type Staged = Const<u32>;
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
/// This is a side-effecting operation that returns `()`.
/// Use with tuples to chain multiple assignments or continue with other expressions.
///
/// # Example
/// ```ignore
/// let x = compiler.var::<i64>();
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
    type Out = ();

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

        // Return cached unit value
        ctx.get_unit_value()
    }
}

/// Create an assignment expression
///
/// Accepts any value that implements `IntoStaged<T>`.
/// This allows ergonomic usage like `assign(var, 42i64)` instead of
/// `assign(var, Const::<i64>::new(42))`.
pub fn assign<T, E>(var: Var<T>, expr: E) -> Assign<Var<T>, E::Staged>
where
    T: StagedType,
    E: IntoStaged<T>,
{
    Assign {
        var,
        expr: expr.into_staged(),
    }
}

/// Create a unit constant
pub fn unit() -> Const<()> {
    Const::new(())
}

// =============================================================================
// InitVar<T, EXPR> - Variable initialization wrapper
// =============================================================================

/// A variable with its initialization expression.
///
/// This type combines a variable reference with its initialization, providing
/// an ergonomic API that doesn't require manual tuple unpacking.
///
/// When used in a tuple for sequencing, it performs the initialization.
/// When used in operations (add, assign, etc.), it derefs to the underlying Var.
///
/// # Example
/// ```ignore
/// let i = compiler.let_var(0u64);  // Returns InitVar<u64, Const<u64>>
/// let expr = (i, add(*i, 5i64));   // i initializes, *i gives Var<u64>
/// ```
pub struct LetVar<T: StagedType, EXPR> {
    var: Var<T>,
    init: EXPR,
}

impl<T: StagedType, EXPR> LetVar<T, EXPR> {
    /// Create a new initialized variable wrapper
    pub(crate) fn new(var: Var<T>, init: EXPR) -> Self {
        LetVar { var, init }
    }

    /// Get the underlying variable reference
    pub fn var(&self) -> Var<T>
    where
        T: Copy,
    {
        self.var
    }
}

// Manually implement Clone (Var<T> is always Copy, clone the expr)
impl<T: StagedType + Copy, EXPR: Clone> Clone for LetVar<T, EXPR> {
    fn clone(&self) -> Self {
        LetVar {
            var: self.var.clone(), // Var<T> is Copy
            init: self.init.clone(),
        }
    }
}

// InitVar is Copy when EXPR is Copy (like Const<T>)
impl<T: StagedType + Copy, EXPR: Copy> Copy for LetVar<T, EXPR> {}

// Deref to allow transparent access to the underlying Var
impl<T: StagedType, EXPR> std::ops::Deref for LetVar<T, EXPR> {
    type Target = Var<T>;

    fn deref(&self) -> &Self::Target {
        &self.var
    }
}

// When InitVar is staged, it performs the initialization
impl<T, EXPR> Staged for LetVar<T, EXPR>
where
    T: StagedType,
    EXPR: Staged<Out = T>,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate code for the initialization value
        let value = self.init.codegen(ctx);

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

        // Return cached unit value
        ctx.get_unit_value()
    }
}

// Allow implicit conversion from InitVar to Var for convenience
impl<T: StagedType, EXPR> From<LetVar<T, EXPR>> for Var<T> {
    fn from(init_var: LetVar<T, EXPR>) -> Var<T> {
        init_var.var
    }
}
