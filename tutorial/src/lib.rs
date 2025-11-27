//! # Tutorial: Building a Partial Evaluation Compiler with Futamura Projections
//!
//! This tutorial teaches you how to build a JIT compiler using staging and partial evaluation.
//! You'll learn by implementing a progressively more sophisticated calculator that generates
//! optimized machine code using Cranelift.
//!
//! ## Learning Path
//!
//! 1. **Lesson 1**: Simple Addition (Example - COMPLETE)
//! 2. **Lesson 1b**: Multi-Parameter Functions (Example - COMPLETE)
//! 3. **Lesson 2**: Constants (Exercise - YOU COMPLETE)
//! 4. **Lesson 3**: Variables (Exercise - YOU COMPLETE)
//! 5. **Lesson 4**: Mixed Type Operations (Exercise - YOU COMPLETE)
//! 6. **Lesson 5**: Boolean Operations (Exercise - YOU COMPLETE)
//!
//! Future lessons will cover arrays, loops, conditionals, and SIMD.
//!
//! ## How to Use This Tutorial
//!
//! Each lesson has:
//! - Explanation of the concept
//! - Working example (if it's an example lesson)
//! - Test functions that are currently failing (if it's an exercise lesson)
//!
//! Run `cargo test -p tutorial` to see which tests are failing.
//! Your job: make them pass by implementing the missing functionality!

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    types, AbiParam, Function, InstBuilder, MemFlags, Signature, Type, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::ops::{Add, BitAnd, BitOr, Mul, Not, Sub};
use thiserror::Error;

// =============================================================================
// MODULE DECLARATIONS
// =============================================================================

/// Generic numeric types supporting all primitive numeric types
pub mod num;

/// Boolean types and comparison operations
pub mod bool;

/// Runtime execution components (ScalarValue, CompiledNary)
pub mod runtime;

// Re-export commonly used types
pub use num::{
    Numeric, PrimType,
    StagedNum, StagedI8, StagedU8, StagedI16, StagedU16,
    StagedI32, StagedU32, StagedI64, StagedU64,
    StagedF32, StagedF64,
};

pub use bool::{Condition, StagedBool};
pub use runtime::{CompiledNary, ScalarValue};

// =============================================================================
// CORE INFRASTRUCTURE - The foundation for all lessons
// =============================================================================

/// Errors that can occur during staging and compilation
#[derive(Error, Debug)]
pub enum StagingError {
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Compilation failed: {reason}")]
    CompilationFailed { reason: String },

    #[error("Execution failed: {reason}")]
    ExecutionFailed { reason: String },
}

/// Core trait for staged values - values that generate code at compile time
///
// =============================================================================
// CORE EXPRESSION TYPE WITH TYPE INFERENCE
// =============================================================================

/// Lightweight reference to a computed value (variable)
///
/// This represents a reference to a value that has been computed and stored.
/// Unlike `Expr`, which represents a computation tree, `Var` is just a handle
/// to an already-computed value. This makes it cheap to clone and pass around.
#[derive(Debug, Clone)]
pub struct Var {
    var: Variable,
    var_type: DataType,
}

impl Var {
    /// Create a new variable reference
    pub fn new(var: Variable, var_type: DataType) -> Self {
        Var { var, var_type }
    }

    /// Get the data type of this variable
    pub fn data_type(&self) -> &DataType {
        &self.var_type
    }

    /// Convert to an expression
    pub fn to_expr(self) -> Expr {
        Expr::Variable(self)
    }

    /// Convert to StagedI64 (panics if not I64 type)
    pub fn to_i64(self) -> StagedI64 {
        assert_eq!(&self.var_type, &DataType::I64, "Expected I64, got {:?}", self.var_type);
        StagedI64::Variable(self.var)
    }

    /// Convert to StagedU64 (panics if not U64 type)
    pub fn to_u64(self) -> StagedU64 {
        assert_eq!(&self.var_type, &DataType::U64, "Expected U64, got {:?}", self.var_type);
        StagedU64::Variable(self.var)
    }

    /// Convert to StagedBool (panics if not Bool type)
    pub fn to_bool(self) -> StagedBool {
        assert_eq!(&self.var_type, &DataType::Bool, "Expected Bool, got {:?}", self.var_type);
        StagedBool::Variable(self.var)
    }
}

/// Generic expression that can be any type
///
/// This wraps type-specific expressions (StagedI64, StagedU64, StagedBool)
/// and adds generic control flow (Let, If) that works with any type.
///
/// Note: Expr is NOT Clone because it represents a computation tree.
/// Cloning would duplicate the entire tree, which is expensive and usually unintended.
/// If you need to reference a computed value multiple times, use Let to bind it to a Var.
#[derive(Debug)]
pub enum Expr {
    /// A 64-bit signed integer expression
    I64(StagedI64),

    /// A 64-bit unsigned integer expression
    U64(StagedU64),

    /// A boolean expression
    Bool(StagedBool),

    /// Variable reference (can be any type)
    Variable(Var),

    /// Let binding: bind values to variables, use them in body
    /// Type is inferred from the value expressions
    Let {
        bindings: Vec<(u32, DataType, Box<Expr>)>,  // (var_id, type, value)
        body: Box<Expr>,
    },

    /// Mutable let binding: declare a mutable variable that can be updated with SetVar
    /// Type is inferred from the initial value
    LetMut {
        var_id: u32,                     // Variable ID
        var_type: DataType,              // Variable type
        initial_value: Box<Expr>,        // Initial value
        body: Box<Expr>,                 // Body where var can be read/updated
    },

    /// If-then-else conditional
    /// Both branches must have the same type
    If {
        condition: Box<Expr>,     // Must be Bool
        then_branch: Box<Expr>,   // Any type T
        else_branch: Box<Expr>,   // Must be same type T
        result_type: DataType,    // Inferred from branches
    },

    /// Get element from array at index
    /// Returns the element type (U64, I64, Bool)
    ArrayGet {
        array: StagedArray,       // The array to index into
        index: Box<Expr>,         // Index expression (must be U64)
        element_type: DataType,   // Type of elements in the array
    },

    /// Set element in array at index
    /// This is a statement that returns unit/void
    ArraySet {
        array: StagedArray,       // The array to write to (must be mutable)
        index: Box<Expr>,         // Index expression (must be U64)
        value: Box<Expr>,         // Value to write (type must match element_type)
    },

    /// Update a variable's value (for loops and mutable state)
    /// This is a statement that returns unit/void
    SetVar {
        var: Var,                 // Variable to update
        value: Box<Expr>,         // New value (must match variable's type)
    },

    /// While loop: while condition { body }
    /// Returns unit/void (while loops are statements)
    WhileLoop {
        condition: Box<Expr>,     // Condition expression (must be Bool)
        body: Box<Expr>,          // Body expression (can be any type, result ignored)
    },
}

impl Expr {
    /// Get the type of this expression (type inference!)
    pub fn data_type(&self) -> DataType {
        match self {
            Expr::I64(_) => DataType::I64,
            Expr::U64(_) => DataType::U64,
            Expr::Bool(_) => DataType::Bool,
            Expr::Variable(var) => var.data_type().clone(),
            Expr::Let { body, .. } => body.data_type(),
            Expr::LetMut { body, .. } => body.data_type(),
            Expr::If { result_type, .. } => result_type.clone(),
            Expr::ArrayGet { element_type, .. } => element_type.clone(),
            Expr::ArraySet { .. } => DataType::Unit,
            Expr::SetVar { .. } => DataType::Unit,
            Expr::WhileLoop { .. } => DataType::Unit,
        }
    }

    /// Extract as I64 (consuming)
    pub fn into_i64(self) -> Option<StagedI64> {
        match self {
            Expr::I64(v) => Some(v),
            Expr::Variable(var) if var.var_type == DataType::I64 => Some(StagedI64::Variable(var.var)),
            _ => None,
        }
    }

    /// Extract as U64 (consuming)
    pub fn into_u64(self) -> Option<StagedU64> {
        match self {
            Expr::U64(v) => Some(v),
            Expr::Variable(var) if var.var_type == DataType::U64 => Some(StagedU64::Variable(var.var)),
            _ => None,
        }
    }

    /// Extract as Bool (consuming)
    pub fn into_bool(self) -> Option<StagedBool> {
        match self {
            Expr::Bool(v) => Some(v),
            Expr::Variable(var) if var.var_type == DataType::Bool => Some(StagedBool::Variable(var.var)),
            _ => None,
        }
    }

    /// Borrow as I64
    pub fn as_i64(&self) -> Option<&StagedI64> {
        match self {
            Expr::I64(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow as U64
    pub fn as_u64(&self) -> Option<&StagedU64> {
        match self {
            Expr::U64(v) => Some(v),
            _ => None,
        }
    }

    /// Borrow as Bool
    pub fn as_bool(&self) -> Option<&StagedBool> {
        match self {
            Expr::Bool(v) => Some(v),
            _ => None,
        }
    }

    /// Create a variable reference
    pub fn variable(var: Variable, var_type: DataType) -> Self {
        Expr::Variable(Var::new(var, var_type))
    }

    /// Generate Cranelift IR code for this expression
    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            Expr::I64(v) => v.codegen(builder),
            Expr::U64(v) => v.codegen(builder),
            Expr::Bool(v) => v.codegen(builder),
            Expr::Variable(var) => builder.use_var(var.var),
            Expr::Let { bindings, body } => {
                // Declare and define all bindings
                for (var_id, var_type, value) in bindings {
                    let var = Variable::from_u32(*var_id);
                    builder.declare_var(var, var_type.to_cranelift_type());

                    // Evaluate the value expression
                    let val = value.codegen(builder);

                    // Store it in the variable
                    builder.def_var(var, val);
                }

                // Evaluate the body (which can reference these variables)
                body.codegen(builder)
            }
            Expr::LetMut { var_id, var_type, initial_value, body } => {
                // Declare the mutable variable
                let var = Variable::from_u32(*var_id);
                builder.declare_var(var, var_type.to_cranelift_type());

                // Evaluate and set the initial value
                let val = initial_value.codegen(builder);
                builder.def_var(var, val);

                // Evaluate the body (which can read and update this variable)
                body.codegen(builder)
            }
            Expr::If { condition, then_branch, else_branch, result_type } => {
                // Evaluate the condition
                let cond_val = condition.codegen(builder);

                // Create blocks for control flow
                let then_block = builder.create_block();
                let else_block = builder.create_block();
                let merge_block = builder.create_block();

                // Add a block parameter to merge block to receive the result
                builder.append_block_param(merge_block, result_type.to_cranelift_type());

                // Branch based on condition
                builder.ins().brif(cond_val, then_block, &[], else_block, &[]);

                // Generate then branch
                builder.switch_to_block(then_block);
                builder.seal_block(then_block);
                let then_val = then_branch.codegen(builder);
                builder.ins().jump(merge_block, &[then_val]);

                // Generate else branch
                builder.switch_to_block(else_block);
                builder.seal_block(else_block);
                let else_val = else_branch.codegen(builder);
                builder.ins().jump(merge_block, &[else_val]);

                // Continue at merge block
                builder.switch_to_block(merge_block);
                builder.seal_block(merge_block);

                // The result is the block parameter (phi node)
                builder.block_params(merge_block)[0]
            }
            Expr::ArrayGet { array, index, element_type } => {
                // Get the array pointer and index
                let arr_ptr = builder.use_var(array.ptr_var);
                let index_val = index.codegen(builder);

                // Calculate element size in bytes
                let element_size = match element_type {
                    DataType::Prim(prim_type) => (prim_type.bit_width() / 8) as i64,
                    DataType::Bool => 1,
                    _ => panic!("Unsupported array element type: {:?}", element_type),
                };

                // Calculate byte offset: index * element_size
                let elem_size_const = builder.ins().iconst(types::I64, element_size);
                let byte_offset = builder.ins().imul(index_val, elem_size_const);

                // Calculate element address: arr_ptr + byte_offset
                let elem_addr = builder.ins().iadd(arr_ptr, byte_offset);

                // Load the element
                let cranelift_type = element_type.to_cranelift_type();
                builder.ins().load(cranelift_type, MemFlags::trusted(), elem_addr, 0)
            }
            Expr::ArraySet { array, index, value } => {
                // Get the array pointer, index, and value
                let arr_ptr = builder.use_var(array.ptr_var);
                let index_val = index.codegen(builder);
                let value_val = value.codegen(builder);

                // Calculate element size in bytes
                let element_size = match &array.element_type {
                    DataType::Prim(prim_type) => (prim_type.bit_width() / 8) as i64,
                    DataType::Bool => 1,
                    _ => panic!("Unsupported array element type: {:?}", array.element_type),
                };

                // Calculate byte offset: index * element_size
                let elem_size_const = builder.ins().iconst(types::I64, element_size);
                let byte_offset = builder.ins().imul(index_val, elem_size_const);

                // Calculate element address: arr_ptr + byte_offset
                let elem_addr = builder.ins().iadd(arr_ptr, byte_offset);

                // Store the value
                builder.ins().store(MemFlags::trusted(), value_val, elem_addr, 0);

                // ArraySet returns unit, represented as 0
                builder.ins().iconst(types::I64, 0)
            }
            Expr::SetVar { var, value } => {
                // Evaluate the new value
                let new_val = value.codegen(builder);

                // Update the variable
                builder.def_var(var.var, new_val);

                // SetVar returns unit, represented as 0
                builder.ins().iconst(types::I64, 0)
            }
            Expr::WhileLoop { condition, body } => {
                // Create blocks for the loop
                let header_block = builder.create_block();
                let body_block = builder.create_block();
                let exit_block = builder.create_block();

                // Jump to header
                builder.ins().jump(header_block, &[]);

                // Header: evaluate condition
                // Note: Don't seal yet - we have a back-edge from the loop body!
                builder.switch_to_block(header_block);

                let cond_val = condition.codegen(builder);
                builder.ins().brif(cond_val, body_block, &[], exit_block, &[]);

                // Body: execute loop body
                builder.switch_to_block(body_block);
                builder.seal_block(body_block);

                // Execute body (result is ignored)
                body.codegen(builder);

                // Jump back to header (this is the back-edge)
                builder.ins().jump(header_block, &[]);

                // NOW we can seal the header - all predecessors are known
                builder.seal_block(header_block);

                // Exit block
                builder.switch_to_block(exit_block);
                builder.seal_block(exit_block);

                // WhileLoop returns unit, represented as 0
                builder.ins().iconst(types::I64, 0)
            }
        }
    }
}

// Conversion helpers for ergonomics
impl From<StagedI64> for Expr {
    fn from(v: StagedI64) -> Self {
        Expr::I64(v)
    }
}

impl From<StagedU64> for Expr {
    fn from(v: StagedU64) -> Self {
        Expr::U64(v)
    }
}

impl From<StagedBool> for Expr {
    fn from(v: StagedBool) -> Self {
        Expr::Bool(v)
    }
}

// Display trait for C-like pretty printing
impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::I64(v) => write!(f, "{}", v),
            Expr::U64(v) => write!(f, "{}", v),
            Expr::Bool(v) => write!(f, "{}", v),
            Expr::Variable(var) => write!(f, "v{}", var.var.as_u32()),
            Expr::Let { bindings, body } => {
                writeln!(f, "{{")?;
                for (var_id, var_type, value) in bindings {
                    writeln!(f, "  {:?} v{} = {};", var_type, var_id, value)?;
                }
                writeln!(f, "  {}", body)?;
                write!(f, "}}")
            }
            Expr::LetMut { var_id, var_type, initial_value, body } => {
                writeln!(f, "{{")?;
                writeln!(f, "  {:?} mut v{} = {};", var_type, var_id, initial_value)?;
                writeln!(f, "  {}", body)?;
                write!(f, "}}")
            }
            Expr::If { condition, then_branch, else_branch, .. } => {
                write!(f, "if ({}) {{ {} }} else {{ {} }}", condition, then_branch, else_branch)
            }
            Expr::ArrayGet { array, index, .. } => {
                write!(f, "array_{}[{}]", array.ptr_var.as_u32(), index)
            }
            Expr::ArraySet { array, index, value } => {
                write!(f, "array_{}[{}] = {}", array.ptr_var.as_u32(), index, value)
            }
            Expr::SetVar { var, value } => {
                write!(f, "v{} = {}", var.var.as_u32(), value)
            }
            Expr::WhileLoop { condition, body } => {
                write!(f, "while ({}) {{ {} }}", condition, body)
            }
        }
    }
}

// Display implementations for StagedI64, StagedU64, and StagedBool are now in their respective modules (num.rs and bool.rs)

// =============================================================================
// STAGED BUILDER: Ergonomic AST Construction with Auto Variable Management
// =============================================================================

/// Builder for constructing staged expressions with automatic variable ID management
///
/// This provides ergonomic helpers for building expression trees, especially for
/// let bindings where we need to track variable IDs.
pub struct StagedBuilder {
    next_var_id: u32,
}

impl StagedBuilder {
    /// Create a new builder
    /// Variable IDs start at 1000 to avoid conflicts with function parameters (0-999)
    pub fn new() -> Self {
        StagedBuilder { next_var_id: 1000 }
    }

    /// Create a single let binding with automatic variable ID management
    ///
    /// # Example
    /// ```
    /// # use tutorial::*;
    /// let mut builder = StagedBuilder::new();
    /// let x = Expr::I64(StagedI64::constant(5));
    ///
    /// let expr = builder.let1(x, |builder, var| {
    ///     // var is a lightweight reference to the bound value
    ///     let y = var.to_i64();
    ///     Expr::I64(y.clone() * y)
    /// });
    /// ```
    pub fn let1<F>(&mut self, value: Expr, body: F) -> Expr
    where
        F: FnOnce(&mut Self, Var) -> Expr,
    {
        let var_id = self.next_var_id;
        self.next_var_id += 1;

        // Infer type from the value expression
        let var_type = value.data_type();

        // Create a variable reference for the closure
        let var = Var::new(Variable::from_u32(var_id), var_type.clone());

        // Build the body
        let body_expr = body(self, var);

        Expr::Let {
            bindings: vec![(var_id, var_type, Box::new(value))],
            body: Box::new(body_expr),
        }
    }

    /// Create multiple let bindings with automatic variable ID management
    ///
    /// This allows binding multiple values at once, making the code cleaner
    /// when you have several intermediate computations to name.
    ///
    /// # Example
    /// ```
    /// # use tutorial::*;
    /// let mut builder = StagedBuilder::new();
    /// let x = Expr::I64(StagedI64::constant(3));
    /// let y = Expr::I64(StagedI64::constant(4));
    ///
    /// let expr = builder.let_n(vec![x, y], |builder, vars| {
    ///     let x = vars[0].to_i64();
    ///     let y = vars[1].to_i64();
    ///     // x^2 + y^2
    ///     Expr::I64((x.clone() * x) + (y.clone() * y))
    /// });
    /// ```
    pub fn let_n<F>(&mut self, values: Vec<Expr>, body: F) -> Expr
    where
        F: FnOnce(&mut Self, &[Var]) -> Expr,
    {
        // Allocate variable IDs and collect bindings
        let mut bindings = Vec::new();
        let mut vars = Vec::new();

        for value in values {
            let var_id = self.next_var_id;
            self.next_var_id += 1;

            let var_type = value.data_type();
            let var = Var::new(Variable::from_u32(var_id), var_type.clone());

            bindings.push((var_id, var_type, Box::new(value)));
            vars.push(var);
        }

        // Build the body
        let body_expr = body(self, &vars);

        Expr::Let {
            bindings,
            body: Box::new(body_expr),
        }
    }

    /// Create an if-then-else expression with type checking
    ///
    /// # Example
    /// ```
    /// # use tutorial::*;
    /// let mut builder = StagedBuilder::new();
    /// let cond = Expr::Bool(StagedBool::constant(true));
    ///
    /// let expr = builder.if_then_else(
    ///     cond,
    ///     |_| Expr::I64(StagedI64::constant(10)),
    ///     |_| Expr::I64(StagedI64::constant(20)),
    /// );
    /// ```
    pub fn if_then_else<T, E>(&mut self, condition: Expr, then_fn: T, else_fn: E) -> Expr
    where
        T: FnOnce(&mut Self) -> Expr,
        E: FnOnce(&mut Self) -> Expr,
    {
        // Verify condition is Bool
        assert_eq!(
            condition.data_type(),
            DataType::Bool,
            "If condition must be boolean, got {:?}",
            condition.data_type()
        );

        let then_branch = then_fn(self);
        let else_branch = else_fn(self);

        // Infer result type from then branch
        let result_type = then_branch.data_type();

        // Type check: branches must match
        assert_eq!(
            result_type,
            else_branch.data_type(),
            "If branches must have same type: then={:?}, else={:?}",
            result_type,
            else_branch.data_type()
        );

        Expr::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
            result_type,
        }
    }

    /// Create a while loop: while condition { body }
    ///
    /// # Example
    /// ```
    /// # use tutorial::*;
    /// let mut builder = StagedBuilder::new();
    /// // while i < 10 { ... }
    /// builder.while_loop(
    ///     |builder| {
    ///         // Return a boolean condition expression
    ///         Expr::Bool(StagedBool::constant(true))
    ///     },
    ///     |builder| {
    ///         // Return body expression
    ///         Expr::U64(StagedU64::constant(0))
    ///     }
    /// );
    /// ```
    pub fn while_loop<C, B>(&mut self, condition: C, body: B) -> Expr
    where
        C: FnOnce(&mut Self) -> Expr,
        B: FnOnce(&mut Self) -> Expr,
    {
        let condition_expr = condition(self);
        let body_expr = body(self);

        Expr::WhileLoop {
            condition: Box::new(condition_expr),
            body: Box::new(body_expr),
        }
    }

    /// Get element from array at index
    ///
    /// # Example
    /// ```ignore
    /// let value = builder.array_get(array, StagedU64::constant(5));
    /// ```
    pub fn array_get(&self, array: StagedArray, index: StagedU64) -> Expr {
        let element_type = array.element_type().clone();
        Expr::ArrayGet {
            array,
            index: Box::new(Expr::U64(index)),
            element_type,
        }
    }

    /// Set element in array at index
    ///
    /// # Example
    /// ```ignore
    /// builder.array_set(array, StagedU64::constant(5), Expr::U64(value));
    /// ```
    pub fn array_set(&self, array: StagedArray, index: StagedU64, value: Expr) -> Expr {
        assert!(array.is_mutable(), "Cannot write to immutable array");
        Expr::ArraySet {
            array,
            index: Box::new(Expr::U64(index)),
            value: Box::new(value),
        }
    }

    /// Update a variable's value (for mutable loop counters, etc.)
    ///
    /// # Example
    /// ```ignore
    /// builder.set_var(i_var, Expr::U64(new_value));
    /// ```
    pub fn set_var(&self, var: Var, value: Expr) -> Expr {
        Expr::SetVar {
            var,
            value: Box::new(value),
        }
    }

    /// Create a mutable let binding: let mut var = value; body
    /// The variable can be read and updated with set_var in the body
    ///
    /// # Example
    /// ```ignore
    /// builder.let_mut(
    ///     Expr::U64(StagedU64::constant(0)),  // i = 0
    ///     |builder, i_var| {
    ///         // i_var can be read and updated
    ///         builder.set_var(i_var.clone(), Expr::U64(StagedU64::constant(1)));
    ///         Expr::U64(i_var.to_u64())
    ///     }
    /// )
    /// ```
    pub fn let_mut<F>(&mut self, value: Expr, body: F) -> Expr
    where
        F: FnOnce(&mut Self, Var) -> Expr,
    {
        let var_type = value.data_type();
        let var_id = self.next_var_id;
        self.next_var_id += 1;

        let var = Var::new(Variable::from_u32(var_id), var_type.clone());
        let body_expr = body(self, var);

        Expr::LetMut {
            var_id,
            var_type,
            initial_value: Box::new(value),
            body: Box::new(body_expr),
        }
    }
}

impl Default for StagedBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// This is the heart of our partial evaluation system. Instead of computing
/// values immediately, we build up a description of the computation that will
/// be compiled to machine code later.
pub trait Staged {
    /// The runtime type this staged value will produce when executed
    type RuntimeType;

    /// Generate Cranelift IR code for this value
    fn codegen(&self, builder: &mut FunctionBuilder) -> Value;

    /// Get the Cranelift type representation
    fn cranelift_type() -> cranelift_codegen::ir::Type;
}

// =============================================================================
// LESSON 1: SIMPLE ADDITION (EXAMPLE - COMPLETE)
// =============================================================================
//
// This lesson demonstrates the complete flow:
// 1. Define a staged type (StagedI64)
// 2. Implement operations (add)
// 3. Compile to machine code
// 4. Execute the compiled function
//
// Study this example carefully - it's the template for all exercises!

/// A staged 64-bit signed integer
///
/// This represents an i64 value that will exist at runtime. At compile time,
/// we're just building a description of how to compute it.
// #[derive(Debug, Clone)]
// pub enum StagedI64 {
//     /// A constant value known at compile time
//     Constant(i64),
//
//     /// A variable (function parameter) known only at runtime
//     Variable(Variable),
//
//     /// Addition of two staged values
//     Add(Box<StagedI64>, Box<StagedI64>),
//
//     /// Subtraction of two staged values
//     Sub(Box<StagedI64>, Box<StagedI64>),
//
//     /// Multiplication of two staged values
//     Mul(Box<StagedI64>, Box<StagedI64>),
// }

// Comparison methods for StagedI64 (returns StagedBool)
impl StagedI64 {
    pub fn lt(self, right: StagedI64) -> StagedBool {
        StagedBool::I64Cmp(Condition::LessThan, self.into(), right.into())
    }

    pub fn gt(self, right: StagedI64) -> StagedBool {
        StagedBool::I64Cmp(Condition::GreaterThan, self.into(), right.into())
    }

    pub fn eq(self, right: StagedI64) -> StagedBool {
        StagedBool::I64Cmp(Condition::Equal, self.into(), right.into())
    }

    pub fn ne(self, right: StagedI64) -> StagedBool {
        StagedBool::I64Cmp(Condition::NotEqual, self.into(), right.into())
    }

    pub fn lte(self, right: StagedI64) -> StagedBool {
        StagedBool::I64Cmp(Condition::LessThanOrEqual, self.into(), right.into())
    }

    pub fn gte(self, right: StagedI64) -> StagedBool {
        StagedBool::I64Cmp(Condition::GreaterThanOrEqual, self.into(), right.into())
    }
}

/// A JIT compiler that can compile staged functions to machine code
pub struct Compiler {
    module: JITModule,
}

impl Compiler {
    /// Create a new compiler instance
    pub fn new() -> Result<Self, StagingError> {
        let isa = cranelift_native::builder()
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to create ISA: {}", e),
            })?
            .finish(settings::Flags::new(settings::builder()))
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to finish ISA: {}", e),
            })?;

        let builder = JITBuilder::with_isa(isa, default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self { module })
    }

    /// Compile a staged function that takes multiple i64 parameters (as a slice) and returns i64
    ///
    /// This is a more general version that supports N parameters!
    /// The key differences:
    /// 1. Instead of individual parameters, we pass a pointer to an array
    /// 2. Variables are looked up by index from this array
    /// 3. The body function receives a vector of Variable handles (one per parameter)
    ///
    /// # Example
    ///
    /// To compile `f(x, y, z) = (x + y) * z`:
    ///
    /// ```
    /// use tutorial::{Compiler, StagedI64};
    ///
    /// let mut compiler = Compiler::new().unwrap();
    /// let compiled = compiler.compile_nary_i64(3, |_, vars| {
    ///     let x = StagedI64::variable(vars[0]);
    ///     let y = StagedI64::variable(vars[1]);
    ///     let z = StagedI64::variable(vars[2]);
    ///     let sum = x + y;
    ///     sum * z
    /// }).unwrap();
    ///
    /// assert_eq!(compiled.call(&[2, 3, 4]), 20); // (2 + 3) * 4 = 20
    /// ```
    pub fn compile_nary_i64(
        &mut self,
        num_params: usize,
        body: impl FnOnce(&mut FunctionBuilder, &[Variable]) -> StagedI64,
    ) -> Result<CompiledNaryI64, StagingError> {
        // Create function signature: *const i64 -> i64
        // The function takes a pointer to an array of i64 parameters
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64)); // pointer to params array
        sig.returns.push(AbiParam::new(types::I64)); // return value

        // Create the function
        let mut func = Function::new();
        func.signature = sig;

        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create entry block with parameter (pointer to array)
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let params_ptr = builder.block_params(entry_block)[0];

        // Load each parameter from the array and assign to variables
        // This is the key insight: we load from params[0], params[1], etc.
        let mut param_vars = Vec::new();
        for i in 0..num_params {
            let var = Variable::from_u32(i as u32);
            builder.declare_var(var, types::I64);

            // Load params[i]: compute address = params_ptr + (i * 8)
            let offset = builder.ins().iconst(types::I64, (i * 8) as i64);
            let param_addr = builder.ins().iadd(params_ptr, offset);
            let param_val = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), param_addr, 0);

            // Assign to variable
            builder.def_var(var, param_val);
            param_vars.push(var);
        }

        // Generate the function body
        // The user's closure receives the list of variables
        let result_expr = body(&mut builder, &param_vars);
        let result_val = result_expr.codegen(&mut builder);

        // Return the result
        builder.ins().return_(&[result_val]);

        // Finalize
        builder.seal_all_blocks();
        builder.finalize();

        // Debug output if requested
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            eprintln!("\n========== CRANELIFT IR ==========");
            eprintln!("{}", func);
            eprintln!("==================================\n");
        }

        // Compile to machine code
        let mut ctx = Context::new();
        ctx.func = func;

        let func_id = self
            .module
            .declare_function("staged_func_nary", Linkage::Export, &ctx.func.signature)
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to declare function: {}", e),
            })?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to define function: {}", e),
            })?;

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to finalize: {}", e),
            })?;

        let code_ptr = self.module.get_finalized_function(func_id);

        Ok(CompiledNaryI64 {
            code_ptr,
            num_params,
        })
    }
}

/// A compiled function that takes a slice of i64 parameters and returns i64
///
/// This is the more general form that supports N-ary functions!
/// Instead of individual parameters, we pass all inputs as a slice.
pub struct CompiledNaryI64 {
    code_ptr: *const u8,
    num_params: usize,
}

impl CompiledNaryI64 {
    /// Execute the compiled function with a slice of arguments
    ///
    /// # Safety
    /// The caller must ensure that `args.len() >= num_params`
    pub fn call(&self, args: &[i64]) -> i64 {
        assert!(
            args.len() >= self.num_params,
            "Expected at least {} arguments, got {}",
            self.num_params,
            args.len()
        );

        unsafe {
            // The compiled function takes a pointer to the array of parameters
            let func: extern "C" fn(*const i64) -> i64 = std::mem::transmute(self.code_ptr);
            func(args.as_ptr())
        }
    }

    /// Get the number of parameters this function expects
    pub fn num_params(&self) -> usize {
        self.num_params
    }
}

// =============================================================================
// LESSON 1b: MULTI-PARAMETER FUNCTIONS (EXAMPLE - COMPLETE)
// =============================================================================
//
// Real functions often need multiple parameters! The compile_unary_i64 method
// only supports single-parameter functions, which is limiting.
//
// We've added compile_nary_i64 which supports N parameters:
// - Instead of passing individual parameters, we pass a pointer to an array
// - Each parameter is loaded from the array at the start of the function
// - This matches how dio3's cranelift_backend works!
//
// Key insight: f(x, y, z) becomes f(*params) where:
//   x = params[0]
//   y = params[1]
//   z = params[2]
//
// This is more flexible and scales to any number of parameters!
//
// See the tests below for examples of using compile_nary_i64.

// =============================================================================
// LESSON 2: CONSTANTS (EXERCISE - YOU COMPLETE)
// =============================================================================
//
// Constants are values we know at compile time. By using constants instead of
// variables, we enable more optimizations. This is the essence of partial
// evaluation!
//
// Example: Instead of compiling `add(x, y)` which needs two inputs,
//          we can compile `add(42, y)` which only needs one input.
//          The constant 42 is "baked into" the machine code!
//
// YOUR TASK: Implement the subtract operation for StagedI64

impl StagedI64 {
    /// Subtract two staged values
    ///
    pub fn sub(left: StagedI64, right: StagedI64) -> Self {
        StagedI64::Sub(left.into(), right.into())
    }
}

// =============================================================================
// LESSON 3: VARIABLES (EXERCISE - YOU COMPLETE)
// =============================================================================
//
// Variables represent values we don't know until runtime (function parameters).
// The key insight: we can mix constants and variables! This is partial evaluation.
//
// Example: `add(constant(10), variable(x))` compiles to code that adds 10 to
//          whatever value x has at runtime.
//
// YOUR TASK: Implement the multiply operation

impl StagedI64 {
    /// Multiply two staged values
    ///
    pub fn mul(left: StagedI64, right: StagedI64) -> Self {
        StagedI64::Mul(left.into(), right.into())
    }
}

// =============================================================================
// LESSON 4: MIXED TYPE OPERATIONS (EXERCISE - YOU COMPLETE)
// =============================================================================
//
// Real programs use multiple types: signed integers, unsigned integers, floats, etc.
// Each type needs its own staged representation and operations.
//
// YOUR TASK: Implement StagedU64 for unsigned 64-bit integers

// Comparison methods for StagedU64 (returns StagedBool)
impl StagedU64 {
    /// Less than comparison: self < other
    pub fn lt(self, other: StagedU64) -> StagedBool {
        StagedBool::LessThan(Box::new(self), Box::new(other))
    }

    /// Less than or equal comparison: self <= other
    pub fn le(self, other: StagedU64) -> StagedBool {
        StagedBool::LessThanOrEqual(Box::new(self), Box::new(other))
    }

    /// Greater than comparison: self > other
    pub fn gt(self, other: StagedU64) -> StagedBool {
        StagedBool::GreaterThan(Box::new(self), Box::new(other))
    }

    /// Greater than or equal comparison: self >= other
    pub fn ge(self, other: StagedU64) -> StagedBool {
        StagedBool::GreaterThanOrEqual(Box::new(self), Box::new(other))
    }

    /// Equality comparison: self == other
    pub fn eq(self, other: StagedU64) -> StagedBool {
        StagedBool::Equal(Box::new(self), Box::new(other))
    }

    /// Inequality comparison: self != other
    pub fn ne(self, other: StagedU64) -> StagedBool {
        StagedBool::NotEqual(Box::new(self), Box::new(other))
    }
}

// =============================================================================
// GENERIC COMPILATION - Supporting multiple types
// =============================================================================
//
// This section provides a generic compilation mechanism that can handle
// different staged types (I64, U64, etc.) similar to how dio3 and dio4
// use ArrayRef with type erasure.

/// Runtime data type for parameters and return values
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DataType {
    /// Primitive numeric type (i8, u8, ..., i64, u64, f32, f64)
    Prim(PrimType),

    /// Boolean type
    Bool,

    /// Array type with element type and mutability flag
    Array {
        element_type: Box<DataType>,
        mutable: bool,
    },

    /// Unit type for statements that don't return a value
    Unit,
}

// Convenience constants for common types
impl DataType {
    pub const I8: DataType = DataType::Prim(PrimType::I8);
    pub const U8: DataType = DataType::Prim(PrimType::U8);
    pub const I16: DataType = DataType::Prim(PrimType::I16);
    pub const U16: DataType = DataType::Prim(PrimType::U16);
    pub const I32: DataType = DataType::Prim(PrimType::I32);
    pub const U32: DataType = DataType::Prim(PrimType::U32);
    pub const I64: DataType = DataType::Prim(PrimType::I64);
    pub const U64: DataType = DataType::Prim(PrimType::U64);
    pub const F32: DataType = DataType::Prim(PrimType::F32);
    pub const F64: DataType = DataType::Prim(PrimType::F64);
}

impl DataType {
    /// Create an immutable array type
    pub fn arr(element_type: DataType) -> Self {
        DataType::Array {
            element_type: Box::new(element_type),
            mutable: false,
        }
    }

    /// Create a mutable array type
    pub fn mut_arr(element_type: DataType) -> Self {
        DataType::Array {
            element_type: Box::new(element_type),
            mutable: true,
        }
    }
}

/// A runtime scalar value that can be passed to compiled functions
///
/// Similar to how dio4 uses ArrayRef for type erasure, ScalarValue provides
/// a type-erased wrapper for scalar parameters. This allows calling compiled
/// functions with heterogeneous parameter types in a type-safe way.
///
/// # Example
///
/// ```
/// use tutorial::{Compiler, DataType, StagedU64, Expr, ScalarValue};
///
/// let mut compiler = Compiler::new().unwrap();
/// let mut compiled = compiler.compile_nary(
///     vec![DataType::U64, DataType::I64],
///     DataType::U64,
///     |_, vars| {
///         let x = StagedU64::variable(vars[0]);
///         let y_as_u64 = StagedU64::variable(vars[1]);
///         Expr::U64(x + y_as_u64)
///     }
/// ).unwrap();
///
/// // Call with ScalarValues instead of raw slices
/// let result = compiled.call(&[
///     ScalarValue::U64(10),
///     ScalarValue::I64(5)
/// ]).unwrap();
///
/// assert_eq!(result, ScalarValue::U64(15));
/// ```
// ScalarValue has been moved to runtime.rs

impl DataType {
    /// Get the Cranelift type for this data type
    fn to_cranelift_type(&self) -> Type {
        match self {
            DataType::Prim(prim_type) => prim_type.to_cranelift_type(),
            DataType::Bool => types::I8, // Booleans are i8 (0 or 1)
            DataType::Array { .. } => types::I64, // Arrays are pointers (i64)
            DataType::Unit => types::I64, // Unit represented as i64 (unused)
        }
    }

    /// Check if this is a scalar type (primitive numeric or boolean)
    pub fn is_scalar(&self) -> bool {
        matches!(self, DataType::Prim(_) | DataType::Bool)
    }

    /// Check if this is an array type
    pub fn is_array(&self) -> bool {
        matches!(self, DataType::Array { .. })
    }

    /// Check if this is a primitive numeric type
    pub fn is_prim(&self) -> bool {
        matches!(self, DataType::Prim(_))
    }
}

/// A type-erased staged value that can hold any staged type
///
/// This is similar to how dio3/dio4 use ArrayRef to abstract over different
/// concrete array types. It allows generic compilation while preserving
/// type information for code generation.
#[derive(Debug, Clone)]
pub enum StagedValue {
    I64(StagedI64),
    U64(StagedU64),
    Bool(StagedBool),
}

impl StagedValue {
    /// Generate Cranelift IR code for this value
    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            StagedValue::I64(v) => v.codegen(builder),
            StagedValue::U64(v) => v.codegen(builder),
            StagedValue::Bool(v) => v.codegen(builder),
        }
    }

    /// Get the runtime data type
    pub fn data_type(&self) -> DataType {
        match self {
            StagedValue::I64(_) => DataType::I64,
            StagedValue::U64(_) => DataType::U64,
            StagedValue::Bool(_) => DataType::Bool,
        }
    }
}

// Conversion traits for ergonomic usage
impl From<StagedI64> for StagedValue {
    fn from(v: StagedI64) -> Self {
        StagedValue::I64(v)
    }
}

impl From<StagedU64> for StagedValue {
    fn from(v: StagedU64) -> Self {
        StagedValue::U64(v)
    }
}

impl From<StagedBool> for StagedValue {
    fn from(v: StagedBool) -> Self {
        StagedValue::Bool(v)
    }
}

// =============================================================================
// LESSON 7: ARRAYS (EXAMPLE - COMPLETE)
// =============================================================================
//
// Arrays allow us to process multiple values efficiently. Unlike scalars which
// hold a single value, arrays hold many values and we can loop over them.
//
// Key concepts:
// - Arrays are passed as (pointer, length) pairs
// - We can index into arrays to read values
// - We can write values back to arrays
// - For loops let us process each element
//
// This is similar to how dio4 passes arrays via *const u8 pointers!

/// Reference to a function parameter (scalar or array)
///
/// This enum distinguishes between scalar parameters (single values stored in variables)
/// and array parameters (pointer + length pairs).
#[derive(Debug, Clone)]
pub enum ParamRef {
    /// A scalar parameter stored in a single variable
    Scalar {
        var: Variable,
        data_type: DataType,
    },

    /// An array parameter stored as pointer + length
    Array {
        ptr_var: Variable,    // Variable holding the array pointer
        len_var: Variable,    // Variable holding the array length
        element_type: DataType, // Type of array elements
        mutable: bool,        // Whether array is mutable (*mut vs *const)
    },
}

impl ParamRef {
    /// Get the underlying data type (scalar type or array type)
    pub fn data_type(&self) -> DataType {
        match self {
            ParamRef::Scalar { data_type, .. } => data_type.clone(),
            ParamRef::Array { element_type, mutable, .. } => DataType::Array {
                element_type: Box::new(element_type.clone()),
                mutable: *mutable,
            },
        }
    }
}

/// A staged array reference
///
/// This represents an array that will exist at runtime. At compile time,
/// we're building up operations on the array (indexing, length checks, etc.)
/// that will be compiled to machine code.
///
/// Arrays work like Rust slices: they have a pointer and a length.
#[derive(Debug, Clone)]
pub struct StagedArray {
    ptr_var: Variable,      // Variable holding pointer to array data
    len_var: Variable,      // Variable holding array length
    element_type: DataType, // Type of elements
    mutable: bool,          // Whether we can write to this array
}

impl StagedArray {
    /// Create an array reference from a ParamRef
    pub fn from_param(param: &ParamRef) -> Self {
        match param {
            ParamRef::Array { ptr_var, len_var, element_type, mutable } => {
                StagedArray {
                    ptr_var: *ptr_var,
                    len_var: *len_var,
                    element_type: element_type.clone(),
                    mutable: *mutable,
                }
            }
            _ => panic!("Expected array parameter, got scalar"),
        }
    }

    /// Create an array reference from raw variables
    pub fn new(ptr_var: Variable, len_var: Variable, element_type: DataType, mutable: bool) -> Self {
        StagedArray {
            ptr_var,
            len_var,
            element_type,
            mutable,
        }
    }

    /// Get the length of this array as a staged U64
    pub fn len(&self) -> StagedU64 {
        StagedU64::Variable(self.len_var)
    }

    /// Get the element type
    pub fn element_type(&self) -> &DataType {
        &self.element_type
    }

    /// Check if array is mutable
    pub fn is_mutable(&self) -> bool {
        self.mutable
    }
}

impl Compiler {
    /// Compile a generic n-ary function with typed parameters
    ///
    /// This is the generic version of compile_nary_i64 that supports multiple types.
    /// Similar to how dio3/dio4 handle ArrayRef, we use DataType to specify parameter
    /// types and StagedValue for type-erased staged values.
    ///
    /// # Example
    ///
    /// ```
    /// use tutorial::{Compiler, StagedI64, StagedU64, Expr, DataType};
    ///
    /// let mut compiler = Compiler::new().unwrap();
    /// let compiled = compiler.compile_nary(
    ///     vec![DataType::U64, DataType::I64],
    ///     DataType::U64,
    ///     |_, vars| {
    ///         let x = StagedU64::variable(vars[0]);
    ///         let y = StagedI64::variable(vars[1]);
    ///         let y_unsigned = StagedU64::variable(vars[1]); // reinterpret as U64
    ///         Expr::U64(x + y_unsigned)
    ///     }
    /// ).unwrap();
    /// ```
    pub fn compile_nary(
        &mut self,
        param_types: Vec<DataType>,
        return_type: DataType,
        body: impl FnOnce(&mut StagedBuilder, &[Variable]) -> Expr,
    ) -> Result<CompiledNary, StagingError> {
        // Count total variables needed (scalars use 1 var, arrays use 2: ptr + len)
        let mut total_vars = 0;
        let mut total_slots = 0;
        for param_type in &param_types {
            match param_type {
                DataType::Array { .. } => {
                    total_vars += 2; // ptr and len
                    total_slots += 2; // ptr and len in parameter array
                }
                _ => {
                    total_vars += 1;
                    total_slots += 1;
                }
            }
        }

        // Create function signature: *const u64 -> <return_type>
        // We pass a pointer to an array of u64 values. Each scalar takes 1 slot,
        // each array takes 2 slots (pointer, length).
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64)); // pointer (as i64) to u64 array
        sig.returns
            .push(AbiParam::new(return_type.to_cranelift_type())); // return value

        // Create the function
        let mut func = Function::new();
        func.signature = sig;

        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create entry block with parameter (pointer to array)
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let params_ptr = builder.block_params(entry_block)[0];

        // Load each parameter from the u64 array and assign to variables
        // Scalars: single value
        // Arrays: two values (pointer, length)
        let mut param_vars = Vec::new();
        let mut slot_offset = 0;
        let mut var_id = 0u32;

        for param_type in &param_types {
            match param_type {
                DataType::Array { .. } => {
                    // Arrays take 2 slots: pointer and length

                    // Load pointer
                    let ptr_var = Variable::from_u32(var_id);
                    var_id += 1;
                    builder.declare_var(ptr_var, types::I64);
                    let ptr_offset = builder.ins().iconst(types::I64, (slot_offset * 8) as i64);
                    let ptr_addr = builder.ins().iadd(params_ptr, ptr_offset);
                    let ptr_val = builder.ins().load(types::I64, MemFlags::trusted(), ptr_addr, 0);
                    builder.def_var(ptr_var, ptr_val);
                    slot_offset += 1;

                    // Load length
                    let len_var = Variable::from_u32(var_id);
                    var_id += 1;
                    builder.declare_var(len_var, types::I64);
                    let len_offset = builder.ins().iconst(types::I64, (slot_offset * 8) as i64);
                    let len_addr = builder.ins().iadd(params_ptr, len_offset);
                    let len_val = builder.ins().load(types::I64, MemFlags::trusted(), len_addr, 0);
                    builder.def_var(len_var, len_val);
                    slot_offset += 1;

                    // For backward compatibility, add both to param_vars
                    // (The user will need to know arrays use 2 consecutive vars)
                    param_vars.push(ptr_var);
                    param_vars.push(len_var);
                }
                _ => {
                    // Scalars take 1 slot
                    let var = Variable::from_u32(var_id);
                    var_id += 1;
                    let cranelift_type = param_type.to_cranelift_type();
                    builder.declare_var(var, cranelift_type);

                    let byte_offset = slot_offset * 8;
                    let offset = builder.ins().iconst(types::I64, byte_offset as i64);
                    let param_addr = builder.ins().iadd(params_ptr, offset);
                    let param_val = builder.ins().load(cranelift_type, MemFlags::trusted(), param_addr, 0);
                    builder.def_var(var, param_val);
                    param_vars.push(var);
                    slot_offset += 1;
                }
            }
        }

        // Generate the function body using StagedBuilder
        let mut staged_builder = StagedBuilder::new();
        let result_expr = body(&mut staged_builder, &param_vars);

        // Debug output: expression tree
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            eprintln!("\n========== EXPRESSION TREE ==========");
            eprintln!("{}", result_expr);
            eprintln!("=====================================\n");
        }

        // Verify return type matches
        if result_expr.data_type() != return_type {
            return Err(StagingError::TypeMismatch {
                expected: format!("{:?}", return_type),
                actual: format!("{:?}", result_expr.data_type()),
            });
        }

        let result_val = result_expr.codegen(&mut builder);

        // Return the result
        builder.ins().return_(&[result_val]);

        // Finalize
        builder.seal_all_blocks();
        builder.finalize();

        // Debug output if requested
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            eprintln!("\n========== CRANELIFT IR ==========");
            eprintln!("{}", func);
            eprintln!("==================================\n");
        }

        // Compile to machine code
        let mut ctx = Context::new();
        ctx.func = func;

        let func_id = self
            .module
            .declare_function("staged_func_nary", Linkage::Export, &ctx.func.signature)
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to declare function: {}", e),
            })?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to define function: {}", e),
            })?;

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to finalize: {}", e),
            })?;

        let code_ptr = self.module.get_finalized_function(func_id);

        Ok(CompiledNary::new(code_ptr, param_types, return_type))
    }
}

// CompiledNary has been moved to runtime.rs

// =============================================================================
// LESSON 5: BOOLEAN OPERATIONS
// =============================================================================
//
// StagedBool, Condition, and all boolean operations have been moved to bool.rs
// This keeps the code organized and separates numeric types from boolean logic.

// =============================================================================
// TESTS - Your guide through the tutorial
// =============================================================================
//
// These tests are ordered by difficulty. Make them pass one at a time!

#[cfg(test)]
mod tests {
    use super::*;

    // -------------------------------------------------------------------------
    // LESSON 1 TESTS: Simple Addition (Example - Should PASS)
    // -------------------------------------------------------------------------

    #[test]
    fn test_lesson1_constant_addition() {
        // This compiles: f(x) = x + 5
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(1, |_, vars| {
                let x = StagedI64::variable(vars[0]);
                let five = StagedI64::constant(5);
                StagedI64::add(x, five)
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 15);
        assert_eq!(compiled.call(&[0]), 5);
        assert_eq!(compiled.call(&[-3]), 2);
    }

    #[test]
    fn test_lesson1_double_addition() {
        // This compiles: f(x) = x + x (which is x * 2)
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(1, |_, vars| {
                let x = StagedI64::variable(vars[0]);
                let x2 = StagedI64::variable(vars[0]);
                StagedI64::add(x, x2)
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 20);
        assert_eq!(compiled.call(&[7]), 14);
    }

    #[test]
    fn test_lesson1_nested_addition() {
        // This compiles: f(x) = (x + 1) + 2
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(1, |_, vars| {
                let x = StagedI64::variable(vars[0]);
                let one = StagedI64::constant(1);
                let two = StagedI64::constant(2);
                let x_plus_1 = StagedI64::add(x, one);
                StagedI64::add(x_plus_1, two)
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 13);
        assert_eq!(compiled.call(&[0]), 3);
    }

    // -------------------------------------------------------------------------
    // LESSON 1b TESTS: Multi-Parameter Functions (Example - Should PASS)
    // -------------------------------------------------------------------------

    #[test]
    fn test_lesson1b_binary_addition() {
        // This compiles: f(x, y) = x + y
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(2, |_, vars| {
                let x = StagedI64::variable(vars[0]);
                let y = StagedI64::variable(vars[1]);
                StagedI64::add(x, y)
            })
            .unwrap();

        assert_eq!(compiled.call(&[10, 5]), 15);
        assert_eq!(compiled.call(&[100, 200]), 300);
        assert_eq!(compiled.call(&[-3, 8]), 5);
    }

    #[test]
    fn test_lesson1b_ternary_expression() {
        // This compiles: f(x, y, z) = (x + y) * z
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(3, |_, vars| {
                let x = StagedI64::variable(vars[0]);
                let y = StagedI64::variable(vars[1]);
                let z = StagedI64::variable(vars[2]);
                let sum = StagedI64::add(x, y);
                StagedI64::mul(sum, z)
            })
            .unwrap();

        // (2 + 3) * 4 = 20
        assert_eq!(compiled.call(&[2, 3, 4]), 20);
        // (10 + 5) * 2 = 30
        assert_eq!(compiled.call(&[10, 5, 2]), 30);
        // (1 + 1) * 100 = 200
        assert_eq!(compiled.call(&[1, 1, 100]), 200);
    }

    #[test]
    fn test_lesson1b_mixed_constants_and_variables() {
        // This compiles: f(x, y) = (x + 10) * y
        // Shows partial evaluation: the constant 10 is baked into the code!
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(2, |_, vars| {
                let x = StagedI64::variable(vars[0]);
                let y = StagedI64::variable(vars[1]);
                let ten = StagedI64::constant(10);
                let x_plus_10 = StagedI64::add(x, ten);
                StagedI64::mul(x_plus_10, y)
            })
            .unwrap();

        // (5 + 10) * 2 = 30
        assert_eq!(compiled.call(&[5, 2]), 30);
        // (0 + 10) * 3 = 30
        assert_eq!(compiled.call(&[0, 3]), 30);
        // (90 + 10) * 1 = 100
        assert_eq!(compiled.call(&[90, 1]), 100);
    }

    #[test]
    fn test_lesson1b_complex_multi_param() {
        // This compiles: f(a, b, c, d) = (a + b) * (c - d)
        // Demonstrates multiple variables and nested operations
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(4, |_, vars| {
                let a = StagedI64::variable(vars[0]);
                let b = StagedI64::variable(vars[1]);
                let c = StagedI64::variable(vars[2]);
                let d = StagedI64::variable(vars[3]);
                let left = StagedI64::add(a, b);
                let right = StagedI64::sub(c, d);
                StagedI64::mul(left, right)
            })
            .unwrap();

        // (1 + 2) * (10 - 5) = 3 * 5 = 15
        assert_eq!(compiled.call(&[1, 2, 10, 5]), 15);
        // (10 + 20) * (100 - 50) = 30 * 50 = 1500
        assert_eq!(compiled.call(&[10, 20, 100, 50]), 1500);
    }

    #[test]
    fn test_lesson1b_zero_params_all_constants() {
        // This compiles: f() = 42 + 58
        // Even though we have "zero runtime params", we still need to pass
        // an empty slice. The result is always constant!
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(0, |_, _vars| {
                let forty_two = StagedI64::constant(42);
                let fifty_eight = StagedI64::constant(58);
                StagedI64::add(forty_two, fifty_eight)
            })
            .unwrap();

        // No matter what we pass (even empty), result is always 100
        assert_eq!(compiled.call(&[]), 100);
        assert_eq!(compiled.call(&[999, 888]), 100); // extra args ignored
    }

    // -------------------------------------------------------------------------
    // LESSON 2 TESTS: Constants (Exercise - Currently FAIL)
    // -------------------------------------------------------------------------

    #[test]
    fn test_lesson2_simple_subtraction() {
        // This should compile: f(x) = x - 3
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(1, |_, vars| {
                let x = StagedI64::variable(vars[0]);
                let three = StagedI64::constant(3);
                StagedI64::sub(x, three)
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 7);
        assert_eq!(compiled.call(&[5]), 2);
        assert_eq!(compiled.call(&[0]), -3);
    }

    #[test]
    fn test_lesson2_constant_only_subtraction() {
        // We're compiling a function with NO parameters - everything is constant!
        // This should compile: f() = 100 - 42

        // Note: We use compile_nary_i64 with 0 parameters
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(0, |_, _vars| {
                let hundred = StagedI64::constant(100);
                let fortytwo = StagedI64::constant(42);
                StagedI64::sub(hundred, fortytwo)
            })
            .unwrap();

        // No matter what we pass, the result is always 58!
        assert_eq!(compiled.call(&[]), 58);
        assert_eq!(compiled.call(&[999]), 58);
    }

    // -------------------------------------------------------------------------
    // LESSON 3 TESTS: Variables (Exercise - Currently FAIL)
    // -------------------------------------------------------------------------

    #[test]
    fn test_lesson3_simple_multiplication() {
        // This should compile: f(x) = x * 2
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(1, |_, vars| {
                let x = StagedI64::variable(vars[0]);
                let two = StagedI64::constant(2);
                StagedI64::mul(x, two)
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 20);
        assert_eq!(compiled.call(&[7]), 14);
        assert_eq!(compiled.call(&[-3]), -6);
    }

    #[test]
    fn test_lesson3_complex_expression() {
        // This should compile: f(x) = (x + 5) * (x - 2)
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary_i64(1, |_, vars| {
                let x1 = StagedI64::variable(vars[0]);
                let x2 = StagedI64::variable(vars[0]);
                let five = StagedI64::constant(5);
                let two = StagedI64::constant(2);

                let left = StagedI64::add(x1, five);
                let right = StagedI64::sub(x2, two);
                StagedI64::mul(left, right)
            })
            .unwrap();

        // When x = 3: (3 + 5) * (3 - 2) = 8 * 1 = 8
        assert_eq!(compiled.call(&[3]), 8);
        // When x = 4: (4 + 5) * (4 - 2) = 9 * 2 = 18
        assert_eq!(compiled.call(&[4]), 18);
    }

    // -------------------------------------------------------------------------
    // LESSON 4 TESTS: Mixed Types
    // -------------------------------------------------------------------------

    #[test]
    fn test_lesson4_unsigned_addition() {
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::U64], DataType::U64, |_, param| {
                let x = StagedU64::variable(param[0]);
                let ten = StagedU64::constant(10);
                (x + ten).into()
            })
            .unwrap();

        assert_eq!(compiled.call_u64(&[5]), 15);
        assert_eq!(compiled.call_u64(&[0]), 10);
        assert_eq!(compiled.call_u64(&[100]), 110);
    }

    // -------------------------------------------------------------------------
    // LESSON 5 TESTS: Booleans (Exercise - Currently FAIL)
    // -------------------------------------------------------------------------

    #[test]
    fn test_lesson5_less_than_comparison() {
        // This should compile: f(x) = (x < 10) ? 1 : 0

        let mut compiler = Compiler::new().unwrap();
        // You'll need to create compile_unary_i64_to_bool
        let mut compiled = compiler
            .compile_nary(vec![DataType::I64], DataType::Bool, |_, param| {
                let x = StagedI64::variable(param[0]);
                let ten = StagedI64::constant(10);
                (x.lt(ten)).into()
            })
            .unwrap();

        assert_eq!(compiled.call(&[ScalarValue::I64(2)]).unwrap().as_bool_unchecked(), true);
        assert_eq!(compiled.call(&[ScalarValue::I64(5)]).unwrap().as_bool_unchecked(), true);
        assert_eq!(compiled.call(&[ScalarValue::I64(10)]).unwrap().as_bool_unchecked(), false);
        assert_eq!(compiled.call(&[ScalarValue::I64(15)]).unwrap().as_bool_unchecked(), false);
    }

    // -------------------------------------------------------------------------
    // LESSON 6 TESTS: LET BINDINGS AND CONDITIONALS (COMPLETE EXAMPLES)
    // -------------------------------------------------------------------------

    #[test]
    fn test_lesson6_simple_if_else() {
        // f(x) = if x < 10 then 1 else 0
        // This demonstrates basic if-then-else control flow
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::I64], DataType::I64, |builder, vars| {
                let x = StagedI64::variable(vars[0]);
                let ten = StagedI64::constant(10);
                let cond = x.lt(ten);
                let one = StagedI64::constant(1);
                let zero = StagedI64::constant(0);
                builder.if_then_else(
                    Expr::Bool(cond),
                    |_| Expr::I64(one),
                    |_| Expr::I64(zero),
                )
            })
            .unwrap();

        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(5)])
                .unwrap()
                .as_i64_unchecked(),
            1
        );
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(15)])
                .unwrap()
                .as_i64_unchecked(),
            0
        );
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(10)])
                .unwrap()
                .as_i64_unchecked(),
            0
        );
    }

    #[test]
    fn test_lesson6_if_else_with_computation() {
        // f(x) = if x < 0 then -x else x (absolute value)
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::I64], DataType::I64, |builder, vars| {
                let x = StagedI64::variable(vars[0]);
                let zero = StagedI64::constant(0);
                let cond = x.clone().lt(zero.clone());
                let neg_x = zero - x.clone();
                builder.if_then_else(
                    Expr::Bool(cond),
                    |_| Expr::I64(neg_x.clone()),
                    |_| Expr::I64(x),
                )
            })
            .unwrap();

        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(-5)])
                .unwrap()
                .as_i64_unchecked(),
            5
        );
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(10)])
                .unwrap()
                .as_i64_unchecked(),
            10
        );
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(0)])
                .unwrap()
                .as_i64_unchecked(),
            0
        );
    }

    #[test]
    fn test_lesson6_simple_let_binding() {
        // f(x) = let y = x + 1 in y * y
        // Demonstrates naming an intermediate computation
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::I64], DataType::I64, |builder, vars| {
                let x = StagedI64::variable(vars[0]);
                let one = StagedI64::constant(1);
                let y_binding = x + one;

                builder.let1(Expr::I64(y_binding), |_builder, var| {
                    // var is the bound variable (automatically managed)
                    let y = var.to_i64();
                    Expr::I64(y.clone() * y)
                })
            })
            .unwrap();

        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(5)])
                .unwrap()
                .as_i64_unchecked(),
            36
        ); // (5+1)^2 = 36
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(0)])
                .unwrap()
                .as_i64_unchecked(),
            1
        ); // (0+1)^2 = 1
    }

    #[test]
    fn test_lesson6_let_binding_avoids_recomputation() {
        // f(x) = let y = x * x in y + y
        // Without let binding: (x*x) + (x*x) - computes x*x twice
        // With let binding: let y = x*x in y + y - computes x*x once
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::I64], DataType::I64, |builder, vars| {
                let x = StagedI64::variable(vars[0]);
                let y_binding = x.clone() * x;

                builder.let1(Expr::I64(y_binding), |_builder, var| {
                    let y = var.to_i64();
                    Expr::I64(y.clone() + y)
                })
            })
            .unwrap();

        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(3)])
                .unwrap()
                .as_i64_unchecked(),
            18
        ); // 3*3 + 3*3 = 18
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(10)])
                .unwrap()
                .as_i64_unchecked(),
            200
        ); // 10*10 + 10*10 = 200
    }

    #[test]
    fn test_lesson6_nested_let_bindings() {
        // f(x) = let y = x + 1 in let z = y * 2 in z + y
        // Demonstrates nested let bindings
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::I64], DataType::I64, |builder, vars| {
                let x = StagedI64::variable(vars[0]);
                let one = StagedI64::constant(1);
                let two = StagedI64::constant(2);

                // let y = x + 1 in ...
                builder.let1(Expr::I64(x + one), |builder, y_var| {
                    let y = y_var.to_i64();

                    // let z = y * 2 in ...
                    builder.let1(Expr::I64(y.clone() * two), |_builder, z_var| {
                        let z = z_var.to_i64();

                        // z + y
                        Expr::I64(z + y)
                    })
                })
            })
            .unwrap();

        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(5)])
                .unwrap()
                .as_i64_unchecked(),
            18
        ); // y=6, z=12, result=18
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(10)])
                .unwrap()
                .as_i64_unchecked(),
            33
        ); // y=11, z=22, result=33
    }

    #[test]
    fn test_lesson6_if_inside_let() {
        // f(x) = let y = (if x < 0 then -x else x) in y * 2
        // Demonstrates if-then-else as a let binding
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::I64], DataType::I64, |builder, vars| {
                let x = StagedI64::variable(vars[0]);
                let zero = StagedI64::constant(0);
                let two = StagedI64::constant(2);

                // Absolute value using if-then-else
                let cond = x.clone().lt(zero.clone());
                let neg_x = zero - x.clone();

                // Compute the if-then-else expression first
                let abs_value = builder.if_then_else(
                    Expr::Bool(cond),
                    |_| Expr::I64(neg_x.clone()),
                    |_| Expr::I64(x.clone()),
                );

                // Bind the if-then-else result and use it
                builder.let1(abs_value, |_builder, y_var| {
                    let y = y_var.to_i64();
                    Expr::I64(y * two)
                })
            })
            .unwrap();

        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(-5)])
                .unwrap()
                .as_i64_unchecked(),
            10
        );
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(7)])
                .unwrap()
                .as_i64_unchecked(),
            14
        );
    }

    #[test]
    fn test_lesson6_let_inside_if() {
        // f(x) = if x < 0 then (let y = -x in y * 2) else x
        // Demonstrates let binding inside a branch
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::I64], DataType::I64, |builder, vars| {
                let x = StagedI64::variable(vars[0]);
                let zero = StagedI64::constant(0);
                let two = StagedI64::constant(2);

                let cond = x.clone().lt(zero.clone());
                let neg_x = zero - x.clone();

                builder.if_then_else(
                    Expr::Bool(cond),
                    |builder| {
                        // Then branch: let y = -x in y * 2
                        builder.let1(Expr::I64(neg_x.clone()), |_builder, y_var| {
                            let y = y_var.to_i64();
                            Expr::I64(y * two)
                        })
                    },
                    |_| Expr::I64(x), // Else branch: just x
                )
            })
            .unwrap();

        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(-5)])
                .unwrap()
                .as_i64_unchecked(),
            10
        );
        assert_eq!(
            compiled
                .call(&[ScalarValue::I64(7)])
                .unwrap()
                .as_i64_unchecked(),
            7
        );
    }

    // -------------------------------------------------------------------------
    // LESSON 6 EXERCISES: Implement these patterns yourself!
    // -------------------------------------------------------------------------
    // TODO: Uncomment these tests and implement them after understanding the examples above

    // #[test]
    // fn test_lesson6_exercise_max() {
    //     // TODO: Implement max(x, y) using if-then-else
    //     // f(x, y) = if x > y then x else y
    //     let mut compiler = Compiler::new().unwrap();
    //     let mut compiled = compiler.compile_nary(
    //         vec![DataType::I64, DataType::I64],
    //         DataType::I64,
    //         |_, vars| {
    //             // YOUR CODE HERE
    //             todo!("Implement max using if_then_else")
    //         }
    //     ).unwrap();
    //
    //     assert_eq!(compiled.call(&[ScalarValue::I64(10), ScalarValue::I64(20)]).unwrap().as_i64_unchecked(), 20);
    //     assert_eq!(compiled.call(&[ScalarValue::I64(30), ScalarValue::I64(15)]).unwrap().as_i64_unchecked(), 30);
    // }

    // #[test]
    // fn test_lesson6_exercise_clamp() {
    //     // TODO: Implement clamp(x, min, max) using nested if-then-else
    //     // Returns min if x < min, max if x > max, otherwise x
    //     let mut compiler = Compiler::new().unwrap();
    //     let mut compiled = compiler.compile_nary(
    //         vec![DataType::I64, DataType::I64, DataType::I64],
    //         DataType::I64,
    //         |_, vars| {
    //             // YOUR CODE HERE
    //             todo!("Implement clamp using nested if_then_else")
    //         }
    //     ).unwrap();
    //
    //     assert_eq!(compiled.call(&[
    //         ScalarValue::I64(5), ScalarValue::I64(10), ScalarValue::I64(20)
    //     ]).unwrap().as_i64_unchecked(), 10); // Below min
    //     assert_eq!(compiled.call(&[
    //         ScalarValue::I64(25), ScalarValue::I64(10), ScalarValue::I64(20)
    //     ]).unwrap().as_i64_unchecked(), 20); // Above max
    //     assert_eq!(compiled.call(&[
    //         ScalarValue::I64(15), ScalarValue::I64(10), ScalarValue::I64(20)
    //     ]).unwrap().as_i64_unchecked(), 15); // Within range
    // }

    // #[test]
    // fn test_lesson6_exercise_pythagorean() {
    //     // TODO: Compute x^2 + y^2 using let bindings to avoid recomputation
    //     // f(x, y) = let x2 = x*x in let y2 = y*y in x2 + y2
    //     let mut compiler = Compiler::new().unwrap();
    //     let mut compiled = compiler.compile_nary(
    //         vec![DataType::I64, DataType::I64],
    //         DataType::I64,
    //         |_, vars| {
    //             // YOUR CODE HERE
    //             todo!("Implement using nested let1 or let_n")
    //         }
    //     ).unwrap();
    //
    //     assert_eq!(compiled.call(&[
    //         ScalarValue::I64(3), ScalarValue::I64(4)
    //     ]).unwrap().as_i64_unchecked(), 25); // 3^2 + 4^2 = 25
    //     assert_eq!(compiled.call(&[
    //         ScalarValue::I64(5), ScalarValue::I64(12)
    //     ]).unwrap().as_i64_unchecked(), 169); // 5^2 + 12^2 = 169
    // }

    // #[test]
    // fn test_lesson6_exercise_sign() {
    //     // TODO: Implement sign(x) that returns -1, 0, or 1 using nested if-then-else
    //     // f(x) = if x < 0 then -1 else (if x > 0 then 1 else 0)
    //     let mut compiler = Compiler::new().unwrap();
    //     let mut compiled = compiler.compile_nary(
    //         vec![DataType::I64],
    //         DataType::I64,
    //         |_, vars| {
    //             // YOUR CODE HERE
    //             todo!("Implement sign using nested if_then_else")
    //         }
    //     ).unwrap();
    //
    //     assert_eq!(compiled.call(&[ScalarValue::I64(-10)]).unwrap().as_i64_unchecked(), -1);
    //     assert_eq!(compiled.call(&[ScalarValue::I64(0)]).unwrap().as_i64_unchecked(), 0);
    //     assert_eq!(compiled.call(&[ScalarValue::I64(10)]).unwrap().as_i64_unchecked(), 1);
    // }

    // -------------------------------------------------------------------------
    // GENERIC COMPILATION TESTS: Testing compile_nary with mixed types
    // -------------------------------------------------------------------------

    #[test]
    fn test_generic_u64_addition() {
        // Compile: f(x: u64) = x + 10
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![DataType::U64], DataType::U64, |_, vars| {
                let x = StagedU64::variable(vars[0]);
                let ten = StagedU64::constant(10);
                Expr::U64(x + ten)
            })
            .unwrap();

        assert_eq!(compiled.call_u64(&[5]), 15);
        assert_eq!(compiled.call_u64(&[0]), 10);
        assert_eq!(compiled.call_u64(&[100]), 110);
    }

    #[test]
    fn test_generic_mixed_types_u64_i64() {
        // Compile: f(x: u64, y: i64) -> u64
        // Note: We reinterpret y as u64 for the addition
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::I64],
                DataType::U64,
                |builder, vars| {
                    let x = StagedU64::variable(vars[0]);
                    // Reinterpret vars[1] as U64 (they're both 64-bit values)
                    let y_as_u64 = StagedU64::variable(vars[1]);
                    Expr::U64(x + y_as_u64)
                },
            )
            .unwrap();

        // Pass as mixed i64 slice (both u64 and i64 fit in i64)
        assert_eq!(compiled.call_mixed(&[10, 5]), 15);
        assert_eq!(compiled.call_mixed(&[100, 200]), 300);
    }

    #[test]
    fn test_generic_i64_operations() {
        // Compile: f(a: i64, b: i64, c: i64) -> i64 = (a + b) * c
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::I64, DataType::I64, DataType::I64],
                DataType::I64,
                |builder, vars| {
                    let a = StagedI64::variable(vars[0]);
                    let b = StagedI64::variable(vars[1]);
                    let c = StagedI64::variable(vars[2]);
                    let sum = StagedI64::add(a, b);
                    Expr::I64(StagedI64::mul(sum, c))
                },
            )
            .unwrap();

        assert_eq!(compiled.call_i64(&[2, 3, 4]), 20); // (2 + 3) * 4 = 20
        assert_eq!(compiled.call_i64(&[10, 5, 2]), 30); // (10 + 5) * 2 = 30
    }

    #[test]
    fn test_generic_u64_multiplication() {
        // Compile: f(x: u64, y: u64) -> u64 = x * y
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::U64,
                |builder, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    Expr::U64(x * y)
                },
            )
            .unwrap();

        assert_eq!(compiled.call_u64(&[3, 4]), 12);
        assert_eq!(compiled.call_u64(&[7, 8]), 56);
        assert_eq!(compiled.call_u64(&[0, 100]), 0);
    }

    #[test]
    fn test_generic_type_mismatch_error() {
        // This should fail: declaring return type as U64 but returning I64
        let mut compiler = Compiler::new().unwrap();
        let result = compiler.compile_nary(
            vec![DataType::I64],
            DataType::U64, // Expecting U64 return
            |builder, vars| {
                let x = StagedI64::variable(vars[0]);
                Expr::I64(x) // But returning I64!
            },
        );

        assert!(result.is_err());
        if let Err(StagingError::TypeMismatch { expected, actual }) = result {
            assert!(expected.contains("U64"));
            assert!(actual.contains("I64"));
        } else {
            panic!("Expected TypeMismatch error");
        }
    }

    #[test]
    fn test_generic_zero_params_constant() {
        // Compile: f() -> u64 = 42
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(vec![], DataType::U64, |builder, _vars| {
                Expr::U64(StagedU64::constant(42))
            })
            .unwrap();

        assert_eq!(compiled.call_u64(&[]), 42);
        assert_eq!(compiled.call_u64(&[999, 888]), 42); // Extra args ignored
    }

    #[test]
    fn test_generic_complex_expression() {
        // Compile: f(a: u64, b: u64, c: u64) -> u64 = (a * b) + (c * 2)
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64, DataType::U64],
                DataType::U64,
                |builder, vars| {
                    let a = StagedU64::variable(vars[0]);
                    let b = StagedU64::variable(vars[1]);
                    let c = StagedU64::variable(vars[2]);
                    let two = StagedU64::constant(2);
                    let ab = a * b;
                    let c2 = c * two;
                    Expr::U64(ab + c2)
                },
            )
            .unwrap();

        // (3 * 4) + (5 * 2) = 12 + 10 = 22
        assert_eq!(compiled.call_u64(&[3, 4, 5]), 22);
        // (10 * 2) + (7 * 2) = 20 + 14 = 34
        assert_eq!(compiled.call_u64(&[10, 2, 7]), 34);
    }

    // -------------------------------------------------------------------------
    // SCALARVALUE TESTS: Type-safe heterogeneous calling (like dio4's ArrayRef)
    // -------------------------------------------------------------------------

    #[test]
    fn test_scalarvalue_homogeneous_u64() {
        // Test calling with ScalarValue instead of raw slices
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::U64,
                |builder, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    Expr::U64(x + y)
                },
            )
            .unwrap();

        let result = compiled
            .call(&[ScalarValue::U64(10), ScalarValue::U64(20)])
            .unwrap();

        assert_eq!(result, ScalarValue::U64(30));
    }

    #[test]
    fn test_scalarvalue_heterogeneous_types() {
        // Test mixing U64 and I64 parameters (like dio4 mixing ArrayRef types)
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::I64, DataType::U64],
                DataType::U64,
                |builder, vars| {
                    let a = StagedU64::variable(vars[0]);
                    let b_as_u64 = StagedU64::variable(vars[1]); // Reinterpret i64 as u64
                    let c = StagedU64::variable(vars[2]);
                    // (a + b) * c
                    let sum = a + b_as_u64;
                    Expr::U64(sum * c)
                },
            )
            .unwrap();

        let result = compiled
            .call(&[
                ScalarValue::U64(10),
                ScalarValue::I64(5),
                ScalarValue::U64(2),
            ])
            .unwrap();

        assert_eq!(result, ScalarValue::U64(30)); // (10 + 5) * 2 = 30
    }

    #[test]
    fn test_scalarvalue_type_checking() {
        // Should fail: passing wrong type
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::U64,
                |builder, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    Expr::U64(x + y)
                },
            )
            .unwrap();

        // Try to pass I64 where U64 is expected
        let result = compiled.call(&[ScalarValue::U64(10), ScalarValue::I64(20)]);

        assert!(result.is_err());
        if let Err(StagingError::TypeMismatch { expected, actual }) = result {
            assert!(expected.contains("U64"));
            assert!(actual.contains("I64"));
        } else {
            panic!("Expected TypeMismatch error");
        }
    }

    #[test]
    fn test_scalarvalue_wrong_arg_count() {
        // Should fail: wrong number of arguments
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::U64,
                |builder, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    Expr::U64(x + y)
                },
            )
            .unwrap();

        // Try to pass only 1 argument when 2 are expected
        let result = compiled.call(&[ScalarValue::U64(10)]);

        assert!(result.is_err());
        if let Err(StagingError::ExecutionFailed { reason }) = result {
            assert!(reason.contains("Expected 2 arguments"));
        } else {
            panic!("Expected ExecutionFailed error");
        }
    }

    #[test]
    fn test_scalarvalue_i64_return_type() {
        // Test I64 return type
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::I64, DataType::I64],
                DataType::I64,
                |builder, vars| {
                    let x = StagedI64::variable(vars[0]);
                    let y = StagedI64::variable(vars[1]);
                    Expr::I64(StagedI64::sub(x, y))
                },
            )
            .unwrap();

        let result = compiled
            .call(&[ScalarValue::I64(10), ScalarValue::I64(3)])
            .unwrap();

        assert_eq!(result, ScalarValue::I64(7));
    }

    #[test]
    fn test_scalarvalue_complex_expression() {
        // Complex expression with mixed operations
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64, DataType::U64],
                DataType::U64,
                |builder, vars| {
                    let a = StagedU64::variable(vars[0]);
                    let b = StagedU64::variable(vars[1]);
                    let c = StagedU64::variable(vars[2]);
                    let ten = StagedU64::constant(10);
                    // ((a + b) * c) + 10
                    let sum = a + b;
                    let product = sum * c;
                    Expr::U64(product + ten)
                },
            )
            .unwrap();

        let result = compiled
            .call(&[
                ScalarValue::U64(2),
                ScalarValue::U64(3),
                ScalarValue::U64(4),
            ])
            .unwrap();

        // ((2 + 3) * 4) + 10 = 20 + 10 = 30
        assert_eq!(result, ScalarValue::U64(30));
    }

    #[test]
    fn test_scalarvalue_unwrap_methods() {
        // Test the unwrap helper methods
        let i64_val = ScalarValue::I64(42);
        let u64_val = ScalarValue::U64(100);
        let bool_val = ScalarValue::Bool(true);

        assert_eq!(i64_val.as_i64_unchecked(), 42);
        assert_eq!(u64_val.as_u64_unchecked(), 100);
        assert_eq!(bool_val.as_bool_unchecked(), true);
    }

    #[test]
    #[should_panic(expected = "Expected U64")]
    fn test_scalarvalue_unwrap_wrong_type() {
        // Should panic when unwrapping as wrong type
        let i64_val = ScalarValue::I64(42);
        i64_val.as_u64_unchecked(); // This should panic
    }

    // -------------------------------------------------------------------------
    // LESSON 7 TESTS: Arrays and Loops (COMPLETE EXAMPLES)
    // -------------------------------------------------------------------------

    #[test]
    fn test_lesson7_array_double() {
        // Compile: f(input: &[u64], output: &mut [u64]) that doubles each element
        // This is an element-wise operation: output[i] = input[i] * 2
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![
                    DataType::arr(DataType::U64),      // input array
                    DataType::mut_arr(DataType::U64),  // output array (mutable)
                ],
                DataType::U64,  // return the length processed
                |builder, vars| {
                    // vars[0] = input_ptr, vars[1] = input_len
                    // vars[2] = output_ptr, vars[3] = output_len
                    let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
                    let output = StagedArray::new(vars[2], vars[3], DataType::U64, true);

                    let len = input.len();

                    // Use let_mut to create a mutable loop counter: let mut i = 0
                    // Then use while loop: while i < len { output[i] = input[i] * 2; i = i + 1 }
                    // This shows how while loops are more fundamental than for loops
                    builder.let_mut(
                        Expr::U64(StagedU64::constant(0)),  // i = 0
                        |builder, i_var| {
                            let loop_expr = builder.while_loop(
                                |builder| {
                                    // Condition: i < len
                                    let i = i_var.clone().to_u64();
                                    Expr::Bool(i.lt(len.clone()))
                                },
                                |builder| {
                                    let i = i_var.clone().to_u64();

                                    // Use let binding to materialize the array element
                                    builder.let1(
                                        builder.array_get(input, i.clone()),
                                        |builder, elem_var| {
                                            // elem_var is now a Var we can use
                                            let elem = elem_var.to_u64();
                                            let doubled = elem * StagedU64::constant(2);

                                            // Set output[i] = doubled, then increment i
                                            builder.let1(
                                                builder.array_set(output, i.clone(), Expr::U64(doubled)),
                                                |builder, _| {
                                                    // i = i + 1
                                                    let i_next = i_var.clone().to_u64() + StagedU64::constant(1);
                                                    builder.set_var(i_var.clone(), Expr::U64(i_next))
                                                }
                                            )
                                        }
                                    )
                                }
                            );

                            // Execute the loop, then return the length
                            builder.let1(loop_expr, |_, _| Expr::U64(len))
                        }
                    )
                },
            )
            .unwrap();

        // Test with actual data
        let input_data: Vec<u64> = vec![1, 2, 3, 4, 5];
        let mut output_data: Vec<u64> = vec![0; 5];

        // Call the compiled function
        // Arguments: [input_ptr, input_len, output_ptr, output_len]
        let args_u64 = vec![
            input_data.as_ptr() as u64,
            input_data.len() as u64,
            output_data.as_mut_ptr() as u64,
            output_data.len() as u64,
        ];

        let result_len = compiled.call_u64(&args_u64);
        assert_eq!(result_len, 5);
        assert_eq!(output_data, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_lesson7_array_simple_length() {
        // Simple test: just return the array length
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::arr(DataType::U64)],  // input array
                DataType::U64,                        // return length
                |_builder, vars| {
                    // vars[0] = input_ptr, vars[1] = input_len
                    let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
                    Expr::U64(input.len())
                },
            )
            .unwrap();

        // Test with actual data
        let input_data: Vec<u64> = vec![1, 6, 3, 8, 5, 9, 2];

        // Call the compiled function
        let args_u64 = vec![
            input_data.as_ptr() as u64,
            input_data.len() as u64,
        ];

        let result = compiled.call_u64(&args_u64);
        assert_eq!(result, 7);
    }
}

// =============================================================================
// WHAT'S NEXT?
// =============================================================================
//
// Once you complete these lessons, you'll understand:
// ✅ Staging - Building code generators instead of computing directly
// ✅ Partial evaluation - Fixing some inputs to specialize code
// ✅ Futamura projections - How specialization creates compilers
// ✅ JIT compilation - Generating machine code at runtime
//
// In future prompts, we'll add:
// - Lesson 6-8: Arrays and loops
// - Lesson 9-11: Conditionals and control flow
// - Lesson 12-14: Functions and inlining
// - Lesson 15+: SIMD and advanced optimizations
//
// The root function will take:
// - input_arrays: Array of pointers to input arrays
// - input_scalars: Array of scalar inputs
// - output_array: Pointer to output array
// - length: Number of elements to process
//
// This matches the NaryOpFn signature from dio4!
