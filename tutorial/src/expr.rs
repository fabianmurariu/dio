//! Expression AST and builder for staged computations.
//!
//! This module provides the core expression types for building staged programs:
//! - `Var`: Lightweight reference to a computed value
//! - `Expr`: Generic expression that can be any type (with type inference)
//! - `StagedBuilder`: Ergonomic builder for constructing expression trees

use cranelift_codegen::ir::{types, FuncRef, InstBuilder, MemFlags, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use std::collections::HashMap;

use crate::bool::StagedBool;
use crate::num::{StagedI64, StagedU64};
use crate::staged_value::StagedValue;
use crate::{DataType, StagedArray};

// =============================================================================
// VAR: LIGHTWEIGHT VARIABLE REFERENCE
// =============================================================================

/// Lightweight reference to a computed value (variable)
///
/// This represents a reference to a value that has been computed and stored.
/// Unlike `Expr`, which represents a computation tree, `Var` is just a handle
/// to an already-computed value. This makes it cheap to clone and pass around.
#[derive(Debug, Clone)]
pub struct Var {
    pub(crate) var: Variable,
    pub(crate) var_type: DataType,
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

// =============================================================================
// EXPR: GENERIC EXPRESSION WITH TYPE INFERENCE
// =============================================================================

/// Generic expression that can be any type
///
/// This uses a trait-based design where any type implementing `StagedValue`
/// can be used, making the system extensible without modifying this enum.
///
/// Note: Expr is NOT Clone because it represents a computation tree.
/// Cloning would duplicate the entire tree, which is expensive and usually unintended.
/// If you need to reference a computed value multiple times, use Let to bind it to a Var.
///
/// All variants now store their computed DataType for O(1) type checking.
#[derive(Debug)]
pub enum Expr {
    /// A staged value (numeric, boolean, or any custom type implementing StagedValue)
    Staged (Box<dyn StagedValue>),

    /// Variable reference (can be any type)
    Variable(Var),

    /// Let binding: bind values to variables, use them in body
    /// Type is inferred from the value expressions
    Let {
        bindings: Vec<(u32, DataType, Box<Expr>)>,  // (var_id, type, value)
        body: Box<Expr>,
        data_type: DataType,              // Cached type (same as body's type)
    },

    /// Mutable let binding: declare a mutable variable that can be updated with SetVar
    /// Type is inferred from the initial value
    LetMut {
        var_id: u32,                     // Variable ID
        var_type: DataType,              // Variable type
        initial_value: Box<Expr>,        // Initial value
        body: Box<Expr>,                 // Body where var can be read/updated
        data_type: DataType,              // Cached type (same as body's type)
    },

    /// If-then-else conditional
    /// Both branches must have the same type
    If {
        condition: Box<Expr>,     // Must be Bool
        then_branch: Box<Expr>,   // Any type T
        else_branch: Box<Expr>,   // Must be same type T
        data_type: DataType,      // Type of both branches
    },

    /// Get element from array at index
    /// Returns the element type (U64, I64, Bool)
    ArrayGet {
        array: StagedArray,       // The array to index into
        index: Box<Expr>,         // Index expression (must be U64)
        // array:DataType
    },

    /// Set element in array at index
    /// This is a statement that returns unit/void
    ArraySet {
        array: StagedArray,       // The array to write to (must be mutable)
        index: Box<Expr>,         // Index expression (must be U64)
        value: Box<Expr>,         // Value to write (type must match element_type)
        // Always DataType::Unit
    },

    /// Update a variable's value (for loops and mutable state)
    /// This is a statement that returns unit/void
    SetVar {
        var: Var,                 // Variable to update
        value: Box<Expr>,         // New value (must match variable's type)
        // Always DataType::Unit
    },

    /// While loop: while condition { body }
    /// Returns unit/void (while loops are statements)
    WhileLoop {
        condition: Box<Expr>,     // Condition expression (must be Bool)
        body: Box<Expr>,          // Body expression (can be any type, result ignored)
        data_type: DataType,      // Always DataType::Unit
    },

    /// Call an external (Rust) function
    /// The function must be registered in the Compiler's function registry
    ExternalCall {
        function_name: String,    // Name of the external function
        args: Vec<Expr>,          // Arguments (types must match signature)
        param_types: Vec<DataType>, // Parameter types from the function signature
        return_type: DataType,    // Return type from the function signature
        // Cached type (same as return_type)
    },

    /// External pointer variable
    /// Used for struct pointers and other opaque pointers
    ExtPtr {
        var: Variable,            // Variable holding the pointer
        name: String,             // Type name for debugging
        data_type: DataType,      // Always DataType::ExtPtr(name)
    },

    /// Load a field from a struct pointer
    /// Generates: load(ptr + offset)
    LoadField {
        ptr: Box<Expr>,           // Pointer expression (must be ExtPtr)
        offset: Box<Expr>,        // Offset in bytes (must be U64)
        field_type: DataType,     // Type of the field to load
    },

    /// Store a value to a struct field
    /// Generates: store(ptr + offset, value)
    StoreField {
        ptr: Box<Expr>,           // Pointer expression (must be ExtPtr)
        offset: Box<Expr>,        // Offset in bytes (must be U64)
        value: Box<Expr>,         // Value to store
    },
}

impl Expr {
    /// Get the type of this expression (O(1) lookup, no computation needed)
    pub fn data_type(&self) -> &DataType {
        match self {
            Expr::Staged (value) => value.data_type(),
            Expr::Variable(var) => var.data_type(),
            Expr::Let { data_type, .. } => data_type,
            Expr::LetMut { data_type, .. } => data_type,
            Expr::If { data_type, .. } => data_type,
            Expr::ArrayGet { array, .. } => array.element_type(),
            Expr::ArraySet {..} => &DataType::Unit,
            Expr::SetVar {..} => &DataType::Unit,
            Expr::WhileLoop { data_type, .. } => data_type,
            Expr::ExternalCall { return_type, .. } => return_type,
            Expr::ExtPtr { data_type, .. } => data_type,
            Expr::LoadField { field_type, .. } => field_type,
            Expr::StoreField {..} => &DataType::Unit,
        }
    }

    /// Extract as I64 (consuming)
    pub fn into_i64(self) -> Option<StagedI64> {
        match self {
            Expr::Staged(v) => v.as_any().downcast_ref::<StagedI64>().cloned(),
            Expr::Variable(var) if var.var_type == DataType::I64 => Some(StagedI64::Variable(var.var)),
            _ => None,
        }
    }

    /// Extract as U64 (consuming)
    pub fn into_u64(self) -> Option<StagedU64> {
        match self {
            Expr::Staged(v) => v.as_any().downcast_ref::<StagedU64>().cloned(),
            Expr::Variable(var) if var.var_type == DataType::U64 => Some(StagedU64::Variable(var.var)),
            _ => None,
        }
    }

    /// Extract as Bool (consuming)
    pub fn into_bool(self) -> Option<StagedBool> {
        match self {
            Expr::Staged(v) => v.as_any().downcast_ref::<StagedBool>().cloned(),
            Expr::Variable(var) if var.var_type == DataType::Bool => Some(StagedBool::Variable(var.var)),
            _ => None,
        }
    }

    /// Borrow as I64
    pub fn as_i64(&self) -> Option<&StagedI64> {
        match self {
            Expr::Staged(v) => v.as_any().downcast_ref::<StagedI64>(),
            _ => None,
        }
    }

    /// Borrow as U64
    pub fn as_u64(&self) -> Option<&StagedU64> {
        match self {
            Expr::Staged(v) => v.as_any().downcast_ref::<StagedU64>(),
            _ => None,
        }
    }

    /// Borrow as Bool
    pub fn as_bool(&self) -> Option<&StagedBool> {
        match self {
            Expr::Staged(v) => v.as_any().downcast_ref::<StagedBool>(),
            _ => None,
        }
    }

    /// Create an I64 expression
    #[allow(non_snake_case)]
    pub fn I64(v: StagedI64) -> Self {
        Expr::Staged(Box::new(v))
    }

    /// Create a U64 expression
    #[allow(non_snake_case)]
    pub fn U64(v: StagedU64) -> Self {
        Expr::Staged(Box::new(v))
    }

    /// Create a Bool expression
    #[allow(non_snake_case)]
    pub fn Bool(v: StagedBool) -> Self {
        Expr::Staged(Box::new(v))
    }

    /// Create a variable reference
    pub fn variable(var: Variable, var_type: DataType) -> Self {
        Expr::Variable(Var::new(var, var_type))
    }

    /// Generate Cranelift IR code for this expression
    pub(crate) fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        self.codegen_with_externals(builder, None)
    }

    /// Generate Cranelift IR code with external function support
    pub(crate) fn codegen_with_externals(
        &self,
        builder: &mut FunctionBuilder,
        external_funcs: Option<&HashMap<String, FuncRef>>,
    ) -> Value {
        match self {
            Expr::Staged(v) => v.codegen(builder),
            Expr::Variable(var) => builder.use_var(var.var),
            Expr::Let { bindings, body, .. } => {
                // Declare and define all bindings
                for (var_id, var_type, value) in bindings {
                    let var = Variable::from_u32(*var_id);
                    builder.declare_var(var, var_type.to_cranelift_type());

                    // Evaluate the value expression
                    let val = value.codegen_with_externals(builder, external_funcs);

                    // Store it in the variable
                    builder.def_var(var, val);
                }

                // Evaluate the body (which can reference these variables)
                body.codegen_with_externals(builder, external_funcs)
            }
            Expr::LetMut { var_id, var_type, initial_value, body, .. } => {
                // Declare the mutable variable
                let var = Variable::from_u32(*var_id);
                builder.declare_var(var, var_type.to_cranelift_type());

                // Evaluate and set the initial value
                let val = initial_value.codegen_with_externals(builder, external_funcs);
                builder.def_var(var, val);

                // Evaluate the body (which can read and update this variable)
                body.codegen_with_externals(builder, external_funcs)
            }
            Expr::If { condition, then_branch, else_branch, data_type } => {
                // Evaluate the condition
                let cond_val = condition.codegen_with_externals(builder, external_funcs);

                // Create blocks for control flow
                let then_block = builder.create_block();
                let else_block = builder.create_block();
                let merge_block = builder.create_block();

                // Add a block parameter to merge block to receive the result
                builder.append_block_param(merge_block, data_type.to_cranelift_type());

                // Branch based on condition
                builder.ins().brif(cond_val, then_block, &[], else_block, &[]);

                // Generate then branch
                builder.switch_to_block(then_block);
                builder.seal_block(then_block);
                let then_val = then_branch.codegen_with_externals(builder, external_funcs);
                builder.ins().jump(merge_block, &[then_val]);

                // Generate else branch
                builder.switch_to_block(else_block);
                builder.seal_block(else_block);
                let else_val = else_branch.codegen_with_externals(builder, external_funcs);
                builder.ins().jump(merge_block, &[else_val]);

                // Continue at merge block
                builder.switch_to_block(merge_block);
                builder.seal_block(merge_block);

                // The result is the block parameter (phi node)
                builder.block_params(merge_block)[0]
            }
            Expr::ArrayGet { array, index } => {
                let element_type = array.element_type();
                // Get the array pointer and index
                let arr_ptr = builder.use_var(array.ptr_var);
                let index_val = index.codegen_with_externals(builder, external_funcs);

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
                let index_val = index.codegen_with_externals(builder, external_funcs);
                let value_val = value.codegen_with_externals(builder, external_funcs);

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
                let new_val = value.codegen_with_externals(builder, external_funcs);

                // Update the variable
                builder.def_var(var.var, new_val);

                // SetVar returns unit, represented as 0
                builder.ins().iconst(types::I64, 0)
            }
            Expr::WhileLoop { condition, body, .. } => {
                // Create blocks for the loop
                let header_block = builder.create_block();
                let body_block = builder.create_block();
                let exit_block = builder.create_block();

                // Jump to header
                builder.ins().jump(header_block, &[]);

                // Header: evaluate condition
                // Note: Don't seal yet - we have a back-edge from the loop body!
                builder.switch_to_block(header_block);

                let cond_val = condition.codegen_with_externals(builder, external_funcs);
                builder.ins().brif(cond_val, body_block, &[], exit_block, &[]);

                // Body: execute loop body
                builder.switch_to_block(body_block);
                builder.seal_block(body_block);

                // Execute body (result is ignored)
                body.codegen_with_externals(builder, external_funcs);

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
            Expr::ExternalCall { function_name, args, return_type, .. } => {
                // Look up the FuncRef for this external function
                let func_ref = external_funcs
                    .and_then(|funcs| funcs.get(function_name))
                    .unwrap_or_else(|| {
                        panic!("External function '{}' not registered or not imported", function_name)
                    });

                // Generate code for all arguments
                let mut arg_vals = Vec::new();
                for arg in args {
                    let arg_val = arg.codegen_with_externals(builder, external_funcs);
                    arg_vals.push(arg_val);
                }

                // Call the external function
                let call_inst = builder.ins().call(*func_ref, &arg_vals);

                // Get the return value (first result of the call)
                // For void functions, we still need to return something
                if matches!(return_type, DataType::Unit) {
                    builder.ins().iconst(types::I64, 0)
                } else {
                    builder.inst_results(call_inst)[0]
                }
            }
            Expr::ExtPtr { var, .. } => {
                // External pointer is just a variable containing a pointer
                builder.use_var(*var)
            }
            Expr::LoadField { ptr, offset, field_type } => {
                // Get the base pointer
                let ptr_val = ptr.codegen_with_externals(builder, external_funcs);

                // Get the offset
                let offset_val = offset.codegen_with_externals(builder, external_funcs);

                // Calculate the field address: ptr + offset
                let field_addr = builder.ins().iadd(ptr_val, offset_val);

                // Load the field value
                let cranelift_type = field_type.to_cranelift_type();
                builder.ins().load(cranelift_type, MemFlags::trusted(), field_addr, 0)
            }
            Expr::StoreField { ptr, offset, value } => {
                // Get the base pointer
                let ptr_val = ptr.codegen_with_externals(builder, external_funcs);

                // Get the offset
                let offset_val = offset.codegen_with_externals(builder, external_funcs);

                // Calculate the field address: ptr + offset
                let field_addr = builder.ins().iadd(ptr_val, offset_val);

                // Get the value to store
                let value_val = value.codegen_with_externals(builder, external_funcs);

                // Store the value
                builder.ins().store(MemFlags::trusted(), value_val, field_addr, 0);

                // StoreField returns unit, represented as 0
                builder.ins().iconst(types::I64, 0)
            }
        }
    }
}

// =============================================================================
// CONVERSION TRAITS
// =============================================================================

impl From<StagedI64> for Expr {
    fn from(v: StagedI64) -> Self {
        Expr::Staged(Box::new(v))
    }
}

impl From<StagedU64> for Expr {
    fn from(v: StagedU64) -> Self {
        Expr::Staged(Box::new(v))
    }
}

impl From<StagedBool> for Expr {
    fn from(v: StagedBool) -> Self {
        Expr::Staged(Box::new(v))
    }
}

// =============================================================================
// DISPLAY TRAIT
// =============================================================================

impl std::fmt::Display for Expr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Expr::Staged(v) => write!(f, "{}", v),
            Expr::Variable(var) => write!(f, "v{}", var.var.as_u32()),
            Expr::Let { bindings, body, .. } => {
                writeln!(f, "{{")?;
                for (var_id, var_type, value) in bindings {
                    writeln!(f, "  {:?} v{} = {};", var_type, var_id, value)?;
                }
                writeln!(f, "  {}", body)?;
                write!(f, "}}")
            }
            Expr::LetMut { var_id, var_type, initial_value, body, .. } => {
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
            Expr::WhileLoop { condition, body, .. } => {
                write!(f, "while ({}) {{ {} }}", condition, body)
            }
            Expr::ExternalCall { function_name, args, .. } => {
                write!(f, "{}(", function_name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
            Expr::ExtPtr { var, name, .. } => {
                write!(f, "ptr<{}>_v{}", name, var.as_u32())
            }
            Expr::LoadField { ptr, offset, field_type } => {
                write!(f, "load<{:?}>({} + {})", field_type, ptr, offset)
            }
            Expr::StoreField { ptr, offset, value } => {
                write!(f, "store({} + {}, {})", ptr, offset, value)
            }
        }
    }
}

// =============================================================================
// STAGED BUILDER
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
        let data_type = body_expr.data_type().clone();

        Expr::Let {
            bindings: vec![(var_id, var_type.clone(), Box::new(value))],
            body: Box::new(body_expr),
            data_type,
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
    ///     let x = vars[0].clone().to_i64();
    ///     let y = vars[1].clone().to_i64();
    ///     // x^2 + y^2
    ///     Expr::I64((x.clone() * x.clone()) + (y.clone() * y.clone()))
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

            bindings.push((var_id, var_type.clone(), Box::new(value)));
            vars.push(var);
        }

        // Build the body
        let body_expr = body(self, &vars);
        let data_type = body_expr.data_type().clone();

        Expr::Let {
            bindings,
            body: Box::new(body_expr),
            data_type,
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
            &DataType::Bool,
            "If condition must be boolean, got {:?}",
            condition.data_type()
        );

        let then_branch = then_fn(self);
        let else_branch = else_fn(self);

        // Infer result type from then branch and clone it before moving
        let result_type = then_branch.data_type().clone();

        // Type check: branches must match
        assert_eq!(
            &result_type,
            else_branch.data_type(),
            "If branches must have same type: then={:?}, else={:?}",
            result_type,
            else_branch.data_type()
        );

        Expr::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
            data_type: result_type,
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
            data_type: DataType::Unit,
        }
    }

    /// Get element from array at index
    ///
    /// # Example
    /// ```ignore
    /// let value = builder.array_get(array, StagedU64::constant(5));
    /// ```
    pub fn array_get(&self, array: StagedArray, index: StagedU64) -> Expr {
        Expr::ArrayGet {
            array,
            index: Box::new(Expr::Staged(Box::new(index))),
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
            index: Box::new(Expr::Staged(Box::new(index))),
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
        let data_type = body_expr.data_type().clone();

        Expr::LetMut {
            var_id,
            var_type: var_type.clone(),
            initial_value: Box::new(value),
            body: Box::new(body_expr),
            data_type,
        }
    }

    /// Call an external (Rust) function with type-safe signature
    ///
    /// The signature is typically obtained from a `#[extern_fn]` annotated function's
    /// generated `{function_name}_signature()` helper.
    ///
    /// # Example
    /// ```ignore
    /// use crate::ffi::iter_next_i64_signature;
    ///
    /// let sig = iter_next_i64_signature();
    /// let option_i64 = builder.call_external(
    ///     &sig,
    ///     vec![Expr::Variable(iter_ptr_var)],
    /// );
    /// ```
    ///
    /// # Panics
    /// Panics if the number of arguments doesn't match the signature,
    /// or if any argument type doesn't match the expected parameter type.
    pub fn call_external(
        &self,
        sig: &impl crate::ffi::Signature,
        args: impl IntoIterator<Item = Expr>,
    ) -> Expr {
        let args: Vec<Expr> = args.into_iter().collect();
        let param_types = sig.param_types();

        // Validate argument count
        assert_eq!(
            args.len(),
            param_types.len(),
            "Function '{}' expects {} arguments, got {}",
            sig.name(),
            param_types.len(),
            args.len()
        );

        // Validate argument types
        for (i, (arg, expected_type)) in args.iter().zip(param_types.iter()).enumerate() {
            let actual_type = arg.data_type();
            assert_eq!(
                actual_type,
                expected_type,
                "Function '{}' argument {} type mismatch: expected {:?}, got {:?}",
                sig.name(),
                i,
                expected_type,
                actual_type
            );
        }

        Expr::ExternalCall {
            function_name: sig.name().to_string(),
            args,
            param_types: param_types.to_vec(),
            return_type: sig.return_type().clone(),
        }
    }
}

impl Default for StagedBuilder {
    fn default() -> Self {
        Self::new()
    }
}
