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
#[derive(Debug, Clone)]
pub enum StagedI64 {
    /// A constant value known at compile time
    Constant(i64),

    /// A variable (function parameter) known only at runtime
    Variable(Variable),

    /// Addition of two staged values
    Add(Box<StagedI64>, Box<StagedI64>),

    /// Subtraction of two staged values
    Sub(Box<StagedI64>, Box<StagedI64>),

    /// Multiplication of two staged values
    Mul(Box<StagedI64>, Box<StagedI64>),
}

impl StagedI64 {
    /// Create a constant staged value
    pub fn constant(value: i64) -> Self {
        StagedI64::Constant(value)
    }

    /// Create a variable staged value (represents a function parameter)
    pub fn variable(var: Variable) -> Self {
        StagedI64::Variable(var)
    }

    /// Add two staged values
    ///
    /// Note: This doesn't perform the addition! It creates a description
    /// that says "when this code runs, add these two values"
    pub fn add(left: StagedI64, right: StagedI64) -> Self {
        StagedI64::Add(Box::new(left), Box::new(right))
    }
}

impl Staged for StagedI64 {
    type RuntimeType = i64;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            StagedI64::Constant(val) => {
                // Generate: iconst.i64 <val>
                builder.ins().iconst(types::I64, *val)
            }
            StagedI64::Variable(var) => {
                // Generate: use_var <var>
                builder.use_var(*var)
            }
            StagedI64::Add(left, right) => {
                // Generate code for left and right, then add them
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                // Generate: iadd <left>, <right>
                builder.ins().iadd(left_val, right_val)
            }
            StagedI64::Sub(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().isub(left_val, right_val)
            }
            StagedI64::Mul(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().imul(left_val, right_val)
            }
        }
    }

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
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
    ///     let sum = StagedI64::add(x, y);
    ///     StagedI64::mul(sum, z)
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

/// A staged 64-bit unsigned integer
///
/// TODO: Implement this following the pattern from StagedI64
/// Remember to handle:
/// 1. Constants and Variables
/// 2. Add operation
/// 3. The Staged trait implementation
/// 4. Use unsigned instructions (iconst with type I64, but interpret as unsigned)
#[derive(Debug, Clone)]
pub enum StagedU64 {
    /// A constant value known at compile time
    Constant(u64),

    /// A variable (function parameter) known only at runtime
    Variable(Variable),

    /// Addition of two staged values
    Add(Box<StagedU64>, Box<StagedU64>),

    /// Subtraction of two staged values
    Sub(Box<StagedU64>, Box<StagedU64>),

    /// Multiplication of two staged values
    Mul(Box<StagedU64>, Box<StagedU64>),
}

// TODO: Implement methods for StagedU64 (constant, variable, add)
impl StagedU64 {
    /// Create a constant staged value
    pub fn constant(value: u64) -> Self {
        StagedU64::Constant(value)
    }

    /// Create a variable staged value (represents a function parameter)
    pub fn variable(var: Variable) -> Self {
        StagedU64::Variable(var)
    }
}

impl Add<StagedU64> for StagedU64 {
    type Output = StagedU64;

    fn add(self, rhs: StagedU64) -> StagedU64 {
        StagedU64::Add(Box::new(self), Box::new(rhs))
    }
}

impl Mul<StagedU64> for StagedU64 {
    type Output = StagedU64;

    fn mul(self, rhs: StagedU64) -> StagedU64 {
        StagedU64::Mul(Box::new(self), Box::new(rhs))
    }
}

impl Sub<StagedU64> for StagedU64 {
    type Output = StagedU64;

    fn sub(self, rhs: StagedU64) -> StagedU64 {
        StagedU64::Sub(Box::new(self), Box::new(rhs))
    }
}

impl Staged for StagedU64 {
    type RuntimeType = u64;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            StagedU64::Constant(val) => {
                // Generate: iconst.i64 <val>
                builder.ins().iconst(types::I64, *val as i64)
            }
            StagedU64::Variable(var) => {
                // Generate: use_var <var>
                builder.use_var(*var)
            }
            StagedU64::Add(left, right) => {
                // Generate code for left and right, then add them
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                // Generate: iadd <left>, <right>
                builder.ins().iadd(left_val, right_val)
            }
            StagedU64::Sub(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().isub(left_val, right_val)
            }
            StagedU64::Mul(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().imul(left_val, right_val)
            }
        }
    }

    fn cranelift_type() -> Type {
        types::I64
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DataType {
    I64,
    U64,
    Bool,
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
/// use tutorial::{Compiler, DataType, StagedI64, StagedU64, StagedValue, ScalarValue};
///
/// let mut compiler = Compiler::new().unwrap();
/// let compiled = compiler.compile_nary(
///     vec![DataType::U64, DataType::I64],
///     DataType::U64,
///     |_, vars| {
///         let x = StagedU64::variable(vars[0]);
///         let y_as_u64 = StagedU64::variable(vars[1]);
///         StagedValue::U64(x + y_as_u64)
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
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarValue {
    I64(i64),
    U64(u64),
    Bool(bool),
}

impl ScalarValue {
    /// Get the data type of this scalar value
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::I64(_) => DataType::I64,
            ScalarValue::U64(_) => DataType::U64,
            ScalarValue::Bool(_) => DataType::Bool,
        }
    }

    /// Convert this scalar value to i64 representation
    /// This is used internally for the calling convention where all
    /// 64-bit values are passed as i64
    fn as_i64(&self) -> i64 {
        match self {
            ScalarValue::I64(v) => *v,
            ScalarValue::U64(v) => *v as i64,
            ScalarValue::Bool(v) => *v as i64,
        }
    }

    /// Convert from i64 representation back to the typed value
    fn from_i64(value: i64, data_type: DataType) -> Self {
        match data_type {
            DataType::I64 => ScalarValue::I64(value),
            DataType::U64 => ScalarValue::U64(value as u64),
            DataType::Bool => ScalarValue::Bool(value != 0),
        }
    }

    /// Unwrap as i64, panics if not I64
    pub fn as_i64_unchecked(&self) -> i64 {
        match self {
            ScalarValue::I64(v) => *v,
            _ => panic!("Expected I64, got {:?}", self.data_type()),
        }
    }

    /// Unwrap as u64, panics if not U64
    pub fn as_u64_unchecked(&self) -> u64 {
        match self {
            ScalarValue::U64(v) => *v,
            _ => panic!("Expected U64, got {:?}", self.data_type()),
        }
    }

    /// Unwrap as bool, panics if not Bool
    pub fn as_bool_unchecked(&self) -> bool {
        match self {
            ScalarValue::Bool(v) => *v,
            _ => panic!("Expected Bool, got {:?}", self.data_type()),
        }
    }
}

impl DataType {
    /// Get the Cranelift type for this data type
    fn to_cranelift_type(&self) -> Type {
        match self {
            DataType::I64 => types::I64,
            DataType::U64 => types::I64, // U64 also uses I64 in Cranelift
            DataType::Bool => types::I8,  // Booleans are i8 (0 or 1)
        }
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
}

impl StagedValue {
    /// Generate Cranelift IR code for this value
    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            StagedValue::I64(v) => v.codegen(builder),
            StagedValue::U64(v) => v.codegen(builder),
        }
    }

    /// Get the runtime data type
    pub fn data_type(&self) -> DataType {
        match self {
            StagedValue::I64(_) => DataType::I64,
            StagedValue::U64(_) => DataType::U64,
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
    /// use tutorial::{Compiler, StagedI64, StagedU64, StagedValue, DataType};
    ///
    /// let mut compiler = Compiler::new().unwrap();
    /// let compiled = compiler.compile_nary(
    ///     vec![DataType::U64, DataType::I64],
    ///     DataType::U64,
    ///     |_, vars| {
    ///         let x = StagedU64::variable(vars[0]);
    ///         let y = StagedI64::variable(vars[1]);
    ///         let y_unsigned = StagedU64::variable(vars[1]); // reinterpret as U64
    ///         StagedValue::U64(x + y_unsigned)
    ///     }
    /// ).unwrap();
    /// ```
    pub fn compile_nary(
        &mut self,
        param_types: Vec<DataType>,
        return_type: DataType,
        body: impl FnOnce(&mut FunctionBuilder, &[Variable]) -> StagedValue,
    ) -> Result<CompiledNary, StagingError> {
        let num_params = param_types.len();

        // Create function signature: *const i64 -> i64
        // The function takes a pointer to an array of parameters (all 64-bit values)
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64)); // pointer to params array
        sig.returns.push(AbiParam::new(return_type.to_cranelift_type())); // return value

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
        let mut param_vars = Vec::new();
        for i in 0..num_params {
            let var = Variable::from_u32(i as u32);
            let cranelift_type = param_types[i].to_cranelift_type();
            builder.declare_var(var, cranelift_type);

            // Load params[i]: compute address = params_ptr + (i * 8)
            let offset = builder.ins().iconst(types::I64, (i * 8) as i64);
            let param_addr = builder.ins().iadd(params_ptr, offset);
            let param_val = builder
                .ins()
                .load(cranelift_type, MemFlags::trusted(), param_addr, 0);

            // Assign to variable
            builder.def_var(var, param_val);
            param_vars.push(var);
        }

        // Generate the function body
        let result_expr = body(&mut builder, &param_vars);

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

        Ok(CompiledNary {
            code_ptr,
            param_types,
            return_type,
        })
    }
}

/// A compiled generic n-ary function
///
/// This is similar to how dio3/dio4 handle compiled functions with type information
/// preserved for proper execution.
pub struct CompiledNary {
    code_ptr: *const u8,
    param_types: Vec<DataType>,
    return_type: DataType,
}

impl CompiledNary {
    /// Execute the compiled function with type-safe ScalarValue arguments
    ///
    /// This is the primary calling interface, similar to how dio4's execute()
    /// takes ArrayRef and extracts raw pointers. Here we take ScalarValues,
    /// perform type checking, extract raw values, and call the compiled function.
    ///
    /// # Example
    ///
    /// ```
    /// use tutorial::{Compiler, DataType, StagedU64, StagedValue, ScalarValue};
    ///
    /// let mut compiler = Compiler::new().unwrap();
    /// let compiled = compiler.compile_nary(
    ///     vec![DataType::U64, DataType::U64],
    ///     DataType::U64,
    ///     |_, vars| {
    ///         let x = StagedU64::variable(vars[0]);
    ///         let y = StagedU64::variable(vars[1]);
    ///         StagedValue::U64(x + y)
    ///     }
    /// ).unwrap();
    ///
    /// let result = compiled.call(&[
    ///     ScalarValue::U64(10),
    ///     ScalarValue::U64(20)
    /// ]).unwrap();
    ///
    /// assert_eq!(result, ScalarValue::U64(30));
    /// ```
    pub fn call(&self, args: &[ScalarValue]) -> Result<ScalarValue, StagingError> {
        // Verify argument count
        if args.len() != self.param_types.len() {
            return Err(StagingError::ExecutionFailed {
                reason: format!(
                    "Expected {} arguments, got {}",
                    self.param_types.len(),
                    args.len()
                ),
            });
        }

        // Verify argument types match expected parameter types
        for (i, (arg, expected_type)) in args.iter().zip(&self.param_types).enumerate() {
            if arg.data_type() != *expected_type {
                return Err(StagingError::TypeMismatch {
                    expected: format!("argument {} type {:?}", i, expected_type),
                    actual: format!("got {:?}", arg.data_type()),
                });
            }
        }

        // Convert all args to i64 representation for calling
        // (all 64-bit values use the same calling convention)
        let i64_args: Vec<i64> = args.iter().map(|a| a.as_i64()).collect();

        // Call the function using the raw i64 calling convention
        let result_i64 = self.call_i64(&i64_args);

        // Convert result back to the typed ScalarValue based on return type
        Ok(ScalarValue::from_i64(result_i64, self.return_type))
    }

    /// Execute the compiled function with i64 arguments (low-level interface)
    ///
    /// Similar to how dio3/dio4's call_nary_op works, we pass a pointer to
    /// an array of parameters. The types are reinterpreted based on param_types.
    ///
    /// # Safety
    /// The caller must ensure that `args.len() >= param_types.len()`
    pub fn call_i64(&self, args: &[i64]) -> i64 {
        assert!(
            args.len() >= self.param_types.len(),
            "Expected at least {} arguments, got {}",
            self.param_types.len(),
            args.len()
        );

        unsafe {
            let func: extern "C" fn(*const i64) -> i64 = std::mem::transmute(self.code_ptr);
            func(args.as_ptr())
        }
    }

    /// Execute the compiled function with u64 arguments (low-level interface)
    ///
    /// # Safety
    /// The caller must ensure that `args.len() >= param_types.len()`
    pub fn call_u64(&self, args: &[u64]) -> u64 {
        assert!(
            args.len() >= self.param_types.len(),
            "Expected at least {} arguments, got {}",
            self.param_types.len(),
            args.len()
        );

        unsafe {
            let func: extern "C" fn(*const u64) -> u64 = std::mem::transmute(self.code_ptr);
            func(args.as_ptr())
        }
    }

    /// Execute with mixed i64/u64 arguments based on parameter types (low-level interface)
    ///
    /// This is the most flexible calling convention - values are passed
    /// as i64 but interpreted according to their declared types.
    pub fn call_mixed(&self, args: &[i64]) -> i64 {
        self.call_i64(args)
    }

    /// Get the parameter types
    pub fn param_types(&self) -> &[DataType] {
        &self.param_types
    }

    /// Get the return type
    pub fn return_type(&self) -> DataType {
        self.return_type
    }
}

// =============================================================================
// LESSON 5: BOOLEAN OPERATIONS (EXERCISE - YOU COMPLETE)
// =============================================================================
//
// Booleans are essential for control flow (if/then/else) and comparisons.
// In Cranelift, booleans are represented as i8 (0 = false, 1 = true).
//
// YOUR TASK: Implement StagedBool with comparison and logical operations

/// A staged boolean value
///
/// TODO: Implement this with support for:
/// 1. Constants (true/false)
/// 2. Comparisons (less than, greater than, equal)
/// 3. Logical operations (and, or, not)
#[derive(Debug, Clone)]
pub enum StagedBool {
    /// A constant boolean value
    Constant(bool),

    /// A variable (function parameter) known only at runtime
    Variable(Variable),

    /// Logical AND of two staged booleans
    And(Box<StagedBool>, Box<StagedBool>),

    /// Logical OR of two staged booleans
    Or(Box<StagedBool>, Box<StagedBool>),

    /// Logical NOT of a staged boolean
    Not(Box<StagedBool>),
}

impl StagedBool {
    /// Create a constant staged boolean
    pub fn constant(value: bool) -> Self {
        StagedBool::Constant(value)
    }

    /// Create a variable staged boolean (represents a function parameter)
    pub fn variable(var: Variable) -> Self {
        StagedBool::Variable(var)
    }
}

impl Staged for StagedBool {
    type RuntimeType = bool;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            StagedBool::Constant(val) => {
                let int_val = if *val { 1 } else { 0 };
                builder.ins().iconst(types::I8, int_val)
            }
            StagedBool::Variable(var) => {
                // Generate: use_var <var>
                builder.use_var(*var)
            }
            StagedBool::And(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                // Generate: and <left>, <right>
                builder.ins().band(left_val, right_val)
            }
            StagedBool::Or(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                // Generate: or <left>, <right>
                builder.ins().bor(left_val, right_val)
            }
            StagedBool::Not(expr) => {
                let expr_val = expr.codegen(builder);
                // Generate: xor <expr>, 1 (to flip the boolean)
                let one = builder.ins().iconst(types::I8, 1);
                builder.ins().bxor(expr_val, one)
            }
        }
    }

    fn cranelift_type() -> Type {
        types::I8
    }
}

impl BitAnd for StagedBool {
    type Output = StagedBool;

    fn bitand(self, rhs: Self) -> Self::Output {
        StagedBool::And(Box::new(self), Box::new(rhs))
    }
}

impl BitOr for StagedBool {
    type Output = StagedBool;

    fn bitor(self, rhs: Self) -> Self::Output {
        StagedBool::Or(Box::new(self), Box::new(rhs))
    }
}

impl Not for StagedBool {
    type Output = StagedBool;

    fn not(self) -> Self::Output {
        StagedBool::Not(Box::new(self))
    }
}

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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
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
        let compiled = compiler
            .compile_nary(vec![DataType::U64],DataType::U64, |_, param| {
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
    #[should_panic(expected = "not yet implemented")]
    fn test_lesson5_less_than_comparison() {
        // TODO: Implement StagedBool to make this test pass
        // This should compile: f(x) = (x < 10) ? 1 : 0

        // Uncomment this code once you've implemented StagedBool:
        /*
        let mut compiler = Compiler::new().unwrap();
        // You'll need to create compile_unary_i64_to_bool
        let compiled = compiler
            .compile_unary_i64_to_bool(|builder, param| {
                let x = StagedI64::variable(param);
                let ten = StagedI64::constant(10);
                StagedBool::less_than(x, ten)
            })
            .unwrap();

        assert_eq!(compiled.call(5), true);
        assert_eq!(compiled.call(10), false);
        assert_eq!(compiled.call(15), false);
        */

        todo!("Implement StagedBool and comparison operations");
    }

    // -------------------------------------------------------------------------
    // GENERIC COMPILATION TESTS: Testing compile_nary with mixed types
    // -------------------------------------------------------------------------

    #[test]
    fn test_generic_u64_addition() {
        // Compile: f(x: u64) = x + 10
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64],
                DataType::U64,
                |_, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let ten = StagedU64::constant(10);
                    StagedValue::U64(x + ten)
                },
            )
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::I64],
                DataType::U64,
                |_, vars| {
                    let x = StagedU64::variable(vars[0]);
                    // Reinterpret vars[1] as U64 (they're both 64-bit values)
                    let y_as_u64 = StagedU64::variable(vars[1]);
                    StagedValue::U64(x + y_as_u64)
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::I64, DataType::I64, DataType::I64],
                DataType::I64,
                |_, vars| {
                    let a = StagedI64::variable(vars[0]);
                    let b = StagedI64::variable(vars[1]);
                    let c = StagedI64::variable(vars[2]);
                    let sum = StagedI64::add(a, b);
                    StagedValue::I64(StagedI64::mul(sum, c))
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::U64,
                |_, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    StagedValue::U64(x * y)
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
            |_, vars| {
                let x = StagedI64::variable(vars[0]);
                StagedValue::I64(x) // But returning I64!
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
        let compiled = compiler
            .compile_nary(vec![], DataType::U64, |_, _vars| {
                StagedValue::U64(StagedU64::constant(42))
            })
            .unwrap();

        assert_eq!(compiled.call_u64(&[]), 42);
        assert_eq!(compiled.call_u64(&[999, 888]), 42); // Extra args ignored
    }

    #[test]
    fn test_generic_complex_expression() {
        // Compile: f(a: u64, b: u64, c: u64) -> u64 = (a * b) + (c * 2)
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64, DataType::U64],
                DataType::U64,
                |_, vars| {
                    let a = StagedU64::variable(vars[0]);
                    let b = StagedU64::variable(vars[1]);
                    let c = StagedU64::variable(vars[2]);
                    let two = StagedU64::constant(2);
                    let ab = a * b;
                    let c2 = c * two;
                    StagedValue::U64(ab + c2)
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::U64,
                |_, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    StagedValue::U64(x + y)
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::I64, DataType::U64],
                DataType::U64,
                |_, vars| {
                    let a = StagedU64::variable(vars[0]);
                    let b_as_u64 = StagedU64::variable(vars[1]); // Reinterpret i64 as u64
                    let c = StagedU64::variable(vars[2]);
                    // (a + b) * c
                    let sum = a + b_as_u64;
                    StagedValue::U64(sum * c)
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::U64,
                |_, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    StagedValue::U64(x + y)
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::U64,
                |_, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    StagedValue::U64(x + y)
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::I64, DataType::I64],
                DataType::I64,
                |_, vars| {
                    let x = StagedI64::variable(vars[0]);
                    let y = StagedI64::variable(vars[1]);
                    StagedValue::I64(StagedI64::sub(x, y))
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
        let compiled = compiler
            .compile_nary(
                vec![DataType::U64, DataType::U64, DataType::U64],
                DataType::U64,
                |_, vars| {
                    let a = StagedU64::variable(vars[0]);
                    let b = StagedU64::variable(vars[1]);
                    let c = StagedU64::variable(vars[2]);
                    let ten = StagedU64::constant(10);
                    // ((a + b) * c) + 10
                    let sum = a + b;
                    let product = sum * c;
                    StagedValue::U64(product + ten)
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
