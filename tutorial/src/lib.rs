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
    types, AbiParam, FuncRef, Function, InstBuilder, MemFlags, Signature, Type, Value,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::collections::HashMap;
use std::ops::{Add, BitAnd, BitOr, Mul, Not, Sub};
use thiserror::Error;

// =============================================================================
// MODULE DECLARATIONS
// =============================================================================

/// Trait abstraction for staged values
pub mod staged_value;

/// Generic numeric types supporting all primitive numeric types
pub mod num;

/// Boolean types and comparison operations
pub mod bool;

/// Expression AST and builder for staged computations
pub mod expr;

/// Runtime execution components (ScalarValue, CompiledNary)
pub mod runtime;

/// Foreign Function Interface for calling Rust functions from JIT code
pub mod ffi;

/// Code generation and JIT compilation infrastructure
pub mod codegen;

// =============================================================================
// MACROS
// =============================================================================

/// Macro to register multiple external function symbols with CompilerBuilder
///
/// This macro automatically extracts function names and pointers, making it easy
/// to register many functions at once (useful for standard libraries).
///
/// # Example
/// ```ignore
/// use tutorial::{CompilerBuilder, register_symbols};
/// use tutorial::ffi::*;
///
/// let compiler = CompilerBuilder::new()
///     .register_symbols!(iter_create_range, iter_next_i64, iter_drop)
///     .build()?;
/// ```
#[macro_export]
macro_rules! register_symbols {
    ($builder:expr, $($fn_name:ident),+ $(,)?) => {
        {
            let mut builder = $builder;
            $(
                builder = builder.with_symbol(stringify!($fn_name), $fn_name as *const u8);
            )+
            builder
        }
    };
}

// Re-export commonly used types
pub use num::{
    Numeric, PrimType,
    StagedNum, StagedI8, StagedU8, StagedI16, StagedU16,
    StagedI32, StagedU32, StagedI64, StagedU64,
    StagedF32, StagedF64,
};

pub use bool::{Condition, StagedBool};
pub use codegen::{Compiler, CompilerBuilder};
pub use expr::{Expr, StagedBuilder, Var};
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

    /// External pointer (opaque pointer to Rust data)
    /// The String describes what it points to (for debugging/error messages)
    ExtPtr(String),

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
            DataType::ExtPtr(_) => types::I64, // External pointers are i64
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
    use std::ops::Add;
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

        assert_eq!(compiled.call_i64(&[10]), 15);
        assert_eq!(compiled.call_i64(&[0]), 5);
        assert_eq!(compiled.call_i64(&[-3]), 2);
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

        assert_eq!(compiled.call_i64(&[10]), 20);
        assert_eq!(compiled.call_i64(&[7]), 14);
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

        assert_eq!(compiled.call_i64(&[10]), 13);
        assert_eq!(compiled.call_i64(&[0]), 3);
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

        assert_eq!(compiled.call_i64(&[10, 5]), 15);
        assert_eq!(compiled.call_i64(&[100, 200]), 300);
        assert_eq!(compiled.call_i64(&[-3, 8]), 5);
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
        assert_eq!(compiled.call_i64(&[2, 3, 4]), 20);
        // (10 + 5) * 2 = 30
        assert_eq!(compiled.call_i64(&[10, 5, 2]), 30);
        // (1 + 1) * 100 = 200
        assert_eq!(compiled.call_i64(&[1, 1, 100]), 200);
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
        assert_eq!(compiled.call_i64(&[5, 2]), 30);
        // (0 + 10) * 3 = 30
        assert_eq!(compiled.call_i64(&[0, 3]), 30);
        // (90 + 10) * 1 = 100
        assert_eq!(compiled.call_i64(&[90, 1]), 100);
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
        assert_eq!(compiled.call_i64(&[1, 2, 10, 5]), 15);
        // (10 + 20) * (100 - 50) = 30 * 50 = 1500
        assert_eq!(compiled.call_i64(&[10, 20, 100, 50]), 1500);
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
        assert_eq!(compiled.call_i64(&[]), 100);
        assert_eq!(compiled.call_i64(&[999, 888]), 100); // extra args ignored
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

        assert_eq!(compiled.call_i64(&[10]), 7);
        assert_eq!(compiled.call_i64(&[5]), 2);
        assert_eq!(compiled.call_i64(&[0]), -3);
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
        assert_eq!(compiled.call_i64(&[]), 58);
        assert_eq!(compiled.call_i64(&[999]), 58);
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

        assert_eq!(compiled.call_i64(&[10]), 20);
        assert_eq!(compiled.call_i64(&[7]), 14);
        assert_eq!(compiled.call_i64(&[-3]), -6);
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
        assert_eq!(compiled.call_i64(&[3]), 8);
        // When x = 4: (4 + 5) * (4 - 2) = 9 * 2 = 18
        assert_eq!(compiled.call_i64(&[4]), 18);
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
                (x + ten)
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
                x.lt(&ten)
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
                let cond = x.lt(&ten);
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
                let cond = x.lt(&zero);
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
                let cond = x.lt(&zero);
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

                let cond = x.lt(&zero);
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
                                    Expr::Bool(i.lt(&len))
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

    // -------------------------------------------------------------------------
    // PROPERTY-BASED TESTS FOR COMPARISONS
    // -------------------------------------------------------------------------

    use proptest::prelude::*;

    // Property: Compiled comparisons should match Rust's native comparisons
    proptest! {
        #[test]
        fn prop_i64_lt_matches_native(x in -1000i64..1000, y in -1000i64..1000) {
            let mut compiler = Compiler::new().unwrap();
            let mut compiled = compiler.compile_nary(
                vec![DataType::I64, DataType::I64],
                DataType::Bool,
                |_, vars| {
                    let x = StagedI64::variable(vars[0]);
                    let y = StagedI64::variable(vars[1]);
                    x.lt(&y)
                }
            ).unwrap();

            let result = compiled.call(&[ScalarValue::I64(x), ScalarValue::I64(y)])
                .unwrap()
                .as_bool_unchecked();
            assert_eq!(result, x < y, "Compiled x < y should match native for x={}, y={}", x, y);
        }

        #[test]
        fn prop_u64_lt_matches_native(x in 0u64..1000, y in 0u64..1000) {
            let mut compiler = Compiler::new().unwrap();
            let mut compiled = compiler.compile_nary(
                vec![DataType::U64, DataType::U64],
                DataType::Bool,
                |_, vars| {
                    let x = StagedU64::variable(vars[0]);
                    let y = StagedU64::variable(vars[1]);
                    x.lt(&y)
                }
            ).unwrap();

            let result = compiled.call(&[ScalarValue::U64(x), ScalarValue::U64(y)])
                .unwrap()
                .as_bool_unchecked();
            assert_eq!(result, x < y, "Compiled x < y should match native for x={}, y={}", x, y);
        }

        #[test]
        fn prop_i64_eq_matches_native(x in -1000i64..1000, y in -1000i64..1000) {
            let mut compiler = Compiler::new().unwrap();
            let mut compiled = compiler.compile_nary(
                vec![DataType::I64, DataType::I64],
                DataType::Bool,
                |_, vars| {
                    let x = StagedI64::variable(vars[0]);
                    let y = StagedI64::variable(vars[1]);
                    x.eq(&y)
                }
            ).unwrap();

            let result = compiled.call(&[ScalarValue::I64(x), ScalarValue::I64(y)])
                .unwrap()
                .as_bool_unchecked();
            assert_eq!(result, x == y, "Compiled x == y should match native for x={}, y={}", x, y);
        }
    }

    // Property: Reflexivity - x == x should always be true
    proptest! {
        #[test]
        fn prop_i64_eq_reflexive(x in -1000i64..1000) {
            let mut compiler = Compiler::new().unwrap();
            let mut compiled = compiler.compile_nary(
                vec![DataType::I64],
                DataType::Bool,
                |_, vars| {
                    let x1 = StagedI64::variable(vars[0]);
                    let x2 = StagedI64::variable(vars[0]);
                    x1.eq(&x2)
                }
            ).unwrap();

            let result = compiled.call(&[ScalarValue::I64(x)])
                .unwrap()
                .as_bool_unchecked();
            assert!(result, "x == x should always be true for x={}", x);
        }

        #[test]
        fn prop_u64_eq_reflexive(x in 0u64..1000) {
            let mut compiler = Compiler::new().unwrap();
            let mut compiled = compiler.compile_nary(
                vec![DataType::U64],
                DataType::Bool,
                |_, vars| {
                    let x1 = StagedU64::variable(vars[0]);
                    let x2 = StagedU64::variable(vars[0]);
                    x1.eq(&x2)
                }
            ).unwrap();

            let result = compiled.call(&[ScalarValue::U64(x)])
                .unwrap()
                .as_bool_unchecked();
            assert!(result, "x == x should always be true for x={}", x);
        }
    }

    // Property: Trichotomy - exactly one of x < y, x == y, x > y is true
    proptest! {
        #[test]
        fn prop_i64_trichotomy(x in -1000i64..1000, y in -1000i64..1000) {
            // Use separate compiler instances to avoid function name conflicts
            let mut compiler_lt = Compiler::new().unwrap();
            let mut lt_compiled = compiler_lt.compile_nary(
                vec![DataType::I64, DataType::I64],
                DataType::Bool,
                |_, vars| {
                    let x = StagedI64::variable(vars[0]);
                    let y = StagedI64::variable(vars[1]);
                    x.lt(&y)
                }
            ).unwrap();

            let mut compiler_eq = Compiler::new().unwrap();
            let mut eq_compiled = compiler_eq.compile_nary(
                vec![DataType::I64, DataType::I64],
                DataType::Bool,
                |_, vars| {
                    let x = StagedI64::variable(vars[0]);
                    let y = StagedI64::variable(vars[1]);
                    x.eq(&y)
                }
            ).unwrap();

            let mut compiler_gt = Compiler::new().unwrap();
            let mut gt_compiled = compiler_gt.compile_nary(
                vec![DataType::I64, DataType::I64],
                DataType::Bool,
                |_, vars| {
                    let x = StagedI64::variable(vars[0]);
                    let y = StagedI64::variable(vars[1]);
                    x.gt(&y)
                }
            ).unwrap();

            let lt = lt_compiled.call(&[ScalarValue::I64(x), ScalarValue::I64(y)])
                .unwrap()
                .as_bool_unchecked();
            let eq = eq_compiled.call(&[ScalarValue::I64(x), ScalarValue::I64(y)])
                .unwrap()
                .as_bool_unchecked();
            let gt = gt_compiled.call(&[ScalarValue::I64(x), ScalarValue::I64(y)])
                .unwrap()
                .as_bool_unchecked();

            let count = (lt as u8) + (eq as u8) + (gt as u8);
            assert_eq!(count, 1, "Exactly one of x < y, x == y, x > y should be true for x={}, y={}", x, y);
        }
    }

    // -------------------------------------------------------------------------
    // FFI TESTS - External function calls and iterators
    // -------------------------------------------------------------------------

    use crate::ffi::*;

    #[test]
    fn test_ffi_option_i64() {
        // Test OptionI64 struct
        let some_val = OptionI64::some(42);
        assert_eq!(some_val.has_value, 1);
        assert_eq!(some_val.value, 42);
        assert_eq!(some_val.to_option(), Some(42));

        let none_val = OptionI64::none();
        assert_eq!(none_val.has_value, 0);
        assert_eq!(none_val.to_option(), None);
    }

    #[test]
    fn test_ffi_iterator_basic() {
        // Create an iterator from 0..5
        let iter_ptr = iter_create_range(0, 5);

        // Iterate and collect values
        let mut values = Vec::new();
        loop {
            let option = iter_next_i64(iter_ptr);
            match option.to_option() {
                Some(val) => values.push(val),
                None => break,
            }
        }

        assert_eq!(values, vec![0, 1, 2, 3, 4]);

        // Clean up
        iter_drop(iter_ptr);
    }

    #[test]
    fn test_ffi_external_function_registry() {
        let mut registry = ExternalFunctionRegistry::new();

        // Register iter_next_i64
        registry.register(
            "iter_next_i64".to_string(),
            vec![DataType::ExtPtr("IterBoxI64".to_string())],
            DataType::ExtPtr("OptionI64".to_string()),
            iter_next_i64 as *const u8,
        );

        assert!(registry.has_function("iter_next_i64"));
        let sig = registry.get_signature("iter_next_i64").unwrap();
        assert_eq!(sig.name, "iter_next_i64");
        assert_eq!(sig.params.len(), 1);
    }

    #[test]
    fn test_ffi_actual_external_call() {
        // This test actually compiles and EXECUTES code that calls an external function
        // Using Option 1C: CompilerBuilder with symbol pre-registration

        // Define the external function
        extern "C" fn add_42(x: i64) -> i64 {
            x + 42
        }

        // Step 1: Register symbol with CompilerBuilder BEFORE creating JIT module
        let mut compiler = CompilerBuilder::new()
            .with_symbol("add_42", add_42 as *const u8)
            .build()
            .unwrap();

        // Step 2: Register signature for type checking
        compiler.register_external_signature(
            "add_42",
            vec![DataType::I64],
            DataType::I64,
        );

        // Step 3: Compile code that calls this external function
        let compile_result = compiler.compile_nary(
            vec![DataType::I64],
            DataType::I64,
            |builder, vars| {
                builder.call_external(
                    "add_42",
                    vec![Expr::I64(StagedI64::variable(vars[0]))],
                    DataType::I64,
                )
            },
        );

        // Step 4: Verify compilation and execution work
        match compile_result {
            Ok(mut compiled) => {
                println!("✓ Compilation succeeded!");

                // Now try to actually execute it
                let exec_result = compiled.call(&[ScalarValue::I64(10)]);
                match exec_result {
                    Ok(result) => {
                        println!("✓ Execution succeeded!");
                        println!("  Result: {:?}", result);
                        assert_eq!(result.as_i64_unchecked(), 52);
                    }
                    Err(e) => {
                        println!("✗ Execution failed: {:?}", e);
                        panic!("Execution failed: {:?}", e);
                    }
                }
            }
            Err(e) => {
                println!("✗ Compilation failed: {:?}", e);
                panic!("Compilation failed: {:?}", e);
            }
        }
    }

    #[test]
    fn test_ffi_register_symbols_macro() {
        // Demonstrate using the register_symbols! macro for convenient multi-function registration
        // This is perfect for building a standard library!

        extern "C" fn add(x: i64, y: i64) -> i64 {
            x + y
        }

        extern "C" fn mul(x: i64, y: i64) -> i64 {
            x * y
        }

        extern "C" fn neg(x: i64) -> i64 {
            -x
        }

        // Use the macro to register all functions at once
        let builder = CompilerBuilder::new();
        let builder = register_symbols!(builder, add, mul, neg);
        let mut compiler = builder.build().unwrap();

        // Register signatures for type checking
        compiler.register_external_signature("add", vec![DataType::I64, DataType::I64], DataType::I64);
        compiler.register_external_signature("mul", vec![DataType::I64, DataType::I64], DataType::I64);
        compiler.register_external_signature("neg", vec![DataType::I64], DataType::I64);

        // Compile a simple expression that uses multiple external functions
        // Just test that we can call all three registered functions
        let mut compiled = compiler.compile_nary(
            vec![DataType::I64, DataType::I64],
            DataType::I64,
            |builder, vars| {
                // Call add(x, y)
                let x = Expr::I64(StagedI64::variable(vars[0]));
                let y = Expr::I64(StagedI64::variable(vars[1]));
                builder.call_external("add", vec![x, y], DataType::I64)
            },
        ).unwrap();

        // Test it works
        let result = compiled.call(&[ScalarValue::I64(10), ScalarValue::I64(5)]).unwrap();
        assert_eq!(result.as_i64_unchecked(), 15);

        println!("✓ Multi-function FFI registration with macro works!");
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
