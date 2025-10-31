//! Example: Generic Rep<T> abstraction (Scala LMS style)
//!
//! This shows how to implement a generic staged computation type
//! similar to Scala LMS's Rep[T], where operations are implemented
//! via traits and conditional compilation.

use cranelift_codegen::ir::{types, InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use std::ops::{Add, Mul, Sub};

// =============================================================================
// Core Abstraction: Rep<T> - A staged computation producing type T
// =============================================================================

/// Represents a staged computation that will produce a value of type T at runtime
#[derive(Clone)]
pub enum Rep<T: Staged> {
    /// A constant value known at compile time
    Constant(T::RuntimeValue),
    /// A variable (function parameter) known only at runtime
    Variable(Variable),
    /// A binary operation on two staged values
    BinOp(Box<Rep<T>>, BinOpKind, Box<Rep<T>>),
}

#[derive(Clone, Debug)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

// =============================================================================
// Trait System: What types can be staged?
// =============================================================================

/// Core trait: Types that can participate in staged computation
pub trait Staged: 'static + Clone {
    /// The actual runtime type (e.g., i64 for I64Type)
    type RuntimeValue: Clone;

    /// Get the Cranelift type representation
    fn cranelift_type() -> cranelift_codegen::ir::Type;

    /// Generate code for a constant value
    fn codegen_constant(value: &Self::RuntimeValue, builder: &mut FunctionBuilder) -> Value;
}

/// Extended trait: Types that support binary operations
pub trait SupportsBinOp: Staged {
    /// Generate code for a binary operation
    fn codegen_binop(
        kind: BinOpKind,
        left: Value,
        right: Value,
        builder: &mut FunctionBuilder,
    ) -> Value;
}

// =============================================================================
// Phantom Types: Marker types for different staged types
// =============================================================================

/// Marker type for i64 values
#[derive(Clone)]
pub struct I64Type;

/// Marker type for u64 values
#[derive(Clone)]
pub struct U64Type;

/// Marker type for boolean values
#[derive(Clone)]
pub struct BoolType;

// =============================================================================
// Implement Staged for concrete types
// =============================================================================

impl Staged for I64Type {
    type RuntimeValue = i64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn codegen_constant(value: &i64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value)
    }
}

impl SupportsBinOp for I64Type {
    fn codegen_binop(
        kind: BinOpKind,
        left: Value,
        right: Value,
        builder: &mut FunctionBuilder,
    ) -> Value {
        match kind {
            BinOpKind::Add => builder.ins().iadd(left, right),
            BinOpKind::Sub => builder.ins().isub(left, right),
            BinOpKind::Mul => builder.ins().imul(left, right),
            BinOpKind::Div => builder.ins().sdiv(left, right),
        }
    }
}

impl Staged for U64Type {
    type RuntimeValue = u64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn codegen_constant(value: &u64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value as i64)
    }
}

impl SupportsBinOp for U64Type {
    fn codegen_binop(
        kind: BinOpKind,
        left: Value,
        right: Value,
        builder: &mut FunctionBuilder,
    ) -> Value {
        match kind {
            BinOpKind::Add => builder.ins().iadd(left, right),
            BinOpKind::Sub => builder.ins().isub(left, right),
            BinOpKind::Mul => builder.ins().imul(left, right),
            BinOpKind::Div => builder.ins().udiv(left, right), // Note: unsigned division
        }
    }
}

impl Staged for BoolType {
    type RuntimeValue = bool;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I8
    }

    fn codegen_constant(value: &bool, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, if *value { 1 } else { 0 })
    }
}

// =============================================================================
// Operator Overloading: Make Rep<T> work with +, -, *, etc.
// =============================================================================

impl<T: SupportsBinOp> Add for Rep<T> {
    type Output = Rep<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Add, Box::new(rhs))
    }
}

impl<T: SupportsBinOp> Sub for Rep<T> {
    type Output = Rep<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Sub, Box::new(rhs))
    }
}

impl<T: SupportsBinOp> Mul for Rep<T> {
    type Output = Rep<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Mul, Box::new(rhs))
    }
}

// =============================================================================
// Helper Methods for Rep<T>
// =============================================================================

impl<T: Staged> Rep<T> {
    /// Create a constant staged value
    pub fn constant(value: T::RuntimeValue) -> Self {
        Rep::Constant(value)
    }

    /// Create a variable staged value (represents a function parameter)
    pub fn variable(var: Variable) -> Self {
        Rep::Variable(var)
    }
}

impl<T: SupportsBinOp> Rep<T> {
    /// Generate Cranelift IR code for this staged computation
    pub fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            Rep::Constant(val) => T::codegen_constant(val, builder),
            Rep::Variable(var) => builder.use_var(*var),
            Rep::BinOp(left, op, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                T::codegen_binop(op.clone(), left_val, right_val, builder)
            }
        }
    }
}

// =============================================================================
// Type Aliases for Convenience
// =============================================================================

/// Staged i64 value
pub type RepI64 = Rep<I64Type>;

/// Staged u64 value
pub type RepU64 = Rep<U64Type>;

/// Staged boolean value
pub type RepBool = Rep<BoolType>;

// =============================================================================
// Example Usage
// =============================================================================

fn main() {
    // Example 1: Using RepI64 with operator overloading
    // This would compile to: f(x) = (x + 5) * 2
    fn example_expression(x: RepI64) -> RepI64 {
        let five = RepI64::constant(5);
        let two = RepI64::constant(2);
        (x + five) * two // Natural operator syntax!
    }

    // Example 2: Generic function that works with any staged type
    fn square<T: SupportsBinOp>(x: Rep<T>) -> Rep<T> {
        x.clone() * x
    }

    // Example 3: Type-safe operations
    let x = RepI64::variable(Variable::from_u32(0));
    let y = RepI64::constant(10);
    let result = x + y; // This compiles!

    // This would NOT compile (type mismatch):
    // let z = RepBool::constant(true);
    // let invalid = x + z; // ERROR: cannot add RepI64 and RepBool

    println!("Rep<T> abstraction demonstrates Scala LMS-style staging in Rust!");
    println!("Key features:");
    println!("  - Generic over types via phantom types");
    println!("  - Operator overloading (+, -, *) via trait implementations");
    println!("  - Type-safe: operations only work when supported");
    println!("  - Extensible: add new types by implementing Staged + SupportsBinOp");
}