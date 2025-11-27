//! Staged boolean type for conditional expressions and comparisons.
//!
//! This module provides `StagedBool` which represents boolean values in staged
//! computations. It supports logical operations (AND, OR, NOT) and comparisons
//! between numeric types.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, InstBuilder, Type, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use std::ops::{BitAnd, BitOr, Not};

use crate::num::{StagedI64, StagedU64};
use crate::Staged;

/// Comparison condition for integer comparisons
#[derive(Debug, Clone, Copy)]
pub enum Condition {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    NotEqual,
    Equal,
}

impl From<Condition> for IntCC {
    fn from(cond: Condition) -> IntCC {
        match cond {
            Condition::LessThan => IntCC::SignedLessThan,
            Condition::LessThanOrEqual => IntCC::SignedLessThanOrEqual,
            Condition::GreaterThan => IntCC::SignedGreaterThan,
            Condition::GreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
            Condition::NotEqual => IntCC::NotEqual,
            Condition::Equal => IntCC::Equal,
        }
    }
}

/// A staged boolean value
///
/// Represents boolean computations that will be compiled to machine code.
/// Booleans are represented as i8 in Cranelift (0 = false, 1 = true).
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

    /// Comparison: less than (unsigned)
    LessThan(Box<StagedU64>, Box<StagedU64>),

    /// Comparison: less than or equal (unsigned)
    LessThanOrEqual(Box<StagedU64>, Box<StagedU64>),

    /// Comparison: greater than (unsigned)
    GreaterThan(Box<StagedU64>, Box<StagedU64>),

    /// Comparison: greater than or equal (unsigned)
    GreaterThanOrEqual(Box<StagedU64>, Box<StagedU64>),

    /// Comparison: equal (unsigned)
    Equal(Box<StagedU64>, Box<StagedU64>),

    /// Comparison: not equal (unsigned)
    NotEqual(Box<StagedU64>, Box<StagedU64>),

    /// Generic comparison between two staged booleans
    Cmp(Condition, Box<StagedBool>, Box<StagedBool>),

    /// Comparison between two staged i64 values
    I64Cmp(Condition, Box<StagedI64>, Box<StagedI64>),
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
                builder.use_var(*var)
            }
            StagedBool::And(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().band(left_val, right_val)
            }
            StagedBool::Or(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().bor(left_val, right_val)
            }
            StagedBool::Not(expr) => {
                let expr_val = expr.codegen(builder);
                // XOR with 1 flips the boolean
                let one = builder.ins().iconst(types::I8, 1);
                builder.ins().bxor(expr_val, one)
            }
            StagedBool::LessThan(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::UnsignedLessThan, left_val, right_val)
            }
            StagedBool::LessThanOrEqual(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::UnsignedLessThanOrEqual, left_val, right_val)
            }
            StagedBool::GreaterThan(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::UnsignedGreaterThan, left_val, right_val)
            }
            StagedBool::GreaterThanOrEqual(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::UnsignedGreaterThanOrEqual, left_val, right_val)
            }
            StagedBool::Equal(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::Equal, left_val, right_val)
            }
            StagedBool::NotEqual(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::NotEqual, left_val, right_val)
            }
            StagedBool::Cmp(cond, left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                let int_cc: IntCC = (*cond).into();
                builder.ins().icmp(int_cc, left_val, right_val)
            }
            StagedBool::I64Cmp(cond, left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                let int_cc: IntCC = (*cond).into();
                builder.ins().icmp(int_cc, left_val, right_val)
            }
        }
    }

    fn cranelift_type() -> Type {
        types::I8
    }
}

// Operator overloading for ergonomic boolean operations

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

// Display implementation for debugging
impl std::fmt::Display for StagedBool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StagedBool::Constant(val) => write!(f, "{}", val),
            StagedBool::Variable(var) => write!(f, "v{}", var.as_u32()),
            StagedBool::And(left, right) => write!(f, "({} && {})", left, right),
            StagedBool::Or(left, right) => write!(f, "({} || {})", left, right),
            StagedBool::Not(expr) => write!(f, "!{}", expr),
            StagedBool::LessThan(left, right) => write!(f, "({} < {})", left, right),
            StagedBool::LessThanOrEqual(left, right) => write!(f, "({} <= {})", left, right),
            StagedBool::GreaterThan(left, right) => write!(f, "({} > {})", left, right),
            StagedBool::GreaterThanOrEqual(left, right) => write!(f, "({} >= {})", left, right),
            StagedBool::Equal(left, right) => write!(f, "({} == {})", left, right),
            StagedBool::NotEqual(left, right) => write!(f, "({} != {})", left, right),
            StagedBool::Cmp(cond, left, right) => write!(f, "({} {:?} {})", left, cond, right),
            StagedBool::I64Cmp(cond, left, right) => write!(f, "({} {:?} {})", left, cond, right),
        }
    }
}
