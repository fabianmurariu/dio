//! Staged boolean type for conditional expressions and comparisons.
//!
//! This module provides `StagedBool` which represents boolean values in staged
//! computations. It supports logical operations (AND, OR, NOT) and comparisons
//! between numeric types.

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{types, InstBuilder, Type, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use std::ops::{BitAnd, BitOr, Not};

use crate::num::*;
use crate::Staged;

/// Comparison condition for integer and float comparisons
#[derive(Debug, Clone, Copy)]
pub enum Condition {
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
    NotEqual,
    Equal,
}

impl Condition {
    /// Convert to signed integer comparison condition
    fn to_signed_int_cc(self) -> IntCC {
        match self {
            Condition::LessThan => IntCC::SignedLessThan,
            Condition::LessThanOrEqual => IntCC::SignedLessThanOrEqual,
            Condition::GreaterThan => IntCC::SignedGreaterThan,
            Condition::GreaterThanOrEqual => IntCC::SignedGreaterThanOrEqual,
            Condition::NotEqual => IntCC::NotEqual,
            Condition::Equal => IntCC::Equal,
        }
    }

    /// Convert to unsigned integer comparison condition
    fn to_unsigned_int_cc(self) -> IntCC {
        match self {
            Condition::LessThan => IntCC::UnsignedLessThan,
            Condition::LessThanOrEqual => IntCC::UnsignedLessThanOrEqual,
            Condition::GreaterThan => IntCC::UnsignedGreaterThan,
            Condition::GreaterThanOrEqual => IntCC::UnsignedGreaterThanOrEqual,
            Condition::NotEqual => IntCC::NotEqual,
            Condition::Equal => IntCC::Equal,
        }
    }

    /// Convert to float comparison condition (ordered comparisons)
    fn to_float_cc(self) -> FloatCC {
        match self {
            Condition::LessThan => FloatCC::LessThan,
            Condition::LessThanOrEqual => FloatCC::LessThanOrEqual,
            Condition::GreaterThan => FloatCC::GreaterThan,
            Condition::GreaterThanOrEqual => FloatCC::GreaterThanOrEqual,
            Condition::NotEqual => FloatCC::NotEqual,
            Condition::Equal => FloatCC::Equal,
        }
    }
}

/// A staged boolean value
///
/// Represents boolean computations that will be compiled to machine code.
/// Booleans are represented as i8 in Cranelift (0 = false, 1 = true).
///
/// # Type-Erased Comparisons
///
/// This enum uses trait objects for comparisons (via `StagedValue`), which allows
/// new numeric types to be added without modifying this enum. This solves the
/// "StagedBool explosion" problem where each numeric type required its own variant.
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

    /// Type-erased comparison between two values
    ///
    /// This variant uses trait objects to support comparison of any two values
    /// that implement `StagedValue`, making the type system more extensible.
    /// The `operand_type` field is used during codegen to determine whether to
    /// generate signed/unsigned integer or float comparison instructions.
    Compare {
        condition: Condition,
        left: Box<dyn crate::staged_value::StagedValue>,
        right: Box<dyn crate::staged_value::StagedValue>,
        operand_type: PrimType,
    },
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
            StagedBool::Variable(var) => builder.use_var(*var),
            StagedBool::And(left, right) => {
                let left_val = Staged::codegen(left.as_ref(), builder);
                let right_val = Staged::codegen(right.as_ref(), builder);
                builder.ins().band(left_val, right_val)
            }
            StagedBool::Or(left, right) => {
                let left_val = Staged::codegen(left.as_ref(), builder);
                let right_val = Staged::codegen(right.as_ref(), builder);
                builder.ins().bor(left_val, right_val)
            }
            StagedBool::Not(expr) => {
                let expr_val = Staged::codegen(expr.as_ref(), builder);
                let one = builder.ins().iconst(types::I8, 1);
                builder.ins().bxor(expr_val, one)
            }
            // Type-erased comparison using trait objects
            StagedBool::Compare {
                condition,
                left,
                right,
                operand_type,
            } => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);

                // Dispatch to appropriate comparison instruction based on type
                if operand_type.is_signed_int() {
                    let int_cc = condition.to_signed_int_cc();
                    builder.ins().icmp(int_cc, left_val, right_val)
                } else if operand_type.is_unsigned_int() {
                    let int_cc = condition.to_unsigned_int_cc();
                    builder.ins().icmp(int_cc, left_val, right_val)
                } else {
                    // Float comparison
                    let float_cc = condition.to_float_cc();
                    builder.ins().fcmp(float_cc, left_val, right_val)
                }
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

/// Helper to format comparison operator
fn format_condition(cond: &Condition) -> &'static str {
    match cond {
        Condition::LessThan => "<",
        Condition::LessThanOrEqual => "<=",
        Condition::GreaterThan => ">",
        Condition::GreaterThanOrEqual => ">=",
        Condition::NotEqual => "!=",
        Condition::Equal => "==",
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
            // Type-erased comparison - uses Display trait on boxed values
            StagedBool::Compare {
                condition,
                left,
                right,
                operand_type: _,
            } => {
                write!(f, "({} {} {})", left, format_condition(condition), right)
            }
        }
    }
}

// =============================================================================
// STAGEDVALUE TRAIT IMPLEMENTATION
// =============================================================================

use crate::staged_value::StagedValue;
use crate::DataType;

impl StagedValue for StagedBool {
    fn data_type(&self) -> &DataType {
        &DataType::Bool
    }

    fn codegen(&self, builder: &mut FunctionBuilder) -> cranelift_codegen::ir::Value {
        // Delegate to the Staged trait implementation
        <Self as Staged>::codegen(self, builder)
    }

    fn clone_box(&self) -> Box<dyn StagedValue> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
