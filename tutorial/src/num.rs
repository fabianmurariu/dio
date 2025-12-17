//! Generic staged numeric types supporting all primitive numeric types.
//!
//! This module provides `StagedNum<T>` which works with i8, u8, i16, u16, i32, u32,
//! i64, u64, f32, and f64. It generates optimized Cranelift IR for each type.
//! Casting between types is handled at the Expr level, not here.

use cranelift_codegen::ir::{types, InstBuilder, Type, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use std::fmt;
use std::ops::{Add, Mul, Sub};

// =============================================================================
// PRIMITIVE TYPE SYSTEM
// =============================================================================

/// Primitive numeric types supported by staged compilation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum PrimType {
    I8 = 0,
    U8 = 1,
    I16 = 2,
    U16 = 3,
    I32 = 4,
    U32 = 5,
    I64 = 6,
    U64 = 7,
    F32 = 8,
    F64 = 9,
}

impl PrimType {
    /// Get the Cranelift IR type for this primitive type
    pub fn to_cranelift_type(self) -> Type {
        match self {
            PrimType::I8 => types::I8,
            PrimType::U8 => types::I8,
            PrimType::I16 => types::I16,
            PrimType::U16 => types::I16,
            PrimType::I32 => types::I32,
            PrimType::U32 => types::I32,
            PrimType::I64 => types::I64,
            PrimType::U64 => types::I64,
            PrimType::F32 => types::F32,
            PrimType::F64 => types::F64,
        }
    }

    pub fn as_index(self) -> usize {
        self as usize
    }

    /// Check if this is a signed integer type
    pub fn is_signed_int(self) -> bool {
        matches!(self, PrimType::I8 | PrimType::I16 | PrimType::I32 | PrimType::I64)
    }

    /// Check if this is an unsigned integer type
    pub fn is_unsigned_int(self) -> bool {
        matches!(self, PrimType::U8 | PrimType::U16 | PrimType::U32 | PrimType::U64)
    }

    /// Check if this is an integer type (signed or unsigned)
    pub fn is_int(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Check if this is a floating point type
    pub fn is_float(self) -> bool {
        matches!(self, PrimType::F32 | PrimType::F64)
    }

    /// Get the bit width of this type
    pub fn bit_width(self) -> u32 {
        match self {
            PrimType::I8 | PrimType::U8 => 8,
            PrimType::I16 | PrimType::U16 => 16,
            PrimType::I32 | PrimType::U32 => 32,
            PrimType::I64 | PrimType::U64 => 64,
            PrimType::F32 => 32,
            PrimType::F64 => 64,
        }
    }
}

impl fmt::Display for PrimType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrimType::I8 => write!(f, "i8"),
            PrimType::U8 => write!(f, "u8"),
            PrimType::I16 => write!(f, "i16"),
            PrimType::U16 => write!(f, "u16"),
            PrimType::I32 => write!(f, "i32"),
            PrimType::U32 => write!(f, "u32"),
            PrimType::I64 => write!(f, "i64"),
            PrimType::U64 => write!(f, "u64"),
            PrimType::F32 => write!(f, "f32"),
            PrimType::F64 => write!(f, "f64"),
        }
    }
}

// =============================================================================
// NUMERIC TRAIT - Encodes Cranelift support for each type
// =============================================================================

/// Trait for types that can be used with StagedNum<T>
///
/// This trait encodes how to generate Cranelift IR for each numeric type:
/// - How to create constants
/// - How to perform arithmetic operations
/// - What Cranelift type to use
pub trait Numeric: Copy + fmt::Debug + fmt::Display + 'static {
    /// Get the PrimType for this numeric type
    fn prim_type() -> PrimType;

    /// Get the Cranelift type for this numeric type
    fn cranelift_type() -> Type {
        Self::prim_type().to_cranelift_type()
    }

    /// Create a Cranelift constant instruction for this value
    fn create_const(builder: &mut FunctionBuilder, value: Self) -> Value;

    /// Create an add instruction (iadd for ints, fadd for floats)
    fn create_add(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value;

    /// Create a subtract instruction (isub for ints, fsub for floats)
    fn create_sub(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value;

    /// Create a multiply instruction (imul for ints, fmul for floats)
    fn create_mul(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value;

    /// Create a divide instruction (sdiv/udiv for ints, fdiv for floats)
    fn create_div(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value;
}

// =============================================================================
// NUMERIC TRAIT IMPLEMENTATIONS
// =============================================================================

// Integer types share most operations, just differ in constants and division
macro_rules! impl_numeric_int {
    ($rust_type:ty, $prim_type:expr, $is_signed:expr) => {
        impl Numeric for $rust_type {
            fn prim_type() -> PrimType {
                $prim_type
            }

            fn create_const(builder: &mut FunctionBuilder, value: Self) -> Value {
                builder.ins().iconst(Self::cranelift_type(), value as i64)
            }

            fn create_add(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
                builder.ins().iadd(left, right)
            }

            fn create_sub(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
                builder.ins().isub(left, right)
            }

            fn create_mul(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
                builder.ins().imul(left, right)
            }

            fn create_div(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
                if $is_signed {
                    builder.ins().sdiv(left, right)
                } else {
                    builder.ins().udiv(left, right)
                }
            }
        }
    };
}

impl_numeric_int!(i8, PrimType::I8, true);
impl_numeric_int!(u8, PrimType::U8, false);
impl_numeric_int!(i16, PrimType::I16, true);
impl_numeric_int!(u16, PrimType::U16, false);
impl_numeric_int!(i32, PrimType::I32, true);
impl_numeric_int!(u32, PrimType::U32, false);
impl_numeric_int!(i64, PrimType::I64, true);
impl_numeric_int!(u64, PrimType::U64, false);

// Float types
impl Numeric for f32 {
    fn prim_type() -> PrimType {
        PrimType::F32
    }

    fn create_const(builder: &mut FunctionBuilder, value: Self) -> Value {
        builder.ins().f32const(value)
    }

    fn create_add(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
        builder.ins().fadd(left, right)
    }

    fn create_sub(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
        builder.ins().fsub(left, right)
    }

    fn create_mul(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
        builder.ins().fmul(left, right)
    }

    fn create_div(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
        builder.ins().fdiv(left, right)
    }
}

impl Numeric for f64 {
    fn prim_type() -> PrimType {
        PrimType::F64
    }

    fn create_const(builder: &mut FunctionBuilder, value: Self) -> Value {
        builder.ins().f64const(value)
    }

    fn create_add(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
        builder.ins().fadd(left, right)
    }

    fn create_sub(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
        builder.ins().fsub(left, right)
    }

    fn create_mul(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
        builder.ins().fmul(left, right)
    }

    fn create_div(builder: &mut FunctionBuilder, left: Value, right: Value) -> Value {
        builder.ins().fdiv(left, right)
    }
}

// =============================================================================
// STAGED NUMERIC VALUE
// =============================================================================

/// A staged numeric value that will generate Cranelift IR at compile time
///
/// This is generic over all primitive numeric types. The type parameter T
/// ensures type safety at the Rust level - you can't add an i32 to an f64
/// without explicit casting (which happens at the Expr level).
#[derive(Debug, Clone)]
pub enum StagedNum<T: Numeric> {
    /// A constant value known at compile time
    Constant(T),

    /// A variable (function parameter) known only at runtime
    Variable(Variable),

    /// Addition of two staged values
    Add(Box<StagedNum<T>>, Box<StagedNum<T>>),

    /// Subtraction of two staged values
    Sub(Box<StagedNum<T>>, Box<StagedNum<T>>),

    /// Multiplication of two staged values
    Mul(Box<StagedNum<T>>, Box<StagedNum<T>>),

    /// Division of two staged values
    Div(Box<StagedNum<T>>, Box<StagedNum<T>>),
}

impl<T: Numeric> StagedNum<T> {
    /// Create a constant staged value
    pub fn constant(value: T) -> Self {
        StagedNum::Constant(value)
    }

    /// Create a variable staged value (represents a function parameter)
    pub fn variable(var: Variable) -> Self {
        StagedNum::Variable(var)
    }

    /// Get the primitive type of this staged value
    pub fn prim_type(&self) -> PrimType {
        T::prim_type()
    }

    /// Generate Cranelift IR code for this value
    pub fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            StagedNum::Constant(val) => T::create_const(builder, *val),
            StagedNum::Variable(var) => builder.use_var(*var),
            StagedNum::Add(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                T::create_add(builder, left_val, right_val)
            }
            StagedNum::Sub(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                T::create_sub(builder, left_val, right_val)
            }
            StagedNum::Mul(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                T::create_mul(builder, left_val, right_val)
            }
            StagedNum::Div(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                T::create_div(builder, left_val, right_val)
            }
        }
    }
}

// =============================================================================
// ARITHMETIC OPERATIONS
// =============================================================================

impl<T: Numeric> Add for StagedNum<T> {
    type Output = StagedNum<T>;

    fn add(self, rhs: StagedNum<T>) -> StagedNum<T> {
        StagedNum::Add(Box::new(self), Box::new(rhs))
    }
}

impl<T: Numeric> Sub for StagedNum<T> {
    type Output = StagedNum<T>;

    fn sub(self, rhs: StagedNum<T>) -> StagedNum<T> {
        StagedNum::Sub(Box::new(self), Box::new(rhs))
    }
}

impl<T: Numeric> Mul for StagedNum<T> {
    type Output = StagedNum<T>;

    fn mul(self, rhs: StagedNum<T>) -> StagedNum<T> {
        StagedNum::Mul(Box::new(self), Box::new(rhs))
    }
}

// =============================================================================
// CONVERSION HELPERS
// =============================================================================

// Allow creating StagedNum from integer literals
macro_rules! impl_from_literal {
    ($rust_type:ty) => {
        impl From<$rust_type> for StagedNum<$rust_type> {
            fn from(value: $rust_type) -> Self {
                StagedNum::Constant(value)
            }
        }
    };
}

impl_from_literal!(i8);
impl_from_literal!(u8);
impl_from_literal!(i16);
impl_from_literal!(u16);
impl_from_literal!(i32);
impl_from_literal!(u32);
impl_from_literal!(i64);
impl_from_literal!(u64);
impl_from_literal!(f32);
impl_from_literal!(f64);

// =============================================================================
// DISPLAY IMPLEMENTATION
// =============================================================================

impl<T: Numeric> fmt::Display for StagedNum<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StagedNum::Constant(val) => write!(f, "{}{}", val, T::prim_type()),
            StagedNum::Variable(var) => write!(f, "v{}", var.as_u32()),
            StagedNum::Add(left, right) => write!(f, "({} + {})", left, right),
            StagedNum::Sub(left, right) => write!(f, "({} - {})", left, right),
            StagedNum::Mul(left, right) => write!(f, "({} * {})", left, right),
            StagedNum::Div(left, right) => write!(f, "({} / {})", left, right),
        }
    }
}

// =============================================================================
// TYPE ALIASES FOR CONVENIENCE
// =============================================================================

pub type StagedI8 = StagedNum<i8>;
pub type StagedU8 = StagedNum<u8>;
pub type StagedI16 = StagedNum<i16>;
pub type StagedU16 = StagedNum<u16>;
pub type StagedI32 = StagedNum<i32>;
pub type StagedU32 = StagedNum<u32>;
pub type StagedI64 = StagedNum<i64>;
pub type StagedU64 = StagedNum<u64>;
pub type StagedF32 = StagedNum<f32>;
pub type StagedF64 = StagedNum<f64>;

// =============================================================================
// COMPARISON OPERATIONS
// =============================================================================

// Import StagedBool and Condition for comparison operations
use crate::bool::{Condition, StagedBool};
use crate::{DataType, PRIM_DATA_TYPES};

/// Macro to implement comparison methods using type-erased Compare variant
///
/// This generates all six comparison operations (lt, le, gt, ge, eq, ne)
/// for the given type, using the unified StagedBool::Compare variant with trait objects.
///
/// Note: These methods take `&self` instead of consuming `self` for ergonomics.
/// The cloning is handled internally, so callers don't need to clone explicitly.
macro_rules! impl_staged_num_comparisons {
    ($rust_type:ty) => {
        impl StagedNum<$rust_type> {
            /// Less than: self < other
            pub fn lt(&self, other: &Self) -> StagedBool {
                StagedBool::Compare {
                    condition: Condition::LessThan,
                    left: Box::new(self.clone()),
                    right: Box::new(other.clone()),
                    operand_type: <$rust_type>::prim_type(),
                }
            }

            /// Less than or equal: self <= other
            pub fn le(&self, other: &Self) -> StagedBool {
                StagedBool::Compare {
                    condition: Condition::LessThanOrEqual,
                    left: Box::new(self.clone()),
                    right: Box::new(other.clone()),
                    operand_type: <$rust_type>::prim_type(),
                }
            }

            /// Greater than: self > other
            pub fn gt(&self, other: &Self) -> StagedBool {
                StagedBool::Compare {
                    condition: Condition::GreaterThan,
                    left: Box::new(self.clone()),
                    right: Box::new(other.clone()),
                    operand_type: <$rust_type>::prim_type(),
                }
            }

            /// Greater than or equal: self >= other
            pub fn ge(&self, other: &Self) -> StagedBool {
                StagedBool::Compare {
                    condition: Condition::GreaterThanOrEqual,
                    left: Box::new(self.clone()),
                    right: Box::new(other.clone()),
                    operand_type: <$rust_type>::prim_type(),
                }
            }

            /// Equal: self == other
            pub fn eq(&self, other: &Self) -> StagedBool {
                StagedBool::Compare {
                    condition: Condition::Equal,
                    left: Box::new(self.clone()),
                    right: Box::new(other.clone()),
                    operand_type: <$rust_type>::prim_type(),
                }
            }

            /// Not equal: self != other
            pub fn ne(&self, other: &Self) -> StagedBool {
                StagedBool::Compare {
                    condition: Condition::NotEqual,
                    left: Box::new(self.clone()),
                    right: Box::new(other.clone()),
                    operand_type: <$rust_type>::prim_type(),
                }
            }
        }
    };
}

// Generate comparison implementations for all numeric types
impl_staged_num_comparisons!(i8);
impl_staged_num_comparisons!(u8);
impl_staged_num_comparisons!(i16);
impl_staged_num_comparisons!(u16);
impl_staged_num_comparisons!(i32);
impl_staged_num_comparisons!(u32);
impl_staged_num_comparisons!(i64);
impl_staged_num_comparisons!(u64);
impl_staged_num_comparisons!(f32);
impl_staged_num_comparisons!(f64);

// =============================================================================
// STAGEDVALUE TRAIT IMPLEMENTATION
// =============================================================================

use crate::staged_value::StagedValue;


impl<T: Numeric> StagedValue for StagedNum<T> {
    fn data_type(&self) -> &crate::DataType {
        &PRIM_DATA_TYPES[T::prim_type().as_index()]
    }

    fn codegen(&self, builder: &mut FunctionBuilder) -> cranelift_codegen::ir::Value {
        // Delegate to the existing codegen method
        self.codegen(builder)
    }

    fn clone_box(&self) -> Box<dyn StagedValue> {
        Box::new(self.clone())
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
