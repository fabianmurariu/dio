//! Capability traits for numeric types.
//!
//! - [`Num`]: shared arithmetic (`+ - * /`) and comparison (`< > ==`) for any
//!   integer or floating-point staged type. Both branches of the hierarchy
//!   refine this.
//! - [`IntNum`]: adds remainder (modulo). Implemented by `i64`, `u64`, `i32`,
//!   `u32`.
//! - [`FloatNum`]: marker for floating-point staged types; reserved for future
//!   float-only operations (sqrt, abs, ...). Implemented by `f64`.
//!
//! `bool` deliberately does not implement `Num` — boolean values use the
//! control-flow combinators (`if_then`, `if_then_else`) rather than arithmetic.

use crate::staged::{CompilationContext, Value};
use crate::types::{ConstantType, CopyType, FloatCmp, IntCmp, StagedType};

// =============================================================================
// Trait hierarchy
// =============================================================================

mod sealed {
    pub trait Sealed {}
}

/// Numeric staged types — share arithmetic and comparison operations.
///
/// This trait is sealed because its methods return raw IR values whose type is
/// trusted by every arithmetic expression.
pub trait Num: StagedType + ConstantType + CopyType + sealed::Sealed + 'static {
    fn codegen_add(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_sub(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_mul(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_div(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_lt(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_gt(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_eq(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_ne(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_le(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_ge(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
}

/// Integer-typed numbers — additionally support remainder (modulo).
pub trait IntNum: Num {
    const SIGNED: bool;

    fn codegen_rem(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_bitand(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_bitor(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_bitxor(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_shl(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_shr(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value;
}

/// Floating-point numbers — marker; reserved for future float-only ops.
pub trait FloatNum: Num {}

// =============================================================================
// Macro-generated impls for primitive integer types
// =============================================================================

macro_rules! impl_int_num {
    ($ty:ty, signed) => {
        impl sealed::Sealed for $ty {}
        impl Num for $ty {
            fn codegen_add(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.iadd(l.leaf(), r.leaf()))
            }
            fn codegen_sub(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.isub(l.leaf(), r.leaf()))
            }
            fn codegen_mul(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.imul(l.leaf(), r.leaf()))
            }
            fn codegen_div(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.sdiv(l.leaf(), r.leaf()))
            }
            fn codegen_lt(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Slt, l.leaf(), r.leaf()))
            }
            fn codegen_gt(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Sgt, l.leaf(), r.leaf()))
            }
            fn codegen_eq(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Eq, l.leaf(), r.leaf()))
            }
            fn codegen_ne(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Ne, l.leaf(), r.leaf()))
            }
            fn codegen_le(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Sle, l.leaf(), r.leaf()))
            }
            fn codegen_ge(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Sge, l.leaf(), r.leaf()))
            }
        }
        impl IntNum for $ty {
            const SIGNED: bool = true;

            fn codegen_rem(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.srem(l.leaf(), r.leaf()))
            }
            fn codegen_bitand(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.band(l.leaf(), r.leaf()))
            }
            fn codegen_bitor(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.bor(l.leaf(), r.leaf()))
            }
            fn codegen_bitxor(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.bxor(l.leaf(), r.leaf()))
            }
            fn codegen_shl(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.ishl(l.leaf(), r.leaf()))
            }
            fn codegen_shr(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.sshr(l.leaf(), r.leaf()))
            }
        }
    };
    ($ty:ty, unsigned) => {
        impl sealed::Sealed for $ty {}
        impl Num for $ty {
            fn codegen_add(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.iadd(l.leaf(), r.leaf()))
            }
            fn codegen_sub(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.isub(l.leaf(), r.leaf()))
            }
            fn codegen_mul(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.imul(l.leaf(), r.leaf()))
            }
            fn codegen_div(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.udiv(l.leaf(), r.leaf()))
            }
            fn codegen_lt(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Ult, l.leaf(), r.leaf()))
            }
            fn codegen_gt(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Ugt, l.leaf(), r.leaf()))
            }
            fn codegen_eq(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Eq, l.leaf(), r.leaf()))
            }
            fn codegen_ne(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Ne, l.leaf(), r.leaf()))
            }
            fn codegen_le(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Ule, l.leaf(), r.leaf()))
            }
            fn codegen_ge(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Uge, l.leaf(), r.leaf()))
            }
        }
        impl IntNum for $ty {
            const SIGNED: bool = false;

            fn codegen_rem(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.urem(l.leaf(), r.leaf()))
            }
            fn codegen_bitand(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.band(l.leaf(), r.leaf()))
            }
            fn codegen_bitor(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.bor(l.leaf(), r.leaf()))
            }
            fn codegen_bitxor(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.bxor(l.leaf(), r.leaf()))
            }
            fn codegen_shl(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.ishl(l.leaf(), r.leaf()))
            }
            fn codegen_shr(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.ushr(l.leaf(), r.leaf()))
            }
        }
    };
}

impl_int_num!(i8, signed);
impl_int_num!(u8, unsigned);
impl_int_num!(i16, signed);
impl_int_num!(u16, unsigned);
impl_int_num!(i64, signed);
impl_int_num!(u64, unsigned);
impl_int_num!(i32, signed);
impl_int_num!(u32, unsigned);

// =============================================================================
// Floating point
// =============================================================================

macro_rules! impl_float_num {
    ($ty:ty) => {
        impl sealed::Sealed for $ty {}
        impl Num for $ty {
            fn codegen_add(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fadd(l.leaf(), r.leaf()))
            }
            fn codegen_sub(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fsub(l.leaf(), r.leaf()))
            }
            fn codegen_mul(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fmul(l.leaf(), r.leaf()))
            }
            fn codegen_div(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fdiv(l.leaf(), r.leaf()))
            }
            fn codegen_lt(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fcmp(FloatCmp::Lt, l.leaf(), r.leaf()))
            }
            fn codegen_gt(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fcmp(FloatCmp::Gt, l.leaf(), r.leaf()))
            }
            fn codegen_eq(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fcmp(FloatCmp::Eq, l.leaf(), r.leaf()))
            }
            fn codegen_ne(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                // Unordered: `NaN != x` is true, the exact negation of `eq`.
                Value::scalar(ctx.fcmp(FloatCmp::Ne, l.leaf(), r.leaf()))
            }
            fn codegen_le(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fcmp(FloatCmp::Le, l.leaf(), r.leaf()))
            }
            fn codegen_ge(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.fcmp(FloatCmp::Ge, l.leaf(), r.leaf()))
            }
        }
        impl FloatNum for $ty {}
    };
}

impl_float_num!(f64);
impl_float_num!(f32);
