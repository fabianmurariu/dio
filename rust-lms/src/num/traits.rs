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

use crate::staged::{CompilationContext, ValueId};
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
    fn codegen_add(left: ValueId, right: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_sub(left: ValueId, right: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_mul(left: ValueId, right: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_div(left: ValueId, right: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_lt(left: ValueId, right: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_gt(left: ValueId, right: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_eq(left: ValueId, right: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
}

/// Integer-typed numbers — additionally support remainder (modulo).
pub trait IntNum: Num {
    const SIGNED: bool;

    fn codegen_rem(left: ValueId, right: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_bitand(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_bitor(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_bitxor(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_shl(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
    fn codegen_shr(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId;
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
            fn codegen_add(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.iadd(l, r)
            }
            fn codegen_sub(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.isub(l, r)
            }
            fn codegen_mul(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.imul(l, r)
            }
            fn codegen_div(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.sdiv(l, r)
            }
            fn codegen_lt(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.icmp(IntCmp::Slt, l, r)
            }
            fn codegen_gt(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.icmp(IntCmp::Sgt, l, r)
            }
            fn codegen_eq(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.icmp(IntCmp::Eq, l, r)
            }
        }
        impl IntNum for $ty {
            const SIGNED: bool = true;

            fn codegen_rem(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.srem(l, r)
            }
            fn codegen_bitand(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.band(l, r)
            }
            fn codegen_bitor(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.bor(l, r)
            }
            fn codegen_bitxor(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.bxor(l, r)
            }
            fn codegen_shl(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.ishl(l, r)
            }
            fn codegen_shr(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.sshr(l, r)
            }
        }
    };
    ($ty:ty, unsigned) => {
        impl sealed::Sealed for $ty {}
        impl Num for $ty {
            fn codegen_add(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.iadd(l, r)
            }
            fn codegen_sub(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.isub(l, r)
            }
            fn codegen_mul(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.imul(l, r)
            }
            fn codegen_div(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.udiv(l, r)
            }
            fn codegen_lt(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.icmp(IntCmp::Ult, l, r)
            }
            fn codegen_gt(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.icmp(IntCmp::Ugt, l, r)
            }
            fn codegen_eq(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.icmp(IntCmp::Eq, l, r)
            }
        }
        impl IntNum for $ty {
            const SIGNED: bool = false;

            fn codegen_rem(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.urem(l, r)
            }
            fn codegen_bitand(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.band(l, r)
            }
            fn codegen_bitor(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.bor(l, r)
            }
            fn codegen_bitxor(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.bxor(l, r)
            }
            fn codegen_shl(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.ishl(l, r)
            }
            fn codegen_shr(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.ushr(l, r)
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
            fn codegen_add(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.fadd(l, r)
            }
            fn codegen_sub(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.fsub(l, r)
            }
            fn codegen_mul(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.fmul(l, r)
            }
            fn codegen_div(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.fdiv(l, r)
            }
            fn codegen_lt(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.fcmp(FloatCmp::Lt, l, r)
            }
            fn codegen_gt(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.fcmp(FloatCmp::Gt, l, r)
            }
            fn codegen_eq(l: ValueId, r: ValueId, ctx: &mut CompilationContext<'_>) -> ValueId {
                ctx.fcmp(FloatCmp::Eq, l, r)
            }
        }
        impl FloatNum for $ty {}
    };
}

impl_float_num!(f64);
impl_float_num!(f32);
