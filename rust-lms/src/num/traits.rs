//! Capability traits for numeric types.
//!
//! - [`Ordered`]: comparison (`< > ==`), for every [`Num`] and for `i128`/`u128`.
//! - [`Num`]: shared arithmetic (`+ - * /`) for any integer or floating-point
//!   staged type. Both branches of the hierarchy refine this.
//! - [`Int`]: the integers, with shifts and bit counts. Implemented by every integer
//!   type, `i128` and `u128` included.
//! - [`IntNum`]: an [`Int`] that is also a [`Num`], adding remainder (modulo). Every
//!   integer type but `i128`/`u128`, which have no division.
//! - [`FloatNum`]: marker for floating-point staged types; reserved for future
//!   float-only operations (sqrt, abs, ...). Implemented by `f64`.
//! - [`Bits`]: bitwise `& | ^`. Implemented by every integer type and by `bool`,
//!   where they are logical and/or/xor (non-short-circuiting).
//!
//! `bool` deliberately does not implement `Num` — boolean values use the
//! control-flow combinators (`if_then`, `if_then_else`) and [`Bits`] rather than
//! arithmetic.

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
pub trait Num: Ordered {
    fn codegen_add(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_sub(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_mul(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_div(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
}

/// Staged types with comparisons (`< > == != <= >=`): every [`Num`], and `i128`/`u128`,
/// which have no division and so are not `Num`s.
///
/// Sealed for the same reason as [`Num`].
pub trait Ordered: StagedType + ConstantType + CopyType + sealed::Sealed + 'static {
    fn codegen_lt(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_gt(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_eq(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_ne(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_le(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_ge(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
}

/// Types supporting bitwise `& | ^`: the integers, and `bool` (where they are
/// logical and/or/xor). A `bool` is an `i8` in `{0, 1}`, a set closed under all
/// three, so one instruction per op is exact for every implementor — this is a
/// marker, with no per-type instruction selection.
///
/// `bool` gets only the logical ops — no shifts:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
/// fn shift(a: Var<bool>, b: Var<bool>) { let _ = a << b; }
/// ```
///
/// no arithmetic:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
/// fn plus(a: Var<bool>, b: Var<bool>) { let _ = a + b; }
/// ```
///
/// and no mixing with integers:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
/// fn mixed(a: Var<bool>, b: Var<i64>) { let _ = a & b; }
/// ```
pub trait Bits: StagedType + ConstantType + CopyType + sealed::Sealed + 'static {}

/// The integers, of every width: bitwise ops, shifts, bit counts and byte swaps, and
/// comparisons by their signedness. `i128`/`u128` are `Int`s without arithmetic.
pub trait Int: Ordered + Bits {
    const SIGNED: bool;

    fn codegen_shl(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value;
    fn codegen_shr(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value;
}

/// Integer-typed numbers — an [`Int`] with arithmetic, adding remainder (modulo).
pub trait IntNum: Num + Int {
    fn codegen_rem(left: Value, right: Value, ctx: &mut CompilationContext<'_>) -> Value;
}

/// Floating-point numbers — marker; reserved for future float-only ops.
pub trait FloatNum: Num {}

// =============================================================================
// Macro-generated impls for primitive integer types
// =============================================================================

macro_rules! impl_int {
    ($ty:ty, signed) => {
        impl Int for $ty {
            const SIGNED: bool = true;

            fn codegen_shl(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.ishl(l.leaf(), r.leaf()))
            }
            fn codegen_shr(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.sshr(l.leaf(), r.leaf()))
            }
        }
    };
    ($ty:ty, unsigned) => {
        impl Int for $ty {
            const SIGNED: bool = false;

            fn codegen_shl(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.ishl(l.leaf(), r.leaf()))
            }
            fn codegen_shr(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.ushr(l.leaf(), r.leaf()))
            }
        }
    };
}

/// `Ordered` by a signedness: the six integer comparisons.
macro_rules! impl_int_ordered {
    ($ty:ty, $lt:ident, $gt:ident, $le:ident, $ge:ident) => {
        impl Ordered for $ty {
            fn codegen_lt(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::$lt, l.leaf(), r.leaf()))
            }
            fn codegen_gt(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::$gt, l.leaf(), r.leaf()))
            }
            fn codegen_eq(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Eq, l.leaf(), r.leaf()))
            }
            fn codegen_ne(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::Ne, l.leaf(), r.leaf()))
            }
            fn codegen_le(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::$le, l.leaf(), r.leaf()))
            }
            fn codegen_ge(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.icmp(IntCmp::$ge, l.leaf(), r.leaf()))
            }
        }
    };
}

macro_rules! impl_int_num {
    ($ty:ty, signed) => {
        impl sealed::Sealed for $ty {}
        impl Bits for $ty {}
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
        }
        impl_int_ordered!($ty, Slt, Sgt, Sle, Sge);
        impl_int!($ty, signed);
        impl IntNum for $ty {
            fn codegen_rem(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.srem(l.leaf(), r.leaf()))
            }
        }
    };
    ($ty:ty, unsigned) => {
        impl sealed::Sealed for $ty {}
        impl Bits for $ty {}
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
        }
        impl_int_ordered!($ty, Ult, Ugt, Ule, Uge);
        impl_int!($ty, unsigned);
        impl IntNum for $ty {
            fn codegen_rem(l: Value, r: Value, ctx: &mut CompilationContext<'_>) -> Value {
                Value::scalar(ctx.urem(l.leaf(), r.leaf()))
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

// The 128-bit integers: ordered, bitwise, shifts and bit counts, no arithmetic.
impl sealed::Sealed for i128 {}
impl Bits for i128 {}
impl_int_ordered!(i128, Slt, Sgt, Sle, Sge);
impl_int!(i128, signed);

impl sealed::Sealed for u128 {}
impl Bits for u128 {}
impl_int_ordered!(u128, Ult, Ugt, Ule, Uge);
impl_int!(u128, unsigned);

impl sealed::Sealed for bool {}
impl Bits for bool {}

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
        }
        impl Ordered for $ty {
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
