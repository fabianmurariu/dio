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

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::types::{ConstantType, CopyType, StagedType};

// =============================================================================
// Trait hierarchy
// =============================================================================

/// Numeric staged types — share arithmetic and comparison operations.
pub trait Num: StagedType + ConstantType + CopyType + 'static {
    fn codegen_add(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
    fn codegen_sub(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
    fn codegen_mul(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
    fn codegen_div(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
    fn codegen_lt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
    fn codegen_gt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
    fn codegen_eq(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
}

/// Integer-typed numbers — additionally support remainder (modulo).
pub trait IntNum: Num {
    fn codegen_rem(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
}

/// Floating-point numbers — marker; reserved for future float-only ops.
pub trait FloatNum: Num {}

// =============================================================================
// Macro-generated impls for primitive integer types
// =============================================================================

macro_rules! impl_int_num {
    ($ty:ty, signed) => {
        impl Num for $ty {
            fn codegen_add(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().iadd(l, r)
            }
            fn codegen_sub(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().isub(l, r)
            }
            fn codegen_mul(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().imul(l, r)
            }
            fn codegen_div(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().sdiv(l, r)
            }
            fn codegen_lt(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().icmp(IntCC::SignedLessThan, l, r)
            }
            fn codegen_gt(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().icmp(IntCC::SignedGreaterThan, l, r)
            }
            fn codegen_eq(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().icmp(IntCC::Equal, l, r)
            }
        }
        impl IntNum for $ty {
            fn codegen_rem(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().srem(l, r)
            }
        }
    };
    ($ty:ty, unsigned) => {
        impl Num for $ty {
            fn codegen_add(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().iadd(l, r)
            }
            fn codegen_sub(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().isub(l, r)
            }
            fn codegen_mul(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().imul(l, r)
            }
            fn codegen_div(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().udiv(l, r)
            }
            fn codegen_lt(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().icmp(IntCC::UnsignedLessThan, l, r)
            }
            fn codegen_gt(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().icmp(IntCC::UnsignedGreaterThan, l, r)
            }
            fn codegen_eq(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().icmp(IntCC::Equal, l, r)
            }
        }
        impl IntNum for $ty {
            fn codegen_rem(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
                b.ins().urem(l, r)
            }
        }
    };
}

impl_int_num!(i64, signed);
impl_int_num!(u64, unsigned);
impl_int_num!(i32, signed);
impl_int_num!(u32, unsigned);

// =============================================================================
// Floating point
// =============================================================================

impl Num for f64 {
    fn codegen_add(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
        b.ins().fadd(l, r)
    }
    fn codegen_sub(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
        b.ins().fsub(l, r)
    }
    fn codegen_mul(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
        b.ins().fmul(l, r)
    }
    fn codegen_div(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
        b.ins().fdiv(l, r)
    }
    fn codegen_lt(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
        b.ins().fcmp(FloatCC::LessThan, l, r)
    }
    fn codegen_gt(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
        b.ins().fcmp(FloatCC::GreaterThan, l, r)
    }
    fn codegen_eq(l: Value, r: Value, b: &mut FunctionBuilder) -> Value {
        b.ins().fcmp(FloatCC::Equal, l, r)
    }
}

impl FloatNum for f64 {}
