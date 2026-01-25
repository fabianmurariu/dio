//! Capability traits for numeric types.
//!
//! These traits define which operations are supported by different staged types.

use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::types::{BoolType, F64Type, I32Type, I64Type, StagedType, U32Type, U64Type};

// =============================================================================
// Capability Traits
// =============================================================================

/// Types that support addition
pub trait SupportsAdd: StagedType {
    /// Generate code for addition
    fn codegen_add(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
}

/// Types that support subtraction
pub trait SupportsSub: StagedType {
    /// Generate code for subtraction
    fn codegen_sub(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
}

/// Types that support multiplication
pub trait SupportsMul: StagedType {
    /// Generate code for multiplication
    fn codegen_mul(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
}

/// Types that support division
pub trait SupportsDiv: StagedType {
    /// Generate code for division
    fn codegen_div(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
}

/// Types that support comparison operations
pub trait SupportsComparison: StagedType {
    /// Generate code for less-than comparison
    fn codegen_lt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;

    /// Generate code for greater-than comparison
    fn codegen_gt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;

    /// Generate code for equality comparison
    fn codegen_eq(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value;
}

// =============================================================================
// Implementations for I64Type
// =============================================================================

impl SupportsAdd for I64Type {
    fn codegen_add(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iadd(left, right)
    }
}

impl SupportsSub for I64Type {
    fn codegen_sub(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().isub(left, right)
    }
}

impl SupportsMul for I64Type {
    fn codegen_mul(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().imul(left, right)
    }
}

impl SupportsDiv for I64Type {
    fn codegen_div(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().sdiv(left, right)
    }
}

impl SupportsComparison for I64Type {
    fn codegen_lt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::SignedLessThan, left, right)
    }

    fn codegen_gt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::SignedGreaterThan, left, right)
    }

    fn codegen_eq(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::Equal, left, right)
    }
}

// =============================================================================
// Implementations for U64Type
// =============================================================================

impl SupportsAdd for U64Type {
    fn codegen_add(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iadd(left, right)
    }
}

impl SupportsSub for U64Type {
    fn codegen_sub(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().isub(left, right)
    }
}

impl SupportsMul for U64Type {
    fn codegen_mul(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().imul(left, right)
    }
}

impl SupportsDiv for U64Type {
    fn codegen_div(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().udiv(left, right) // Note: unsigned division
    }
}

impl SupportsComparison for U64Type {
    fn codegen_lt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::UnsignedLessThan, left, right)
    }

    fn codegen_gt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::UnsignedGreaterThan, left, right)
    }

    fn codegen_eq(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::Equal, left, right)
    }
}

// =============================================================================
// Implementations for I32Type
// =============================================================================

impl SupportsAdd for I32Type {
    fn codegen_add(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iadd(left, right)
    }
}

impl SupportsSub for I32Type {
    fn codegen_sub(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().isub(left, right)
    }
}

impl SupportsMul for I32Type {
    fn codegen_mul(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().imul(left, right)
    }
}

impl SupportsDiv for I32Type {
    fn codegen_div(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().sdiv(left, right)
    }
}

impl SupportsComparison for I32Type {
    fn codegen_lt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::SignedLessThan, left, right)
    }

    fn codegen_gt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::SignedGreaterThan, left, right)
    }

    fn codegen_eq(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::Equal, left, right)
    }
}

// =============================================================================
// Implementations for U32Type
// =============================================================================

impl SupportsAdd for U32Type {
    fn codegen_add(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iadd(left, right)
    }
}

impl SupportsSub for U32Type {
    fn codegen_sub(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().isub(left, right)
    }
}

impl SupportsMul for U32Type {
    fn codegen_mul(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().imul(left, right)
    }
}

impl SupportsDiv for U32Type {
    fn codegen_div(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().udiv(left, right) // Note: unsigned division
    }
}

impl SupportsComparison for U32Type {
    fn codegen_lt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::UnsignedLessThan, left, right)
    }

    fn codegen_gt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::UnsignedGreaterThan, left, right)
    }

    fn codegen_eq(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::Equal, left, right)
    }
}

// =============================================================================
// Implementations for F64Type
// =============================================================================

impl SupportsAdd for F64Type {
    fn codegen_add(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().fadd(left, right)
    }
}

impl SupportsSub for F64Type {
    fn codegen_sub(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().fsub(left, right)
    }
}

impl SupportsMul for F64Type {
    fn codegen_mul(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().fmul(left, right)
    }
}

impl SupportsDiv for F64Type {
    fn codegen_div(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        builder.ins().fdiv(left, right)
    }
}

impl SupportsComparison for F64Type {
    fn codegen_lt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::FloatCC;
        builder.ins().fcmp(FloatCC::LessThan, left, right)
    }

    fn codegen_gt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::FloatCC;
        builder.ins().fcmp(FloatCC::GreaterThan, left, right)
    }

    fn codegen_eq(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::FloatCC;
        builder.ins().fcmp(FloatCC::Equal, left, right)
    }
}

// =============================================================================
// Implementations for BoolType (limited operations)
// =============================================================================

impl SupportsComparison for BoolType {
    fn codegen_lt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::UnsignedLessThan, left, right)
    }

    fn codegen_gt(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::UnsignedGreaterThan, left, right)
    }

    fn codegen_eq(left: Value, right: Value, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;
        builder.ins().icmp(IntCC::Equal, left, right)
    }
}
