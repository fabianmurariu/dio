//! Type system for staged computations.
//!
//! This module defines:
//! - `StagedType`: Base trait for all types that can participate in staged computation
//! - `ConstantType`: Trait for types that can be compile-time constants
//! - Concrete type markers: `I64Type`, `U64Type`, `BoolType`, etc.

use cranelift_codegen::ir::{types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

// =============================================================================
// Core Traits
// =============================================================================

/// Base trait for all types that can participate in staged computations.
///
/// This trait associates a Rust type with:
/// - Its runtime value representation
/// - Its Cranelift IR type
pub trait StagedType: 'static {
    /// The actual runtime type (e.g., i64 for I64Type)
    type RuntimeValue<'a>;

    /// Get the Cranelift IR type representation
    fn cranelift_type() -> cranelift_codegen::ir::Type;
}

/// Types that can be compile-time constants.
///
/// Not all StagedType values can be constants (e.g., function types cannot),
/// so this is a separate trait.
pub trait ConstantType: StagedType {
    /// Generate code for a constant value
    fn codegen_constant(value: &Self::RuntimeValue<'static>, builder: &mut FunctionBuilder) -> Value;
}

// =============================================================================
// Concrete Type Markers
// =============================================================================

/// Marker type for i64 values
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct I64Type;

/// Marker type for u64 values
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct U64Type;

/// Marker type for boolean values
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoolType;

/// Marker type for f64 values
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct F64Type;

/// Marker type for unit (no value) - used for side-effect-only expressions
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UnitType;

// =============================================================================
// StagedType implementations
// =============================================================================

impl StagedType for I64Type {
    type RuntimeValue<'a> = i64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }
}

impl ConstantType for I64Type {
    fn codegen_constant(value: &i64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value)
    }
}

impl StagedType for U64Type {
    type RuntimeValue<'a> = u64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }
}

impl ConstantType for U64Type {
    fn codegen_constant(value: &u64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value as i64)
    }
}

impl StagedType for BoolType {
    type RuntimeValue<'a> = bool;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I8
    }
}

impl ConstantType for BoolType {
    fn codegen_constant(value: &bool, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, if *value { 1 } else { 0 })
    }
}

impl StagedType for F64Type {
    type RuntimeValue<'a> = f64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::F64
    }
}

impl ConstantType for F64Type {
    fn codegen_constant(value: &f64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().f64const(*value)
    }
}

impl StagedType for UnitType {
    type RuntimeValue<'a> = ();

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I8 // Minimal representation, value is ignored
    }
}

impl ConstantType for UnitType {
    fn codegen_constant(_value: &(), builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, 0)
    }
}