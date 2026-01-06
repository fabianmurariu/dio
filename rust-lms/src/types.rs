//! Type system for staged computations.
//!
//! This module defines:
//! - `StagedType`: Trait for types that can participate in staged computation
//! - Concrete type markers: `I64Type`, `U64Type`, `BoolType`, etc.

use cranelift_codegen::ir::{types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

// =============================================================================
// Core Trait: StagedType
// =============================================================================

/// Types that can participate in staged computations.
///
/// This trait associates a Rust type with:
/// - Its runtime value representation
/// - Its Cranelift IR type
/// - How to generate code for constant values
pub trait StagedType: 'static {
    /// The actual runtime type (e.g., i64 for I64Type)
    type RuntimeValue: Clone;

    /// Get the Cranelift IR type representation
    fn cranelift_type() -> cranelift_codegen::ir::Type;

    /// Generate code for a constant value
    fn codegen_constant(value: &Self::RuntimeValue, builder: &mut FunctionBuilder) -> Value;
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

// =============================================================================
// StagedType implementations
// =============================================================================

impl StagedType for I64Type {
    type RuntimeValue = i64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn codegen_constant(value: &i64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value)
    }
}

impl StagedType for U64Type {
    type RuntimeValue = u64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn codegen_constant(value: &u64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value as i64)
    }
}

impl StagedType for BoolType {
    type RuntimeValue = bool;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I8
    }

    fn codegen_constant(value: &bool, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, if *value { 1 } else { 0 })
    }
}

impl StagedType for F64Type {
    type RuntimeValue = f64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::F64
    }

    fn codegen_constant(value: &f64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().f64const(*value)
    }
}