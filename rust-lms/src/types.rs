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
/// - Size and alignment information for struct layout
pub trait StagedType {
    /// The actual runtime type (e.g., i64 for I64Type)
    type RuntimeValue;

    /// Get the Cranelift IR type representation.
    /// For primitives, this is the actual type (I64, F64, etc.)
    /// For structs, this is I64 (pointer to stack slot)
    fn cranelift_type() -> cranelift_codegen::ir::Type;

    /// Size of this type in bytes (for struct layout calculations)
    fn size_of() -> usize {
        // Default: use Cranelift type size
        match Self::cranelift_type() {
            types::I8 => 1,
            types::I16 => 2,
            types::I32 | types::F32 => 4,
            types::I64 | types::F64 => 8,
            _ => 8, // Default to pointer size
        }
    }

    /// Alignment of this type in bytes (for struct layout calculations)
    fn align_of() -> usize {
        // Default: alignment equals size for primitives
        Self::size_of()
    }

    /// Returns true if this is a Copy struct that should be passed by value.
    /// When true, the type is passed in registers at the ABI boundary but
    /// stored to a stack slot internally for field access via pointer.
    fn is_copy_struct() -> bool {
        false
    }

    /// Returns true if this struct should be passed by pointer at the ABI level.
    ///
    /// On ARM64, structs larger than 16 bytes are passed by pointer according
    /// to the C ABI (caller allocates memory, passes pointer). This method
    /// detects that case to generate correct calling convention code.
    ///
    /// For structs ≤16 bytes, returns false (pass in registers).
    /// For structs >16 bytes, returns true (pass by pointer).
    fn should_pass_by_pointer() -> bool {
        // Only applies to copy structs larger than 16 bytes
        Self::is_copy_struct() && Self::size_of() > 16
    }

    /// Number of primitive values this type flattens to at the ABI boundary.
    /// For primitives: 1
    /// For structs ≤16 bytes: number of register-sized values needed
    /// For structs >16 bytes: 1 (pointer)
    fn num_abi_values() -> usize {
        1
    }

    /// Get the Cranelift types for each ABI value.
    /// For primitives: just the cranelift_type
    /// For structs: sequence of I64s (or I64+F64 mix if we support floats in structs)
    fn abi_types() -> Vec<cranelift_codegen::ir::Type> {
        vec![Self::cranelift_type()]
    }
}

/// Types that can be compile-time constants.
///
/// Not all StagedType values can be constants (e.g., function types cannot),
/// so this is a separate trait.
pub trait ConstantType: StagedType {
    /// Generate code for a constant value
    fn codegen_constant(
        value: &Self::RuntimeValue,
        builder: &mut FunctionBuilder,
    ) -> Value;
}

/// Marker trait for types that are Copy at the semantic level.
///
/// This trait indicates that a type can be copied by value (in Rust semantics),
/// even though the Cranelift representation may use pointers for structs.
///
/// Primitive types (i64, f64, bool) are always CopyType.
/// Structs are CopyType only if all their fields are CopyType.
pub trait CopyType: StagedType {}

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
    type RuntimeValue = i64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn size_of() -> usize {
        8
    }

    fn align_of() -> usize {
        8
    }
}

impl ConstantType for I64Type {
    fn codegen_constant(value: &i64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value)
    }
}

impl CopyType for I64Type {}

impl StagedType for U64Type {
    type RuntimeValue = u64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn size_of() -> usize {
        8
    }

    fn align_of() -> usize {
        8
    }
}

impl ConstantType for U64Type {
    fn codegen_constant(value: &u64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value as i64)
    }
}

impl CopyType for U64Type {}

impl StagedType for BoolType {
    type RuntimeValue = bool;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I8
    }

    fn size_of() -> usize {
        1
    }

    fn align_of() -> usize {
        1
    }
}

impl ConstantType for BoolType {
    fn codegen_constant(value: &bool, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, if *value { 1 } else { 0 })
    }
}

impl CopyType for BoolType {}

impl StagedType for F64Type {
    type RuntimeValue = f64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::F64
    }

    fn size_of() -> usize {
        8
    }

    fn align_of() -> usize {
        8
    }
}

impl ConstantType for F64Type {
    fn codegen_constant(value: &f64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().f64const(*value)
    }
}

impl CopyType for F64Type {}

impl StagedType for UnitType {
    type RuntimeValue = ();

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I8 // Minimal representation, value is ignored
    }

    fn size_of() -> usize {
        0
    }

    fn align_of() -> usize {
        1
    }
}

impl ConstantType for UnitType {
    fn codegen_constant(_value: &(), builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, 0)
    }
}

impl CopyType for UnitType {}
