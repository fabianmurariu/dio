//! Type system for staged computations.
//!
//! This module defines:
//! - `StagedType`: Base trait for all types that can participate in staged computation
//! - `ConstantType`: Trait for types that can be compile-time constants
//! - Concrete type markers: `i64`, `u64`, `bool`, etc.

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
///
/// Prefer `#[derive(StagedType)]` for `#[repr(C)]` structs. Manual
/// implementations are part of the compiler's trusted boundary.
///
/// # Safety
///
/// `RuntimeValue`, `cranelift_type`, `size_of`, and `align_of` must describe one
/// consistent runtime representation. The ABI methods must classify that same
/// representation correctly, and every value produced for this type must be a
/// valid `RuntimeValue`. Incorrect implementations can make generated code
/// perform invalid loads, stores, calls, or Rust function-pointer conversions.
///
/// The derive macro rejects field markers with incompatible runtime types:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// #[repr(C)]
/// #[derive(Clone, Copy, StagedType)]
/// struct InvalidField {
///     #[staged(bool)]
///     byte: u8,
/// }
/// ```
pub unsafe trait StagedType {
    /// The actual runtime type (e.g., i64 for i64)
    type RuntimeValue;

    /// Compile-time representation checks emitted by `#[derive(StagedType)]`.
    /// Manual unsafe implementations are responsible for validating their own
    /// layout and may use the default.
    #[doc(hidden)]
    const LAYOUT_VALID: () = ();

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

    /// Returns true if this is a fat pointer (e.g., slice reference).
    /// Fat pointers are 2 x i64 (ptr, len) that can be stored in separate
    /// registers instead of a stack slot for better performance.
    fn is_fat_pointer() -> bool {
        false
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
/// # Safety
///
/// [`Self::codegen_constant`] must produce the exact IR type and bit-level
/// representation declared by [`StagedType`] and must represent `value`.
pub unsafe trait ConstantType: StagedType {
    /// Generate code for a constant value
    fn codegen_constant(value: &Self::RuntimeValue, builder: &mut FunctionBuilder) -> Value;
}

/// Marker trait for types that are Copy at the semantic level.
///
/// This trait indicates that a type can be copied by value (in Rust semantics),
/// even though the Cranelift representation may use pointers for structs.
///
/// Primitive types (i64, f64, bool) are always CopyType.
/// Structs are CopyType only if all their fields are CopyType.
/// # Safety
///
/// Values using this staged representation must be valid to duplicate by
/// copying [`StagedType::size_of`] bytes, and the associated `RuntimeValue`
/// must have ordinary Rust copy semantics.
pub unsafe trait CopyType: StagedType<RuntimeValue: Copy> + Copy {}

mod direct_value_sealed {
    pub trait Sealed {}
}

/// Staged scalar values represented directly by one Cranelift SSA value.
///
/// This sealed bound excludes aggregate `CopyType` values whose staged value is
/// an address. Generic memory operations use it when they must load or store
/// the value directly rather than invoke aggregate copy lowering.
pub trait DirectValue: ConstantType + CopyType + direct_value_sealed::Sealed {}

/// Maps a staged function parameter to the Rust value accepted by one safe
/// invocation of generated code.
///
/// Unlike [`StagedType::RuntimeValue`], this mapping is generic over the
/// invocation lifetime. Reference markers can therefore expose `&'call T` or
/// `&'call mut T` without baking the marker's staging-only lifetime into a
/// compiled entry point.
///
/// # Safety
///
/// `Arg<'call>` must have the same calling-convention representation as
/// [`StagedType::RuntimeValue`]. Every safe `Arg<'call>` value must satisfy the
/// validity and aliasing requirements that generated code assumes for this
/// staged type for the duration of the call.
pub unsafe trait RuntimeParam: StagedType {
    type Arg<'call>;
}

/// Maps a staged function result to the Rust value returned by one safe
/// invocation of generated code.
///
/// # Safety
///
/// `Output<'call>` must have the same calling-convention representation as
/// [`StagedType::RuntimeValue`]. Generated code must only produce values valid
/// for `Output<'call>`; any borrow in the output must remain valid for the
/// invocation lifetime selected by the safe entry point.
pub unsafe trait RuntimeResult: StagedType {
    type Output<'call>;
}

macro_rules! impl_by_value_runtime_type {
    ($($ty:ty),+ $(,)?) => {
        $(
            unsafe impl RuntimeParam for $ty {
                type Arg<'call> = $ty;
            }

            unsafe impl RuntimeResult for $ty {
                type Output<'call> = $ty;
            }
        )+
    };
}

// =============================================================================
// Concrete Type Markers
// =============================================================================
//
// The staged type markers ARE the Rust primitives themselves (`u64`, `i32`,
// `f64`, `bool`, `()`, …) — there are no `XType` aliases. `StagedType` is
// implemented directly on each primitive below.

// =============================================================================
// StagedType implementations
// =============================================================================

macro_rules! impl_int_staged_type {
    ($ty:ty, $ir_ty:expr) => {
        unsafe impl StagedType for $ty {
            type RuntimeValue = $ty;

            fn cranelift_type() -> cranelift_codegen::ir::Type {
                $ir_ty
            }

            fn size_of() -> usize {
                std::mem::size_of::<$ty>()
            }

            fn align_of() -> usize {
                std::mem::align_of::<$ty>()
            }
        }

        unsafe impl ConstantType for $ty {
            fn codegen_constant(value: &$ty, builder: &mut FunctionBuilder) -> Value {
                builder.ins().iconst($ir_ty, *value as i64)
            }
        }

        unsafe impl CopyType for $ty {}
    };
}

impl_int_staged_type!(i8, types::I8);
impl_int_staged_type!(u8, types::I8);
impl_int_staged_type!(i16, types::I16);
impl_int_staged_type!(u16, types::I16);

unsafe impl StagedType for i64 {
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

unsafe impl ConstantType for i64 {
    fn codegen_constant(value: &i64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value)
    }
}

unsafe impl CopyType for i64 {}

unsafe impl StagedType for u64 {
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

unsafe impl ConstantType for u64 {
    fn codegen_constant(value: &u64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value as i64)
    }
}

unsafe impl CopyType for u64 {}

unsafe impl StagedType for i32 {
    type RuntimeValue = i32;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I32
    }

    fn size_of() -> usize {
        4
    }

    fn align_of() -> usize {
        4
    }
}

unsafe impl ConstantType for i32 {
    fn codegen_constant(value: &i32, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I32, *value as i64)
    }
}

unsafe impl CopyType for i32 {}

unsafe impl StagedType for u32 {
    type RuntimeValue = u32;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I32
    }

    fn size_of() -> usize {
        4
    }

    fn align_of() -> usize {
        4
    }
}

unsafe impl ConstantType for u32 {
    fn codegen_constant(value: &u32, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I32, *value as i64)
    }
}

unsafe impl CopyType for u32 {}

unsafe impl StagedType for f32 {
    type RuntimeValue = f32;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::F32
    }

    fn size_of() -> usize {
        4
    }

    fn align_of() -> usize {
        4
    }
}

unsafe impl ConstantType for f32 {
    fn codegen_constant(value: &f32, builder: &mut FunctionBuilder) -> Value {
        builder.ins().f32const(*value)
    }
}

unsafe impl CopyType for f32 {}

unsafe impl StagedType for bool {
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

unsafe impl ConstantType for bool {
    fn codegen_constant(value: &bool, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, if *value { 1 } else { 0 })
    }
}

unsafe impl CopyType for bool {}

unsafe impl StagedType for f64 {
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

unsafe impl ConstantType for f64 {
    fn codegen_constant(value: &f64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().f64const(*value)
    }
}

unsafe impl CopyType for f64 {}

unsafe impl StagedType for () {
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

unsafe impl ConstantType for () {
    fn codegen_constant(_value: &(), builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, 0)
    }
}

unsafe impl CopyType for () {}

impl_by_value_runtime_type!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, bool, ());

macro_rules! impl_direct_value {
    ($($ty:ty),+ $(,)?) => {
        $(
            impl direct_value_sealed::Sealed for $ty {}
            impl DirectValue for $ty {}
        )+
    };
}

impl_direct_value!(i8, u8, i16, u16, i32, u32, i64, u64, f32, f64, bool);
