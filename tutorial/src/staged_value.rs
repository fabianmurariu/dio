//! Trait-based abstraction for staged values.
//!
//! This module provides the core `StagedValue` trait that unifies all staged types
//! (numeric, boolean, arrays, etc.) under a common interface. This allows `Expr` to
//! be more extensible - new types can be added without modifying the `Expr` enum.

use cranelift_codegen::ir::Value;
use cranelift_frontend::FunctionBuilder;
use std::fmt::{Debug, Display};

use crate::DataType;

/// A value that can be staged for JIT compilation.
///
/// This trait unifies all staged types (StagedNum<T>, StagedBool, etc.) under a
/// common interface. Any type that implements this trait can be used as a staged
/// value in expressions.
///
/// # Design
///
/// This trait uses dynamic dispatch (trait objects) to allow heterogeneous collections
/// of staged values. While this has a slight runtime cost compared to enum dispatch,
/// it provides much better extensibility - new types can be added without modifying
/// core code.
///
/// # Safety
///
/// Implementations must ensure that `codegen()` produces values of the type indicated
/// by `data_type()`. Mismatches will cause undefined behavior or JIT compilation failures.
pub trait StagedValue: Debug + Display {
    /// Get the runtime data type of this value
    fn data_type(&self) -> &DataType;

    /// Generate Cranelift IR code for this value
    ///
    /// # Safety
    ///
    /// The generated value must match the type returned by `data_type()`.
    fn codegen(&self, builder: &mut FunctionBuilder) -> Value;

    /// Clone this staged value into a Box
    ///
    /// This is needed because `Clone` is not object-safe, but we need to clone
    /// trait objects. Implementations should delegate to their regular `Clone` impl.
    fn clone_box(&self) -> Box<dyn StagedValue>;

    /// Attempt to downcast to a concrete type
    ///
    /// This is useful when you need to extract the underlying type for optimization
    /// or specific operations. Returns `None` if the downcast fails.
    fn as_any(&self) -> &dyn std::any::Any;
}

/// Implement Clone for Box<dyn StagedValue>
impl Clone for Box<dyn StagedValue> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}
