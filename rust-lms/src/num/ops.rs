//! Operation structs for numeric staged computations.

use cranelift_codegen::ir::Value;
use cranelift_frontend::FunctionBuilder;

use crate::staged::Staged;
use crate::types::{BoolType, StagedType};

use super::traits::{SupportsAdd, SupportsComparison, SupportsDiv, SupportsMul, SupportsSub};

// =============================================================================
// Arithmetic Operations
// =============================================================================

/// Addition operation: takes two Staged values with same Out type, produces same type
#[derive(Clone)]
pub struct Add<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Add<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsAdd,
{
    type Out = T;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        let lv = self.left.codegen(builder);
        let rv = self.right.codegen(builder);
        T::codegen_add(lv, rv, builder)
    }
}

/// Subtraction operation
#[derive(Clone)]
pub struct Sub<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Sub<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsSub,
{
    type Out = T;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        let lv = self.left.codegen(builder);
        let rv = self.right.codegen(builder);
        T::codegen_sub(lv, rv, builder)
    }
}

/// Multiplication operation
#[derive(Clone)]
pub struct Mul<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Mul<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsMul,
{
    type Out = T;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        let lv = self.left.codegen(builder);
        let rv = self.right.codegen(builder);
        T::codegen_mul(lv, rv, builder)
    }
}

/// Division operation
#[derive(Clone)]
pub struct Div<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Div<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsDiv,
{
    type Out = T;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        let lv = self.left.codegen(builder);
        let rv = self.right.codegen(builder);
        T::codegen_div(lv, rv, builder)
    }
}

// =============================================================================
// Comparison Operations (produce BoolType)
// =============================================================================

/// Less-than comparison: takes two values of same type, produces Bool
#[derive(Clone)]
pub struct Lt<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Lt<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsComparison,
{
    type Out = BoolType; // Always returns boolean!

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        let lv = self.left.codegen(builder);
        let rv = self.right.codegen(builder);
        T::codegen_lt(lv, rv, builder)
    }
}

/// Equality comparison: takes two values of same type, produces Bool
#[derive(Clone)]
pub struct Eq<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Eq<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsComparison,
{
    type Out = BoolType; // Always returns boolean!

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        let lv = self.left.codegen(builder);
        let rv = self.right.codegen(builder);
        T::codegen_eq(lv, rv, builder)
    }
}

// =============================================================================
// Helper Functions for Ergonomics
// =============================================================================

/// Create an addition operation
pub fn add<L, R, T>(left: L, right: R) -> Add<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsAdd,
{
    Add { left, right }
}

/// Create a subtraction operation
pub fn sub<L, R, T>(left: L, right: R) -> Sub<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsSub,
{
    Sub { left, right }
}

/// Create a multiplication operation
pub fn mul<L, R, T>(left: L, right: R) -> Mul<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsMul,
{
    Mul { left, right }
}

/// Create a division operation
pub fn div<L, R, T>(left: L, right: R) -> Div<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsDiv,
{
    Div { left, right }
}

/// Create a less-than comparison
pub fn lt<L, R, T>(left: L, right: R) -> Lt<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsComparison,
{
    Lt { left, right }
}

/// Create an equality comparison
pub fn eq<L, R, T>(left: L, right: R) -> Eq<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsComparison,
{
    Eq { left, right }
}