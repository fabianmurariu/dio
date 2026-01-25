//! Operation structs for numeric staged computations.

use cranelift_codegen::ir::{InstBuilder, Value};

use crate::staged::{CompilationContext, IntoStaged, Staged};
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

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_add(lv, rv, ctx.builder)
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

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_sub(lv, rv, ctx.builder)
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

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_mul(lv, rv, ctx.builder)
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

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_div(lv, rv, ctx.builder)
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

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_lt(lv, rv, ctx.builder)
    }
}

#[derive(Clone)]
pub struct Gt<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Gt<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: StagedType + SupportsComparison,
{
    type Out = BoolType; // Always returns boolean!

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_gt(lv, rv, ctx.builder)
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

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_eq(lv, rv, ctx.builder)
    }
}

// =============================================================================
// Helper Functions for Ergonomics
// =============================================================================

/// Create an addition operation
///
/// Accepts any values that can be converted into staged expressions.
/// This allows ergonomic usage like `add(x, 5i64)` instead of
/// `add(x, Const::<I64Type>::new(5))`.
pub fn add<T, L, R>(left: L, right: R) -> Add<L::Staged, R::Staged>
where
    T: StagedType + SupportsAdd,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Add {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Create a subtraction operation
pub fn sub<T, L, R>(left: L, right: R) -> Sub<L::Staged, R::Staged>
where
    T: StagedType + SupportsSub,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Sub {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Create a multiplication operation
pub fn mul<T, L, R>(left: L, right: R) -> Mul<L::Staged, R::Staged>
where
    T: StagedType + SupportsMul,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Mul {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Create a division operation
pub fn div<T, L, R>(left: L, right: R) -> Div<L::Staged, R::Staged>
where
    T: StagedType + SupportsDiv,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Div {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Create a less-than comparison
///
/// Accepts any values that can be converted into staged expressions.
/// This allows ergonomic usage like `lt(x, 100i64)` instead of
/// `lt(x, Const::<I64Type>::new(100))`.
pub fn lt<T, L, R>(left: L, right: R) -> Lt<L::Staged, R::Staged>
where
    T: StagedType + SupportsComparison,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Lt {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Create a greater-than comparison
pub fn gt<T, L, R>(left: L, right: R) -> Gt<L::Staged, R::Staged>
where
    T: StagedType + SupportsComparison,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Gt {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Create an equality comparison
pub fn eq<T, L, R>(left: L, right: R) -> Eq<L::Staged, R::Staged>
where
    T: StagedType + SupportsComparison,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Eq {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

// =============================================================================
// Conditional Select Operation (branchless)
// =============================================================================

/// Select operation: if condition is true, return if_true, else return if_false
/// This compiles to a branchless cmov instruction on x86-64.
#[derive(Clone)]
pub struct Select<C, T, F> {
    condition: C,
    if_true: T,
    if_false: F,
}

impl<C, T, F, Out> Staged for Select<C, T, F>
where
    C: Staged<Out = BoolType>,
    T: Staged<Out = Out>,
    F: Staged<Out = Out>,
    Out: StagedType,
{
    type Out = Out;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let cond = self.condition.codegen(ctx);
        let true_val = self.if_true.codegen(ctx);
        let false_val = self.if_false.codegen(ctx);
        ctx.builder.ins().select(cond, true_val, false_val)
    }
}

/// Create a conditional select operation (branchless)
///
/// Returns `if_true` if `condition` is true, otherwise returns `if_false`.
/// This compiles to a branchless cmov instruction.
///
/// # Example
/// ```ignore
/// // Branchless min
/// let new_min = select(lt(*val, *min), *val, *min);
/// ```
pub fn select<C, T, F, Out>(condition: C, if_true: T, if_false: F) -> Select<C::Staged, T::Staged, F::Staged>
where
    C: IntoStaged<BoolType>,
    T: IntoStaged<Out>,
    F: IntoStaged<Out>,
    Out: StagedType,
{
    Select {
        condition: condition.into_staged(),
        if_true: if_true.into_staged(),
        if_false: if_false.into_staged(),
    }
}

/// Branchless minimum of two values
pub fn min<T, L, R>(left: L, right: R) -> Select<Lt<L::Staged, R::Staged>, L::Staged, R::Staged>
where
    T: StagedType + SupportsComparison,
    L: IntoStaged<T> + Clone,
    R: IntoStaged<T> + Clone,
    L::Staged: Clone,
    R::Staged: Clone,
{
    let left_s = left.into_staged();
    let right_s = right.into_staged();
    Select {
        condition: Lt { left: left_s.clone(), right: right_s.clone() },
        if_true: left_s,
        if_false: right_s,
    }
}

/// Branchless maximum of two values
pub fn max<T, L, R>(left: L, right: R) -> Select<Gt<L::Staged, R::Staged>, L::Staged, R::Staged>
where
    T: StagedType + SupportsComparison,
    L: IntoStaged<T> + Clone,
    R: IntoStaged<T> + Clone,
    L::Staged: Clone,
    R::Staged: Clone,
{
    let left_s = left.into_staged();
    let right_s = right.into_staged();
    Select {
        condition: Gt { left: left_s.clone(), right: right_s.clone() },
        if_true: left_s,
        if_false: right_s,
    }
}
