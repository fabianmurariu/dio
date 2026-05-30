//! Operation structs for numeric staged computations.

use cranelift_codegen::ir::{InstBuilder, Value};

use crate::staged::{CompilationContext, Const, IntoStaged, Staged, Var};
use crate::types::{BoolType, StagedType};

use super::traits::{IntNum, Num};

// =============================================================================
// Arithmetic Operations
// =============================================================================

/// Addition operation: takes two Staged values with same Out type, produces same type.
#[derive(Clone)]
pub struct Add<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Add<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: Num,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_add(lv, rv, ctx.builder)
    }
}

/// Subtraction operation.
#[derive(Clone)]
pub struct Sub<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Sub<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: Num,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_sub(lv, rv, ctx.builder)
    }
}

/// Multiplication operation.
#[derive(Clone)]
pub struct Mul<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Mul<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: Num,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_mul(lv, rv, ctx.builder)
    }
}

/// Division operation.
#[derive(Clone)]
pub struct Div<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Div<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: Num,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_div(lv, rv, ctx.builder)
    }
}

/// Remainder (modulo) operation. Only integers support this.
#[derive(Clone)]
pub struct Rem<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Rem<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: IntNum,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_rem(lv, rv, ctx.builder)
    }
}

// =============================================================================
// Comparison Operations (produce BoolType)
// =============================================================================

/// Less-than comparison.
#[derive(Clone)]
pub struct Lt<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Lt<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: Num,
{
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_lt(lv, rv, ctx.builder)
    }
}

/// Greater-than comparison.
#[derive(Clone)]
pub struct Gt<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Gt<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: Num,
{
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_gt(lv, rv, ctx.builder)
    }
}

/// Equality comparison.
#[derive(Clone)]
pub struct Eq<L, R> {
    left: L,
    right: R,
}

impl<L, R, T> Staged for Eq<L, R>
where
    L: Staged<Out = T>,
    R: Staged<Out = T>,
    T: Num,
{
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let lv = self.left.codegen(ctx);
        let rv = self.right.codegen(ctx);
        T::codegen_eq(lv, rv, ctx.builder)
    }
}

// =============================================================================
// Helper Functions for Ergonomics
// =============================================================================

/// Build a staged addition. Accepts any `IntoStaged<T>` where `T: Num`.
pub fn add<T, L, R>(left: L, right: R) -> Add<L::Staged, R::Staged>
where
    T: Num,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Add {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Build a staged subtraction.
pub fn sub<T, L, R>(left: L, right: R) -> Sub<L::Staged, R::Staged>
where
    T: Num,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Sub {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Build a staged multiplication.
pub fn mul<T, L, R>(left: L, right: R) -> Mul<L::Staged, R::Staged>
where
    T: Num,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Mul {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Build a staged division.
pub fn div<T, L, R>(left: L, right: R) -> Div<L::Staged, R::Staged>
where
    T: Num,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Div {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Build a staged remainder (modulo). Only integers support this.
pub fn rem<T, L, R>(left: L, right: R) -> Rem<L::Staged, R::Staged>
where
    T: IntNum,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Rem {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Build a staged less-than comparison.
pub fn lt<T, L, R>(left: L, right: R) -> Lt<L::Staged, R::Staged>
where
    T: Num,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Lt {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Build a staged greater-than comparison.
pub fn gt<T, L, R>(left: L, right: R) -> Gt<L::Staged, R::Staged>
where
    T: Num,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
{
    Gt {
        left: left.into_staged(),
        right: right.into_staged(),
    }
}

/// Build a staged equality comparison.
pub fn eq<T, L, R>(left: L, right: R) -> Eq<L::Staged, R::Staged>
where
    T: Num,
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

/// Select: branchless `if condition { if_true } else { if_false }`.
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

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let cond = self.condition.codegen(ctx);
        let true_val = self.if_true.codegen(ctx);
        let false_val = self.if_false.codegen(ctx);
        ctx.builder.ins().select(cond, true_val, false_val)
    }
}

pub fn select<C, T, F, Out>(
    condition: C,
    if_true: T,
    if_false: F,
) -> Select<C::Staged, T::Staged, F::Staged>
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

/// Branchless minimum.
pub fn min<T, L, R>(left: L, right: R) -> Select<Lt<L::Staged, R::Staged>, L::Staged, R::Staged>
where
    T: Num,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
    L::Staged: Clone,
    R::Staged: Clone,
{
    let left_s = left.into_staged();
    let right_s = right.into_staged();
    Select {
        condition: Lt {
            left: left_s.clone(),
            right: right_s.clone(),
        },
        if_true: left_s,
        if_false: right_s,
    }
}

/// Branchless maximum.
pub fn max<T, L, R>(left: L, right: R) -> Select<Gt<L::Staged, R::Staged>, L::Staged, R::Staged>
where
    T: Num,
    L: IntoStaged<T>,
    R: IntoStaged<T>,
    L::Staged: Clone,
    R::Staged: Clone,
{
    let left_s = left.into_staged();
    let right_s = right.into_staged();
    Select {
        condition: Gt {
            left: left_s.clone(),
            right: right_s.clone(),
        },
        if_true: left_s,
        if_false: right_s,
    }
}

// =============================================================================
// std::ops integration — let users write `x + y`, `var % 2`, etc.
//
// We impl each of {Add, Sub, Mul, Div, Rem} for the staged carriers (Var<T>,
// Const<T>, and each op struct). The macro keeps the boilerplate to a single
// line per carrier.
// =============================================================================

/// Impls `core::ops::{Add, Sub, Mul, Div}` (requires `Num`) for the given type
/// pattern. The type must implement `Staged<Out = T>` for some `T: Num`.
macro_rules! impl_num_ops_for {
    ([$($gen:tt)*] $self:ty) => {
        impl<$($gen)*, __Out, __R> ::core::ops::Add<__R> for $self
        where
            $self: Staged<Out = __Out> + 'static,
            __Out: Num + 'static,
            __R: IntoStaged<__Out>,
            __R::Staged: 'static,
        {
            type Output = Add<Self, __R::Staged>;
            fn add(self, rhs: __R) -> Self::Output {
                add::<__Out, _, _>(self, rhs)
            }
        }

        impl<$($gen)*, __Out, __R> ::core::ops::Sub<__R> for $self
        where
            $self: Staged<Out = __Out> + 'static,
            __Out: Num + 'static,
            __R: IntoStaged<__Out>,
            __R::Staged: 'static,
        {
            type Output = Sub<Self, __R::Staged>;
            fn sub(self, rhs: __R) -> Self::Output {
                sub::<__Out, _, _>(self, rhs)
            }
        }

        impl<$($gen)*, __Out, __R> ::core::ops::Mul<__R> for $self
        where
            $self: Staged<Out = __Out> + 'static,
            __Out: Num + 'static,
            __R: IntoStaged<__Out>,
            __R::Staged: 'static,
        {
            type Output = Mul<Self, __R::Staged>;
            fn mul(self, rhs: __R) -> Self::Output {
                mul::<__Out, _, _>(self, rhs)
            }
        }

        impl<$($gen)*, __Out, __R> ::core::ops::Div<__R> for $self
        where
            $self: Staged<Out = __Out> + 'static,
            __Out: Num + 'static,
            __R: IntoStaged<__Out>,
            __R::Staged: 'static,
        {
            type Output = Div<Self, __R::Staged>;
            fn div(self, rhs: __R) -> Self::Output {
                div::<__Out, _, _>(self, rhs)
            }
        }
    };
}

/// Impls `core::ops::Rem` (requires `IntNum`) for the given type pattern.
macro_rules! impl_rem_op_for {
    ([$($gen:tt)*] $self:ty) => {
        impl<$($gen)*, __Out, __R> ::core::ops::Rem<__R> for $self
        where
            $self: Staged<Out = __Out> + 'static,
            __Out: IntNum + 'static,
            __R: IntoStaged<__Out>,
            __R::Staged: 'static,
        {
            type Output = Rem<Self, __R::Staged>;
            fn rem(self, rhs: __R) -> Self::Output {
                rem::<__Out, _, _>(self, rhs)
            }
        }
    };
}

// Carriers: Var<T>, Const<T>, and the op structs themselves.
impl_num_ops_for!([T: StagedType] Var<T>);
impl_rem_op_for!([T: StagedType] Var<T>);

impl_num_ops_for!([T: crate::types::ConstantType] Const<T>);
impl_rem_op_for!([T: crate::types::ConstantType] Const<T>);

impl_num_ops_for!([L, R] Add<L, R>);
impl_rem_op_for!([L, R] Add<L, R>);

impl_num_ops_for!([L, R] Sub<L, R>);
impl_rem_op_for!([L, R] Sub<L, R>);

impl_num_ops_for!([L, R] Mul<L, R>);
impl_rem_op_for!([L, R] Mul<L, R>);

impl_num_ops_for!([L, R] Div<L, R>);
impl_rem_op_for!([L, R] Div<L, R>);

impl_num_ops_for!([L, R] Rem<L, R>);
impl_rem_op_for!([L, R] Rem<L, R>);

// Conditional select carries through operators too.
impl_num_ops_for!([C, T, F] Select<C, T, F>);
impl_rem_op_for!([C, T, F] Select<C, T, F>);

// LetVar acts like a Var when used in expressions.
impl_num_ops_for!([T: StagedType, E] crate::staged::LetVar<T, E>);
impl_rem_op_for!([T: StagedType, E] crate::staged::LetVar<T, E>);

// Slice access carriers (length, element reads). The carriers are now unified
// across immutable and mutable slices, so one impl each covers both.
impl_num_ops_for!([S] crate::slice::SliceLen<S>);
impl_rem_op_for!([S] crate::slice::SliceLen<S>);

impl_num_ops_for!([S, I] crate::slice::SliceGetUnchecked<S, I>);
impl_rem_op_for!([S, I] crate::slice::SliceGetUnchecked<S, I>);

// Reference-load carriers.
impl_num_ops_for!(['a, P] crate::refer::LoadRef<'a, P>);
impl_rem_op_for!(['a, P] crate::refer::LoadRef<'a, P>);

impl_num_ops_for!(['a, P] crate::refer::LoadMutRef<'a, P>);
impl_rem_op_for!(['a, P] crate::refer::LoadMutRef<'a, P>);

impl_num_ops_for!(['a, P, I] crate::refer::ArrayIndex<'a, P, I>);
impl_rem_op_for!(['a, P, I] crate::refer::ArrayIndex<'a, P, I>);

// Struct field accessors.
impl_num_ops_for!([P, F] crate::r#struct::LoadField<P, F>);
impl_rem_op_for!([P, F] crate::r#struct::LoadField<P, F>);

impl_num_ops_for!([P, F] crate::r#struct::FieldPath<P, F>);
impl_rem_op_for!([P, F] crate::r#struct::FieldPath<P, F>);
