//! Accumulator traits and fold result type for staged iterators.

use crate::func::VarBuilder;
use crate::staged::{assign, Assign, CompilationContext, Const, LetVar, Staged, Var};
use crate::types::{StagedType, UnitType};
use cranelift_codegen::ir::Value;

// =============================================================================
// FoldExpr - wraps init + loop + result into a single Staged value
// =============================================================================

/// Result of a single-accumulator fold (sum, count, min, max).
///
/// Codegens as: initialize accumulator → run loop → return final value.
/// This lets `sum`/`count`/`min`/`max` be used directly as staged expressions.
pub struct FoldExpr<Init, Loop, T: StagedType> {
    pub(crate) init: Init,
    pub(crate) loop_expr: Loop,
    pub(crate) result_var: Var<T>,
}

impl<Init, Loop, T> Staged for FoldExpr<Init, Loop, T>
where
    Init: Staged<Out = UnitType>,
    Loop: Staged<Out = UnitType>,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.init.codegen(ctx);
        self.loop_expr.codegen(ctx);
        self.result_var.codegen(ctx)
    }
}

// =============================================================================
// Accumulator trait
// =============================================================================

/// A type that can serve as a fold accumulator.
///
/// `create_vars` returns both the initialization expression (a `LetVar` or
/// tuple thereof) and the plain `Var` references.  The caller is responsible
/// for including the init expression in the staged sequence before the loop.
pub trait Accumulator: Sized {
    /// The staged variable type(s) — `Var<T>` for single, tuple of `Var` for multi.
    type Vars: Clone;
    /// References passed into the fold closure.
    type Refs: Clone;
    /// Staged expression that initializes the accumulator variable(s).
    /// Must implement `Staged<Out = UnitType>`.
    type Init: Staged<Out = UnitType>;

    /// Allocate and initialize accumulator variable(s).
    ///
    /// Returns `(init_expr, vars)`. The caller must include `init_expr` in the
    /// staged sequence (before the loop) so that the variable is declared and
    /// given its initial value in the generated code.
    fn create_vars(builder: &mut VarBuilder, init: Self) -> (Self::Init, Self::Vars);

    /// Extract references from variables for passing to the fold closure.
    fn as_refs(vars: &Self::Vars) -> Self::Refs;
}

// =============================================================================
// IntoAccumulatorUpdate trait — concrete associated type so Clone is provable
// =============================================================================

/// Converts a user expression into a sequence of variable-update assignments.
///
/// The associated `Update` type is concrete so callers can require `Clone`
/// on it — which is necessary because loop bodies must be cloneable.
pub trait IntoAccumulatorUpdate<Acc: Accumulator> {
    /// The concrete update expression type — `Assign<Var<T>, Expr>` for single,
    /// tuple of Assigns for multi. Must be `Staged<Out = UnitType> + Clone`.
    type Update: Staged<Out = UnitType> + Clone;

    /// Produce assignment(s) that update the accumulator variables.
    fn apply_update(self, vars: Acc::Vars) -> Self::Update;
}

// =============================================================================
// Single-value accumulator impls
// =============================================================================

macro_rules! impl_accumulator_for_primitive {
    ($rt:ty, $staged:ty) => {
        impl Accumulator for $rt {
            type Vars = Var<$staged>;
            type Refs = Var<$staged>;
            type Init = LetVar<$staged, Const<$staged>>;

            fn create_vars(builder: &mut VarBuilder, init: Self) -> (Self::Init, Self::Vars) {
                let lv = builder.let_var(init);
                let var = lv.var();
                (lv, var)
            }

            fn as_refs(vars: &Self::Vars) -> Self::Refs {
                *vars
            }
        }

        impl<Expr> IntoAccumulatorUpdate<$rt> for Expr
        where
            Expr: Staged<Out = $staged> + Clone,
        {
            type Update = Assign<Var<$staged>, Expr>;

            fn apply_update(self, var: Var<$staged>) -> Self::Update {
                assign(var, self)
            }
        }
    };
}

impl_accumulator_for_primitive!(i32, crate::types::I32Type);
impl_accumulator_for_primitive!(i64, crate::types::I64Type);
impl_accumulator_for_primitive!(u32, crate::types::U32Type);
impl_accumulator_for_primitive!(u64, crate::types::U64Type);
impl_accumulator_for_primitive!(f64, crate::types::F64Type);

// =============================================================================
// 2-tuple accumulator impls
// =============================================================================

macro_rules! impl_accumulator_for_tuple2 {
    (($rt_a:ty, $staged_a:ty), ($rt_b:ty, $staged_b:ty)) => {
        impl Accumulator for ($rt_a, $rt_b) {
            type Vars = (Var<$staged_a>, Var<$staged_b>);
            type Refs = (Var<$staged_a>, Var<$staged_b>);
            type Init = (
                LetVar<$staged_a, Const<$staged_a>>,
                LetVar<$staged_b, Const<$staged_b>>,
            );

            fn create_vars(
                builder: &mut VarBuilder,
                (a, b): Self,
            ) -> (Self::Init, Self::Vars) {
                let lv_a = builder.let_var(a);
                let var_a = lv_a.var();
                let lv_b = builder.let_var(b);
                let var_b = lv_b.var();
                ((lv_a, lv_b), (var_a, var_b))
            }

            fn as_refs(vars: &Self::Vars) -> Self::Refs {
                *vars
            }
        }

        impl<ExprA, ExprB> IntoAccumulatorUpdate<($rt_a, $rt_b)> for (ExprA, ExprB)
        where
            ExprA: Staged<Out = $staged_a> + Clone,
            ExprB: Staged<Out = $staged_b> + Clone,
        {
            type Update = (
                Assign<Var<$staged_a>, ExprA>,
                Assign<Var<$staged_b>, ExprB>,
            );

            fn apply_update(
                self,
                (var_a, var_b): (Var<$staged_a>, Var<$staged_b>),
            ) -> Self::Update {
                (assign(var_a, self.0), assign(var_b, self.1))
            }
        }
    };
}

impl_accumulator_for_tuple2!((u64, crate::types::U64Type), (f64, crate::types::F64Type));
impl_accumulator_for_tuple2!((u64, crate::types::U64Type), (u64, crate::types::U64Type));
impl_accumulator_for_tuple2!((f64, crate::types::F64Type), (f64, crate::types::F64Type));
impl_accumulator_for_tuple2!((i64, crate::types::I64Type), (f64, crate::types::F64Type));
impl_accumulator_for_tuple2!((i64, crate::types::I64Type), (i64, crate::types::I64Type));

// =============================================================================
// 3-tuple accumulator impls
// =============================================================================

macro_rules! impl_accumulator_for_tuple3 {
    (($rt_a:ty, $staged_a:ty), ($rt_b:ty, $staged_b:ty), ($rt_c:ty, $staged_c:ty)) => {
        impl Accumulator for ($rt_a, $rt_b, $rt_c) {
            type Vars = (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>);
            type Refs = (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>);
            type Init = (
                LetVar<$staged_a, Const<$staged_a>>,
                LetVar<$staged_b, Const<$staged_b>>,
                LetVar<$staged_c, Const<$staged_c>>,
            );

            fn create_vars(
                builder: &mut VarBuilder,
                (a, b, c): Self,
            ) -> (Self::Init, Self::Vars) {
                let lv_a = builder.let_var(a);
                let var_a = lv_a.var();
                let lv_b = builder.let_var(b);
                let var_b = lv_b.var();
                let lv_c = builder.let_var(c);
                let var_c = lv_c.var();
                ((lv_a, lv_b, lv_c), (var_a, var_b, var_c))
            }

            fn as_refs(vars: &Self::Vars) -> Self::Refs {
                *vars
            }
        }

        impl<ExprA, ExprB, ExprC> IntoAccumulatorUpdate<($rt_a, $rt_b, $rt_c)>
            for (ExprA, ExprB, ExprC)
        where
            ExprA: Staged<Out = $staged_a> + Clone,
            ExprB: Staged<Out = $staged_b> + Clone,
            ExprC: Staged<Out = $staged_c> + Clone,
        {
            type Update = (
                Assign<Var<$staged_a>, ExprA>,
                Assign<Var<$staged_b>, ExprB>,
                Assign<Var<$staged_c>, ExprC>,
            );

            fn apply_update(
                self,
                (var_a, var_b, var_c): (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>),
            ) -> Self::Update {
                (assign(var_a, self.0), assign(var_b, self.1), assign(var_c, self.2))
            }
        }
    };
}

impl_accumulator_for_tuple3!(
    (u64, crate::types::U64Type),
    (f64, crate::types::F64Type),
    (f64, crate::types::F64Type)
);

// =============================================================================
// 4-tuple accumulator impls
// =============================================================================

macro_rules! impl_accumulator_for_tuple4 {
    (($rt_a:ty, $staged_a:ty), ($rt_b:ty, $staged_b:ty), ($rt_c:ty, $staged_c:ty), ($rt_d:ty, $staged_d:ty)) => {
        impl Accumulator for ($rt_a, $rt_b, $rt_c, $rt_d) {
            type Vars = (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>, Var<$staged_d>);
            type Refs = (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>, Var<$staged_d>);
            type Init = (
                LetVar<$staged_a, Const<$staged_a>>,
                LetVar<$staged_b, Const<$staged_b>>,
                LetVar<$staged_c, Const<$staged_c>>,
                LetVar<$staged_d, Const<$staged_d>>,
            );

            fn create_vars(
                builder: &mut VarBuilder,
                (a, b, c, d): Self,
            ) -> (Self::Init, Self::Vars) {
                let lv_a = builder.let_var(a);
                let var_a = lv_a.var();
                let lv_b = builder.let_var(b);
                let var_b = lv_b.var();
                let lv_c = builder.let_var(c);
                let var_c = lv_c.var();
                let lv_d = builder.let_var(d);
                let var_d = lv_d.var();
                ((lv_a, lv_b, lv_c, lv_d), (var_a, var_b, var_c, var_d))
            }

            fn as_refs(vars: &Self::Vars) -> Self::Refs {
                *vars
            }
        }

        impl<ExprA, ExprB, ExprC, ExprD> IntoAccumulatorUpdate<($rt_a, $rt_b, $rt_c, $rt_d)>
            for (ExprA, ExprB, ExprC, ExprD)
        where
            ExprA: Staged<Out = $staged_a> + Clone,
            ExprB: Staged<Out = $staged_b> + Clone,
            ExprC: Staged<Out = $staged_c> + Clone,
            ExprD: Staged<Out = $staged_d> + Clone,
        {
            type Update = (
                Assign<Var<$staged_a>, ExprA>,
                Assign<Var<$staged_b>, ExprB>,
                Assign<Var<$staged_c>, ExprC>,
                Assign<Var<$staged_d>, ExprD>,
            );

            fn apply_update(
                self,
                (var_a, var_b, var_c, var_d): (
                    Var<$staged_a>,
                    Var<$staged_b>,
                    Var<$staged_c>,
                    Var<$staged_d>,
                ),
            ) -> Self::Update {
                (
                    assign(var_a, self.0),
                    assign(var_b, self.1),
                    assign(var_c, self.2),
                    assign(var_d, self.3),
                )
            }
        }
    };
}

impl_accumulator_for_tuple4!(
    (u64, crate::types::U64Type),
    (u64, crate::types::U64Type),
    (f64, crate::types::F64Type),
    (f64, crate::types::F64Type)
);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_accumulator_traits_compile() {
        fn _test<T: Accumulator>() {}
        _test::<f64>();
        _test::<u64>();
        _test::<i64>();
        _test::<(u64, f64)>();
        _test::<(u64, f64, f64)>();
        _test::<(u64, u64, f64, f64)>();
    }
}
