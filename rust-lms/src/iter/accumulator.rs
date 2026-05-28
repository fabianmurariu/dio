//! Accumulator traits for fold operations.

use crate::func::VarBuilder;
use crate::staged::{assign, Staged, Var};
use crate::types::UnitType;

/// A type that can serve as a fold accumulator.
pub trait Accumulator: Sized {
    /// The staged variable type(s).
    type Vars: Clone;
    /// References to variables for the fold function.
    type Refs: Clone;

    /// Create and initialize accumulator variable(s).
    fn create_vars(builder: &mut VarBuilder, init: Self) -> Self::Vars;

    /// Get references from variables.
    fn as_refs(vars: &Self::Vars) -> Self::Refs;
}

/// Update expression for accumulators.
pub trait IntoAccumulatorUpdate<Acc: Accumulator> {
    /// Generate assignment(s) to update the accumulator variables.
    fn apply_update(self, vars: Acc::Vars) -> impl Staged<Out = UnitType>;
}

// =============================================================================
// Implementations for single values - concrete types
// =============================================================================

// Helper macro to reduce boilerplate for single value accumulators
macro_rules! impl_accumulator_for_primitive {
    ($rt:ty, $staged:ty) => {
        impl Accumulator for $rt {
            type Vars = Var<$staged>;
            type Refs = Var<$staged>;

            fn create_vars(builder: &mut VarBuilder, init: Self) -> Self::Vars {
                *builder.let_var(init)
            }

            fn as_refs(vars: &Self::Vars) -> Self::Refs {
                *vars
            }
        }

        impl<Expr> IntoAccumulatorUpdate<$rt> for Expr
        where
            Expr: Staged<Out = $staged>,
        {
            fn apply_update(self, var: Var<$staged>) -> impl Staged<Out = UnitType> {
                assign(var, self)
            }
        }
    };
}

// Implement for common numeric types
impl_accumulator_for_primitive!(i32, crate::types::I32Type);
impl_accumulator_for_primitive!(i64, crate::types::I64Type);
impl_accumulator_for_primitive!(u32, crate::types::U32Type);
impl_accumulator_for_primitive!(u64, crate::types::U64Type);
impl_accumulator_for_primitive!(f64, crate::types::F64Type);

// =============================================================================
// Implementations for 2-tuples
// =============================================================================

// Helper macro for 2-tuple accumulators
macro_rules! impl_accumulator_for_tuple2 {
    (($rt_a:ty, $staged_a:ty), ($rt_b:ty, $staged_b:ty)) => {
        impl Accumulator for ($rt_a, $rt_b) {
            type Vars = (Var<$staged_a>, Var<$staged_b>);
            type Refs = (Var<$staged_a>, Var<$staged_b>);

            fn create_vars(builder: &mut VarBuilder, (a, b): Self) -> Self::Vars {
                (*builder.let_var(a), *builder.let_var(b))
            }

            fn as_refs(vars: &Self::Vars) -> Self::Refs {
                *vars
            }
        }

        impl<ExprA, ExprB> IntoAccumulatorUpdate<($rt_a, $rt_b)> for (ExprA, ExprB)
        where
            ExprA: Staged<Out = $staged_a>,
            ExprB: Staged<Out = $staged_b>,
        {
            fn apply_update(
                self,
                (var_a, var_b): (Var<$staged_a>, Var<$staged_b>),
            ) -> impl Staged<Out = UnitType> {
                (assign(var_a, self.0), assign(var_b, self.1))
            }
        }
    };
}

// Common 2-tuple combinations
impl_accumulator_for_tuple2!((u64, crate::types::U64Type), (f64, crate::types::F64Type));
impl_accumulator_for_tuple2!((u64, crate::types::U64Type), (u64, crate::types::U64Type));
impl_accumulator_for_tuple2!((f64, crate::types::F64Type), (f64, crate::types::F64Type));
impl_accumulator_for_tuple2!((i64, crate::types::I64Type), (f64, crate::types::F64Type));
impl_accumulator_for_tuple2!((i64, crate::types::I64Type), (i64, crate::types::I64Type));

// =============================================================================
// Implementations for 3-tuples
// =============================================================================

// Helper macro for 3-tuple accumulators
macro_rules! impl_accumulator_for_tuple3 {
    (($rt_a:ty, $staged_a:ty), ($rt_b:ty, $staged_b:ty), ($rt_c:ty, $staged_c:ty)) => {
        impl Accumulator for ($rt_a, $rt_b, $rt_c) {
            type Vars = (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>);
            type Refs = (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>);

            fn create_vars(builder: &mut VarBuilder, (a, b, c): Self) -> Self::Vars {
                (*builder.let_var(a), *builder.let_var(b), *builder.let_var(c))
            }

            fn as_refs(vars: &Self::Vars) -> Self::Refs {
                *vars
            }
        }

        impl<ExprA, ExprB, ExprC> IntoAccumulatorUpdate<($rt_a, $rt_b, $rt_c)>
            for (ExprA, ExprB, ExprC)
        where
            ExprA: Staged<Out = $staged_a>,
            ExprB: Staged<Out = $staged_b>,
            ExprC: Staged<Out = $staged_c>,
        {
            fn apply_update(
                self,
                (var_a, var_b, var_c): (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>),
            ) -> impl Staged<Out = UnitType> {
                (
                    assign(var_a, self.0),
                    assign(var_b, self.1),
                    assign(var_c, self.2),
                )
            }
        }
    };
}

// Common 3-tuple combinations
impl_accumulator_for_tuple3!(
    (u64, crate::types::U64Type),
    (f64, crate::types::F64Type),
    (f64, crate::types::F64Type)
);

// =============================================================================
// Implementations for 4-tuples
// =============================================================================

// Helper macro for 4-tuple accumulators
macro_rules! impl_accumulator_for_tuple4 {
    (($rt_a:ty, $staged_a:ty), ($rt_b:ty, $staged_b:ty), ($rt_c:ty, $staged_c:ty), ($rt_d:ty, $staged_d:ty)) => {
        impl Accumulator for ($rt_a, $rt_b, $rt_c, $rt_d) {
            type Vars = (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>, Var<$staged_d>);
            type Refs = (Var<$staged_a>, Var<$staged_b>, Var<$staged_c>, Var<$staged_d>);

            fn create_vars(builder: &mut VarBuilder, (a, b, c, d): Self) -> Self::Vars {
                (
                    *builder.let_var(a),
                    *builder.let_var(b),
                    *builder.let_var(c),
                    *builder.let_var(d),
                )
            }

            fn as_refs(vars: &Self::Vars) -> Self::Refs {
                *vars
            }
        }

        impl<ExprA, ExprB, ExprC, ExprD> IntoAccumulatorUpdate<($rt_a, $rt_b, $rt_c, $rt_d)>
            for (ExprA, ExprB, ExprC, ExprD)
        where
            ExprA: Staged<Out = $staged_a>,
            ExprB: Staged<Out = $staged_b>,
            ExprC: Staged<Out = $staged_c>,
            ExprD: Staged<Out = $staged_d>,
        {
            fn apply_update(
                self,
                (var_a, var_b, var_c, var_d): (
                    Var<$staged_a>,
                    Var<$staged_b>,
                    Var<$staged_c>,
                    Var<$staged_d>,
                ),
            ) -> impl Staged<Out = UnitType> {
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

// Common 4-tuple combinations
impl_accumulator_for_tuple4!(
    (u64, crate::types::U64Type),
    (u64, crate::types::U64Type),
    (f64, crate::types::F64Type),
    (f64, crate::types::F64Type)
);

#[cfg(test)]
mod tests {
    use super::*;

    // Tests will be in integration tests where we can actually create VarBuilder
    // Here we just verify the trait implementations compile

    #[test]
    fn test_accumulator_traits_compile() {
        // Verify single value implements Accumulator
        fn _test_single<T: Accumulator>() {}
        _test_single::<f64>();
        _test_single::<u64>();
        _test_single::<i64>();

        // Verify tuples implement Accumulator
        _test_single::<(u64, f64)>();
        _test_single::<(u64, f64, f64)>();
        _test_single::<(u64, u64, f64, f64)>();
    }
}
