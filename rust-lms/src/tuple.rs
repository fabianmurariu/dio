//! Tuple support for staged computations.
//!
//! This module implements `Staged` for tuples, where each element implements `Staged`.
//! The last element determines the return type, while all previous elements are executed
//! for their side effects (similar to `Seq`).
//!
//! # Example
//! ```ignore
//! let x = compiler.var_unchecked::<I64Type>();
//! let y = compiler.var_unchecked::<I64Type>();
//!
//! // Instead of: seq(assign(x, Const::new(5)), seq(assign(y, Const::new(10)), add(x, y)))
//! // You can write:
//! let expr = (
//!     assign(x, Const::new(5)),
//!     assign(y, Const::new(10)),
//!     add(x, y)
//! );
//! ```

use crate::staged::{CompilationContext, Staged};
use crate::types::StagedType;
use cranelift_codegen::ir::Value;

// Macro to generate Staged implementations for tuples of various sizes
macro_rules! tuple_impls {
    ($(
        $len:tt => ($($T:ident $idx:tt),*) -> $Last:ident $last_idx:tt
    ),+ $(,)?) => {
        $(
            impl<$($T,)* $Last, OUT> Staged for ($($T,)* $Last,)
            where
                $($T: Staged,)*
                $Last: Staged<Out = OUT>,
                OUT: StagedType,
            {
                type Out = OUT;

                fn codegen(&self, ctx: &mut CompilationContext) -> Value {
                    // Execute all elements except the last for side effects
                    $(let _ = self.$idx.codegen(ctx);)*
                    // Return the last element's value
                    self.$last_idx.codegen(ctx)
                }
            }
        )+
    };
}

// Generate implementations for tuples from size 2 to 25
tuple_impls! {
    2 => (T0 0) -> T1 1,
    3 => (T0 0, T1 1) -> T2 2,
    4 => (T0 0, T1 1, T2 2) -> T3 3,
    5 => (T0 0, T1 1, T2 2, T3 3) -> T4 4,
    6 => (T0 0, T1 1, T2 2, T3 3, T4 4) -> T5 5,
    7 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5) -> T6 6,
    8 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6) -> T7 7,
    9 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7) -> T8 8,
    10 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8) -> T9 9,
    11 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9) -> T10 10,
    12 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10) -> T11 11,
    13 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11) -> T12 12,
    14 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12) -> T13 13,
    15 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13) -> T14 14,
    16 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14) -> T15 15,
    17 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15) -> T16 16,
    18 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15, T16 16) -> T17 17,
    19 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15, T16 16, T17 17) -> T18 18,
    20 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15, T16 16, T17 17, T18 18) -> T19 19,
    21 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15, T16 16, T17 17, T18 18, T19 19) -> T20 20,
    22 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15, T16 16, T17 17, T18 18, T19 19, T20 20) -> T21 21,
    23 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15, T16 16, T17 17, T18 18, T19 19, T20 20, T21 21) -> T22 22,
    24 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15, T16 16, T17 17, T18 18, T19 19, T20 20, T21 21, T22 22) -> T23 23,
    25 => (T0 0, T1 1, T2 2, T3 3, T4 4, T5 5, T6 6, T7 7, T8 8, T9 9, T10 10, T11 11, T12 12, T13 13, T14 14, T15 15, T16 16, T17 17, T18 18, T19 19, T20 20, T21 21, T22 22, T23 23) -> T24 24,
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;

    #[test]
    fn test_tuple_2() {
        let compiler = Compiler::new();

        // (5, 10) => 10 (first ignored, second returned)
        let expr = (
            Const::<I64Type>::new(5),
            Const::<I64Type>::new(10),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 10);
    }

    #[test]
    fn test_tuple_3() {
        let compiler = Compiler::new();

        // (5, 10, 15) => 15
        let expr = (
            Const::<I64Type>::new(5),
            Const::<I64Type>::new(10),
            Const::<I64Type>::new(15),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 15);
    }

    #[test]
    fn test_tuple_with_assignments() {
        let mut compiler = Compiler::new();

        let x = compiler.var_unchecked::<I64Type>();
        let y = compiler.var_unchecked::<I64Type>();

        // Assign x=5, y=10, return x+y
        let expr = (
            assign(x, Const::<I64Type>::new(5)),
            assign(y, Const::<I64Type>::new(10)),
            add(x, y),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 15);
    }

    #[test]
    fn test_tuple_vs_seq_equivalence() {
        let mut compiler1 = Compiler::new();
        let mut compiler2 = Compiler::new();

        let x1 = compiler1.var_unchecked::<I64Type>();
        let x2 = compiler2.var_unchecked::<I64Type>();

        // Using tuple
        let expr1 = (
            assign(x1, Const::<I64Type>::new(42)),
            mul(x1, Const::<I64Type>::new(2)),
        );

        // Using seq (equivalent)
        let expr2 = seq(
            assign(x2, Const::<I64Type>::new(42)),
            mul(x2, Const::<I64Type>::new(2)),
        );

        let result1 = compiler1.compile(expr1).expect("compilation failed").run();
        let result2 = compiler2.compile(expr2).expect("compilation failed").run();

        assert_eq!(result1, 84);
        assert_eq!(result2, 84);
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_large_tuple() {
        let mut compiler = Compiler::new();

        let vars: Vec<_> = (0..10).map(|_| compiler.var_unchecked::<I64Type>()).collect();

        // Assign 10 variables, then sum them all
        let expr = (
            assign(vars[0], Const::<I64Type>::new(1)),
            assign(vars[1], Const::<I64Type>::new(2)),
            assign(vars[2], Const::<I64Type>::new(3)),
            assign(vars[3], Const::<I64Type>::new(4)),
            assign(vars[4], Const::<I64Type>::new(5)),
            assign(vars[5], Const::<I64Type>::new(6)),
            assign(vars[6], Const::<I64Type>::new(7)),
            assign(vars[7], Const::<I64Type>::new(8)),
            assign(vars[8], Const::<I64Type>::new(9)),
            assign(vars[9], Const::<I64Type>::new(10)),
            // Sum: 1+2+3+4+5+6+7+8+9+10 = 55
            add(add(add(add(add(add(add(add(add(
                vars[0], vars[1]), vars[2]), vars[3]), vars[4]),
                vars[5]), vars[6]), vars[7]), vars[8]), vars[9]),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 55);
    }

    #[test]
    fn test_nested_tuples() {
        let mut compiler = Compiler::new();

        let x = compiler.var_unchecked::<I64Type>();
        let y = compiler.var_unchecked::<I64Type>();

        // Nested tuples work too
        let expr = (
            (
                assign(x, Const::<I64Type>::new(10)),
                assign(y, Const::<I64Type>::new(20)),
            ),
            add(x, y),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 30);
    }

    #[test]
    fn test_while_loop_with_tuple() {
        let mut compiler = Compiler::new();

        let i = compiler.var_unchecked::<I64Type>();
        let sum = compiler.var_unchecked::<I64Type>();

        // Compute sum of 1..=10 using tuple syntax instead of deeply nested seq
        let sum_to_10 = compiler.fun1("sum_to_10", |_n: VarRef<I64Type>| {
            (
                assign(i, Const::<I64Type>::new(1)),
                assign(sum, Const::<I64Type>::new(0)),
                while_loop(
                    lt(i, Const::<I64Type>::new(11)),
                    (
                        assign(sum, add(sum, i)),
                        assign(i, add(i, Const::<I64Type>::new(1))),
                    ),
                ),
                sum,
            )
        });

        let compiled = compiler.compile(call1(sum_to_10, Const::<I64Type>::new(0)))
            .expect("compilation failed");
        assert_eq!(compiled.run(), 55); // 1+2+3+...+10 = 55
    }
}

