//! Tuple support for staged computations.
//!
//! This module implements `Staged` for tuples, where each element implements `Staged`.
//! The last element determines the return type, while all previous elements are executed
//! for their side effects (similar to `Seq`).
//!
//! # Example
//! ```ignore
//! let x = compiler.var_unchecked::<i64>();
//! let y = compiler.var_unchecked::<i64>();
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
            unsafe impl<$($T,)* $Last, OUT> Staged for ($($T,)* $Last,)
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
        // Note: Top-level tuples need Const, but operations can use primitives via IntoStaged
        let expr = (Const::<i64>::new(5), Const::<i64>::new(10));

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 10);
    }

    #[test]
    fn test_tuple_3() {
        let compiler = Compiler::new();

        // (5, 10, 15) => 15
        // Note: Top-level tuples need Const, but operations can use primitives via IntoStaged
        let expr = (Const::from(5i64), Const::from(10i64), Const::from(15i64));

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 15);
    }

    #[test]
    fn test_tuple_with_assignments() {
        let mut compiler = Compiler::new();

        let f = compiler.fun0("tuple_assign", |ctx| {
            let x = ctx.let_var(0i64);
            let y = ctx.let_var(0i64);
            (assign(*x, 5i64), assign(*y, 10i64), add(*x, *y))
        });

        let compiled = compiler.compile(call0(f)).expect("compilation failed");
        assert_eq!(compiled.run(), 15);
    }

    #[test]
    fn test_large_tuple() {
        let mut compiler = Compiler::new();

        let f = compiler.fun0("large_tuple", |ctx| {
            let v0 = ctx.let_var(1i64);
            let v1 = ctx.let_var(2i64);
            let v2 = ctx.let_var(3i64);
            let v3 = ctx.let_var(4i64);
            let v4 = ctx.let_var(5i64);
            let v5 = ctx.let_var(6i64);
            let v6 = ctx.let_var(7i64);
            let v7 = ctx.let_var(8i64);
            let v8 = ctx.let_var(9i64);
            let v9 = ctx.let_var(10i64);
            // Sum: 1+2+3+4+5+6+7+8+9+10 = 55
            (
                v0,
                v1,
                v2,
                v3,
                v4,
                v5,
                v6,
                v7,
                v8,
                v9,
                add(
                    add(
                        add(
                            add(add(add(add(add(add(*v0, *v1), *v2), *v3), *v4), *v5), *v6),
                            *v7,
                        ),
                        *v8,
                    ),
                    *v9,
                ),
            )
        });

        let compiled = compiler.compile(call0(f)).expect("compilation failed");
        assert_eq!(compiled.run(), 55);
    }

    #[test]
    fn test_nested_tuples() {
        let mut compiler = Compiler::new();

        let f = compiler.fun0("nested_tuples", |ctx| {
            let x = ctx.let_var(0i64);
            let y = ctx.let_var(0i64);
            ((x, y, assign(*x, 10i64), assign(*y, 20i64)), add(*x, *y))
        });

        let compiled = compiler.compile(call0(f)).expect("compilation failed");
        assert_eq!(compiled.run(), 30);
    }
}
