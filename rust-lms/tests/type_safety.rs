//! Tests to verify compile-time type safety guarantees

use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

#[test]
fn test_homogeneous_operations() {
    for_each_backend(|compiler| {
        let x = Const::<i64>::new(5);
        let y = Const::<i64>::new(10);

        // These should all work - same types
        let _expr1 = add(x, y);
        let _expr2 = sub(x, y);
        let _expr3 = mul(x, y);
        let _expr4 = div(x, y);

        // Compile to verify it's valid
        let compiled = compiler.compile(add(x, y)).expect("compilation failed");
        assert_eq!(compiled.run(), 15);
    });
}

#[test]
fn test_heterogeneous_operations() {
    for_each_backend(|compiler| {
        let x = Const::<i64>::new(5);
        let y = Const::<i64>::new(10);

        // Comparisons change type to Bool
        let comparison = lt(x, y); // 5 < 10 = true

        // Verify comparison works
        let compiled = compiler.compile(comparison).expect("compilation failed");
        assert!(compiled.run());
    });
}

#[test]
fn test_bool_comparison() {
    // Comparisons produce bool. To combine two booleans we use control
    // flow (`if_then_else`) rather than arithmetic — booleans are no longer
    // a `Num` so `eq(bool, bool)` is intentionally rejected at compile time.
    for_each_backend(|compiler| {
        let x = Const::<i64>::new(5);
        let y = Const::<i64>::new(10);

        // `lt` produces bool; `if_then_else` then yields an i64.
        let expr = if_then_else(lt(x, y), Const::<i64>::new(1), Const::<i64>::new(0));

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 1); // 5 < 10 → true → 1
    });
}

#[test]
fn test_varref_is_copy() {
    fn assert_copy<T: Copy>() {}

    assert_copy::<Var<SRef<i64>>>();

    for_each_backend(|mut compiler| {
        // Use x multiple times - no clone needed!
        let f = compiler.fun1("f", |_ctx, x: Var<i64>| {
            // x used 4 times in one expression
            add(add(x, x), add(x, x))
        });

        let expr = call1(f, Const::<i64>::new(5));
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 20); // 5 + 5 + 5 + 5 = 20
    });
}

#[test]
fn test_const_is_copy() {
    for_each_backend(|compiler| {
        let c = Const::<i64>::new(42);

        // Use c multiple times
        let _expr1 = add(c, c);
        let _expr2 = mul(c, c);
        let _expr3 = add(c, Const::new(1));

        let compiled = compiler.compile(add(c, c)).expect("compilation failed");
        assert_eq!(compiled.run(), 84);
    });
}

#[test]
fn test_nested_expressions() {
    for_each_backend(|mut compiler| {
        // Build (x + 3) * (10 - x) where x = 2
        // = (2 + 3) * (10 - 2) = 5 * 8 = 40
        let f = compiler.fun1("f", |_ctx, x: Var<i64>| {
            let a = Const::<i64>::new(3);
            let b = Const::<i64>::new(10);
            let left = add(x, a);
            let right = sub(b, x);
            mul(left, right)
        });

        let expr = call1(f, Const::<i64>::new(2));
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 40);
    });
}

#[test]
fn test_multiple_types_i64() {
    for_each_backend(|compiler| {
        let expr = add(Const::<i64>::new(10), Const::new(20));
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 30);
    });
}

#[test]
fn test_multiple_types_u64() {
    for_each_backend(|compiler| {
        let expr = mul(Const::<u64>::new(10), Const::new(20));
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 200);
    });
}

#[test]
fn test_multiple_types_f64() {
    for_each_backend(|compiler| {
        let expr = div(Const::<f64>::new(10.0), Const::new(4.0));
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert!((compiled.run() - 2.5).abs() < 0.0001);
    });
}

#[test]
fn test_boxing() {
    let c = Const::<i64>::new(42);
    let _boxed: Box<dyn Staged<Out = i64>> = c.boxed();

    // Can box operations too
    let expr = add(c, Const::new(5));
    let _boxed_expr: Box<dyn Staged<Out = i64>> = expr.boxed();
}

#[test]
fn test_function_reuse() {
    for_each_backend(|mut compiler| {
        // Define a function and call it multiple times
        let double = compiler.fun1("double", |_ctx, x: Var<i64>| add(x, x));

        // Call double twice with different arguments and add results
        // double(3) + double(4) = 6 + 8 = 14
        let expr = add(
            call1(double, Const::<i64>::new(3)),
            call1(double, Const::<i64>::new(4)),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 14);
    });
}

#[derive(Clone, Copy)]
struct MismatchedBackendAdd;

unsafe impl Staged for MismatchedBackendAdd {
    type Out = i64;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let integer = ctx.iconst(ScalarType::I64, 1);
        let float = ctx.f64const(2.0);
        Value::Scalar(ctx.iadd(integer, float))
    }
}

#[test]
fn neutral_ir_rejects_mismatched_operation_leaves() {
    for_each_backend(|compiler| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = compiler.compile(MismatchedBackendAdd);
        }));
        assert!(result.is_err(), "ill-typed iadd reached the backend");
    });
}

#[derive(Clone, Copy)]
struct IntegerUsedAsPointer;

unsafe impl Staged for IntegerUsedAsPointer {
    type Out = i64;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let integer = ctx.iconst(ScalarType::I64, 0);
        Value::Scalar(ctx.load(ScalarType::I64, integer, 0))
    }
}

#[test]
fn neutral_ir_rejects_integer_used_as_pointer() {
    for_each_backend(|compiler| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = compiler.compile(IntegerUsedAsPointer);
        }));
        assert!(
            result.is_err(),
            "integer leaf reached a pointer-only operation"
        );
    });
}

#[derive(Clone, Copy)]
struct WrongDeclaredResult;

unsafe impl Staged for WrongDeclaredResult {
    type Out = i64;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        Value::Scalar(ctx.f64const(1.0))
    }
}

#[test]
fn neutral_ir_rejects_leaf_that_disagrees_with_staged_output() {
    for_each_backend(|compiler| {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = compiler.compile(WrongDeclaredResult);
        }));
        assert!(
            result.is_err(),
            "ill-typed Staged result reached native code"
        );
    });
}

#[test]
fn null_pointer_branch_is_typed_on_every_backend() {
    for_each_backend(|mut compiler| {
        let function = compiler.fun0("null_pointer_branch", |ctx| {
            match_opt_ref(
                ctx,
                opt_ref_none::<i64>(),
                |_ctx, _reference| Const::<i64>::new(1),
                Const::<i64>::new(2),
            )
        });
        let compiled = compiler.compile(function).expect("compilation failed");
        assert_eq!(compiled.as_fn().call(), 2);
    });
}

// The following tests are compile-fail tests - they should NOT compile
// Uncomment them to verify that type errors are caught at compile time

// #[test]
// fn test_type_mismatch_fails() {
//     let compiler = Compiler::new();
//     let x = Const::<i64>::new(5);
//     let comparison = lt(x, Const::new(10));
//
//     // This should fail: can't add i64 and bool
//     let _bad = add(x, comparison);
// }

// #[test]
// fn test_mixed_numeric_types_fails() {
//     let compiler = Compiler::new();
//     let i64_val = Const::<i64>::new(5);
//     let f64_val = Const::<f64>::new(3.14);
//
//     // This should fail: can't add i64 and f64
//     let _bad = add(i64_val, f64_val);
// }
