//! Tests to verify compile-time type safety guarantees

use rust_lms::prelude::*;

#[test]
fn test_homogeneous_operations() {
    let compiler = Compiler::new();
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
}

#[test]
fn test_heterogeneous_operations() {
    let compiler = Compiler::new();
    let x = Const::<i64>::new(5);
    let y = Const::<i64>::new(10);

    // Comparisons change type to Bool
    let comparison = lt(x, y); // 5 < 10 = true

    // Verify comparison works
    let compiled = compiler.compile(comparison).expect("compilation failed");
    assert!(compiled.run());
}

#[test]
fn test_bool_comparison() {
    // Comparisons produce bool. To combine two booleans we use control
    // flow (`if_then_else`) rather than arithmetic — booleans are no longer
    // a `Num` so `eq(bool, bool)` is intentionally rejected at compile time.
    let compiler = Compiler::new();
    let x = Const::<i64>::new(5);
    let y = Const::<i64>::new(10);

    // `lt` produces bool; `if_then_else` then yields an i64.
    let expr = if_then_else(lt(x, y), Const::<i64>::new(1), Const::<i64>::new(0));

    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 1); // 5 < 10 → true → 1
}

#[test]
fn test_varref_is_copy() {
    let mut compiler = Compiler::new();

    // Use x multiple times - no clone needed!
    let f = compiler.fun1("f", |_ctx, x: Var<i64>| {
        // x used 4 times in one expression
        add(add(x, x), add(x, x))
    });

    let expr = call1(f, Const::<i64>::new(5));
    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 20); // 5 + 5 + 5 + 5 = 20
}

#[test]
fn test_const_is_copy() {
    let compiler = Compiler::new();
    let c = Const::<i64>::new(42);

    // Use c multiple times
    let _expr1 = add(c, c);
    let _expr2 = mul(c, c);
    let _expr3 = add(c, Const::new(1));

    let compiled = compiler.compile(add(c, c)).expect("compilation failed");
    assert_eq!(compiled.run(), 84);
}

#[test]
fn test_nested_expressions() {
    let mut compiler = Compiler::new();

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
}

#[test]
fn test_multiple_types_i64() {
    let compiler = Compiler::new();
    let expr = add(Const::<i64>::new(10), Const::new(20));
    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 30);
}

#[test]
fn test_multiple_types_u64() {
    let compiler = Compiler::new();
    let expr = mul(Const::<u64>::new(10), Const::new(20));
    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 200);
}

#[test]
fn test_multiple_types_f64() {
    let compiler = Compiler::new();
    let expr = div(Const::<f64>::new(10.0), Const::new(4.0));
    let compiled = compiler.compile(expr).expect("compilation failed");
    assert!((compiled.run() - 2.5).abs() < 0.0001);
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
    let mut compiler = Compiler::new();

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
