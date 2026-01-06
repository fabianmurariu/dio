//! Tests to verify compile-time type safety guarantees

use cranelift_frontend::Variable;
use rust_lms::prelude::*;

#[test]
fn test_homogeneous_operations() {
    let x = Var::<I64Type>::new(Variable::from_u32(0));
    let y = Const::<I64Type>::new(10);

    // These should all work - same types
    let _expr1 = add(x, y);
    let _expr2 = sub(x, y);
    let _expr3 = mul(x, y);
    let _expr4 = div(x, y);
}

#[test]
fn test_heterogeneous_operations() {
    let x = Var::<I64Type>::new(Variable::from_u32(0));
    let y = Const::<I64Type>::new(10);

    // Comparisons change type to Bool
    let comparison = lt(x, y);
    let equality = eq(x, y);

    // We can compare the results (both are BoolType)
    let _bool_comparison = eq(comparison, equality);
}

#[test]
fn test_var_is_copy() {
    let x = Var::<I64Type>::new(Variable::from_u32(0));
    let y = Const::<I64Type>::new(5);

    // Use x multiple times - no clone needed!
    let _expr1 = add(x, y);
    let _expr2 = mul(x, y);
    let _expr3 = sub(x, y);
    let _expr4 = add(x, x); // Can even use x twice in one operation
}

#[test]
fn test_const_is_copy() {
    let c = Const::<I64Type>::new(42);
    let x = Var::<I64Type>::new(Variable::from_u32(0));

    // Use c multiple times
    let _expr1 = add(x, c);
    let _expr2 = mul(x, c);
    let _expr3 = add(c, c);
}

#[test]
fn test_nested_expressions() {
    let x = Var::<I64Type>::new(Variable::from_u32(0));
    let a = Const::<I64Type>::new(3);
    let b = Const::<I64Type>::new(10);

    // Build (x + 3) * (10 - x)
    let left = add(x, a);
    let right = sub(b, x);
    let _complex = mul(left, right);
}

#[test]
fn test_multiple_types() {
    let i64_var = Var::<I64Type>::new(Variable::from_u32(0));
    let u64_var = Var::<U64Type>::new(Variable::from_u32(1));
    let f64_var = Var::<F64Type>::new(Variable::from_u32(2));

    let _i64_expr = add(i64_var, Const::<I64Type>::new(10));
    let _u64_expr = mul(u64_var, Const::<U64Type>::new(20));
    let _f64_expr = div(f64_var, Const::<F64Type>::new(2.0));
}

#[test]
fn test_boxing() {
    let x = Var::<I64Type>::new(Variable::from_u32(0));
    let _boxed: Box<dyn Staged<Out = I64Type>> = x.boxed();

    // Can box operations too
    let expr = add(x, Const::new(5));
    let _boxed_expr: Box<dyn Staged<Out = I64Type>> = expr.boxed();
}

// The following tests are compile-fail tests - they should NOT compile
// Uncomment them to verify that type errors are caught at compile time

// #[test]
// fn test_type_mismatch_fails() {
//     let x = Var::<I64Type>::new(Variable::from_u32(0));
//     let comparison = lt(x, Const::new(10));
//
//     // This should fail: can't add I64Type and BoolType
//     let _bad = add(x, comparison);
// }

// #[test]
// fn test_mixed_numeric_types_fails() {
//     let i64_var = Var::<I64Type>::new(Variable::from_u32(0));
//     let f64_var = Var::<F64Type>::new(Variable::from_u32(1));
//
//     // This should fail: can't add I64Type and F64Type
//     let _bad = add(i64_var, f64_var);
// }