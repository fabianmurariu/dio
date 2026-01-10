//! Integration tests for struct support with derive macro
//!
//! These tests verify that structs are passed BY VALUE:
//! - `Var<Point>` means `fn(Point)` - pass struct by value
//! - `Var<SRef<Point>>` would mean `fn(&Point)` - pass by reference (not tested here)

use rust_lms::prelude::*;
use rust_lms::refer::SRef;

// Test with simple Copy struct
// Note: Structs MUST be Copy for pass-by-value semantics
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Point {
    #[staged(I64Type)]
    x: i64,
    #[staged(F64Type)]
    y: f64,
}

#[test]
fn test_simple_struct_field_access() {
    let mut compiler = Compiler::new();

    // Create a function that reads the x field from a Point
    // fn get_x(pt: Point) -> i64  -- NOTE: Pass by VALUE!
    let get_x = compiler.fun1("get_x", |pt: Var<Point>| {
        pt.get(PointType::x)
    });

    let compiled = compiler.compile(get_x).expect("compilation failed");
    let f = compiled.as_fn();

    // Create a test point - passed by VALUE
    let point = Point { x: 42, y: 3.14 };
    let result = f(point);  // Pass by value, not &point

    assert_eq!(result, 42);
}

#[test]
fn test_struct_multiple_fields() {
    let mut compiler = Compiler::new();

    // fn sum_fields(pt: Point) -> i64
    let sum_fields = compiler.fun1("sum_fields", |pt: Var<Point>| {
        let x = pt.get(PointType::x);
        let _y = pt.get(PointType::y);  // Just to verify we can access y

        // Add 3 to x for testing
        add(x, Const::<I64Type>::new(3))
    });

    let compiled = compiler.compile(sum_fields).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 10, y: 3.14 };
    let result = f(point);  // Pass by value

    assert_eq!(result, 13);  // 10 + 3
}

#[test]
fn test_struct_pass_by_value_semantics() {
    let mut compiler = Compiler::new();

    // This test verifies that structs are truly passed by value
    // fn read_x(pt: Point) -> i64
    let read_x = compiler.fun1("read_x", |pt: Var<Point>| {
        pt.get(PointType::x)
    });

    let compiled = compiler.compile(read_x).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 99, y: 2.71 };
    let result = f(point);  // Pass by value

    assert_eq!(result, 99);
}

// Test nested struct (struct with struct field)
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Inner {
    #[staged(I64Type)]
    value: i64,
}

#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Outer {
    #[staged(Inner)]
    inner: Inner,
    #[staged(I64Type)]
    extra: i64,
}

#[test]
fn test_nested_struct_access() {
    let mut compiler = Compiler::new();

    // fn get_inner_value(outer: Outer) -> i64
    // Access: outer.inner.value
    // Note: Outer is passed by VALUE
    let get_inner_value = compiler.fun1("get_inner_value", |outer: Var<Outer>| {
        // Get reference to inner field, then get its value
        outer.get_ref(OuterType::inner).get(InnerType::value)
    });

    let compiled = compiler.compile(get_inner_value).expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 777 },
        extra: 123,
    };
    let result = f(test_struct);  // Pass by value

    assert_eq!(result, 777);
}

#[test]
fn test_nested_struct_by_ref_access() {
    let mut compiler = Compiler::new();

    // fn get_inner_value_by_ref(outer: SRef<Outer>) -> i64
    // Access: outer.inner.value
    // Note: Outer is passed by REFERENCE
    let get_inner_value_by_ref = compiler.fun1("get_inner_value_by_ref", |outer: Var<SRef<Outer>>| {
        // Get reference to inner field, then get its value
        outer.get_ref(OuterType::inner).get(InnerType::value)
    });

    let compiled = compiler.compile(get_inner_value_by_ref).expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 555 },
        extra: 321,
    };
    let result = f(&test_struct);  // Pass by reference

    assert_eq!(result, 555);
}

#[test]
fn test_nested_struct_multiple_access() {
    let mut compiler = Compiler::new();

    // fn sum_outer(outer: Outer) -> i64
    // Returns outer.inner.value + outer.extra
    let sum_outer = compiler.fun1("sum_outer", |outer: Var<Outer>| {
        let inner_val = outer.get_ref(OuterType::inner).get(InnerType::value);
        let extra = outer.get(OuterType::extra);

        add(inner_val, extra)
    });

    let compiled = compiler.compile(sum_outer).expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 100 },
        extra: 50,
    };
    let result = f(test_struct);  // Pass by value

    assert_eq!(result, 150);  // 100 + 50
}

#[test]
fn test_struct_copy_semantics() {
    let mut compiler = Compiler::new();

    // Test that Point is CopyType and passed by value
    // fn double_x(pt: Point) -> i64
    let double_x = compiler.fun1("double_x", |pt: Var<Point>| {
        let x = pt.get(PointType::x);
        mul(x, Const::<I64Type>::new(2))
    });

    let compiled = compiler.compile(double_x).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 21, y: 1.0 };
    let result = f(point);  // Pass by value

    assert_eq!(result, 42);
}

// Test accessing the second field of a nested struct's parent (to isolate the issue)
#[test]
fn test_outer_extra_field() {
    let mut compiler = Compiler::new();

    // fn get_extra(outer: Outer) -> i64  -- just get the extra field, no nesting
    let get_extra = compiler.fun1("get_extra", |outer: Var<Outer>| {
        outer.get(OuterType::extra)
    });

    let compiled = compiler.compile(get_extra).expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 777 },
        extra: 999,
    };
    let result = f(test_struct);

    assert_eq!(result, 999);
}
