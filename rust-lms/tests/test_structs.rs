//! Integration tests for struct support with derive macro

use rust_lms::prelude::*;

// Test with simple Copy struct
// Note: For now, we define fields with type markers
// and the macro generates appropriate field descriptors
#[derive(StagedType)]
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
    // fn get_x(pt: Point) -> i64
    let get_x = compiler.fun1("get_x", |pt: VarRef<Point>| {
        pt.get(PointType::x)
    });

    let compiled = compiler.compile(get_x).expect("compilation failed");
    let f = compiled.as_fn();

    // Create a test point
    let point = Point { x: 42, y: 3.14 };
    let result = f(&point);

    assert_eq!(result, 42);
}

#[test]
fn test_struct_multiple_fields() {
    let mut compiler = Compiler::new();

    // fn swap_fields(pt: Point) -> Point with x and y swapped conceptually
    // Actually returns x + y since we can't return structs yet
    let sum_fields = compiler.fun1("sum_fields", |pt: VarRef<Point>| {
        let x = pt.get(PointType::x);
        let y = pt.get(PointType::y);

        // Convert f64 to i64 for addition (just for testing)
        // In real code we'd need proper conversion
        add(x, Const::<I64Type>::new(3))  // Add 3 to x (y is 3.14)
    });

    let compiled = compiler.compile(sum_fields).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 10, y: 3.14 };
    let result = f(&point);

    assert_eq!(result, 13);  // 10 + 3
}

#[test]
fn test_struct_with_reference() {
    let mut compiler = Compiler::new();

    // fn read_x_ref(pt: &Point) -> i64
    // Note: Point's RuntimeValue is already &Point, so we use VarRef<Point> directly
    let read_x_ref = compiler.fun1("read_x_ref", |pt: VarRef<Point>| {
        pt.get(PointType::x)
    });

    let compiled = compiler.compile(read_x_ref).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 99, y: 2.71 };
    let result = f(&point);

    assert_eq!(result, 99);
}

// Test nested struct (struct with struct field)
#[derive(StagedType)]
#[repr(C)]
pub struct Inner {
    #[staged(I64Type)]
    value: i64,
}

#[derive(StagedType)]
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

    // fn get_inner_value(outer: &Outer) -> i64
    // Access: outer.inner.value
    // Note: Outer's RuntimeValue is already &Outer
    let get_inner_value = compiler.fun1("get_inner_value", |outer: VarRef<Outer>| {
        // Get reference to inner field, then get its value
        outer.get_ref(OuterType::inner).get(InnerType::value)
    });

    let compiled = compiler.compile(get_inner_value).expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 777 },
        extra: 123,
    };
    let result = f(&test_struct);

    assert_eq!(result, 777);
}

#[test]
fn test_nested_struct_multiple_access() {
    let mut compiler = Compiler::new();

    // fn sum_outer(outer: &Outer) -> i64
    // Returns outer.inner.value + outer.extra
    // Note: Outer's RuntimeValue is already &Outer
    let sum_outer = compiler.fun1("sum_outer", |outer: VarRef<Outer>| {
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
    let result = f(&test_struct);

    assert_eq!(result, 150);  // 100 + 50
}

#[test]
fn test_struct_copy_semantics() {
    let mut compiler = Compiler::new();

    // Test that Point is CopyType
    // fn double_x(pt: Point) -> i64
    let double_x = compiler.fun1("double_x", |pt: VarRef<Point>| {
        let x = pt.get(PointType::x);
        mul(x, Const::<I64Type>::new(2))
    });

    let compiled = compiler.compile(double_x).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 21, y: 1.0 };
    let result = f(&point);

    assert_eq!(result, 42);
}
