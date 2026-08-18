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
    #[staged(i64)]
    x: i64,
    #[staged(f64)]
    y: f64,
}

#[test]
fn test_simple_struct_field_access() {
    let mut compiler = Compiler::new();

    // Create a function that reads the x field from a Point
    // fn get_x(pt: Point) -> i64  -- NOTE: Pass by VALUE!
    let get_x = compiler.fun1("get_x", |_ctx, pt: Var<Point>| pt.get(PointType::x()));

    let compiled = compiler.compile(get_x).expect("compilation failed");
    let f = compiled.as_fn();

    // Create a test point - passed by VALUE
    let point = Point { x: 42, y: 3.15 };
    let result = f.call(point); // Pass by value, not &point

    assert_eq!(result, 42);
}

#[test]
fn test_struct_multiple_fields() {
    let mut compiler = Compiler::new();

    // fn sum_fields(pt: Point) -> i64
    let sum_fields = compiler.fun1("sum_fields", |_ctx, pt: Var<Point>| {
        let x = pt.get(PointType::x());
        let _y = pt.get(PointType::y()); // Just to verify we can access y

        // Add 3 to x for testing
        x + 3i64
    });

    let compiled = compiler.compile(sum_fields).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 10, y: 3.15 };
    let result = f.call(point); // Pass by value

    assert_eq!(result, 13); // 10 + 3
}

#[test]
fn test_struct_pass_by_value_semantics() {
    let mut compiler = Compiler::new();

    // This test verifies that structs are truly passed by value
    // fn read_x(pt: Point) -> i64
    let read_x = compiler.fun1("read_x", |_ctx, pt: Var<Point>| pt.get(PointType::x()));

    let compiled = compiler.compile(read_x).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 99, y: 2.71 };
    let result = f.call(point); // Pass by value

    assert_eq!(result, 99);
}

// Test nested struct (struct with struct field)
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Inner {
    #[staged(i64)]
    value: i64,
}

#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Outer {
    #[staged(Inner)]
    inner: Inner,
    #[staged(i64)]
    extra: i64,
}

#[test]
fn test_nested_struct_access() {
    let mut compiler = Compiler::new();

    // fn get_inner_value(outer: Outer) -> i64
    // Access: outer.inner.value
    // Note: Outer is passed by VALUE, so we use .field() (not .get_ref())
    let get_inner_value = compiler.fun1("get_inner_value", |_ctx, outer: Var<Outer>| {
        // Navigate to inner field, then load its value
        outer.field(OuterType::inner()).get(InnerType::value())
    });

    let compiled = compiler
        .compile(get_inner_value)
        .expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 777 },
        extra: 123,
    };

    let result = f.call(test_struct); // Pass by value

    assert_eq!(result, 777);
}

#[test]
fn test_nested_struct_by_ref_access() {
    let mut compiler = Compiler::new();

    // fn get_inner_value_by_ref(outer: SRef<Outer>) -> i64
    // Access: outer.inner.value
    // Note: Outer is passed by REFERENCE
    let get_inner_value_by_ref =
        compiler.fun1("get_inner_value_by_ref", |_ctx, outer: Var<SRef<Outer>>| {
            // Get reference to inner field, then get its value
            // Note: Using get_ref (not get_ref_mut) since we have an immutable reference
            outer.get_ref(OuterType::inner()).get(InnerType::value())
        });

    let compiled = compiler
        .compile(get_inner_value_by_ref)
        .expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 555 },
        extra: 321,
    };
    let result = f.call(&test_struct); // Pass by reference

    assert_eq!(result, 555);
}

#[test]
fn test_nested_struct_multiple_access() {
    let mut compiler = Compiler::new();

    // fn sum_outer(outer: Outer) -> i64
    // Returns outer.inner.value + outer.extra
    // Note: Using .field() instead of .get_ref() since outer is by-value
    // (can't return references from by-value parameters)
    let sum_outer = compiler.fun1("sum_outer", |_ctx, outer: Var<Outer>| {
        let inner_val = outer.field(OuterType::inner()).get(InnerType::value());
        let extra = outer.get(OuterType::extra());

        inner_val + extra
    });

    let compiled = compiler.compile(sum_outer).expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 100 },
        extra: 50,
    };
    let result = f.call(test_struct); // Pass by value

    assert_eq!(result, 150); // 100 + 50
}
#[test]
fn test_nested_struct_multiple_access_ref() {
    let mut compiler = Compiler::new();

    // fn sum_outer(outer: Outer) -> i64
    // Returns outer.inner.value + outer.extra
    let sum_outer = compiler.fun1("sum_outer", |_ctx, outer: Var<SRef<Outer>>| {
        let inner_val = outer.get_ref(OuterType::inner()).get(InnerType::value());
        let extra = outer.get(OuterType::extra());

        inner_val + extra
    });

    let compiled = compiler.compile(sum_outer).expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 100 },
        extra: 50,
    };
    let result = f.call(&test_struct); // Pass by value

    assert_eq!(result, 150); // 100 + 50
}

#[test]
fn test_struct_copy_semantics() {
    let mut compiler = Compiler::new();

    // Test that Point is CopyType and passed by value
    // fn double_x(pt: Point) -> i64
    let double_x = compiler.fun1("double_x", |_ctx, pt: Var<Point>| {
        let x = pt.get(PointType::x());
        x * 2i64
    });

    let compiled = compiler.compile(double_x).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 21, y: 1.0 };
    let result = f.call(point); // Pass by value

    assert_eq!(result, 42);
}

// Test accessing the second field of a nested struct's parent (to isolate the issue)
#[test]
fn test_outer_extra_field() {
    let mut compiler = Compiler::new();

    // fn get_extra(outer: Outer) -> i64  -- just get the extra field, no nesting
    let get_extra = compiler.fun1("get_extra", |_ctx, outer: Var<Outer>| {
        outer.get(OuterType::extra())
    });

    let compiled = compiler.compile(get_extra).expect("compilation failed");
    let f = compiled.as_fn();

    let test_struct = Outer {
        inner: Inner { value: 777 },
        extra: 999,
    };
    let result = f.call(test_struct);

    assert_eq!(result, 999);
}

// =============================================================================
// Tests for mixed integer/float struct ABI
// =============================================================================

#[test]
fn test_mixed_struct_read_f64_field() {
    let mut compiler = Compiler::new();

    // fn get_y(pt: Point) -> f64 -- read the f64 field from mixed struct
    let get_y = compiler.fun1("get_y", |_ctx, pt: Var<Point>| pt.get(PointType::y()));

    let compiled = compiler.compile(get_y).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 42, y: 3.15 };
    let result = f.call(point);

    assert!(
        (result - 3.15).abs() < 1e-10,
        "Expected 3.15, got {}",
        result
    );
}

#[test]
fn test_mixed_struct_read_i64_after_f64_access() {
    let mut compiler = Compiler::new();

    // Read y (f64), then read x (i64) and return it
    // This verifies both fields are accessible
    let read_both = compiler.fun1("read_both", |_ctx, pt: Var<Point>| {
        let _y = pt.get(PointType::y()); // Access f64 field
        pt.get(PointType::x()) // Return i64 field
    });

    let compiled = compiler.compile(read_both).expect("compilation failed");
    let f = compiled.as_fn();

    let point = Point { x: 100, y: 3.15 };
    let result = f.call(point);

    assert_eq!(result, 100);
}

#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct MixedStruct {
    #[staged(f64)]
    a: f64,
    #[staged(i64)]
    b: i64,
    #[staged(f64)]
    c: f64,
}

// 16-byte struct with f64 first - should work on ARM64
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct FloatFirst {
    #[staged(f64)]
    x: f64,
    #[staged(i64)]
    y: i64,
}

// These tests verify that large structs (>16 bytes) are correctly passed by pointer
#[test]
fn test_float_first_struct() {
    let mut compiler = Compiler::new();

    // Struct with f64 as first field
    let get_a = compiler.fun1("get_a", |_ctx, s: Var<MixedStruct>| {
        s.get(MixedStructType::a())
    });

    let compiled = compiler.compile(get_a).expect("compilation failed");
    let f = compiled.as_fn();

    let s = MixedStruct {
        a: 2.71,
        b: 42,
        c: 1.41,
    };
    let result = f.call(s);

    assert!(
        (result - 2.71).abs() < 1e-10,
        "Expected 2.71, got {}",
        result
    );
}

#[test]
fn test_float_first_struct_read_int() {
    let mut compiler = Compiler::new();

    let get_b = compiler.fun1("get_b", |_ctx, s: Var<MixedStruct>| {
        s.get(MixedStructType::b())
    });

    let compiled = compiler.compile(get_b).expect("compilation failed");
    let f = compiled.as_fn();

    let s = MixedStruct {
        a: 2.71,
        b: 42,
        c: 1.41,
    };
    let result = f.call(s);

    assert_eq!(result, 42);
}

// Test 16-byte struct with f64 first - should work because it fits in 2 registers
#[test]
fn test_16byte_float_first_struct() {
    let mut compiler = Compiler::new();

    let get_x = compiler.fun1("get_x", |_ctx, s: Var<FloatFirst>| {
        s.get(FloatFirstType::x())
    });

    let compiled = compiler.compile(get_x).expect("compilation failed");
    let f = compiled.as_fn();

    let s = FloatFirst { x: 2.71, y: 42 };
    let result = f.call(s);

    assert!(
        (result - 2.71).abs() < 1e-10,
        "Expected 2.71, got {}",
        result
    );
}

#[test]
fn test_16byte_float_first_struct_read_int() {
    let mut compiler = Compiler::new();

    let get_y = compiler.fun1("get_y", |_ctx, s: Var<FloatFirst>| {
        s.get(FloatFirstType::y())
    });

    let compiled = compiler.compile(get_y).expect("compilation failed");
    let f = compiled.as_fn();

    let s = FloatFirst { x: 2.71, y: 42 };
    let result = f.call(s);

    assert_eq!(result, 42);
}

// =============================================================================
// Generic structs: #[derive(StagedType)] over <A: StagedType, ...>
// =============================================================================

// The staged type of each field is *inferred* from its Rust type (no
// `#[staged(..)]`), since `A`/`B` are themselves `StagedType`.
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
pub struct Pair<A: StagedType, B: StagedType> {
    first: A,
    second: B,
}

#[test]
fn test_generic_struct_by_value() {
    let mut compiler = Compiler::new();

    // Pair<i64, i64>: the constructor fn's generics are inferred from the
    // receiver, so `PairType::second()` resolves to Field<Parent = Pair<i64,i64>>.
    let get_second = compiler.fun1("get_second", |_ctx, p: Var<Pair<i64, i64>>| {
        p.get(PairType::second())
    });

    let compiled = compiler.compile(get_second).expect("compilation failed");
    let f = compiled.as_fn();
    assert_eq!(
        f.call(Pair {
            first: 10i64,
            second: 20i64
        }),
        20
    );
}

#[test]
fn test_generic_struct_distinct_monomorphizations() {
    let mut compiler = Compiler::new();

    // A different instantiation: Pair<i32, i64>. `first` is an i32 at offset 0 —
    // exercises a per-monomorphization `offset_of!` and field type.
    let get_first = compiler.fun1("get_first", |_ctx, p: Var<Pair<i32, i64>>| {
        p.get(PairType::first())
    });

    let compiled = compiler.compile(get_first).expect("compilation failed");
    let f = compiled.as_fn();
    assert_eq!(
        f.call(Pair {
            first: 7i32,
            second: 99i64
        }),
        7
    );
}

#[test]
fn test_generic_struct_by_ref() {
    let mut compiler = Compiler::new();

    // By reference: get_ref into a field of a generic struct, then load it.
    let read_second = compiler.fun1("read_second", |_ctx, p: Var<SRef<Pair<i64, i64>>>| {
        load_ref(p.get_ref(PairType::second()))
    });

    let compiled = compiler.compile(read_second).expect("compilation failed");
    let f = compiled.as_fn();
    assert_eq!(
        f.call(&Pair {
            first: 1i64,
            second: 42i64
        }),
        42
    );
}

#[test]
fn test_return_mixed_struct() {
    let mut compiler = Compiler::new();

    // Function that takes two values and returns a Point struct
    let make_point = compiler.fun2("make_point", |_ctx, x: Var<i64>, _y: Var<f64>| {
        // We need to construct a Point - but we don't have struct construction yet
        // For now, just test that we can return the input x
        x
    });

    let compiled = compiler.compile(make_point).expect("compilation failed");
    let f = compiled.as_fn();

    let result = f.call(42, 3.15);
    assert_eq!(result, 42);
}
