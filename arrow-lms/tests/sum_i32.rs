//! End-to-end prototype: pass an Arrow `RecordBatch` column into a JIT'd kernel
//! and operate on it through the `rust-lms` iterator framework.

use std::sync::Arc;

use arrow::array::Int32Array;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

use arrow_lms::{get_primitive_i32, FfiArray, StagedArrowArrayI32};
use rust_lms::prelude::*;

/// Build a single-column `Int32` RecordBatch.
fn batch(values: Vec<i32>) -> RecordBatch {
    let col = Int32Array::from(values);
    let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
    RecordBatch::try_new(schema, vec![Arc::new(col)]).unwrap()
}

#[test]
fn sum_i32_column() {
    let rb = batch(vec![1, 2, 3, 4, 5]);
    let ffi = get_primitive_i32(&rb, 0);

    // JIT: fn(&FfiArray) -> i32 that sums the column via the iterator framework.
    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum_i32", |ctx, arr: Var<SRef<FfiArray>>| {
        let array = StagedArrowArrayI32::load(ctx, arr);
        array.values().sum(ctx)
    });
    let compiled = compiler.compile(f).expect("compile");
    let sum = compiled.as_fn();

    assert_eq!(sum(&ffi), 15);

    // Cross-check against Arrow's own aggregate kernel.
    let arrow_sum =
        arrow::compute::sum(rb.column(0).as_any().downcast_ref::<Int32Array>().unwrap()).unwrap();
    assert_eq!(sum(&ffi), arrow_sum);
}

#[test]
fn empty_column_sums_to_zero() {
    let rb = batch(vec![]);
    let ffi = get_primitive_i32(&rb, 0);

    let mut compiler = Compiler::new();
    let f = compiler.fun1("sum_i32", |ctx, arr: Var<SRef<FfiArray>>| {
        StagedArrowArrayI32::load(ctx, arr).values().sum(ctx)
    });
    let sum = compiler.compile(f).unwrap().as_fn();

    assert_eq!(sum(&ffi), 0);
}

#[test]
fn combinators_compose_over_a_column() {
    let rb = batch(vec![1, 2, 3, 4, 5, 6]);
    let ffi = get_primitive_i32(&rb, 0);

    // Count elements > 3 (branchless), and sum the squares of the even ones.
    let mut compiler = Compiler::new();

    let count_gt3 = compiler.fun1("count_gt3", |ctx, arr: Var<SRef<FfiArray>>| {
        StagedArrowArrayI32::load(ctx, arr)
            .values()
            .count_if(ctx, |x| gt(x, 3i32))
    });
    let count = compiler.compile(count_gt3).unwrap().as_fn();
    assert_eq!(count(&ffi), 3); // 4, 5, 6

    let mut compiler = Compiler::new();
    let sum_even_sq = compiler.fun1("sum_even_sq", |ctx, arr: Var<SRef<FfiArray>>| {
        StagedArrowArrayI32::load(ctx, arr)
            .values()
            .filter(|x| eq(x % 2i32, 0i32))
            .map(|x| x * x)
            .sum(ctx)
    });
    let sum = compiler.compile(sum_even_sq).unwrap().as_fn();
    assert_eq!(sum(&ffi), 4 + 16 + 36); // 2² + 4² + 6²
}
