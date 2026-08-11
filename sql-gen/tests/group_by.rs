//! GROUP BY over a single Int64 key: count(*) / count(col) / sum, checked against
//! a Rust oracle. Group order is hash-map-dependent, so results are compared as
//! sorted key → value maps.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use sql_gen::exec_jit;

fn batch(keys: Vec<i64>, values: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap()
}

/// A batch with a non-null key and a **nullable** value column.
fn batch_nullable(keys: Vec<i64>, values: Vec<Option<i64>>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap()
}

fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}

/// Result as a `key -> Option<agg>` map, preserving nulls in the aggregate column.
fn as_opt_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<i64, Option<i64>> {
    let keys = i64s(rb, 0);
    let aggs = i64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| {
            let v = if aggs.is_null(i) {
                None
            } else {
                Some(aggs.value(i))
            };
            (keys.value(i), v)
        })
        .collect()
}

/// Result as a `key -> agg` map (column 0 = key, `agg_col` = aggregate).
fn as_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<i64, i64> {
    let keys = i64s(rb, 0);
    let aggs = i64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| (keys.value(i), aggs.value(i)))
        .collect()
}

#[test]
fn group_by_count_star() {
    let rb = batch(vec![1, 1, 2, 1, 2], vec![10, 20, 30, 40, 50]);
    let out = exec_jit("SELECT key, count(*) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 2);
    assert_eq!(as_map(&out, 1), BTreeMap::from([(1, 3), (2, 2)]));
}

#[test]
fn group_by_where_clause() {
    let rb = batch(vec![1, 1, 2, 1, 2], vec![10, 20, 30, 40, 50]);
    let out = exec_jit(
        "SELECT key, sum(value), min(value) FROM t where key = 1 and value > 10 GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 1);
    assert_eq!(as_map(&out, 1), BTreeMap::from([(1, 60)]));
    assert_eq!(as_map(&out, 2), BTreeMap::from([(1, 20)]));
}

#[test]
fn group_by_sum() {
    let rb = batch(vec![1, 1, 2, 1, 2], vec![10, 20, 30, 40, 50]);
    let out = exec_jit("SELECT key, sum(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(as_map(&out, 1), BTreeMap::from([(1, 70), (2, 80)]));
}

#[test]
fn group_by_count_and_sum() {
    let rb = batch(vec![5, 5, 7, 5, 7, 9], vec![1, 2, 3, 4, 5, 6]);
    let out = exec_jit(
        "SELECT key, count(*), sum(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 3);
    assert_eq!(as_map(&out, 1), BTreeMap::from([(5, 3), (7, 2), (9, 1)])); // count
    assert_eq!(as_map(&out, 2), BTreeMap::from([(5, 7), (7, 8), (9, 6)])); // sum
}

#[test]
fn group_by_sum_expr() {
    let rb = batch(vec![1, 2, 1, 2, 1], vec![1, 2, 3, 4, 5]);
    let out = exec_jit("SELECT key, sum(value)+2 FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 2);
    assert_eq!(as_map(&out, 1), BTreeMap::from([(1, 11), (2, 8)])); // sum
}

#[test]
fn group_by_single_group() {
    let rb = batch(vec![42, 42, 42], vec![1, 2, 3]);
    let out = exec_jit("SELECT key, sum(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 1);
    assert_eq!(as_map(&out, 1), BTreeMap::from([(42, 6)]));
}

#[test]
fn group_by_all_distinct() {
    let rb = batch(vec![1, 2, 3, 4], vec![10, 20, 30, 40]);
    let out = exec_jit("SELECT key, count(*) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 4);
    assert_eq!(
        as_map(&out, 1),
        BTreeMap::from([(1, 1), (2, 1), (3, 1), (4, 1)])
    );
}

#[test]
fn group_by_min_max() {
    // negatives too, so the min/max identities (i64::MAX / i64::MIN) matter.
    let rb = batch(vec![1, 1, 2, 1, 2], vec![-5, 20, 30, 40, -100]);
    let out = exec_jit(
        "SELECT key, min(value), max(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(as_map(&out, 1), BTreeMap::from([(1, -5), (2, -100)])); // min
    assert_eq!(as_map(&out, 2), BTreeMap::from([(1, 40), (2, 30)])); // max
}

#[test]
fn group_by_sum_with_nulls() {
    // key 1: values [10, null, 40] -> sum 50, count(value) 2
    // key 2: values [null, null]   -> sum NULL, count(value) 0
    // key 3: values [7]            -> sum 7, count(value) 1
    let rb = batch_nullable(
        vec![1, 2, 1, 2, 1, 3],
        vec![Some(10), None, None, None, Some(40), Some(7)],
    );
    let out = exec_jit(
        "SELECT key, sum(value), count(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    // sum is NULL for the all-null group; count(value) counts non-nulls (never null).
    assert_eq!(
        as_opt_map(&out, 1),
        BTreeMap::from([(1, Some(50)), (2, None), (3, Some(7))])
    );
    assert_eq!(
        as_opt_map(&out, 2),
        BTreeMap::from([(1, Some(2)), (2, Some(0)), (3, Some(1))])
    );
}

#[test]
fn group_by_min_max_with_nulls() {
    // key 1: [null, 5, null, -3] -> min -3, max 5
    // key 2: [null]              -> min NULL, max NULL
    let rb = batch_nullable(
        vec![1, 2, 1, 1, 1],
        vec![None, None, Some(5), None, Some(-3)],
    );
    let out = exec_jit(
        "SELECT key, min(value), max(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(
        as_opt_map(&out, 1),
        BTreeMap::from([(1, Some(-3)), (2, None)])
    );
    assert_eq!(
        as_opt_map(&out, 2),
        BTreeMap::from([(1, Some(5)), (2, None)])
    );
}

fn f64s(rb: &RecordBatch, col: usize) -> &Float64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}

/// avg column as `key -> Option<f64>`.
fn as_opt_f64_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<i64, Option<f64>> {
    let keys = i64s(rb, 0);
    let aggs = f64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| {
            let v = if aggs.is_null(i) {
                None
            } else {
                Some(aggs.value(i))
            };
            (keys.value(i), v)
        })
        .collect()
}

#[test]
fn group_by_avg() {
    // key 1: [10, 20, 60] avg 30; key 2: [5, 15] avg 10
    let rb = batch(vec![1, 1, 2, 1, 2], vec![10, 20, 5, 60, 15]);
    let out = exec_jit("SELECT key, avg(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(
        as_opt_f64_map(&out, 1),
        BTreeMap::from([(1, Some(30.0)), (2, Some(10.0))])
    );
}

#[test]
fn group_by_avg_with_nulls() {
    // key 1: [10, null, 40] avg 25; key 2: [null, null] avg NULL (count 0 -> null)
    let rb = batch_nullable(
        vec![1, 2, 1, 2, 1],
        vec![Some(10), None, None, None, Some(40)],
    );
    let out = exec_jit("SELECT key, avg(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(
        as_opt_f64_map(&out, 1),
        BTreeMap::from([(1, Some(25.0)), (2, None)])
    );
}

#[test]
fn group_by_avg_and_sum() {
    // avg alongside another aggregate, exercising the hidden count column layout.
    let rb = batch(vec![1, 1, 2], vec![10, 30, 50]);
    let out = exec_jit(
        "SELECT key, sum(value), avg(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(
        as_opt_map(&out, 1),
        BTreeMap::from([(1, Some(40)), (2, Some(50))])
    ); // sum
    assert_eq!(
        as_opt_f64_map(&out, 2),
        BTreeMap::from([(1, Some(20.0)), (2, Some(50.0))])
    ); // avg
}

#[test]
fn group_by_count_star_ignores_nulls() {
    // count(*) counts rows regardless of value nulls.
    let rb = batch_nullable(vec![1, 1, 2], vec![None, Some(5), None]);
    let out = exec_jit("SELECT key, count(*) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(
        as_opt_map(&out, 1),
        BTreeMap::from([(1, Some(2)), (2, Some(1))])
    );
}

#[test]
fn group_by_many_groups_grows_records() {
    // Every row is its own group → the records buffer must grow ~N times
    // (reallocating and moving), exercising the O(groups) grow + returned-record-ptr
    // path. Each group's sum is just its single value.
    let n = 1000i64;
    let keys: Vec<i64> = (0..n).collect();
    let values: Vec<i64> = (0..n).map(|k| k * 3).collect();
    let rb = batch(keys, values);
    let out = exec_jit("SELECT key, sum(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), n as usize);
    let expected: BTreeMap<i64, i64> = (0..n).map(|k| (k, k * 3)).collect();
    assert_eq!(as_map(&out, 1), expected);
}

#[test]
fn group_by_all_aggs() {
    let rb = batch(vec![1, 1, 2], vec![10, 30, 50]);
    let out = exec_jit(
        "SELECT key, count(*), sum(value), min(value), max(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(as_map(&out, 1), BTreeMap::from([(1, 2), (2, 1)])); // count
    assert_eq!(as_map(&out, 2), BTreeMap::from([(1, 40), (2, 50)])); // sum
    assert_eq!(as_map(&out, 3), BTreeMap::from([(1, 10), (2, 50)])); // min
    assert_eq!(as_map(&out, 4), BTreeMap::from([(1, 30), (2, 50)])); // max
}

// --- Float64 aggregate value columns: sum/min/max accumulate in `f64`. ---

/// A batch with a non-null Int64 key and a non-null `Float64` value column.
fn batch_f64(keys: Vec<i64>, values: Vec<f64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Float64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap()
}

/// A batch with a non-null key and a **nullable** `Float64` value column.
fn batch_f64_nullable(keys: Vec<i64>, values: Vec<Option<f64>>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Float64, true),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Float64Array::from(values)),
        ],
    )
    .unwrap()
}

#[test]
fn group_by_float_sum_min_max() {
    // key 1: [1.5, 2.5, 4.0]  sum 8.0, min 1.5, max 4.0
    // key 2: [10.25, -3.75]   sum 6.5, min -3.75, max 10.25
    let rb = batch_f64(vec![1, 1, 2, 1, 2], vec![1.5, 2.5, 10.25, 4.0, -3.75]);
    let out = exec_jit(
        "SELECT key, sum(value), min(value), max(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(
        as_opt_f64_map(&out, 1),
        BTreeMap::from([(1, Some(8.0)), (2, Some(6.5))])
    ); // sum
    assert_eq!(
        as_opt_f64_map(&out, 2),
        BTreeMap::from([(1, Some(1.5)), (2, Some(-3.75))])
    ); // min
    assert_eq!(
        as_opt_f64_map(&out, 3),
        BTreeMap::from([(1, Some(4.0)), (2, Some(10.25))])
    ); // max
}

#[test]
fn group_by_float_min_max_with_nulls() {
    // key 1: [2.0, null, 0.5] min 0.5, max 2.0
    // key 2: [null, null]     all-null -> NULL (never beats the ±∞ identity, count 0)
    let rb = batch_f64_nullable(
        vec![1, 2, 1, 2, 1],
        vec![Some(2.0), None, None, None, Some(0.5)],
    );
    let out = exec_jit(
        "SELECT key, min(value), max(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(
        as_opt_f64_map(&out, 1),
        BTreeMap::from([(1, Some(0.5)), (2, None)])
    ); // min
    assert_eq!(
        as_opt_f64_map(&out, 2),
        BTreeMap::from([(1, Some(2.0)), (2, None)])
    ); // max
}

#[test]
fn group_by_float_sum_all_null_is_null() {
    // A group whose float inputs are all NULL yields SQL NULL for sum (count 0),
    // not 0.0 — the seen-bit guards it.
    let rb = batch_f64_nullable(vec![1, 1, 2], vec![None, None, Some(3.5)]);
    let out = exec_jit("SELECT key, sum(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(
        as_opt_f64_map(&out, 1),
        BTreeMap::from([(1, None), (2, Some(3.5))])
    );
}
