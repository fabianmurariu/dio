//! GROUP BY over a single Int64 key: count(*) / count(col) / sum, checked against
//! a Rust oracle. Group order is hash-map-dependent, so results are compared as
//! sorted key → value maps.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
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

fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
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
fn group_by_count_star_where_clause() {
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
