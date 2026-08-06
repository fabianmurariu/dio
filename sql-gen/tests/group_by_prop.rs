//! Property test for GROUP BY over a single Int64 key against a Rust oracle:
//! `count(*)`, `sum`, `min`, `max` per group over generated `(key, value)` rows,
//! with a small key domain so groups collide heavily.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use proptest::prelude::*;
use sql_gen::exec_jit;

fn batch(rows: &[(i64, i64)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let keys: Vec<i64> = rows.iter().map(|(k, _)| *k).collect();
    let vals: Vec<i64> = rows.iter().map(|(_, v)| *v).collect();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(keys)),
            Arc::new(Int64Array::from(vals)),
        ],
    )
    .unwrap()
}

fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}

/// `key -> aggregate` from the result (column 0 = key).
fn result_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<i64, i64> {
    let keys = i64s(rb, 0);
    let aggs = i64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| (keys.value(i), aggs.value(i)))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 300, ..ProptestConfig::default() })]

    #[test]
    fn group_by_matches_oracle(
        rows in prop::collection::vec((0i64..8, -1000i64..1000), 1..40),
    ) {
        let rb = batch(&rows);
        let out = exec_jit(
            "SELECT key, count(*), sum(value), min(value), max(value) FROM t GROUP BY key",
            "t",
            &rb,
        ).expect("exec_jit");

        // Oracle: fold rows into per-key aggregates.
        let mut count = BTreeMap::new();
        let mut sum = BTreeMap::new();
        let mut min = BTreeMap::new();
        let mut max = BTreeMap::new();
        for &(k, v) in &rows {
            *count.entry(k).or_insert(0i64) += 1;
            *sum.entry(k).or_insert(0i64) += v;
            min.entry(k).and_modify(|m: &mut i64| *m = (*m).min(v)).or_insert(v);
            max.entry(k).and_modify(|m: &mut i64| *m = (*m).max(v)).or_insert(v);
        }

        prop_assert_eq!(out.num_rows(), count.len());
        prop_assert_eq!(result_map(&out, 1), count);
        prop_assert_eq!(result_map(&out, 2), sum);
        prop_assert_eq!(result_map(&out, 3), min);
        prop_assert_eq!(result_map(&out, 4), max);
    }
}
