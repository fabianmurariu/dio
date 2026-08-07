//! Property test for GROUP BY over a single Int64 key against a Rust oracle:
//! `count(*)`, `count(value)`, `sum`, `min`, `max`, `avg` per group over generated
//! `(key, Option<value>)` rows — nulls included — with a small key domain so groups
//! collide heavily and all-null groups occur.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use proptest::prelude::*;
use sql_gen::exec_jit;

fn batch(rows: &[(i64, Option<i64>)]) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, true),
    ]));
    let keys: Vec<i64> = rows.iter().map(|(k, _)| *k).collect();
    let vals: Vec<Option<i64>> = rows.iter().map(|(_, v)| *v).collect();
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

/// `key -> non-null i64` (for `count`, which is never null).
fn map_i64(rb: &RecordBatch, col: usize) -> BTreeMap<i64, i64> {
    let keys = i64s(rb, 0);
    let a = i64s(rb, col);
    (0..rb.num_rows())
        .map(|i| (keys.value(i), a.value(i)))
        .collect()
}

/// `key -> Option<i64>` (nulls preserved).
fn map_opt_i64(rb: &RecordBatch, col: usize) -> BTreeMap<i64, Option<i64>> {
    let keys = i64s(rb, 0);
    let a = i64s(rb, col);
    (0..rb.num_rows())
        .map(|i| (keys.value(i), (!a.is_null(i)).then(|| a.value(i))))
        .collect()
}

/// `key -> Option<f64>` (avg; nulls preserved).
fn map_opt_f64(rb: &RecordBatch, col: usize) -> BTreeMap<i64, Option<f64>> {
    let keys = i64s(rb, 0);
    let a = rb
        .column(col)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    (0..rb.num_rows())
        .map(|i| (keys.value(i), (!a.is_null(i)).then(|| a.value(i))))
        .collect()
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 400, ..ProptestConfig::default() })]

    #[test]
    fn group_by_all_aggs_matches_oracle(
        rows in prop::collection::vec(
            (0i64..6, prop::option::of(-1000i64..1000)),
            1..40,
        ),
    ) {
        let rb = batch(&rows);
        let out = exec_jit(
            "SELECT key, count(*), count(value), sum(value), min(value), max(value), avg(value) \
             FROM t GROUP BY key",
            "t",
            &rb,
        ).expect("exec_jit");

        // Oracle.
        let mut cstar = BTreeMap::new();
        let mut cval = BTreeMap::new();
        let mut sum: BTreeMap<i64, Option<i64>> = BTreeMap::new();
        let mut min: BTreeMap<i64, Option<i64>> = BTreeMap::new();
        let mut max: BTreeMap<i64, Option<i64>> = BTreeMap::new();
        for &(k, v) in &rows {
            *cstar.entry(k).or_insert(0i64) += 1;
            cval.entry(k).or_insert(0i64);
            sum.entry(k).or_insert(None);
            min.entry(k).or_insert(None);
            max.entry(k).or_insert(None);
            if let Some(v) = v {
                *cval.get_mut(&k).unwrap() += 1;
                *sum.get_mut(&k).unwrap() = Some(sum[&k].unwrap_or(0) + v);
                *min.get_mut(&k).unwrap() = Some(min[&k].map_or(v, |m| m.min(v)));
                *max.get_mut(&k).unwrap() = Some(max[&k].map_or(v, |m| m.max(v)));
            }
        }
        let avg: BTreeMap<i64, Option<f64>> = cstar
            .keys()
            .map(|&k| {
                let c = cval[&k];
                (k, (c > 0).then(|| sum[&k].unwrap() as f64 / c as f64))
            })
            .collect();

        prop_assert_eq!(out.num_rows(), cstar.len());
        prop_assert_eq!(map_i64(&out, 1), cstar);
        prop_assert_eq!(map_i64(&out, 2), cval);
        prop_assert_eq!(map_opt_i64(&out, 3), sum);
        prop_assert_eq!(map_opt_i64(&out, 4), min);
        prop_assert_eq!(map_opt_i64(&out, 5), max);

        // avg: same IEEE division on both sides, so compare with a tiny tolerance.
        let got_avg = map_opt_f64(&out, 6);
        for (k, want) in &avg {
            match (got_avg.get(k), want) {
                (Some(Some(g)), Some(w)) => prop_assert!((g - w).abs() < 1e-9, "avg[{}]: {} vs {}", k, g, w),
                (Some(None), None) => {}
                other => prop_assert!(false, "avg[{}] mismatch: {:?}", k, other),
            }
        }
    }
}
