//! GROUP BY over a single Int64 key: count(*) / count(col) / sum, checked against
//! a Rust oracle. Group order is hash-map-dependent, so results are compared as
//! sorted key → value maps.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::{Array, Float64Array, Int64Array, StringViewArray};
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

// --- Utf8View string group keys: hash/eq on content, keys copied into the table's
// pool, emitted as a Utf8View column. ---

/// A batch with a non-null `Utf8View` key column and a non-null Int64 value.
fn batch_str(keys: Vec<&str>, values: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8View, false),
        Field::new("value", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringViewArray::from(keys)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap()
}

/// Result as a `name -> agg` map (column 0 = Utf8View key, `agg_col` = i64 aggregate).
fn as_str_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<String, i64> {
    let keys: &StringViewArray = rb.column(0).as_any().downcast_ref().unwrap();
    let aggs = i64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| (keys.value(i).to_string(), aggs.value(i)))
        .collect()
}

#[test]
fn group_by_string_key() {
    // short (inline) strings; a/b/a/c/b/a
    let rb = batch_str(
        vec!["apple", "beet", "apple", "cherry", "beet", "apple"],
        vec![1, 2, 3, 4, 5, 6],
    );
    let out = exec_jit(
        "SELECT name, count(*), sum(value) FROM t GROUP BY name",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 3);
    assert_eq!(
        as_str_map(&out, 1),
        BTreeMap::from([
            ("apple".into(), 3),
            ("beet".into(), 2),
            ("cherry".into(), 1),
        ])
    ); // count: apple×3, beet×2, cherry×1
    assert_eq!(
        as_str_map(&out, 2),
        BTreeMap::from([
            ("apple".into(), 10),
            ("beet".into(), 7),
            ("cherry".into(), 4),
        ])
    ); // sum: apple 1+3+6, beet 2+5, cherry 4
}

#[test]
fn group_by_long_string_key() {
    // Strings >12 bytes: their Utf8View carries a buffer index/offset that differs
    // per occurrence, so grouping MUST hash/compare content, not the raw view.
    let long_a = "this is a long grouping key over twelve bytes";
    let long_b = "another sufficiently long key value here!!";
    let rb = batch_str(
        vec![long_a, long_b, long_a, long_a, long_b],
        vec![10, 20, 30, 40, 50],
    );
    let out = exec_jit("SELECT name, sum(value) FROM t GROUP BY name", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 2);
    assert_eq!(
        as_str_map(&out, 1),
        BTreeMap::from([(long_a.into(), 80), (long_b.into(), 70)])
    );
}

#[test]
fn group_by_string_key_survives_input_drop() {
    // The key bytes are copied into the table's pool, so the result is valid after
    // the input batch is dropped — the contract that lets inputs stream.
    let out = {
        let rb = batch_str(vec!["x", "yy", "x", "zzz", "yy"], vec![1, 2, 3, 4, 5]);
        exec_jit("SELECT name, sum(value) FROM t GROUP BY name", "t", &rb).unwrap()
        // rb dropped here
    };
    assert_eq!(
        as_str_map(&out, 1),
        BTreeMap::from([("x".into(), 4), ("yy".into(), 7), ("zzz".into(), 4)])
    );
}

// --- Float64 group keys: keyed on the f64 bits (via the u64 table), with -0.0/NaN
// canonicalized so they group per SQL semantics. ---

/// A batch with a non-null `Float64` key column and a non-null Int64 value.
fn batch_f64_key(keys: Vec<f64>, values: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Float64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(keys)),
            Arc::new(Int64Array::from(values)),
        ],
    )
    .unwrap()
}

/// Result as a `bits(key) -> agg` map (f64 keys aren't Ord/Hash-friendly, so key on bits).
fn as_f64key_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<u64, i64> {
    let keys = f64s(rb, 0);
    let aggs = i64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| (keys.value(i).to_bits(), aggs.value(i)))
        .collect()
}

#[test]
fn group_by_float_key() {
    // keys 1.5 / 2.5 / 1.5 / 3.5 / 2.5 / 1.5
    let rb = batch_f64_key(vec![1.5, 2.5, 1.5, 3.5, 2.5, 1.5], vec![1, 2, 3, 4, 5, 6]);
    let out = exec_jit(
        "SELECT key, count(*), sum(value) FROM t GROUP BY key",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 3);
    assert_eq!(
        as_f64key_map(&out, 1),
        BTreeMap::from([
            (1.5f64.to_bits(), 3),
            (2.5f64.to_bits(), 2),
            (3.5f64.to_bits(), 1),
        ])
    ); // count
    assert_eq!(
        as_f64key_map(&out, 2),
        BTreeMap::from([
            (1.5f64.to_bits(), 10), // 1+3+6
            (2.5f64.to_bits(), 7),  // 2+5
            (3.5f64.to_bits(), 4),  // 4
        ])
    ); // sum
}

#[test]
fn group_by_float_key_neg_zero_and_nan() {
    // -0.0 must group with +0.0 (different bits, equal value), and two NaNs with
    // *different payloads* must group together (bit grouping alone would split them;
    // canonicalization collapses them).
    let nan1 = f64::from_bits(0x7ff8_0000_0000_0001);
    let nan2 = f64::from_bits(0x7ff8_0000_0000_0002);
    assert_ne!(nan1.to_bits(), nan2.to_bits());
    let rb = batch_f64_key(vec![0.0, -0.0, nan1, 0.0, nan2], vec![1, 2, 3, 4, 5]);
    let out = exec_jit("SELECT key, sum(value) FROM t GROUP BY key", "t", &rb).unwrap();
    // two groups: {0.0, -0.0, 0.0} -> 1+2+4=7 ; {nan, nan} -> 3+5=8
    assert_eq!(out.num_rows(), 2);
    let keys = f64s(&out, 0);
    let sums = i64s(&out, 1);
    let mut zero_sum = None;
    let mut nan_sum = None;
    for i in 0..out.num_rows() {
        if keys.value(i).is_nan() {
            nan_sum = Some(sums.value(i));
        } else {
            assert_eq!(keys.value(i), 0.0);
            zero_sum = Some(sums.value(i));
        }
    }
    assert_eq!(zero_sum, Some(7));
    assert_eq!(nan_sum, Some(8));
}

// --- Nullable GROUP BY keys: NULLs form one group (SQL semantics), kept out of the
// hash table (tracked as a separate null group). Works for int / float / string. ---

fn batch_null_int_key(keys: Vec<Option<i64>>, values: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, true), // nullable key
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

/// Output `Option<i64> key -> i64 agg`, preserving a NULL key.
fn as_opt_key_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<Option<i64>, i64> {
    let keys = i64s(rb, 0);
    let aggs = i64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| {
            let k = if keys.is_null(i) {
                None
            } else {
                Some(keys.value(i))
            };
            (k, aggs.value(i))
        })
        .collect()
}

#[test]
fn group_by_null_int_key() {
    // keys 1/null/1/null/2 -> {1: 1+3=4}, {null: 2+4=6}, {2: 5}
    let rb = batch_null_int_key(
        vec![Some(1), None, Some(1), None, Some(2)],
        vec![1, 2, 3, 4, 5],
    );
    let out = exec_jit("SELECT key, sum(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 3);
    assert_eq!(
        as_opt_key_map(&out, 1),
        BTreeMap::from([(Some(1), 4), (None, 6), (Some(2), 5)])
    );
}

#[test]
fn group_by_nullable_key_no_nulls_present() {
    // A nullable key column but no actual nulls -> no null group is created.
    let rb = batch_null_int_key(vec![Some(7), Some(7), Some(9)], vec![1, 2, 3]);
    let out = exec_jit("SELECT key, sum(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 2);
    assert_eq!(
        as_opt_key_map(&out, 1),
        BTreeMap::from([(Some(7), 3), (Some(9), 3)])
    );
}

#[test]
fn group_by_null_float_key() {
    // nullable Float64 key: 1.5 / null / 1.5 / null
    let schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Float64, true),
        Field::new("value", DataType::Int64, false),
    ]));
    let rb = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Float64Array::from(vec![Some(1.5), None, Some(1.5), None])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        ],
    )
    .unwrap();
    let out = exec_jit("SELECT key, sum(value) FROM t GROUP BY key", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 2);
    let keys = f64s(&out, 0);
    let sums = i64s(&out, 1);
    let mut real = None;
    let mut null = None;
    for i in 0..out.num_rows() {
        if keys.is_null(i) {
            null = Some(sums.value(i));
        } else {
            assert_eq!(keys.value(i), 1.5);
            real = Some(sums.value(i));
        }
    }
    assert_eq!(real, Some(4)); // 1+3
    assert_eq!(null, Some(6)); // 2+4
}

#[test]
fn group_by_null_string_key() {
    // nullable Utf8View key: "a" / null / "a" / "b" / null
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8View, true),
        Field::new("value", DataType::Int64, false),
    ]));
    let rb = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringViewArray::from(vec![
                Some("a"),
                None,
                Some("a"),
                Some("b"),
                None,
            ])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4, 5])),
        ],
    )
    .unwrap();
    let out = exec_jit("SELECT name, sum(value) FROM t GROUP BY name", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 3);
    let keys: &StringViewArray = out.column(0).as_any().downcast_ref().unwrap();
    let sums = i64s(&out, 1);
    let mut map: BTreeMap<Option<String>, i64> = BTreeMap::new();
    for i in 0..out.num_rows() {
        let k = if keys.is_null(i) {
            None
        } else {
            Some(keys.value(i).to_string())
        };
        map.insert(k, sums.value(i));
    }
    assert_eq!(
        map,
        BTreeMap::from([
            (Some("a".into()), 4), // 1+3
            (Some("b".into()), 4), // 4
            (None, 7),             // 2+5
        ])
    );
}

// --- Composite (multi-column) keys: packed into a byte key (fixed-width columns),
// nulls in the packed bitmap so each (a, b, ...) combination is its own group. ---

fn batch_2int(a: Vec<i64>, b: Vec<i64>, v: Vec<i64>) -> RecordBatch {
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(a)),
            Arc::new(Int64Array::from(b)),
            Arc::new(Int64Array::from(v)),
        ],
    )
    .unwrap()
}

/// Output `(a, b) -> agg` map for a two-int-key result.
fn as_pair_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<(i64, i64), i64> {
    let a = i64s(rb, 0);
    let b = i64s(rb, 1);
    let agg = i64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| ((a.value(i), b.value(i)), agg.value(i)))
        .collect()
}

#[test]
fn group_by_two_int_keys() {
    // (a,b): (1,10),(1,10),(2,20),(1,30) -> (1,10):1+2=3, (2,20):3, (1,30):4
    let rb = batch_2int(vec![1, 1, 2, 1], vec![10, 10, 20, 30], vec![1, 2, 3, 4]);
    let out = exec_jit("SELECT a, b, sum(v) FROM t GROUP BY a, b", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 3);
    assert_eq!(
        as_pair_map(&out, 2),
        BTreeMap::from([((1, 10), 3), ((2, 20), 3), ((1, 30), 4)])
    );
}

#[test]
fn group_by_int_float_composite() {
    // key (int a, float b): (1,1.5),(1,1.5),(2,1.5),(1,2.5)
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Float64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let rb = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![1, 1, 2, 1])),
            Arc::new(Float64Array::from(vec![1.5, 1.5, 1.5, 2.5])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        ],
    )
    .unwrap();
    let out = exec_jit("SELECT a, b, sum(v) FROM t GROUP BY a, b", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 3);
    let a = i64s(&out, 0);
    let b = f64s(&out, 1);
    let s = i64s(&out, 2);
    let mut map: BTreeMap<(i64, u64), i64> = BTreeMap::new();
    for i in 0..out.num_rows() {
        map.insert((a.value(i), b.value(i).to_bits()), s.value(i));
    }
    assert_eq!(
        map,
        BTreeMap::from([
            ((1, 1.5f64.to_bits()), 3), // 1+2
            ((2, 1.5f64.to_bits()), 3),
            ((1, 2.5f64.to_bits()), 4),
        ])
    );
}

#[test]
fn group_by_composite_with_nulls() {
    // Nullable columns: (NULL,5) must differ from (0,5), and group with other (NULL,5).
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Int64, true),
        Field::new("b", DataType::Int64, true),
        Field::new("v", DataType::Int64, false),
    ]));
    let rb = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int64Array::from(vec![None, Some(0), None, Some(0)])),
            Arc::new(Int64Array::from(vec![Some(5), Some(5), Some(5), None])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        ],
    )
    .unwrap();
    // rows: (NULL,5):1, (0,5):2, (NULL,5):3, (0,NULL):4
    //  -> (NULL,5): 1+3=4 ; (0,5): 2 ; (0,NULL): 4
    let out = exec_jit("SELECT a, b, sum(v) FROM t GROUP BY a, b", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 3);
    let a = i64s(&out, 0);
    let b = i64s(&out, 1);
    let s = i64s(&out, 2);
    let mut map: BTreeMap<(Option<i64>, Option<i64>), i64> = BTreeMap::new();
    for i in 0..out.num_rows() {
        let ka = if a.is_null(i) { None } else { Some(a.value(i)) };
        let kb = if b.is_null(i) { None } else { Some(b.value(i)) };
        map.insert((ka, kb), s.value(i));
    }
    assert_eq!(
        map,
        BTreeMap::from([
            ((None, Some(5)), 4),
            ((Some(0), Some(5)), 2),
            ((Some(0), None), 4),
        ])
    );
}

// --- Composite keys containing string columns: built via the host key-builder into a
// flat byte key (per-column [len|content] for strings), unpacked with a running offset. ---

#[test]
fn group_by_string_int_composite() {
    // (name: Utf8View, cat: Int64): ("a",1),("a",1),("b",1),("a",2)
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8View, false),
        Field::new("cat", DataType::Int64, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let rb = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringViewArray::from(vec!["a", "a", "b", "a"])),
            Arc::new(Int64Array::from(vec![1, 1, 1, 2])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        ],
    )
    .unwrap();
    let out = exec_jit(
        "SELECT name, cat, sum(v) FROM t GROUP BY name, cat",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 3);
    let name: &StringViewArray = out.column(0).as_any().downcast_ref().unwrap();
    let cat = i64s(&out, 1);
    let s = i64s(&out, 2);
    let mut map: BTreeMap<(String, i64), i64> = BTreeMap::new();
    for i in 0..out.num_rows() {
        map.insert((name.value(i).to_string(), cat.value(i)), s.value(i));
    }
    assert_eq!(
        map,
        BTreeMap::from([
            (("a".into(), 1), 3), // 1+2
            (("b".into(), 1), 3),
            (("a".into(), 2), 4),
        ])
    );
}

#[test]
fn group_by_two_string_composite_long() {
    // Two string columns, including >12-byte content (view differs per occurrence, so
    // grouping must be on content). ("first", LONG), (LONG, "x"), ("first", LONG)
    let long = "a sufficiently long string over twelve bytes";
    let schema = Arc::new(Schema::new(vec![
        Field::new("a", DataType::Utf8View, false),
        Field::new("b", DataType::Utf8View, false),
        Field::new("v", DataType::Int64, false),
    ]));
    let rb = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringViewArray::from(vec!["first", long, "first"])),
            Arc::new(StringViewArray::from(vec![long, "x", long])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();
    let out = exec_jit("SELECT a, b, sum(v) FROM t GROUP BY a, b", "t", &rb).unwrap();
    assert_eq!(out.num_rows(), 2);
    let a: &StringViewArray = out.column(0).as_any().downcast_ref().unwrap();
    let b: &StringViewArray = out.column(1).as_any().downcast_ref().unwrap();
    let s = i64s(&out, 2);
    let mut map: BTreeMap<(String, String), i64> = BTreeMap::new();
    for i in 0..out.num_rows() {
        map.insert((a.value(i).to_string(), b.value(i).to_string()), s.value(i));
    }
    assert_eq!(
        map,
        BTreeMap::from([
            (("first".into(), long.into()), 40), // 10+30
            ((long.into(), "x".into()), 20),
        ])
    );
}

#[test]
fn group_by_string_composite_with_nulls() {
    // (name?: Utf8View, cat: Int64): (NULL,1),("a",1),(NULL,1),("a",NULL)
    let schema = Arc::new(Schema::new(vec![
        Field::new("name", DataType::Utf8View, true),
        Field::new("cat", DataType::Int64, true),
        Field::new("v", DataType::Int64, false),
    ]));
    let rb = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(StringViewArray::from(vec![
                None,
                Some("a"),
                None,
                Some("a"),
            ])),
            Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(1), None])),
            Arc::new(Int64Array::from(vec![1, 2, 3, 4])),
        ],
    )
    .unwrap();
    // (NULL,1):1+3=4 ; ("a",1):2 ; ("a",NULL):4
    let out = exec_jit(
        "SELECT name, cat, sum(v) FROM t GROUP BY name, cat",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 3);
    let name: &StringViewArray = out.column(0).as_any().downcast_ref().unwrap();
    let cat = i64s(&out, 1);
    let s = i64s(&out, 2);
    let mut map: BTreeMap<(Option<String>, Option<i64>), i64> = BTreeMap::new();
    for i in 0..out.num_rows() {
        let kn = if name.is_null(i) {
            None
        } else {
            Some(name.value(i).to_string())
        };
        let kc = if cat.is_null(i) {
            None
        } else {
            Some(cat.value(i))
        };
        map.insert((kn, kc), s.value(i));
    }
    assert_eq!(
        map,
        BTreeMap::from([
            ((None, Some(1)), 4),
            ((Some("a".into()), Some(1)), 2),
            ((Some("a".into()), None), 4),
        ])
    );
}

// --- HAVING: a post-aggregation filter on the emitted groups. It's just a `Filter`
// above the `Aggregate`, so it composes for free with the push-model emit. ---

#[test]
fn having_on_aggregate() {
    // key 1: sum 70 ; key 2: sum 80 ; HAVING sum(value) > 75 -> only key 2
    let rb = batch(vec![1, 1, 2, 1, 2], vec![10, 20, 30, 40, 50]);
    let out = exec_jit(
        "SELECT key, sum(value) FROM t GROUP BY key HAVING sum(value) > 75",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(as_map(&out, 1), BTreeMap::from([(2, 80)]));
}

#[test]
fn having_with_projection_expr() {
    // HAVING references the raw aggregate while the projection transforms it.
    let rb = batch(vec![1, 1, 2, 1, 2], vec![10, 20, 30, 40, 50]);
    let out = exec_jit(
        "SELECT key, sum(value) + 1 FROM t GROUP BY key HAVING sum(value) > 75",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(as_map(&out, 1), BTreeMap::from([(2, 81)])); // 80 + 1
}

#[test]
fn having_on_count_and_where() {
    // WHERE filters rows pre-aggregation; HAVING filters groups post-aggregation.
    let rb = batch(vec![1, 1, 2, 1, 2, 3], vec![10, 20, 30, 40, 50, 5]);
    let out = exec_jit(
        "SELECT key, count(*) FROM t WHERE value > 8 GROUP BY key HAVING count(*) >= 2",
        "t",
        &rb,
    )
    .unwrap();
    // after WHERE value>8: key1×3, key2×2, key3×0 (row dropped) -> HAVING count>=2: key1,key2
    assert_eq!(as_map(&out, 1), BTreeMap::from([(1, 3), (2, 2)]));
}

#[test]
fn having_all_groups_pass() {
    let rb = batch(vec![1, 2, 3], vec![10, 20, 30]);
    let out = exec_jit(
        "SELECT key, sum(value) FROM t GROUP BY key HAVING sum(value) > 0",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 3);
}

#[test]
fn having_no_groups_pass() {
    let rb = batch(vec![1, 2, 3], vec![10, 20, 30]);
    let out = exec_jit(
        "SELECT key, sum(value) as s FROM t GROUP BY key HAVING s > 1000",
        "t",
        &rb,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 0);
}
