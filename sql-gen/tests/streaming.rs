//! Multi-batch streaming: push several `RecordBatch`es of one table through the
//! JIT kernel via `exec_jit_stream` and check the result equals the same query
//! over the concatenated single batch. Exercises the two-level scan loop (outer
//! batch loop + inner row loop) for real — scan/filter pass-through, scalar
//! aggregates, and GROUP BY all folding across batch boundaries in ONE kernel.

use std::collections::BTreeMap;
use std::sync::{Arc, Weak};

use arrow::array::{Array, ArrayRef, Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use sql_gen::exec_jit_stream;

fn schema_kv() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]))
}

fn kv_batch(keys: Vec<i64>, values: Vec<i64>) -> RecordBatch {
    RecordBatch::try_new(
        schema_kv(),
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

/// `key -> value` map over the whole result (order-independent).
fn as_map(rb: &RecordBatch, agg_col: usize) -> BTreeMap<i64, i64> {
    let keys = i64s(rb, 0);
    let aggs = i64s(rb, agg_col);
    (0..rb.num_rows())
        .map(|i| (keys.value(i), aggs.value(i)))
        .collect()
}

/// Column `col` flattened to a `Vec<i64>`, preserving row order.
fn col_vec(rb: &RecordBatch, col: usize) -> Vec<i64> {
    let a = i64s(rb, col);
    (0..rb.num_rows()).map(|i| a.value(i)).collect()
}

#[test]
fn passthrough_across_three_batches() {
    // Three batches; a plain SELECT should emit every row, batch after batch, in
    // order — proving the outer loop advances and the inner loop refills per batch.
    let batches = vec![
        kv_batch(vec![1, 2], vec![10, 20]),
        kv_batch(vec![3], vec![30]),
        kv_batch(vec![4, 5, 6], vec![40, 50, 60]),
    ];
    let out = exec_jit_stream("SELECT key, value FROM t", "t", schema_kv(), batches).unwrap();
    assert_eq!(out.num_rows(), 6);
    assert_eq!(col_vec(&out, 0), vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(col_vec(&out, 1), vec![10, 20, 30, 40, 50, 60]);
}

#[test]
fn filter_across_batches() {
    let batches = vec![
        kv_batch(vec![1, 2, 3], vec![5, 15, 25]),
        kv_batch(vec![4, 5], vec![35, 8]),
    ];
    let out = exec_jit_stream(
        "SELECT key FROM t WHERE value > 10",
        "t",
        schema_kv(),
        batches,
    )
    .unwrap();
    // survivors: value 15 (key 2), 25 (key 3), 35 (key 4)
    assert_eq!(col_vec(&out, 0), vec![2, 3, 4]);
}

#[test]
fn scalar_sum_and_count_across_batches() {
    let batches = vec![
        kv_batch(vec![1, 1], vec![10, 20]),
        kv_batch(vec![2], vec![30]),
        kv_batch(vec![3, 3], vec![40, 50]),
    ];
    // Accumulators are registers that must persist across the outer batch loop.
    let out = exec_jit_stream(
        "SELECT sum(value), count(*) FROM t",
        "t",
        schema_kv(),
        batches,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 1);
    assert_eq!(i64s(&out, 0).value(0), 150); // sum
    assert_eq!(i64s(&out, 1).value(0), 5); // count(*)
}

#[test]
fn group_by_across_batches() {
    // The same keys appear in different batches, so the host GROUP BY state must
    // accumulate across batch boundaries (not reset per batch).
    let batches = vec![
        kv_batch(vec![1, 2], vec![10, 100]),
        kv_batch(vec![1, 3], vec![20, 300]),
        kv_batch(vec![2, 1], vec![200, 5]),
    ];
    let out = exec_jit_stream(
        "SELECT key, sum(value) FROM t GROUP BY key",
        "t",
        schema_kv(),
        batches,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 3);
    assert_eq!(
        as_map(&out, 1),
        BTreeMap::from([(1, 35), (2, 300), (3, 300)])
    );
}

#[test]
fn group_by_stream_matches_single_batch_oracle() {
    // Split vs. whole must agree.
    let keys: Vec<i64> = (0..50).map(|i| i % 7).collect();
    let values: Vec<i64> = (0..50).collect();

    let whole = kv_batch(keys.clone(), values.clone());
    let single = exec_jit_stream(
        "SELECT key, sum(value) FROM t GROUP BY key",
        "t",
        schema_kv(),
        [whole],
    )
    .unwrap();

    // Same rows, chopped into three batches.
    let batches = vec![
        kv_batch(keys[..17].to_vec(), values[..17].to_vec()),
        kv_batch(keys[17..33].to_vec(), values[17..33].to_vec()),
        kv_batch(keys[33..].to_vec(), values[33..].to_vec()),
    ];
    let streamed = exec_jit_stream(
        "SELECT key, sum(value) FROM t GROUP BY key",
        "t",
        schema_kv(),
        batches,
    )
    .unwrap();

    assert_eq!(as_map(&single, 1), as_map(&streamed, 1));
}

// --- String output across batches ---

fn schema_name() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new(
        "name",
        DataType::Utf8View,
        false,
    )]))
}

fn name_batch(names: Vec<&str>) -> RecordBatch {
    RecordBatch::try_new(schema_name(), vec![Arc::new(StringViewArray::from(names))]).unwrap()
}

#[test]
fn string_output_across_batches_survives_input_drop() {
    // Each batch's strings are appended into the output builder; since the builder
    // COPIES bytes, the result stays valid after every input batch is dropped.
    let batches = vec![
        name_batch(vec!["alice", "bob"]),
        name_batch(vec!["carol"]),
        name_batch(vec!["dave", "erin"]),
    ];
    let out = exec_jit_stream("SELECT name FROM t", "t", schema_name(), batches).unwrap();
    let names: &StringViewArray = out.column(0).as_any().downcast_ref().unwrap();
    let got: Vec<&str> = (0..out.num_rows()).map(|i| names.value(i)).collect();
    assert_eq!(got, vec!["alice", "bob", "carol", "dave", "erin"]);
}

// --- Residency: at most one input batch alive at a time ---

/// An iterator that asserts the *previous* batch it yielded was fully dropped
/// before the next is pulled — i.e. the stream never holds two input batches at
/// once. `ScanStream` drops `current` at the top of `next_batch`, so by the time
/// this `next` runs the prior batch's Arrow arrays must be at strong-count 0.
struct ResidencyChecked {
    inner: std::vec::IntoIter<RecordBatch>,
    last: Option<Weak<dyn Array>>,
}

impl Iterator for ResidencyChecked {
    type Item = RecordBatch;
    fn next(&mut self) -> Option<RecordBatch> {
        if let Some(prev) = self.last.take() {
            assert!(
                prev.upgrade().is_none(),
                "previous input batch was still resident when the next was pulled",
            );
        }
        let rb = self.inner.next()?;
        // A `Weak` to the batch's first column: its only strong ref now lives
        // inside `rb`, so once `ScanStream` drops `rb`, `upgrade()` returns `None`.
        let col: ArrayRef = rb.column(0).clone();
        self.last = Some(Arc::downgrade(&col));
        Some(rb)
    }
}

#[test]
fn only_one_input_batch_resident_at_a_time() {
    let batches = vec![
        kv_batch(vec![1, 2], vec![10, 20]),
        kv_batch(vec![3, 4], vec![30, 40]),
        kv_batch(vec![5, 6], vec![50, 60]),
    ];
    let iter = ResidencyChecked {
        inner: batches.into_iter(),
        last: None,
    };
    // The GROUP BY still accumulates correctly across the drops.
    let out = exec_jit_stream(
        "SELECT key, sum(value) FROM t GROUP BY key",
        "t",
        schema_kv(),
        iter,
    )
    .unwrap();
    assert_eq!(out.num_rows(), 6);
    assert_eq!(
        as_map(&out, 1),
        BTreeMap::from([(1, 10), (2, 20), (3, 30), (4, 40), (5, 50), (6, 60)])
    );
}
