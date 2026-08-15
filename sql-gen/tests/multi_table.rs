//! Table-id routing: register several tables (each with its own stream) and check
//! that a query scanning a given table reads *that* table's stream — i.e. the
//! table id baked into the `Scan` selects the right `Inputs.streams` slot. No joins
//! yet (a query scans one table), but this is the plumbing a join threads.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use sql_gen::{StreamTable, exec_jit_multi};

/// A one-column `Int64` table named `n`.
fn n_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]))
}

fn n_batch(vals: Vec<i64>) -> RecordBatch {
    RecordBatch::try_new(n_schema(), vec![Arc::new(Int64Array::from(vals))]).unwrap()
}

fn n_table(name: &str, vals: Vec<i64>) -> StreamTable {
    StreamTable::new(name, n_schema(), vec![n_batch(vals)])
}

fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}

fn col_vec(rb: &RecordBatch, col: usize) -> Vec<i64> {
    let a = i64s(rb, col);
    (0..rb.num_rows()).map(|i| a.value(i)).collect()
}

#[test]
fn scan_first_table() {
    let out = exec_jit_multi(
        "SELECT n FROM a",
        vec![
            n_table("a", vec![1, 2, 3]),
            n_table("b", vec![100, 200, 300]),
        ],
    )
    .unwrap();
    assert_eq!(col_vec(&out, 0), vec![1, 2, 3]);
}

#[test]
fn scan_second_table_reads_its_own_stream() {
    // `b` is id 1. If the baked table id were wrong (0), this would return `a`'s
    // rows [1,2,3]; it must return `b`'s [100,200,300].
    let out = exec_jit_multi(
        "SELECT n FROM b",
        vec![
            n_table("a", vec![1, 2, 3]),
            n_table("b", vec![100, 200, 300]),
        ],
    )
    .unwrap();
    assert_eq!(col_vec(&out, 0), vec![100, 200, 300]);
}

#[test]
fn table_id_follows_registration_order() {
    // Same tables, reversed order: now `b` is id 0 and `a` is id 1. Each query must
    // still read its own table's data — routing follows the assigned id, not a slot.
    let tables = || {
        vec![
            n_table("b", vec![100, 200, 300]),
            n_table("a", vec![1, 2, 3]),
        ]
    };
    let from_a = exec_jit_multi("SELECT n FROM a", tables()).unwrap();
    let from_b = exec_jit_multi("SELECT n FROM b", tables()).unwrap();
    assert_eq!(col_vec(&from_a, 0), vec![1, 2, 3]);
    assert_eq!(col_vec(&from_b, 0), vec![100, 200, 300]);
}

#[test]
fn filter_on_second_table() {
    let out = exec_jit_multi(
        "SELECT n FROM b WHERE n > 150",
        vec![
            n_table("a", vec![1, 2, 3]),
            n_table("b", vec![100, 200, 300]),
        ],
    )
    .unwrap();
    assert_eq!(col_vec(&out, 0), vec![200, 300]);
}

#[test]
fn group_by_on_second_table_across_batches() {
    // Aggregate over a non-zero table id, split across batches — the baked id must
    // route through the group-by fold path too.
    let kv_schema = Arc::new(Schema::new(vec![
        Field::new("key", DataType::Int64, false),
        Field::new("value", DataType::Int64, false),
    ]));
    let kv = |keys: Vec<i64>, vals: Vec<i64>| {
        RecordBatch::try_new(
            kv_schema.clone(),
            vec![
                Arc::new(Int64Array::from(keys)),
                Arc::new(Int64Array::from(vals)),
            ],
        )
        .unwrap()
    };
    let out = exec_jit_multi(
        "SELECT key, sum(value) FROM b GROUP BY key",
        vec![
            n_table("a", vec![1, 2, 3]),
            StreamTable::new(
                "b",
                kv_schema.clone(),
                vec![kv(vec![1, 2], vec![10, 100]), kv(vec![1, 3], vec![20, 300])],
            ),
        ],
    )
    .unwrap();
    let map: BTreeMap<i64, i64> = (0..out.num_rows())
        .map(|i| (i64s(&out, 0).value(i), i64s(&out, 1).value(i)))
        .collect();
    assert_eq!(map, BTreeMap::from([(1, 30), (2, 100), (3, 300)]));
}

#[test]
fn unregistered_table_errors() {
    let err = exec_jit_multi("SELECT n FROM missing", vec![n_table("a", vec![1])]);
    assert!(err.is_err());
}
