//! Phase 1 hash join: inner equi-join on a single Int key. The LEFT input is the
//! materialized build side, the RIGHT streams as the probe. Output is
//! `[left cols | right cols]`. Results are compared as sorted tuple multisets
//! (row order is probe-order × build-insertion-order, but we don't rely on it).

use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use sql_gen::{StreamTable, exec_jit_multi};

fn schema(cols: &[&str]) -> SchemaRef {
    Arc::new(Schema::new(
        cols.iter()
            .map(|c| Field::new(*c, DataType::Int64, false))
            .collect::<Vec<_>>(),
    ))
}

fn batch(s: &SchemaRef, cols: Vec<Vec<i64>>) -> RecordBatch {
    RecordBatch::try_new(
        s.clone(),
        cols.into_iter()
            .map(|c| Arc::new(Int64Array::from(c)) as _)
            .collect(),
    )
    .unwrap()
}

fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}

/// The whole result as a sorted `Vec` of row tuples (order-independent compare).
fn rows(rb: &RecordBatch) -> Vec<Vec<i64>> {
    let cols: Vec<&Int64Array> = (0..rb.num_columns()).map(|c| i64s(rb, c)).collect();
    let mut out: Vec<Vec<i64>> = (0..rb.num_rows())
        .map(|r| cols.iter().map(|c| c.value(r)).collect())
        .collect();
    out.sort();
    out
}

/// `a` (build/left) and `b` (probe/right), each one batch, joined on `a.ak = b.bk`.
fn join_ab(a: Vec<(i64, i64)>, b: Vec<(i64, i64)>) -> RecordBatch {
    let sa = schema(&["ak", "av"]);
    let sb = schema(&["bk", "bv"]);
    let a_rb = batch(
        &sa,
        vec![
            a.iter().map(|x| x.0).collect(),
            a.iter().map(|x| x.1).collect(),
        ],
    );
    let b_rb = batch(
        &sb,
        vec![
            b.iter().map(|x| x.0).collect(),
            b.iter().map(|x| x.1).collect(),
        ],
    );
    exec_jit_multi(
        "SELECT ak, av, bk, bv FROM a JOIN b ON a.ak = b.bk",
        vec![
            StreamTable::new("a", sa, vec![a_rb]),
            StreamTable::new("b", sb, vec![b_rb]),
        ],
    )
    .unwrap()
}

#[test]
fn inner_join_basic_and_multiplicity() {
    // ak=1 appears twice in the build → each matching probe row multiplies.
    let out = join_ab(
        vec![(1, 10), (2, 20), (1, 30), (3, 40)],
        vec![(1, 100), (2, 200), (5, 500)],
    );
    assert_eq!(
        rows(&out),
        vec![
            vec![1, 10, 1, 100],
            vec![1, 30, 1, 100],
            vec![2, 20, 2, 200],
        ]
    );
}

#[test]
fn inner_join_no_matches_is_empty() {
    let out = join_ab(vec![(1, 10), (2, 20)], vec![(3, 30), (4, 40)]);
    assert_eq!(out.num_rows(), 0);
}

#[test]
fn inner_join_empty_build_is_empty() {
    let out = join_ab(vec![], vec![(1, 10), (2, 20)]);
    assert_eq!(out.num_rows(), 0);
}

#[test]
fn inner_join_negative_keys() {
    // Negative keys exercise the i64-as-u64 reinterpret on both sides.
    let out = join_ab(vec![(-5, 1), (-5, 2), (7, 3)], vec![(-5, 100), (7, 200)]);
    assert_eq!(
        rows(&out),
        vec![
            vec![-5, 1, -5, 100],
            vec![-5, 2, -5, 100],
            vec![7, 3, 7, 200]
        ]
    );
}

#[test]
fn inner_join_multi_batch_build_and_probe() {
    // Build side split across 2 batches, probe side across 2 batches.
    let sa = schema(&["ak", "av"]);
    let sb = schema(&["bk", "bv"]);
    let a = vec![
        batch(&sa, vec![vec![1, 2], vec![10, 20]]),
        batch(&sa, vec![vec![1, 3], vec![30, 40]]), // ak=1 again, in a later batch
    ];
    let b = vec![
        batch(&sb, vec![vec![1], vec![100]]),
        batch(&sb, vec![vec![2, 1], vec![200, 101]]),
    ];
    let out = exec_jit_multi(
        "SELECT ak, av, bk, bv FROM a JOIN b ON a.ak = b.bk",
        vec![StreamTable::new("a", sa, a), StreamTable::new("b", sb, b)],
    )
    .unwrap();
    // ak=1 rows: (av 10 in batch0) and (av 30 in batch1); probe bk=1 twice (bv 100, 101).
    assert_eq!(
        rows(&out),
        vec![
            vec![1, 10, 1, 100],
            vec![1, 10, 1, 101],
            vec![1, 30, 1, 100],
            vec![1, 30, 1, 101],
            vec![2, 20, 2, 200],
        ]
    );
}

#[test]
fn inner_join_null_keys_never_match() {
    // Nullable key columns; NULL never matches NULL (or anything).
    let sa = Arc::new(Schema::new(vec![
        Field::new("ak", DataType::Int64, true),
        Field::new("av", DataType::Int64, false),
    ]));
    let sb = Arc::new(Schema::new(vec![
        Field::new("bk", DataType::Int64, true),
        Field::new("bv", DataType::Int64, false),
    ]));
    let a_rb = RecordBatch::try_new(
        sa.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), None, Some(2)])),
            Arc::new(Int64Array::from(vec![10, 20, 30])),
        ],
    )
    .unwrap();
    let b_rb = RecordBatch::try_new(
        sb.clone(),
        vec![
            Arc::new(Int64Array::from(vec![Some(1), None])),
            Arc::new(Int64Array::from(vec![100, 200])),
        ],
    )
    .unwrap();
    let out = exec_jit_multi(
        "SELECT ak, av, bk, bv FROM a JOIN b ON a.ak = b.bk",
        vec![
            StreamTable::new("a", sa, vec![a_rb]),
            StreamTable::new("b", sb, vec![b_rb]),
        ],
    )
    .unwrap();
    // Only ak=1 ⋈ bk=1 survives; the NULL rows on both sides match nothing.
    assert_eq!(rows(&out), vec![vec![1, 10, 1, 100]]);
}

#[test]
fn inner_join_then_filter() {
    // A WHERE above the join lowers to Filter over Join — should compose.
    let sa = schema(&["ak", "av"]);
    let sb = schema(&["bk", "bv"]);
    let a_rb = batch(&sa, vec![vec![1, 2, 3], vec![10, 20, 30]]);
    let b_rb = batch(&sb, vec![vec![1, 2, 3], vec![100, 200, 300]]);
    let out = exec_jit_multi(
        "SELECT ak, bv FROM a JOIN b ON a.ak = b.bk WHERE bv > 150",
        vec![
            StreamTable::new("a", sa, vec![a_rb]),
            StreamTable::new("b", sb, vec![b_rb]),
        ],
    )
    .unwrap();
    assert_eq!(rows(&out), vec![vec![2, 200], vec![3, 300]]);
}
