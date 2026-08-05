//! Reading a `Utf8View` (StringView) string column and integrating it into SQL:
//! `octet_length` (reads the view's byte length) and `count` (reads validity).

use std::sync::Arc;

use arrow::array::{Array, Int32Array, Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use sql_gen::exec_jit;

/// A batch with a `name: Utf8View` column (nullability inferred).
fn batch(names: StringViewArray) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "name",
        DataType::Utf8View,
        names.null_count() > 0,
    )]));
    RecordBatch::try_new(schema, vec![Arc::new(names)]).unwrap()
}

/// A batch with two `Utf8View` columns `a` and `b`.
fn batch2(a: StringViewArray, b: StringViewArray) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![
        Field::new("a", DataType::Utf8View, a.null_count() > 0),
        Field::new("b", DataType::Utf8View, b.null_count() > 0),
    ]));
    RecordBatch::try_new(schema, vec![Arc::new(a), Arc::new(b)]).unwrap()
}

fn run(sql: &str, rb: &RecordBatch) -> RecordBatch {
    exec_jit(sql, "t", rb).expect("exec_jit")
}

fn i32s(rb: &RecordBatch, col: usize) -> &Int32Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}
fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}
fn strs(rb: &RecordBatch, col: usize) -> &StringViewArray {
    rb.column(col).as_any().downcast_ref().unwrap()
}

#[test]
fn octet_length_of_string_column() {
    // Mix of inline (<=12 bytes) and indirect (>12) strings, incl. multi-byte.
    let rb = batch(StringViewArray::from(vec![
        "ok",                   // 2
        "hello",                // 5
        "a-longer-string-here", // 20 (indirect)
        "café",                 // 5 bytes (é = 2)
    ]));
    let out = run("SELECT octet_length(name) FROM t", &rb);

    assert_eq!(out.num_rows(), 4);
    // octet_length = byte length, straight from each view's low 32 bits.
    assert_eq!(i32s(&out, 0).values(), &[2, 5, 20, 5]);
}

#[test]
fn octet_length_with_filter() {
    let rb = batch(StringViewArray::from(vec!["a", "bbbb", "cc", "ddddddd"]));
    // octet_length(name) for the rows longer than 2 bytes
    let out = run(
        "SELECT octet_length(name) FROM t WHERE octet_length(name) > 2",
        &rb,
    );
    assert_eq!(i32s(&out, 0).values(), &[4, 7]);
}

#[test]
fn count_reads_string_validity() {
    let rb = batch(StringViewArray::from(vec![
        Some("x"),
        None,
        Some("this-one-is-indirect"),
        None,
    ]));
    let out = run("SELECT count(name) FROM t", &rb);
    // count(col) counts non-null strings
    assert_eq!(i64s(&out, 0).value(0), 2);
}

#[test]
fn equality_short_literal_filter() {
    // `active` (6 bytes) is inline; equality is a pure staged view compare.
    let rb = batch(StringViewArray::from(vec![
        "active",
        "idle",
        "active",
        "a-much-longer-status-value", // indirect, must NOT match
        "active",
    ]));
    let out = run("SELECT count(*) FROM t WHERE name = 'active'", &rb);
    assert_eq!(i64s(&out, 0).value(0), 3);
}

#[test]
fn not_equal_short_literal() {
    let rb = batch(StringViewArray::from(vec!["ok", "no", "ok", "maybe"]));
    let out = run("SELECT count(*) FROM t WHERE name <> 'ok'", &rb);
    assert_eq!(i64s(&out, 0).value(0), 2);
}

#[test]
fn equality_null_row_dropped() {
    // NULL = 'x' is unknown -> the null row is filtered out.
    let rb = batch(StringViewArray::from(vec![
        Some("x"),
        None,
        Some("x"),
        Some("y"),
    ]));
    let out = run("SELECT count(*) FROM t WHERE name = 'x'", &rb);
    assert_eq!(i64s(&out, 0).value(0), 2);
}

#[test]
fn octet_length_with_string_filter() {
    let rb = batch(StringViewArray::from(vec!["ok", "fail", "ok"]));
    let out = run("SELECT octet_length(name) FROM t WHERE name = 'ok'", &rb);
    assert_eq!(i32s(&out, 0).values(), &[2, 2]);
}

#[test]
fn equality_long_literal_uses_extern() {
    // 'in-progress-status' is 18 bytes (> 12) -> indirect view -> extern fallback.
    let rb = batch(StringViewArray::from(vec![
        "in-progress-status",
        "done", // inline, no match
        "in-progress-status",
        "in-progress-statuz", // same length + prefix, differs at the end
        "in-progress-status",
    ]));
    let out = run(
        "SELECT count(*) FROM t WHERE name = 'in-progress-status'",
        &rb,
    );
    // exactly the 3 exact matches; the near-miss must be rejected by the extern
    assert_eq!(i64s(&out, 0).value(0), 3);
}

#[test]
fn column_equals_column() {
    // Mix of matching / non-matching rows, short (inline) and long (indirect),
    // to exercise the byte-compare extern across both view shapes.
    let a = StringViewArray::from(vec![
        "ok",                  // == b -> match
        "left",                // != b
        "a-long-shared-value", // 19 bytes, indirect, == b -> match
        "a-long-shared-value", // != b (near-miss below)
    ]);
    let b = StringViewArray::from(vec![
        "ok",
        "right",
        "a-long-shared-value",
        "a-long-shared-valuE", // differs only in last byte
    ]);
    let rb = batch2(a, b);
    let out = run("SELECT count(*) FROM t WHERE a = b", &rb);
    assert_eq!(i64s(&out, 0).value(0), 2);
}

#[test]
fn column_equals_column_with_nulls() {
    // NULL = anything is unknown -> dropped, even NULL = NULL.
    let a = StringViewArray::from(vec![Some("x"), None, Some("y"), None]);
    let b = StringViewArray::from(vec![Some("x"), Some("z"), None, None]);
    let rb = batch2(a, b);
    let out = run("SELECT count(*) FROM t WHERE a = b", &rb);
    assert_eq!(i64s(&out, 0).value(0), 1);
}

#[test]
fn select_string_column_passthrough() {
    // Mix of inline (<=12) and indirect (>12) and multi-byte, all materialized out.
    let rb = batch(StringViewArray::from(vec![
        "ok",
        "a-longer-indirect-value",
        "café",
        "x",
    ]));
    let out = run("SELECT name FROM t", &rb);
    let s = strs(&out, 0);
    assert_eq!(s.len(), 4);
    assert_eq!(s.value(0), "ok");
    assert_eq!(s.value(1), "a-longer-indirect-value");
    assert_eq!(s.value(2), "café");
    assert_eq!(s.value(3), "x");
}

#[test]
fn select_string_column_filtered() {
    let rb = batch(StringViewArray::from(vec![
        "keep-this-long-one",
        "no",
        "keep-this-long-one",
        "yes",
    ]));
    let out = run("SELECT name FROM t WHERE octet_length(name) > 3", &rb);
    let s = strs(&out, 0);
    assert_eq!(s.len(), 2);
    assert_eq!(s.value(0), "keep-this-long-one");
    assert_eq!(s.value(1), "keep-this-long-one");
}

#[test]
fn select_string_column_with_nulls() {
    let rb = batch(StringViewArray::from(vec![
        Some("a"),
        None,
        Some("this-is-indirect-long"),
        None,
    ]));
    let out = run("SELECT name FROM t", &rb);
    let s = strs(&out, 0);
    assert_eq!(s.len(), 4);
    assert!(s.is_valid(0));
    assert_eq!(s.value(0), "a");
    assert!(s.is_null(1));
    assert!(s.is_valid(2));
    assert_eq!(s.value(2), "this-is-indirect-long");
    assert!(s.is_null(3));
}

#[test]
fn select_string_literal_short() {
    // A short (inline) literal, materialized once per surviving row.
    let rb = batch(StringViewArray::from(vec!["a", "b", "c"]));
    let out = run("SELECT 'hi' FROM t", &rb);
    let s = strs(&out, 0);
    assert_eq!(s.len(), 3);
    assert_eq!(s.value(0), "hi");
    assert_eq!(s.value(1), "hi");
    assert_eq!(s.value(2), "hi");
}

#[test]
fn select_string_literal_long() {
    // A long (>12, indirect) literal, interned in the pool and output.
    let rb = batch(StringViewArray::from(vec!["x", "y"]));
    let out = run("SELECT 'a-fairly-long-literal-value' FROM t", &rb);
    let s = strs(&out, 0);
    assert_eq!(s.len(), 2);
    assert_eq!(s.value(0), "a-fairly-long-literal-value");
    assert_eq!(s.value(1), "a-fairly-long-literal-value");
}

#[test]
fn select_literal_alongside_column() {
    let rb = batch(StringViewArray::from(vec!["keep", "drop", "keep"]));
    let out = run("SELECT name, 'tag' FROM t WHERE name = 'keep'", &rb);
    let names = strs(&out, 0);
    let tags = strs(&out, 1);
    assert_eq!(names.len(), 2);
    assert_eq!(names.value(0), "keep");
    assert_eq!(names.value(1), "keep");
    assert_eq!(tags.value(0), "tag");
    assert_eq!(tags.value(1), "tag");
}

#[test]
fn long_literal_octet_length_and_filter() {
    let rb = batch(StringViewArray::from(vec![
        "alpha",
        "a-fairly-long-name-value",
        "a-fairly-long-name-value",
    ]));
    let out = run(
        "SELECT octet_length(name) FROM t WHERE name = 'a-fairly-long-name-value'",
        &rb,
    );
    // both surviving rows are 24 bytes
    assert_eq!(i32s(&out, 0).values(), &[24, 24]);
}
