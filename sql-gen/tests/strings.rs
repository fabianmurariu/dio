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

fn run(sql: &str, rb: &RecordBatch) -> RecordBatch {
    exec_jit(sql, "t", rb).expect("exec_jit")
}

fn i32s(rb: &RecordBatch, col: usize) -> &Int32Array {
    rb.column(col).as_any().downcast_ref().unwrap()
}
fn i64s(rb: &RecordBatch, col: usize) -> &Int64Array {
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
