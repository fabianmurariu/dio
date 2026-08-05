//! Property tests for `Utf8View` string equality and output, against a plain-Rust
//! oracle. Columns are arbitrary UTF-8 (0..=100 chars per value), nullable, so
//! every view shape is exercised: empty, inline (≤12 bytes), and indirect (>12,
//! where equality must go past the length+prefix into the data buffer).

use std::sync::Arc;

use arrow::array::{Array, Int64Array, StringViewArray};
use arrow::datatypes::{DataType, Field, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use proptest::prelude::*;
use sql_gen::exec_jit;

/// A one-column `name: Utf8View` batch (nullable iff any value is null).
fn batch(values: &[Option<String>]) -> RecordBatch {
    let schema = Arc::new(ArrowSchema::new(vec![Field::new(
        "name",
        DataType::Utf8View,
        values.iter().any(Option::is_none),
    )]));
    let arr = StringViewArray::from(values.iter().map(Option::as_deref).collect::<Vec<_>>());
    RecordBatch::try_new(schema, vec![Arc::new(arr)]).unwrap()
}

/// A SQL single-quoted string literal (doubling embedded quotes).
fn sql_lit(s: &str) -> String {
    format!("'{}'", s.replace('\'', "''"))
}

fn count(rb: &RecordBatch, sql: &str) -> i64 {
    let out = exec_jit(sql, "t", rb).expect("exec_jit");
    out.column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .value(0)
}

/// A value guaranteed absent from `values` (and SQL-safe: pure ASCII).
fn absent_value(values: &[Option<String>]) -> String {
    let mut s = String::from("z");
    while values.iter().any(|v| v.as_deref() == Some(s.as_str())) {
        s.push('z');
    }
    s
}

/// A string value, 0..=100 chars, safe to embed in a SQL literal. Mixes two
/// shapes so both matter: arbitrary UTF-8 (multibyte, control-free) and a tiny
/// `a/b/c` alphabet — the latter makes length+prefix *collisions* common, which
/// forces equality past the view fast-reject into the byte compare for long
/// strings, and stresses the ≤12 / >12 boundary.
fn utf8_string() -> impl Strategy<Value = String> {
    prop_oneof![
        prop::collection::vec(
            any::<char>().prop_filter("no control chars", |c| !c.is_control()),
            0..=100,
        )
        .prop_map(|cs| cs.into_iter().collect::<String>()),
        prop::collection::vec(prop::sample::select(vec!['a', 'b', 'c']), 0..=100)
            .prop_map(|cs| cs.into_iter().collect::<String>()),
    ]
}

fn nullable_column() -> impl Strategy<Value = Vec<Option<String>>> {
    prop::collection::vec(prop::option::of(utf8_string()), 1..=15)
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 128, ..ProptestConfig::default() })]

    /// `WHERE name = <matching literal>` counts exactly the non-null rows whose
    /// bytes equal the literal; `<absent literal>` counts zero. Nulls never match.
    #[test]
    fn eq_literal_matches_oracle(
        values in nullable_column(),
        pick in any::<prop::sample::Index>(),
    ) {
        let rb = batch(&values);

        // A literal present in the column (an existing non-null value).
        let non_null: Vec<&String> = values.iter().flatten().collect();
        prop_assume!(!non_null.is_empty());
        let present = non_null[pick.index(non_null.len())].clone();

        let got = count(
            &rb,
            &format!("SELECT count(*) FROM t WHERE name = {}", sql_lit(&present)),
        );
        let expected = values
            .iter()
            .filter(|v| v.as_deref() == Some(present.as_str()))
            .count() as i64;
        prop_assert_eq!(got, expected, "literal {:?}", present);

        // A literal absent from the column.
        let absent = absent_value(&values);
        let zero = count(
            &rb,
            &format!("SELECT count(*) FROM t WHERE name = {}", sql_lit(&absent)),
        );
        prop_assert_eq!(zero, 0, "absent literal {:?}", absent);
    }

    /// `SELECT name` materializes the column verbatim, preserving values and nulls.
    #[test]
    fn select_name_roundtrip(values in nullable_column()) {
        let rb = batch(&values);
        let out = exec_jit("SELECT name FROM t", "t", &rb).expect("exec_jit");
        let s = out
            .column(0)
            .as_any()
            .downcast_ref::<StringViewArray>()
            .unwrap();
        prop_assert_eq!(s.len(), values.len());
        for (i, v) in values.iter().enumerate() {
            match v {
                Some(x) => {
                    prop_assert!(s.is_valid(i));
                    prop_assert_eq!(s.value(i), x.as_str());
                }
                None => prop_assert!(s.is_null(i)),
            }
        }
    }

    /// `octet_length(name)` equals each value's UTF-8 byte length (nulls stay null).
    #[test]
    fn octet_length_matches_oracle(values in nullable_column()) {
        let rb = batch(&values);
        let out = exec_jit("SELECT octet_length(name) FROM t", "t", &rb).expect("exec_jit");
        let lens = out
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::Int32Array>()
            .unwrap();
        for (i, v) in values.iter().enumerate() {
            match v {
                Some(x) => {
                    prop_assert!(lens.is_valid(i));
                    prop_assert_eq!(lens.value(i) as usize, x.len());
                }
                None => prop_assert!(lens.is_null(i)),
            }
        }
    }
}
