//! Shared test harness: run every query on **both** codegen backends and assert the
//! results agree.
//!
//! The trick is that these functions have the *same names and signatures* as the
//! `sql_gen::exec_jit*` entry points, so a test file opts in by changing only its
//! `use` line (`use sql_gen::exec_jit` → `use common::exec_jit`) — no test-body edits.
//!
//! Each wrapper runs the query on Cranelift and, when the `llvm` feature is on, again on
//! LLVM/MLIR, then asserts the two `RecordBatch`es are identical before returning the
//! Cranelift result to the test's own assertions. A default `cargo test` runs exactly one
//! Cranelift pass (behaviourally identical to calling `sql_gen::exec_jit`); `cargo test
//! --features llvm` turns it into a live differential oracle over the whole SQL suite.
//!
//! Errors are backend-independent (they come from planning, before codegen), so an `Err`
//! from Cranelift is returned as-is without a second pass — error-path tests keep working.

// Not every test file uses every wrapper.
#![allow(dead_code)]

use arrow::record_batch::RecordBatch;
use datafusion_common::Result;
use rust_lms::prelude::JitBackend;
pub use sql_gen::StreamTable;

/// Assert two results agree: both `Ok` and batch-equal, or defer to the caller on `Err`.
/// Returns the first (Cranelift) result for the test to assert on.
fn cross_check(
    sql: &str,
    cranelift: Result<RecordBatch>,
    _llvm: impl FnOnce() -> Result<RecordBatch>,
) -> Result<RecordBatch> {
    #[cfg(feature = "llvm")]
    if let Ok(ref want) = cranelift {
        match _llvm() {
            Ok(got) => assert_eq!(
                *want, got,
                "Cranelift and LLVM produced different results for query:\n  {sql}"
            ),
            Err(e) => panic!("query succeeded on Cranelift but failed on LLVM:\n  {sql}\n  {e:?}"),
        }
    }
    let _ = sql;
    cranelift
}

/// Both-backends [`sql_gen::exec_jit`].
pub fn exec_jit(sql: &str, table: &str, rb: &RecordBatch) -> Result<RecordBatch> {
    let cranelift = sql_gen::exec_jit_with(JitBackend::Cranelift, sql, table, rb);
    cross_check(sql, cranelift, || {
        #[cfg(feature = "llvm")]
        {
            sql_gen::exec_jit_with(JitBackend::Llvm, sql, table, rb)
        }
        #[cfg(not(feature = "llvm"))]
        unreachable!()
    })
}

/// Both-backends [`sql_gen::exec_jit_stream`]. The batch stream is materialized once so it
/// can be replayed on each backend.
pub fn exec_jit_stream<I>(
    sql: &str,
    table: &str,
    schema: arrow::datatypes::SchemaRef,
    batches: I,
) -> Result<RecordBatch>
where
    I: IntoIterator<Item = RecordBatch>,
{
    let batches: Vec<RecordBatch> = batches.into_iter().collect();
    let cranelift = sql_gen::exec_jit_stream_with(
        JitBackend::Cranelift,
        sql,
        table,
        schema.clone(),
        batches.clone(),
    );
    cross_check(sql, cranelift, || {
        #[cfg(feature = "llvm")]
        {
            sql_gen::exec_jit_stream_with(
                JitBackend::Llvm,
                sql,
                table,
                schema.clone(),
                batches.clone(),
            )
        }
        #[cfg(not(feature = "llvm"))]
        unreachable!()
    })
}

/// Both-backends [`sql_gen::exec_jit_multi`]. Each table's stream is materialized once so it
/// can be replayed as a fresh `StreamTable` on each backend.
pub fn exec_jit_multi(sql: &str, tables: Vec<StreamTable>) -> Result<RecordBatch> {
    // Drain each stream to a replayable Vec (batches are cheap Arc clones).
    let mats: Vec<(String, arrow::datatypes::SchemaRef, Vec<RecordBatch>)> = tables
        .into_iter()
        .map(|t| (t.name, t.schema, t.batches.collect::<Vec<_>>()))
        .collect();
    let build = || {
        mats.iter()
            .map(|(n, s, b)| StreamTable::new(n.clone(), s.clone(), b.clone()))
            .collect::<Vec<_>>()
    };
    let cranelift = sql_gen::exec_jit_multi_with(JitBackend::Cranelift, sql, build());
    cross_check(sql, cranelift, || {
        #[cfg(feature = "llvm")]
        {
            sql_gen::exec_jit_multi_with(JitBackend::Llvm, sql, build())
        }
        #[cfg(not(feature = "llvm"))]
        unreachable!()
    })
}
