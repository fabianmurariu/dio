//! One-call JIT execution: a SQL string over one `RecordBatch` → a result
//! `RecordBatch`. The single entry point for running queries.

use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArray, PreparedOutput, prepare_record_batch};
use datafusion_common::{DataFusionError, Result};
use rust_lms::prelude::*;

use crate::codegen::gen_collect;
use crate::sql::sql_to_operator;

/// Parse `sql` over `table` (bound to `rb`'s schema), JIT-compile it, run it over
/// `rb`, and return the result as a `RecordBatch` — whether the query
/// materializes rows (`SELECT … WHERE …`) or folds to one row (scalar
/// aggregates like `count(*)`, `min`/`max`/`sum`).
pub fn exec_jit(sql: &str, table: &str, rb: &RecordBatch) -> Result<RecordBatch> {
    let op = sql_to_operator(sql, table, rb.schema())?;
    let out_schema = op.output_schema();
    let capacity = op.max_output_rows(rb.num_rows());

    let prepared_in = prepare_record_batch(rb).map_err(exec_err)?;
    let mut out = PreparedOutput::alloc(out_schema, capacity);

    let mut compiler = Compiler::new();
    let f = compiler.fun2(
        "query",
        |ctx, batch: Var<SRef<Slice<FfiArray>>>, sink: Var<SRefMut<Slice<FfiArray>>>| {
            gen_collect(ctx, batch, sink, &op, &op.output_schema())
        },
    );
    let compiled = compiler.compile(f).map_err(exec_err)?;

    let n = compiled.as_fn()(prepared_in.arrays(), out.as_ffi_mut());
    Ok(out.into_record_batch(n as usize))
}

fn exec_err(e: impl std::fmt::Debug) -> DataFusionError {
    DataFusionError::Execution(format!("{e:?}"))
}
