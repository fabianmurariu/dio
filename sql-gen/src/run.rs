//! One-call JIT execution: a SQL string over one `RecordBatch` → a result
//! `RecordBatch`. The single entry point for running queries.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArray, PreparedOutput, prepare_record_batch};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::Expr;
use rust_lms::pool::BytesPool;
use rust_lms::prelude::*;

use crate::codegen::{Cx, agg_identities, collect_str_literals, gen_collect, gen_grouped};
use crate::group::GroupTable;
use crate::plan::Operator;
use crate::runtime::Runtime;
use crate::sql::sql_to_operator;

/// Normalize a plan's output schema for materialization: every string type maps
/// to `Utf8View`, since the executor always produces `StringViewArray` output
/// (datafusion types a string *literal* as `Utf8`, but we build it as a view).
fn normalize_out_schema(schema: &SchemaRef) -> SchemaRef {
    let fields = schema
        .fields()
        .iter()
        .map(|f| {
            let dt = match f.data_type() {
                DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => DataType::Utf8View,
                other => other.clone(),
            };
            Arc::new(Field::new(f.name(), dt, f.is_nullable()))
        })
        .collect::<Vec<_>>();
    Arc::new(Schema::new(fields))
}

/// Parse `sql` over `table` (bound to `rb`'s schema), JIT-compile it, run it over
/// `rb`, and return the result as a `RecordBatch` — whether the query
/// materializes rows (`SELECT … WHERE …`) or folds to one row (scalar
/// aggregates like `count(*)`, `min`/`max`/`sum`).
pub fn exec_jit(sql: &str, table: &str, rb: &RecordBatch) -> Result<RecordBatch> {
    let op = sql_to_operator(sql, table, rb.schema())?;

    // GROUP BY takes a dedicated kernel: it needs the output arrays + the
    // Rust-hosted hash table, which the push-model path doesn't thread.
    if let Some(g) = grouped_plan(&op) {
        return exec_grouped(g, &op, rb);
    }

    let out_schema = normalize_out_schema(&op.output_schema());
    let capacity = op.max_output_rows(rb.num_rows());

    let prepared_in = prepare_record_batch(rb).map_err(exec_err)?;
    let mut out = PreparedOutput::alloc(out_schema.clone(), capacity);

    let mut compiler = Compiler::new();
    let rt = Runtime::register(&mut compiler);

    // Intern every string literal into a pool that outlives the run; codegen bakes
    // each literal's stable pointer as a constant. The pool must stay alive until
    // after `as_fn` (the baked pointers reference its bytes).
    let mut pool = BytesPool::new();
    let mut lit_strs = Vec::new();
    collect_str_literals(&op, &mut lit_strs);
    let mut lits: HashMap<String, u64> = HashMap::new();
    for s in lit_strs {
        if !lits.contains_key(s) {
            lits.insert(s.to_string(), pool.append(s.as_bytes()) as u64);
        }
    }
    let cx = Cx {
        rt,
        lits: Arc::new(lits),
    };

    let f = compiler.fun2(
        "query",
        move |ctx, batch: Var<SRef<Slice<FfiArray>>>, sink: Var<SRefMut<Slice<FfiArray>>>| {
            gen_collect(ctx, batch, sink, &op, &out_schema, &cx)
        },
    );
    let compiled = compiler.compile(f).map_err(exec_err)?;

    let n = compiled.as_fn()(prepared_in.arrays(), out.as_ffi_mut());
    let result = out.into_record_batch(n as usize);
    drop(pool); // keep interned-literal bytes alive across the run, then free
    Ok(result)
}

/// A recognized GROUP BY plan: the aggregate's group keys, aggregates, input, and
/// output schema (`[group keys | aggregates]`).
struct Grouped {
    group_exprs: Vec<Expr>,
    aggs: Vec<Expr>,
    input: Operator,
    schema: SchemaRef,
}

/// Recognize a GROUP BY: a grouped `Aggregate`, optionally under a (trivial)
/// projection. Output columns follow the aggregate's own order for now.
fn grouped_plan(op: &Operator) -> Option<Grouped> {
    let agg = match op {
        Operator::Aggregate { group_exprs, .. } if !group_exprs.is_empty() => op,
        Operator::Project { input, .. } => match input.as_ref() {
            Operator::Aggregate { group_exprs, .. } if !group_exprs.is_empty() => input.as_ref(),
            _ => return None,
        },
        _ => return None,
    };
    match agg {
        Operator::Aggregate {
            group_exprs,
            aggs,
            input,
            schema,
        } => Some(Grouped {
            group_exprs: group_exprs.clone(),
            aggs: aggs.clone(),
            input: (**input).clone(),
            schema: schema.clone(),
        }),
        _ => unreachable!(),
    }
}

/// Execute a GROUP BY: fold input rows into per-group accumulator arrays (sized to
/// the row count) via the JIT kernel + Rust-hosted [`GroupTable`], then truncate
/// to the number of groups.
fn exec_grouped(g: Grouped, op: &Operator, rb: &RecordBatch) -> Result<RecordBatch> {
    let out_schema = normalize_out_schema(&g.schema);
    let capacity = rb.num_rows(); // groups ≤ rows

    let prepared_in = prepare_record_batch(rb).map_err(exec_err)?;
    let mut out = PreparedOutput::alloc(out_schema.clone(), capacity);
    // Accumulator columns start at 1 (column 0 is the group key). Pre-fill min/max
    // with their identity so the fold is correct from the first row.
    for (j, id) in agg_identities(&g.aggs).into_iter().enumerate() {
        out.fill_i64(1 + j, id);
    }

    let mut compiler = Compiler::new();
    let rt = Runtime::register(&mut compiler);

    let mut pool = BytesPool::new();
    let mut lit_strs = Vec::new();
    collect_str_literals(op, &mut lit_strs);
    let mut lits: HashMap<String, u64> = HashMap::new();
    for s in lit_strs {
        if !lits.contains_key(s) {
            lits.insert(s.to_string(), pool.append(s.as_bytes()) as u64);
        }
    }
    let cx = Cx {
        rt,
        lits: Arc::new(lits),
    };

    let mut table = GroupTable::new();
    let Grouped {
        group_exprs,
        aggs,
        input,
        ..
    } = g;

    let f = compiler.fun3(
        "group_by",
        move |ctx,
              batch: Var<SRef<Slice<FfiArray>>>,
              table: Var<SRefMut<Opaque<GroupTable>>>,
              sink: Var<SRefMut<Slice<FfiArray>>>| {
            gen_grouped(ctx, batch, table, sink, &group_exprs, &aggs, &input, &cx)
        },
    );
    let compiled = compiler.compile(f).map_err(exec_err)?;

    let num_groups = compiled.as_fn()(prepared_in.arrays(), &mut table, out.as_ffi_mut());
    let result = out.into_record_batch(num_groups as usize);
    drop(pool);
    Ok(result)
}

fn exec_err(e: impl std::fmt::Debug) -> DataFusionError {
    DataFusionError::Execution(format!("{e:?}"))
}
