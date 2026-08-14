//! One-call JIT execution: a SQL string over one `RecordBatch` → a result
//! `RecordBatch`. The single entry point for running queries.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use arrow_lms::{FfiArray, PreparedOutput, prepare_record_batch};
use datafusion_common::{DataFusionError, Result};
use rust_lms::pool::BytesPool;
use rust_lms::prelude::*;

use crate::codegen::{
    CodegenCtx, GroupHandle, agg_output_types, collect_str_literals, gen_collect, group_template,
};
use crate::group::{GroupState, KeyKind};
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
    run_operator(op, rb)
}

/// Compile and run a whole operator tree over `rb` in ONE kernel. A GROUP BY is a
/// push operator inside it — we just allocate its Rust-hosted [`GroupState`] up
/// front (it outlives the run) and bake the pointers into the codegen context; the
/// projection/filter above it run in the same kernel.
fn run_operator(op: Operator, rb: &RecordBatch) -> Result<RecordBatch> {
    let out_schema = normalize_out_schema(&op.output_schema());
    let capacity = op.max_output_rows(rb.num_rows());

    // Reject GROUP BY key types we don't support yet. A single key may be
    // `Int32`/`Int64`, `Float64`, or `Utf8View` (nullable or not). A composite
    // (multi-column) key is fixed-width only for now — each column `Int32`/`Int64`/
    // `Float64` (strings inside a composite key are a follow-up).
    if let Some(Operator::Aggregate {
        schema,
        group_exprs,
        ..
    }) = find_grouped(&op)
    {
        check_group_key(schema, group_exprs.len())?;
    }

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
    let mut lits: HashMap<String, *const u8> = HashMap::new();
    for s in lit_strs {
        if !lits.contains_key(s) {
            lits.insert(s.to_string(), pool.append(s.as_bytes()));
        }
    }

    // Allocate the GROUP BY state (sized to the row count, groups ≤ rows) and bake
    // its pointers. Must outlive `as_fn` — kept in `group_state` below.
    let mut group_state = match find_grouped(&op) {
        Some(Operator::Aggregate {
            aggs,
            group_exprs,
            schema,
            ..
        }) => {
            let agg_tys = agg_output_types(schema, group_exprs.len());
            let template = group_template(aggs, &agg_tys, schema);
            Some(GroupState::new(
                template,
                group_key_kind(schema, group_exprs.len()),
            ))
        }
        _ => None,
    };
    let group = group_state.as_mut().map(|gs| {
        Rc::new(GroupHandle {
            state: gs as *mut GroupState,
        })
    });

    let cx = CodegenCtx {
        rt,
        lits: Rc::new(lits),
        group,
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
    // Keep the interned-literal bytes and the group state alive across the run.
    drop(pool);
    drop(group_state);
    Ok(result)
}

/// The [`KeyKind`] for a single GROUP BY key column type, or `None` if unsupported.
fn key_kind(dt: &DataType) -> Option<KeyKind> {
    match dt {
        DataType::Int32 | DataType::Int64 => Some(KeyKind::Int),
        DataType::Float64 => Some(KeyKind::Float),
        DataType::Utf8View => Some(KeyKind::Str),
        _ => None,
    }
}

/// The [`KeyKind`] for the whole GROUP BY key — a composite key is a packed byte key,
/// so it uses the `Str` (bytes) table. Assumes [`check_group_key`] already passed.
fn group_key_kind(schema: &SchemaRef, n_keys: usize) -> KeyKind {
    if n_keys > 1 {
        KeyKind::Str
    } else {
        key_kind(schema.field(0).data_type()).expect("checked")
    }
}

/// Reject unsupported GROUP BY key columns (single: int/float/string; composite:
/// int/float/string columns).
fn check_group_key(schema: &SchemaRef, n_keys: usize) -> Result<()> {
    if n_keys == 1 {
        let ty = schema.field(0).data_type();
        if key_kind(ty).is_none() {
            return Err(DataFusionError::NotImplemented(format!(
                "GROUP BY key type {ty}"
            )));
        }
    } else {
        for i in 0..n_keys {
            let ty = schema.field(i).data_type();
            if !matches!(
                ty,
                DataType::Int32 | DataType::Int64 | DataType::Float64 | DataType::Utf8View
            ) {
                return Err(DataFusionError::NotImplemented(format!(
                    "composite GROUP BY key column type {ty}"
                )));
            }
        }
    }
    Ok(())
}

/// Find the (single) grouped `Aggregate` node in the plan, if any.
fn find_grouped(op: &Operator) -> Option<&Operator> {
    match op {
        Operator::Aggregate { group_exprs, .. } if !group_exprs.is_empty() => Some(op),
        Operator::Aggregate { input, .. }
        | Operator::Filter { input, .. }
        | Operator::Project { input, .. } => find_grouped(input),
        Operator::Scan { .. } => None,
    }
}

fn exec_err(e: impl std::fmt::Debug) -> DataFusionError {
    DataFusionError::Execution(format!("{e:?}"))
}
