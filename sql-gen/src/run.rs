//! One-call JIT execution: a SQL string over one `RecordBatch` → a result
//! `RecordBatch`. The single entry point for running queries.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::{DataFusionError, Result};
use rust_lms::opaque::Opaque;
use rust_lms::pool::BytesPool;
use rust_lms::prelude::*;

use datafusion_expr::Expr;

use crate::codegen::{
    CodegenCtx, GroupHandle, JoinHandle, agg_output_types, collect_str_literals, gen_build,
    gen_build_index, gen_collect, group_template,
};
use crate::group::{GroupState, KeyKind};
use crate::join::JoinState;
use crate::output::{OutCols, OutputHandle};
use crate::plan::Operator;
use crate::runtime::Runtime;
use crate::scan::{Inputs, ScanStream};
use crate::sql::{sql_to_operator, sql_to_operator_multi};

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
    run_operator(op, Inputs::single(rb.clone()))
}

/// Like [`exec_jit`] but over a **stream** of batches (all sharing `schema`): the
/// kernel pulls one batch at a time, so only one input batch is resident at once
/// (except where an operator deliberately retains — a future JOIN build side).
/// Scalar aggregates and GROUP BY fold across the whole stream in a single kernel.
/// `batches` is any `'static` iterator of same-schema batches (a `Vec`, a file
/// reader, …).
pub fn exec_jit_stream<I>(
    sql: &str,
    table: &str,
    schema: SchemaRef,
    batches: I,
) -> Result<RecordBatch>
where
    I: IntoIterator<Item = RecordBatch>,
    I::IntoIter: 'static,
{
    let op = sql_to_operator(sql, table, schema.clone())?;
    let inputs = Inputs::new(vec![ScanStream::new(schema, Box::new(batches.into_iter()))]);
    run_operator(op, inputs)
}

/// One named input table for [`exec_jit_multi`]: its schema and its batch stream.
/// The table's **id** is its position in the `tables` vector passed to
/// `exec_jit_multi`, so `Scan`s resolve to the matching stream.
pub struct StreamTable {
    pub name: String,
    pub schema: SchemaRef,
    pub batches: Box<dyn Iterator<Item = RecordBatch>>,
}

impl StreamTable {
    /// A table backed by a fixed set of batches.
    pub fn new(name: impl Into<String>, schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        StreamTable {
            name: name.into(),
            schema,
            batches: Box::new(batches.into_iter()),
        }
    }
}

/// Run `sql` over several named tables, each with its own batch stream. Table ids
/// are assigned by position in `tables`, and every `Scan` pulls from the matching
/// stream. A single query still scans one table (joins arrive later); this is the
/// id-routing plumbing joins will build on.
pub fn exec_jit_multi(sql: &str, tables: Vec<StreamTable>) -> Result<RecordBatch> {
    // Build the plan against the table names/schemas (ids = positions), then take
    // ownership of each stream in the same id order.
    let op = {
        let defs: Vec<(&str, SchemaRef)> = tables
            .iter()
            .map(|t| (t.name.as_str(), t.schema.clone()))
            .collect();
        sql_to_operator_multi(sql, &defs)?
    };
    let streams = tables
        .into_iter()
        .map(|t| ScanStream::new(t.schema, t.batches))
        .collect();
    run_operator(op, Inputs::new(streams))
}

/// Compile and run a whole operator tree over `inputs` in ONE kernel. A GROUP BY is
/// a push operator inside it — we just allocate its Rust-hosted [`GroupState`] up
/// front (it outlives the run) and bake the pointers into the codegen context; the
/// projection/filter above it run in the same kernel.
fn run_operator(op: Operator, mut inputs: Inputs) -> Result<RecordBatch> {
    // Hash join: materialize + index the LEFT (build) side first, then run the probe
    // kernel over the RIGHT with the built `JoinState` baked in.
    if let Some(Operator::Join { left, on, .. }) = find_join(&op) {
        let left_schema = left.output_schema();
        let key_col = left_key_col(&on[0].0, &left_schema)?;
        match left_schema.field(key_col).data_type() {
            DataType::Int32 | DataType::Int64 => {}
            other => {
                return Err(DataFusionError::NotImplemented(format!(
                    "join key type {other} (only Int for now)"
                )));
            }
        }
        // `run_operator` owns the `JoinState` for the whole join, so it lives at one
        // stable address across the build and probe kernels — the builders fill it in
        // place. Create the empty shell (a bare `Scan` clones its input batches into
        // the relation; a filtered/projected build starts empty), then fill it: a
        // bare scan → JIT index over the clone; else → ONE fused kernel that
        // materializes the surviving rows AND indexes them in a single pass.
        let mut join_state = match left.as_ref() {
            Operator::Scan { table, .. } => JoinState::from_clone(inputs.drain_table(*table)?),
            _ => JoinState::empty_relation(),
        };
        match left.as_ref() {
            Operator::Scan { .. } => build_join_index(&mut join_state, &left_schema, key_col)?,
            build_subtree => {
                build_join_materialized(&mut join_state, build_subtree, &on[0].0, &mut inputs)?
            }
        }
        // The baked pointer aliases `join_state`, which is not touched again until
        // after the probe kernel runs — the same contract as `GroupState`.
        let handle = Rc::new(JoinHandle {
            state: &mut join_state as *mut JoinState,
        });
        let result = run_kernel(&op, &mut inputs, Some(handle));
        drop(join_state);
        return result;
    }
    run_kernel(&op, &mut inputs, None)
}

/// Compile and run the join **build index kernel**: it iterates the (already
/// materialized) build relation and, per non-null row, reads the key column with
/// its known type and `join_insert`s the locator — the JIT counterpart of a host
/// per-value key loop. Takes no inputs and produces no output; its whole job is the
/// side effect of populating `js`'s key index.
fn build_join_index(js: &mut JoinState, rel_schema: &SchemaRef, key_col: usize) -> Result<()> {
    let mut compiler = Compiler::new();
    let rt = Runtime::register(&mut compiler);
    let handle = Rc::new(JoinHandle {
        state: js as *mut JoinState,
    });
    // Minimal codegen context — the index kernel only uses `rt` + `join`.
    let cx = CodegenCtx {
        rt,
        lits: Rc::new(HashMap::new()),
        group: None,
        out: Rc::new(OutputHandle { cols: Vec::new() }),
        join: Some(handle),
    };
    let rel_schema = rel_schema.clone();
    let f = compiler.fun0("build_index", move |ctx| {
        gen_build_index(ctx, rel_schema.clone(), key_col, &cx);
        unit()
    });
    let compiled = compiler.compile(f).map_err(exec_err)?;
    compiled.call();
    Ok(())
}

/// Fill `js` (an empty relation) via the **fused** build kernel for a non-bare-scan
/// build subtree: ONE pass that runs the subtree, writes each surviving row into the
/// build relation, and `join_insert`s its key + position — then installs the single
/// materialized batch, whose row indices the inserted locators reference.
fn build_join_materialized(
    js: &mut JoinState,
    build: &Operator,
    key_expr: &Expr,
    inputs: &mut Inputs,
) -> Result<()> {
    let out_schema = normalize_out_schema(&build.output_schema());
    let mut out = OutCols::alloc(&out_schema);

    let mut compiler = Compiler::new();
    let rt = Runtime::register(&mut compiler);

    // Intern the build subtree's string literals (kept alive until after the run).
    let mut pool = BytesPool::new();
    let mut lit_strs = Vec::new();
    collect_str_literals(build, &mut lit_strs);
    let mut lits: HashMap<String, *const u8> = HashMap::new();
    for s in lit_strs {
        if !lits.contains_key(s) {
            lits.insert(s.to_string(), pool.append(s.as_bytes()));
        }
    }

    let handle = Rc::new(JoinHandle {
        state: &mut *js as *mut JoinState,
    });
    let cx = CodegenCtx {
        rt,
        lits: Rc::new(lits),
        group: None,
        out: Rc::new(out.handle()),
        join: Some(handle),
    };
    let build_c = build.clone();
    let key_expr = key_expr.clone();
    let out_schema_c = out_schema.clone();
    let f = compiler.fun1(
        "build_materialize",
        move |ctx, in_v: Var<SRefMut<Opaque<Inputs>>>| {
            let inputs = ctx.bind(ref_mut_as_ptr(in_v));
            gen_build(ctx, inputs, &build_c, &key_expr, &out_schema_c, &cx)
        },
    );
    let compiled = compiler.compile(f).map_err(exec_err)?;
    let n = compiled.call(inputs);
    inputs.take_error()?;
    // Install the materialized batch: the locators inserted during the run reference
    // row indices into it.
    js.push_batch(out.into_record_batch(n as usize));
    Ok(())
}

/// Compile and run ONE kernel over `inputs`: allocate the output columns and any
/// GROUP BY state, bake them (plus an optional pre-built join handle) into the
/// codegen context, and materialize the result into a `RecordBatch`. Reused both
/// for a whole query and for materializing a join's build subtree.
fn run_kernel(
    op: &Operator,
    inputs: &mut Inputs,
    join: Option<Rc<JoinHandle>>,
) -> Result<RecordBatch> {
    let out_schema = normalize_out_schema(&op.output_schema());

    // Reject GROUP BY key types we don't support yet. A single key may be
    // `Int32`/`Int64`, `Float64`, or `Utf8View` (nullable or not). A composite
    // (multi-column) key is fixed-width only for now — each column `Int32`/`Int64`/
    // `Float64` (strings inside a composite key are a follow-up).
    if let Some(Operator::Aggregate {
        schema,
        group_exprs,
        ..
    }) = find_grouped(op)
    {
        check_group_key(schema, group_exprs.len())?;
    }

    let mut out = OutCols::alloc(&out_schema);

    let mut compiler = Compiler::new();
    let rt = Runtime::register(&mut compiler);

    // Intern every string literal into a pool that outlives the run; codegen bakes
    // each literal's stable pointer as a constant. The pool must stay alive until
    // after `as_fn` (the baked pointers reference its bytes).
    let mut pool = BytesPool::new();
    let mut lit_strs = Vec::new();
    collect_str_literals(op, &mut lit_strs);
    let mut lits: HashMap<String, *const u8> = HashMap::new();
    for s in lit_strs {
        if !lits.contains_key(s) {
            lits.insert(s.to_string(), pool.append(s.as_bytes()));
        }
    }

    // Allocate the GROUP BY state (sized to the row count, groups ≤ rows) and bake
    // its pointers. Must outlive `as_fn` — kept in `group_state` below.
    let mut group_state = match find_grouped(op) {
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

    // Bake the growable output columns' stable pointers into the codegen context
    // (like the GROUP BY state). The kernel appends rows through them; `out` must
    // outlive the run (its control blocks / builders are referenced by the kernel).
    let cx = CodegenCtx {
        rt,
        lits: Rc::new(lits),
        group,
        out: Rc::new(out.handle()),
        join,
    };

    let f = compiler.fun1("query", move |ctx, in_v: Var<SRefMut<Opaque<Inputs>>>| {
        let inputs = ctx.bind(ref_mut_as_ptr(in_v));
        gen_collect(ctx, inputs, op, &out_schema, &cx)
    });
    let compiled = compiler.compile(f).map_err(exec_err)?;

    let n = compiled.call(inputs);
    inputs.take_error()?;
    let result = out.into_record_batch(n as usize);
    // Keep the interned-literal bytes and group state alive across the run (the
    // kernel holds pointers into them).
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
        // No GROUP BY inside a join subtree yet (Phase 1 joins have none).
        Operator::Scan { .. } | Operator::Join { .. } => None,
    }
}

/// Find the (single) `Join` node on the plan's spine, if any.
fn find_join(op: &Operator) -> Option<&Operator> {
    match op {
        Operator::Join { .. } => Some(op),
        Operator::Filter { input, .. }
        | Operator::Project { input, .. }
        | Operator::Aggregate { input, .. } => find_join(input),
        Operator::Scan { .. } => None,
    }
}

/// Resolve a join key expression to a column index in the build side's schema.
/// Phase 1: the key must be a plain column (the host build reads it directly).
fn left_key_col(expr: &Expr, schema: &SchemaRef) -> Result<usize> {
    match expr {
        Expr::Column(c) => schema
            .index_of(&c.name)
            .map_err(|_| DataFusionError::Plan(format!("join key column not found: {}", c.name))),
        other => Err(DataFusionError::NotImplemented(format!(
            "join key must be a column (got {other:?})"
        ))),
    }
}

fn exec_err(e: impl std::fmt::Debug) -> DataFusionError {
    DataFusionError::Execution(format!("{e:?}"))
}
