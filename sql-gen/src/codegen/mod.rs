//! Staged code generation: walk an [`Operator`] tree and *emit* rust-lms staged
//! ops that JIT-compile to a kernel — the first Futamura projection for
//! relational algebra (per `docs/sql_to_c.pdf`). One entry, [`gen_collect`],
//! writes results into an output batch: surviving rows for `Scan`/`Filter`/
//! `Project`, a single folded row for a scalar `Aggregate`.
//!
//! Scalar expressions are datafusion [`Expr`] values (columns, literals,
//! comparison / boolean / arithmetic operators). Nulls follow static
//! nullability: only nullable columns/exprs carry an `is_valid` bit, propagated
//! through evaluation; non-nullable ones emit no validity IR.

mod aggregate;
mod expr;
mod grouping;
mod join;
mod numeric;
mod strings;

pub use grouping::{agg_output_types, group_template};

use std::cell::Cell;
use std::collections::HashMap;
use std::rc::Rc;

use arrow::array::{StringViewArray, StringViewBuilder};
use arrow::datatypes::{DataType, Field, SchemaRef};
use arrow_lms::ffi::FfiArrayType;
use arrow_lms::{ArrayBatchOps, FfiArray, PrimitiveArrayView};
use datafusion_common::ScalarValue;
use datafusion_expr::Expr;
use rust_lms::prelude::*;
use rust_lms_std::SVec;

use crate::group::GroupState;
use crate::join::JoinState;
use crate::output::{OutColHandle, OutputHandle};
use crate::plan::Operator;
use crate::runtime::Runtime;
use crate::scan::Inputs;

fn string_builder_mut(
    builder: *mut StringViewBuilder,
) -> impl Staged<Out = SMutPtr<Opaque<StringViewBuilder>>> + Copy {
    const_mut_ptr::<Opaque<StringViewBuilder>>(builder)
}

use crate::value::{ColVal, Nullness, Row, StrVal};

use aggregate::{Agg, CountFast, count_fast};
use expr::{gen_expr, gen_predicate};
use grouping::gen_grouped;
use join::gen_join;
pub(crate) use join::{gen_build, gen_build_index};
use numeric::{coerce_f64, coerce_i32, tag, to_i64};
use strings::resolve;

/// Codegen context threaded through the walk: the runtime extern handles plus the
/// build-time-interned string literals (value → stable `BytesPool` pointer). The
/// `Rc` is because the push-model `yld` closures are `'static` and can't borrow;
/// codegen is single-threaded, so `Rc` (not `Arc`) is the honest sharing type.
#[derive(Clone)]
pub struct CodegenCtx {
    pub rt: Runtime,
    pub lits: Rc<HashMap<String, *const u8>>,
    /// Baked pointers to the GROUP BY's host-side state, when the plan has one.
    /// Single group-by for now; a `Vec` indexed by plan order generalizes to CTEs.
    pub group: Option<Rc<GroupHandle>>,
    /// Baked pointers to the growable output columns (see `output::OutCols`). The
    /// kernel appends each emitted row into these via `SVec::push` / a string builder.
    pub out: Rc<OutputHandle>,
    /// Baked pointer to the hash-join build state, when the plan has a join. Single
    /// join for now; a `Vec` indexed by plan order generalizes to multiple joins.
    pub join: Option<Rc<JoinHandle>>,
    /// Kernel-wide "poison" flag: a `Var<bool>` set to `true` when a fallible
    /// callback returns its error sentinel (see [`crate::status`]). The scan's batch
    /// loop checks it so a failed run stops pulling work and returns to the driver,
    /// which then surfaces the recorded `DataFusionError`. Populated once at kernel
    /// entry ([`gen_collect`]); `None` for helper kernels that never fold fallible
    /// callbacks. Shared across `clone()` so the setter and checkers agree.
    pub poison: Rc<Cell<Option<Var<bool>>>>,
}

/// Typed host pointers into a GROUP BY's Rust-hosted state (see `group::GroupState`),
/// captured before compilation and baked (as constants) into the kernel. Real
/// pointers, not `u64`s — the pointee type is checked at stage 0.
pub struct GroupHandle {
    /// The whole host-side GROUP BY state (hash table + growable records buffer),
    /// handed to the `group_upsert`/`group_len`/`group_records_base` externs. One
    /// baked pointer: growth happens inside the externs, so nothing here dangles.
    pub state: *mut GroupState,
}

/// Baked pointer to the hash-join's host-side [`JoinState`] (build relation + index),
/// handed to the `join_probe_*` / `join_left_batch` externs. Built before compile,
/// outlives the run.
pub struct JoinHandle {
    pub state: *mut JoinState,
}

/// Collect the string-literal values in `op`'s expressions, so `exec_jit` can
/// intern each into the `BytesPool` before codegen (see [`CodegenCtx::lits`]).
pub fn collect_str_literals<'a>(op: &'a Operator, out: &mut Vec<&'a str>) {
    fn expr_lits<'b>(e: &'b Expr, out: &mut Vec<&'b str>) {
        match e {
            Expr::Literal(
                ScalarValue::Utf8(Some(s))
                | ScalarValue::Utf8View(Some(s))
                | ScalarValue::LargeUtf8(Some(s)),
                _,
            ) => out.push(s),
            Expr::BinaryExpr(be) => {
                expr_lits(&be.left, out);
                expr_lits(&be.right, out);
            }
            Expr::ScalarFunction(f) => f.args.iter().for_each(|a| expr_lits(a, out)),
            _ => {}
        }
    }
    match op {
        Operator::Scan { .. } => {}
        Operator::Filter { predicate, input } => {
            expr_lits(predicate, out);
            collect_str_literals(input, out);
        }
        Operator::Project { exprs, input, .. } => {
            exprs.iter().for_each(|e| expr_lits(e, out));
            collect_str_literals(input, out);
        }
        Operator::Aggregate { aggs, input, .. } => {
            aggs.iter().for_each(|e| expr_lits(e, out));
            collect_str_literals(input, out);
        }
        Operator::Join { left, right, .. } => {
            collect_str_literals(left, out);
            collect_str_literals(right, out);
        }
    }
}

/// Dispatch a runtime arrow `DataType` to the matching [`Prim`] type parameter,
/// running `$body` (in which `$m` is a type alias for the primitive) once for the
/// selected type. Turns the per-type `match dt` into one place.
macro_rules! dispatch_prim {
    ($dt:expr, $m:ident => $body:expr) => {
        match $dt {
            DataType::Int32 => {
                type $m = i32;
                $body
            }
            DataType::Int64 => {
                type $m = i64;
                $body
            }
            DataType::Float64 => {
                type $m = f64;
                $body
            }
            other => panic!("unsupported column type: {other}"),
        }
    };
}

/// A primitive column type: the staged type used to read/write it, plus how it
/// maps to and from the type-erased [`ColVal`]. Adding a type is one `impl` here
/// and one arm in [`dispatch_prim!`].
trait Prim: StagedType + CopyType + 'static {
    /// Wrap a staged value of this type into a `ColVal`.
    fn wrap(value: Var<Self>, null: Nullness) -> ColVal;
    /// Coerce any numeric `ColVal` to a staged value of this type.
    fn coerce(ctx: &mut Ctx, cv: ColVal) -> Var<Self>;
}

impl Prim for i32 {
    fn wrap(value: Var<Self>, null: Nullness) -> ColVal {
        ColVal::I32(value, null)
    }
    fn coerce(ctx: &mut Ctx, cv: ColVal) -> Var<Self> {
        coerce_i32(ctx, cv)
    }
}

impl Prim for i64 {
    fn wrap(value: Var<Self>, null: Nullness) -> ColVal {
        ColVal::I64(value, null)
    }
    fn coerce(ctx: &mut Ctx, cv: ColVal) -> Var<Self> {
        to_i64(ctx, cv)
    }
}

impl Prim for f64 {
    fn wrap(value: Var<Self>, null: Nullness) -> ColVal {
        ColVal::F64(value, null)
    }
    fn coerce(ctx: &mut Ctx, cv: ColVal) -> Var<Self> {
        coerce_f64(ctx, cv)
    }
}

/// A staged expression yielding one lifetime-free `(ptr, len)` descriptor array.
/// The scan or join owner retains the pointed-to batch for the enclosing loop.
pub trait BatchSource: Staged<Out = FatSliceType<FfiArray>> + Copy + 'static {}
impl<T> BatchSource for T where T: Staged<Out = FatSliceType<FfiArray>> + Copy + 'static {}

/// Raw staged pointer derived from the kernel's real `&mut Inputs` parameter.
/// It is threaded through the operator walk so [`gen_scan`] can call `scan_next`
/// without fabricating an internal reference lifetime.
pub trait InputsSource: Staged<Out = SMutPtr<Opaque<Inputs>>> + Copy + 'static {}
impl<T> InputsSource for T where T: Staged<Out = SMutPtr<Opaque<Inputs>>> + Copy + 'static {}

/// Downstream continuation, invoked once per emitted row at code-generation
/// time; the [`Row`] rides by value (cheap `Copy` handles).
pub(crate) type Yld = Box<dyn FnOnce(&mut Ctx, Row) + 'static>;

/// Emit a kernel that appends `plan`'s emitted rows into the baked output columns
/// (`cx.out`) and returns the row count. The single entry point — a scalar
/// `Aggregate` is just a push operator that emits one row (see [`gen_op`]).
pub fn gen_collect<I: InputsSource>(
    ctx: &mut Ctx,
    inputs: I,
    plan: &Operator,
    out_schema: &SchemaRef,
    cx: &CodegenCtx,
) -> Var<u64> {
    let n = ctx.var(0u64);
    // Kernel-wide error-stop flag: fallible callbacks set it via their sentinel
    // (see `stop_if_failed`) and the scan batch loop checks it. Created once here so
    // it dominates every loop in the kernel.
    let poison = ctx.var(false);
    cx.poison.set(Some(poison));
    let fields = out_schema.fields().clone();
    let cx_c = cx.clone();
    gen_op(
        plan,
        ctx,
        inputs,
        cx,
        Box::new(move |ctx, row| {
            for (c, (cv, field)) in row.iter().zip(fields.iter()).enumerate() {
                write_col(ctx, c, field, *cv, &cx_c);
            }
            ctx.store(n, add(n, 1u64));
        }),
    );
    n
}

pub(crate) fn gen_op<I: InputsSource>(
    op: &Operator,
    ctx: &mut Ctx,
    inputs: I,
    cx: &CodegenCtx,
    yld: Yld,
) {
    match op {
        Operator::Scan { table, schema } => {
            gen_scan(ctx, inputs, *table as u64, schema.clone(), cx, yld)
        }

        Operator::Filter { predicate, input } => {
            let predicate = predicate.clone();
            let schema = input.output_schema();
            let cx_c = cx.clone();
            gen_op(
                input,
                ctx,
                inputs,
                cx,
                Box::new(move |ctx, row| {
                    let keep = gen_predicate(ctx, &predicate, &schema, &row, &cx_c);
                    ctx.if_then(keep, move |ctx| yld(ctx, row));
                }),
            );
        }

        Operator::Project { exprs, input, .. } => {
            let exprs = exprs.clone();
            let schema = input.output_schema();
            let cx_c = cx.clone();
            gen_op(
                input,
                ctx,
                inputs,
                cx,
                Box::new(move |ctx, row| {
                    let projected: Row = exprs
                        .iter()
                        .map(|e| gen_expr(ctx, e, &schema, &row, &cx_c))
                        .collect();
                    yld(ctx, projected);
                }),
            );
        }

        // GROUP BY: fold into the Rust-hosted group state, then emit one row per
        // group downstream — a normal push operator, so the projection/filter
        // above it run in the same kernel.
        Operator::Aggregate { group_exprs, .. } if !group_exprs.is_empty() => {
            gen_grouped(ctx, inputs, op, cx, yld);
        }

        // Scalar aggregate: fold every input row into accumulators, then emit
        // exactly one result row downstream (after the loop).
        Operator::Aggregate {
            aggs,
            input,
            schema,
            ..
        } => {
            // Whole-batch fast path: a lone `count(*)` / `count(col)` directly over
            // a `Scan` (no filter) needs no per-row loop — the answer is the sum of
            // per-batch counts. `count(*)` adds the row count; `count(col)` adds the
            // non-null count (`len - null_count`, folding both nullable and
            // non-nullable columns since `null_count` is 0 for the latter).
            if aggs.len() == 1
                && let Operator::Scan {
                    table,
                    schema: scan_schema,
                } = input.as_ref()
                && let Some(cf) = count_fast(&aggs[0], scan_schema)
            {
                let count = ctx.var(0i64);
                for_each_batch(
                    ctx,
                    inputs,
                    *table as u64,
                    scan_schema.clone(),
                    cx,
                    move |ctx, batch, len| {
                        let inc = match cf {
                            CountFast::Star => len,
                            CountFast::Col(col) => {
                                // SAFETY: the scan schema proves `col` exists;
                                // this path reads only validity metadata, not values.
                                let nulls = ctx.bind(
                                    unsafe { batch.primitive::<u64>(col) }
                                        .validity()
                                        .null_count(),
                                );
                                ctx.bind(sub(len, nulls))
                            }
                        };
                        ctx.store(count, add(count, int_cast::<i64, u64, _>(inc)));
                    },
                );
                yld(ctx, vec![ColVal::I64(count, Nullness::NonNull)]);
                return;
            }

            let fields = schema.fields().clone();
            let accs: Vec<Agg> = aggs
                .iter()
                .enumerate()
                .map(|(c, e)| Agg::init(ctx, e, &fields[c]))
                .collect();

            let loop_accs = accs.clone();
            let input_schema = input.output_schema();
            let cx_c = cx.clone();
            gen_op(
                input,
                ctx,
                inputs,
                cx,
                Box::new(move |ctx, row| {
                    for agg in &loop_accs {
                        agg.update(ctx, &row, &input_schema, &cx_c);
                    }
                }),
            );

            let mut result: Row = Vec::with_capacity(accs.len());
            for agg in &accs {
                result.push(agg.finalize(ctx));
            }
            yld(ctx, result);
        }

        // Hash join: the left (build) side was materialized + indexed host-side
        // before compile; here we codegen the streaming probe over the right side.
        Operator::Join { .. } => gen_join(ctx, inputs, op, cx, yld),
    }
}

/// The staged raw slice descriptor received by a [`for_each_batch`] body.
type ScanBatch = Var<FatSliceType<FfiArray>>;

/// Drive the OUTER batch loop of table `table`: pull batches from the stream
/// (`scan_next`, null = exhausted → break), rebuild each `&[FfiArray]` batch, and
/// hand `body` the batch plus its row count. The single place that knows the
/// batch-pull shape; both the row scan and the `count(*)` shortcut build on it.
/// `table` is baked as a stage-0 constant, so `scan_next`'s multi-stream dispatch
/// costs nothing.
fn for_each_batch<I: InputsSource>(
    ctx: &mut Ctx,
    inputs: I,
    table: u64,
    schema: SchemaRef,
    cx: &CodegenCtx,
    body: impl FnOnce(&mut Ctx, ScanBatch, Var<u64>) + 'static,
) {
    let ncols = schema.fields().len() as u64;
    let key_dt = schema.field(0).data_type().clone();
    let scan_next = cx.rt.scan_next;
    let poison = cx.poison.get();

    ctx.while_loop(Const::<bool>::new(true), move |ctx| {
        // Stop pulling batches once a fallible callback has poisoned the run — the
        // driver will surface the recorded error after we return.
        if let Some(p) = poison {
            ctx.if_then(p, |ctx| ctx.break_loop());
        }
        // Pull the next batch; a null descriptor pointer means the stream is done.
        // SAFETY: the compiled entry point receives exclusive access to live
        // `Inputs`; scan callbacks are sequenced by this outer loop.
        let descs = ctx
            .bind(unsafe { call_extern2_unchecked(scan_next, inputs, Const::<u64>::new(table)) });
        ctx.if_then(ptr_is_null(descs), |ctx| ctx.break_loop());
        // Rebuild a lifetime-free raw batch descriptor from (ptr, column count).
        // SAFETY: `Inputs` retains the current descriptor array until the next
        // callback, and Phase 1 schema validation proves it has `ncols` entries.
        let batch = ctx.bind(unsafe {
            slice_from_raw_parts::<FfiArray, _, _>(descs, Const::<u64>::new(ncols))
        });
        let len = gen_len(ctx, batch, &key_dt);
        body(ctx, batch, len);
    });
}

/// Scan a table row by row: the outer batch loop ([`for_each_batch`]) wrapping an
/// inner row loop that reads every column and `yld`s each row. Cross-batch
/// accumulators live in registers/host state above this, so folding across the
/// whole stream is free (see `docs/table_scan.md` §6).
fn gen_scan<I: InputsSource>(
    ctx: &mut Ctx,
    inputs: I,
    table: u64,
    schema: SchemaRef,
    cx: &CodegenCtx,
    yld: Yld,
) {
    let fields = schema.fields().clone();
    let i = ctx.var(0u64);
    for_each_batch(ctx, inputs, table, schema, cx, move |ctx, batch, len| {
        ctx.store(i, 0u64);
        ctx.while_loop(lt(i, len), move |ctx| {
            let row: Row = fields
                .iter()
                .enumerate()
                .map(|(col, f)| gen_read(ctx, batch, col, f, i))
                .collect();
            yld(ctx, row);
            ctx.store(i, add(i, 1u64));
        });
    });
}

/// Emit the error-stop check for a fallible pointer-returning callback (the
/// `group_upsert*` family): if `ptr` is the null error sentinel (see
/// [`crate::status`]), set the kernel poison flag and break the current loop, so the
/// caller never dereferences null and the batch loop unwinds to the driver — which
/// then surfaces the recorded `DataFusionError`. No-op for helper kernels with no
/// poison flag.
pub(crate) fn stop_if_null(ctx: &mut Ctx, ptr: Var<SMutPtr<u8>>, cx: &CodegenCtx) {
    if let Some(p) = cx.poison.get() {
        ctx.if_then(ptr_is_null(ptr), move |ctx| {
            ctx.store(p, true);
            ctx.break_loop();
        });
    }
}

pub(crate) fn gen_len<B: BatchSource>(ctx: &mut Ctx, batch: B, dt: &DataType) -> Var<u64> {
    // A `Utf8View` column's `values` holds one 16-byte view per row, so its
    // element count (read via any type) is the row count.
    if matches!(dt, DataType::Utf8View) {
        // SAFETY: the batch was schema-validated; only the descriptor length is
        // read, so the placeholder physical marker is never dereferenced.
        return ctx.bind(unsafe { batch.primitive::<u64>(0) }.len());
    }
    dispatch_prim!(dt, M => {
        // SAFETY: dispatch selects the staged type matching the validated field.
        ctx.bind(unsafe { batch.primitive::<M>(0) }.len())
    })
}

/// Read column `col` at row `i`, attaching validity iff the field is nullable.
pub(crate) fn gen_read<B: BatchSource>(
    ctx: &mut Ctx,
    batch: B,
    col: usize,
    field: &Field,
    i: Var<u64>,
) -> ColVal {
    if matches!(field.data_type(), DataType::Utf8View) {
        return gen_read_str(ctx, batch, col, field, i);
    }
    let nullable = field.is_nullable();
    dispatch_prim!(field.data_type(), M => {
        // SAFETY: schema validation proves `col` exists with physical type `M`.
        let view = unsafe { batch.primitive::<M>(col) };
        // SAFETY: all callers invoke `gen_read` inside the batch row loop.
        let value = ctx.bind(unsafe { view.value_unchecked(i) });
        M::wrap(value, read_nullness(ctx, view, nullable, i))
    })
}

/// Read a `Utf8View` string at row `i`. The views buffer is reinterpreted as
/// `u64` pairs (view `i` = elements `2i`/`2i+1`); the octet length is the low 32
/// bits of the first half — no data-buffer access needed.
fn gen_read_str<B: BatchSource>(
    ctx: &mut Ctx,
    batch: B,
    col: usize,
    field: &Field,
    i: Var<u64>,
) -> ColVal {
    // SAFETY: a validated Utf8View descriptor stores 16-byte views; this code
    // intentionally reads each view as its two u64 halves.
    let views = unsafe { batch.primitive::<u64>(col) };
    let base = ctx.bind(mul(i, 2u64));
    // SAFETY: `i` is in the row loop, so halves `2*i` and `2*i+1` exist.
    let lo = ctx.bind(unsafe { views.value_unchecked(base) });
    let hi = ctx.bind(unsafe { views.value_unchecked(add(base, 1u64)) });
    // SAFETY: `col` is schema-validated and the current batch retains this
    // descriptor and its originating `StringViewArray` for the row loop.
    let descriptor = unsafe { slice_get_ptr_unchecked(batch, col as u64) };
    let array = ctx.bind(ptr_cast::<Opaque<StringViewArray>, u8, _>(unsafe {
        load_field_unchecked(descriptor, FfiArrayType::array())
    }));
    let null = read_nullness(ctx, views, field.is_nullable(), i);
    ColVal::Str(
        StrVal::Column {
            lo,
            hi,
            array,
            row: i,
        },
        null,
    )
}

fn read_nullness<P, M>(
    ctx: &mut Ctx,
    view: PrimitiveArrayView<P, M>,
    nullable: bool,
    i: Var<u64>,
) -> Nullness
where
    P: Staged<Out = SPtr<FfiArray>> + Clone + 'static,
    M: StagedType + 'static,
{
    if nullable {
        // SAFETY: `i` is bounded by this batch's row loop.
        Nullness::Nullable(ctx.bind(unsafe { view.validity().is_valid(i) }))
    } else {
        Nullness::NonNull
    }
}

// =============================================================================
// Output materialization
// =============================================================================

/// Append column `c`'s value to its baked output buffer (see [`crate::output`]).
/// Fixed-width columns `SVec::push` the value inline (and, if nullable, push a
/// `bool` validity flag to a parallel `SVec` so the two stay length-aligned);
/// string columns append through the builder. One append per emitted row, in
/// order, so every column ends at the same length.
pub(crate) fn write_col(ctx: &mut Ctx, c: usize, field: &Field, cv: ColVal, cx: &CodegenCtx) {
    match &cx.out.cols[c] {
        OutColHandle::Str { builder } => write_str_col(ctx, *builder, cv, cx),
        OutColHandle::Fixed { values, validity } => {
            dispatch_prim!(field.data_type(), M => {
                // SAFETY: the output owns this control block with element type
                // `M` and retains it for the compiled kernel's lifetime.
                let vals = unsafe { SVec::<M>::from_raw_unchecked(*values, cx.rt.svec_grow) };
                let v = M::coerce(ctx, cv);
                vals.push(ctx, v);
                // A nullable column pushes a validity flag for *every* row (the
                // value is pushed unconditionally, so no branch is needed).
                if let Some(valid_ctrl) = validity {
                    let flag = match cv.nullness() {
                        Nullness::NonNull => ctx.bind(Const::<bool>::new(true)),
                        Nullness::Nullable(valid) => valid,
                    };
                    // SAFETY: nullable output owns a bool control block and
                    // retains it for the compiled kernel's lifetime.
                    unsafe {
                        SVec::<bool>::from_raw_unchecked(*valid_ctrl, cx.rt.svec_grow)
                    }
                    .push(ctx, flag);
                }
            })
        }
    }
}

/// Append a `Utf8View` value to the baked output `StringViewBuilder`. Any string
/// container works: the value is `resolve`d to bytes and appended; a null row
/// appends a null.
fn write_str_col(ctx: &mut Ctx, builder: *mut StringViewBuilder, cv: ColVal, cx: &CodegenCtx) {
    let sv = match cv {
        ColVal::Str(sv, _) => sv,
        other => panic!(
            "string output column got a non-string value: {}",
            tag(other)
        ),
    };
    let append = cx.rt.strview_append_bytes;
    let append_null = cx.rt.strview_append_null;
    // `resolve` is safe even on a null row (its `str_ptr` reads an empty view), so
    // resolve unconditionally and branch only on which extern to call.
    let (ptr, len) = resolve(ctx, sv, cx.rt.str_ptr);
    let emit_append = move |ctx: &mut Ctx| {
        // SAFETY: the resolved string owner remains live for this append call,
        // `len` is its byte length, and `OutCols` exclusively owns the builder.
        let bytes = unsafe { slice_from_raw_parts::<u8, _, _>(ptr, len) };
        ctx.emit(unsafe { call_extern2_unchecked(append, string_builder_mut(builder), bytes) });
    };
    match cv.nullness() {
        Nullness::NonNull => emit_append(ctx),
        Nullness::Nullable(valid) => {
            ctx.if_then(valid, emit_append);
            ctx.if_then(not(valid), move |ctx| {
                // SAFETY: `OutCols` retains and exclusively owns the builder
                // throughout the generated kernel call.
                ctx.emit(unsafe {
                    call_extern1_unchecked(append_null, string_builder_mut(builder))
                });
            });
        }
    }
}
