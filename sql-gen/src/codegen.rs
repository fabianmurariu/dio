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

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{StringViewArray, StringViewBuilder};
use arrow::datatypes::{DataType, Field, SchemaRef};
use arrow_lms::ffi::FfiArrayType;
use arrow_lms::{ArrayBatchOps, FfiArray, MutBatchOps, PrimitiveArrayView};
use datafusion_common::ScalarValue;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{BinaryExpr, Expr, Operator as DfOp};
use rust_lms::prelude::*;

use crate::group::GroupTable;
use crate::plan::Operator;
use crate::runtime::{Runtime, StrPtrExtern};
use crate::value::{ColVal, Nullness, Row, StrVal};

/// Codegen context threaded through the walk: the runtime extern handles plus the
/// build-time-interned string literals (value → stable `BytesPool` pointer). The
/// `Arc` is because the push-model `yld` closures are `'static` and can't borrow.
#[derive(Clone)]
pub struct Cx {
    pub rt: Runtime,
    pub lits: Arc<HashMap<String, u64>>,
}

/// Collect the string-literal values in `op`'s expressions, so `exec_jit` can
/// intern each into the `BytesPool` before codegen (see [`Cx::lits`]).
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

/// A staged expression yielding the read-only input batch (`&[FfiArray]`).
pub trait BatchSource: Staged<Out = SRef<'static, Slice<FfiArray>>> + Copy + 'static {}
impl<T> BatchSource for T where T: Staged<Out = SRef<'static, Slice<FfiArray>>> + Copy + 'static {}

/// A staged expression yielding the `&mut` output batch (`&mut [FfiArray]`).
pub trait OutSink: Staged<Out = SRefMut<'static, Slice<FfiArray>>> + Copy + 'static {}
impl<T> OutSink for T where T: Staged<Out = SRefMut<'static, Slice<FfiArray>>> + Copy + 'static {}

/// Downstream continuation, invoked once per emitted row at code-generation
/// time; the [`Row`] rides by value (cheap `Copy` handles).
type Yld = Box<dyn FnOnce(&mut Ctx, Row) + 'static>;

/// Emit a kernel that writes `plan`'s emitted rows into `out` at a running
/// cursor and returns the row count. The single entry point — a scalar
/// `Aggregate` is just a push operator that emits one row (see [`gen_op`]).
pub fn gen_collect<B: BatchSource, O: OutSink>(
    ctx: &mut Ctx,
    batch: B,
    out: O,
    plan: &Operator,
    out_schema: &SchemaRef,
    cx: &Cx,
) -> Var<u64> {
    let n = ctx.var(0u64);
    let fields = out_schema.fields().clone();
    let cx_c = cx.clone();
    gen_op(
        plan,
        ctx,
        batch,
        cx,
        Box::new(move |ctx, row| {
            for (c, (cv, field)) in row.iter().zip(fields.iter()).enumerate() {
                write_col(ctx, out, c, field, n, *cv, &cx_c);
            }
            ctx.store(n, add(n, 1u64));
        }),
    );
    n
}

fn gen_op<B: BatchSource>(op: &Operator, ctx: &mut Ctx, batch: B, cx: &Cx, yld: Yld) {
    match op {
        Operator::Scan { schema } => gen_scan(ctx, batch, schema.clone(), yld),

        Operator::Filter { predicate, input } => {
            let predicate = predicate.clone();
            let schema = input.output_schema();
            let cx_c = cx.clone();
            gen_op(
                input,
                ctx,
                batch,
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
                batch,
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

        // Scalar aggregate: fold every input row into accumulators, then emit
        // exactly one result row downstream (after the loop). GROUP BY is handled
        // by a dedicated kernel in `run.rs` (it needs the output arrays + the
        // hash table), not through this push path.
        Operator::Aggregate {
            group_exprs,
            aggs,
            input,
            schema,
        } => {
            assert!(
                group_exprs.is_empty(),
                "grouped aggregate should be routed to gen_grouped, not gen_op"
            );
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
                batch,
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
    }
}

/// GROUP BY kernel: fold each input row into a group slot, keeping the hot loop in
/// the JIT. Per row: compute the group key, `intern` it into the Rust-hosted
/// `GroupTable` for a dense group index, write the key into output column 0, and
/// fold each aggregate into its output column at that index. Returns the number of
/// groups (the output row count). Output layout is `[group key | aggregates…]` and
/// the accumulator columns are sized to the row count (groups ≤ rows) so indices
/// never overflow. Single non-null `Int64` key; `count(*)`/`count(col)`/`sum` for
/// now. `min`/`max` need identity-initialized accumulators — added next.
#[allow(clippy::too_many_arguments)]
pub fn gen_grouped<B: BatchSource, O: OutSink>(
    ctx: &mut Ctx,
    batch: B,
    table: Var<SRefMut<'static, Opaque<GroupTable>>>,
    out: O,
    group_exprs: &[Expr],
    aggs: &[Expr],
    input: &Operator,
    out_schema: &SchemaRef,
    cx: &Cx,
) -> Var<u64> {
    assert_eq!(
        group_exprs.len(),
        1,
        "only a single GROUP BY key is supported"
    );
    let group_expr = group_exprs[0].clone();
    let parsed: Vec<GroupedAgg> = aggs.iter().map(GroupedAgg::parse).collect();
    // Whether each aggregate's output column carries a validity bitmap (so the
    // fold marks a group valid on its first non-null input).
    let nullable: Vec<bool> = (0..parsed.len())
        .map(|j| out_schema.field(1 + j).is_nullable())
        .collect();
    // `avg` needs a per-group count alongside its running sum; give each one a
    // hidden `i64` count column appended after the visible ones (see `exec_grouped`).
    let mut avg_count: Vec<Option<usize>> = vec![None; parsed.len()];
    let mut next_hidden = 1 + parsed.len();
    for (j, a) in parsed.iter().enumerate() {
        if a.kind == AggKind::Avg {
            avg_count[j] = Some(next_hidden);
            next_hidden += 1;
        }
    }
    // (avg sum column, count column) pairs, for the finalize divide.
    let avg_finalize: Vec<(usize, usize)> = (0..parsed.len())
        .filter_map(|j| avg_count[j].map(|cc| (1 + j, cc)))
        .collect();

    let input_schema = input.output_schema();
    let cx_c = cx.clone();

    gen_op(
        input,
        ctx,
        batch,
        cx,
        Box::new(move |ctx, row| {
            // Key → group index (the extern owns the Rust hash map). `table` is
            // already a typed `&mut GroupTable` kernel param, so pass it directly.
            let key_cv = gen_expr(ctx, &group_expr, &input_schema, &row, &cx_c);
            let key = to_i64(ctx, key_cv);
            let gidx = ctx.bind(call_extern2(cx_c.rt.group_intern, table, key));
            // Group-key column (col 0), idempotent per group.
            out.column_mut::<i64>(0).set(ctx, gidx, key);
            // Fold each aggregate into its column at `gidx`.
            for (j, agg) in parsed.iter().enumerate() {
                agg.fold(
                    ctx,
                    out,
                    1 + j,
                    nullable[j],
                    avg_count[j],
                    gidx,
                    &row,
                    &input_schema,
                    &cx_c,
                );
            }
        }),
    );

    let num_groups = ctx.bind(call_extern1(cx.rt.group_len, table));

    // Finalize `avg`: divide each group's running sum by its count (valid iff the
    // count is > 0, i.e. the group had a non-null input; otherwise it stays null).
    if !avg_finalize.is_empty() {
        let g = ctx.var(0u64);
        ctx.while_loop(lt(g, num_groups), move |ctx| {
            for &(sum_col, count_col) in &avg_finalize {
                let count = out.column_mut::<i64>(count_col).get(ctx, g);
                let cond = ctx.bind(gt(count, 0i64));
                ctx.if_then(cond, move |ctx| {
                    let sview = out.column_mut::<f64>(sum_col);
                    let sum = sview.get(ctx, g);
                    let cf = ctx.bind(int_to_float::<f64, i64, _>(count));
                    let avg = ctx.bind(div(sum, cf));
                    sview.set(ctx, g, avg);
                    sview.validity_mut().set_valid(ctx, g);
                });
            }
            ctx.store(g, add(g, 1u64));
        });
    }

    num_groups
}

/// A parsed grouped aggregate (kind + optional argument), folded per row into an
/// output accumulator column indexed by group.
struct GroupedAgg {
    kind: AggKind,
    arg: Option<Expr>,
}

impl GroupedAgg {
    #[allow(deprecated)]
    fn parse(expr: &Expr) -> GroupedAgg {
        let (name, arg) = match expr {
            Expr::AggregateFunction(af) => (
                af.func.name().to_ascii_lowercase(),
                af.params
                    .args
                    .first()
                    .cloned()
                    .filter(|e| !matches!(e, Expr::Wildcard { .. })),
            ),
            other => panic!("expected aggregate function, got {other:?}"),
        };
        let kind = match name.as_str() {
            "count" => AggKind::Count,
            "sum" => AggKind::Sum,
            "min" => AggKind::Min,
            "max" => AggKind::Max,
            "avg" => AggKind::Avg,
            other => panic!("grouped aggregate not yet supported: {other}"),
        };
        GroupedAgg { kind, arg }
    }

    /// The aggregate's input value (`None` for `count(*)`) and its per-row validity
    /// bit (`true` when there's no argument or the argument is non-null).
    fn arg_value(
        &self,
        ctx: &mut Ctx,
        row: &Row,
        schema: &SchemaRef,
        cx: &Cx,
    ) -> (Option<ColVal>, Var<bool>) {
        match &self.arg {
            None => (None, ctx.bind(Const::<bool>::new(true))),
            Some(e) => {
                let cv = gen_expr(ctx, e, schema, row, cx);
                let valid = match cv.nullness() {
                    Nullness::NonNull => ctx.bind(Const::<bool>::new(true)),
                    Nullness::Nullable(b) => b,
                };
                (Some(cv), valid)
            }
        }
    }

    /// Fold this row's contribution into output column `col` at group `gidx`. Only
    /// non-null inputs contribute; for nullable `sum`/`min`/`max` the group is
    /// marked valid on its first non-null input (columns start all-null), so an
    /// all-null group stays null. `count` counts non-null rows (or all rows for
    /// `count(*)`) and is never null. `avg` folds an `f64` running sum into `col`
    /// and a count into hidden `avg_count` col; the divide happens in the finalize
    /// pass (see [`gen_grouped`]).
    #[allow(clippy::too_many_arguments)]
    fn fold<O: OutSink>(
        &self,
        ctx: &mut Ctx,
        out: O,
        col: usize,
        nullable: bool,
        avg_count: Option<usize>,
        gidx: Var<u64>,
        row: &Row,
        schema: &SchemaRef,
        cx: &Cx,
    ) {
        let (val_cv, valid) = self.arg_value(ctx, row, schema, cx);
        match self.kind {
            AggKind::Count => {
                let view = out.column_mut::<i64>(col);
                ctx.if_then(valid, move |ctx| {
                    let cur = view.get(ctx, gidx);
                    let next = ctx.bind(add(cur, 1i64));
                    view.set(ctx, gidx, next);
                });
            }
            AggKind::Sum => {
                let view = out.column_mut::<i64>(col);
                let v = to_i64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    if nullable {
                        view.validity_mut().set_valid(ctx, gidx);
                    }
                    let cur = view.get(ctx, gidx);
                    let next = ctx.bind(add(cur, v));
                    view.set(ctx, gidx, next);
                });
            }
            // min/max fold from an identity-initialized column (i64::MAX / MIN).
            AggKind::Min => {
                let view = out.column_mut::<i64>(col);
                let v = to_i64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    if nullable {
                        view.validity_mut().set_valid(ctx, gidx);
                    }
                    let cur = view.get(ctx, gidx);
                    let next = ctx.bind(min(cur, v));
                    view.set(ctx, gidx, next);
                });
            }
            AggKind::Max => {
                let view = out.column_mut::<i64>(col);
                let v = to_i64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    if nullable {
                        view.validity_mut().set_valid(ctx, gidx);
                    }
                    let cur = view.get(ctx, gidx);
                    let next = ctx.bind(max(cur, v));
                    view.set(ctx, gidx, next);
                });
            }
            // avg: fold (f64 sum, i64 count); the finalize pass divides.
            AggKind::Avg => {
                let cc = avg_count.expect("avg needs a hidden count column");
                let v = to_f64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    let sview = out.column_mut::<f64>(col);
                    let cur = sview.get(ctx, gidx);
                    let next = ctx.bind(add(cur, v));
                    sview.set(ctx, gidx, next);

                    let cview = out.column_mut::<i64>(cc);
                    let ccur = cview.get(ctx, gidx);
                    let cnext = ctx.bind(add(ccur, 1i64));
                    cview.set(ctx, gidx, cnext);
                });
            }
        }
    }
}

/// Identity fill for each aggregate's accumulator column (aligned to output
/// columns `1..`): `0` for `count`/`sum`/`avg`, `i64::MAX`/`i64::MIN` for
/// `min`/`max`. The host pre-fills these before the fold (see [`gen_grouped`]).
pub fn agg_identities(aggs: &[Expr]) -> Vec<i64> {
    aggs.iter()
        .map(|e| match GroupedAgg::parse(e).kind {
            AggKind::Min => i64::MAX,
            AggKind::Max => i64::MIN,
            _ => 0,
        })
        .collect()
}

/// Number of hidden scratch columns the grouped kernel needs appended after the
/// visible output columns — one `i64` per-group count per `avg` (its running sum
/// lives in the visible column; the finalize pass divides).
pub fn grouped_scratch_cols(aggs: &[Expr]) -> usize {
    aggs.iter()
        .filter(|e| GroupedAgg::parse(e).kind == AggKind::Avg)
        .count()
}

fn gen_scan<B: BatchSource>(ctx: &mut Ctx, batch: B, schema: SchemaRef, yld: Yld) {
    let len = gen_len(ctx, batch, schema.field(0).data_type());
    let i = ctx.var(0u64);
    let fields = schema.fields().clone();

    ctx.while_loop(lt(i, len), move |ctx| {
        let row: Row = fields
            .iter()
            .enumerate()
            .map(|(col, f)| gen_read(ctx, batch, col, f, i))
            .collect();
        yld(ctx, row);
        ctx.store(i, add(i, 1u64));
    });
}

fn gen_len<B: BatchSource>(ctx: &mut Ctx, batch: B, dt: &DataType) -> Var<u64> {
    // A `Utf8View` column's `values` holds one 16-byte view per row, so its
    // element count (read via any type) is the row count.
    if matches!(dt, DataType::Utf8View) {
        return ctx.bind(batch.primitive::<u64>(0).len());
    }
    dispatch_prim!(dt, M => ctx.bind(batch.primitive::<M>(0).len()))
}

/// Read column `col` at row `i`, attaching validity iff the field is nullable.
fn gen_read<B: BatchSource>(
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
        let view = batch.primitive::<M>(col);
        let value = ctx.bind(view.value_unchecked(i));
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
    let views = batch.primitive::<u64>(col);
    let base = ctx.bind(mul(i, 2u64));
    let lo = ctx.bind(views.value_unchecked(base));
    let hi = ctx.bind(views.value_unchecked(add(base, 1u64)));
    // The opaque pointer back to the arrow array, for the extern fallback.
    let array = ctx.bind(load_field(
        batch.get_ref_unchecked(col as u64),
        FfiArrayType::array(),
    ));
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
    P: Staged<Out = SRef<'static, FfiArray>> + Clone + 'static,
    M: StagedType + 'static,
{
    if nullable {
        Nullness::Nullable(ctx.bind(view.validity().is_valid(i)))
    } else {
        Nullness::NonNull
    }
}

// =============================================================================
// Output materialization
// =============================================================================

fn write_col<O: OutSink>(
    ctx: &mut Ctx,
    out: O,
    c: usize,
    field: &Field,
    n: Var<u64>,
    cv: ColVal,
    cx: &Cx,
) {
    if matches!(field.data_type(), DataType::Utf8View) {
        return write_str_col(ctx, out, c, cv, cx);
    }
    dispatch_prim!(field.data_type(), M => {
        let view = out.column_mut::<M>(c);
        let v = M::coerce(ctx, cv);
        view.set(ctx, n, v);
        write_null(ctx, view, field, n, cv);
    })
}

/// Append a `Utf8View` value to output column `c`'s builder (whose pointer rides
/// in the output descriptor's opaque `array` field). Any string container works:
/// the value is `resolve`d to bytes and appended. A null row appends a null.
/// Appends happen once per emitted row, in order, so the builder ends
/// length-aligned with the fixed-width columns.
fn write_str_col<O: OutSink>(ctx: &mut Ctx, out: O, c: usize, cv: ColVal, cx: &Cx) {
    let sv = match cv {
        ColVal::Str(sv, _) => sv,
        other => panic!(
            "string output column got a non-string value: {}",
            tag(other)
        ),
    };
    let builder = ctx.bind(load_field(
        out.get_mut_unchecked(c as u64),
        FfiArrayType::array(),
    ));
    let append = cx.rt.strview_append_bytes;
    let append_null = cx.rt.strview_append_null;
    // `resolve` is safe even on a null row (its `str_ptr` reads an empty view), so
    // resolve unconditionally and branch only on which extern to call.
    let (ptr, len) = resolve(ctx, sv, cx.rt.str_ptr);
    let emit_append = move |ctx: &mut Ctx| {
        ctx.emit(call_extern2(
            append,
            opaque_ref_mut::<StringViewBuilder, _>(builder),
            slice_from_raw_parts::<u8, _, _>(ptr, len),
        ));
    };
    match cv.nullness() {
        Nullness::NonNull => emit_append(ctx),
        Nullness::Nullable(valid) => {
            ctx.if_then(valid, emit_append);
            ctx.if_then(not(valid), move |ctx| {
                ctx.emit(call_extern1(
                    append_null,
                    opaque_ref_mut::<StringViewBuilder, _>(builder),
                ));
            });
        }
    }
}

fn write_null<P, M>(
    ctx: &mut Ctx,
    view: PrimitiveArrayView<P, M>,
    field: &Field,
    n: Var<u64>,
    cv: ColVal,
) where
    P: Staged<Out = SRefMut<'static, FfiArray>> + Clone + 'static,
    M: StagedType + 'static,
{
    // Only nullable outputs need validity writes, and only nullable values can
    // be null (the bitmap starts all-valid, so non-null rows need no write).
    if let (true, Nullness::Nullable(valid)) = (field.is_nullable(), cv.nullness()) {
        let validity = view.validity_mut();
        ctx.if_then(not(valid), move |ctx| validity.set_null(ctx, n));
    }
}

// =============================================================================
// Expression evaluation
// =============================================================================

fn gen_predicate(ctx: &mut Ctx, e: &Expr, schema: &SchemaRef, row: &Row, cx: &Cx) -> Var<bool> {
    let cv = gen_expr(ctx, e, schema, row, cx);
    let cond = as_bool(cv);
    // SQL: a NULL predicate does not pass the filter -> keep iff (valid && cond).
    match cv.nullness() {
        Nullness::NonNull => cond,
        Nullness::Nullable(valid) => ctx.bind(select(valid, cond, Const::<bool>::new(false))),
    }
}

fn gen_expr(ctx: &mut Ctx, e: &Expr, schema: &SchemaRef, row: &Row, cx: &Cx) -> ColVal {
    match e {
        Expr::Column(c) => {
            let idx = schema
                .index_of(&c.name)
                .unwrap_or_else(|_| panic!("unknown column: {}", c.name));
            row[idx]
        }
        Expr::Literal(sv, _) => gen_literal(ctx, sv, cx),
        Expr::BinaryExpr(be) => gen_binary(ctx, be, schema, row, cx),
        Expr::ScalarFunction(f) => gen_scalar_fn(ctx, f, schema, row, cx),
        other => panic!("unsupported expression: {other:?}"),
    }
}

fn gen_scalar_fn(
    ctx: &mut Ctx,
    f: &ScalarFunction,
    schema: &SchemaRef,
    row: &Row,
    cx: &Cx,
) -> ColVal {
    match f.func.name().to_ascii_lowercase().as_str() {
        // Octet length: the byte length, from the view (`lo & 0xFFFF_FFFF`) for a
        // column or directly from `len` for resolved bytes.
        "octet_length" => match gen_expr(ctx, &f.args[0], schema, row, cx) {
            ColVal::Str(sv, null) => {
                let len = match sv {
                    StrVal::Column { lo, .. } => ctx.bind(bitand::<u64, _, _>(lo, 0xFFFF_FFFFu64)),
                    StrVal::Bytes { len, .. } => len,
                };
                ColVal::I64(ctx.bind(int_cast::<i64, u64, _>(len)), null)
            }
            other => panic!("octet_length expects a string, got {}", tag(other)),
        },
        other => panic!("unsupported scalar function: {other}"),
    }
}

/// Resolve a string to its bytes `(ptr, len)` — the generic representation every
/// container produces. A `Column` reads its byte pointer via the `str_ptr` extern
/// (its length comes free from the view, `lo & 0xFFFF_FFFF`); `Bytes` (interned
/// literal or produced string) is already resolved. Takes the extern handle (not
/// `&Cx`) so it can be called from `'static` branch closures.
fn resolve(ctx: &mut Ctx, sv: StrVal, str_ptr: ExternRef<StrPtrExtern>) -> (Var<u64>, Var<u64>) {
    match sv {
        StrVal::Column { lo, array, row, .. } => {
            let ptr = ctx.bind(call_extern2(
                str_ptr,
                opaque_ref::<StringViewArray, _>(array),
                row,
            ));
            let len = ctx.bind(bitand::<u64, _, _>(lo, 0xFFFF_FFFFu64));
            (ptr, len)
        }
        StrVal::Bytes { ptr, len, .. } => (ptr, len),
    }
}

/// The 16-byte view halves of a string, as staged `u64`s — for the equality fast
/// path. A `Column` has them as `Var`s; a literal's are host constants baked in.
/// A produced `Bytes` (no precomputed view) has none → resolve to bytes instead.
fn view_of(ctx: &mut Ctx, sv: StrVal) -> Option<(Var<u64>, Var<u64>)> {
    match sv {
        StrVal::Column { lo, hi, .. } => Some((lo, hi)),
        StrVal::Bytes {
            view: Some((lo, hi)),
            ..
        } => Some((ctx.var(lo), ctx.var(hi))),
        StrVal::Bytes { view: None, .. } => None,
    }
}

/// String equality over `Utf8View`, using every bit of the 16-byte view before
/// touching bytes (the view layout: `lo` = `[len:u32][first 4 content bytes]`,
/// `hi` = the inline tail for `len ≤ 12`, else buffer index/offset):
///
/// 1. `lo` differs  → **not equal** (differing length or first-4 bytes) — no resolve.
/// 2. `len ≤ 12`    → `hi` settles it (both fully inline) — no resolve.
/// 3. `len > 12`    → prefixes match but the rest is array-relative → resolve both
///    sides and `bytes_eq`.
///
/// This is correct across containers: a column row and a literal share the same
/// `lo`/`hi` encoding, and step 3 never trusts the array-relative `hi`. Only when
/// an operand has no view (a produced string) do we resolve outright.
fn str_eq(ctx: &mut Ctx, l: StrVal, r: StrVal, cx: &Cx) -> Var<bool> {
    let str_ptr = cx.rt.str_ptr;
    let bytes_eq = cx.rt.bytes_eq;
    match (view_of(ctx, l), view_of(ctx, r)) {
        (Some((a_lo, a_hi)), Some((b_lo, b_hi))) => {
            let result = ctx.var(false);
            let lo_eq = ctx.bind(eq(a_lo, b_lo));
            // Only when `lo` matches (same length + first 4 content bytes) is there
            // anything more to check.
            ctx.if_then(lo_eq, move |ctx| {
                let len = ctx.bind(bitand::<u64, _, _>(a_lo, 0xFFFF_FFFFu64));
                let long = ctx.bind(gt(len, 12u64));
                // len <= 12: the views are the whole strings — `hi` is definitive.
                let hi_eq = ctx.bind(eq(a_hi, b_hi));
                ctx.if_then(not(long), move |ctx| ctx.store(result, hi_eq));
                // len > 12: prefixes match; compare the full bytes.
                ctx.if_then(long, move |ctx| {
                    let (ap, al) = resolve(ctx, l, str_ptr);
                    let (bp, bl) = resolve(ctx, r, str_ptr);
                    let eq = ctx.bind(call_extern2(
                        bytes_eq,
                        slice_from_raw_parts::<u8, _, _>(ap, al),
                        slice_from_raw_parts::<u8, _, _>(bp, bl),
                    ));
                    ctx.store(result, eq);
                });
            });
            result
        }
        // At least one operand is a produced string with no view: resolve both.
        _ => {
            let (ap, al) = resolve(ctx, l, str_ptr);
            let (bp, bl) = resolve(ctx, r, str_ptr);
            ctx.bind(call_extern2(
                bytes_eq,
                slice_from_raw_parts::<u8, _, _>(ap, al),
                slice_from_raw_parts::<u8, _, _>(bp, bl),
            ))
        }
    }
}

fn is_str(cv: ColVal) -> bool {
    matches!(cv, ColVal::Str(..))
}

/// Equality: string operands compare by view/bytes, numeric operands by value.
fn gen_eq(ctx: &mut Ctx, l: ColVal, r: ColVal, cx: &Cx) -> Var<bool> {
    match (l, r) {
        (ColVal::Str(a, _), ColVal::Str(b, _)) => str_eq(ctx, a, b, cx),
        _ if is_str(l) || is_str(r) => panic!("string compared with non-string"),
        _ => num_cmp(ctx, Cmp::Eq, l, r),
    }
}

/// The `Utf8View` view halves `(lo, hi)` for a literal of *any* length, matching
/// arrow's on-wire encoding so they compare bit-identically to a column row's
/// view. `lo` = `[len:u32][first ≤4 content bytes, zero-padded]` — valid for all
/// lengths (it's the length + prefix). `hi` = the inline tail (bytes 4..12) for
/// `len ≤ 12`; for longer strings `hi` is `0` (a column's `hi` there is a
/// buffer/offset, which `str_eq` never compares — it resolves instead).
fn literal_view_halves(s: &str) -> (u64, u64) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut lo = [0u8; 8];
    lo[0..4].copy_from_slice(&(len as u32).to_le_bytes());
    let head = &bytes[..len.min(4)];
    lo[4..4 + head.len()].copy_from_slice(head);
    let lo = u64::from_le_bytes(lo);
    let hi = if len <= 12 {
        let mut hi = [0u8; 8];
        let tail = &bytes[len.min(4)..len]; // bytes 4..len (or empty)
        hi[..tail.len()].copy_from_slice(tail);
        u64::from_le_bytes(hi)
    } else {
        0
    };
    (lo, hi)
}

fn gen_literal(ctx: &mut Ctx, sv: &ScalarValue, cx: &Cx) -> ColVal {
    match sv {
        ScalarValue::Int32(Some(v)) => ColVal::I32(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Int64(Some(v)) => ColVal::I64(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Float64(Some(v)) => ColVal::F64(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Boolean(Some(v)) => {
            ColVal::Bool(ctx.bind(Const::<bool>::new(*v)), Nullness::NonNull)
        }
        ScalarValue::Utf8(Some(s))
        | ScalarValue::Utf8View(Some(s))
        | ScalarValue::LargeUtf8(Some(s)) => gen_str_literal(ctx, s, cx),
        other => panic!("unsupported literal: {other:?}"),
    }
}

/// A string literal as resolved [`StrVal::Bytes`]: `ptr` is the literal's stable
/// address in the `BytesPool` (interned once at build time — see [`Cx::lits`]),
/// and `view` carries the view halves (for *any* length) so `str_eq`'s length +
/// prefix fast-reject and inline compare apply.
fn gen_str_literal(ctx: &mut Ctx, s: &str, cx: &Cx) -> ColVal {
    let ptr = *cx
        .lits
        .get(s)
        .unwrap_or_else(|| panic!("literal not interned: {s:?}"));
    ColVal::Str(
        StrVal::Bytes {
            ptr: ctx.var(ptr),
            len: ctx.var(s.len() as u64),
            view: Some(literal_view_halves(s)),
        },
        Nullness::NonNull,
    )
}

fn gen_binary(ctx: &mut Ctx, be: &BinaryExpr, schema: &SchemaRef, row: &Row, cx: &Cx) -> ColVal {
    let l = gen_expr(ctx, &be.left, schema, row, cx);
    let r = gen_expr(ctx, &be.right, schema, row, cx);
    let null = combine_null(ctx, l.nullness(), r.nullness());
    match be.op {
        DfOp::Eq => ColVal::Bool(gen_eq(ctx, l, r, cx), null),
        DfOp::NotEq => {
            let e = gen_eq(ctx, l, r, cx);
            ColVal::Bool(ctx.bind(not(e)), null)
        }
        DfOp::Lt => ColVal::Bool(num_cmp(ctx, Cmp::Lt, l, r), null),
        DfOp::Gt => ColVal::Bool(num_cmp(ctx, Cmp::Gt, l, r), null),
        DfOp::LtEq => {
            let e = num_cmp(ctx, Cmp::Gt, l, r);
            ColVal::Bool(ctx.bind(not(e)), null)
        }
        DfOp::GtEq => {
            let e = num_cmp(ctx, Cmp::Lt, l, r);
            ColVal::Bool(ctx.bind(not(e)), null)
        }
        DfOp::And => {
            let (a, b) = (as_bool(l), as_bool(r));
            ColVal::Bool(ctx.bind(select(a, b, Const::<bool>::new(false))), null)
        }
        DfOp::Or => {
            let (a, b) = (as_bool(l), as_bool(r));
            ColVal::Bool(ctx.bind(select(a, Const::<bool>::new(true), b)), null)
        }
        DfOp::Plus => arith(ctx, Arith::Add, l, r, null),
        DfOp::Minus => arith(ctx, Arith::Sub, l, r, null),
        DfOp::Multiply => arith(ctx, Arith::Mul, l, r, null),
        other => panic!("unsupported binary operator: {other:?}"),
    }
}

/// `NonNull` unless an operand is nullable; two nullable operands AND their bits.
fn combine_null(ctx: &mut Ctx, a: Nullness, b: Nullness) -> Nullness {
    match (a, b) {
        (Nullness::NonNull, Nullness::NonNull) => Nullness::NonNull,
        (Nullness::Nullable(v), Nullness::NonNull) | (Nullness::NonNull, Nullness::Nullable(v)) => {
            Nullness::Nullable(v)
        }
        (Nullness::Nullable(x), Nullness::Nullable(y)) => {
            Nullness::Nullable(ctx.bind(select(x, y, Const::<bool>::new(false))))
        }
    }
}

enum Cmp {
    Eq,
    Lt,
    Gt,
}

/// Compare two values -> bool. Floats compare as `f64`; ints widen to `i64`.
fn num_cmp(ctx: &mut Ctx, kind: Cmp, l: ColVal, r: ColVal) -> Var<bool> {
    if let (ColVal::F64(x, _), ColVal::F64(y, _)) = (l, r) {
        return match kind {
            Cmp::Eq => ctx.bind(eq(x, y)),
            Cmp::Lt => ctx.bind(lt(x, y)),
            Cmp::Gt => ctx.bind(gt(x, y)),
        };
    }
    let x = to_i64(ctx, l);
    let y = to_i64(ctx, r);
    match kind {
        Cmp::Eq => ctx.bind(eq(x, y)),
        Cmp::Lt => ctx.bind(lt(x, y)),
        Cmp::Gt => ctx.bind(gt(x, y)),
    }
}

enum Arith {
    Add,
    Sub,
    Mul,
}

/// Arithmetic -> numeric ColVal. Floats stay `f64`; ints widen to `i64`.
fn arith(ctx: &mut Ctx, kind: Arith, l: ColVal, r: ColVal, null: Nullness) -> ColVal {
    if let (ColVal::F64(x, _), ColVal::F64(y, _)) = (l, r) {
        let v = match kind {
            Arith::Add => ctx.bind(add(x, y)),
            Arith::Sub => ctx.bind(sub(x, y)),
            Arith::Mul => ctx.bind(mul(x, y)),
        };
        return ColVal::F64(v, null);
    }
    let x = to_i64(ctx, l);
    let y = to_i64(ctx, r);
    let v = match kind {
        Arith::Add => ctx.bind(add(x, y)),
        Arith::Sub => ctx.bind(sub(x, y)),
        Arith::Mul => ctx.bind(mul(x, y)),
    };
    ColVal::I64(v, null)
}

// =============================================================================
// Value extraction / coercion
// =============================================================================

fn to_i64(ctx: &mut Ctx, cv: ColVal) -> Var<i64> {
    match cv {
        ColVal::I64(v, _) => v,
        ColVal::I32(v, _) => ctx.bind(int_cast::<i64, i32, _>(v)),
        other => panic!("expected integer operand, got {}", tag(other)),
    }
}

fn coerce_i32(ctx: &mut Ctx, cv: ColVal) -> Var<i32> {
    match cv {
        ColVal::I32(v, _) => v,
        ColVal::I64(v, _) => ctx.bind(int_cast::<i32, i64, _>(v)),
        other => panic!("cannot coerce {} to i32", tag(other)),
    }
}

fn coerce_f64(_ctx: &mut Ctx, cv: ColVal) -> Var<f64> {
    match cv {
        ColVal::F64(v, _) => v,
        other => panic!("cannot coerce {} to f64", tag(other)),
    }
}

/// Convert any numeric column value to `f64` (int → float). Used by `avg`.
fn to_f64(ctx: &mut Ctx, cv: ColVal) -> Var<f64> {
    match cv {
        ColVal::F64(v, _) => v,
        ColVal::I64(v, _) => ctx.bind(int_to_float::<f64, i64, _>(v)),
        ColVal::I32(v, _) => ctx.bind(int_to_float::<f64, i32, _>(v)),
        other => panic!("cannot convert {} to f64", tag(other)),
    }
}

fn as_bool(cv: ColVal) -> Var<bool> {
    match cv {
        ColVal::Bool(v, _) => v,
        other => panic!("expected bool, got {}", tag(other)),
    }
}

fn tag(cv: ColVal) -> &'static str {
    match cv {
        ColVal::I32(..) => "i32",
        ColVal::I64(..) => "i64",
        ColVal::F64(..) => "f64",
        ColVal::Bool(..) => "bool",
        ColVal::Str { .. } => "str",
    }
}

// =============================================================================
// Scalar aggregation (no GROUP BY): accumulators folded by `gen_op`'s Aggregate
// arm, which emits one result row.
// =============================================================================

#[derive(Clone, Copy, PartialEq, Eq)]
enum AggKind {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

#[derive(Clone, Copy)]
enum AccVar {
    I64(Var<i64>),
    F64(Var<f64>),
    /// `avg` folds a running sum and count (both `f64`) and divides at the end.
    Avg {
        sum: Var<f64>,
        count: Var<f64>,
    },
}

/// A resolved aggregate plus its accumulator variable(s).
#[derive(Clone)]
struct Agg {
    kind: AggKind,
    /// The aggregated argument (`None` / a literal for `count(*)`).
    arg: Option<Expr>,
    acc: AccVar,
    /// Whether any non-null value has been folded — min/max/sum are NULL if not.
    seen: Var<bool>,
}

impl Agg {
    // `Expr::Wildcard` is deprecated in datafusion but still appears as
    // `count(*)`'s argument, so we match it deliberately.
    #[allow(deprecated)]
    fn init(ctx: &mut Ctx, expr: &Expr, out_field: &Field) -> Agg {
        let (name, arg) = match expr {
            Expr::AggregateFunction(af) => (
                af.func.name().to_ascii_lowercase(),
                // `count(*)`'s arg is a `Wildcard` — not evaluable; treat as
                // "no argument", i.e. count every row.
                af.params
                    .args
                    .first()
                    .cloned()
                    .filter(|e| !matches!(e, Expr::Wildcard { .. })),
            ),
            other => panic!("expected aggregate function, got {other:?}"),
        };
        let kind = match name.as_str() {
            "count" => AggKind::Count,
            "sum" => AggKind::Sum,
            "min" => AggKind::Min,
            "max" => AggKind::Max,
            "avg" => AggKind::Avg,
            other => panic!("unsupported aggregate: {other}"),
        };
        // Accumulator domain from the output type; min/max seed with the extreme.
        let is_float = matches!(out_field.data_type(), DataType::Float64);
        let acc = match (kind, is_float) {
            (AggKind::Count, _) => AccVar::I64(ctx.var(0i64)),
            (AggKind::Sum, false) => AccVar::I64(ctx.var(0i64)),
            (AggKind::Sum, true) => AccVar::F64(ctx.var(0.0f64)),
            (AggKind::Min, false) => AccVar::I64(ctx.var(i64::MAX)),
            (AggKind::Min, true) => AccVar::F64(ctx.var(f64::MAX)),
            (AggKind::Max, false) => AccVar::I64(ctx.var(i64::MIN)),
            (AggKind::Max, true) => AccVar::F64(ctx.var(f64::MIN)),
            (AggKind::Avg, _) => AccVar::Avg {
                sum: ctx.var(0.0f64),
                count: ctx.var(0.0f64),
            },
        };
        // count is never null; min/max/sum start "unseen".
        let seen = ctx.var(Const::<bool>::new(matches!(kind, AggKind::Count)));
        Agg {
            kind,
            arg,
            acc,
            seen,
        }
    }

    fn update(&self, ctx: &mut Ctx, row: &Row, schema: &SchemaRef, cx: &Cx) {
        let arg = self.arg.as_ref().map(|e| gen_expr(ctx, e, schema, row, cx));
        // The row contributes iff its argument is non-null (always, for count(*)).
        let row_valid = match arg.map(|cv| cv.nullness()) {
            Some(Nullness::Nullable(b)) => b,
            _ => ctx.bind(Const::<bool>::new(true)),
        };

        match (self.kind, self.acc) {
            (AggKind::Count, AccVar::I64(acc)) => {
                let one = select(row_valid, Const::<i64>::new(1), Const::<i64>::new(0));
                ctx.store(acc, add(acc, one));
            }
            (AggKind::Sum, AccVar::I64(acc)) => {
                let v = to_i64(ctx, arg.unwrap());
                let contrib = select(row_valid, v, Const::<i64>::new(0));
                ctx.store(acc, add(acc, contrib));
                self.mark_seen(ctx, row_valid);
            }
            (AggKind::Sum, AccVar::F64(acc)) => {
                let v = coerce_f64(ctx, arg.unwrap());
                let contrib = select(row_valid, v, Const::<f64>::new(0.0));
                ctx.store(acc, add(acc, contrib));
                self.mark_seen(ctx, row_valid);
            }
            (AggKind::Min, AccVar::I64(acc)) => {
                let v = to_i64(ctx, arg.unwrap());
                ctx.store(acc, select(row_valid, min(acc, v), acc));
                self.mark_seen(ctx, row_valid);
            }
            (AggKind::Min, AccVar::F64(acc)) => {
                let v = coerce_f64(ctx, arg.unwrap());
                ctx.store(acc, select(row_valid, min(acc, v), acc));
                self.mark_seen(ctx, row_valid);
            }
            (AggKind::Max, AccVar::I64(acc)) => {
                let v = to_i64(ctx, arg.unwrap());
                ctx.store(acc, select(row_valid, max(acc, v), acc));
                self.mark_seen(ctx, row_valid);
            }
            (AggKind::Max, AccVar::F64(acc)) => {
                let v = coerce_f64(ctx, arg.unwrap());
                ctx.store(acc, select(row_valid, max(acc, v), acc));
                self.mark_seen(ctx, row_valid);
            }
            (AggKind::Avg, AccVar::Avg { sum, count }) => {
                let v = to_f64(ctx, arg.unwrap());
                ctx.store(sum, add(sum, select(row_valid, v, Const::<f64>::new(0.0))));
                let inc = select(row_valid, Const::<f64>::new(1.0), Const::<f64>::new(0.0));
                ctx.store(count, add(count, inc));
                self.mark_seen(ctx, row_valid);
            }
            _ => unreachable!("mismatched aggregate kind / accumulator"),
        }
    }

    fn mark_seen(&self, ctx: &mut Ctx, row_valid: Var<bool>) {
        ctx.store(
            self.seen,
            select(self.seen, Const::<bool>::new(true), row_valid),
        );
    }

    /// Finalize into the output row's `ColVal`: `count` is never null; the others
    /// are null when nothing was folded (`seen == false`). `avg` divides its
    /// running sum by its count here (once, for the single output row).
    fn finalize(&self, ctx: &mut Ctx) -> ColVal {
        let null = match self.kind {
            AggKind::Count => Nullness::NonNull,
            _ => Nullness::Nullable(self.seen),
        };
        match self.acc {
            AccVar::I64(v) => ColVal::I64(v, null),
            AccVar::F64(v) => ColVal::F64(v, null),
            // sum/count = NaN when count == 0, but `seen == false` there masks it
            // with a NULL, so the divide is safe.
            AccVar::Avg { sum, count } => ColVal::F64(ctx.bind(div(sum, count)), null),
        }
    }
}
