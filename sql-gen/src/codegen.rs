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
use std::rc::Rc;

use arrow::array::{StringViewArray, StringViewBuilder};
use arrow::datatypes::{DataType, Field, SchemaRef};
use arrow_lms::ffi::FfiArrayType;
use arrow_lms::{ArrayBatchOps, FfiArray, MutBatchOps, PrimitiveArrayView};
use datafusion_common::ScalarValue;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{BinaryExpr, Expr, Operator as DfOp};
use rust_lms::prelude::*;
use rust_lms_std::{FieldHandle, RecordLayout};

use crate::group::GroupState;
use crate::plan::Operator;
use crate::runtime::{Runtime, StrPtrExtern};
use crate::value::{ColVal, Nullness, Row, StrVal};

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
    cx: &CodegenCtx,
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

fn gen_op<B: BatchSource>(op: &Operator, ctx: &mut Ctx, batch: B, cx: &CodegenCtx, yld: Yld) {
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

        // GROUP BY: fold into the Rust-hosted group state, then emit one row per
        // group downstream — a normal push operator, so the projection/filter
        // above it run in the same kernel.
        Operator::Aggregate { group_exprs, .. } if !group_exprs.is_empty() => {
            gen_grouped(ctx, batch, op, cx, yld);
        }

        // Scalar aggregate: fold every input row into accumulators, then emit
        // exactly one result row downstream (after the loop).
        Operator::Aggregate {
            aggs,
            input,
            schema,
            ..
        } => {
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

/// GROUP BY as a push operator: fold each input row into the Rust-hosted group
/// state (baked buffers indexed by `gidx`), then emit one *manifested* Row per
/// group downstream. Keeps the hot fold loop in the JIT and lets the projection /
/// filter above it run in the same kernel (see [`CodegenCtx::group`]). `avg` divides at
/// emit; nullability rides in the emitted `ColVal` (so `write_col` handles nulls).
fn gen_grouped<B: BatchSource>(ctx: &mut Ctx, batch: B, op: &Operator, cx: &CodegenCtx, yld: Yld) {
    let Operator::Aggregate {
        group_exprs,
        aggs,
        input,
        schema: out_schema,
    } = op
    else {
        unreachable!("gen_grouped called on a non-Aggregate operator")
    };
    assert_eq!(
        group_exprs.len(),
        1,
        "only a single GROUP BY key is supported"
    );
    let handle = cx
        .group
        .clone()
        .expect("grouped aggregate without a baked GroupState");
    let agg_tys = agg_output_types(out_schema, group_exprs.len());
    let record = group_record(aggs, &agg_tys);
    let layout = record.layout;
    let key_field = record.key;
    let state = handle.state;
    // Resolve each aggregate to its record fields (byte offsets) + its argument.
    let resolved: Vec<ResolvedAgg> = record
        .aggs
        .iter()
        .zip(aggs)
        .map(|(a, e)| ResolvedAgg {
            kind: a.kind,
            acc: a.acc,
            value_off: a.value_off,
            count_off: a.count_off,
            arg: GroupedAgg::parse(e).arg,
        })
        .collect();

    let group_expr = group_exprs[0].clone();
    let input_schema = input.output_schema();

    // Fold: per input row, intern the key and fold each aggregate into the group's
    // packed record at `records[gidx]`.
    let cx_c = cx.clone();
    let resolved_f = resolved.clone();
    let group_expr_f = group_expr.clone();
    let schema_f = input_schema.clone();
    gen_op(
        input,
        ctx,
        batch,
        cx,
        Box::new(move |ctx, row| {
            let key_cv = gen_expr(ctx, &group_expr_f, &schema_f, &row, &cx_c);
            let key = to_i64(ctx, key_cv);
            // The table keys on `u64` bits (grouping is sign-agnostic); the cast is a
            // no-op reinterpret. The signed `key` still goes into the record for emit.
            let key_bits = ctx.bind(int_cast::<u64, i64, _>(key));
            // Find-or-insert the group AND materialise its record in one proxy call:
            // `group_upsert` grows the records buffer (host-side) when it mints a new
            // group, and returns that group's record pointer — so the fold never bakes
            // a records base that a grow could dangle. Valid until the next `upsert`.
            let rec = ctx.bind(call_extern2(
                cx_c.rt.group_upsert,
                const_opaque_mut::<GroupState>(state),
                key_bits,
            ));
            key_field.set(ctx, rec, key);
            for agg in &resolved_f {
                agg.fold(ctx, rec, &row, &schema_f, &cx_c);
            }
        }),
    );

    // Emit: one manifested Row per group, pushed downstream in the same kernel. The
    // records buffer is fully grown now, so its base is stable — fetch it once.
    let num_groups = ctx.bind(call_extern1(
        cx.rt.group_len,
        const_opaque::<GroupState>(state),
    ));
    let base = ctx.bind(call_extern1(
        cx.rt.group_records_base,
        const_opaque_mut::<GroupState>(state),
    ));
    let g = ctx.var(0u64);
    ctx.while_loop(lt(g, num_groups), move |ctx| {
        let rec = layout.record(ctx, base, g);
        let key = key_field.get(ctx, rec);
        let mut row: Row = Vec::with_capacity(1 + resolved.len());
        row.push(ColVal::I64(key, Nullness::NonNull));
        for agg in &resolved {
            row.push(agg.finalize(ctx, rec));
        }
        yld(ctx, row);
        ctx.store(g, add(g, 1u64));
    });
}

/// The physical cell type an aggregate accumulates in: an `i64` cell or an `f64`
/// cell (the buffers are `i64`-typed 8-byte slots either way; `F64` reinterprets the
/// same bytes). Chosen from the aggregate's *output* type — `sum`/`min`/`max` over a
/// `Float64` column accumulate in `f64`; `count` is always `i64`; `avg` always sums
/// in `f64`.
#[derive(Clone, Copy, PartialEq)]
enum AccTy {
    I64,
    F64,
}

/// The accumulator cell type for an aggregate, from its datafusion output `DataType`.
fn acc_ty(kind: AggKind, out: &DataType) -> AccTy {
    match kind {
        AggKind::Count => AccTy::I64,
        AggKind::Avg => AccTy::F64,
        AggKind::Sum | AggKind::Min | AggKind::Max => match out {
            DataType::Float64 | DataType::Float32 => AccTy::F64,
            _ => AccTy::I64,
        },
    }
}

/// The aggregates' output `DataType`s — the schema fields *after* the leading group
/// keys (so `min(f64)` reports `Float64`, `sum(i32)` reports `Int64`, etc.).
pub fn agg_output_types(schema: &SchemaRef, num_keys: usize) -> Vec<DataType> {
    schema.fields()[num_keys..]
        .iter()
        .map(|f| f.data_type().clone())
        .collect()
}

/// The record fields an aggregate uses, and the cell type it folds in. Offsets are
/// byte offsets into the group's packed record (see [`group_record`]).
struct AggFields {
    kind: AggKind,
    acc: AccTy,
    value_off: usize,
    /// Non-null input count (for nullability + `avg`'s divide); `None` for `count`.
    count_off: Option<usize>,
}

/// The packed record of a group-by's state: field 0 is the `i64` key, then a value
/// (+ count) field per aggregate. One [`RecordLayout`], derived deterministically
/// from the aggregates, is shared by the host allocator ([`group_template`]) and
/// codegen so they agree on field offsets — exactly like a `#[repr(C)]` struct, but
/// query-shaped.
struct GroupRecord {
    layout: RecordLayout,
    key: FieldHandle<i64>,
    aggs: Vec<AggFields>,
}

/// Build the packed-record layout from the aggregates and their output `DataType`s
/// (parallel to `aggs`). The output type picks the accumulator cell ([`acc_ty`]);
/// `sum`/`min`/`max` over `Float64` fold in an `f64` field, everything else `i64`.
fn group_record(aggs: &[Expr], agg_tys: &[DataType]) -> GroupRecord {
    let mut layout = RecordLayout::new();
    let key = layout.field::<i64>();
    let mut agg_fields = Vec::with_capacity(aggs.len());
    for (e, out) in aggs.iter().zip(agg_tys) {
        let kind = GroupedAgg::parse(e).kind;
        let acc = acc_ty(kind, out);
        let value_off = match acc {
            AccTy::I64 => layout.field::<i64>().offset(),
            AccTy::F64 => layout.field::<f64>().offset(),
        };
        // sum/min/max/avg track a non-null count (seen / divisor); count does not.
        let count_off = matches!(
            kind,
            AggKind::Sum | AggKind::Min | AggKind::Max | AggKind::Avg
        )
        .then(|| layout.field::<i64>().offset());
        agg_fields.push(AggFields {
            kind,
            acc,
            value_off,
            count_off,
        });
    }
    GroupRecord {
        layout,
        key,
        aggs: agg_fields,
    }
}

/// A value field's identity fill, as raw `i64` bits (an `f64` field reuses the same
/// 8 bytes): `0` for count/sum/avg, `i64::MAX`/`MIN` for integer min/max, `±∞` bits
/// for float min/max.
fn value_init_bits(kind: AggKind, acc: AccTy) -> i64 {
    match (kind, acc) {
        (AggKind::Min, AccTy::I64) => i64::MAX,
        (AggKind::Min, AccTy::F64) => f64::INFINITY.to_bits() as i64,
        (AggKind::Max, AccTy::I64) => i64::MIN,
        (AggKind::Max, AccTy::F64) => f64::NEG_INFINITY.to_bits() as i64,
        _ => 0,
    }
}

/// The identity record: `stride/8` `u64` words a group starts at (key `0`, each
/// value field at its identity, counts `0`). The host fills every group slot with
/// this before the fold (see `group::GroupState::new`). All fields are 8-byte
/// `i64`/`f64` cells, so every offset is a whole word.
pub fn group_template(aggs: &[Expr], agg_tys: &[DataType]) -> Vec<u64> {
    let record = group_record(aggs, agg_tys);
    let words = record.layout.stride() / 8;
    let mut template = vec![0u64; words];
    for a in &record.aggs {
        template[a.value_off / 8] = value_init_bits(a.kind, a.acc) as u64;
    }
    template
}

// --- record-field accessors: reconstruct a typed `FieldHandle` from a stored byte
// offset (the query planner keeps offsets + `AccTy`, re-types the leaf here). ---

fn field_i64(offset: usize) -> FieldHandle<i64> {
    FieldHandle::from_offset(offset)
}

fn field_f64(offset: usize) -> FieldHandle<f64> {
    FieldHandle::from_offset(offset)
}

/// A parsed grouped aggregate (kind + optional argument).
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
}

/// An aggregate resolved to its record-field byte offsets — the fold/emit form used
/// inside [`gen_grouped`].
#[derive(Clone)]
struct ResolvedAgg {
    kind: AggKind,
    acc: AccTy,
    value_off: usize,
    count_off: Option<usize>,
    arg: Option<Expr>,
}

/// The aggregate's input value (`None` for `count(*)`) and its per-row validity bit
/// (`true` with no argument or a non-null argument).
fn agg_arg_value(
    arg: &Option<Expr>,
    ctx: &mut Ctx,
    row: &Row,
    schema: &SchemaRef,
    cx: &CodegenCtx,
) -> (Option<ColVal>, Var<bool>) {
    match arg {
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

impl ResolvedAgg {
    /// Fold this row into the group's record `rec`. Only non-null inputs contribute;
    /// `sum`/`min`/`max`/`avg` also bump a non-null count (the "seen" bit and, for
    /// `avg`, the divisor).
    fn fold(
        &self,
        ctx: &mut Ctx,
        rec: Var<SMutPtr<u8>>,
        row: &Row,
        schema: &SchemaRef,
        cg_ctx: &CodegenCtx,
    ) {
        let (val_cv, valid) = agg_arg_value(&self.arg, ctx, row, schema, cg_ctx);
        let kind = self.kind;
        let value_off = self.value_off;
        let count_off = self.count_off;
        match kind {
            AggKind::Count => ctx.if_then(valid, move |ctx| {
                let f = field_i64(value_off);
                let cur = f.get(ctx, rec);
                let next = ctx.bind(add(cur, 1i64));
                f.set(ctx, rec, next);
            }),
            // sum/min/max fold in the accumulator's cell type: `i64` for integer
            // inputs (i32 widened), `f64` for a `Float64` column.
            AggKind::Sum | AggKind::Min | AggKind::Max => match self.acc {
                AccTy::I64 => {
                    let v = to_i64(ctx, val_cv.unwrap());
                    ctx.if_then(valid, move |ctx| {
                        bump_count(ctx, count_off, rec);
                        let f = field_i64(value_off);
                        let cur = f.get(ctx, rec);
                        let next = combine_i64(ctx, kind, cur, v);
                        f.set(ctx, rec, next);
                    });
                }
                AccTy::F64 => {
                    let v = to_f64(ctx, val_cv.unwrap());
                    ctx.if_then(valid, move |ctx| {
                        bump_count(ctx, count_off, rec);
                        let f = field_f64(value_off);
                        let cur = f.get(ctx, rec);
                        let next = combine_f64(ctx, kind, cur, v);
                        f.set(ctx, rec, next);
                    });
                }
            },
            AggKind::Avg => {
                let v = to_f64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    bump_count(ctx, count_off, rec);
                    let f = field_f64(value_off);
                    let cur = f.get(ctx, rec);
                    let next = ctx.bind(add(cur, v));
                    f.set(ctx, rec, next);
                });
            }
        }
    }

    /// Manifest this aggregate's final `ColVal` from the group's record `rec`:
    /// `count` is never null; `sum`/`min`/`max` are null when the count is 0; `avg`
    /// divides `sum/count` (null when count 0).
    fn finalize(&self, ctx: &mut Ctx, rec: Var<SMutPtr<u8>>) -> ColVal {
        match self.kind {
            AggKind::Count => {
                ColVal::I64(field_i64(self.value_off).get(ctx, rec), Nullness::NonNull)
            }
            AggKind::Sum | AggKind::Min | AggKind::Max => {
                let seen = self.seen(ctx, rec);
                match self.acc {
                    AccTy::I64 => ColVal::I64(
                        field_i64(self.value_off).get(ctx, rec),
                        Nullness::Nullable(seen),
                    ),
                    AccTy::F64 => ColVal::F64(
                        field_f64(self.value_off).get(ctx, rec),
                        Nullness::Nullable(seen),
                    ),
                }
            }
            AggKind::Avg => {
                let sum = field_f64(self.value_off).get(ctx, rec);
                let count = field_i64(self.count_off.unwrap()).get(ctx, rec);
                let cf = ctx.bind(int_to_float::<f64, i64, _>(count));
                let avg = ctx.bind(div(sum, cf));
                let seen = ctx.bind(gt(count, 0i64));
                ColVal::F64(avg, Nullness::Nullable(seen))
            }
        }
    }

    /// Whether group `rec` saw any non-null input (its non-null count > 0).
    fn seen(&self, ctx: &mut Ctx, rec: Var<SMutPtr<u8>>) -> Var<bool> {
        let count = field_i64(self.count_off.expect("nullable agg has a count")).get(ctx, rec);
        ctx.bind(gt(count, 0i64))
    }
}

/// `rec.count += 1` (the non-null count field for a nullable aggregate).
fn bump_count(ctx: &mut Ctx, count_off: Option<usize>, rec: Var<SMutPtr<u8>>) {
    let f = field_i64(count_off.expect("nullable aggregate has a count field"));
    let cur = f.get(ctx, rec);
    let next = ctx.bind(add(cur, 1i64));
    f.set(ctx, rec, next);
}

/// Fold one `i64` input into a `sum`/`min`/`max` accumulator (`add`/`min`/`max`).
/// The arms return distinct staged types, so each binds to a `Var` before returning.
fn combine_i64(ctx: &mut Ctx, kind: AggKind, cur: Var<i64>, v: Var<i64>) -> Var<i64> {
    match kind {
        AggKind::Sum => ctx.bind(add(cur, v)),
        AggKind::Min => ctx.bind(min(cur, v)),
        AggKind::Max => ctx.bind(max(cur, v)),
        _ => unreachable!("combine_i64 is only for sum/min/max"),
    }
}

/// `f64` twin of [`combine_i64`] — for `sum`/`min`/`max` over a `Float64` column.
fn combine_f64(ctx: &mut Ctx, kind: AggKind, cur: Var<f64>, v: Var<f64>) -> Var<f64> {
    match kind {
        AggKind::Sum => ctx.bind(add(cur, v)),
        AggKind::Min => ctx.bind(min(cur, v)),
        AggKind::Max => ctx.bind(max(cur, v)),
        _ => unreachable!("combine_f64 is only for sum/min/max"),
    }
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
    // The typed opaque reference back to the arrow array, for the extern fallback.
    let array = ctx.bind(opaque_ref::<StringViewArray, _>(load_field(
        batch.get_ref_unchecked(col as u64),
        FfiArrayType::array(),
    )));
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
    cx: &CodegenCtx,
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
fn write_str_col<O: OutSink>(ctx: &mut Ctx, out: O, col_idx: usize, cv: ColVal, cx: &CodegenCtx) {
    let sv = match cv {
        ColVal::Str(sv, _) => sv,
        other => panic!(
            "string output column got a non-string value: {}",
            tag(other)
        ),
    };
    let builder = ctx.bind(load_field(
        out.get_mut_unchecked(col_idx as u64),
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

fn gen_predicate(
    ctx: &mut Ctx,
    e: &Expr,
    schema: &SchemaRef,
    row: &Row,
    cx: &CodegenCtx,
) -> Var<bool> {
    let cv = gen_expr(ctx, e, schema, row, cx);
    let cond = as_bool(cv);
    // SQL: a NULL predicate does not pass the filter -> keep iff (valid && cond).
    match cv.nullness() {
        Nullness::NonNull => cond,
        Nullness::Nullable(valid) => ctx.bind(select(valid, cond, Const::<bool>::new(false))),
    }
}

fn gen_expr(ctx: &mut Ctx, e: &Expr, schema: &SchemaRef, row: &Row, cx: &CodegenCtx) -> ColVal {
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
    cx: &CodegenCtx,
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
/// `&CodegenCtx`) so it can be called from `'static` branch closures.
fn resolve(
    ctx: &mut Ctx,
    sv: StrVal,
    str_ptr: ExternRef<StrPtrExtern>,
) -> (Var<SPtr<u8>>, Var<u64>) {
    match sv {
        StrVal::Column { lo, array, row, .. } => {
            let ptr = ctx.bind(call_extern2(str_ptr, array, row));
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
fn str_eq(ctx: &mut Ctx, l: StrVal, r: StrVal, cx: &CodegenCtx) -> Var<bool> {
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
fn gen_eq(ctx: &mut Ctx, l: ColVal, r: ColVal, cx: &CodegenCtx) -> Var<bool> {
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

fn gen_literal(ctx: &mut Ctx, sv: &ScalarValue, cx: &CodegenCtx) -> ColVal {
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
/// address in the `BytesPool` (interned once at build time — see [`CodegenCtx::lits`]),
/// and `view` carries the view halves (for *any* length) so `str_eq`'s length +
/// prefix fast-reject and inline compare apply.
fn gen_str_literal(ctx: &mut Ctx, s: &str, cx: &CodegenCtx) -> ColVal {
    let ptr = *cx
        .lits
        .get(s)
        .unwrap_or_else(|| panic!("literal not interned: {s:?}"));
    ColVal::Str(
        StrVal::Bytes {
            ptr: ctx.bind(const_ptr::<u8>(ptr)),
            len: ctx.var(s.len() as u64),
            view: Some(literal_view_halves(s)),
        },
        Nullness::NonNull,
    )
}

fn gen_binary(
    ctx: &mut Ctx,
    be: &BinaryExpr,
    schema: &SchemaRef,
    row: &Row,
    cx: &CodegenCtx,
) -> ColVal {
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

    fn update(&self, ctx: &mut Ctx, row: &Row, schema: &SchemaRef, cx: &CodegenCtx) {
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
