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
use arrow_lms::{ArrayBatchOps, FfiArray, PrimitiveArrayView};
use datafusion_common::ScalarValue;
use datafusion_expr::expr::ScalarFunction;
use datafusion_expr::{BinaryExpr, Expr, Operator as DfOp};
use rust_lms::prelude::*;
use rust_lms_std::{DynamicRecord, FieldId, RecordLayout, SVec};

use crate::group::GroupState;
use crate::output::{OutColHandle, OutputHandle};
use crate::plan::Operator;
use crate::runtime::{Runtime, StrPtrExtern};
use crate::scan::Inputs;
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
    /// Baked pointers to the growable output columns (see `output::OutCols`). The
    /// kernel appends each emitted row into these via `SVec::push` / a string builder.
    pub out: Rc<OutputHandle>,
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

/// A staged expression yielding a single read-only input batch (`&[FfiArray]`) —
/// one stream batch, produced inside [`gen_scan`] and consumed by the row reads.
pub trait BatchSource: Staged<Out = SRef<'static, Slice<FfiArray>>> + Copy + 'static {}
impl<T> BatchSource for T where T: Staged<Out = SRef<'static, Slice<FfiArray>>> + Copy + 'static {}

/// A staged expression yielding the input streams (`&mut Inputs`), threaded through
/// the operator walk so [`gen_scan`] can pull batches via `scan_next`.
pub trait InputsSource: Staged<Out = SRefMut<'static, Opaque<Inputs>>> + Copy + 'static {}
impl<T> InputsSource for T where T: Staged<Out = SRefMut<'static, Opaque<Inputs>>> + Copy + 'static {}

/// Downstream continuation, invoked once per emitted row at code-generation
/// time; the [`Row`] rides by value (cheap `Copy` handles).
type Yld = Box<dyn FnOnce(&mut Ctx, Row) + 'static>;

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

fn gen_op<I: InputsSource>(op: &Operator, ctx: &mut Ctx, inputs: I, cx: &CodegenCtx, yld: Yld) {
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
                                let nulls =
                                    ctx.bind(batch.primitive::<u64>(col).validity().null_count());
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
    }
}

/// GROUP BY as a push operator: fold each input row into the Rust-hosted group
/// state (baked buffers indexed by `gidx`), then emit one *manifested* Row per
/// group downstream. Keeps the hot fold loop in the JIT and lets the projection /
/// filter above it run in the same kernel (see [`CodegenCtx::group`]). `avg` divides at
/// emit; nullability rides in the emitted `ColVal` (so `write_col` handles nulls).
fn gen_grouped<I: InputsSource>(
    ctx: &mut Ctx,
    inputs: I,
    op: &Operator,
    cx: &CodegenCtx,
    yld: Yld,
) {
    let Operator::Aggregate {
        group_exprs,
        aggs,
        input,
        schema: out_schema,
    } = op
    else {
        unreachable!("gen_grouped called on a non-Aggregate operator")
    };
    let handle = cx
        .group
        .clone()
        .expect("grouped aggregate without a baked GroupState");
    let agg_tys = agg_output_types(out_schema, group_exprs.len());
    let key_spec = KeySpec::from_schema(out_schema, group_exprs.len());
    let record = group_record(aggs, &agg_tys, &key_spec);
    let layout = record.layout;
    let key_fields = record.key;
    let key_valid = record.key_valid;
    let state = handle.state;
    // How the fold computes the key: one expression (single key) or a packed layout
    // over all key columns (composite).
    let key_source = match &key_spec {
        KeySpec::Single { .. } => KeySource::Single(group_exprs[0].clone()),
        KeySpec::Composite(cols) => {
            // A string column forces the variable-length builder; otherwise the fast
            // fixed-size stack pack.
            if cols.iter().any(|(ty, _)| *ty == DataType::Utf8View) {
                KeySource::CompositeBytes(bytes_key(group_exprs, cols))
            } else {
                KeySource::CompositeFixed(packed_key(group_exprs, cols))
            }
        }
    };
    let n_keys = group_exprs.len();
    // Resolve each aggregate to its record fields (byte offsets) + its argument.
    let resolved: Vec<ResolvedAgg> = record
        .aggs
        .iter()
        .zip(aggs)
        .map(|(a, e)| ResolvedAgg {
            kind: a.kind,
            value: a.value,
            count: a.count,
            arg: GroupedAgg::parse(e).arg,
        })
        .collect();

    let input_schema = input.output_schema();

    // Fold: per input row, find-or-insert the group and fold each aggregate into its
    // packed record. The `upsert` extern grows the records buffer (host-side), writes
    // the key into the record, and returns the record pointer — valid until the next
    // `upsert`.
    let cx_c = cx.clone();
    let resolved_f = resolved.clone();
    let key_source_f = key_source.clone();
    let schema_f = input_schema.clone();
    gen_op(
        input,
        ctx,
        inputs,
        cx,
        Box::new(move |ctx, row| {
            let rec_ptr = match &key_source_f {
                KeySource::Single(expr) => {
                    let key_cv = gen_expr(ctx, expr, &schema_f, &row, &cx_c);
                    fold_single_key(ctx, key_fields, key_valid, key_cv, state, &cx_c, layout)
                }
                KeySource::CompositeFixed(packed) => {
                    pack_and_upsert(ctx, packed, &row, &schema_f, &cx_c, state)
                }
                KeySource::CompositeBytes(bk) => {
                    build_bytes_key(ctx, bk, &row, &schema_f, &cx_c, state)
                }
            };
            let rec = layout.wrap(rec_ptr);
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
        let key_cols = match &key_source {
            KeySource::Single(_) => {
                vec![emit_single_key(ctx, key_fields, key_valid, rec)]
            }
            KeySource::CompositeFixed(packed) => {
                let KeyFields::Composite { ptr } = key_fields else {
                    unreachable!("composite key source without composite key fields")
                };
                let packed_ptr = rec.get(ctx, ptr);
                unpack_composite(ctx, packed, packed_ptr)
            }
            KeySource::CompositeBytes(bk) => {
                let KeyFields::Composite { ptr } = key_fields else {
                    unreachable!("composite key source without composite key fields")
                };
                let flat_ptr = rec.get(ctx, ptr);
                unpack_bytes(ctx, bk, flat_ptr)
            }
        };
        let mut row: Row = Vec::with_capacity(n_keys + resolved.len());
        row.extend(key_cols);
        for agg in &resolved {
            row.push(agg.finalize(ctx, rec));
        }
        yld(ctx, row);
        ctx.store(g, add(g, 1u64));
    });
}

/// Fold-time upsert for a **single** key: dispatch on the key type, then finish (null
/// handling) via [`finish_upsert`]. The key computation runs unconditionally but is
/// harmless on a null row (unused; `str_ptr` reads a null row safely).
fn fold_single_key(
    ctx: &mut Ctx,
    key_fields: KeyFields,
    key_valid: Option<FieldId<i64>>,
    key_cv: ColVal,
    state: *mut GroupState,
    cx: &CodegenCtx,
    layout: RecordLayout,
) -> Var<SMutPtr<u8>> {
    match key_fields {
        KeyFields::Int(_) => {
            // The table keys on `u64` bits (grouping is sign-agnostic); a no-op cast.
            let key = to_i64(ctx, key_cv);
            let key_bits = ctx.bind(int_cast::<u64, i64, _>(key));
            let call = call_extern2(
                cx.rt.group_upsert,
                const_opaque_mut::<GroupState>(state),
                key_bits,
            );
            finish_upsert(ctx, call, key_valid, key_cv, state, cx, layout)
        }
        KeyFields::Float(_) => {
            // Key on the f64's bits: canonicalize (`-0.0`/NaN), then bitcast.
            let key = coerce_f64(ctx, key_cv);
            let canon = canonical_f64(ctx, key);
            let key_bits = ctx.bind(bitcast::<u64, f64, _>(canon));
            let call = call_extern2(
                cx.rt.group_upsert,
                const_opaque_mut::<GroupState>(state),
                key_bits,
            );
            finish_upsert(ctx, call, key_valid, key_cv, state, cx, layout)
        }
        KeyFields::Str { .. } => {
            // Pass the key's content bytes; the extern copies them into its pool.
            let (ptr, len) = resolve(ctx, as_str(key_cv), cx.rt.str_ptr);
            let call = call_extern3(
                cx.rt.group_upsert_str,
                const_opaque_mut::<GroupState>(state),
                ptr,
                len,
            );
            finish_upsert(ctx, call, key_valid, key_cv, state, cx, layout)
        }
        KeyFields::Composite { .. } => {
            unreachable!("composite key in the single-key fold")
        }
    }
}

/// Emit the single key `ColVal` for one group's record.
fn emit_single_key(
    ctx: &mut Ctx,
    key_fields: KeyFields,
    key_valid: Option<FieldId<i64>>,
    rec: DynamicRecord,
) -> ColVal {
    // For a nullable key, the null group's record carries `key_valid == 0`; emit NULL.
    let key_null = match key_valid {
        None => Nullness::NonNull,
        Some(vf) => {
            let v = rec.get(ctx, vf);
            Nullness::Nullable(ctx.bind(gt(v, 0i64)))
        }
    };
    match key_fields {
        KeyFields::Int(f) => ColVal::I64(rec.get(ctx, f), key_null),
        KeyFields::Float(f) => ColVal::F64(rec.get(ctx, f), key_null),
        KeyFields::Str { ptr, len } => {
            let ptr = rec.get(ctx, ptr);
            let len = rec.get(ctx, len);
            ColVal::Str(
                StrVal::Bytes {
                    ptr,
                    len,
                    view: None,
                },
                key_null,
            )
        }
        KeyFields::Composite { .. } => unreachable!("composite key in the single-key emit"),
    }
}

/// Complete a fold-row's upsert: for a non-nullable key, just run `valid_call` (the
/// key-specific `group_upsert*`). For a nullable key, branch on the key's validity —
/// a real key runs `valid_call`, a NULL key routes to `group_upsert_null` — and record
/// the validity (`1`/`0`) in the record's key-valid cell for emit. `valid_call`'s key
/// computation has already run in the caller (harmless on a null row); only the extern
/// call is guarded by the branch.
fn finish_upsert<C>(
    ctx: &mut Ctx,
    valid_call: C,
    key_valid: Option<FieldId<i64>>,
    key_cv: ColVal,
    state: *mut GroupState,
    cx: &CodegenCtx,
    layout: RecordLayout,
) -> Var<SMutPtr<u8>>
where
    C: Staged<Out = SMutPtr<u8>> + 'static,
{
    match key_valid {
        None => ctx.bind(valid_call),
        Some(valid_field) => {
            let valid = match key_cv.nullness() {
                Nullness::Nullable(b) => b,
                Nullness::NonNull => ctx.bind(Const::<bool>::new(true)),
            };
            let null_call = call_extern1(
                cx.rt.group_upsert_null,
                const_opaque_mut::<GroupState>(state),
            );
            let rec_ptr = ctx.bind(if_then_else(valid, valid_call, null_call));
            let valid_bits = ctx.bind(select(valid, 1i64, 0i64));
            layout.wrap(rec_ptr).set(ctx, valid_field, valid_bits);
            rec_ptr
        }
    }
}

/// How the fold obtains the key: a single key expression, or a packed layout over all
/// key columns (composite). Cloned into the fold closure and kept for emit.
#[derive(Clone)]
enum KeySource {
    Single(Expr),
    /// All-fixed-width composite — packed into a fixed-size stack key (fast path).
    CompositeFixed(PackedKey),
    /// Composite with ≥1 variable-length (string) column — built via the host key
    /// builder into a flat byte key (`group_key_*` + `group_upsert_composite`).
    CompositeBytes(BytesKey),
}

/// A composite key containing string columns: an ordered list of columns (fixed or
/// string) plus a null bitmap. The fold pushes each column's bytes into the host
/// scratch (`group_key_*`), then interns the assembled flat key; emit unpacks it with
/// a running byte offset (string lengths make offsets runtime-dependent).
#[derive(Clone)]
struct BytesKey {
    cols: Vec<BytesCol>,
}

#[derive(Clone)]
struct BytesCol {
    kind: BytesColKind,
    /// Null-bitmap bit index (= column position).
    bit: usize,
    nullable: bool,
    expr: Expr,
}

#[derive(Clone, Copy, PartialEq)]
enum BytesColKind {
    I64,
    F64,
    Str,
}

/// Build the byte-key descriptor for a string-containing composite key.
fn bytes_key(group_exprs: &[Expr], cols: &[(DataType, bool)]) -> BytesKey {
    let cols = group_exprs
        .iter()
        .zip(cols)
        .enumerate()
        .map(|(i, (expr, (ty, nullable)))| {
            let kind = match ty {
                DataType::Int32 | DataType::Int64 => BytesColKind::I64,
                DataType::Float64 => BytesColKind::F64,
                DataType::Utf8View => BytesColKind::Str,
                other => panic!("unsupported composite key column type: {other}"),
            };
            BytesCol {
                kind,
                bit: i,
                nullable: *nullable,
                expr: expr.clone(),
            }
        })
        .collect();
    BytesKey { cols }
}

/// A composite key's **packed** layout: one 8-byte cell per key column (`i64`/`f64`)
/// plus a trailing `u64` null bitmap (bit `i` = column `i` is NULL). The fold writes
/// the canonicalized column values + bitmap into a stack scratch of this shape — the
/// bytes handed to `group_upsert_str` — and emit reads them back to produce the output
/// key columns. One `RecordLayout` is shared by pack and unpack, so offsets agree.
#[derive(Clone)]
struct PackedKey {
    layout: RecordLayout,
    cols: Vec<PackedCol>,
    bitmap: FieldId<u64>,
    nbytes: usize,
}

#[derive(Clone)]
struct PackedCol {
    field: PackedColField,
    /// Null-bitmap bit index (= column position).
    bit: usize,
    nullable: bool,
    /// The group-by expression producing this column's value.
    expr: Expr,
}

#[derive(Clone, Copy)]
enum PackedColField {
    I64(FieldId<i64>),
    F64(FieldId<f64>),
}

/// Build the packed-key layout from the key columns (fixed-width only for now).
fn packed_key(group_exprs: &[Expr], cols: &[(DataType, bool)]) -> PackedKey {
    let mut layout = RecordLayout::new();
    let packed_cols = group_exprs
        .iter()
        .zip(cols)
        .enumerate()
        .map(|(i, (expr, (ty, nullable)))| {
            let field = match ty {
                DataType::Int32 | DataType::Int64 => PackedColField::I64(layout.field::<i64>()),
                DataType::Float64 => PackedColField::F64(layout.field::<f64>()),
                other => panic!("unsupported composite key column type: {other}"),
            };
            PackedCol {
                field,
                bit: i,
                nullable: *nullable,
                expr: expr.clone(),
            }
        })
        .collect();
    let bitmap = layout.field::<u64>();
    let nbytes = layout.stride();
    PackedKey {
        layout,
        cols: packed_cols,
        bitmap,
        nbytes,
    }
}

/// Pack the composite key columns into a fresh stack scratch (canonicalized values +
/// null bitmap) then find-or-insert via `group_upsert_str` (which copies the packed
/// bytes into the pool on a miss). Returns the group's record pointer.
fn pack_and_upsert(
    ctx: &mut Ctx,
    packed: &PackedKey,
    row: &Row,
    schema: &SchemaRef,
    cx: &CodegenCtx,
    state: *mut GroupState,
) -> Var<SMutPtr<u8>> {
    let scratch = ctx.bind(stack_alloc(packed.nbytes));
    let prec = packed.layout.wrap(scratch);
    let bitmap = ctx.var(0u64);
    for col in &packed.cols {
        let cv = gen_expr(ctx, &col.expr, schema, row, cx);
        let valid = match cv.nullness() {
            Nullness::Nullable(b) => Some(b),
            Nullness::NonNull => None,
        };
        // NULL columns pack a canonical value (0 / +0.0); the bitmap distinguishes them.
        match col.field {
            PackedColField::I64(f) => {
                let v = to_i64(ctx, cv);
                let stored = match valid {
                    Some(vb) => ctx.bind(select(vb, v, 0i64)),
                    None => v,
                };
                prec.set(ctx, f, stored);
            }
            PackedColField::F64(f) => {
                let v = coerce_f64(ctx, cv);
                let canon = canonical_f64(ctx, v);
                let stored = match valid {
                    Some(vb) => ctx.bind(select(vb, canon, 0.0f64)),
                    None => canon,
                };
                prec.set(ctx, f, stored);
            }
        }
        if let Some(vb) = valid {
            // null → set bit `col.bit`; valid → 0.
            let bit = ctx.bind(select(vb, 0u64, 1u64 << col.bit));
            let next = ctx.bind(bitor(bitmap, bit));
            ctx.store(bitmap, next);
        }
    }
    prec.set(ctx, packed.bitmap, bitmap);
    ctx.bind(call_extern3(
        cx.rt.group_upsert_str,
        const_opaque_mut::<GroupState>(state),
        ptr_as_const(scratch),
        Const::<u64>::new(packed.nbytes as u64),
    ))
}

/// Unpack a group's pooled packed key into its output key `ColVal`s (nulls from the
/// bitmap).
fn unpack_composite(
    ctx: &mut Ctx,
    packed: &PackedKey,
    packed_ptr: Var<SMutPtr<u8>>,
) -> Vec<ColVal> {
    let prec = packed.layout.wrap(packed_ptr);
    let bitmap = prec.get(ctx, packed.bitmap);
    packed
        .cols
        .iter()
        .map(|col| {
            let null = if col.nullable {
                let shifted = ctx.bind(shr(bitmap, col.bit as u64));
                let bit = ctx.bind(bitand(shifted, 1u64));
                Nullness::Nullable(ctx.bind(eq(bit, 0u64)))
            } else {
                Nullness::NonNull
            };
            match col.field {
                PackedColField::I64(f) => ColVal::I64(prec.get(ctx, f), null),
                PackedColField::F64(f) => ColVal::F64(prec.get(ctx, f), null),
            }
        })
        .collect()
}

/// A column's pushable form for the variable-length composite key builder.
enum Pushable {
    /// A fixed-width column's canonicalized bits (or the null bitmap).
    U64(Var<u64>),
    /// A string column's content reference.
    Bytes(Var<SPtr<u8>>, Var<u64>),
}

/// Build a **variable-length** composite key: evaluate each column (computing its
/// pushable form + the null bitmap), then push the bitmap and each column's bytes into
/// the host scratch (`group_key_*`) and intern the assembled flat key
/// (`group_upsert_composite`). Returns the group's record pointer.
fn build_bytes_key(
    ctx: &mut Ctx,
    bk: &BytesKey,
    row: &Row,
    schema: &SchemaRef,
    cx: &CodegenCtx,
    state: *mut GroupState,
) -> Var<SMutPtr<u8>> {
    ctx.emit(call_extern1(
        cx.rt.group_key_reset,
        const_opaque_mut::<GroupState>(state),
    ));
    let bitmap = ctx.var(0u64);
    let mut pushables = Vec::with_capacity(bk.cols.len());
    for col in &bk.cols {
        let cv = gen_expr(ctx, &col.expr, schema, row, cx);
        let valid = match cv.nullness() {
            Nullness::Nullable(b) => Some(b),
            Nullness::NonNull => None,
        };
        // NULL columns push a canonical value (0 / +0.0 / empty); the bitmap tells apart.
        let pushable = match col.kind {
            BytesColKind::I64 => {
                let v = to_i64(ctx, cv);
                let v = match valid {
                    Some(vb) => ctx.bind(select(vb, v, 0i64)),
                    None => v,
                };
                Pushable::U64(ctx.bind(int_cast::<u64, i64, _>(v)))
            }
            BytesColKind::F64 => {
                let v = coerce_f64(ctx, cv);
                let canon = canonical_f64(ctx, v);
                let v = match valid {
                    Some(vb) => ctx.bind(select(vb, canon, 0.0f64)),
                    None => canon,
                };
                Pushable::U64(ctx.bind(bitcast::<u64, f64, _>(v)))
            }
            BytesColKind::Str => {
                let (ptr, len) = resolve(ctx, as_str(cv), cx.rt.str_ptr);
                Pushable::Bytes(ptr, len)
            }
        };
        if let Some(vb) = valid {
            let bit = ctx.bind(select(vb, 0u64, 1u64 << col.bit));
            let next = ctx.bind(bitor(bitmap, bit));
            ctx.store(bitmap, next);
        }
        pushables.push(pushable);
    }
    // Push the bitmap first, then each column in order.
    ctx.emit(call_extern2(
        cx.rt.group_key_push_u64,
        const_opaque_mut::<GroupState>(state),
        bitmap,
    ));
    for p in pushables {
        match p {
            Pushable::U64(v) => ctx.emit(call_extern2(
                cx.rt.group_key_push_u64,
                const_opaque_mut::<GroupState>(state),
                v,
            )),
            Pushable::Bytes(ptr, len) => ctx.emit(call_extern3(
                cx.rt.group_key_push_bytes,
                const_opaque_mut::<GroupState>(state),
                ptr,
                len,
            )),
        }
    }
    ctx.bind(call_extern1(
        cx.rt.group_upsert_composite,
        const_opaque_mut::<GroupState>(state),
    ))
}

/// Load a value of type `T` at byte offset `off` in `base` (a runtime offset, since a
/// string column's length shifts everything after it).
fn load_at<T: StagedType + CopyType + 'static>(
    ctx: &mut Ctx,
    base: Var<SMutPtr<u8>>,
    off: Var<i64>,
) -> Var<T> {
    ctx.bind(load_ref_mut(ptr_cast_mut::<T, u8, _>(ptr_offset_mut(
        base, off,
    ))))
}

/// A `*const u8` at byte offset `off` in `base`.
fn ptr_at(ctx: &mut Ctx, base: Var<SMutPtr<u8>>, off: Var<i64>) -> Var<SPtr<u8>> {
    ctx.bind(ptr_as_const(ptr_offset_mut(base, off)))
}

/// Unpack a variable-length composite key (the pooled flat bytes: `[bitmap | col0 |
/// …]`, each fixed column 8 bytes, each string `[len | content]`) into its output key
/// `ColVal`s, walking a running byte offset.
fn unpack_bytes(ctx: &mut Ctx, bk: &BytesKey, flat_ptr: Var<SMutPtr<u8>>) -> Vec<ColVal> {
    let off0 = ctx.var(0i64);
    let bitmap = load_at::<u64>(ctx, flat_ptr, off0);
    let offset = ctx.var(8i64); // running byte offset (past the bitmap)
    let mut out = Vec::with_capacity(bk.cols.len());
    for col in &bk.cols {
        let null = if col.nullable {
            let shifted = ctx.bind(shr(bitmap, col.bit as u64));
            let bit = ctx.bind(bitand(shifted, 1u64));
            Nullness::Nullable(ctx.bind(eq(bit, 0u64)))
        } else {
            Nullness::NonNull
        };
        match col.kind {
            BytesColKind::I64 => {
                out.push(ColVal::I64(load_at::<i64>(ctx, flat_ptr, offset), null));
                let next = ctx.bind(add(offset, 8i64));
                ctx.store(offset, next);
            }
            BytesColKind::F64 => {
                out.push(ColVal::F64(load_at::<f64>(ctx, flat_ptr, offset), null));
                let next = ctx.bind(add(offset, 8i64));
                ctx.store(offset, next);
            }
            BytesColKind::Str => {
                let len = load_at::<u64>(ctx, flat_ptr, offset);
                let content_off = ctx.bind(add(offset, 8i64));
                let ptr = ptr_at(ctx, flat_ptr, content_off);
                out.push(ColVal::Str(
                    StrVal::Bytes {
                        ptr,
                        len,
                        view: None,
                    },
                    null,
                ));
                // advance past [len | content] = 8 + len.
                let len_i = ctx.bind(int_cast::<i64, u64, _>(len));
                let next = ctx.bind(add(content_off, len_i));
                ctx.store(offset, next);
            }
        }
    }
    out
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

/// An aggregate's value field, typed by the cell it folds in: an `i64` field
/// (integer inputs, `count`) or an `f64` field (`Float64` `sum`/`min`/`max`, `avg`).
/// The `FieldId` is the layout-bound token used for every access.
#[derive(Clone, Copy)]
enum AggValueField {
    I64(FieldId<i64>),
    F64(FieldId<f64>),
}

impl AggValueField {
    fn offset(&self) -> usize {
        match self {
            AggValueField::I64(f) => f.offset(),
            AggValueField::F64(f) => f.offset(),
        }
    }
}

/// The record fields an aggregate uses: its value field (typed) and, for
/// `sum`/`min`/`max`/`avg`, a non-null count field (`i64`).
struct AggFields {
    kind: AggKind,
    value: AggValueField,
    count: Option<FieldId<i64>>,
}

/// The group key's field(s) in the record — an `i64` for an integer key, or a
/// `(ptr, len)` byte reference for a `Utf8View` key (the extern writes the pooled
/// pointer; emit reads it to produce the output string). Always the record's leading
/// field(s), at offset 0 — the `group_upsert*` externs rely on this.
#[derive(Clone, Copy)]
enum KeyFields {
    Int(FieldId<i64>),
    /// A `Float64` key. Stored (by the `Int` upsert extern) as its `u64` bits in the
    /// leading cell; read back as `f64` at emit (the token reinterprets those bytes).
    Float(FieldId<f64>),
    Str {
        ptr: FieldId<SPtr<u8>>,
        len: FieldId<u64>,
    },
    /// A **composite** (multi-column) key, stored like a string key — a pointer to the
    /// pooled *packed* key bytes (the `len` cell the extern also writes is reserved but
    /// unread; the packed length is a compile-time constant). The fold packs the
    /// columns; emit unpacks them (see [`PackedKey`]). Nulls ride in the packed bitmap,
    /// not a `key_valid` cell, so each `(a, b, …)` combination is its own group.
    Composite {
        ptr: FieldId<SMutPtr<u8>>,
    },
}

/// What the GROUP BY key is: one column (specialised int/float/string paths) or
/// several (a packed byte key). Drives the record's key fields and the fold/emit.
enum KeySpec {
    Single { ty: DataType, nullable: bool },
    Composite(Vec<(DataType, bool)>), // (type, nullable) per key column, in order
}

impl KeySpec {
    /// Derive the key spec from the aggregate's output schema (its first `n_keys`
    /// fields are the group keys).
    fn from_schema(out_schema: &SchemaRef, n_keys: usize) -> KeySpec {
        if n_keys == 1 {
            let f = out_schema.field(0);
            KeySpec::Single {
                ty: f.data_type().clone(),
                nullable: f.is_nullable(),
            }
        } else {
            KeySpec::Composite(
                (0..n_keys)
                    .map(|i| {
                        let f = out_schema.field(i);
                        (f.data_type().clone(), f.is_nullable())
                    })
                    .collect(),
            )
        }
    }
}

/// The packed record of a group-by's state: the key field(s), then a value (+ count)
/// field per aggregate. One [`RecordLayout`], derived deterministically from the
/// aggregates + key type, is shared by the host allocator ([`group_template`]) and
/// codegen so they agree on field offsets — exactly like a `#[repr(C)]` struct, but
/// query-shaped. Fields are addressed only through typed [`FieldId`] tokens.
struct GroupRecord {
    layout: RecordLayout,
    key: KeyFields,
    /// A key-validity cell — `Some` only when the key column is **nullable**. The fold
    /// stores `1` for a real key and `0` for the null group; emit reads it to decide
    /// whether the key is NULL. Absent (and zero overhead) for non-nullable keys.
    key_valid: Option<FieldId<i64>>,
    aggs: Vec<AggFields>,
}

/// Reserve the leading key field(s) for a single-column key of type `key_ty`. Must be
/// first (offset 0) — the `group_upsert*` externs write the key there.
fn key_fields_single(layout: &mut RecordLayout, key_ty: &DataType) -> KeyFields {
    match key_ty {
        DataType::Int32 | DataType::Int64 => KeyFields::Int(layout.field::<i64>()),
        DataType::Float64 => KeyFields::Float(layout.field::<f64>()),
        DataType::Utf8View => KeyFields::Str {
            ptr: layout.field::<SPtr<u8>>(),
            len: layout.field::<u64>(),
        },
        other => panic!("unsupported GROUP BY key type: {other}"),
    }
}

/// Build the packed-record layout from the key spec and the aggregates + their output
/// `DataType`s (parallel to `aggs`). The output type picks the value cell ([`acc_ty`]);
/// `sum`/`min`/`max` over `Float64` fold in an `f64` field, everything else `i64`.
fn group_record(aggs: &[Expr], agg_tys: &[DataType], key_spec: &KeySpec) -> GroupRecord {
    let mut layout = RecordLayout::new();
    let (key, key_valid) = match key_spec {
        KeySpec::Single { ty, nullable } => {
            let key = key_fields_single(&mut layout, ty);
            // A nullable single key routes NULLs to the null group via a validity cell.
            (key, nullable.then(|| layout.field::<i64>()))
        }
        KeySpec::Composite(_) => {
            // A `(ptr, len)` to the pooled packed key (like a string key); nulls are in
            // the packed bitmap, so no separate validity cell.
            let ptr = layout.field::<SMutPtr<u8>>();
            let _len = layout.field::<u64>(); // reserved for the extern's len write; unread
            (KeyFields::Composite { ptr }, None)
        }
    };
    let mut agg_fields = Vec::with_capacity(aggs.len());
    for (e, out) in aggs.iter().zip(agg_tys) {
        let kind = GroupedAgg::parse(e).kind;
        let value = match acc_ty(kind, out) {
            AccTy::I64 => AggValueField::I64(layout.field::<i64>()),
            AccTy::F64 => AggValueField::F64(layout.field::<f64>()),
        };
        // sum/min/max/avg track a non-null count (seen / divisor); count does not.
        let count = matches!(
            kind,
            AggKind::Sum | AggKind::Min | AggKind::Max | AggKind::Avg
        )
        .then(|| layout.field::<i64>());
        agg_fields.push(AggFields { kind, value, count });
    }
    GroupRecord {
        layout,
        key,
        key_valid,
        aggs: agg_fields,
    }
}

/// A value field's identity fill, as raw `i64` bits (an `f64` field reuses the same
/// 8 bytes): `0` for count/sum/avg, `i64::MAX`/`MIN` for integer min/max, `±∞` bits
/// for float min/max.
fn value_init_bits(kind: AggKind, value: &AggValueField) -> i64 {
    match (kind, value) {
        (AggKind::Min, AggValueField::I64(_)) => i64::MAX,
        (AggKind::Min, AggValueField::F64(_)) => f64::INFINITY.to_bits() as i64,
        (AggKind::Max, AggValueField::I64(_)) => i64::MIN,
        (AggKind::Max, AggValueField::F64(_)) => f64::NEG_INFINITY.to_bits() as i64,
        _ => 0,
    }
}

/// The identity record: `stride/8` `u64` words a group starts at (key `0`, each
/// value field at its identity, counts `0`). The host fills every group slot with
/// this before the fold (see `group::GroupState::new`). All fields are 8-byte
/// `i64`/`f64` cells, so every offset is a whole word.
pub fn group_template(aggs: &[Expr], agg_tys: &[DataType], out_schema: &SchemaRef) -> Vec<u64> {
    let n_keys = out_schema.fields().len() - agg_tys.len();
    let key_spec = KeySpec::from_schema(out_schema, n_keys);
    let record = group_record(aggs, agg_tys, &key_spec);
    let words = record.layout.stride() / 8;
    let mut template = vec![0u64; words];
    for a in &record.aggs {
        template[a.value.offset() / 8] = value_init_bits(a.kind, &a.value) as u64;
    }
    template
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

/// An aggregate resolved to its record-field tokens — the fold/emit form used
/// inside [`gen_grouped`].
#[derive(Clone)]
struct ResolvedAgg {
    kind: AggKind,
    value: AggValueField,
    count: Option<FieldId<i64>>,
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
        rec: DynamicRecord,
        row: &Row,
        schema: &SchemaRef,
        cg_ctx: &CodegenCtx,
    ) {
        let (val_cv, valid) = agg_arg_value(&self.arg, ctx, row, schema, cg_ctx);
        let kind = self.kind;
        let count = self.count;
        match (kind, self.value) {
            (AggKind::Count, AggValueField::I64(f)) => ctx.if_then(valid, move |ctx| {
                let cur = rec.get(ctx, f);
                let next = ctx.bind(add(cur, 1i64));
                rec.set(ctx, f, next);
            }),
            // sum/min/max fold in the value cell's type: `i64` for integer inputs
            // (i32 widened), `f64` for a `Float64` column.
            (AggKind::Sum | AggKind::Min | AggKind::Max, AggValueField::I64(f)) => {
                let v = to_i64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    bump_count(ctx, count, rec);
                    let cur = rec.get(ctx, f);
                    let next = combine_i64(ctx, kind, cur, v);
                    rec.set(ctx, f, next);
                });
            }
            (AggKind::Sum | AggKind::Min | AggKind::Max, AggValueField::F64(f)) => {
                let v = to_f64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    bump_count(ctx, count, rec);
                    let cur = rec.get(ctx, f);
                    let next = combine_f64(ctx, kind, cur, v);
                    rec.set(ctx, f, next);
                });
            }
            (AggKind::Avg, AggValueField::F64(f)) => {
                let v = to_f64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    bump_count(ctx, count, rec);
                    let cur = rec.get(ctx, f);
                    let next = ctx.bind(add(cur, v));
                    rec.set(ctx, f, next);
                });
            }
            (k, _) => unreachable!("aggregate {k:?} has a mismatched value-field type"),
        }
    }

    /// Manifest this aggregate's final `ColVal` from the group's record `rec`:
    /// `count` is never null; `sum`/`min`/`max` are null when the count is 0; `avg`
    /// divides `sum/count` (null when count 0).
    fn finalize(&self, ctx: &mut Ctx, rec: DynamicRecord) -> ColVal {
        match (self.kind, self.value) {
            (AggKind::Count, AggValueField::I64(f)) => {
                ColVal::I64(rec.get(ctx, f), Nullness::NonNull)
            }
            (AggKind::Sum | AggKind::Min | AggKind::Max, AggValueField::I64(f)) => {
                ColVal::I64(rec.get(ctx, f), Nullness::Nullable(self.seen(ctx, rec)))
            }
            (AggKind::Sum | AggKind::Min | AggKind::Max, AggValueField::F64(f)) => {
                ColVal::F64(rec.get(ctx, f), Nullness::Nullable(self.seen(ctx, rec)))
            }
            (AggKind::Avg, AggValueField::F64(f)) => {
                let sum = rec.get(ctx, f);
                let count = rec.get(ctx, self.count.unwrap());
                let cf = ctx.bind(int_to_float::<f64, i64, _>(count));
                let avg = ctx.bind(div(sum, cf));
                let seen = ctx.bind(gt(count, 0i64));
                ColVal::F64(avg, Nullness::Nullable(seen))
            }
            (k, _) => unreachable!("aggregate {k:?} has a mismatched value-field type"),
        }
    }

    /// Whether group `rec` saw any non-null input (its non-null count > 0).
    fn seen(&self, ctx: &mut Ctx, rec: DynamicRecord) -> Var<bool> {
        let count = rec.get(ctx, self.count.expect("nullable agg has a count"));
        ctx.bind(gt(count, 0i64))
    }
}

/// `rec.count += 1` (the non-null count field for a nullable aggregate).
fn bump_count(ctx: &mut Ctx, count: Option<FieldId<i64>>, rec: DynamicRecord) {
    let f = count.expect("nullable aggregate has a count field");
    let cur = rec.get(ctx, f);
    let next = ctx.bind(add(cur, 1i64));
    rec.set(ctx, f, next);
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

/// The staged batch a [`for_each_batch`] body receives: `&[FfiArray]` rebuilt from
/// the stream's current descriptor pointer (a `BatchSource`).
type ScanBatch = Var<SRef<'static, Slice<FfiArray>>>;

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

    ctx.while_loop(Const::<bool>::new(true), move |ctx| {
        // Pull the next batch; a null descriptor pointer means the stream is done.
        let descs = ctx.bind(call_extern2(scan_next, inputs, Const::<u64>::new(table)));
        ctx.if_then(ptr_is_null(descs), |ctx| ctx.break_loop());
        // Rebuild the borrowed batch (`&[FfiArray]`) from (ptr, column count).
        let batch = ctx.bind(slice_ref_from_raw_parts::<FfiArray, _, _>(
            descs,
            Const::<u64>::new(ncols),
        ));
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

/// Append column `c`'s value to its baked output buffer (see [`crate::output`]).
/// Fixed-width columns `SVec::push` the value inline (and, if nullable, push a
/// `bool` validity flag to a parallel `SVec` so the two stay length-aligned);
/// string columns append through the builder. One append per emitted row, in
/// order, so every column ends at the same length.
fn write_col(ctx: &mut Ctx, c: usize, field: &Field, cv: ColVal, cx: &CodegenCtx) {
    match &cx.out.cols[c] {
        OutColHandle::Str { builder } => write_str_col(ctx, *builder, cv, cx),
        OutColHandle::Fixed { values, validity } => {
            dispatch_prim!(field.data_type(), M => {
                let vals = SVec::<M>::new(*values, cx.rt.svec_grow);
                let v = M::coerce(ctx, cv);
                vals.push(ctx, v);
                // A nullable column pushes a validity flag for *every* row (the
                // value is pushed unconditionally, so no branch is needed).
                if let Some(valid_ctrl) = validity {
                    let flag = match cv.nullness() {
                        Nullness::NonNull => ctx.bind(Const::<bool>::new(true)),
                        Nullness::Nullable(valid) => valid,
                    };
                    SVec::<bool>::new(*valid_ctrl, cx.rt.svec_grow).push(ctx, flag);
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
        ctx.emit(call_extern2(
            append,
            const_opaque_mut::<StringViewBuilder>(builder),
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
                    const_opaque_mut::<StringViewBuilder>(builder),
                ));
            });
        }
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
        // An alias is a pure rename: evaluate the inner expression. The output
        // schema already carries the alias name, so nothing else is needed.
        Expr::Alias(a) => gen_expr(ctx, &a.expr, schema, row, cx),
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

fn as_str(cv: ColVal) -> StrVal {
    match cv {
        ColVal::Str(sv, _) => sv,
        other => panic!("expected string group key, got {}", tag(other)),
    }
}

/// Canonicalize an `f64` GROUP BY key before bit-keying it, so bit-equality matches
/// SQL float grouping: map `-0.0` to `+0.0` (they compare equal) and every NaN to one
/// canonical NaN (NaN ≠ NaN, so distinct payloads would otherwise split). Branchless.
fn canonical_f64(ctx: &mut Ctx, key: Var<f64>) -> Var<f64> {
    let is_zero = ctx.bind(eq(key, 0.0f64));
    let no_neg_zero = ctx.bind(select(is_zero, 0.0f64, key));
    let is_number = ctx.bind(eq(no_neg_zero, no_neg_zero)); // false only for NaN
    ctx.bind(select(is_number, no_neg_zero, f64::NAN))
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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum AggKind {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

/// A `count` aggregate that qualifies for the whole-batch shortcut (see [`gen_op`]).
#[derive(Clone, Copy)]
enum CountFast {
    /// `count(*)` — counts rows: `+= batch_len`.
    Star,
    /// `count(col)` — counts non-nulls of column `col`: `+= batch_len - null_count`.
    Col(usize),
}

/// Classify `expr` as a whole-batch-countable `count`, resolving a `count(col)`
/// argument to its column index in `scan_schema`. Returns `None` for anything that
/// needs per-row work: non-`count` aggregates, `DISTINCT`, a `FILTER (WHERE …)`, or
/// a computed argument like `count(a + b)`.
#[allow(deprecated)] // `Expr::Wildcard` is deprecated but still how `*` arrives.
fn count_fast(expr: &Expr, scan_schema: &SchemaRef) -> Option<CountFast> {
    let Expr::AggregateFunction(af) = expr else {
        return None;
    };
    if !af.func.name().eq_ignore_ascii_case("count") {
        return None;
    }
    if af.params.distinct || af.params.filter.is_some() || af.params.args.len() != 1 {
        return None;
    }
    match &af.params.args[0] {
        Expr::Wildcard { .. } => Some(CountFast::Star),
        Expr::Column(c) => scan_schema.index_of(&c.name).ok().map(CountFast::Col),
        _ => None,
    }
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
