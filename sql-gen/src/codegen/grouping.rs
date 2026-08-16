//! GROUP BY: fold each input row into the Rust-hosted group state, then emit one
//! manifested row per group. The packed group-record layout, the single/composite
//! key handling, and the grouped-aggregate fold/finalize all live here.

use arrow::datatypes::{DataType, SchemaRef};
use datafusion_expr::Expr;
use rust_lms::prelude::*;
use rust_lms_std::{DynamicRecord, FieldId, RecordLayout};

use crate::group::GroupState;
use crate::plan::Operator;
use crate::value::{ColVal, Nullness, Row, StrVal};

use super::aggregate::AggKind;
use super::expr::gen_expr;
use super::numeric::{canonical_f64, coerce_f64, to_f64, to_i64};
use super::strings::{as_str, resolve};
use super::{CodegenCtx, InputsSource, Yld, gen_op};

/// GROUP BY as a push operator: fold each input row into the Rust-hosted group
/// state (baked buffers indexed by `gidx`), then emit one *manifested* Row per
/// group downstream. Keeps the hot fold loop in the JIT and lets the projection /
/// filter above it run in the same kernel (see [`CodegenCtx::group`]). `avg` divides at
/// emit; nullability rides in the emitted `ColVal` (so `write_col` handles nulls).
pub(crate) fn gen_grouped<I: InputsSource>(
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
