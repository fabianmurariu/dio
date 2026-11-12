//! GROUP BY: fold each input row into the Rust-hosted group state, then emit one
//! manifested row per group. The packed group-record layout, the single/composite
//! key handling, and the grouped-aggregate fold/finalize all live here.

use arrow::datatypes::{DataType, SchemaRef};
use datafusion_expr::Expr;
use rust_lms::prelude::*;
use rust_lms_std::{DynamicRecord, FieldId, RecordLayout};

use crate::group::{GroupState, KeyKind};
use crate::plan::Operator;
use crate::value::{ColVal, Nullness, Row};

use super::aggregate::AggKind;
use super::expr::gen_expr;
use super::numeric::{canonical_f64, coerce_f64, to_f64, to_i64};
use super::strings::as_str;
use super::{CodegenCtx, InputsSource, Yld, gen_op};

fn group_ref(state: *mut GroupState) -> impl Staged<Out = SPtr<Opaque<GroupState>>> + Copy {
    const_ptr::<Opaque<GroupState>>(state)
}

fn group_mut(state: *mut GroupState) -> impl Staged<Out = SMutPtr<Opaque<GroupState>>> + Copy {
    const_mut_ptr::<Opaque<GroupState>>(state)
}

/// GROUP BY as a push operator: fold each input row into the Rust-hosted group
/// state (baked buffers indexed by `gidx`), then emit one *manifested* Row per
/// group downstream. Keeps the hot fold loop in the JIT and lets the projection /
/// filter above it run in the same kernel (see [`CodegenCtx::group`]). `avg` divides at
/// emit; nullability rides in the emitted `ColVal` (so `write_col` handles nulls).
pub(crate) fn gen_grouped<I: InputsSource>(
    ctx: &Ctx,
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
    // How the fold computes the key: one expression (an integer or float key) or a
    // packed layout over every key column.
    let key_source = match key_spec {
        KeySpec::Single { .. } => KeySource::Single(group_exprs[0].clone()),
        KeySpec::Packed(packed) => KeySource::Packed(packed, group_exprs.clone()),
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
                KeySource::Packed(packed, exprs) => {
                    pack_and_upsert(ctx, packed, exprs, &row, &schema_f, &cx_c, state)
                }
            };
            // A null record pointer is the upsert's error sentinel (e.g. group
            // count exceeded `u32`): poison the run and break so we never fold
            // through null. The driver surfaces the recorded error.
            super::stop_if_null(ctx, rec_ptr, &cx_c);
            // SAFETY: each upsert returns a record allocated with `layout` that
            // remains live until the next upsert; folding performs no upsert.
            let rec = unsafe { layout.wrap(rec_ptr) };
            for agg in &resolved_f {
                agg.fold(ctx, rec, &row, &schema_f, &cx_c);
            }
        }),
    );

    // Emit: one manifested Row per group, pushed downstream in the same kernel. The
    // records buffer is fully grown now, so its base is stable — fetch it once.
    // SAFETY: `run_kernel` retains `state`; the fold is complete, so these
    // shared/mutable calls are sequenced without host or generated aliases.
    let num_groups = ctx.bind(unsafe { call_extern1_unchecked(cx.rt.group_len, group_ref(state)) });
    let base =
        ctx.bind(unsafe { call_extern1_unchecked(cx.rt.group_records_base, group_mut(state)) });
    // The interned strings' table, which a packed key's string views index.
    let strings =
        ctx.bind(unsafe { call_extern1_unchecked(cx.rt.group_strings, group_ref(state)) });
    range(0u64, num_groups).for_each(ctx, move |ctx, g| {
        // SAFETY: `g < num_groups`, and `base` addresses the stable records
        // buffer after the fold has completed.
        let rec = unsafe { layout.record(ctx, base, g) };
        let key_cols = match &key_source {
            KeySource::Single(_) => {
                vec![emit_single_key(ctx, key_fields, key_valid, rec)]
            }
            KeySource::Packed(packed, _) => unpack_packed(ctx, packed, rec, strings),
        };
        let mut row: Row = Vec::with_capacity(n_keys + resolved.len());
        row.extend(key_cols);
        for agg in &resolved {
            row.push(agg.finalize(ctx, rec));
        }
        yld(ctx, row);
    });
}

/// Fold-time upsert for a **single** integer or float key: dispatch on the key type,
/// then finish (null handling) via [`finish_upsert`]. The key computation runs
/// unconditionally but is harmless on a null row (its value is unused).
fn fold_single_key(
    ctx: &Ctx,
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
            // SAFETY: `run_kernel` retains and exclusively lends `state` to
            // generated calls, which are sequenced by the emitted program.
            let call =
                unsafe { call_extern2_unchecked(cx.rt.group_upsert, group_mut(state), key_bits) };
            finish_upsert(ctx, call, key_valid, key_cv, state, cx, layout)
        }
        KeyFields::Float(_) => {
            // Key on the f64's bits: canonicalize (`-0.0`/NaN), then bitcast.
            let key = coerce_f64(ctx, key_cv);
            let canon = canonical_f64(ctx, key);
            let key_bits = ctx.bind(bitcast::<u64, f64, _>(canon));
            // SAFETY: as above, this call has exclusive access to live state.
            let call =
                unsafe { call_extern2_unchecked(cx.rt.group_upsert, group_mut(state), key_bits) };
            finish_upsert(ctx, call, key_valid, key_cv, state, cx, layout)
        }
        KeyFields::Packed => unreachable!("a packed key in the single-key fold"),
    }
}

/// Emit the single key `ColVal` for one group's record.
fn emit_single_key(
    ctx: &Ctx,
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
        KeyFields::Packed => unreachable!("a packed key in the single-key emit"),
    }
}

/// Complete a fold-row's upsert: for a non-nullable key, just run `valid_call` (the
/// key-specific `group_upsert*`). For a nullable key, branch on the key's validity —
/// a real key runs `valid_call`, a NULL key routes to `group_upsert_null` — and record
/// the validity (`1`/`0`) in the record's key-valid cell for emit. `valid_call`'s key
/// computation has already run in the caller (harmless on a null row); only the extern
/// call is guarded by the branch.
fn finish_upsert<C>(
    ctx: &Ctx,
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
            // SAFETY: `state` remains live and is exclusively accessed by the
            // selected generated branch.
            let null_call =
                unsafe { call_extern1_unchecked(cx.rt.group_upsert_null, group_mut(state)) };
            let rec_ptr = ctx.bind(if_then_else(valid, valid_call, null_call));
            let valid_bits = ctx.bind(select(valid, 1i64, 0i64));
            // SAFETY: both upsert branches return a writable record allocated
            // with `layout` and no further upsert occurs before this write.
            unsafe { layout.wrap(rec_ptr) }.set(ctx, valid_field, valid_bits);
            rec_ptr
        }
    }
}

/// How the fold obtains the key: a single key expression, or a packed layout over all
/// key columns and their expressions, in order. Cloned into the fold closure and kept
/// for emit.
#[derive(Clone)]
enum KeySource {
    Single(Expr),
    Packed(PackedKey, Vec<Expr>),
}

/// A packed key's layout: one cell per key column — 8 bytes for an `i64`/`f64`, a
/// string's 16-byte view as two `u64` halves — plus a trailing `u64` null bitmap (bit
/// `i` = column `i` is NULL). Strings are interned, so equal keys have equal bytes. The
/// fold writes the canonicalized values + bitmap into a stack scratch of this shape —
/// the bytes `group_upsert_packed` hashes and stores at the head of the group's record
/// — and emit reads them back from the record, whose layout starts as this one.
#[derive(Clone)]
struct PackedKey {
    layout: RecordLayout,
    cols: Vec<PackedCol>,
    bitmap: FieldId<u64>,
}

#[derive(Clone)]
struct PackedCol {
    field: PackedColField,
    /// Null-bitmap bit index (= column position).
    bit: usize,
    nullable: bool,
}

#[derive(Clone, Copy)]
enum PackedColField {
    I64(FieldId<i64>),
    F64(FieldId<f64>),
    /// A string's interned view, low and high halves.
    Str {
        lo: FieldId<u64>,
        hi: FieldId<u64>,
    },
}

impl PackedKey {
    /// The layout for key columns `cols`, `(type, nullable)` each, in order.
    fn new(cols: &[(DataType, bool)]) -> Self {
        let mut layout = RecordLayout::new();
        let cols = cols
            .iter()
            .enumerate()
            .map(|(bit, (ty, nullable))| {
                let field = match ty {
                    DataType::Int32 | DataType::Int64 => PackedColField::I64(layout.field()),
                    DataType::Float64 => PackedColField::F64(layout.field()),
                    DataType::Utf8View => PackedColField::Str {
                        lo: layout.field(),
                        hi: layout.field(),
                    },
                    other => panic!("unsupported GROUP BY key column type: {other}"),
                };
                PackedCol {
                    field,
                    bit,
                    nullable: *nullable,
                }
            })
            .collect();
        let bitmap = layout.field::<u64>();
        PackedKey {
            layout,
            cols,
            bitmap,
        }
    }

    /// The key's length in bytes.
    fn len(&self) -> usize {
        self.layout.stride()
    }
}

/// Pack the key columns into a fresh stack scratch (canonicalized values, interned
/// string views, null bitmap), then find-or-insert via `group_upsert_packed`, which
/// copies the bytes into a new group's record. Returns the group's record pointer.
fn pack_and_upsert(
    ctx: &Ctx,
    packed: &PackedKey,
    exprs: &[Expr],
    row: &Row,
    schema: &SchemaRef,
    cx: &CodegenCtx,
    state: *mut GroupState,
) -> Var<SMutPtr<u8>> {
    let scratch = ctx.bind(stack_alloc(packed.len()));
    // SAFETY: the stack allocation is exactly one record using `packed.layout`
    // and remains live throughout this generated function.
    let prec = unsafe { packed.layout.wrap(scratch) };
    let bitmap = ctx.var(0u64);
    for (col, expr) in packed.cols.iter().zip(exprs) {
        let cv = gen_expr(ctx, expr, schema, row, cx);
        let valid = match cv.nullness() {
            Nullness::Nullable(b) => Some(b),
            Nullness::NonNull => None,
        };
        // NULL columns pack a canonical value (0 / +0.0 / the empty string); the bitmap
        // distinguishes them.
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
            PackedColField::Str { lo, hi } => {
                // A null row's value is the empty string already (`gen_read`).
                let view = intern(ctx, as_str(cv), state, cx);
                prec.set(ctx, lo, ctx.bind(int_cast::<u64, u128, _>(view)));
                prec.set(
                    ctx,
                    hi,
                    ctx.bind(int_cast::<u64, u128, _>(shr(view, 64u128))),
                );
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
    // SAFETY: `scratch` holds exactly one initialized packed key of `packed.len()`
    // bytes, alive for the call.
    let key = unsafe {
        slice_from_raw_parts::<u8, _, _>(
            ptr_as_const(scratch),
            Const::<u64>::new(packed.len() as u64),
        )
    };
    // SAFETY: state is exclusively available to this call.
    ctx.bind(unsafe { call_extern2_unchecked(cx.rt.group_upsert_packed, group_mut(state), key) })
}

/// The interned view of `s`: its own view when it is short, else the one
/// `group_intern` gives every equal string.
fn intern(ctx: &Ctx, s: Var<Str>, state: *mut GroupState, cx: &CodegenCtx) -> Var<u128> {
    let slot = ctx.bind(stack_array::<u128>(1));
    // SAFETY: the slot holds one `u128`.
    ctx.emit(unsafe { store(slot, s.view()) });
    let intern = cx.rt.group_intern;
    ctx.if_then(not(s.is_inline()), move |ctx| {
        let bytes = s.as_slice(ctx);
        // SAFETY: state is exclusively available; `bytes` are the string's, alive for
        // the call, and `slot` is a `u128` on this kernel's stack.
        ctx.emit(unsafe { call_extern3_unchecked(intern, group_mut(state), bytes, slot) });
    });
    // SAFETY: the slot was written above.
    ctx.bind(unsafe { load_mut(slot) })
}

/// A group's key columns, read from the head of its record (nulls from the bitmap; a
/// string's view against the interned strings' table).
fn unpack_packed(
    ctx: &Ctx,
    packed: &PackedKey,
    rec: DynamicRecord,
    strings: Var<SPtr<ByteBuffer>>,
) -> Vec<ColVal> {
    let bitmap = rec.get(ctx, packed.bitmap);
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
                PackedColField::I64(f) => ColVal::I64(rec.get(ctx, f), null),
                PackedColField::F64(f) => ColVal::F64(rec.get(ctx, f), null),
                PackedColField::Str { lo, hi } => {
                    let lo = int_cast::<u128, u64, _>(rec.get(ctx, lo));
                    let hi = int_cast::<u128, u64, _>(rec.get(ctx, hi));
                    let view = ctx.bind(bitor(lo, shl(hi, 64u128)));
                    // SAFETY: the view is one `group_intern` returned (or a short one),
                    // and the group state keeps the interned strings, unchanged, while
                    // groups are emitted.
                    let s = ctx.bind(unsafe { ByteView::<Utf8>::from_parts(view, strings) });
                    ColVal::Str(s, null)
                }
            }
        })
        .collect()
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

/// The group key's field in the record, its leading one — the `group_upsert*` externs
/// write the key there.
#[derive(Clone, Copy)]
enum KeyFields {
    Int(FieldId<i64>),
    /// A `Float64` key. Stored (by the `Int` upsert extern) as its `u64` bits in the
    /// leading cell; read back as `f64` at emit (the token reinterprets those bytes).
    Float(FieldId<f64>),
    /// A packed key: the record's layout starts as the key's (see [`PackedKey`]), and
    /// nulls ride in its bitmap, not a `key_valid` cell, so each `(a, b, …)` combination
    /// is its own group.
    Packed,
}

/// What the GROUP BY key is: one integer or float column (the `u64`-keyed table, a
/// separate null group), or a packed key over every key column — strings, several
/// columns.
enum KeySpec {
    Single { ty: DataType, nullable: bool },
    Packed(PackedKey),
}

impl KeySpec {
    /// Derive the key spec from the aggregate's output schema (its first `n_keys`
    /// fields are the group keys).
    fn from_schema(out_schema: &SchemaRef, n_keys: usize) -> KeySpec {
        let cols: Vec<(DataType, bool)> = out_schema.fields()[..n_keys]
            .iter()
            .map(|f| (f.data_type().clone(), f.is_nullable()))
            .collect();
        match cols.as_slice() {
            [(ty, nullable)] if *ty != DataType::Utf8View => KeySpec::Single {
                ty: ty.clone(),
                nullable: *nullable,
            },
            _ => KeySpec::Packed(PackedKey::new(&cols)),
        }
    }
}

/// The table a GROUP BY with this output schema and `n_keys` keys uses.
pub fn group_key_kind(out_schema: &SchemaRef, n_keys: usize) -> KeyKind {
    match KeySpec::from_schema(out_schema, n_keys) {
        KeySpec::Single {
            ty: DataType::Float64,
            ..
        } => KeyKind::Float,
        KeySpec::Single { .. } => KeyKind::Int,
        KeySpec::Packed(packed) => KeyKind::Packed(packed.len()),
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
    /// A key-validity cell — `Some` only when a single key column is **nullable**. The
    /// fold stores `1` for a real key and `0` for the null group; emit reads it to
    /// decide whether the key is NULL. Absent (and zero overhead) otherwise.
    key_valid: Option<FieldId<i64>>,
    aggs: Vec<AggFields>,
}

/// Reserve the leading key field for a single integer or float key of type `key_ty`.
/// Must be first (offset 0) — the `group_upsert` extern writes the key there.
fn key_fields_single(layout: &mut RecordLayout, key_ty: &DataType) -> KeyFields {
    match key_ty {
        DataType::Int32 | DataType::Int64 => KeyFields::Int(layout.field::<i64>()),
        DataType::Float64 => KeyFields::Float(layout.field::<f64>()),
        other => panic!("unsupported GROUP BY key type: {other}"),
    }
}

/// Build the packed-record layout from the key spec and the aggregates + their output
/// `DataType`s (parallel to `aggs`). The output type picks the value cell ([`acc_ty`]);
/// `sum`/`min`/`max` over `Float64` fold in an `f64` field, everything else `i64`.
fn group_record(aggs: &[Expr], agg_tys: &[DataType], key_spec: &KeySpec) -> GroupRecord {
    let (mut layout, key, key_valid) = match key_spec {
        KeySpec::Single { ty, nullable } => {
            let mut layout = RecordLayout::new();
            let key = key_fields_single(&mut layout, ty);
            // A nullable single key routes NULLs to the null group via a validity cell.
            let key_valid = nullable.then(|| layout.field::<i64>());
            (layout, key, key_valid)
        }
        // The key's own fields lead the record; nulls are in its bitmap.
        KeySpec::Packed(packed) => (packed.layout, KeyFields::Packed, None),
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
    ctx: &Ctx,
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
        ctx: &Ctx,
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
    fn finalize(&self, ctx: &Ctx, rec: DynamicRecord) -> ColVal {
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
    fn seen(&self, ctx: &Ctx, rec: DynamicRecord) -> Var<bool> {
        let count = rec.get(ctx, self.count.expect("nullable agg has a count"));
        ctx.bind(gt(count, 0i64))
    }
}

/// `rec.count += 1` (the non-null count field for a nullable aggregate).
fn bump_count(ctx: &Ctx, count: Option<FieldId<i64>>, rec: DynamicRecord) {
    let f = count.expect("nullable aggregate has a count field");
    let cur = rec.get(ctx, f);
    let next = ctx.bind(add(cur, 1i64));
    rec.set(ctx, f, next);
}

/// Fold one `i64` input into a `sum`/`min`/`max` accumulator (`add`/`min`/`max`).
/// The arms return distinct staged types, so each binds to a `Var` before returning.
fn combine_i64(ctx: &Ctx, kind: AggKind, cur: Var<i64>, v: Var<i64>) -> Var<i64> {
    match kind {
        AggKind::Sum => ctx.bind(add(cur, v)),
        AggKind::Min => ctx.bind(min(cur, v)),
        AggKind::Max => ctx.bind(max(cur, v)),
        _ => unreachable!("combine_i64 is only for sum/min/max"),
    }
}

/// `f64` twin of [`combine_i64`] — for `sum`/`min`/`max` over a `Float64` column.
fn combine_f64(ctx: &Ctx, kind: AggKind, cur: Var<f64>, v: Var<f64>) -> Var<f64> {
    match kind {
        AggKind::Sum => ctx.bind(add(cur, v)),
        AggKind::Min => ctx.bind(min(cur, v)),
        AggKind::Max => ctx.bind(max(cur, v)),
        _ => unreachable!("combine_f64 is only for sum/min/max"),
    }
}
