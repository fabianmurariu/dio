//! Hash-join codegen — both sides JIT-ed:
//! - **Bare-scan build** ([`gen_build_index`]): the input RBs are Arc-cloned into
//!   the relation host-side; this kernel iterates them, reads each row's key with
//!   its known type (no dispatch) and `join_insert`s its [`Locator`].
//! - **Materialized build** ([`gen_build`]): a filtered/projected build subtree —
//!   ONE fused pass that writes each surviving row's columns into the relation *and*
//!   `join_insert`s its key+position, exactly the paper's `left.exec { rec => hm +=
//!   (lkey(rec), rec) }`.
//! - **Probe** ([`gen_join`]): per right row, compute the key, walk the matching
//!   `Locator`s, `gen_read` the located left columns, and emit `[left | right]`.
//!
//! Phase 1: inner join, single `Int` key.

use arrow::datatypes::{Field, SchemaRef};
use arrow_lms::FfiArray;
use datafusion_expr::Expr;
use rust_lms::prelude::*;

use crate::join::{JoinState, LocatorType};
use crate::plan::Operator;
use crate::value::{Nullness, Row};

use super::expr::gen_expr;
use super::numeric::to_i64;
use super::{CodegenCtx, InputsSource, Yld, gen_len, gen_op, gen_read, write_col};

/// Insert the current build row's key + `Locator(rb_pos, row)` into the host
/// multimap (the JIT proxy for `hm += (key, rec)`). `key_cv` is the row's key
/// value; a NULL key is skipped (it never matches).
fn emit_key_insert(
    ctx: &mut Ctx,
    state: *mut JoinState,
    rt: crate::runtime::Runtime,
    key_cv: crate::value::ColVal,
    rb_pos: Var<u32>,
    row: Var<u32>,
) {
    let key_i64 = to_i64(ctx, key_cv);
    let key = ctx.bind(int_cast::<u64, i64, _>(key_i64));
    let insert = move |ctx: &mut Ctx| {
        ctx.emit(call_extern4(
            rt.join_insert,
            const_opaque_mut::<JoinState>(state),
            key,
            rb_pos,
            row,
        ));
    };
    match key_cv.nullness() {
        Nullness::NonNull => insert(ctx),
        Nullness::Nullable(valid) => ctx.if_then(valid, insert),
    }
}

/// Bare-scan build index kernel: iterate the (host-cloned) relation and, per
/// non-null row, read the `Int` key column (typed) and `join_insert` its
/// `Locator(batch_idx, row)`.
pub(crate) fn gen_build_index(
    ctx: &mut Ctx,
    rel_schema: SchemaRef,
    key_col: usize,
    cx: &CodegenCtx,
) {
    let handle = cx
        .join
        .clone()
        .expect("build index without a baked JoinState");
    let state = handle.state;
    let rt = cx.rt;
    let ncols = rel_schema.fields().len() as u64;
    let key_field: Field = rel_schema.field(key_col).clone();
    let key_dt = key_field.data_type().clone();

    let count = ctx.bind(call_extern1(
        rt.join_rel_count,
        const_opaque::<JoinState>(state),
    ));
    let b = ctx.var(0u64);
    ctx.while_loop(lt(b, count), move |ctx| {
        let descs = ctx.bind(call_extern2(
            rt.join_left_batch,
            const_opaque::<JoinState>(state),
            b,
        ));
        let batch = ctx.bind(slice_ref_from_raw_parts::<FfiArray, _, _>(
            descs,
            Const::<u64>::new(ncols),
        ));
        let len = gen_len(ctx, batch, &key_dt);
        let row = ctx.var(0u64);
        ctx.store(row, 0u64);
        let key_field = key_field.clone();
        ctx.while_loop(lt(row, len), move |ctx| {
            let key_cv = gen_read(ctx, batch, key_col, &key_field, row);
            let rb_pos = ctx.bind(int_cast::<u32, u64, _>(b));
            let row_u32 = ctx.bind(int_cast::<u32, u64, _>(row));
            emit_key_insert(ctx, state, rt, key_cv, rb_pos, row_u32);
            ctx.store(row, add(row, 1u64));
        });
        ctx.store(b, add(b, 1u64));
    });
}

/// Materialized build kernel — ONE fused pass. Runs the build subtree; per
/// surviving row it (a) writes the row's columns into the build relation
/// (`write_col`) and (b) `join_insert`s its key + position `Locator(0, n)` (the
/// relation is a single batch, so `rb_pos == 0`). Returns the row count `n` (for
/// finalizing the relation's `RecordBatch`).
pub(crate) fn gen_build<I: InputsSource>(
    ctx: &mut Ctx,
    inputs: I,
    build: &Operator,
    key_expr: &Expr,
    out_schema: &SchemaRef,
    cx: &CodegenCtx,
) -> Var<u64> {
    let handle = cx
        .join
        .clone()
        .expect("fused build without a baked JoinState");
    let state = handle.state;
    let rt = cx.rt;
    let n = ctx.var(0u64);
    let fields = out_schema.fields().clone();
    let build_schema = build.output_schema();
    let key_expr = key_expr.clone();
    let cx_c = cx.clone();
    gen_op(
        build,
        ctx,
        inputs,
        cx,
        Box::new(move |ctx, row: Row| {
            // (a) materialize the row into the relation's columns.
            for (c, (cv, field)) in row.iter().zip(fields.iter()).enumerate() {
                write_col(ctx, c, field, *cv, &cx_c);
            }
            // (b) index: key from the row + Locator(rb_pos=0, row=n).
            let key_cv = gen_expr(ctx, &key_expr, &build_schema, &row, &cx_c);
            let rb0 = ctx.bind(Const::<u32>::new(0));
            let row_u32 = ctx.bind(int_cast::<u32, u64, _>(n));
            emit_key_insert(ctx, state, rt, key_cv, rb0, row_u32);
            ctx.store(n, add(n, 1u64));
        }),
    );
    n
}

/// Emit the probe. `op` is the `Join`; its left side is already built, so we only
/// codegen `gen_op(right, …)` with a per-row match loop.
pub(crate) fn gen_join<I: InputsSource>(
    ctx: &mut Ctx,
    inputs: I,
    op: &Operator,
    cx: &CodegenCtx,
    yld: Yld,
) {
    let Operator::Join {
        left, right, on, ..
    } = op
    else {
        unreachable!("gen_join called on a non-Join operator")
    };
    let handle = cx
        .join
        .clone()
        .expect("hash join without a baked JoinState");
    let state = handle.state;
    let left_schema = left.output_schema();
    let ncols_left = left_schema.fields().len();
    let right_key = on[0].1.clone();
    let right_schema = right.output_schema();
    let rt = cx.rt;
    let cx_c = cx.clone();

    gen_op(
        right,
        ctx,
        inputs,
        cx,
        Box::new(move |ctx, right_row: Row| {
            // Key of the probe row, as the same `u64` the build side indexed
            // (`i64 as u64` reinterpret — negatives keep their bits both sides).
            let key_cv = gen_expr(ctx, &right_key, &right_schema, &right_row, &cx_c);
            let key_i64 = to_i64(ctx, key_cv);
            let key = ctx.bind(int_cast::<u64, i64, _>(key_i64));

            // The per-match loop: look up the key's `Locator` run, then for each
            // read `.rb_pos`/`.row`, `gen_read` the located left row, emit `[l | r]`.
            let probe = move |ctx: &mut Ctx| {
                let count = ctx.bind(call_extern2(
                    rt.join_probe_count,
                    const_opaque::<JoinState>(state),
                    key,
                ));
                let base = ctx.bind(call_extern2(
                    rt.join_probe_base,
                    const_opaque::<JoinState>(state),
                    key,
                ));
                let i = ctx.var(0u64);
                ctx.while_loop(lt(i, count), move |ctx| {
                    let loc = ctx.bind(ptr_offset(base, int_cast::<i64, u64, _>(i)));
                    let rb_pos = ctx.bind(load_field(loc, LocatorType::rb_pos()));
                    let row_u32 = ctx.bind(load_field(loc, LocatorType::row()));
                    let b = ctx.bind(int_cast::<u64, u32, _>(rb_pos));
                    let r = ctx.bind(int_cast::<u64, u32, _>(row_u32));
                    let descs = ctx.bind(call_extern2(
                        rt.join_left_batch,
                        const_opaque::<JoinState>(state),
                        b,
                    ));
                    let left_batch = ctx.bind(slice_ref_from_raw_parts::<FfiArray, _, _>(
                        descs,
                        Const::<u64>::new(ncols_left as u64),
                    ));
                    let mut row: Row = (0..ncols_left)
                        .map(|c| gen_read(ctx, left_batch, c, left_schema.field(c), r))
                        .collect();
                    row.extend(right_row.iter().copied());
                    yld(ctx, row);
                    ctx.store(i, add(i, 1u64));
                });
            };

            // A NULL join key never matches — skip the probe on a null key.
            match key_cv.nullness() {
                Nullness::NonNull => probe(ctx),
                Nullness::Nullable(valid) => ctx.if_then(valid, probe),
            }
        }),
    );
}
