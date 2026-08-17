//! Hash-join codegen — both sides JIT-ed:
//! - [`gen_build_index`] iterates the build relation (host-owned `RecordBatch`es),
//!   reads each row's key column with its *known* type (`gen_read`, no dynamic
//!   dispatch) and calls the `join_insert` proxy — like GROUP BY's fold loop.
//! - [`gen_join`] emits the streaming probe over the right side: per right row,
//!   compute the key, walk the matching build-row locators, `gen_read` the located
//!   left columns, and emit `[left | right]`.
//!
//! Phase 1: inner join, single `Int` key.

use arrow::datatypes::{Field, SchemaRef};
use arrow_lms::FfiArray;
use rust_lms::prelude::*;

use crate::join::JoinState;
use crate::plan::Operator;
use crate::value::{Nullness, Row};

use super::expr::gen_expr;
use super::numeric::to_i64;
use super::{CodegenCtx, InputsSource, Yld, gen_len, gen_op, gen_read};

/// Emit the build **index kernel**: iterate the build relation and, per non-null
/// row, read the `Int` key column (typed — no host dispatch), pack the locator
/// `(batch_idx<<32)|row`, and `join_insert` it into the host multimap. `rel_schema`
/// is the relation's schema; `key_col` the key column's index in it.
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
            let key_i64 = to_i64(ctx, key_cv);
            let key = ctx.bind(int_cast::<u64, i64, _>(key_i64));
            // locator = (batch_idx << 32) | row_idx
            let hi = ctx.bind(shl(b, 32u64));
            let loc = ctx.bind(bitor(hi, row));
            // Skip NULL keys (a NULL join key never matches).
            let insert = move |ctx: &mut Ctx| {
                ctx.emit(call_extern3(
                    rt.join_insert,
                    const_opaque_mut::<JoinState>(state),
                    key,
                    loc,
                ));
            };
            match key_cv.nullness() {
                Nullness::NonNull => insert(ctx),
                Nullness::Nullable(valid) => ctx.if_then(valid, insert),
            }
            ctx.store(row, add(row, 1u64));
        });
        ctx.store(b, add(b, 1u64));
    });
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

            // The per-match loop: look up the key's locator run, then for each
            // locator `gen_read` the located left row and emit `[left | right]`.
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
                    // locator = (batch_idx << 32) | row_idx
                    let loc = ctx.bind(array_index(base, int_cast::<i64, u64, _>(i)));
                    let b = ctx.bind(shr(loc, 32u64));
                    let r = ctx.bind(bitand(loc, 0xFFFF_FFFFu64));
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
