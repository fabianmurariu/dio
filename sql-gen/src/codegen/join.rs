//! Hash-join probe codegen. The left (build) side is materialized + indexed
//! host-side before compile (see `crate::join`); this emits the streaming probe
//! over the right side: per right row, compute the key, walk the matching build-row
//! locators, `gen_read` the located left columns, and emit `[left | right]`.
//!
//! Phase 1: inner join, single `Int` key.

use arrow_lms::FfiArray;
use rust_lms::prelude::*;

use crate::join::JoinState;
use crate::plan::Operator;
use crate::value::{Nullness, Row};

use super::expr::gen_expr;
use super::numeric::to_i64;
use super::{CodegenCtx, InputsSource, Yld, gen_op, gen_read};

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
