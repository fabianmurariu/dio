//! Scalar aggregation (no GROUP BY): accumulators folded by `gen_op`'s Aggregate
//! arm, which emits one result row.

use arrow::datatypes::{DataType, Field, SchemaRef};
use datafusion_expr::Expr;
use rust_lms::prelude::*;

use crate::value::{ColVal, Nullness, Row};

use super::CodegenCtx;
use super::expr::gen_expr;
use super::numeric::{coerce_f64, to_f64, to_i64};

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum AggKind {
    Count,
    Sum,
    Min,
    Max,
    Avg,
}

/// A `count` aggregate that qualifies for the whole-batch shortcut (see [`gen_op`]).
#[derive(Clone, Copy)]
pub(crate) enum CountFast {
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
pub(crate) fn count_fast(expr: &Expr, scan_schema: &SchemaRef) -> Option<CountFast> {
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
pub(crate) struct Agg {
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
    pub(crate) fn init(ctx: &mut Ctx, expr: &Expr, out_field: &Field) -> Agg {
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

    pub(crate) fn update(&self, ctx: &mut Ctx, row: &Row, schema: &SchemaRef, cx: &CodegenCtx) {
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
    pub(crate) fn finalize(&self, ctx: &mut Ctx) -> ColVal {
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
