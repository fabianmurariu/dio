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

use arrow::datatypes::{DataType, Field, SchemaRef};
use arrow_lms::{ArrayBatchOps, FfiArray, MutBatchOps, PrimitiveArrayView};
use datafusion_common::ScalarValue;
use datafusion_expr::{BinaryExpr, Expr, Operator as DfOp};
use rust_lms::prelude::*;

use crate::plan::Operator;
use crate::value::{ColVal, Nullness, Row};

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
) -> Var<u64> {
    let n = ctx.var(0u64);
    let fields = out_schema.fields().clone();
    gen_op(
        plan,
        ctx,
        batch,
        Box::new(move |ctx, row| {
            for (c, (cv, field)) in row.iter().zip(fields.iter()).enumerate() {
                write_col(ctx, out, c, field, n, *cv);
            }
            ctx.store(n, add(n, 1u64));
        }),
    );
    n
}

fn gen_op<B: BatchSource>(op: &Operator, ctx: &mut Ctx, batch: B, yld: Yld) {
    match op {
        Operator::Scan { schema } => gen_scan(ctx, batch, schema.clone(), yld),

        Operator::Filter { predicate, input } => {
            let predicate = predicate.clone();
            let schema = input.output_schema();
            gen_op(
                input,
                ctx,
                batch,
                Box::new(move |ctx, row| {
                    let keep = gen_predicate(ctx, &predicate, &schema, &row);
                    ctx.if_then(keep, move |ctx| yld(ctx, row));
                }),
            );
        }

        Operator::Project { exprs, input, .. } => {
            let exprs = exprs.clone();
            let schema = input.output_schema();
            gen_op(
                input,
                ctx,
                batch,
                Box::new(move |ctx, row| {
                    let projected: Row = exprs
                        .iter()
                        .map(|e| gen_expr(ctx, e, &schema, &row))
                        .collect();
                    yld(ctx, projected);
                }),
            );
        }

        // Scalar aggregate: fold every input row into accumulators, then emit
        // exactly one result row downstream (after the loop).
        Operator::Aggregate {
            aggs,
            input,
            schema,
        } => {
            let fields = schema.fields().clone();
            let accs: Vec<Agg> = aggs
                .iter()
                .enumerate()
                .map(|(c, e)| Agg::init(ctx, e, &fields[c]))
                .collect();

            let loop_accs = accs.clone();
            let input_schema = input.output_schema();
            gen_op(
                input,
                ctx,
                batch,
                Box::new(move |ctx, row| {
                    for agg in &loop_accs {
                        agg.update(ctx, &row, &input_schema);
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
    let nullable = field.is_nullable();
    dispatch_prim!(field.data_type(), M => {
        let view = batch.primitive::<M>(col);
        let value = ctx.bind(view.value_unchecked(i));
        M::wrap(value, read_nullness(ctx, view, nullable, i))
    })
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

fn write_col<O: OutSink>(ctx: &mut Ctx, out: O, c: usize, field: &Field, n: Var<u64>, cv: ColVal) {
    dispatch_prim!(field.data_type(), M => {
        let view = out.column_mut::<M>(c);
        let v = M::coerce(ctx, cv);
        view.set(ctx, n, v);
        write_null(ctx, view, field, n, cv);
    })
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

fn gen_predicate(ctx: &mut Ctx, e: &Expr, schema: &SchemaRef, row: &Row) -> Var<bool> {
    let cv = gen_expr(ctx, e, schema, row);
    let cond = as_bool(cv);
    // SQL: a NULL predicate does not pass the filter -> keep iff (valid && cond).
    match cv.nullness() {
        Nullness::NonNull => cond,
        Nullness::Nullable(valid) => ctx.bind(select(valid, cond, Const::<bool>::new(false))),
    }
}

fn gen_expr(ctx: &mut Ctx, e: &Expr, schema: &SchemaRef, row: &Row) -> ColVal {
    match e {
        Expr::Column(c) => {
            let idx = schema
                .index_of(&c.name)
                .unwrap_or_else(|_| panic!("unknown column: {}", c.name));
            row[idx]
        }
        Expr::Literal(sv, _) => gen_literal(ctx, sv),
        Expr::BinaryExpr(be) => gen_binary(ctx, be, schema, row),
        other => panic!("unsupported expression: {other:?}"),
    }
}

fn gen_literal(ctx: &mut Ctx, sv: &ScalarValue) -> ColVal {
    match sv {
        ScalarValue::Int32(Some(v)) => ColVal::I32(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Int64(Some(v)) => ColVal::I64(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Float64(Some(v)) => ColVal::F64(ctx.var(*v), Nullness::NonNull),
        ScalarValue::Boolean(Some(v)) => {
            ColVal::Bool(ctx.bind(Const::<bool>::new(*v)), Nullness::NonNull)
        }
        other => panic!("unsupported literal: {other:?}"),
    }
}

fn gen_binary(ctx: &mut Ctx, be: &BinaryExpr, schema: &SchemaRef, row: &Row) -> ColVal {
    let l = gen_expr(ctx, &be.left, schema, row);
    let r = gen_expr(ctx, &be.right, schema, row);
    let null = combine_null(ctx, l.nullness(), r.nullness());
    match be.op {
        DfOp::Eq => ColVal::Bool(num_cmp(ctx, Cmp::Eq, l, r), null),
        DfOp::NotEq => {
            let e = num_cmp(ctx, Cmp::Eq, l, r);
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
    }
}

// =============================================================================
// Scalar aggregation (no GROUP BY): accumulators folded by `gen_op`'s Aggregate
// arm, which emits one result row.
// =============================================================================

#[derive(Clone, Copy)]
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

    fn update(&self, ctx: &mut Ctx, row: &Row, schema: &SchemaRef) {
        let arg = self.arg.as_ref().map(|e| gen_expr(ctx, e, schema, row));
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
