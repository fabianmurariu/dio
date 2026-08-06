//! Our push-based relational operators.
//!
//! The operators are *ours* (datafusion's logical operators are pull-based), but
//! the scalar expression language is reused verbatim from datafusion
//! ([`datafusion_expr::Expr`]) and schemas are plain arrow [`SchemaRef`]s. A
//! datafusion `LogicalPlan` is lowered into this tree by [`crate::sql`].

use arrow::datatypes::SchemaRef;
use datafusion_expr::Expr;

/// Push-model relational operators (a starter subset).
///
/// `Scan` is the single input batch; `Filter`/`Project` transform the row
/// stream; `Aggregate` (no GROUP BY) folds the stream into a single row. Each
/// carries the arrow schema of the rows it emits, so the operator above resolves
/// column references and output types against it.
#[derive(Clone, Debug)]
pub enum Operator {
    Scan {
        schema: SchemaRef,
    },
    Filter {
        predicate: Expr,
        input: Box<Operator>,
    },
    Project {
        exprs: Vec<Expr>,
        /// Output schema (types + nullability), from datafusion's `Projection`.
        schema: SchemaRef,
        input: Box<Operator>,
    },
    /// Aggregation. With no `group_exprs` it's a scalar aggregate — one output
    /// row. With `group_exprs` it's a GROUP BY — one row per distinct key, output
    /// columns `[group keys… | aggregates…]`. Each `agg` is a datafusion
    /// `Expr::AggregateFunction` (count/min/max/sum/…).
    Aggregate {
        group_exprs: Vec<Expr>,
        aggs: Vec<Expr>,
        /// Output schema (`[group keys | aggregates]`), from datafusion's `Aggregate`.
        schema: SchemaRef,
        input: Box<Operator>,
    },
}

impl Operator {
    /// Schema of the rows this operator *emits*.
    pub fn output_schema(&self) -> SchemaRef {
        match self {
            Operator::Scan { schema } => schema.clone(),
            Operator::Filter { input, .. } => input.output_schema(),
            Operator::Project { schema, .. } | Operator::Aggregate { schema, .. } => schema.clone(),
        }
    }

    /// Upper bound on emitted rows, given the input row count — the output
    /// buffer size. A scalar aggregate emits exactly one row (even for empty
    /// input); filters/projections emit at most one per input row.
    pub fn max_output_rows(&self, input_rows: usize) -> usize {
        match self {
            // Scalar aggregate → 1 row; GROUP BY → at most one group per input row.
            Operator::Aggregate { group_exprs, .. } => {
                if group_exprs.is_empty() {
                    1
                } else {
                    input_rows
                }
            }
            Operator::Scan { .. } => input_rows,
            Operator::Filter { input, .. } | Operator::Project { input, .. } => {
                input.max_output_rows(input_rows)
            }
        }
    }
}
