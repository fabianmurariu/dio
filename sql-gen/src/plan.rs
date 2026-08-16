//! Our push-based relational operators.
//!
//! The operators are *ours* (datafusion's logical operators are pull-based), but
//! the scalar expression language is reused verbatim from datafusion
//! ([`datafusion_expr::Expr`]) and schemas are plain arrow [`SchemaRef`]s. A
//! datafusion `LogicalPlan` is lowered into this tree by [`crate::sql`].

use arrow::datatypes::SchemaRef;
use datafusion_expr::{Expr, JoinType};

/// Push-model relational operators (a starter subset).
///
/// `Scan` is the single input batch; `Filter`/`Project` transform the row
/// stream; `Aggregate` (no GROUP BY) folds the stream into a single row. Each
/// carries the arrow schema of the rows it emits, so the operator above resolves
/// column references and output types against it.
#[derive(Clone, Debug)]
pub enum Operator {
    Scan {
        /// Table id — an index into the kernel's `Inputs.streams`, assigned during
        /// lowering (registration order). A single-table query is always id 0.
        table: usize,
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
    /// Equi-join. The **left** input is the materialized *build* side; the **right**
    /// streams as the *probe* side. `on` is the equijoin key pairs `(left, right)`
    /// (Phase 1: exactly one). Output columns are `[left fields | right fields]`.
    Join {
        left: Box<Operator>,
        right: Box<Operator>,
        on: Vec<(Expr, Expr)>,
        join_type: JoinType,
        /// Output schema (`[left | right]`), from datafusion's `Join`.
        schema: SchemaRef,
    },
}

impl Operator {
    /// Schema of the rows this operator *emits*.
    pub fn output_schema(&self) -> SchemaRef {
        match self {
            Operator::Scan { schema, .. } => schema.clone(),
            Operator::Filter { input, .. } => input.output_schema(),
            Operator::Project { schema, .. }
            | Operator::Aggregate { schema, .. }
            | Operator::Join { schema, .. } => schema.clone(),
        }
    }
}
