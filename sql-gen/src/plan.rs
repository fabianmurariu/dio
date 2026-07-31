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
/// stream. Each carries the arrow schema of the rows it emits, so the operator
/// above resolves column references and output types against it.
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
        /// Output schema (types + nullability), taken from datafusion's
        /// `Projection` node so we don't re-infer expression types.
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
            Operator::Project { schema, .. } => schema.clone(),
        }
    }
}
