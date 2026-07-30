//! Our push-based relational operators.
//!
//! The operators are *ours* (datafusion's logical operators are pull-based), but
//! the scalar expression language is reused verbatim from datafusion
//! ([`datafusion_expr::Expr`]) and the schema is a plain arrow [`SchemaRef`].
//! A datafusion `LogicalPlan` is lowered into this tree by [`crate::sql`].

use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion_expr::Expr;

/// Push-model relational operators (a starter subset).
///
/// `Scan` is the single input batch; `Filter`/`Project` transform the row
/// stream. Each operator describes *what records it emits to its consumer*.
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
        input: Box<Operator>,
    },
}

impl Operator {
    /// Schema of the rows this operator *emits* — i.e. the schema against which
    /// the next operator up resolves its column references.
    pub fn output_schema(&self) -> SchemaRef {
        match self {
            Operator::Scan { schema } => schema.clone(),
            Operator::Filter { input, .. } => input.output_schema(),
            Operator::Project { exprs, input } => {
                // Column-reference projections only, for now; that is all the
                // lowering produces.
                let in_schema = input.output_schema();
                let fields = exprs
                    .iter()
                    .map(|e| match e {
                        Expr::Column(c) => in_schema
                            .field_with_name(&c.name)
                            .expect("projected column exists")
                            .clone(),
                        other => panic!("unsupported projection expression: {other:?}"),
                    })
                    .collect::<Vec<_>>();
                Arc::new(Schema::new(fields))
            }
        }
    }
}
