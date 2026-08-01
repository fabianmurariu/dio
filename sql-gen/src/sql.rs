//! SQL front-end: parse with datafusion, then *lower* its (pull-based) logical
//! plan into our (push-based) [`Operator`] tree, reusing datafusion `Expr`
//! verbatim inside `Filter`/`Project`.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_sql::parser::DFParser;
use datafusion_sql::planner::SqlToRel;

use crate::catalog::Catalog;
use crate::plan::Operator;

/// Parse `sql` (a single statement over one table) and lower it to an
/// [`Operator`] tree.
pub fn sql_to_operator(sql: &str, table: &str, schema: SchemaRef) -> Result<Operator> {
    let catalog = Catalog::new().with_table(table, schema);
    let mut statements = DFParser::parse_sql(sql)?;
    let statement = statements
        .pop_front()
        .ok_or_else(|| DataFusionError::Plan("empty SQL statement".into()))?;
    let plan = SqlToRel::new(&catalog).statement_to_plan(statement)?;
    lower(&plan)
}

/// Lower a datafusion `LogicalPlan` node into our push operators.
fn lower(plan: &LogicalPlan) -> Result<Operator> {
    match plan {
        LogicalPlan::TableScan(scan) => Ok(Operator::Scan {
            schema: scan.source.schema(),
        }),
        LogicalPlan::Filter(filter) => Ok(Operator::Filter {
            predicate: filter.predicate.clone(),
            input: Box::new(lower(filter.input.as_ref())?),
        }),
        LogicalPlan::Projection(projection) => Ok(Operator::Project {
            exprs: projection.expr.clone(),
            schema: Arc::new(projection.schema.as_arrow().clone()),
            input: Box::new(lower(projection.input.as_ref())?),
        }),
        LogicalPlan::Aggregate(aggregate) => {
            if !aggregate.group_expr.is_empty() {
                return Err(DataFusionError::NotImplemented("GROUP BY".into()));
            }
            Ok(Operator::Aggregate {
                // aggr_expr items are `AggregateFunction` or `Alias` — unwrap.
                aggs: aggregate
                    .aggr_expr
                    .iter()
                    .map(unwrap_alias)
                    .cloned()
                    .collect(),
                schema: Arc::new(aggregate.schema.as_arrow().clone()),
                input: Box::new(lower(aggregate.input.as_ref())?),
            })
        }
        other => Err(DataFusionError::NotImplemented(format!(
            "unsupported logical operator: {}",
            other.display()
        ))),
    }
}

fn unwrap_alias(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(alias) => &alias.expr,
        other => other,
    }
}
