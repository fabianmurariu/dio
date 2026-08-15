//! SQL front-end: parse with datafusion, then *lower* its (pull-based) logical
//! plan into our (push-based) [`Operator`] tree, reusing datafusion `Expr`
//! verbatim inside `Filter`/`Project`.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{Expr, LogicalPlan};
use datafusion_sql::parser::DFParser;
use datafusion_sql::planner::SqlToRel;

use crate::catalog::Catalog;
use crate::plan::Operator;

/// Parse `sql` (a single statement over one table) and lower it to an
/// [`Operator`] tree — the single-table shorthand for [`sql_to_operator_multi`]
/// (table id 0).
pub fn sql_to_operator(sql: &str, table: &str, schema: SchemaRef) -> Result<Operator> {
    sql_to_operator_multi(sql, &[(table, schema)])
}

/// Parse `sql` over several named tables and lower it. Each table's **id** is its
/// index in `tables`; a `Scan` carries that id so the kernel pulls from the
/// matching `Inputs.streams` slot. (A single query scans one table today — joins
/// arrive later — but the id plumbing is what a join will thread.)
pub fn sql_to_operator_multi(sql: &str, tables: &[(&str, SchemaRef)]) -> Result<Operator> {
    let mut catalog = Catalog::new();
    let mut ids: HashMap<String, usize> = HashMap::new();
    for (id, (name, schema)) in tables.iter().enumerate() {
        catalog = catalog.with_table(*name, schema.clone());
        ids.insert((*name).to_string(), id);
    }
    let mut statements = DFParser::parse_sql(sql)?;
    let statement = statements
        .pop_front()
        .ok_or_else(|| DataFusionError::Plan("empty SQL statement".into()))?;
    let plan = SqlToRel::new(&catalog).statement_to_plan(statement)?;
    lower(&plan, &ids)
}

/// Lower a datafusion `LogicalPlan` node into our push operators, resolving each
/// scanned table name to its id via `ids`.
fn lower(plan: &LogicalPlan, ids: &HashMap<String, usize>) -> Result<Operator> {
    match plan {
        LogicalPlan::TableScan(scan) => {
            let name = scan.table_name.table();
            let table = *ids
                .get(name)
                .ok_or_else(|| DataFusionError::Plan(format!("table not registered: {name}")))?;
            Ok(Operator::Scan {
                table,
                schema: scan.source.schema(),
            })
        }
        LogicalPlan::Filter(filter) => Ok(Operator::Filter {
            predicate: filter.predicate.clone(),
            input: Box::new(lower(filter.input.as_ref(), ids)?),
        }),
        LogicalPlan::Projection(projection) => Ok(Operator::Project {
            exprs: projection.expr.clone(),
            schema: Arc::new(projection.schema.as_arrow().clone()),
            input: Box::new(lower(projection.input.as_ref(), ids)?),
        }),
        LogicalPlan::Aggregate(aggregate) => Ok(Operator::Aggregate {
            group_exprs: aggregate
                .group_expr
                .iter()
                .map(unwrap_alias)
                .cloned()
                .collect(),
            // aggr_expr items are `AggregateFunction` or `Alias` — unwrap.
            aggs: aggregate
                .aggr_expr
                .iter()
                .map(unwrap_alias)
                .cloned()
                .collect(),
            schema: Arc::new(aggregate.schema.as_arrow().clone()),
            input: Box::new(lower(aggregate.input.as_ref(), ids)?),
        }),
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
