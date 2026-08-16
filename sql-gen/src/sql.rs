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
        // A derived table `(SELECT …) alias` — a transparent rename; lower its input.
        LogicalPlan::SubqueryAlias(sa) => lower(sa.input.as_ref(), ids),
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
        LogicalPlan::Join(join) => {
            // Phase 1: inner equi-join, exactly one key pair.
            if join.join_type != datafusion_expr::JoinType::Inner {
                return Err(DataFusionError::NotImplemented(format!(
                    "join type {:?} (only INNER for now)",
                    join.join_type
                )));
            }
            // Raw `SqlToRel` (no optimizer) leaves `on` empty and puts the equijoin
            // in `filter`; the `ExtractEquijoinPredicate` rule normally moves it. We
            // do that extraction ourselves: accept `on` if populated, else pull the
            // single `left.col = right.col` pair out of `filter`.
            let on = if !join.on.is_empty() {
                if join.on.len() != 1 || join.filter.is_some() {
                    return Err(DataFusionError::NotImplemented(
                        "join with composite keys or a residual filter".into(),
                    ));
                }
                join.on.clone()
            } else if let Some(filter) = &join.filter {
                vec![extract_equijoin(
                    filter,
                    join.left.as_ref(),
                    join.right.as_ref(),
                )?]
            } else {
                return Err(DataFusionError::NotImplemented(
                    "cross join (no join condition)".into(),
                ));
            };
            Ok(Operator::Join {
                left: Box::new(lower(join.left.as_ref(), ids)?),
                right: Box::new(lower(join.right.as_ref(), ids)?),
                on,
                join_type: join.join_type,
                schema: Arc::new(join.schema.as_arrow().clone()),
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

/// Pull a single `left.col = right.col` equijoin pair out of a join `filter`,
/// ordered `(left-side expr, right-side expr)` using the input schemas to decide
/// which column belongs to which side.
fn extract_equijoin(
    filter: &Expr,
    left: &LogicalPlan,
    right: &LogicalPlan,
) -> Result<(Expr, Expr)> {
    if let Expr::BinaryExpr(be) = filter
        && be.op == datafusion_expr::Operator::Eq
        && let (Expr::Column(lc), Expr::Column(rc)) = (be.left.as_ref(), be.right.as_ref())
    {
        // `lc = rc`: assign each column to its side by which input schema holds it.
        if left.schema().index_of_column(lc).is_ok() && right.schema().index_of_column(rc).is_ok() {
            return Ok(((*be.left).clone(), (*be.right).clone()));
        } else if left.schema().index_of_column(rc).is_ok()
            && right.schema().index_of_column(lc).is_ok()
        {
            return Ok(((*be.right).clone(), (*be.left).clone()));
        }
    }
    Err(DataFusionError::NotImplemented(
        "join condition must be a single `left.col = right.col` equijoin".into(),
    ))
}
