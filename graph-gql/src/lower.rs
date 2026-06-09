//! Parse a GraphQL request (async-graphql parser) and lower it to a [`Plan`].
//!
//! Unknown fields/arguments are rejected here — schema-driven validation, since
//! async-graphql does not expose a standalone validate-only entry point.

use async_graphql::parser::types::{
    DocumentOperations, ExecutableDocument, Field, OperationType, Selection, SelectionSet,
};
use async_graphql_value::Value;

use crate::plan::{GraphSel, History, NodeByName, NodeList, NodeSel, Plan};

/// Parse `query` and lower the (single) operation to a [`Plan`].
pub fn parse_and_lower(query: &str) -> Result<Plan, String> {
    let doc = async_graphql::parser::parse_query(query).map_err(|e| e.to_string())?;
    lower(&doc)
}

/// `true` if the request's first field is an introspection meta-field
/// (`__schema` / `__type`), which we route to async-graphql instead of the
/// data executor.
pub fn is_introspection(query: &str) -> bool {
    let Ok(doc) = async_graphql::parser::parse_query(query) else {
        return false;
    };
    let op = match &doc.operations {
        DocumentOperations::Single(op) => &op.node,
        DocumentOperations::Multiple(m) => match m.values().next() {
            Some(o) => &o.node,
            None => return false,
        },
    };
    first_field(&op.selection_set.node)
        .map(|f| f.name.node.as_str().starts_with("__"))
        .unwrap_or(false)
}

fn lower(doc: &ExecutableDocument) -> Result<Plan, String> {
    let op = match &doc.operations {
        DocumentOperations::Single(op) => &op.node,
        DocumentOperations::Multiple(m) => &m.values().next().ok_or("no operation")?.node,
    };
    let ss = &op.selection_set.node;
    match op.ty {
        OperationType::Query => lower_query(ss),
        OperationType::Mutation => lower_mutation(ss),
        OperationType::Subscription => Err("subscriptions are not supported".into()),
    }
}

// --- selection-set helpers ---------------------------------------------------

fn fields(ss: &SelectionSet) -> impl Iterator<Item = &Field> {
    ss.items.iter().filter_map(|s| match &s.node {
        Selection::Field(f) => Some(&f.node),
        _ => None,
    })
}

fn first_field(ss: &SelectionSet) -> Option<&Field> {
    fields(ss).next()
}

fn field_named<'a>(ss: &'a SelectionSet, name: &str) -> Option<&'a Field> {
    fields(ss).find(|f| f.name.node.as_str() == name)
}

fn output_key(f: &Field) -> String {
    f.alias
        .as_ref()
        .map(|a| a.node.to_string())
        .unwrap_or_else(|| f.name.node.to_string())
}

fn arg<'a>(f: &'a Field, name: &str) -> Option<&'a Value> {
    f.arguments
        .iter()
        .find(|(n, _)| n.node.as_str() == name)
        .map(|(_, v)| &v.node)
}

fn string_arg(f: &Field, name: &str) -> Result<String, String> {
    match arg(f, name) {
        Some(Value::String(s)) => Ok(s.clone()),
        _ => Err(format!("expected string argument `{name}`")),
    }
}

fn int_arg(f: &Field, name: &str) -> Result<i64, String> {
    match arg(f, name) {
        Some(Value::Number(n)) => n
            .as_i64()
            .ok_or_else(|| format!("argument `{name}` is not an integer")),
        _ => Err(format!("expected integer argument `{name}`")),
    }
}

// --- query lowering ----------------------------------------------------------

fn lower_query(ss: &SelectionSet) -> Result<Plan, String> {
    let g = first_field(ss).ok_or("empty query")?;
    if g.name.node.as_str() != "graph" {
        return Err(format!("expected `graph`, got `{}`", g.name.node));
    }
    let graph = string_arg(g, "path")?;
    let graph_key = output_key(g);
    let sel = lower_graph_sel(&g.selection_set.node)?;
    Ok(Plan::Query {
        graph_key,
        graph,
        sel,
    })
}

fn lower_graph_sel(ss: &SelectionSet) -> Result<GraphSel, String> {
    let mut sel = GraphSel::default();
    for f in fields(ss) {
        match f.name.node.as_str() {
            "node" => {
                sel.node = Some(NodeByName {
                    key: output_key(f),
                    name: string_arg(f, "name")?,
                    sel: lower_node_sel(&f.selection_set.node)?,
                });
            }
            "nodes" => sel.nodes = Some(lower_node_list(f)?),
            other => return Err(format!("unknown field `{other}` on `graph`")),
        }
    }
    Ok(sel)
}

/// A collection field (`nodes`/`neighbours`) wraps its items in a `list` field.
fn lower_node_list(f: &Field) -> Result<NodeList, String> {
    let list = field_named(&f.selection_set.node, "list")
        .ok_or_else(|| format!("`{}` requires a `list` selection", f.name.node))?;
    Ok(NodeList {
        key: output_key(f),
        list_key: output_key(list),
        sel: Box::new(lower_node_sel(&list.selection_set.node)?),
    })
}

fn lower_node_sel(ss: &SelectionSet) -> Result<NodeSel, String> {
    let mut sel = NodeSel::default();
    for f in fields(ss) {
        match f.name.node.as_str() {
            "name" => sel.name_key = Some(output_key(f)),
            "neighbours" => sel.neighbours = Some(lower_node_list(f)?),
            "history" => sel.history = Some(lower_history(f)?),
            other => return Err(format!("unknown field `{other}` on `Node`")),
        }
    }
    Ok(sel)
}

fn lower_history(f: &Field) -> Result<History, String> {
    let list = field_named(&f.selection_set.node, "list")
        .ok_or("`history` requires a `list` selection")?;
    let mut h = History {
        key: output_key(f),
        list_key: output_key(list),
        ts_key: None,
        eid_key: None,
    };
    for ef in fields(&list.selection_set.node) {
        match ef.name.node.as_str() {
            "timestamp" => h.ts_key = Some(output_key(ef)),
            "eventId" => h.eid_key = Some(output_key(ef)),
            other => return Err(format!("unknown field `{other}` on `Event`")),
        }
    }
    Ok(h)
}

// --- mutation lowering -------------------------------------------------------

fn lower_mutation(ss: &SelectionSet) -> Result<Plan, String> {
    let f = first_field(ss).ok_or("empty mutation")?;
    if f.name.node.as_str() != "addEdge" {
        return Err(format!("expected `addEdge`, got `{}`", f.name.node));
    }
    Ok(Plan::AddEdge {
        result_key: output_key(f),
        graph: string_arg(f, "graph")?,
        time: int_arg(f, "time")?,
        src: int_arg(f, "src")? as u64,
        dst: int_arg(f, "dst")? as u64,
    })
}
