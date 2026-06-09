//! The Plan IR — the seam between GraphQL and execution.
//!
//! Lowered from the parsed GraphQL query; executed by [`crate::exec`] now, and
//! (later) compiled by rust-lms. Node sets line up with the staged sources:
//! a `nodes`/`node(name)` source is `ScanNodes`, a nested `neighbours` is
//! `Neighbours`.
//!
//! Client-facing identity is the **external** id (the node's `name`); traversal
//! uses the **internal** VID and never leaks it.

#[derive(Debug, Clone)]
// A 2-variant request plan; the size gap (query selection vs mutation args) is
// fine — plans are built once per request.
#[allow(clippy::large_enum_variant)]
pub enum Plan {
    /// `graph(path) { … }`
    Query {
        graph_key: String,
        graph: String,
        sel: GraphSel,
    },
    /// `addEdge(graph, time, src, dst)`
    AddEdge {
        result_key: String,
        graph: String,
        time: i64,
        src: u64,
        dst: u64,
    },
}

/// Selections under `graph(path) { … }`.
#[derive(Debug, Clone, Default)]
pub struct GraphSel {
    /// `node(name: …) { … }` — a single node.
    pub node: Option<NodeByName>,
    /// `nodes { list { … } }` — all nodes.
    pub nodes: Option<NodeList>,
}

/// `node(name: "…") { <NodeSel> }`.
#[derive(Debug, Clone)]
pub struct NodeByName {
    pub key: String,
    pub name: String,
    pub sel: NodeSel,
}

/// A `…  { list { <NodeSel> } }` collection (used by `nodes` and `neighbours`).
#[derive(Debug, Clone)]
pub struct NodeList {
    /// Output key of the collection field (e.g. "neighbours").
    pub key: String,
    /// Output key of the inner `list` field.
    pub list_key: String,
    pub sel: Box<NodeSel>,
}

/// What to emit for each node, and any nested traversal.
#[derive(Debug, Clone, Default)]
pub struct NodeSel {
    /// Emit the external name under this output key (`name`).
    pub name_key: Option<String>,
    /// `neighbours { list { … } }`.
    pub neighbours: Option<NodeList>,
    /// `history { list { timestamp eventId } }`.
    pub history: Option<History>,
}

/// `history { list { timestamp eventId } }`.
#[derive(Debug, Clone)]
pub struct History {
    pub key: String,
    pub list_key: String,
    /// Output key for `timestamp`, if selected.
    pub ts_key: Option<String>,
    /// Output key for `eventId`, if selected.
    pub eid_key: Option<String>,
}
