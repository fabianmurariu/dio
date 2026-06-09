//! The POC executor: walk a [`Plan`] over raphtory iterators, streaming the
//! GraphQL JSON response into a [`Sink`] as it goes.
//!
//! Identity rule: traverse with the internal VID (`NodeView::new_internal`),
//! emit the external `name` (`.name()`); the VID is never returned.

use raphtory::core::entities::VID;
use raphtory::db::graph::node::NodeView;
use raphtory::prelude::*;

use crate::plan::{GraphSel, History, NodeList, NodeSel, Plan};
use crate::sink::Sink;
use crate::store::GraphStore;

/// Execute a plan against the store, streaming the JSON response into `sink`.
pub fn run(plan: &Plan, store: &GraphStore, sink: &mut dyn Sink) {
    match plan {
        Plan::Query {
            graph_key,
            graph,
            sel,
        } => {
            sink.begin_obj();
            sink.key("data");
            sink.begin_obj();
            sink.key(graph_key);
            let found = store.with(graph, |g| write_graph(g, sel, sink));
            if found.is_none() {
                sink.null();
            }
            sink.end_obj();
            sink.end_obj();
        }
        Plan::AddEdge {
            result_key,
            graph,
            time,
            src,
            dst,
        } => {
            let ok = store.add_edge(graph, *time, *src, *dst).is_ok();
            sink.begin_obj();
            sink.key("data");
            sink.begin_obj();
            sink.key(result_key);
            sink.bool(ok);
            sink.end_obj();
            sink.end_obj();
        }
    }
}

fn write_graph(g: &Graph, sel: &GraphSel, sink: &mut dyn Sink) {
    sink.begin_obj();
    let mut first = true;
    if let Some(nb) = &sel.node {
        sink.key(&nb.key);
        match g.node(nb.name.as_str()) {
            Some(nv) => write_node(g, &nb.sel, nv.node.as_u64(), sink),
            None => sink.null(),
        }
        first = false;
    }
    if let Some(nl) = &sel.nodes {
        if !first {
            sink.comma();
        }
        let vids = g.nodes().into_iter().map(|n| n.node.as_u64());
        write_node_list(g, nl, vids, sink);
    }
    sink.end_obj();
}

/// `<key>: { <list_key>: [ <node>… ] }`
fn write_node_list(g: &Graph, nl: &NodeList, vids: impl Iterator<Item = u64>, sink: &mut dyn Sink) {
    sink.key(&nl.key);
    sink.begin_obj();
    sink.key(&nl.list_key);
    sink.begin_arr();
    for (i, vid) in vids.enumerate() {
        if i > 0 {
            sink.comma();
        }
        write_node(g, &nl.sel, vid, sink);
    }
    sink.end_arr();
    sink.end_obj();
}

fn write_node(g: &Graph, sel: &NodeSel, vid: u64, sink: &mut dyn Sink) {
    sink.begin_obj();
    let mut first = true;
    if let Some(name_key) = &sel.name_key {
        sink.key(name_key);
        sink.string(&NodeView::new_internal(g, VID(vid as usize)).name());
        first = false;
    }
    if let Some(nl) = &sel.neighbours {
        if !first {
            sink.comma();
        }
        let nbrs = NodeView::new_internal(g, VID(vid as usize))
            .neighbours()
            .into_iter()
            .map(|x| x.node.as_u64());
        write_node_list(g, nl, nbrs, sink);
        first = false;
    }
    if let Some(h) = &sel.history {
        if !first {
            sink.comma();
        }
        write_history(g, h, vid, sink);
    }
    sink.end_obj();
}

/// `<key>: { <list_key>: [ {timestamp, eventId}… ] }`
fn write_history(g: &Graph, h: &History, vid: u64, sink: &mut dyn Sink) {
    sink.key(&h.key);
    sink.begin_obj();
    sink.key(&h.list_key);
    sink.begin_arr();
    let nv = NodeView::new_internal(g, VID(vid as usize));
    for (i, ev) in nv.history().iter().enumerate() {
        if i > 0 {
            sink.comma();
        }
        sink.begin_obj();
        let mut first = true;
        if let Some(ts_key) = &h.ts_key {
            sink.key(ts_key);
            sink.i64(ev.0); // EventTime(.0 = timestamp i64, .1 = eventId usize)
            first = false;
        }
        if let Some(eid_key) = &h.eid_key {
            if !first {
                sink.comma();
            }
            sink.key(eid_key);
            sink.u64(ev.1 as u64);
        }
        sink.end_obj();
    }
    sink.end_arr();
    sink.end_obj();
}
