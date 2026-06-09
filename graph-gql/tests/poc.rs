//! End-to-end POC: GraphQL string → plan → interpret over raphtory → JSON.

use graph_gql::{execute, store::GraphStore};
use raphtory::prelude::*;

/// Graph "a": edges 1→2 and 1→3 (external ids 1,2,3 → names "1","2","3").
fn seeded() -> GraphStore {
    let s = GraphStore::new();
    let g = Graph::new();
    g.add_edge(0, 1u64, 2u64, NO_PROPS, None).unwrap();
    g.add_edge(0, 1u64, 3u64, NO_PROPS, None).unwrap();
    s.insert("a", g);
    s
}

#[test]
fn node_by_name_with_neighbours() {
    let out = execute(
        r#"{ graph(path: "a") { node(name: "1") { name neighbours { list { name } } } } }"#,
        &seeded(),
    );
    assert_eq!(
        out,
        r#"{"data":{"graph":{"node":{"name":"1","neighbours":{"list":[{"name":"2"},{"name":"3"}]}}}}}"#
    );
}

#[test]
fn all_nodes() {
    let out = execute(
        r#"{ graph(path: "a") { nodes { list { name } } } }"#,
        &seeded(),
    );
    assert_eq!(
        out,
        r#"{"data":{"graph":{"nodes":{"list":[{"name":"1"},{"name":"2"},{"name":"3"}]}}}}"#
    );
}

#[test]
fn node_history() {
    // node 1 was updated twice at t=0 (two edges); eventId is raphtory's
    // secondary index disambiguating same-timestamp events.
    let out = execute(
        r#"{ graph(path: "a") { node(name: "1") { history { list { timestamp eventId } } } } }"#,
        &seeded(),
    );
    assert_eq!(
        out,
        r#"{"data":{"graph":{"node":{"history":{"list":[{"timestamp":0,"eventId":0},{"timestamp":0,"eventId":1}]}}}}}"#
    );
}

#[test]
fn add_edge_mutation() {
    let out = execute(
        r#"mutation { addEdge(graph: "a", time: 5, src: 1, dst: 4) }"#,
        &seeded(),
    );
    assert_eq!(out, r#"{"data":{"addEdge":true}}"#);
}

#[test]
fn unknown_field_is_rejected() {
    let out = execute(
        r#"{ graph(path: "a") { node(name: "1") { bogus } } }"#,
        &seeded(),
    );
    assert!(out.contains("errors") && out.contains("bogus"), "{out}");
}

#[test]
fn missing_graph_is_null() {
    let out = execute(
        r#"{ graph(path: "nope") { nodes { list { name } } } }"#,
        &GraphStore::new(),
    );
    assert_eq!(out, r#"{"data":{"graph":null}}"#);
}
