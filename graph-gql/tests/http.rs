//! Integration test: a real GraphQL client (gql_client) over HTTP against the
//! running axum service, validating the responses deserialize as proper GraphQL.

use std::sync::Arc;

use gql_client::Client;
use graph_gql::{http::app, store::GraphStore};
use raphtory::prelude::*;
use serde::Deserialize;

/// Start the service on an ephemeral port; return the `/graphql` URL.
async fn spawn_server() -> String {
    let store = Arc::new(GraphStore::new());
    let g = Graph::new();
    g.add_edge(0, 1u64, 2u64, NO_PROPS, None).unwrap();
    g.add_edge(0, 1u64, 3u64, NO_PROPS, None).unwrap();
    store.insert("a", g);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, app(store)).await.unwrap();
    });
    format!("http://{addr}/graphql")
}

#[derive(Deserialize)]
struct Query {
    graph: GraphData,
}
#[derive(Deserialize)]
struct GraphData {
    node: NodeData,
}
#[derive(Deserialize)]
struct NodeData {
    name: String,
    neighbours: NameList,
}
#[derive(Deserialize)]
struct NameList {
    list: Vec<NameOnly>,
}
#[derive(Deserialize)]
struct NameOnly {
    name: String,
}

#[tokio::test]
async fn node_neighbours_over_http() {
    let url = spawn_server().await;
    let client = Client::new(url);

    let data: Query = client
        .query::<Query>(
            r#"{ graph(path: "a") { node(name: "1") { name neighbours { list { name } } } } }"#,
        )
        .await
        .expect("graphql request failed")
        .expect("no data");

    assert_eq!(data.graph.node.name, "1");
    let neighbours: Vec<&str> = data
        .graph
        .node
        .neighbours
        .list
        .iter()
        .map(|n| n.name.as_str())
        .collect();
    assert_eq!(neighbours, ["2", "3"]);
}

#[derive(Deserialize)]
struct Introspect {
    #[serde(rename = "__schema")]
    schema: SchemaInfo,
}
#[derive(Deserialize)]
struct SchemaInfo {
    #[serde(rename = "queryType")]
    query_type: NamedType,
    types: Vec<NamedType>,
}
#[derive(Deserialize)]
struct NamedType {
    name: Option<String>,
}

#[tokio::test]
async fn introspection_over_http() {
    let url = spawn_server().await;
    let client = Client::new(url);

    let data: Introspect = client
        .query::<Introspect>("{ __schema { queryType { name } types { name } } }")
        .await
        .expect("introspection failed")
        .expect("no data");

    assert_eq!(data.schema.query_type.name.as_deref(), Some("Query"));
    let names: Vec<&str> = data
        .schema
        .types
        .iter()
        .filter_map(|t| t.name.as_deref())
        .collect();
    for expected in ["Graph", "Node", "NodeList", "Event", "EventList"] {
        assert!(
            names.contains(&expected),
            "missing type {expected} in {names:?}"
        );
    }
}

#[derive(Deserialize)]
struct AddEdge {
    #[serde(rename = "addEdge")]
    add_edge: bool,
}

#[tokio::test]
async fn mutation_add_edge_over_http() {
    let url = spawn_server().await;
    let client = Client::new(url);

    let data: AddEdge = client
        .query::<AddEdge>(r#"mutation { addEdge(graph: "a", time: 9, src: 2, dst: 3) }"#)
        .await
        .expect("graphql request failed")
        .expect("no data");
    assert!(data.add_edge);
}
