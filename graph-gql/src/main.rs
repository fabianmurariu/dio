//! Runnable graph-gql service: seeds a sample graph and serves `/graphql`.

use std::sync::Arc;

use graph_gql::{http::app, store::GraphStore};
use raphtory::prelude::*;

#[tokio::main]
async fn main() {
    let store = Arc::new(GraphStore::new());

    // Seed a sample graph "a": edges 1→2 and 1→3.
    let g = Graph::new();
    g.add_edge(0, 1u64, 2u64, NO_PROPS, None).unwrap();
    g.add_edge(0, 1u64, 3u64, NO_PROPS, None).unwrap();
    store.insert("a", g);

    let listener = tokio::net::TcpListener::bind("127.0.0.1:8000")
        .await
        .unwrap();
    println!("graph-gql listening on http://127.0.0.1:8000/graphql");
    axum::serve(listener, app(store)).await.unwrap();
}
