//! axum HTTP service: `POST /graphql`, streaming the response body.
//!
//! The handler parses + lowers synchronously (so parse errors are a normal
//! response), then runs the synchronous executor on a blocking thread, pushing
//! byte chunks through a channel that the response body drains — the response is
//! never fully buffered.

use std::convert::Infallible;
use std::sync::Arc;

use async_graphql::dynamic::Schema;
use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{StatusCode, header},
    response::{IntoResponse, Response},
    routing::post,
};
use bytes::{Bytes, BytesMut};
use serde::Deserialize;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;

use crate::sink::{Sink, VecSink};
use crate::store::GraphStore;
use crate::{exec, lower, schema, write_error};

/// Shared handler state: the graph store + the introspection-only schema.
#[derive(Clone)]
struct AppState {
    store: Arc<GraphStore>,
    schema: Schema,
}

/// Standard GraphQL-over-HTTP request envelope.
#[derive(Deserialize)]
pub struct GqlRequest {
    pub query: String,
    /// Accepted but ignored in the POC.
    #[serde(default)]
    pub variables: Option<serde_json::Value>,
    /// Accepted but ignored in the POC.
    #[serde(default, rename = "operationName")]
    pub operation_name: Option<String>,
}

/// Build the axum app over a shared graph store.
pub fn app(store: Arc<GraphStore>) -> Router {
    let state = AppState {
        store,
        schema: schema::build(),
    };
    Router::new()
        .route("/graphql", post(handler))
        .with_state(state)
}

async fn handler(State(state): State<AppState>, Json(req): Json<GqlRequest>) -> Response {
    // Introspection (`__schema` / `__type`) is answered by async-graphql.
    if lower::is_introspection(&req.query) {
        let resp = state.schema.execute(req.query).await;
        let body = serde_json::to_vec(&resp).unwrap_or_default();
        return json_bytes(body);
    }

    // Parse + lower up front so syntax/lowering errors are a normal response.
    let plan = match lower::parse_and_lower(&req.query) {
        Ok(p) => p,
        Err(msg) => {
            let mut buf = VecSink::new();
            write_error(&mut buf, &msg);
            return json_bytes(buf.0);
        }
    };

    // Stream: the sync executor runs on a blocking thread and pushes chunks into
    // a channel; the response body drains it.
    let store = state.store;
    let (tx, rx) = mpsc::channel::<Result<Bytes, Infallible>>(16);
    tokio::task::spawn_blocking(move || {
        let mut sink = ChannelSink::new(tx);
        exec::run(&plan, &store, &mut sink);
        sink.finish();
    });

    Response::builder()
        .status(StatusCode::OK)
        .header(header::CONTENT_TYPE, "application/json")
        .body(Body::from_stream(ReceiverStream::new(rx)))
        .unwrap()
}

fn json_bytes(bytes: Vec<u8>) -> Response {
    ([(header::CONTENT_TYPE, "application/json")], bytes).into_response()
}

/// A [`Sink`] that batches bytes and streams them over a channel to the HTTP
/// response — the full response is never held in memory at once.
struct ChannelSink {
    tx: mpsc::Sender<Result<Bytes, Infallible>>,
    buf: BytesMut,
}

const CHUNK: usize = 8 * 1024;

impl ChannelSink {
    fn new(tx: mpsc::Sender<Result<Bytes, Infallible>>) -> Self {
        Self {
            tx,
            buf: BytesMut::with_capacity(CHUNK),
        }
    }

    fn flush(&mut self) {
        if !self.buf.is_empty() {
            let chunk = self.buf.split().freeze();
            let _ = self.tx.blocking_send(Ok(chunk));
        }
    }

    fn finish(mut self) {
        self.flush();
    }
}

impl Sink for ChannelSink {
    fn put(&mut self, bytes: &[u8]) {
        self.buf.extend_from_slice(bytes);
        if self.buf.len() >= CHUNK {
            self.flush();
        }
    }
}
