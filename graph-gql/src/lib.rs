//! GraphQL-over-raphtory POC.
//!
//! Pipeline: GraphQL query string → parse (async-graphql) → lower to a small
//! [`plan`] IR (`ScanNodes`/`Neighbours`/…) → execute with the [`exec`]
//! interpreter, streaming the response into a [`sink::Sink`] as it generates.
//! The `Plan` IR + `Sink` are the seams where the interpreter will later be
//! swapped for a rust-lms-compiled kernel writing into the same sink.

pub mod exec;
pub mod http;
pub mod lower;
pub mod plan;
pub mod schema;
pub mod sink;
pub mod store;

use sink::{Sink, VecSink};
use store::GraphStore;

/// Parse, lower, and execute a request, collecting the JSON response into a
/// `String` (convenience for tests / the synchronous path).
pub fn execute(query: &str, store: &GraphStore) -> String {
    let mut sink = VecSink::new();
    execute_into(query, store, &mut sink);
    sink.into_string()
}

/// Parse, lower, and execute a request, streaming the JSON response into `sink`.
pub fn execute_into(query: &str, store: &GraphStore, sink: &mut dyn Sink) {
    match lower::parse_and_lower(query) {
        Ok(plan) => exec::run(&plan, store, sink),
        Err(msg) => write_error(sink, &msg),
    }
}

/// Emit a GraphQL `{"errors":[{"message": …}]}` envelope into `sink`.
pub fn write_error(sink: &mut dyn Sink, msg: &str) {
    sink.begin_obj();
    sink.key("errors");
    sink.begin_arr();
    sink.begin_obj();
    sink.key("message");
    sink.string(msg);
    sink.end_obj();
    sink.end_arr();
    sink.end_obj();
}
