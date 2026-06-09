//! The GraphQL schema, defined with async-graphql's dynamic schema builder.
//!
//! async-graphql owns the **type system**: it answers introspection (`__schema`
//! / `__type`) and produces the SDL, so the endpoint is discoverable, valid
//! GraphQL. It is built in `introspection_only` mode — data fields are never
//! resolved here; our streaming executor (see [`crate::exec`]) produces data.
//! Validation of data queries is enforced against this same shape in
//! [`crate::lower`].

use async_graphql::dynamic::{Field, FieldFuture, FieldValue, InputValue, Object, Schema, TypeRef};

/// A field with a never-called stub resolver (introspection-only schema). The
/// closure is passed inline so it infers the required `for<'a>` resolver type.
fn f(name: &str, ty: TypeRef) -> Field {
    Field::new(name, ty, |_| {
        FieldFuture::new(async { Ok(Some(FieldValue::NULL)) })
    })
}

/// Build the introspection-only schema.
pub fn build() -> Schema {
    let event = Object::new("Event")
        .field(f("timestamp", TypeRef::named_nn(TypeRef::INT)))
        .field(f("eventId", TypeRef::named_nn(TypeRef::INT)));

    let event_list = Object::new("EventList").field(f("list", TypeRef::named_nn_list_nn("Event")));

    let node = Object::new("Node")
        .field(f("name", TypeRef::named_nn(TypeRef::STRING)))
        .field(f("neighbours", TypeRef::named_nn("NodeList")))
        .field(f("history", TypeRef::named_nn("EventList")));

    let node_list = Object::new("NodeList").field(f("list", TypeRef::named_nn_list_nn("Node")));

    let graph = Object::new("Graph")
        .field(
            f("node", TypeRef::named("Node"))
                .argument(InputValue::new("name", TypeRef::named_nn(TypeRef::STRING))),
        )
        .field(f("nodes", TypeRef::named_nn("NodeList")));

    let query = Object::new("Query").field(
        f("graph", TypeRef::named("Graph"))
            .argument(InputValue::new("path", TypeRef::named_nn(TypeRef::STRING))),
    );

    let mutation = Object::new("Mutation").field(
        f("addEdge", TypeRef::named_nn(TypeRef::BOOLEAN))
            .argument(InputValue::new("graph", TypeRef::named_nn(TypeRef::STRING)))
            .argument(InputValue::new("time", TypeRef::named_nn(TypeRef::INT)))
            .argument(InputValue::new("src", TypeRef::named_nn(TypeRef::INT)))
            .argument(InputValue::new("dst", TypeRef::named_nn(TypeRef::INT))),
    );

    Schema::build("Query", Some("Mutation"), None)
        .introspection_only()
        .register(event)
        .register(event_list)
        .register(node)
        .register(node_list)
        .register(graph)
        .register(query)
        .register(mutation)
        .finish()
        .expect("valid schema")
}
