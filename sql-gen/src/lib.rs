//! `sql-gen`: SQL → staged `rust-lms` kernels over Arrow columns.
//!
//! Follows the Rompf & Amin "SQL to C in 500 lines" design (`docs/sql_to_c.pdf`):
//! [`codegen`] walks a relational-operator tree and *emits* JIT code — the first
//! Futamura projection. Rows are mixed-stage ([`value::Row`]): a static arrow
//! schema plus a `Vec` of stage-1 `Var` handles.
//!
//! Pipeline: datafusion parses SQL into a (pull-based) `LogicalPlan`, which
//! [`sql`] lowers into our (push-based) [`plan::Operator`] tree — reusing
//! datafusion [`datafusion_expr::Expr`] verbatim for scalar expressions.
//! [`exec_jit`] runs the whole thing and returns a `RecordBatch`. Scope:
//! `Scan`/`Filter`/`Project` + scalar (no-GROUP-BY) aggregates, primitive
//! columns.

pub mod catalog;
pub mod codegen;
pub mod group;
pub mod output;
pub mod plan;
pub mod run;
pub mod runtime;
pub mod sql;
pub mod value;

pub use catalog::Catalog;
pub use codegen::{BatchSource, gen_collect, group_template};
pub use plan::Operator;
pub use run::exec_jit;
pub use sql::sql_to_operator;
pub use value::{ColVal, Nullness, Row};
