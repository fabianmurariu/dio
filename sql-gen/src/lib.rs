//! `sql-gen`: SQL → staged `rust-lms` kernels over Arrow columns.
//!
//! Follows the Rompf & Amin "SQL to C in 500 lines" design (`docs/sql_to_c.pdf`):
//! a push-model interpreter over relational operators ([`exec`]) has a staged
//! twin ([`codegen`]) that *emits* JIT code instead of running — the first
//! Futamura projection. Rows are mixed-stage ([`value::Row`]): a static arrow
//! schema plus a `Vec` of stage-1 `Var` handles.
//!
//! Pipeline: datafusion parses SQL and produces a (pull-based) `LogicalPlan`,
//! which [`sql`] lowers into our (push-based) [`plan::Operator`] tree — reusing
//! datafusion [`datafusion_expr::Expr`] verbatim for scalar expressions. Current
//! scope is a thin vertical slice: `Scan`/`Filter`/`Project` with a `count(*)`
//! terminal and primitive columns.

pub mod catalog;
pub mod codegen;
pub mod exec;
pub mod plan;
pub mod sql;
pub mod value;

pub use catalog::Catalog;
pub use codegen::{BatchSource, OutSink, gen_collect, gen_count};
pub use plan::Operator;
pub use sql::sql_to_operator;
pub use value::{ColVal, Nullness, Row};
