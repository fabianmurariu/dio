//! `sql-gen`: relational-algebra → staged `rust-lms` kernels over Arrow columns.
//!
//! Follows the Rompf & Amin "SQL to C in 500 lines" design (`docs/sql_to_c.pdf`):
//! a push-model interpreter over relational operators ([`exec`]) has a staged
//! twin ([`codegen`]) that *emits* JIT code instead of running — the first
//! Futamura projection. Rows are mixed-stage ([`value::Row`]): a static
//! [`plan::Schema`] plus a `Vec` of stage-1 `Var` handles.
//!
//! Current scope is a thin vertical slice: `Scan`/`Filter`/`Project` with a
//! `count(*)` terminal, primitive columns, hand-built plans (no SQL parser yet).

pub mod codegen;
pub mod exec;
pub mod plan;
pub mod value;

pub use codegen::{BatchSource, gen_count};
pub use plan::{Expr, Operator, Predicate, Schema};
pub use value::{ColVal, Row};
