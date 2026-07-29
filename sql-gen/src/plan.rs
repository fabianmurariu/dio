//! Minimal relational-algebra plan — the paper's `Operator` tree.
//!
//! Hand-built for now; a SQL front-end (parser → `Operator`) is a later step.
//! The plan is entirely stage-0 (static) data: it is walked at code-generation
//! time to *emit* a kernel, never at run time.

use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;

/// Static column schema: the physical type of each column, positional.
///
/// This is the static half of the mixed-stage record — known at code-generation
/// time and used to drive per-column type dispatch in [`crate::gen`].
#[derive(Clone, Debug)]
pub struct Schema(pub Vec<DataType>);

impl Schema {
    pub fn from_record_batch(rb: &RecordBatch) -> Self {
        Schema(
            rb.schema()
                .fields()
                .iter()
                .map(|f| f.data_type().clone())
                .collect(),
        )
    }
}

/// A scalar reference in a predicate: a column (by index) or a literal.
#[derive(Clone, Debug)]
pub enum Expr {
    Col(usize),
    LitI32(i32),
}

/// A boolean predicate over a row.
#[derive(Clone, Debug)]
pub enum Predicate {
    Eq(Expr, Expr),
    Lt(Expr, Expr),
}

/// Push-model relational operators (a starter subset).
///
/// `Scan` is the single input batch; `Filter`/`Project` transform the row
/// stream. Each operator describes *what records it emits to its consumer*,
/// mirroring the paper's `execOp`.
#[derive(Clone, Debug)]
pub enum Operator {
    Scan,
    Filter(Predicate, Box<Operator>),
    Project(Vec<usize>, Box<Operator>),
}
