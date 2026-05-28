//! Push-based staged iterators with CPS-style fusion.
//!
//! This module provides composable iterators that compile to efficient fused loops.
//!
//! # Core Concepts
//!
//! - **Push-based iteration**: Iterator controls the loop, consumer provides callback
//! - **CPS-style fusion**: Combinators wrap callbacks, enabling single-loop compilation
//! - **IndexedStagedIterator**: Subset of iterators supporting `zip` (like Rayon)
//!
//! # Example
//!
//! ```ignore
//! slice.staged_iter()
//!     .map(|x| add(*x, 2.0))
//!     .filter(|x| gt(*x, 3.0))
//!     .fold(builder, (0u64, 0.0f64), |(count, sum), x| {
//!         (add(*count, 1u64), add(*sum, *x))
//!     })
//! ```

mod traits;
mod accumulator;
mod slice_iter;
mod range_iter;
mod map;
mod filter;
mod zip;

pub use traits::{StagedIterator, IndexedStagedIterator, IndexedSource, IntoStagedIterator, MinMax};
pub use accumulator::{Accumulator, FoldExpr, IntoAccumulatorUpdate};
pub use slice_iter::SliceIter;
pub use range_iter::{RangeIter, range};
pub use map::Map;
pub use filter::Filter;
pub use zip::Zip;
