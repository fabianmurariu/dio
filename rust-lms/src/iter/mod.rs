//! Push-based staged iterators with imperative consumer API.
//!
//! # Core concepts
//!
//! - **Imperative consumers**: `for_each(ctx, |ctx, elem| { ctx.assign(...); })` — no `Clone` constraints.
//! - **Combinators**: `map`, `filter` wrap consumers before passing to the source iterator.
//! - **IndexedStagedIterator**: slices and ranges, enabling `zip`.
//!
//! # Example
//!
//! ```ignore
//! let sum = ctx.var(0.0f64);
//! slice.staged_iter()
//!      .filter(|x| gt(x, 0.0))
//!      .for_each(ctx, move |ctx, elem| {
//!          ctx.assign(sum, add(sum, elem));
//!      });
//! ```

mod traits;
mod slice_iter;
mod range_iter;
mod map;
mod filter;
mod zip;
mod accumulator;
mod early_exit;

pub use traits::{
    IndexedSource, IndexedStagedIterator, IntoStagedIterator, MinMax, StagedIterator,
};
pub use early_exit::IndexedEarlyExit;
pub use slice_iter::SliceIter;
pub use range_iter::{range, range_step, RangeIter, RangeStep};
pub use map::Map;
pub use filter::Filter;
pub use zip::Zip;
