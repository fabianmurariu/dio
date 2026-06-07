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

mod filter;
mod filter_map;
mod map;
pub mod opaque;
mod range_iter;
mod scan;
mod skip_while;
mod slice_iter;
mod take_while;
mod traits;
mod zip;

pub use filter::Filter;
pub use filter_map::FilterMap;
pub use map::Map;
pub use opaque::{
    box_dyn_exact_iter, box_dyn_iter, emplace_iter, DynExactIter, DynIter, ExactSizeOpaqueIter,
    ExactSizeOpaqueIterFns, ExactSizeOpaqueIterKind, OpaqueHandle, OpaqueIter, OpaqueIterFns,
    OpaqueIterKind, OpaqueIterSlot, RegisterScalar, ReusedOpaqueIter, ReusedOpaqueIterFns,
    ReusedOpaqueIterKind, OPAQUE_ITER_INLINE_CAP,
};
pub use range_iter::{range, range_step, RangeIter, RangeStep};
pub use scan::Scan;
pub use skip_while::SkipWhile;
pub use slice_iter::SliceIter;
pub use take_while::TakeWhile;
pub use traits::{
    IndexedSource, IndexedStagedIterator, IntoStagedIterator, MinMax, StagedIterator,
};
pub use zip::Zip;
