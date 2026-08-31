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

mod enumerate;
mod filter;
mod filter_map;
mod from_fn;
mod map;
pub mod opaque;
mod range_iter;
mod scan;
mod skip_while;
mod slice_iter;
mod take_while;
mod traits;
mod zip;

pub use enumerate::Enumerate;
pub use filter::Filter;
pub use filter_map::FilterMap;
pub use from_fn::{FromFn, from_fn};
pub use map::Map;
pub use opaque::{
    DynExactIter, DynIter, ExactOpaqueIterOwner, ExactSizeOpaqueIter, ExactSizeOpaqueIterFns,
    ExactSizeOpaqueIterKind, OPAQUE_ITER_INLINE_CAP, OpaqueHandle, OpaqueIter, OpaqueIterFns,
    OpaqueIterItem, OpaqueIterKind, OpaqueIterOwner, OpaqueIterSlot, ReusedOpaqueIter,
    ReusedOpaqueIterFns, ReusedOpaqueIterKind, box_dyn_exact_iter, box_dyn_iter, emplace_iter,
};
pub use range_iter::{RangeIter, RangeStep, range, range_step};
pub use scan::Scan;
pub use skip_while::SkipWhile;
pub use slice_iter::SliceIter;
pub use take_while::TakeWhile;
pub use traits::{
    IndexedSource, IndexedStagedIterator, IntoStagedIterator, MinMax, StagedIterator,
};
pub use zip::{Pair, Zip, ZipGetAt, ZipItem, ZipItemAccess, ZipItemType, ZipLen};
