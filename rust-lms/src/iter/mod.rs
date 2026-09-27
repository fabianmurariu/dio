//! Staged iterators: pull-based cursors, driven by one fused loop.
//!
//! # Core concepts
//!
//! - **Pull protocol**: every iterator [`open`](StagedIterator::open)s into a
//!   [`Cursor`] whose step either jumps to a `done` label or yields an item
//!   (see `docs/pull_iter.md`). That is what makes `zip` and `chain` work on
//!   any pair of iterators — slices, ranges, extern streams, filtered views.
//! - **Imperative consumers**: `for_each(ctx, |ctx, elem| { ctx.assign(...); })`
//!   drives the cursor in a single loop — no `Clone` constraints.
//! - **Random access**: slice, range and `rev` cursors (and `map`/`enumerate`/
//!   `take`/`skip`/`zip` over them) can be driven by an index, so a `zip` of
//!   indexed sides shares one counter.
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

mod chain;
mod enumerate;
mod filter;
mod filter_map;
mod flat_map;
mod from_fn;
mod map;
mod merge;
pub mod opaque;
mod range_iter;
mod rev;
mod scan;
mod skip_while;
mod slice_iter;
mod take;
mod take_while;
mod traits;
mod zip;

pub use chain::{Chain, ChainCursor};
pub use enumerate::{Enumerate, EnumerateCursor};
pub use filter::{Filter, FilterCursor};
pub use filter_map::{FilterMap, FilterMapCursor};
pub use flat_map::{CloseIfActive, FlatMap, FlatMapCursor};
pub use from_fn::{FromFn, from_fn};
pub use map::{Map, MapCursor};
pub use merge::{MergeBy, MergeCursor};
pub use opaque::{
    DynExactIter, DynIter, ExactOpaqueIterOwner, ExactSizeOpaqueIter, ExactSizeOpaqueIterFns,
    ExactSizeOpaqueIterKind, OPAQUE_ITER_INLINE_CAP, OpaqueHandle, OpaqueIter, OpaqueIterFns,
    OpaqueIterItem, OpaqueIterKind, OpaqueIterOwner, OpaqueIterSlot, ReusedOpaqueIter,
    ReusedOpaqueIterFns, ReusedOpaqueIterKind, box_dyn_exact_iter, box_dyn_iter, emplace_iter,
};
pub use range_iter::{RangeCursor, RangeIter, RangeLen, RangeStep, range, range_step};
pub use rev::{Rev, RevCursor};
pub use scan::{Scan, ScanCursor};
pub use skip_while::{SkipWhile, SkipWhileCursor};
pub use slice_iter::{SliceCursor, SliceIter};
pub use take::{Skip, SkipCursor, Take, TakeCursor};
pub use take_while::{TakeWhile, TakeWhileCursor};
pub use traits::{
    Close, Cursor, IndexedSource, IndexedStagedIterator, IntoStagedIterator, MinMax, StagedIterator,
};
pub use zip::{
    Pair, SharedIndex, Zip, ZipCursor, ZipGetAt, ZipItem, ZipItemAccess, ZipItemType, ZipLen,
};

/// Every combinator is its own [`IntoStagedIterator`], so it can be the
/// argument of `zip`/`chain` (slices and ranges have their own impls).
macro_rules! into_staged_iter_is_self {
    ($($ty:ident<$($p:ident),*>),* $(,)?) => {$(
        impl<$($p),*> IntoStagedIterator for $ty<$($p),*>
        where
            $ty<$($p),*>: StagedIterator,
        {
            type Iter = Self;
            fn staged_iter(self) -> Self {
                self
            }
        }
    )*};
}

into_staged_iter_is_self!(
    Chain<A, B>,
    Enumerate<I>,
    Filter<I, P>,
    FilterMap<I, F>,
    FlatMap<I, F>,
    FromFn<F, O>,
    Map<I, F, U>,
    MergeBy<A, B, P>,
    Rev<I>,
    Scan<I, St, Init, F>,
    Skip<I, N>,
    SkipWhile<I, P>,
    Take<I, N>,
    TakeWhile<I, P>,
    Zip<A, B>,
);

impl<K: OpaqueIterKind, H> IntoStagedIterator for OpaqueIter<K, H>
where
    OpaqueIter<K, H>: StagedIterator,
{
    type Iter = Self;
    fn staged_iter(self) -> Self {
        self
    }
}

impl<K: ExactSizeOpaqueIterKind, H> IntoStagedIterator for ExactSizeOpaqueIter<K, H>
where
    ExactSizeOpaqueIter<K, H>: StagedIterator,
{
    type Iter = Self;
    fn staged_iter(self) -> Self {
        self
    }
}

impl<K: ReusedOpaqueIterKind> IntoStagedIterator for ReusedOpaqueIter<K> {
    type Iter = Self;
    fn staged_iter(self) -> Self {
        self
    }
}
