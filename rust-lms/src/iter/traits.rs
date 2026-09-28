//! Core traits for staged iteration.

use crate::control::not;
use crate::func::Ctx;
use crate::num::{Le, Lt, Num, add, gt, le, lt, select};
use crate::staged::{Const, Staged, Var};
use crate::staged_opt::StagedOpt;
use crate::types::{ConstantType, CopyType, DirectValue, StagedType};

use crate::label::Label;
use crate::staged::IntoStaged;

use super::{
    Chain, Filter, FilterMap, FlatMap, IntersectBy, Map, MergeBy, Scan, Skip, SkipWhile, Take,
    TakeWhile, Zip,
};

// =============================================================================
// MinMax sentinels for min/max reductions
// =============================================================================

pub trait MinMax: Copy {
    fn min_sentinel() -> Self; // smallest value, used as max-reduction start
    fn max_sentinel() -> Self; // largest value, used as min-reduction start
}

impl MinMax for i64 {
    fn min_sentinel() -> Self {
        i64::MIN
    }
    fn max_sentinel() -> Self {
        i64::MAX
    }
}
impl MinMax for i8 {
    fn min_sentinel() -> Self {
        i8::MIN
    }
    fn max_sentinel() -> Self {
        i8::MAX
    }
}
impl MinMax for u8 {
    fn min_sentinel() -> Self {
        u8::MIN
    }
    fn max_sentinel() -> Self {
        u8::MAX
    }
}
impl MinMax for i16 {
    fn min_sentinel() -> Self {
        i16::MIN
    }
    fn max_sentinel() -> Self {
        i16::MAX
    }
}
impl MinMax for u16 {
    fn min_sentinel() -> Self {
        u16::MIN
    }
    fn max_sentinel() -> Self {
        u16::MAX
    }
}
impl MinMax for u64 {
    fn min_sentinel() -> Self {
        u64::MIN
    }
    fn max_sentinel() -> Self {
        u64::MAX
    }
}
impl MinMax for f32 {
    fn min_sentinel() -> Self {
        f32::NEG_INFINITY
    }
    fn max_sentinel() -> Self {
        f32::INFINITY
    }
}
impl MinMax for f64 {
    fn min_sentinel() -> Self {
        f64::NEG_INFINITY
    }
    fn max_sentinel() -> Self {
        f64::INFINITY
    }
}
impl MinMax for i32 {
    fn min_sentinel() -> Self {
        i32::MIN
    }
    fn max_sentinel() -> Self {
        i32::MAX
    }
}
impl MinMax for u32 {
    fn min_sentinel() -> Self {
        u32::MIN
    }
    fn max_sentinel() -> Self {
        u32::MAX
    }
}

// =============================================================================
// Cursor / Close: the pull protocol
// =============================================================================

/// An opened iterator: one staged advance (see `docs/pull_iter.md`).
///
/// [`next`](Self::next) emits code that either jumps to `done` (exhausted) or
/// falls through with the next item bound. It consumes the cursor, so each
/// cursor's step is staged **at most once** and code size stays linear in the
/// pipeline — the runtime loop re-runs that one piece of code.
///
/// The step is written in *direct style*: the caller decides what exhaustion
/// means by choosing the label it passes (`iterate`'s exit, a chain's "switch
/// to the second half", a merge's "this side is dry").
///
/// **Fused contract:** once `done` has been taken, the caller never runs the
/// step again. The callers (the driver, `chain`, `zip`) guarantee it, so
/// sources need no "fused" flag.
pub trait Cursor: Sized {
    type Item: StagedType;
    /// What must run after the traversal, on every exit (see [`Close`]).
    type Close: Close;

    /// Emit one advance: jump to `done`, or fall through with the item.
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<Self::Item>, Self::Close);

    /// Stage-0 probe for random access: `Some(len)` when this cursor can also
    /// be driven by an external index through [`next_at`](Self::next_at).
    ///
    /// `zip` asks while it opens, before the driving loop, so any code needed
    /// for the length (a range's element count) lands there and is emitted only
    /// when someone asks. The decision itself is ordinary Rust at stage 0, so it
    /// costs nothing at runtime.
    fn indexed_len(&mut self, ctx: &mut Ctx) -> Option<Var<u64>> {
        let _ = ctx;
        None
    }

    /// Emit the element at `index`, instead of the cursor's own position.
    ///
    /// # Safety
    ///
    /// Only valid when [`indexed_len`](Self::indexed_len) returned `Some(len)`,
    /// and `index < len` at execution.
    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<Self::Item>, Self::Close) {
        let _ = (ctx, index);
        unreachable!("next_at on a cursor without random access")
    }
}

/// Releases what a traversal holds (an extern iterator handle, …).
///
/// Emitted once after the driving loop, which both exhaustion and an early
/// `break_loop` reach, so every opened resource is released exactly once on
/// every exit. `Copy` because it is only stage-0 handles (vars, extern refs).
pub trait Close: Copy {
    fn close(self, ctx: &mut Ctx);
}

impl Close for () {
    fn close(self, _ctx: &mut Ctx) {}
}

impl<A: Close, B: Close> Close for (A, B) {
    fn close(self, ctx: &mut Ctx) {
        self.0.close(ctx);
        self.1.close(ctx);
    }
}

/// The one push driver: open, loop over `next`, close.
///
/// `done` and a consumer's `break_loop` both leave through `iterate`'s exit, so
/// the close runs on every path out of the loop.
pub(crate) fn drive<C, F>(cursor: C, ctx: &mut Ctx, consumer: F)
where
    C: Cursor,
    F: FnOnce(&mut Ctx, Var<C::Item>),
{
    let mut close = None;
    ctx.iterate(|ctx, done| {
        let (item, c) = cursor.next(ctx, done);
        close = Some(c);
        consumer(ctx, item);
    });
    close.expect("the step is staged exactly once").close(ctx);
}

// =============================================================================
// StagedIterator
// =============================================================================

/// A staged iterator.
///
/// The required method is the pull protocol, [`open`](Self::open), which
/// returns a [`Cursor`]. Everything else is provided: [`for_each`](Self::for_each)
/// drives the cursor in one loop, the combinators wrap cursors, and the
/// terminals are built on `for_each`. An iterator with a better push loop may
/// override `for_each`.
///
/// # Example
/// ```ignore
/// let sum = ctx.var(0.0f64);
/// arr.staged_iter().for_each(ctx, move |ctx, elem| {
///     ctx.assign(sum, add(sum, elem));
/// });
/// // `sum` now holds the accumulated result
/// ```
pub trait StagedIterator: Sized {
    type Item: StagedType;
    type Cursor: Cursor<Item = Self::Item>;

    /// Emit the setup — bind handles, call producers, declare state vars — and
    /// return the cursor. Runs once, before the loop that drives it.
    fn open(self, ctx: &mut Ctx) -> Self::Cursor;

    /// Drive a loop over all elements.
    ///
    /// `consumer` is called once at staging time; it emits the per-element
    /// body via the `Ctx` it receives. No `Clone` constraint required.
    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>),
    {
        let cursor = self.open(ctx);
        drive(cursor, ctx, consumer);
    }

    // =========================================================================
    // Combinators
    // =========================================================================

    fn map<U, F, MapOut>(self, f: F) -> Map<Self, F, U>
    where
        U: StagedType,
        F: Fn(Var<Self::Item>) -> MapOut,
        MapOut: Staged<Out = U>,
    {
        Map::new(self, f)
    }

    fn filter<P, Cond>(self, p: P) -> Filter<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = bool>,
    {
        Filter::new(self, p)
    }

    /// Stateful map: thread a mutable `Var<St>` accumulator (initialized to
    /// `init`) through the iteration. `f(ctx, state, elem)` updates `state`;
    /// the post-update `state` is emitted as each element.
    ///
    /// Example — prefix sums:
    /// `iter.scan(0i64, |ctx, acc, x| ctx.store(acc, acc + x))`.
    fn scan<St, Init, F>(self, init: Init, f: F) -> Scan<Self, St, Init, F>
    where
        St: StagedType + crate::types::CopyType + 'static,
        Init: crate::staged::IntoStaged<St>,
        F: Fn(&mut Ctx, Var<St>, Var<Self::Item>) + 'static,
    {
        Scan::new(self, init, f)
    }

    /// Map-and-filter fused: `f` returns a [`StagedOpt`] (typically via
    /// `cond.then_some(value)`); `Some` payloads are kept, `None` dropped. No
    /// `Option` is materialized — one branch per element, value in a register.
    fn filter_map<F, O>(self, f: F) -> FilterMap<Self, F>
    where
        F: Fn(Var<Self::Item>) -> O,
        O: StagedOpt,
    {
        FilterMap::new(self, f)
    }

    /// Yield elements while `p` holds; stop the whole iteration at the first
    /// element where it fails (short-circuits via `break_loop`).
    fn take_while<P, Cond>(self, p: P) -> TakeWhile<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = bool>,
    {
        TakeWhile::new(self, p)
    }

    /// Skip leading elements while `p` holds; yield the rest (starting at the
    /// first element where `p` fails).
    fn skip_while<P, Cond>(self, p: P) -> SkipWhile<Self, P>
    where
        P: Fn(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = bool>,
    {
        SkipWhile::new(self, p)
    }

    /// Pair each element with its zero-based position.
    ///
    /// `it.enumerate().for_each(ctx, |ctx, i, x| ..)` receives the index and the
    /// element separately; used through the combinators the item is a
    /// `ZipItem<u64, _>`, read with `.first()`/`.second()`.
    fn enumerate(self) -> super::Enumerate<Self> {
        super::Enumerate::new(self)
    }

    /// Pair elements of two iterators, stopping at the shorter one — any two
    /// iterators, indexed or not (a slice, an extern stream, a filtered view…).
    ///
    /// When both sides have random access the pair shares one counter against
    /// the hoisted `min` of the lengths, exactly like a hand-written zip loop.
    fn zip<B>(self, other: B) -> Zip<Self, B::Iter>
    where
        B: IntoStagedIterator,
    {
        Zip::new(self, other.staged_iter())
    }

    /// All elements of `self`, then all elements of `other`.
    ///
    /// The consumer is emitted once; a phase flag picks the side. An early exit
    /// (`any`, `take_while`, …) leaves the whole chain.
    fn chain<B>(self, other: B) -> Chain<Self, B::Iter>
    where
        B: IntoStagedIterator,
        B::Iter: StagedIterator<Item = Self::Item>,
    {
        Chain::new(self, other.staged_iter())
    }

    /// Map each element to an iterator and yield the inner elements in order.
    ///
    /// Consumed directly (a terminal, `for_each`) it is two nested loops; pulled
    /// (inside `zip`/`chain`/`merge_by`/`take`…) it is a state machine over one
    /// loop. Either way an early exit releases the open inner iterator.
    fn flat_map<J, F>(self, f: F) -> FlatMap<Self, F>
    where
        F: Fn(Var<Self::Item>) -> J,
        J: IntoStagedIterator,
    {
        FlatMap::new(self, f)
    }

    /// Stable merge of two streams sorted by `le` (`le(x, y)`: `x` may come
    /// before `y`). On ties the element of `self` comes first.
    fn merge_by<B, P, Cond>(self, other: B, le: P) -> MergeBy<Self, B::Iter, P>
    where
        B: IntoStagedIterator,
        B::Iter: StagedIterator<Item = Self::Item>,
        P: Fn(Var<Self::Item>, Var<Self::Item>) -> Cond,
        Cond: Staged<Out = bool> + 'static,
    {
        MergeBy::new(self, other.staged_iter(), le)
    }

    /// Stable merge of two ascending numeric streams.
    #[allow(clippy::type_complexity)]
    fn merge<B>(
        self,
        other: B,
    ) -> MergeBy<
        Self,
        B::Iter,
        fn(Var<Self::Item>, Var<Self::Item>) -> Le<Var<Self::Item>, Var<Self::Item>>,
    >
    where
        B: IntoStagedIterator,
        B::Iter: StagedIterator<Item = Self::Item>,
        Self::Item: Num,
    {
        MergeBy::new(self, other.staged_iter(), le)
    }

    /// The elements `self` and `other` have in common, both sorted by the
    /// strict order `lt` — a merge join. One-to-one in order (multiset
    /// intersection); yields `self`'s element; ends when either side does.
    fn intersect_by<B, P, Cond>(self, other: B, lt: P) -> IntersectBy<Self, B::Iter, P>
    where
        B: IntoStagedIterator,
        B::Iter: StagedIterator<Item = Self::Item>,
        P: Fn(Var<Self::Item>, Var<Self::Item>) -> Cond,
        Cond: Staged<Out = bool> + 'static,
    {
        IntersectBy::new(self, other.staged_iter(), lt)
    }

    /// Common elements of two ascending numeric streams (e.g. common
    /// neighbours from two sorted adjacency lists).
    #[allow(clippy::type_complexity)]
    fn intersect<B>(
        self,
        other: B,
    ) -> IntersectBy<
        Self,
        B::Iter,
        fn(Var<Self::Item>, Var<Self::Item>) -> Lt<Var<Self::Item>, Var<Self::Item>>,
    >
    where
        B: IntoStagedIterator,
        B::Iter: StagedIterator<Item = Self::Item>,
        Self::Item: Num,
    {
        IntersectBy::new(self, other.staged_iter(), lt)
    }

    /// At most the first `n` elements.
    fn take<N>(self, n: N) -> Take<Self, N::Staged>
    where
        N: IntoStaged<u64>,
    {
        Take::new(self, n.into_staged())
    }

    /// All but the first `n` elements.
    fn skip<N>(self, n: N) -> Skip<Self, N::Staged>
    where
        N: IntoStaged<u64>,
    {
        Skip::new(self, n.into_staged())
    }

    // =========================================================================
    // Terminal operations
    // =========================================================================

    /// Fold every element into an accumulator: `acc = f(acc, elem)`, starting
    /// from `init`. Returns the accumulator.
    ///
    /// `f` is called once at staging time and builds the update expression; the
    /// accumulator lives in a register across the loop.
    ///
    /// ```ignore
    /// let total = slice.staged_iter().fold(ctx, 0i64, |acc, x| acc + x);
    /// ```
    ///
    /// Several accumulators at once, or updates with side effects, are a
    /// [`for_each`](Self::for_each) over vars declared before the loop.
    fn fold<A, Init, F, E>(self, ctx: &mut Ctx, init: Init, f: F) -> Var<A>
    where
        A: StagedType + CopyType + 'static,
        Init: IntoStaged<A>,
        Init::Staged: 'static,
        F: FnOnce(Var<A>, Var<Self::Item>) -> E,
        E: Staged<Out = A> + 'static,
    {
        let acc = ctx.var(init);
        self.for_each(ctx, move |ctx, elem| {
            ctx.store(acc, f(acc, elem));
        });
        acc
    }

    /// Branchless conditional fold: `acc = if pred(elem) { f(acc, elem) } else
    /// { acc }`, lowered as a `select` rather than a branch.
    ///
    /// Equivalent to `self.filter(pred).fold(ctx, init, f)`, but the loop body
    /// has no data-dependent branch, so it stays vectorizable/unrollable (and
    /// `count_if` becomes a single conditional increment, e.g. `csinc`).
    ///
    /// **`f` is evaluated for every element**, whether `pred` holds or not, so
    /// it must be cheap and total — no division `pred` is guarding against a
    /// zero, no load `pred` is guarding against being out of range. For those,
    /// use `filter(..).fold(..)`, which branches.
    fn fold_if<A, Init, P, Cond, F, E>(self, ctx: &mut Ctx, init: Init, pred: P, f: F) -> Var<A>
    where
        A: DirectValue + 'static,
        Init: IntoStaged<A>,
        Init::Staged: 'static,
        Self::Item: CopyType + 'static,
        P: FnOnce(Var<Self::Item>) -> Cond,
        Cond: Staged<Out = bool> + 'static,
        F: FnOnce(Var<A>, Var<Self::Item>) -> E,
        E: Staged<Out = A> + 'static,
    {
        self.fold(ctx, init, move |acc, elem| {
            select(pred(elem), f(acc, elem), acc)
        })
    }

    /// Sum all elements, starting from zero.
    fn sum(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: Num,
        <Self::Item as StagedType>::RuntimeValue: Default,
    {
        self.fold(
            ctx,
            Const::<Self::Item>::new(Default::default()),
            |acc, x| acc + x,
        )
    }

    /// Count elements passing through (including any upstream filter).
    fn count(self, ctx: &mut Ctx) -> Var<u64>
    where
        Self::Item: 'static,
    {
        self.fold(ctx, 0u64, |acc, _| acc + 1u64)
    }

    /// Branchless count of elements satisfying `pred` — a [`fold_if`](Self::fold_if).
    fn count_if<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<u64>
    where
        Self::Item: CopyType + 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = bool> + 'static,
    {
        self.fold_if(ctx, 0u64, pred, |acc, _| acc + 1u64)
    }

    /// Branchless sum of elements satisfying `pred` — a [`fold_if`](Self::fold_if).
    fn sum_if<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<Self::Item>
    where
        Self::Item: Num + DirectValue,
        <Self::Item as StagedType>::RuntimeValue: Default,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = bool> + 'static,
    {
        let zero = Const::<Self::Item>::new(Default::default());
        self.fold_if(ctx, zero, pred, |acc, x| acc + x)
    }

    /// Find the minimum element. Starts at the type's maximum sentinel.
    fn min(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: Num,
        <Self::Item as StagedType>::RuntimeValue: MinMax,
    {
        let sentinel = <Self::Item as StagedType>::RuntimeValue::max_sentinel();
        // Branchless: an unconditional store of a cmov keeps the body
        // vectorizable/unrollable.
        self.fold(ctx, Const::<Self::Item>::new(sentinel), |acc, x| {
            select(lt(x, acc), x, acc)
        })
    }

    /// Find the maximum element. Starts at the type's minimum sentinel.
    fn max(self, ctx: &mut Ctx) -> Var<Self::Item>
    where
        Self::Item: Num,
        <Self::Item as StagedType>::RuntimeValue: MinMax,
    {
        let sentinel = <Self::Item as StagedType>::RuntimeValue::min_sentinel();
        self.fold(ctx, Const::<Self::Item>::new(sentinel), |acc, x| {
            select(gt(x, acc), x, acc)
        })
    }

    // =========================================================================
    // Short-circuiting terminals
    //
    // These drive the ordinary `for_each` loop but `break_loop` out of it once
    // the answer is known. Because combinators (`map`/`filter`/…) introduce
    // only `if_then`s — never loops — the `break_loop` always targets the
    // source's single iteration loop, so these compose freely after `filter`,
    // `map`, etc. (Eager terminals like `sum`/`fold` emit no break and keep
    // their optimal loop body.)
    // =========================================================================

    /// `true` as soon as any element satisfies `pred` (short-circuits).
    fn any<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<bool>
    where
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = bool> + 'static,
    {
        let found = ctx.var(false);
        self.for_each(ctx, move |ctx, elem| {
            ctx.if_then(pred(elem), move |ctx| {
                ctx.store(found, true);
                ctx.break_loop();
            });
        });
        found
    }

    /// `true` only if every element satisfies `pred` (short-circuits on the
    /// first failure).
    fn all<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<bool>
    where
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = bool> + 'static,
    {
        let result = ctx.var(true);
        self.for_each(ctx, move |ctx, elem| {
            ctx.if_then(not(pred(elem)), move |ctx| {
                ctx.store(result, false);
                ctx.break_loop();
            });
        });
        result
    }

    /// Index (in this iterator's sequence, i.e. after any `filter`) of the
    /// first element satisfying `pred`, or the total element count if none
    /// match (short-circuits).
    fn position<P, Cond>(self, ctx: &mut Ctx, pred: P) -> Var<u64>
    where
        Self::Item: 'static,
        P: Fn(Var<Self::Item>) -> Cond + 'static,
        Cond: Staged<Out = bool> + 'static,
    {
        // `idx` counts elements seen; on a match we break *before* incrementing,
        // so it holds the match position. With no match it ends at the count.
        let idx = ctx.var(0u64);
        self.for_each(ctx, move |ctx, elem| {
            ctx.if_then(pred(elem), move |ctx| {
                ctx.break_loop();
            });
            ctx.store(idx, add(idx, 1u64));
        });
        idx
    }

    /// First `Some` produced by `f`, short-circuiting. Returns `(value, found)`:
    /// when `found` is `false` the iterator was exhausted and `value` holds the
    /// default — check `found` before using `value`. No `Option` is
    /// materialized; `f` is typically `|x| cond.then_some(mapped)`.
    fn find_map<F, O>(self, ctx: &mut Ctx, f: F) -> (Var<O::Item>, Var<bool>)
    where
        F: Fn(Var<Self::Item>) -> O + 'static,
        O: StagedOpt + 'static,
        O::Item: ConstantType + CopyType + 'static,
        <O::Item as StagedType>::RuntimeValue: Default,
    {
        let result = ctx.var(Const::<O::Item>::new(Default::default()));
        let found = ctx.var(false);
        self.for_each(ctx, move |ctx, elem| {
            f(elem).eliminate(
                ctx,
                move |ctx, v| {
                    ctx.store(result, v);
                    ctx.store(found, true);
                    ctx.break_loop();
                },
                |_| {},
            );
        });
        (result, found)
    }
}

// =============================================================================
// IndexedStagedIterator
// =============================================================================

/// A staged iterator with a length known before iterating, enabling `rev`.
#[allow(clippy::len_without_is_empty)]
pub trait IndexedStagedIterator: StagedIterator {
    /// The type of the length expression (e.g. `SliceLen<S>`, `Sub<End, Start>`).
    type LenExpr: Staged<Out = u64> + Clone + 'static;

    /// Return the number of elements as a staged expression.
    fn len(&self) -> Self::LenExpr;

    /// Iterate from the last element to the first.
    ///
    /// Needs random access, so it is bounded on `IndexedSource`: a stream can
    /// only be pulled forwards.
    fn rev(self) -> super::Rev<Self>
    where
        Self: IndexedSource,
    {
        super::Rev::new(self)
    }
}

// =============================================================================
// IndexedSource
// =============================================================================

/// A typed random-access data source: element `i` as an expression. Powers
/// `rev` and random access into zipped pairs.
pub trait IndexedSource: Clone + 'static {
    type Item: StagedType;
    type LenExpr: Staged<Out = u64> + Clone + 'static;
    type GetExpr: Staged<Out = Self::Item> + 'static;

    fn count(&self) -> Self::LenExpr;
    /// Return an expression that reads the item at `index` without checking it.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than the length returned by `len` for
    /// this source.
    unsafe fn get_at(self, index: Var<u64>) -> Self::GetExpr;
}

// =============================================================================
// IntoStagedIterator
// =============================================================================

pub trait IntoStagedIterator {
    type Iter: StagedIterator;
    fn staged_iter(self) -> Self::Iter;

    /// Alias for [`staged_iter`](Self::staged_iter), reading more naturally at
    /// call sites that consume `self`: `arr.into_staged_iter()`.
    fn into_staged_iter(self) -> Self::Iter
    where
        Self: Sized,
    {
        self.staged_iter()
    }
}
