//! Zip combinator — pairs elements from two sources at the same index.

use crate::types::IntCmp;
use rust_lms_derive::StagedType;

use crate::func::Ctx;
use crate::label::Label;
use crate::num::{add, ge};
use crate::staged::{CompilationContext, Staged, Value, Var};
use crate::r#struct::{Field, LoadField, load_field_unchecked};
use crate::types::{CopyType, Shape, StagedType};

use super::traits::{Cursor, IndexedSource, IndexedStagedIterator, StagedIterator};

/// Element yielded by a zipped iterator.
///
/// The `StagedType`/`CopyType` impls and the `ZipItemType` field-token module
/// are macro-generated. `Copy`/`Clone` are hand-written so the bounds land on
/// `A::RuntimeValue`/`B::RuntimeValue` rather than the marker types `A`/`B`.
#[derive(StagedType)]
#[repr(C)]
pub struct ZipItem<A, B>
where
    A: StagedType,
    B: StagedType,
{
    #[staged(A)]
    pub first: A::RuntimeValue,
    #[staged(B)]
    pub second: B::RuntimeValue,
}

impl<A, B> Clone for ZipItem<A, B>
where
    A: StagedType,
    B: StagedType,
    A::RuntimeValue: Copy,
    B::RuntimeValue: Copy,
{
    fn clone(&self) -> Self {
        *self
    }
}

impl<A, B> Copy for ZipItem<A, B>
where
    A: StagedType,
    B: StagedType,
    A::RuntimeValue: Copy,
    B::RuntimeValue: Copy,
{
}

/// Convenience field access for staged zipped items.
pub trait ZipItemAccess<A, B>: Staged<Out = ZipItem<A, B>> + Sized
where
    A: CopyType,
    B: CopyType,
{
    fn first(self) -> LoadField<Self, ZipItemType::__field_first<A, B>> {
        // SAFETY: `Self::Out` is exactly the field descriptor's `ZipItem`
        // parent, established by this trait's bound.
        unsafe { load_field_unchecked(self, ZipItemType::first::<A, B>()) }
    }

    fn second(self) -> LoadField<Self, ZipItemType::__field_second<A, B>> {
        // SAFETY: `Self::Out` is exactly the field descriptor's `ZipItem`
        // parent, established by this trait's bound.
        unsafe { load_field_unchecked(self, ZipItemType::second::<A, B>()) }
    }
}

impl<A, B, S> ZipItemAccess<A, B> for S
where
    A: CopyType,
    B: CopyType,
    S: Staged<Out = ZipItem<A, B>> + Sized,
{
}

/// Combinator that pairs elements of two iterators, stopping at the shorter.
///
/// Created by [`StagedIterator::zip`]. Any two iterators zip; two indexed ones
/// share a single counter (see [`Cursor::indexed_len`]).
pub struct Zip<I, S> {
    pub(crate) iter: I,
    pub(crate) other: S,
}

impl<I, S> Zip<I, S> {
    pub(crate) fn new(iter: I, other: S) -> Self {
        Zip { iter, other }
    }
}

impl<I: Clone, S: Clone> Clone for Zip<I, S> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            other: self.other.clone(),
        }
    }
}

impl<I: Copy, S: Copy> Copy for Zip<I, S> {}

/// Length of a zip: the smaller of its two source lengths.
pub struct ZipLen<L, R> {
    left: L,
    right: R,
}

impl<L, R> ZipLen<L, R> {
    fn new(left: L, right: R) -> Self {
        Self { left, right }
    }
}

impl<L: Clone, R: Clone> Clone for ZipLen<L, R> {
    fn clone(&self) -> Self {
        Self {
            left: self.left.clone(),
            right: self.right.clone(),
        }
    }
}

impl<L: Copy, R: Copy> Copy for ZipLen<L, R> {}

unsafe impl<L, R> Staged for ZipLen<L, R>
where
    L: Staged<Out = u64>,
    R: Staged<Out = u64>,
{
    type Out = u64;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let left = self.left.codegen(ctx);
        let right = self.right.codegen(ctx);
        let left_is_shorter = ctx.icmp(IntCmp::Ult, left.leaf(), right.leaf());
        Value::scalar(ctx.select(left_is_shorter, left.leaf(), right.leaf()))
    }
}

/// Random access expression for a zipped pair at `index`.
pub struct ZipGetAt<I, S> {
    iter: I,
    other: S,
    index: Var<u64>,
}

impl<I, S> ZipGetAt<I, S> {
    fn new(iter: I, other: S, index: Var<u64>) -> Self {
        Self { iter, other, index }
    }
}

impl<I: Clone, S: Clone> Clone for ZipGetAt<I, S> {
    fn clone(&self) -> Self {
        Self {
            iter: self.iter.clone(),
            other: self.other.clone(),
            index: self.index,
        }
    }
}

impl<I: Copy, S: Copy> Copy for ZipGetAt<I, S> {}

unsafe impl<I, S> Staged for ZipGetAt<I, S>
where
    I: IndexedSource + Clone,
    S: IndexedSource + Clone,
    <I as IndexedSource>::Item: CopyType + 'static,
    <S as IndexedSource>::Item: CopyType + 'static,
    <I as IndexedSource>::GetExpr: 'static,
    <S as IndexedSource>::GetExpr: 'static,
{
    type Out = ZipItem<<I as IndexedSource>::Item, <S as IndexedSource>::Item>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // SAFETY: `ZipGetAt` is only constructed by a bounded zip loop or by
        // `IndexedSource::get_at`, whose caller supplies the same bound.
        let first = unsafe { IndexedSource::get_at(self.iter.clone(), self.index) }.codegen(ctx);
        // SAFETY: the zip length is the minimum of both source lengths.
        let second = unsafe { IndexedSource::get_at(self.other.clone(), self.index) }.codegen(ctx);

        pair::<<I as IndexedSource>::Item, <S as IndexedSource>::Item>(ctx, first, second)
    }
}

/// Builds a [`ZipItem`] from two arbitrary staged expressions.
///
/// [`ZipGetAt`] pairs two *indexed sources* at the same index; this pairs two
/// values that are already in hand, which is what [`Enumerate`](super::Enumerate)
/// needs — its source may be any iterator, indexed or not.
pub struct Pair<A, B> {
    first: A,
    second: B,
}

impl<A, B> Pair<A, B> {
    pub fn new(first: A, second: B) -> Self {
        Pair { first, second }
    }
}

impl<A: Clone, B: Clone> Clone for Pair<A, B> {
    fn clone(&self) -> Self {
        Pair {
            first: self.first.clone(),
            second: self.second.clone(),
        }
    }
}

impl<A: Copy, B: Copy> Copy for Pair<A, B> {}

// SAFETY: yields a `ZipItem` in the shape it declares: both fields' leaves, or a stack
// slot sized and aligned for it with both fields at their declared offsets.
unsafe impl<A, B> Staged for Pair<A, B>
where
    A: Staged,
    B: Staged,
    A::Out: CopyType + 'static,
    B::Out: CopyType + 'static,
{
    type Out = ZipItem<A::Out, B::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let first = self.first.codegen(ctx);
        let second = self.second.codegen(ctx);

        pair::<A::Out, B::Out>(ctx, first, second)
    }
}

/// A `ZipItem<A, B>` of two values.
fn pair<A: CopyType, B: CopyType>(
    ctx: &mut CompilationContext,
    first: Value,
    second: Value,
) -> Value {
    ctx.struct_value::<ZipItem<A, B>>(&[
        (
            Shape::of::<A>(),
            ZipItemType::__field_first::<A, B>::OFFSET,
            first,
        ),
        (
            Shape::of::<B>(),
            ZipItemType::__field_second::<A, B>::OFFSET,
            second,
        ),
    ])
}

impl<A, B> StagedIterator for Zip<A, B>
where
    A: StagedIterator,
    B: StagedIterator,
    A::Item: CopyType + 'static,
    B::Item: CopyType + 'static,
{
    type Item = ZipItem<A::Item, B::Item>;
    type Cursor = ZipCursor<A::Cursor, B::Cursor>;

    /// Open both sides, then choose the plan at stage 0: when both have random
    /// access, one shared counter against the hoisted `min` of the lengths —
    /// the loop a hand-written zip over two slices would be. Otherwise each
    /// side pulls its own element.
    fn open(self, ctx: &Ctx) -> Self::Cursor {
        let mut a = self.iter.open(ctx);
        let mut b = self.other.open(ctx);
        let shared = match (a.indexed_len(ctx), b.indexed_len(ctx)) {
            (Some(len_a), Some(len_b)) => Some(SharedIndex {
                len: ctx.bind(ZipLen::new(len_a, len_b)),
                pos: ctx.var(0u64),
            }),
            _ => None,
        };
        ZipCursor { a, b, shared }
    }
}

/// A counter shared by both sides of an indexed zip.
#[derive(Clone, Copy)]
pub struct SharedIndex {
    len: Var<u64>,
    pos: Var<u64>,
}

/// The cursor of a [`Zip`]. Random access when both sides are.
pub struct ZipCursor<A, B> {
    a: A,
    b: B,
    /// `Some` when both sides are indexed (decided in `open`).
    shared: Option<SharedIndex>,
}

impl<A, B> Cursor for ZipCursor<A, B>
where
    A: Cursor,
    B: Cursor,
    A::Item: CopyType + 'static,
    B::Item: CopyType + 'static,
{
    type Item = ZipItem<A::Item, B::Item>;
    type Close = (A::Close, B::Close);

    fn next(self, ctx: &Ctx, done: Label<'_>) -> (Var<Self::Item>, Self::Close) {
        match self.shared {
            Some(SharedIndex { len, pos }) => {
                ctx.exit_if(ge(pos, len), done);
                // SAFETY: `pos < len = min(len_a, len_b)`.
                let (x, ca) = unsafe { self.a.next_at(ctx, pos) };
                // SAFETY: as above.
                let (y, cb) = unsafe { self.b.next_at(ctx, pos) };
                ctx.store(pos, add(pos, 1u64));
                (ctx.bind(Pair::new(x, y)), (ca, cb))
            }
            None => {
                // `a` first: if `b` is then exhausted, `a`'s element is dropped,
                // as in `std::iter::Zip`.
                let (x, ca) = self.a.next(ctx, done);
                let (y, cb) = self.b.next(ctx, done);
                (ctx.bind(Pair::new(x, y)), (ca, cb))
            }
        }
    }

    fn indexed_len(&mut self, _ctx: &Ctx) -> Option<Var<u64>> {
        self.shared.map(|s| s.len)
    }

    unsafe fn next_at(self, ctx: &Ctx, index: Var<u64>) -> (Var<Self::Item>, Self::Close) {
        // SAFETY: `index < len = min(len_a, len_b)`, forwarded from the caller.
        let (x, ca) = unsafe { self.a.next_at(ctx, index) };
        // SAFETY: as above.
        let (y, cb) = unsafe { self.b.next_at(ctx, index) };
        (ctx.bind(Pair::new(x, y)), (ca, cb))
    }
}

impl<I, S> IndexedStagedIterator for Zip<I, S>
where
    I: IndexedStagedIterator + IndexedSource + Clone + 'static,
    <I as IndexedSource>::Item: CopyType + 'static,
    S: StagedIterator + IndexedSource + Clone + 'static,
    <I as StagedIterator>::Item: CopyType + 'static,
    <S as StagedIterator>::Item: CopyType + 'static,
    <S as IndexedSource>::Item: CopyType + 'static,
    <I as IndexedSource>::GetExpr: 'static,
    <S as IndexedSource>::GetExpr: 'static,
{
    type LenExpr = ZipLen<<I as IndexedSource>::LenExpr, <S as IndexedSource>::LenExpr>;

    fn len(&self) -> Self::LenExpr {
        ZipLen::new(
            IndexedSource::count(&self.iter),
            IndexedSource::count(&self.other),
        )
    }
}

impl<I, S> IndexedSource for Zip<I, S>
where
    I: IndexedSource + Clone + 'static,
    <I as IndexedSource>::Item: CopyType + 'static,
    S: IndexedSource + Clone + 'static,
    <S as IndexedSource>::Item: CopyType + 'static,
    <I as IndexedSource>::GetExpr: 'static,
    <S as IndexedSource>::GetExpr: 'static,
{
    type Item = ZipItem<<I as IndexedSource>::Item, <S as IndexedSource>::Item>;
    type LenExpr = ZipLen<<I as IndexedSource>::LenExpr, <S as IndexedSource>::LenExpr>;
    type GetExpr = ZipGetAt<I, S>;

    fn count(&self) -> Self::LenExpr {
        ZipLen::new(
            IndexedSource::count(&self.iter),
            IndexedSource::count(&self.other),
        )
    }

    unsafe fn get_at(self, index: Var<u64>) -> Self::GetExpr {
        ZipGetAt::new(self.iter, self.other, index)
    }
}
