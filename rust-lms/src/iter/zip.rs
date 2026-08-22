//! Zip combinator — pairs elements from two sources at the same index.

use cranelift_codegen::ir::{
    condcodes::IntCC, StackSlotData, StackSlotKind, Value,
};

use rust_lms_derive::StagedType;

use crate::func::Ctx;
use crate::num::{add, lt};
use crate::r#struct::{load_field_unchecked, Field, LoadField};
use crate::staged::{CompilationContext, Staged, Var};
use crate::types::{CopyType, StagedType};

use super::traits::{IndexedSource, IndexedStagedIterator, StagedIterator};

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

/// Combinator that pairs elements from two sources at the same 0-based index.
///
/// Created by `indexed_iter.zip(secondary)`. Use `for_each` to drive the loop.
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
        let left_is_shorter = ctx.icmp(IntCC::UnsignedLessThan, left, right);
        ctx.select(left_is_shorter, left, right)
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

        let align_shift = Self::Out::align_of().trailing_zeros() as u8;
        let stack_slot = ctx.create_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            Self::Out::size_of() as u32,
            align_shift,
        ));
        let slot_ptr = ctx.stack_addr(stack_slot, 0);

        store_value::<<I as IndexedSource>::Item>(
            ctx,
            first,
            slot_ptr,
            ZipItemType::__field_first::<
                <I as IndexedSource>::Item,
                <S as IndexedSource>::Item,
            >::OFFSET as i32,
        );
        store_value::<<S as IndexedSource>::Item>(
            ctx,
            second,
            slot_ptr,
            ZipItemType::__field_second::<
                <I as IndexedSource>::Item,
                <S as IndexedSource>::Item,
            >::OFFSET as i32,
        );

        slot_ptr
    }
}

fn store_value<T: StagedType>(ctx: &mut CompilationContext, value: Value, ptr: Value, offset: i32) {
    if T::is_copy_struct() {
        let destination = ctx.ptr_offset_const(ptr, i64::from(offset));
        ctx.copy_nonoverlapping(destination, value, T::size_of(), T::align_of());
    } else {
        ctx.store(value, ptr, offset);
    }
}

impl<I, S> StagedIterator for Zip<I, S>
where
    I: IndexedStagedIterator + IndexedSource + Clone + 'static,
    <I as IndexedSource>::Item: CopyType + 'static,
    S: IndexedSource + Clone + 'static,
    <S as IndexedSource>::Item: CopyType + 'static,
    <I as IndexedSource>::GetExpr: 'static,
    <S as IndexedSource>::GetExpr: 'static,
{
    type Item = ZipItem<<I as IndexedSource>::Item, <S as IndexedSource>::Item>;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
    {
        let i = ctx.var(0u64);
        let len = ctx.bind(ZipLen::new(
            IndexedSource::len(&self.iter),
            IndexedSource::len(&self.other),
        ));
        let prim = self.iter;
        let sec = self.other;

        ctx.while_loop(lt(i, len), move |ctx| {
            let elem = ctx.bind(ZipGetAt::new(prim.clone(), sec.clone(), i));
            consumer(ctx, elem);
            ctx.store(i, add(i, 1u64));
        });
    }
}

impl<I, S> IndexedStagedIterator for Zip<I, S>
where
    I: IndexedStagedIterator + IndexedSource + Clone + 'static,
    <I as IndexedSource>::Item: CopyType + 'static,
    S: IndexedSource + Clone + 'static,
    <S as IndexedSource>::Item: CopyType + 'static,
    <I as IndexedSource>::GetExpr: 'static,
    <S as IndexedSource>::GetExpr: 'static,
{
    type LenExpr = ZipLen<<I as IndexedSource>::LenExpr, <S as IndexedSource>::LenExpr>;

    fn len(&self) -> Self::LenExpr {
        ZipLen::new(
            IndexedSource::len(&self.iter),
            IndexedSource::len(&self.other),
        )
    }
}

impl<I, S> IndexedSource for Zip<I, S>
where
    I: IndexedStagedIterator + IndexedSource + Clone + 'static,
    <I as IndexedSource>::Item: CopyType + 'static,
    S: IndexedSource + Clone + 'static,
    <S as IndexedSource>::Item: CopyType + 'static,
    <I as IndexedSource>::GetExpr: 'static,
    <S as IndexedSource>::GetExpr: 'static,
{
    type Item = ZipItem<<I as IndexedSource>::Item, <S as IndexedSource>::Item>;
    type LenExpr = ZipLen<<I as IndexedSource>::LenExpr, <S as IndexedSource>::LenExpr>;
    type GetExpr = ZipGetAt<I, S>;

    fn len(&self) -> Self::LenExpr {
        ZipLen::new(
            IndexedSource::len(&self.iter),
            IndexedSource::len(&self.other),
        )
    }

    unsafe fn get_at(self, index: Var<u64>) -> Self::GetExpr {
        ZipGetAt::new(self.iter, self.other, index)
    }
}

impl<I, S> Zip<I, S>
where
    I: IndexedStagedIterator + IndexedSource + 'static,
    <I as IndexedSource>::Item: StagedType + CopyType + 'static,
    S: IndexedSource + 'static,
    <S as IndexedSource>::Item: StagedType + CopyType + 'static,
    <I as IndexedSource>::GetExpr: 'static,
    <S as IndexedSource>::GetExpr: 'static,
{
    /// Drive a loop over `(primary_elem, secondary_elem)` pairs.
    ///
    /// Both sources are accessed at the same 0-based index and iteration stops
    /// at the shorter source.
    pub fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<<I as IndexedSource>::Item>, Var<<S as IndexedSource>::Item>)
            + 'static,
    {
        let i = ctx.var(0u64);
        let len = ctx.bind(ZipLen::new(
            IndexedSource::len(&self.iter),
            IndexedSource::len(&self.other),
        ));
        let prim = self.iter;
        let sec = self.other;

        ctx.while_loop(lt(i, len), move |ctx| {
            let pair = ctx.bind(ZipGetAt::new(prim.clone(), sec.clone(), i));
            let elem1 = ctx.bind(pair.first());
            let elem2 = ctx.bind(pair.second());
            consumer(ctx, elem1, elem2);
            ctx.store(i, add(i, 1u64));
        });
    }
}
