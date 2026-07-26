//! Zip combinator — pairs elements from two sources at the same index.

use cranelift_codegen::ir::{types, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value};

use rust_lms_derive::StagedType;

use crate::func::Ctx;
use crate::num::{add, lt};
use crate::r#struct::{load_field, Field, LoadField};
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
    A: StagedType,
    B: StagedType,
{
    fn first(self) -> LoadField<Self, ZipItemType::__field_first<A, B>>
    where
        A: CopyType,
    {
        load_field(self, ZipItemType::first::<A, B>())
    }

    fn second(self) -> LoadField<Self, ZipItemType::__field_second<A, B>>
    where
        B: CopyType,
    {
        load_field(self, ZipItemType::second::<A, B>())
    }
}

impl<A, B, S> ZipItemAccess<A, B> for S
where
    A: StagedType,
    B: StagedType,
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

impl<I, S> Staged for ZipGetAt<I, S>
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
        let first = IndexedSource::get_at(self.iter.clone(), self.index).codegen(ctx);
        let second = IndexedSource::get_at(self.other.clone(), self.index).codegen(ctx);

        let align_shift = Self::Out::align_of().trailing_zeros() as u8;
        let stack_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            Self::Out::size_of() as u32,
            align_shift,
        ));
        let slot_ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);

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
        for i in 0..T::num_abi_values() {
            let field_offset = (i * 8) as i32;
            let chunk =
                ctx.builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), value, field_offset);
            ctx.builder
                .ins()
                .store(MemFlags::trusted(), chunk, ptr, offset + field_offset);
        }
    } else {
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), value, ptr, offset);
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
        let prim_len = IndexedSource::len(&self.iter);
        let prim = self.iter;
        let sec = self.other;

        ctx.while_loop(lt(i, prim_len), move |ctx| {
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
    type LenExpr = <I as IndexedSource>::LenExpr;

    fn len(&self) -> Self::LenExpr {
        IndexedSource::len(&self.iter)
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
    type LenExpr = <I as IndexedSource>::LenExpr;
    type GetExpr = ZipGetAt<I, S>;

    fn len(&self) -> Self::LenExpr {
        IndexedSource::len(&self.iter)
    }

    fn get_at(self, index: Var<u64>) -> Self::GetExpr {
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
    /// The primary source's length drives the loop; the secondary is accessed
    /// at the same 0-based index. Caller must ensure equal-length sources.
    pub fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<<I as IndexedSource>::Item>, Var<<S as IndexedSource>::Item>)
            + 'static,
    {
        let i = ctx.var(0u64);
        let prim_len = IndexedSource::len(&self.iter);
        let prim = self.iter;
        let sec = self.other;

        ctx.while_loop(lt(i, prim_len), move |ctx| {
            let pair = ctx.bind(ZipGetAt::new(prim.clone(), sec.clone(), i));
            let elem1 = ctx.bind(pair.first());
            let elem2 = ctx.bind(pair.second());
            consumer(ctx, elem1, elem2);
            ctx.store(i, add(i, 1u64));
        });
    }
}
