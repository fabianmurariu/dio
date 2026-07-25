//! Staged views over read-only primitive Arrow arrays.
//!
//! The host FFI layout is intentionally erased. Staged code regains the typed
//! primitive view with `batch.primitive::<T>(idx)`, relying on schema binding to
//! guarantee that `T` and `idx` match the prepared Arrow batch.

use std::marker::PhantomData;

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types, BlockArg, InstBuilder, MemFlags, Value};
use rust_lms::prelude::*;

use crate::ffi::{FfiArray, FfiArrayBatch, FfiBuffer, FfiSlice, FfiValidity};

fn load_i64(ctx: &mut CompilationContext, base: Value, offset: usize) -> Value {
    ctx.builder
        .ins()
        .load(types::I64, MemFlags::trusted(), base, offset as i32)
}

fn static_array_values_ptr_offset() -> usize {
    std::mem::offset_of!(FfiArray<'static>, values) + std::mem::offset_of!(FfiBuffer<'static>, ptr)
}

fn static_array_values_len_offset() -> usize {
    std::mem::offset_of!(FfiArray<'static>, values) + std::mem::offset_of!(FfiBuffer<'static>, len)
}

fn static_array_validity_ptr_offset() -> usize {
    std::mem::offset_of!(FfiArray<'static>, validity)
        + std::mem::offset_of!(FfiValidity<'static>, ptr)
}

fn static_array_validity_bit_offset_offset() -> usize {
    std::mem::offset_of!(FfiArray<'static>, validity)
        + std::mem::offset_of!(FfiValidity<'static>, bit_offset)
}

fn static_array_validity_len_offset() -> usize {
    std::mem::offset_of!(FfiArray<'static>, validity)
        + std::mem::offset_of!(FfiValidity<'static>, len)
}

fn static_array_validity_null_count_offset() -> usize {
    std::mem::offset_of!(FfiArray<'static>, validity)
        + std::mem::offset_of!(FfiValidity<'static>, null_count)
}

fn static_batch_arrays_ptr_offset() -> usize {
    std::mem::offset_of!(FfiArrayBatch<'static, 'static>, arrays)
        + std::mem::offset_of!(FfiSlice<'static, FfiArray<'static>>, ptr)
}

/// Staged operation for extracting a column descriptor pointer from a batch.
pub struct BatchColumn<B> {
    batch: B,
    index: usize,
}

impl<B: Clone> Clone for BatchColumn<B> {
    fn clone(&self) -> Self {
        Self {
            batch: self.batch.clone(),
            index: self.index,
        }
    }
}

impl<B: Copy> Copy for BatchColumn<B> {}

impl<'r, 'arrays, 'data, B> Staged for BatchColumn<B>
where
    'data: 'arrays,
    'arrays: 'r,
    B: Staged<Out = SRef<'r, FfiArrayBatch<'arrays, 'data>>>,
{
    type Out = SRef<'r, FfiArray<'data>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let batch_ptr = self.batch.codegen(ctx);
        let arrays_ptr = load_i64(ctx, batch_ptr, static_batch_arrays_ptr_offset());
        let byte_offset = ctx.builder.ins().iconst(
            types::I64,
            (self.index * std::mem::size_of::<FfiArray<'static>>()) as i64,
        );
        ctx.builder.ins().iadd(arrays_ptr, byte_offset)
    }
}

/// Typed staged primitive array view.
pub struct PrimitiveArrayView<P, M> {
    array: P,
    _elem: PhantomData<M>,
}

impl<P: Clone, M> Clone for PrimitiveArrayView<P, M> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, M> Copy for PrimitiveArrayView<P, M> {}

impl<P, M> PrimitiveArrayView<P, M> {
    pub fn physical_values(self) -> PhysicalValues<P, M> {
        PhysicalValues { array: self }
    }

    pub fn non_null_values(self) -> NonNullValues<P, M> {
        NonNullValues { array: self }
    }

    pub fn validity(self) -> ValidityView<P> {
        ValidityView { array: self.array }
    }

    pub fn len(self) -> ArrayLen<P> {
        ArrayLen { array: self.array }
    }

    pub fn value_unchecked<I>(self, index: I) -> PrimitiveValueAt<P, I::Staged, M>
    where
        I: IntoStaged<u64>,
    {
        PrimitiveValueAt {
            array: self.array,
            index: index.into_staged(),
            _elem: PhantomData,
        }
    }
}

/// Staged operations on a prepared FFI array batch.
pub trait FfiArrayBatchOps<'r, 'arrays, 'data>:
    Staged<Out = SRef<'r, FfiArrayBatch<'arrays, 'data>>> + Sized
where
    'data: 'arrays,
    'arrays: 'r,
{
    fn primitive<M>(self, index: usize) -> PrimitiveArrayView<BatchColumn<Self>, M>
    where
        M: StagedType,
    {
        PrimitiveArrayView {
            array: BatchColumn { batch: self, index },
            _elem: PhantomData,
        }
    }
}

impl<'r, 'arrays, 'data, B> FfiArrayBatchOps<'r, 'arrays, 'data> for B
where
    'data: 'arrays,
    'arrays: 'r,
    B: Staged<Out = SRef<'r, FfiArrayBatch<'arrays, 'data>>> + Sized,
{
}

/// Staged operations on a directly-passed FFI array descriptor.
pub trait FfiArrayOps<'r, 'data>: Staged<Out = SRef<'r, FfiArray<'data>>> + Sized
where
    'data: 'r,
{
    fn as_primitive<M>(self) -> PrimitiveArrayView<Self, M>
    where
        M: StagedType,
    {
        PrimitiveArrayView {
            array: self,
            _elem: PhantomData,
        }
    }
}

impl<'r, 'data, A> FfiArrayOps<'r, 'data> for A
where
    'data: 'r,
    A: Staged<Out = SRef<'r, FfiArray<'data>>> + Sized,
{
}

/// Staged length load for a primitive array.
pub struct ArrayLen<P> {
    array: P,
}

impl<P: Clone> Clone for ArrayLen<P> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
        }
    }
}

impl<P: Copy> Copy for ArrayLen<P> {}

impl<'r, 'data, P> Staged for ArrayLen<P>
where
    'data: 'r,
    P: Staged<Out = SRef<'r, FfiArray<'data>>>,
{
    type Out = u64;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let array_ptr = self.array.codegen(ctx);
        load_i64(ctx, array_ptr, static_array_values_len_offset())
    }
}

/// Staged primitive value load by physical row index.
pub struct PrimitiveValueAt<P, I, M> {
    array: P,
    index: I,
    _elem: PhantomData<M>,
}

impl<P: Clone, I: Clone, M> Clone for PrimitiveValueAt<P, I, M> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            index: self.index.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, I: Copy, M> Copy for PrimitiveValueAt<P, I, M> {}

impl<'r, 'data, P, I, M> Staged for PrimitiveValueAt<P, I, M>
where
    'data: 'r,
    P: Staged<Out = SRef<'r, FfiArray<'data>>>,
    I: Staged<Out = u64>,
    M: StagedType,
{
    type Out = M;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let array_ptr = self.array.codegen(ctx);
        let values_ptr = load_i64(ctx, array_ptr, static_array_values_ptr_offset());
        let index = self.index.codegen(ctx);
        let scale = ctx.builder.ins().iconst(types::I64, M::size_of() as i64);
        let byte_offset = ctx.builder.ins().imul(index, scale);
        let value_ptr = ctx.builder.ins().iadd(values_ptr, byte_offset);
        ctx.builder
            .ins()
            .load(M::cranelift_type(), MemFlags::trusted(), value_ptr, 0)
    }
}

/// Row-preserving physical values iterator.
pub struct PhysicalValues<P, M> {
    array: PrimitiveArrayView<P, M>,
}

impl<P: Clone, M> Clone for PhysicalValues<P, M> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
        }
    }
}

impl<P: Copy, M> Copy for PhysicalValues<P, M> {}

impl<'r, 'data, P, M> StagedIterator for PhysicalValues<P, M>
where
    'r: 'static,
    'data: 'static,
    P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
    M: StagedType + CopyType + 'static,
{
    type Item = M;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
    {
        let i = ctx.var(0u64);
        let array = self.array;

        ctx.while_loop(lt(i, array.clone().len()), move |ctx| {
            let elem = ctx.bind(array.clone().value_unchecked(i));
            consumer(ctx, elem);
            ctx.store(i, add(i, 1u64));
        });
    }
}

impl<'r, 'data, P, M> IndexedStagedIterator for PhysicalValues<P, M>
where
    'r: 'static,
    'data: 'static,
    P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
    M: StagedType + CopyType + 'static,
{
    type LenExpr = ArrayLen<P>;

    fn len(&self) -> Self::LenExpr {
        self.array.clone().len()
    }
}

impl<'r, 'data, P, M> IndexedSource for PhysicalValues<P, M>
where
    'r: 'static,
    'data: 'static,
    P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
    M: StagedType + CopyType + 'static,
{
    type Item = M;
    type LenExpr = ArrayLen<P>;
    type GetExpr = PrimitiveValueAt<P, Var<u64>, M>;

    fn len(&self) -> Self::LenExpr {
        self.array.clone().len()
    }

    fn get_at(self, index: Var<u64>) -> Self::GetExpr {
        self.array.value_unchecked(index)
    }
}

/// Null-skipping primitive values iterator.
///
/// This is intentionally not an [`IndexedStagedIterator`]: filtering invalid
/// rows breaks direct position correspondence.
pub struct NonNullValues<P, M> {
    array: PrimitiveArrayView<P, M>,
}

impl<P: Clone, M> Clone for NonNullValues<P, M> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
        }
    }
}

impl<P: Copy, M> Copy for NonNullValues<P, M> {}

impl<'r, 'data, P, M> StagedIterator for NonNullValues<P, M>
where
    'r: 'static,
    'data: 'static,
    P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
    M: StagedType + CopyType + 'static,
{
    type Item = M;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
    {
        let i = ctx.var(0u64);
        let array = self.array;

        ctx.while_loop(lt(i, array.clone().len()), move |ctx| {
            ctx.if_then(array.clone().validity().is_valid(i), move |ctx| {
                let elem = ctx.bind(array.clone().value_unchecked(i));
                consumer(ctx, elem);
            });
            ctx.store(i, add(i, 1u64));
        });
    }
}

/// Staged validity bitmap view for an array.
pub struct ValidityView<P> {
    array: P,
}

impl<P: Clone> Clone for ValidityView<P> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
        }
    }
}

impl<P: Copy> Copy for ValidityView<P> {}

impl<P> ValidityView<P> {
    pub fn len(self) -> ValidityLen<P> {
        ValidityLen { array: self.array }
    }

    pub fn null_count(self) -> ValidityNullCount<P> {
        ValidityNullCount { array: self.array }
    }

    pub fn is_valid<I>(self, index: I) -> ValidityIsValid<P, I::Staged>
    where
        I: IntoStaged<u64>,
    {
        ValidityIsValid {
            array: self.array,
            index: index.into_staged(),
        }
    }

    pub fn iter(self) -> ValidityIter<P> {
        ValidityIter { validity: self }
    }
}

pub struct ValidityLen<P> {
    array: P,
}

impl<P: Clone> Clone for ValidityLen<P> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
        }
    }
}

impl<P: Copy> Copy for ValidityLen<P> {}

impl<'r, 'data, P> Staged for ValidityLen<P>
where
    'data: 'r,
    P: Staged<Out = SRef<'r, FfiArray<'data>>>,
{
    type Out = u64;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let array_ptr = self.array.codegen(ctx);
        load_i64(ctx, array_ptr, static_array_validity_len_offset())
    }
}

pub struct ValidityNullCount<P> {
    array: P,
}

impl<P: Clone> Clone for ValidityNullCount<P> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
        }
    }
}

impl<P: Copy> Copy for ValidityNullCount<P> {}

impl<'r, 'data, P> Staged for ValidityNullCount<P>
where
    'data: 'r,
    P: Staged<Out = SRef<'r, FfiArray<'data>>>,
{
    type Out = u64;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let array_ptr = self.array.codegen(ctx);
        load_i64(ctx, array_ptr, static_array_validity_null_count_offset())
    }
}

/// Staged validity bitmap random access.
pub struct ValidityIsValid<P, I> {
    array: P,
    index: I,
}

impl<P: Clone, I: Clone> Clone for ValidityIsValid<P, I> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            index: self.index.clone(),
        }
    }
}

impl<P: Copy, I: Copy> Copy for ValidityIsValid<P, I> {}

impl<'r, 'data, P, I> Staged for ValidityIsValid<P, I>
where
    'data: 'r,
    P: Staged<Out = SRef<'r, FfiArray<'data>>>,
    I: Staged<Out = u64>,
{
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let array_ptr = self.array.codegen(ctx);
        let validity_ptr = load_i64(ctx, array_ptr, static_array_validity_ptr_offset());
        let ptr_is_null = ctx.builder.ins().icmp_imm(IntCC::Equal, validity_ptr, 0);

        let all_valid_block = ctx.builder.create_block();
        let bitmap_block = ctx.builder.create_block();
        let merge_block = ctx.builder.create_block();
        let bool_ty = ctx.builder.func.dfg.value_type(ptr_is_null);
        ctx.builder.append_block_param(merge_block, bool_ty);

        ctx.builder
            .ins()
            .brif(ptr_is_null, all_valid_block, &[], bitmap_block, &[]);

        ctx.builder.switch_to_block(all_valid_block);
        ctx.builder.seal_block(all_valid_block);
        let true_value = ctx
            .builder
            .ins()
            .icmp(IntCC::Equal, validity_ptr, validity_ptr);
        let true_args = [BlockArg::Value(true_value)];
        ctx.builder.ins().jump(merge_block, &true_args);

        ctx.builder.switch_to_block(bitmap_block);
        ctx.builder.seal_block(bitmap_block);
        let bit_offset = load_i64(ctx, array_ptr, static_array_validity_bit_offset_offset());
        let index = self.index.codegen(ctx);
        let logical_bit = ctx.builder.ins().iadd(bit_offset, index);
        let byte_index = ctx.builder.ins().ushr_imm(logical_bit, 3);
        let bit_in_byte = ctx.builder.ins().band_imm(logical_bit, 7);
        let byte_ptr = ctx.builder.ins().iadd(validity_ptr, byte_index);
        let byte = ctx
            .builder
            .ins()
            .load(types::I8, MemFlags::trusted(), byte_ptr, 0);
        let byte64 = ctx.builder.ins().uextend(types::I64, byte);
        let one = ctx.builder.ins().iconst(types::I64, 1);
        let mask = ctx.builder.ins().ishl(one, bit_in_byte);
        let masked = ctx.builder.ins().band(byte64, mask);
        let bit_is_set = ctx.builder.ins().icmp_imm(IntCC::NotEqual, masked, 0);
        let bitmap_args = [BlockArg::Value(bit_is_set)];
        ctx.builder.ins().jump(merge_block, &bitmap_args);

        ctx.builder.switch_to_block(merge_block);
        ctx.builder.seal_block(merge_block);
        ctx.builder.block_params(merge_block)[0]
    }
}

/// Row-preserving validity iterator.
pub struct ValidityIter<P> {
    validity: ValidityView<P>,
}

impl<P: Clone> Clone for ValidityIter<P> {
    fn clone(&self) -> Self {
        Self {
            validity: self.validity.clone(),
        }
    }
}

impl<P: Copy> Copy for ValidityIter<P> {}

impl<'r, 'data, P> StagedIterator for ValidityIter<P>
where
    'r: 'static,
    'data: 'static,
    P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
{
    type Item = bool;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<Self::Item>) + 'static,
    {
        let i = ctx.var(0u64);
        let validity = self.validity;

        ctx.while_loop(lt(i, validity.clone().len()), move |ctx| {
            let is_valid = ctx.bind(validity.clone().is_valid(i));
            consumer(ctx, is_valid);
            ctx.store(i, add(i, 1u64));
        });
    }
}

impl<'r, 'data, P> IndexedStagedIterator for ValidityIter<P>
where
    'r: 'static,
    'data: 'static,
    P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
{
    type LenExpr = ValidityLen<P>;

    fn len(&self) -> Self::LenExpr {
        self.validity.clone().len()
    }
}

impl<'r, 'data, P> IndexedSource for ValidityIter<P>
where
    'r: 'static,
    'data: 'static,
    P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
{
    type Item = bool;
    type LenExpr = ValidityLen<P>;
    type GetExpr = ValidityIsValid<P, Var<u64>>;

    fn len(&self) -> Self::LenExpr {
        self.validity.clone().len()
    }

    fn get_at(self, index: Var<u64>) -> Self::GetExpr {
        self.validity.is_valid(index)
    }
}
