//! Staged views over read-only primitive Arrow arrays.
//!
//! Arrow's FFI descriptors are erased `(ptr, len)` buffers. This module
//! recovers typed staged slices with the reusable `rust-lms` repr-slice adapter
//! and keeps Arrow-specific code limited to validity bitmap interpretation.

use std::marker::PhantomData;

use rust_lms::prelude::*;

use crate::ffi::{
    FfiArray, FfiArrayBatch, FfiArrayBatchType, FfiArrayType, FfiValidity, FfiValidityType,
};

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
    pub fn values<'r, 'data>(
        &self,
    ) -> impl Staged<Out = SRef<'r, Slice<M>>> + Clone + 'r + use<'r, 'data, P, M>
    where
        'data: 'r,
        P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'r,
        M: StagedType + 'r,
    {
        field_addr(self.array.clone(), FfiArrayType::values()).as_slice::<M>()
    }

    pub fn physical_values<'r, 'data>(
        &self,
    ) -> impl IndexedStagedIterator<Item = M> + IndexedSource<Item = M> + 'r + use<'r, 'data, P, M>
    where
        'r: 'static,
        'data: 'static,
        P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
        M: StagedType + CopyType + ConstantType + 'static,
        M::RuntimeValue: Default,
    {
        self.values().staged_iter()
    }

    pub fn non_null_values(&self) -> NonNullValues<P, M>
    where
        P: Clone,
    {
        NonNullValues {
            array: self.clone(),
        }
    }

    pub fn validity<'r, 'data>(
        &self,
    ) -> ValidityView<
        impl Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'r + use<'r, 'data, P, M>,
    >
    where
        'data: 'r,
        P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'r,
    {
        ValidityView {
            validity: field_addr(self.array.clone(), FfiArrayType::validity()),
        }
    }

    pub fn len<'r, 'data>(&self) -> impl Staged<Out = u64> + Clone + 'r + use<'r, 'data, P, M>
    where
        'data: 'r,
        P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'r,
        M: StagedType + 'r,
    {
        self.values().len()
    }

    pub fn value_unchecked<'r, 'data, I>(
        &self,
        index: I,
    ) -> impl Staged<Out = M> + Clone + 'r + use<'r, 'data, P, M, I>
    where
        'data: 'r,
        P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'r,
        M: StagedType + CopyType + 'r,
        I: IntoStaged<u64>,
        I::Staged: Clone + 'r,
    {
        self.values().get_unchecked(index)
    }

    // pub fn staged_iter<'r, 'data>(&self) -> impl IndexedStagedIterator
    // where
    //     'r: 'static,
    //     'data: 'static,
    //     P: Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'static,
    //     M: StagedType + CopyType + ConstantType + 'static,
    //     M::RuntimeValue: Default,
    // {
    //     let values = self.physical_values();
    //     let nulls = self.validity().iter();
    //     values.zip(nulls)
    // }
}

/// Staged operations on a prepared FFI array batch.
pub trait FfiArrayBatchOps<'r, 'arrays, 'data>:
    Staged<Out = SRef<'r, FfiArrayBatch<'arrays, 'data>>> + Sized
where
    'data: 'arrays,
    'arrays: 'r,
{
    fn arrays(self) -> impl Staged<Out = SRef<'r, Slice<FfiArray<'data>>>> + Clone + 'r
    where
        Self: Clone + 'r,
    {
        field_addr(self, FfiArrayBatchType::arrays()).as_slice::<FfiArray<'data>>()
    }

    fn primitive<M>(
        self,
        index: usize,
    ) -> PrimitiveArrayView<impl Staged<Out = SRef<'r, FfiArray<'data>>> + Clone + 'r, M>
    where
        Self: Clone + 'r,
        M: StagedType + 'r,
    {
        PrimitiveArrayView {
            array: self.arrays().get_ref_unchecked(index as u64),
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

/// Staged validity bitmap view.
pub struct ValidityView<V> {
    validity: V,
}

impl<V: Clone> Clone for ValidityView<V> {
    fn clone(&self) -> Self {
        Self {
            validity: self.validity.clone(),
        }
    }
}

impl<V: Copy> Copy for ValidityView<V> {}

impl<V> ValidityView<V> {
    pub fn bytes<'r, 'data>(
        &self,
    ) -> impl Staged<Out = SRef<'r, Slice<u8>>> + Clone + 'r + use<'r, 'data, V>
    where
        'data: 'r,
        V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'r,
    {
        field_addr(self.validity.clone(), FfiValidityType::bytes()).as_slice::<u8>()
    }

    pub fn len<'r, 'data>(&self) -> ValidityLen<V>
    where
        'data: 'r,
        V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'r,
    {
        ValidityLen {
            validity: self.validity.clone(),
        }
    }

    pub fn null_count<'r, 'data>(&self) -> ValidityNullCount<V>
    where
        'data: 'r,
        V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'r,
    {
        ValidityNullCount {
            validity: self.validity.clone(),
        }
    }

    pub fn bit_offset<'r, 'data>(&self) -> impl Staged<Out = u64> + Clone + 'r + use<'r, 'data, V>
    where
        'data: 'r,
        V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'r,
    {
        load_field(self.validity.clone(), FfiValidityType::bit_offset())
    }

    pub fn is_valid<I>(&self, index: I) -> ValidityIsValid<V, I::Staged>
    where
        V: Clone,
        I: IntoStaged<u64>,
    {
        ValidityIsValid {
            validity: self.validity.clone(),
            index: index.into_staged(),
        }
    }

    pub fn iter(&self) -> ValidityIter<V>
    where
        V: Clone,
    {
        ValidityIter {
            validity: self.clone(),
        }
    }
}

/// Staged validity bitmap length.
pub struct ValidityLen<V> {
    validity: V,
}

impl<V: Clone> Clone for ValidityLen<V> {
    fn clone(&self) -> Self {
        Self {
            validity: self.validity.clone(),
        }
    }
}

impl<V: Copy> Copy for ValidityLen<V> {}

impl<'r, 'data, V> Staged for ValidityLen<V>
where
    'data: 'r,
    V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'r,
{
    type Out = u64;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        load_field(self.validity.clone(), FfiValidityType::bit_len()).codegen(ctx)
    }
}

/// Staged validity null count.
pub struct ValidityNullCount<V> {
    validity: V,
}

impl<V: Clone> Clone for ValidityNullCount<V> {
    fn clone(&self) -> Self {
        Self {
            validity: self.validity.clone(),
        }
    }
}

impl<V: Copy> Copy for ValidityNullCount<V> {}

impl<'r, 'data, V> Staged for ValidityNullCount<V>
where
    'data: 'r,
    V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'r,
{
    type Out = u64;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        load_field(self.validity.clone(), FfiValidityType::null_count()).codegen(ctx)
    }
}

/// Staged validity bitmap random access.
pub struct ValidityIsValid<V, I> {
    validity: V,
    index: I,
}

impl<V: Clone, I: Clone> Clone for ValidityIsValid<V, I> {
    fn clone(&self) -> Self {
        Self {
            validity: self.validity.clone(),
            index: self.index.clone(),
        }
    }
}

impl<V: Copy, I: Copy> Copy for ValidityIsValid<V, I> {}

impl<'r, 'data, V, I> Staged for ValidityIsValid<V, I>
where
    'data: 'r,
    V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'r,
    I: Staged<Out = u64> + Clone + 'r,
{
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let view = ValidityView {
            validity: self.validity.clone(),
        };
        let bit_index = add::<u64, _, _>(view.clone().bit_offset(), self.index.clone());
        let byte_index = shr::<u64, _, _>(bit_index.clone(), 3u64);
        let bit_in_byte = bitand::<u64, _, _>(bit_index, 7u64);
        let byte = view.clone().bytes().get_unchecked(byte_index);
        let byte64 = int_cast::<u64, u8, _>(byte);
        let mask = shl::<u64, _, _>(Const::<u64>::new(1), bit_in_byte);
        let bit_is_set = not(eq(bitand::<u64, _, _>(byte64, mask), 0u64));

        if_then_else(
            eq(view.null_count(), 0u64),
            Const::<bool>::new(true),
            bit_is_set,
        )
        .codegen(ctx)
    }
}

/// Row-preserving validity iterator.
pub struct ValidityIter<V> {
    validity: ValidityView<V>,
}

impl<V: Clone> Clone for ValidityIter<V> {
    fn clone(&self) -> Self {
        Self {
            validity: self.validity.clone(),
        }
    }
}

impl<V: Copy> Copy for ValidityIter<V> {}

impl<'r, 'data, V> StagedIterator for ValidityIter<V>
where
    'r: 'static,
    'data: 'static,
    V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'static,
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

impl<'r, 'data, V> IndexedStagedIterator for ValidityIter<V>
where
    'r: 'static,
    'data: 'static,
    V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'static,
{
    type LenExpr = ValidityLen<V>;

    fn len(&self) -> Self::LenExpr {
        self.validity.clone().len()
    }
}

impl<'r, 'data, V> IndexedSource for ValidityIter<V>
where
    'r: 'static,
    'data: 'static,
    V: Staged<Out = SRef<'r, FfiValidity<'data>>> + Clone + 'static,
{
    type Item = bool;
    type LenExpr = ValidityLen<V>;
    type GetExpr = ValidityIsValid<V, Var<u64>>;

    fn len(&self) -> Self::LenExpr {
        self.validity.clone().len()
    }

    fn get_at(self, index: Var<u64>) -> Self::GetExpr {
        self.validity.is_valid(index)
    }
}
