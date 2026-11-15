//! Staged views over primitive Arrow arrays.
//!
//! A batch is a `Slice<FfiArray>`, read through `SRef`. Bits are read through
//! [`crate::validity`]; everything else is a `rust-lms` slice reinterpretation.

use std::marker::PhantomData;

use rust_lms::prelude::*;

use crate::ffi::{FfiArray, FfiArrayType};
use crate::validity::{FfiValidity, ValidityView};

/// A lifetime-free staged pointer to an `FfiArray` descriptor.
pub trait ArraySource: Staged<Out = SPtr<FfiArray>> + Clone {}
impl<T> ArraySource for T where T: Staged<Out = SPtr<FfiArray>> + Clone {}

// =============================================================================
// Batch access: a batch is just SRef<Slice<FfiArray>>
// =============================================================================

/// Staged read operations on any slice representation of `FfiArray`.
pub trait ArrayBatchOps: Staged + Sized
where
    Self::Out: SliceType<Elem = FfiArray>,
{
    /// A typed view of column `index`.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds and its Arrow physical values buffer must be
    /// represented by `M` for every batch supplied to generated code.
    unsafe fn primitive<M>(
        self,
        index: usize,
    ) -> PrimitiveArrayView<impl Staged<Out = SPtr<FfiArray>> + Clone, M>
    where
        Self: Clone,
        M: StagedType,
    {
        PrimitiveArrayView {
            // SAFETY: forwarded from `ArrayBatchOps::primitive`'s caller.
            array: unsafe { self.get_ptr_unchecked(index as u64) },
            _elem: PhantomData,
        }
    }

    /// A view of column `index` as Arrow `Boolean` values.
    ///
    /// # Safety
    ///
    /// `index` must be in bounds and its descriptor must come from a `BooleanArray`
    /// for every batch supplied to generated code.
    unsafe fn boolean(
        self,
        index: usize,
    ) -> BooleanArrayView<impl Staged<Out = SPtr<FfiArray>> + Clone>
    where
        Self: Clone,
    {
        BooleanArrayView {
            // SAFETY: forwarded from `ArrayBatchOps::boolean`'s caller.
            array: unsafe { self.get_ptr_unchecked(index as u64) },
        }
    }
}

impl<B> ArrayBatchOps for B
where
    B: Staged + Sized,
    B::Out: SliceType<Elem = FfiArray>,
{
}

/// View a single `&FfiArray` as a typed column.
pub trait FfiArrayOps<'r>: Staged<Out = SRef<FfiArray>> + Sized + Clone {
    /// Interpret this erased descriptor's values as `M`.
    ///
    /// # Safety
    ///
    /// The descriptor's Arrow physical values buffer must be represented by
    /// `M` for every generated-code use.
    unsafe fn into_primitive<M: StagedType>(
        self,
    ) -> PrimitiveArrayView<impl Staged<Out = SPtr<FfiArray>> + Clone, M> {
        PrimitiveArrayView {
            array: ref_as_ptr(self),
            _elem: PhantomData,
        }
    }

    /// Interpret this erased descriptor as Arrow `Boolean` values.
    ///
    /// # Safety
    ///
    /// The descriptor must come from a `BooleanArray`.
    unsafe fn into_boolean(self) -> BooleanArrayView<impl Staged<Out = SPtr<FfiArray>> + Clone> {
        BooleanArrayView {
            array: ref_as_ptr(self),
        }
    }
}

// =============================================================================
// BooleanArrayView: bit-packed values
// =============================================================================

/// Staged view of an Arrow `Boolean` array. Its descriptor's `values` addresses the
/// bit-packed values from their first bit, LSB first like the validity bitmap, with
/// one element per row: the buffer's length is the row count, not its byte count.
pub struct BooleanArrayView<P> {
    pub(crate) array: P,
}

impl<P: Clone> Clone for BooleanArrayView<P> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
        }
    }
}

impl<P: Copy> Copy for BooleanArrayView<P> {}

impl<P: ArraySource + 'static> BooleanArrayView<P> {
    /// The value bytes, indexed by byte; its `len` is the row count.
    fn bytes(&self) -> impl Staged<Out = SRef<Slice<u8>>> + Clone + use<P> {
        // SAFETY: the view's constructor established that the descriptor addresses a
        // live bitmap of at least `len` bits for the whole kernel run; the read is
        // shared.
        unsafe {
            field_addr(self.array.clone(), FfiArrayType::values())
                .into_raw_slice::<u8>()
                .assume_shared()
        }
    }

    /// The row count.
    pub fn len(&self) -> impl Staged<Out = u64> + Clone + use<P> {
        self.bytes().len()
    }

    /// Read row `index`'s value without checking the length.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this array's length.
    pub unsafe fn value_unchecked<I>(&self, index: I) -> impl Staged<Out = bool> + Clone + use<P, I>
    where
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        let bit = index.into_staged();
        let byte_index = shr::<u64, _, _>(bit.clone(), 3u64);
        let mask = shl::<u64, _, _>(Const::<u64>::new(1), bitand::<u64, _, _>(bit, 7u64));
        // SAFETY: `index < len` bits, so its byte is inside the bitmap.
        let byte = int_cast::<u64, u8, _>(unsafe { self.bytes().get_unchecked(byte_index) });
        not(eq(bitand::<u64, _, _>(byte, mask), 0u64))
    }

    pub fn validity(&self) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone + use<P>> {
        // SAFETY: part of the descriptor, alive as long as it is (the view's constructor).
        unsafe { ValidityView::new(field_addr(self.array.clone(), FfiArrayType::validity())) }
    }
}

impl<'r, A> FfiArrayOps<'r> for A where A: Staged<Out = SRef<FfiArray>> + Sized + Clone {}

// =============================================================================
// PrimitiveArrayView: read methods
// =============================================================================

/// Typed staged primitive array view.
pub struct PrimitiveArrayView<P, M> {
    pub(crate) array: P,
    pub(crate) _elem: PhantomData<M>,
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
    /// The array's values as a **trusted** staged slice.
    ///
    /// This is the descriptor-construction boundary, so the provenance claim is
    /// made here, once, rather than at every read. Callers get the ordinary
    /// slice surface — `len`, `get_or`, `get_range`, `staged_iter`,
    /// `subslice_unchecked` — instead of an unsafe read per element.
    ///
    /// Safe: the obligation is already discharged at the descriptor boundary.
    /// Both constructors — [`ArrayBatchOps::primitive`] and
    /// [`FfiArrayOps::into_primitive`] — are `unsafe fn` and require the Arrow
    /// values buffer to be represented by `M` for every generated-code use.
    /// Holding a `PrimitiveArrayView` *is* that proof.
    pub fn values(
        &self,
    ) -> impl Staged<Out = SRef<Slice<M>>>
    + Clone
    + LifetimeErased<Out = SRef<Slice<M>>, ErasedOut = SRef<Slice<M>>>
    + use<P, M>
    where
        P: ArraySource + 'static,
        M: StagedType + 'static,
    {
        // SAFETY: this view's construction contract establishes that the
        // descriptor addresses a live Arrow buffer of `M` for the whole kernel
        // run; the read is shared.
        unsafe {
            field_addr(self.array.clone(), FfiArrayType::values())
                .into_raw_slice::<M>()
                .assume_shared()
        }
    }

    pub fn len(&self) -> impl Staged<Out = u64> + Clone + use<P, M>
    where
        P: ArraySource + 'static,
        M: StagedType + 'static,
    {
        self.values().len()
    }

    /// Read an element without checking the descriptor length.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than this array's length.
    pub unsafe fn value_unchecked<I>(&self, index: I) -> impl Staged<Out = M> + Clone + use<P, M, I>
    where
        P: ArraySource + 'static,
        M: StagedType + CopyType + 'static,
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: forwarded from `PrimitiveArrayView::value_unchecked`'s caller
        // — only the index bound is left, since `values()` is already trusted.
        unsafe { self.values().get_unchecked(index) }
    }

    pub fn validity(&self) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone + use<P, M>>
    where
        P: ArraySource + 'static,
    {
        // SAFETY: part of the descriptor, alive as long as it is (the view's constructor).
        unsafe { ValidityView::new(field_addr(self.array.clone(), FfiArrayType::validity())) }
    }
}
