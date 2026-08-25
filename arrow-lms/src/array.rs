//! Staged views over primitive Arrow arrays.
//!
//! A batch is a `Slice<FfiArray>`; `SRef` gives read access, `SRefMut` write
//! access (see [`crate::ffi_mut`]). Arrow-specific logic is confined to validity
//! bitmap interpretation; everything else is a `rust-lms` slice reinterpretation.

use std::marker::PhantomData;

use rust_lms::prelude::*;

use crate::ffi::{FfiArray, FfiArrayType, FfiValidity, FfiValidityType};

/// A lifetime-free staged pointer to an `FfiArray` descriptor.
pub trait ArraySource: Staged<Out = SPtr<FfiArray>> + Clone {}
impl<T> ArraySource for T where T: Staged<Out = SPtr<FfiArray>> + Clone {}

/// A lifetime-free staged pointer to read-only validity metadata.
pub trait ValiditySource: Staged<Out = SPtr<FfiValidity>> + Clone {}
impl<T> ValiditySource for T where T: Staged<Out = SPtr<FfiValidity>> + Clone {}

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
            array: unsafe { slice_get_ptr_unchecked(self, index as u64) },
            _elem: PhantomData,
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
pub trait FfiArrayOps<'r>: Staged<Out = SRef<'r, FfiArray>> + Sized + Clone {
    /// Interpret this erased descriptor's values as `M`.
    ///
    /// # Safety
    ///
    /// The descriptor's Arrow physical values buffer must be represented by
    /// `M` for every generated-code use.
    unsafe fn as_primitive<M: StagedType>(
        self,
    ) -> PrimitiveArrayView<impl Staged<Out = SPtr<FfiArray>> + Clone, M> {
        PrimitiveArrayView {
            array: ref_as_ptr(self),
            _elem: PhantomData,
        }
    }
}

impl<'r, A> FfiArrayOps<'r> for A where A: Staged<Out = SRef<'r, FfiArray>> + Sized + Clone {}

// =============================================================================
// PrimitiveArrayView: read methods
// =============================================================================

/// Typed staged primitive array view. Read methods below; write methods
/// (`set`/`set_null`) live in [`crate::ffi_mut`] on the `SRefMut` flavor.
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
    pub fn values(&self) -> impl Staged<Out = FatSliceType<M>> + Clone + use<P, M>
    where
        P: ArraySource,
        M: StagedType,
    {
        // SAFETY: PrimitiveArrayView's element type must match the descriptor
        // created for FfiArray::values, and that storage must outlive execution.
        unsafe { field_addr(self.array.clone(), FfiArrayType::values()).into_raw_slice::<M>() }
    }

    pub fn len(&self) -> impl Staged<Out = u64> + Clone + use<P, M>
    where
        P: ArraySource,
        M: StagedType,
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
        P: ArraySource,
        M: StagedType + CopyType,
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: forwarded from `PrimitiveArrayView::value_unchecked`'s caller.
        unsafe { self.values().get_unchecked(index) }
    }

    pub fn validity(&self) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone + use<P, M>>
    where
        P: ArraySource,
    {
        ValidityView {
            validity: field_addr(self.array.clone(), FfiArrayType::validity()),
        }
    }
}

// =============================================================================
// Validity: read + shared bit addressing
// =============================================================================

/// Byte index and single-bit mask for global row `index` in a validity bitmap.
/// Shared by the read (`is_valid`) and write (`set_null`) paths so the bit
/// arithmetic lives in exactly one place.
pub(crate) unsafe fn bit_location<V, I>(
    validity: V,
    index: I,
) -> (
    impl Staged<Out = u64> + Clone,
    impl Staged<Out = u64> + Clone,
)
where
    V: Staged + Clone,
    V::Out: StagedType,
    I: IntoStaged<u64>,
    I::Staged: Clone,
{
    // SAFETY: callers guarantee `validity` is a reference to `FfiValidity`.
    let bit_offset = unsafe { load_field_unchecked(validity, FfiValidityType::bit_offset()) };
    let bit = add::<u64, _, _>(bit_offset, index.into_staged());
    let byte_index = shr::<u64, _, _>(bit.clone(), 3u64);
    let mask = shl::<u64, _, _>(Const::<u64>::new(1), bitand::<u64, _, _>(bit, 7u64));
    (byte_index, mask)
}

/// Staged validity bitmap view.
pub struct ValidityView<V> {
    pub(crate) validity: V,
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
    /// Wrap a staged `&FfiValidity` / `&mut FfiValidity`. Lets a validity bitmap
    /// be manipulated on its own, independent of any primitive array.
    pub fn new(validity: V) -> Self {
        Self { validity }
    }

    pub fn bytes(&self) -> impl Staged<Out = FatSliceType<u8>> + Clone + use<V>
    where
        V: ValiditySource,
    {
        // SAFETY: FfiValidity::bytes describes the live byte buffer backing the
        // bitmap for the duration of generated execution.
        unsafe {
            field_addr(self.validity.clone(), FfiValidityType::bytes()).into_raw_slice::<u8>()
        }
    }

    pub fn len(&self) -> LoadField<V, FfiValidityType::__field_bit_len>
    where
        V: ValiditySource,
    {
        // SAFETY: `ValiditySource` proves this expression is an
        // `SRef<FfiValidity>` and the generated field token matches it.
        unsafe { load_field_unchecked(self.validity.clone(), FfiValidityType::bit_len()) }
    }

    pub fn null_count(&self) -> LoadField<V, FfiValidityType::__field_null_count>
    where
        V: ValiditySource,
    {
        // SAFETY: `ValiditySource` proves this expression is an
        // `SRef<FfiValidity>` and the generated field token matches it.
        unsafe { load_field_unchecked(self.validity.clone(), FfiValidityType::null_count()) }
    }

    /// Test a validity bit without checking `bit_len`.
    ///
    /// # Safety
    ///
    /// At execution, `index` must be less than the validity descriptor's bit
    /// length.
    pub unsafe fn is_valid<I>(&self, index: I) -> ValidityIsValid<V, I::Staged>
    where
        V: Clone,
        I: IntoStaged<u64>,
    {
        ValidityIsValid {
            validity: self.validity.clone(),
            index: index.into_staged(),
        }
    }
}

/// Staged validity bitmap length / null count — plain field loads.
pub type ValidityLen<V> = LoadField<V, FfiValidityType::__field_bit_len>;
pub type ValidityNullCount<V> = LoadField<V, FfiValidityType::__field_null_count>;

/// Staged validity bitmap random access: is row `index` valid?
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

unsafe impl<V, I> Staged for ValidityIsValid<V, I>
where
    V: ValiditySource,
    I: Staged<Out = u64> + Clone,
{
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> rust_lms::staged::ValueId {
        let view = ValidityView {
            validity: self.validity.clone(),
        };
        // SAFETY: `ValiditySource` proves the base is an `SRef<FfiValidity>`.
        let (byte_index, mask) = unsafe { bit_location(self.validity.clone(), self.index.clone()) };
        // SAFETY: `bit_location` establishes a byte index within the validity
        // bitmap when the caller satisfies `is_valid`'s row bound.
        let byte =
            int_cast::<u64, u8, _>(unsafe { view.clone().bytes().get_unchecked(byte_index) });
        let bit_is_set = not(eq(bitand::<u64, _, _>(byte, mask), 0u64));

        if_then_else(
            eq(view.null_count(), 0u64),
            Const::<bool>::new(true),
            bit_is_set,
        )
        .codegen(ctx)
    }
}
