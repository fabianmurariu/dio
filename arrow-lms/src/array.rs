//! Staged views over primitive Arrow arrays.
//!
//! A batch is a `Slice<FfiArray>`; `SRef` gives read access, `SRefMut` write
//! access (see [`crate::ffi_mut`]). Arrow-specific logic is confined to validity
//! bitmap interpretation; everything else is a `rust-lms` slice reinterpretation.

use std::marker::PhantomData;

use rust_lms::prelude::*;

use crate::ffi::{FfiArray, FfiArrayType, FfiValidity, FfiValidityType};

/// A staged `&FfiArray` (read) — the source a column view reads from.
pub trait ArraySource<'r>: Staged<Out = SRef<'r, FfiArray>> + Clone + 'r {}
impl<'r, T> ArraySource<'r> for T where T: Staged<Out = SRef<'r, FfiArray>> + Clone + 'r {}

/// A staged `&FfiValidity` (read).
pub trait ValiditySource<'r>: Staged<Out = SRef<'r, FfiValidity>> + Clone + 'r {}
impl<'r, T> ValiditySource<'r> for T where T: Staged<Out = SRef<'r, FfiValidity>> + Clone + 'r {}

// =============================================================================
// Batch access: a batch is just SRef<Slice<FfiArray>>
// =============================================================================

/// Staged read operations on a batch (`&[FfiArray]`).
pub trait ArrayBatchOps<'r>: Staged<Out = SRef<'r, Slice<FfiArray>>> + Sized {
    /// A typed view of column `index`.
    fn primitive<M>(
        self,
        index: usize,
    ) -> PrimitiveArrayView<impl Staged<Out = SRef<'r, FfiArray>> + Clone + 'r, M>
    where
        Self: Clone + 'r,
        M: StagedType + 'r,
    {
        PrimitiveArrayView {
            array: self.get_ref_unchecked(index as u64),
            _elem: PhantomData,
        }
    }
}

impl<'r, B> ArrayBatchOps<'r> for B where B: Staged<Out = SRef<'r, Slice<FfiArray>>> + Sized {}

/// View a single `&FfiArray` as a typed column.
pub trait FfiArrayOps<'r>: Staged<Out = SRef<'r, FfiArray>> + Sized {
    fn as_primitive<M: StagedType>(self) -> PrimitiveArrayView<Self, M> {
        PrimitiveArrayView {
            array: self,
            _elem: PhantomData,
        }
    }
}

impl<'r, A> FfiArrayOps<'r> for A where A: Staged<Out = SRef<'r, FfiArray>> + Sized {}

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
    pub fn values<'r>(&self) -> impl Staged<Out = SRef<'r, Slice<M>>> + Clone + 'r + use<'r, P, M>
    where
        P: ArraySource<'r>,
        M: StagedType + 'r,
    {
        field_addr(self.array.clone(), FfiArrayType::values()).as_slice::<M>()
    }

    pub fn len<'r>(&self) -> impl Staged<Out = u64> + Clone + 'r + use<'r, P, M>
    where
        P: ArraySource<'r>,
        M: StagedType + 'r,
    {
        self.values().len()
    }

    pub fn value_unchecked<'r, I>(
        &self,
        index: I,
    ) -> impl Staged<Out = M> + Clone + 'r + use<'r, P, M, I>
    where
        P: ArraySource<'r>,
        M: StagedType + CopyType + 'r,
        I: IntoStaged<u64>,
        I::Staged: Clone + 'r,
    {
        self.values().get_unchecked(index)
    }

    pub fn validity<'r>(
        &self,
    ) -> ValidityView<impl Staged<Out = SRef<'r, FfiValidity>> + Clone + 'r + use<'r, P, M>>
    where
        P: ArraySource<'r>,
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
pub(crate) fn bit_location<V, I>(
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
    let bit = add::<u64, _, _>(
        load_field(validity, FfiValidityType::bit_offset()),
        index.into_staged(),
    );
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
    pub fn bytes<'r>(&self) -> impl Staged<Out = SRef<'r, Slice<u8>>> + Clone + 'r + use<'r, V>
    where
        V: ValiditySource<'r>,
    {
        field_addr(self.validity.clone(), FfiValidityType::bytes()).as_slice::<u8>()
    }

    pub fn len<'r>(&self) -> LoadField<V, FfiValidityType::__field_bit_len>
    where
        V: ValiditySource<'r>,
    {
        load_field(self.validity.clone(), FfiValidityType::bit_len())
    }

    pub fn null_count<'r>(&self) -> LoadField<V, FfiValidityType::__field_null_count>
    where
        V: ValiditySource<'r>,
    {
        load_field(self.validity.clone(), FfiValidityType::null_count())
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

impl<'r, V, I> Staged for ValidityIsValid<V, I>
where
    V: ValiditySource<'r>,
    I: Staged<Out = u64> + Clone + 'r,
{
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value {
        let view = ValidityView {
            validity: self.validity.clone(),
        };
        let (byte_index, mask) = bit_location(self.validity.clone(), self.index.clone());
        let byte = int_cast::<u64, u8, _>(view.clone().bytes().get_unchecked(byte_index));
        let bit_is_set = not(eq(bitand::<u64, _, _>(byte, mask), 0u64));

        if_then_else(
            eq(view.null_count(), 0u64),
            Const::<bool>::new(true),
            bit_is_set,
        )
        .codegen(ctx)
    }
}
