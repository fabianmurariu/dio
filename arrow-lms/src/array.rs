//! Typed Arrow arrays, the [`ArrowArray`] trait over them, and [`FfiArray`], the one
//! that holds any of them.
//!
//! Each Arrow type a kernel reads has its own descriptor, built only from an Arrow array
//! of that type: [`FfiPrimArray<T>`] for the primitive types, [`FfiBoolArray`] for
//! `Boolean`, and [`FfiByteViewArray<K>`] for `Utf8View` and `BinaryView`. Generic code is generic in
//! [`ArrowArray`], at stage 0; a kernel only ever sees one concrete layout.
//!
//! A batch's columns have different types, so a batch holds [`FfiArray`]s: a Rust enum
//! of the descriptors, matched as one on the host. A kernel reads one as the array its
//! column's type says it is ([`FfiArrayOps::downcast`]), a claim that is `unsafe`. A
//! batch makes it once, against its schema ([`SchemaBatch`](crate::SchemaBatch)).

use arrow::{
    array::{
        Array, BooleanArray, PrimitiveArray,
        types::{
            ArrowPrimitiveType, Float32Type, Float64Type, Int8Type, Int16Type, Int32Type,
            Int64Type, UInt8Type, UInt16Type, UInt32Type, UInt64Type,
        },
    },
    datatypes::{ArrowNativeType, ByteViewType, DataType},
};
use rust_lms::{
    prelude::*,
    types::{Leaves, ScalarType},
};

use crate::{
    byte_view::{ArrowByteKind, FfiByteViewArray},
    validity::{BitmapView, FfiBitmap, FfiValidity, ValidityView},
};

// =============================================================================
// The trait
// =============================================================================

mod sealed {
    pub trait Sealed {}
}

/// An Arrow array descriptor of one type: what generic code reads a column through.
///
/// Its reads take a staged pointer to the descriptor; call them as methods of a staged
/// reference with [`ArrowArrayOps`].
pub trait ArrowArray:
    StagedType<RuntimeValue = Self> + CopyType + Into<FfiArray> + sealed::Sealed + 'static
{
    /// What a row holds.
    type Value: StagedType;

    /// Whether an Arrow array of `data_type` is described by `Self`.
    fn accepts(data_type: &DataType) -> bool;

    /// `array` as `Self`, if that is what it holds.
    fn downcast_ref(array: &FfiArray) -> Option<&Self>;

    /// The row count.
    fn len_of<P>(array: P) -> impl Staged<Out = u64> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static;

    fn validity_of<P>(array: P) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone>
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static;

    /// Row `index`'s value.
    ///
    /// # Safety
    ///
    /// At execution, `index` is less than the row count, and for a byte-view array the
    /// row is valid: Arrow makes no promise about a null row's view.
    unsafe fn value_of<P, I>(array: P, index: I) -> impl Staged<Out = Self::Value> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
        I: IntoStaged<u64>,
        I::Staged: Clone;
}

/// [`ArrowArray`]'s reads, as methods of a staged reference to the array.
#[expect(
    clippy::len_without_is_empty,
    reason = "a staged length is compared in the kernel; `is_empty` would only wrap it"
)]
pub trait ArrowArrayOps<A: ArrowArray>: Staged<Out = SRef<A>> + Clone + 'static {
    /// The row count.
    fn len(self) -> impl Staged<Out = u64> + Clone {
        A::len_of(ref_as_ptr(self))
    }

    fn validity(self) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone> {
        A::validity_of(ref_as_ptr(self))
    }

    /// Row `index`'s value.
    ///
    /// # Safety
    ///
    /// As [`ArrowArray::value_of`].
    unsafe fn value_unchecked<I>(self, index: I) -> impl Staged<Out = A::Value> + Clone
    where
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: forwarded from the caller.
        unsafe { A::value_of(ref_as_ptr(self), index) }
    }
}

impl<A: ArrowArray, S> ArrowArrayOps<A> for S where S: Staged<Out = SRef<A>> + Clone + 'static {}

// =============================================================================
// Primitive arrays
// =============================================================================

/// A primitive type an Arrow array holds, with its Arrow type: the integers of 8 to 64
/// bits, `f32` and `f64`.
pub trait ArrowNative:
    StagedType<RuntimeValue = Self> + CopyType + ArrowNativeType + sealed::Sealed + 'static
{
    type Arrow: ArrowPrimitiveType<Native = Self>;

    #[doc(hidden)]
    fn wrap(array: FfiPrimArray<Self>) -> FfiArray;

    #[doc(hidden)]
    fn unwrap(array: &FfiArray) -> Option<&FfiPrimArray<Self>>;
}

/// An Arrow primitive array of `T`: one `T` per row, and validity.
#[repr(C)]
#[derive(StagedType)]
pub struct FfiPrimArray<T: ArrowNative> {
    values: FfiSlice<T>,
    validity: FfiValidity,
}

impl<T: ArrowNative> Clone for FfiPrimArray<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: ArrowNative> Copy for FfiPrimArray<T> {}

impl<T: ArrowNative> std::fmt::Debug for FfiPrimArray<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FfiPrimArray")
            .field("values", &self.values)
            .field("validity", &self.validity)
            .finish()
    }
}

impl<T: ArrowNative> FfiPrimArray<T> {
    /// `array`'s values and validity, from its first row.
    pub fn new(array: &PrimitiveArray<T::Arrow>) -> Self {
        Self {
            values: FfiSlice::new(array.values().as_ref()),
            validity: FfiValidity::from_nulls(array.nulls()),
        }
    }

    /// No rows.
    pub const fn empty() -> Self {
        Self {
            values: FfiSlice::empty(),
            validity: FfiValidity::all_valid(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn null_count(&self) -> u64 {
        self.validity.null_count()
    }
}

/// Staged reads only a primitive array has.
pub trait FfiPrimArrayOps<T: ArrowNative>:
    Staged<Out = SRef<FfiPrimArray<T>>> + Clone + 'static
{
    /// The values, one per row, as a **trusted** slice: `len`, `get_or`, `staged_iter`
    /// and the rest of the slice surface, with no claim left to make.
    fn values(
        self,
    ) -> impl Staged<Out = SRef<Slice<T>>>
    + Clone
    + LifetimeErased<Out = SRef<Slice<T>>, ErasedOut = SRef<Slice<T>>> {
        // SAFETY: the descriptor was built from an Arrow array of `T`, which the host
        // keeps alive and unchanged while the kernel reads it; the read is shared.
        unsafe {
            field_addr(ref_as_ptr(self), FfiPrimArrayType::values::<T>())
                .into_raw_slice::<T>()
                .assume_shared()
        }
    }
}

impl<T: ArrowNative, S> FfiPrimArrayOps<T> for S where
    S: Staged<Out = SRef<FfiPrimArray<T>>> + Clone + 'static
{
}

impl<T: ArrowNative> sealed::Sealed for FfiPrimArray<T> {}

impl<T: ArrowNative> From<FfiPrimArray<T>> for FfiArray {
    fn from(array: FfiPrimArray<T>) -> Self {
        T::wrap(array)
    }
}

impl<T: ArrowNative> ArrowArray for FfiPrimArray<T> {
    type Value = T;

    fn accepts(data_type: &DataType) -> bool {
        *data_type == T::Arrow::DATA_TYPE
    }

    fn downcast_ref(array: &FfiArray) -> Option<&Self> {
        T::unwrap(array)
    }

    fn len_of<P>(array: P) -> impl Staged<Out = u64> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        values::<T, P>(array).len()
    }

    fn validity_of<P>(array: P) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone>
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        // SAFETY: part of the descriptor, alive as long as it is.
        unsafe { ValidityView::new(field_addr(array, FfiPrimArrayType::validity::<T>())) }
    }

    unsafe fn value_of<P, I>(array: P, index: I) -> impl Staged<Out = T> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: `index` is a row (the caller).
        unsafe { values::<T, P>(array).get_unchecked(index) }
    }
}

/// The values of the primitive array at `array`.
fn values<T: ArrowNative, P>(array: P) -> impl Staged<Out = SRef<Slice<T>>> + Clone
where
    P: Staged<Out = SPtr<FfiPrimArray<T>>> + Clone + 'static,
{
    // SAFETY: as `FfiPrimArrayOps::values`.
    unsafe {
        field_addr(array, FfiPrimArrayType::values::<T>())
            .into_raw_slice::<T>()
            .assume_shared()
    }
}

// =============================================================================
// Boolean arrays
// =============================================================================

/// An Arrow `Boolean` array: a bit per row, and validity.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiBoolArray {
    values: FfiBitmap<u8>,
    validity: FfiValidity,
}

impl FfiBoolArray {
    /// `array`'s values and validity, from its first row, at whatever bit it starts.
    pub fn new(array: &BooleanArray) -> Self {
        let values = array.values();
        Self {
            values: FfiBitmap::new(values.inner().as_slice(), values.offset(), values.len()),
            validity: FfiValidity::from_nulls(array.nulls()),
        }
    }

    /// No rows.
    pub const fn empty() -> Self {
        Self {
            values: FfiBitmap::empty(),
            validity: FfiValidity::all_valid(),
        }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn null_count(&self) -> u64 {
        self.validity.null_count()
    }
}

impl sealed::Sealed for FfiBoolArray {}

impl From<FfiBoolArray> for FfiArray {
    fn from(array: FfiBoolArray) -> Self {
        Self::Boolean(array)
    }
}

impl ArrowArray for FfiBoolArray {
    type Value = bool;

    fn accepts(data_type: &DataType) -> bool {
        *data_type == DataType::Boolean
    }

    fn downcast_ref(array: &FfiArray) -> Option<&Self> {
        match array {
            FfiArray::Boolean(array) => Some(array),
            _ => None,
        }
    }

    fn len_of<P>(array: P) -> impl Staged<Out = u64> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        bool_values(array).len()
    }

    fn validity_of<P>(array: P) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone>
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        // SAFETY: part of the descriptor, alive as long as it is.
        unsafe { ValidityView::new(field_addr(array, FfiBoolArrayType::validity())) }
    }

    unsafe fn value_of<P, I>(array: P, index: I) -> impl Staged<Out = bool> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: `index` is a row (the caller), and the bitmap has a bit per row.
        unsafe { bool_values(array).bit(index) }
    }
}

/// The values of the Boolean array at `array`.
fn bool_values<P>(array: P) -> BitmapView<impl Staged<Out = SPtr<FfiBitmap<u8>>> + Clone, u8>
where
    P: Staged<Out = SPtr<FfiBoolArray>> + Clone + 'static,
{
    // SAFETY: part of the descriptor, built from a `BooleanArray`'s values, which the host
    // keeps alive and unchanged while the kernel reads them.
    unsafe { BitmapView::new(field_addr(array, FfiBoolArrayType::values())) }
}

// =============================================================================
// Byte-view arrays
// =============================================================================

impl<K: ArrowByteKind> sealed::Sealed for FfiByteViewArray<K> {}

impl<K: ArrowByteKind> From<FfiByteViewArray<K>> for FfiArray {
    fn from(array: FfiByteViewArray<K>) -> Self {
        K::wrap(array)
    }
}

impl<K: ArrowByteKind> ArrowArray for FfiByteViewArray<K> {
    type Value = ByteView<K>;

    fn accepts(data_type: &DataType) -> bool {
        *data_type == K::Arrow::DATA_TYPE
    }

    fn downcast_ref(array: &FfiArray) -> Option<&Self> {
        K::unwrap(array)
    }

    fn len_of<P>(array: P) -> impl Staged<Out = u64> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        Self::staged_len(array)
    }

    fn validity_of<P>(array: P) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone>
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        Self::staged_validity(array)
    }

    unsafe fn value_of<P, I>(array: P, index: I) -> impl Staged<Out = ByteView<K>> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: forwarded from the caller.
        unsafe { Self::staged_value(array, index) }
    }
}

// =============================================================================
// FfiArray: any of them
// =============================================================================

/// Any array a kernel reads: what a batch holds per column. The host matches it; a
/// kernel reads it as the array its column's type says ([`FfiArrayOps::downcast`]).
///
/// `repr(C, u8)`: a tag, then the arrays overlaid as a union, so every array starts at
/// the same offset (8 bytes in).
#[repr(C, u8)]
#[derive(Clone, Copy, Debug)]
pub enum FfiArray {
    Int8(FfiPrimArray<i8>),
    Int16(FfiPrimArray<i16>),
    Int32(FfiPrimArray<i32>),
    Int64(FfiPrimArray<i64>),
    UInt8(FfiPrimArray<u8>),
    UInt16(FfiPrimArray<u16>),
    UInt32(FfiPrimArray<u32>),
    UInt64(FfiPrimArray<u64>),
    Float32(FfiPrimArray<f32>),
    Float64(FfiPrimArray<f64>),
    Boolean(FfiBoolArray),
    Utf8View(FfiByteViewArray<Utf8>),
    BinaryView(FfiByteViewArray<Binary>),
}

macro_rules! arrow_native {
    ($($ty:ty => $arrow:ty, $variant:ident);* $(;)?) => {$(
        impl sealed::Sealed for $ty {}

        impl ArrowNative for $ty {
            type Arrow = $arrow;

            fn wrap(array: FfiPrimArray<Self>) -> FfiArray {
                FfiArray::$variant(array)
            }

            fn unwrap(array: &FfiArray) -> Option<&FfiPrimArray<Self>> {
                match array {
                    FfiArray::$variant(array) => Some(array),
                    _ => None,
                }
            }
        }
    )*};
}

arrow_native!(
    i8 => Int8Type, Int8;
    i16 => Int16Type, Int16;
    i32 => Int32Type, Int32;
    i64 => Int64Type, Int64;
    u8 => UInt8Type, UInt8;
    u16 => UInt16Type, UInt16;
    u32 => UInt32Type, UInt32;
    u64 => UInt64Type, UInt64;
    f32 => Float32Type, Float32;
    f64 => Float64Type, Float64;
);

/// Calls `$f` on the array an [`FfiArray`] holds, whatever its type.
macro_rules! each_array {
    ($array:expr, $a:ident => $f:expr) => {
        match $array {
            FfiArray::Int8($a) => $f,
            FfiArray::Int16($a) => $f,
            FfiArray::Int32($a) => $f,
            FfiArray::Int64($a) => $f,
            FfiArray::UInt8($a) => $f,
            FfiArray::UInt16($a) => $f,
            FfiArray::UInt32($a) => $f,
            FfiArray::UInt64($a) => $f,
            FfiArray::Float32($a) => $f,
            FfiArray::Float64($a) => $f,
            FfiArray::Boolean($a) => $f,
            FfiArray::Utf8View($a) => $f,
            FfiArray::BinaryView($a) => $f,
        }
    };
}

impl FfiArray {
    /// No rows, as an array of `data_type`; `None` if no array has that type.
    pub fn empty(data_type: &DataType) -> Option<Self> {
        macro_rules! empty {
            ($($variant:ident),*) => {
                match data_type {
                    $(DataType::$variant => Some(Self::$variant(FfiPrimArray::empty())),)*
                    DataType::Boolean => Some(Self::Boolean(FfiBoolArray::empty())),
                    DataType::Utf8View => Some(Self::Utf8View(FfiByteViewArray::empty())),
                    DataType::BinaryView => Some(Self::BinaryView(FfiByteViewArray::empty())),
                    _ => None,
                }
            };
        }
        empty!(
            Int8, Int16, Int32, Int64, UInt8, UInt16, UInt32, UInt64, Float32, Float64
        )
    }

    /// The array, as `A`, if that is what it holds.
    pub fn downcast_ref<A: ArrowArray>(&self) -> Option<&A> {
        A::downcast_ref(self)
    }

    pub fn len(&self) -> usize {
        each_array!(self, array => array.len())
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn null_count(&self) -> u64 {
        each_array!(self, array => array.null_count())
    }
}

/// Where every array starts in an [`FfiArray`]: after the tag, at the arrays' alignment.
pub(crate) const PAYLOAD: usize = {
    let array = FfiArray::Boolean(FfiBoolArray::empty());
    let start = (&raw const array).cast::<u8>();
    match &array {
        // SAFETY: both pointers are into `array`.
        FfiArray::Boolean(inner) => unsafe {
            (inner as *const FfiBoolArray)
                .cast::<u8>()
                .offset_from(start) as usize
        },
        _ => unreachable!(),
    }
};

const _: () = {
    // `repr(C, u8)` overlays the arrays, so every one starts at `PAYLOAD`; the largest
    // decides the size.
    assert!(PAYLOAD == 8);
    assert!(std::mem::size_of::<FfiArray>() == PAYLOAD + 72);
    assert!(std::mem::align_of::<FfiArray>() == 8);
};

// SAFETY: an 80-byte, 8-aligned value a kernel holds by its address (no leaves), as a
// struct is; it is only read through `downcast`, at `PAYLOAD`, as the array it holds.
unsafe impl StagedType for FfiArray {
    type RuntimeValue = FfiArray;

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }

    fn size_of() -> usize {
        std::mem::size_of::<FfiArray>()
    }

    fn align_of() -> usize {
        std::mem::align_of::<FfiArray>()
    }

    fn is_copy_struct() -> bool {
        true
    }

    fn leaves() -> Option<Leaves> {
        None
    }
}

unsafe impl CopyType for FfiArray {}

/// Staged reads of a `&FfiArray`.
pub trait FfiArrayOps: Staged<Out = SRef<FfiArray>> + Clone + 'static {
    /// The array, read as an `A`.
    ///
    /// # Safety
    ///
    /// At execution, the `FfiArray` holds an `A`.
    unsafe fn downcast<A: ArrowArray>(self) -> impl Staged<Out = SRef<A>> + Clone {
        // SAFETY: forwarded from the caller.
        unsafe { downcast_ptr(ref_as_ptr(self)) }
    }
}

impl<S> FfiArrayOps for S where S: Staged<Out = SRef<FfiArray>> + Clone + 'static {}

/// The array at `array`, read as an `A`.
///
/// # Safety
///
/// At execution, `array` addresses a live `FfiArray` that holds an `A` and that nothing
/// writes while the reference is used.
pub(crate) unsafe fn downcast_ptr<A: ArrowArray, P>(array: P) -> impl Staged<Out = SRef<A>> + Clone
where
    P: Staged<Out = SPtr<FfiArray>> + Clone + 'static,
{
    const _: () = assert!(PAYLOAD.is_multiple_of(8));
    let words = ptr_cast::<u64, FfiArray, _>(array);
    // SAFETY: `PAYLOAD` is inside the `FfiArray`, where its array starts, and the array
    // there is an `A` (the caller).
    unsafe {
        ptr_as_ref(ptr_cast::<A, u64, _>(ptr_offset(
            words,
            Const::<i64>::new((PAYLOAD / 8) as i64),
        )))
    }
}

/// Staged reads of a batch's columns, a slice of [`FfiArray`].
pub trait ArrayBatchOps: Staged + Clone + Sized + 'static
where
    Self::Out: SliceType<Elem = FfiArray>,
{
    /// Column `index`, read as an `A`.
    ///
    /// # Safety
    ///
    /// `index` is a column of every batch the kernel reads, and holds an `A` in each.
    unsafe fn column<A: ArrowArray>(self, index: usize) -> impl Staged<Out = SRef<A>> + Clone {
        // SAFETY: forwarded from the caller.
        unsafe { downcast_ptr(self.get_ptr_unchecked(index as u64)) }
    }
}

impl<B> ArrayBatchOps for B
where
    B: Staged + Clone + Sized + 'static,
    B::Out: SliceType<Elem = FfiArray>,
{
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::{array::Int32Array, buffer::NullBuffer};

    #[test]
    fn an_array_is_the_type_it_was_built_from() {
        let nulls = NullBuffer::from(vec![true, false, true]);
        let ints = FfiArray::from(FfiPrimArray::<i32>::new(&Int32Array::new(
            vec![1, 99, 3].into(),
            Some(nulls),
        )));
        assert_eq!((ints.len(), ints.null_count()), (3, 1));
        assert!(ints.downcast_ref::<FfiPrimArray<i32>>().is_some());
        assert!(ints.downcast_ref::<FfiPrimArray<u32>>().is_none());
        assert!(ints.downcast_ref::<FfiBoolArray>().is_none());

        let bools = FfiArray::from(FfiBoolArray::new(&BooleanArray::from(vec![true; 11])));
        assert_eq!((bools.len(), bools.null_count()), (11, 0));
        assert!(bools.downcast_ref::<FfiBoolArray>().is_some());
    }

    #[test]
    fn an_empty_array_has_its_type() {
        for data_type in [
            DataType::Int8,
            DataType::Float64,
            DataType::Boolean,
            DataType::Utf8View,
        ] {
            let empty = FfiArray::empty(&data_type).unwrap();
            assert!(empty.is_empty(), "{data_type}");
        }
        assert!(FfiArray::empty(&DataType::Utf8).is_none());
        let empty = FfiArray::empty(&DataType::Utf8View).unwrap();
        assert!(empty.downcast_ref::<FfiByteViewArray<Utf8>>().is_some());
    }

    #[test]
    fn each_array_accepts_its_own_type() {
        assert!(FfiPrimArray::<i64>::accepts(&DataType::Int64));
        assert!(!FfiPrimArray::<i64>::accepts(&DataType::UInt64));
        assert!(FfiBoolArray::accepts(&DataType::Boolean));
        assert!(FfiByteViewArray::<Utf8>::accepts(&DataType::Utf8View));
        assert!(!FfiByteViewArray::<Utf8>::accepts(&DataType::Utf8));
    }
}
