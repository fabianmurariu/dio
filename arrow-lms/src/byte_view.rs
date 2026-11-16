//! Arrow byte-view arrays (`Utf8View`, `BinaryView`) as kernels read them.
//!
//! A [`FfiByteViewArray<K>`] describes one array: its views (one `u128` per row), its
//! validity, and a window of a buffer table, the array's data buffers by `buf_idx`. Row
//! `i` is `ByteView { views[i], table }`: a copy of the view, never a conversion, so
//! describing an array costs one table entry per data buffer and nothing per row.
//!
//! Arrow's `data_buffers()` is `&[Buffer]`, not `repr(C)`, so the table has to be built,
//! and a `Copy` descriptor cannot own it. [`ByteViewDescriptors<K>`] does: it builds one
//! table for a set of arrays and the descriptors that point into it. That self-reference
//! is sound because a `Box`'s heap memory does not move with its owner and nothing ever
//! mutates either box; it is confined to this type, behind private fields.

use std::marker::PhantomData;

use arrow::array::{Array, GenericByteViewArray};
use arrow::datatypes::{BinaryViewType, ByteViewType, StringViewType};
use rust_lms::prelude::*;
use rust_lms::types::{Leaves, ScalarType};

use crate::array::FfiArray;
use crate::validity::{FfiValidity, ValidityView};

/// A byte kind with its Arrow array type: `Utf8` reads `StringViewArray`, `Binary`
/// reads `BinaryViewArray`.
pub trait ArrowByteKind: ByteKind + 'static {
    type Arrow: ByteViewType + ?Sized;

    #[doc(hidden)]
    fn wrap(array: FfiByteViewArray<Self>) -> FfiArray;

    #[doc(hidden)]
    fn unwrap(array: &FfiArray) -> Option<&FfiByteViewArray<Self>>;
}

macro_rules! arrow_byte_kind {
    ($($kind:ty => $arrow:ty, $variant:ident);*) => {$(
        impl ArrowByteKind for $kind {
            type Arrow = $arrow;

            fn wrap(array: FfiByteViewArray<Self>) -> FfiArray {
                FfiArray::$variant(array)
            }

            fn unwrap(array: &FfiArray) -> Option<&FfiByteViewArray<Self>> {
                match array {
                    FfiArray::$variant(array) => Some(array),
                    _ => None,
                }
            }
        }
    )*};
}

arrow_byte_kind!(Utf8 => StringViewType, Utf8View; Binary => BinaryViewType, BinaryView);

/// The layout of a [`FfiByteViewArray<K>`], whatever its kind.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
struct RawByteViewArray {
    /// One view per row.
    views: FfiSlice<u128>,
    validity: FfiValidity,
    /// The array's data buffers, by `buf_idx`.
    buffers: FfiSlice<ByteBuffer>,
}

/// One Arrow byte-view array of kind `K`, as a kernel reads it. Built only by
/// [`ByteViewDescriptors`] (or [`empty`](Self::empty)), so its kind is the array's.
#[repr(transparent)]
#[derive(Clone, Copy, Debug)]
pub struct FfiByteViewArray<K: ByteKind> {
    raw: RawByteViewArray,
    _kind: PhantomData<fn() -> K>,
}

impl<K: ByteKind> FfiByteViewArray<K> {
    /// No rows: what a consumer stands in where a column is missing, so a row lookup
    /// finds no value instead of branching on presence. Its buffers are null.
    pub const fn empty() -> Self {
        Self {
            raw: RawByteViewArray {
                views: FfiSlice::empty(),
                validity: FfiValidity::all_valid(),
                buffers: FfiSlice::empty(),
            },
            _kind: PhantomData,
        }
    }

    /// The row count.
    pub fn len(&self) -> usize {
        self.raw.views.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn null_count(&self) -> u64 {
        self.raw.validity.null_count()
    }

    /// How many data buffers its views index.
    pub fn num_buffers(&self) -> usize {
        self.raw.buffers.len()
    }
}

// SAFETY: `FfiByteViewArray<K>` is `repr(transparent)` over `RawByteViewArray` (the
// marker is zero-sized), so every layout question is answered by it.
unsafe impl<K: ByteKind> StagedType for FfiByteViewArray<K> {
    type RuntimeValue = FfiByteViewArray<K>;

    fn scalar_type() -> ScalarType {
        RawByteViewArray::scalar_type()
    }

    fn size_of() -> usize {
        RawByteViewArray::size_of()
    }

    fn align_of() -> usize {
        RawByteViewArray::align_of()
    }

    fn is_copy_struct() -> bool {
        true
    }

    fn leaves() -> Option<Leaves> {
        RawByteViewArray::leaves()
    }
}

unsafe impl<K: ByteKind> CopyType for FfiByteViewArray<K> {}

unsafe impl<K: ByteKind> RuntimeParam for FfiByteViewArray<K> {
    type Arg<'call> = FfiByteViewArray<K>;
}

unsafe impl<K: ByteKind> RuntimeResult for FfiByteViewArray<K> {
    type Output<'call> = FfiByteViewArray<K>;
}

/// Descriptors for a set of Arrow byte-view arrays of kind `K`, and the buffer table
/// they point into. Built once, never mutated; see the `byte_view` module docs.
///
/// Nothing is copied: the table and descriptors point into the arrays' own buffers, so
/// they are read only while those arrays are alive (as [`FfiArray`](crate::FfiArray)'s).
#[derive(Debug)]
pub struct ByteViewDescriptors<K: ByteKind> {
    /// Every array's data buffers, back to back. Read only through `arrays`.
    #[allow(dead_code)]
    buffers: Box<[ByteBuffer]>,
    /// One per array given, in order; each one's `buffers` is a window of `buffers`.
    arrays: Box<[FfiByteViewArray<K>]>,
}

impl<K: ArrowByteKind> ByteViewDescriptors<K> {
    /// Describe `arrays`, in order; `None` (no such column) is
    /// [`FfiByteViewArray::empty`]. Costs one table entry per data buffer.
    pub fn new<'a>(
        arrays: impl IntoIterator<Item = Option<&'a GenericByteViewArray<K::Arrow>>>,
    ) -> Self {
        let arrays: Vec<_> = arrays.into_iter().collect();
        // The table first, boxed, so the windows below point at where it stays.
        let buffers: Box<[ByteBuffer]> = arrays
            .iter()
            .flatten()
            .flat_map(|array| array.data_buffers())
            .map(|buffer| ByteBuffer::new(buffer.as_slice()))
            .collect();
        let mut start = 0;
        let arrays = arrays
            .into_iter()
            .map(|array| {
                let Some(array) = array else {
                    return FfiByteViewArray::empty();
                };
                let window = &buffers[start..start + array.data_buffers().len()];
                start += window.len();
                FfiByteViewArray {
                    raw: RawByteViewArray {
                        views: FfiSlice::new(array.views().as_ref()),
                        validity: FfiValidity::from_nulls(array.nulls()),
                        buffers: FfiSlice::new(window),
                    },
                    _kind: PhantomData,
                }
            })
            .collect();
        Self { buffers, arrays }
    }
}

impl<K: ByteKind> ByteViewDescriptors<K> {
    /// The descriptors, one per array given: what a kernel reads.
    pub fn arrays(&self) -> &[FfiByteViewArray<K>] {
        &self.arrays
    }
}

// =============================================================================
// Staged reads
// =============================================================================

/// The staged reads behind [`ArrowArray`](crate::ArrowArray), for any kind.
impl<K: ByteKind> FfiByteViewArray<K> {
    /// The array at `array` as its kind-free layout.
    fn raw<P>(array: P) -> impl Staged<Out = SPtr<RawByteViewArray>> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        ptr_cast::<RawByteViewArray, Self, _>(array)
    }

    /// The views, one per row.
    fn views<P>(array: P) -> impl Staged<Out = SRef<Slice<u128>>> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        // SAFETY: a descriptor is built only from an Arrow array's views buffer (or
        // empty), which the host keeps alive for the kernel run; the read is shared.
        unsafe {
            field_addr(Self::raw(array), RawByteViewArrayType::views())
                .into_raw_slice::<u128>()
                .assume_shared()
        }
    }

    pub(crate) fn staged_len<P>(array: P) -> impl Staged<Out = u64> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        Self::views(array).len()
    }

    pub(crate) fn staged_validity<P>(
        array: P,
    ) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone>
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
    {
        // SAFETY: part of the descriptor, alive as long as it is.
        unsafe {
            ValidityView::new(field_addr(
                Self::raw(array),
                RawByteViewArrayType::validity(),
            ))
        }
    }

    /// Row `index`: its view and the array's buffer table. A copy of the view, nothing
    /// more.
    ///
    /// # Safety
    ///
    /// At execution, `index` is less than the row count and the row is valid.
    pub(crate) unsafe fn staged_value<P, I>(
        array: P,
        index: I,
    ) -> impl Staged<Out = ByteView<K>> + Clone
    where
        P: Staged<Out = SPtr<Self>> + Clone + 'static,
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: a window of its `ByteViewDescriptors`' table, alive and unchanged for
        // the kernel run.
        let buffers = unsafe {
            field_addr(Self::raw(array.clone()), RawByteViewArrayType::buffers())
                .into_raw_slice::<ByteBuffer>()
                .into_ptr()
        };
        // SAFETY: the row is in bounds and valid (the caller), so its view is one Arrow
        // validated against this array's data buffers, which `buffers` is.
        unsafe { ByteView::<K>::from_parts(Self::views(array).get_unchecked(index), buffers) }
    }
}
