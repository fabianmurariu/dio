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

use crate::array::ValidityView;
use crate::ffi::{FfiBuffer, FfiValidity};

/// A byte kind with its Arrow array type: `Utf8` reads `StringViewArray`, `Binary`
/// reads `BinaryViewArray`.
pub trait ArrowByteKind: ByteKind {
    type Arrow: ByteViewType + ?Sized;
}

impl ArrowByteKind for Utf8 {
    type Arrow = StringViewType;
}

impl ArrowByteKind for Binary {
    type Arrow = BinaryViewType;
}

/// The layout of a [`FfiByteViewArray<K>`], whatever its kind.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
struct RawByteViewArray {
    /// `SliceRepr<u128>`: one view per row.
    #[staged(FfiBuffer)]
    views: FfiBuffer,
    #[staged(FfiValidity)]
    validity: FfiValidity,
    /// `SliceRepr<ByteBuffer>`: the array's data buffers, by `buf_idx`.
    #[staged(FfiBuffer)]
    buffers: FfiBuffer,
}

/// One Arrow byte-view array of kind `K`, as a kernel reads it. Built only by
/// [`ByteViewDescriptors`] (or [`empty`](Self::empty)), so its kind is the array's.
#[repr(transparent)]
#[derive(Debug)]
pub struct FfiByteViewArray<K: ByteKind> {
    raw: RawByteViewArray,
    _kind: PhantomData<fn() -> K>,
}

impl<K: ByteKind> Clone for FfiByteViewArray<K> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<K: ByteKind> Copy for FfiByteViewArray<K> {}

impl<K: ByteKind> FfiByteViewArray<K> {
    /// No rows: what a consumer stands in where a column is missing, so a row lookup
    /// finds no value instead of branching on presence. Its buffers are null.
    pub const fn empty() -> Self {
        Self {
            raw: RawByteViewArray {
                views: FfiBuffer::empty(),
                validity: FfiValidity::all_valid(0),
                buffers: FfiBuffer::empty(),
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
        self.raw.validity.null_count
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
/// they point into. Built once, never mutated; see the [module docs](self).
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
                        views: FfiBuffer::from_typed_slice(array.views().as_ref()),
                        validity: FfiValidity::from_nulls(array.nulls(), array.len()),
                        buffers: FfiBuffer::from_typed_slice(window),
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

/// View a staged `&FfiByteViewArray<K>` as a column of `ByteView<K>`.
pub trait FfiByteViewArrayOps<K: ByteKind>:
    Staged<Out = SRef<FfiByteViewArray<K>>> + Sized + Clone
{
    /// The column. Safe: the descriptor's kind was checked when it was built.
    fn byte_views(
        self,
    ) -> ByteViewArrayView<impl Staged<Out = SPtr<FfiByteViewArray<K>>> + Clone, K> {
        ByteViewArrayView {
            array: ref_as_ptr(self),
            _kind: PhantomData,
        }
    }
}

impl<K: ByteKind, A> FfiByteViewArrayOps<K> for A where
    A: Staged<Out = SRef<FfiByteViewArray<K>>> + Sized + Clone
{
}

/// A staged Arrow byte-view array of kind `K`: its length, validity and rows.
pub struct ByteViewArrayView<P, K> {
    array: P,
    _kind: PhantomData<fn() -> K>,
}

impl<P: Clone, K> Clone for ByteViewArrayView<P, K> {
    fn clone(&self) -> Self {
        Self {
            array: self.array.clone(),
            _kind: PhantomData,
        }
    }
}

impl<P: Copy, K> Copy for ByteViewArrayView<P, K> {}

impl<P, K> ByteViewArrayView<P, K>
where
    P: Staged<Out = SPtr<FfiByteViewArray<K>>> + Clone + 'static,
    K: ByteKind,
{
    /// The array as its kind-free layout.
    fn raw(&self) -> impl Staged<Out = SPtr<RawByteViewArray>> + Clone + use<P, K> {
        ptr_cast::<RawByteViewArray, FfiByteViewArray<K>, _>(self.array.clone())
    }

    /// The views, one per row, as a **trusted** slice.
    pub fn views(&self) -> impl Staged<Out = SRef<Slice<u128>>> + Clone + use<P, K> {
        // SAFETY: a descriptor is built only from an Arrow array's views buffer (or
        // empty), which the host keeps alive for the kernel run; the read is shared.
        unsafe {
            field_addr(self.raw(), RawByteViewArrayType::views())
                .into_raw_slice::<u128>()
                .assume_shared()
        }
    }

    /// The table its views' `buf_idx` index.
    pub fn buffers(&self) -> impl Staged<Out = SPtr<ByteBuffer>> + Clone + use<P, K> {
        // SAFETY: as `views`: a window of its `ByteViewDescriptors`' table, alive and
        // unchanged for the kernel run.
        unsafe {
            field_addr(self.raw(), RawByteViewArrayType::buffers())
                .into_raw_slice::<ByteBuffer>()
                .into_ptr()
        }
    }

    /// The row count.
    pub fn len(&self) -> impl Staged<Out = u64> + Clone + use<P, K> {
        self.views().len()
    }

    pub fn validity(
        &self,
    ) -> ValidityView<impl Staged<Out = SPtr<FfiValidity>> + Clone + use<P, K>> {
        ValidityView::new(field_addr(self.raw(), RawByteViewArrayType::validity()))
    }

    /// Row `index`: its view and the array's table. A copy of the view, nothing more.
    ///
    /// # Safety
    ///
    /// At execution, `index` is less than the row count and the row is valid: Arrow
    /// makes no promise about a null row's view.
    pub unsafe fn value_unchecked<I>(
        &self,
        index: I,
    ) -> impl Staged<Out = ByteView<K>> + Clone + use<P, K, I>
    where
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: the row is in bounds (the caller) and valid, so its view is one Arrow
        // validated against this array's data buffers, which `buffers()` is.
        unsafe { ByteView::<K>::from_parts(self.views().get_unchecked(index), self.buffers()) }
    }
}
