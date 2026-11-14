//! Byte strings as Arrow's **byte views**: [`ByteView<K>`], the staged `&str` (`Str`) and
//! `&[u8]` (`Bin`).
//!
//! A view is Arrow's 16-byte `u128` (`GenericByteViewArray`), little-endian:
//!
//! ```text
//! len <= 12:  [ len:u32 | data: 12 bytes, zero padded                ]
//! len  > 12:  [ len:u32 | prefix: 4 bytes | buf_idx:u32 | offset:u32  ]
//!              0      31                 63            95          127
//! ```
//!
//! A long value's bytes are `buffers[buf_idx].ptr + offset`, in a table of
//! [`ByteBuffer`]s. A [`ByteView<K>`] is the view **and the table it indexes**, so values
//! from different arrays, or a stored value and a literal, compare without carrying their
//! arrays along, and nothing ever rewrites a view. An inline value never reads its table.
//!
//! The kind `K` is Arrow's `ByteViewType`: [`Utf8`] values are valid UTF-8, [`Binary`]
//! ones any bytes. Byte operations (equality, order, `starts_with`, …) are on every kind;
//! [`len`](Var::len) counts characters and exists for `Utf8` alone.
//!
//! Equality and order follow arrow's `compare_byte_view`: the length and 4-byte prefix
//! first, the inline data next, and the bytes, 8 at a time, last. Everything is staged
//! except a long string's character count, the [`utf8_len`] extern.

use std::marker::PhantomData;

use rust_lms_derive::{StagedType, extern_fn};

use crate::ffi::{ExternRef, FatSlice, RawSlice, call_extern1, slice_from_raw_parts, stack_array};
use crate::func::Ctx;
use crate::num::{
    bitand, bitxor, count_ones, eq, ge, gt, int_cast, le, lt, min, ne, select, shl, shr, swap_bytes,
};
use crate::refer::{SPtr, const_ptr, ptr_as_const, ptr_cast, ptr_offset, read_unaligned, store};
use crate::staged::{CompilationContext, Const, IntoStaged, Staged, Value, Var};
use crate::r#struct::{Field, load_field_unchecked};
use crate::types::{
    ConstantType, CopyType, Leaves, RuntimeParam, RuntimeResult, ScalarType, Shape, StagedType,
};

/// The longest value held in the view itself.
pub const MAX_INLINE_LEN: usize = 12;

// =============================================================================
// Host side: buffers, views and their encoding
// =============================================================================

/// One entry of a buffer table: a block of bytes that long views point into.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, StagedType)]
pub struct ByteBuffer {
    #[staged(SPtr<u8>)]
    ptr: *const u8,
    #[staged(u64)]
    len: u64,
}

impl ByteBuffer {
    /// The entry for `bytes`; whoever builds the table keeps them alive and in place.
    pub fn new(bytes: &[u8]) -> Self {
        Self {
            ptr: bytes.as_ptr(),
            len: bytes.len() as u64,
        }
    }

    /// The bytes it describes.
    ///
    /// # Safety
    ///
    /// The bytes it was built from are alive and unchanged.
    pub unsafe fn as_slice<'a>(&self) -> &'a [u8] {
        if self.len == 0 {
            return &[];
        }
        // SAFETY: forwarded from this function's contract.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len as usize) }
    }
}

/// The view of `bytes`, as Arrow's `make_view`: inline when it fits, else its length,
/// prefix, and `(buffer_index, offset)`, where the caller has put (or will put) them.
pub fn make_view(bytes: &[u8], buffer_index: u32, offset: u32) -> u128 {
    let len = u32::try_from(bytes.len()).expect("a byte view is shorter than 4 GiB");
    let mut view = [0u8; 16];
    view[..4].copy_from_slice(&len.to_le_bytes());
    if bytes.len() <= MAX_INLINE_LEN {
        view[4..4 + bytes.len()].copy_from_slice(bytes);
    } else {
        view[4..8].copy_from_slice(&bytes[..4]);
        view[8..12].copy_from_slice(&buffer_index.to_le_bytes());
        view[12..16].copy_from_slice(&offset.to_le_bytes());
    }
    u128::from_le_bytes(view)
}

/// A view's length.
pub fn view_len(view: u128) -> usize {
    view as u32 as usize
}

/// The runtime value of a [`ByteView<K>`]: an Arrow view and the table it indexes.
///
/// Kernels read a long value's bytes through its table without a check, and a
/// `RawByteView` reaches them safely (a [`Const`], a kernel argument). So only an inline
/// value, which never reads its table, is built safely; a long one comes from
/// [`new`](Self::new), whose caller vouches for the table.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, StagedType)]
pub struct RawByteView {
    #[staged(u128)]
    view: u128,
    #[staged(SPtr<ByteBuffer>)]
    buffers: *const ByteBuffer,
}

/// The empty string: an inline view of no bytes, with no table.
impl Default for RawByteView {
    fn default() -> Self {
        Self {
            view: 0,
            buffers: std::ptr::null(),
        }
    }
}

impl RawByteView {
    /// A view and the table it indexes (null for an inline view).
    ///
    /// # Safety
    ///
    /// For a long view, `buffers[buf_idx]` and the bytes the view names inside it stay
    /// alive and unchanged while this value, or any copy of it (a kernel it is baked
    /// into or passed to), is read; an inline view's data past its length is zero.
    pub unsafe fn new(view: u128, buffers: *const ByteBuffer) -> Self {
        Self { view, buffers }
    }

    /// An inline value, with no table. `None` when `bytes` is too long to be one.
    pub fn inline(bytes: &[u8]) -> Option<Self> {
        (bytes.len() <= MAX_INLINE_LEN).then(|| Self {
            view: make_view(bytes, 0, 0),
            buffers: std::ptr::null(),
        })
    }

    /// The Arrow view.
    pub fn view(&self) -> u128 {
        self.view
    }

    /// The bytes it stands for.
    ///
    /// # Safety
    ///
    /// For a long view, the table it was built with is still alive (see [`new`](Self::new)).
    pub unsafe fn bytes(&self) -> &[u8] {
        let len = view_len(self.view);
        if len <= MAX_INLINE_LEN {
            // SAFETY: `view` is 16 bytes of `self`; the data starts at byte 4, and
            // `len <= 12` of them are the value.
            let data = unsafe { (&raw const self.view as *const u8).add(4) };
            return unsafe { std::slice::from_raw_parts(data, len) };
        }
        let buf_idx = (self.view >> 64) as u32 as usize;
        let offset = (self.view >> 96) as u32 as usize;
        // SAFETY: forwarded from this function's contract.
        unsafe {
            let buffer = &*self.buffers.add(buf_idx);
            &buffer.as_slice()[offset..offset + len]
        }
    }
}

// =============================================================================
// Kinds
// =============================================================================

mod sealed {
    pub trait Sealed {}
}

/// What a view's bytes are, as Arrow's `ByteViewType`. `Copy`, like every kind marker,
/// so types generic over a kind derive `Clone` and `Copy`.
pub trait ByteKind: sealed::Sealed + Copy + 'static {}

/// Valid UTF-8, as Arrow's `StringViewType`.
#[derive(Clone, Copy, Debug)]
pub enum Utf8 {}

/// Any bytes, as Arrow's `BinaryViewType`.
#[derive(Clone, Copy, Debug)]
pub enum Binary {}

impl sealed::Sealed for Utf8 {}
impl sealed::Sealed for Binary {}
impl ByteKind for Utf8 {}
impl ByteKind for Binary {}

// =============================================================================
// The staged type
// =============================================================================

/// A byte string of kind `K`: an Arrow view and its buffer table (a [`RawByteView`] at
/// run time). `Copy`, no lifetime; the host keeps the table and its blocks alive for the
/// kernel call.
#[derive(Clone, Copy)]
pub struct ByteView<K: ByteKind>(PhantomData<fn() -> K>);

/// A staged `&str`: valid UTF-8.
pub type Str = ByteView<Utf8>;

/// A staged `&[u8]`.
pub type Bin = ByteView<Binary>;

// SAFETY: every layout question is answered by `RawByteView`, its runtime value.
unsafe impl<K: ByteKind> StagedType for ByteView<K> {
    type RuntimeValue = RawByteView;

    fn scalar_type() -> ScalarType {
        RawByteView::scalar_type()
    }

    fn size_of() -> usize {
        RawByteView::size_of()
    }

    fn align_of() -> usize {
        RawByteView::align_of()
    }

    fn is_copy_struct() -> bool {
        true
    }

    fn leaves() -> Option<Leaves> {
        RawByteView::leaves()
    }
}

unsafe impl<K: ByteKind> CopyType for ByteView<K> {}

unsafe impl<K: ByteKind> RuntimeParam for ByteView<K> {
    type Arg<'call> = RawByteView;
}

unsafe impl<K: ByteKind> RuntimeResult for ByteView<K> {
    type Output<'call> = RawByteView;
}

// SAFETY: the view and the table's address, as `RawByteView` lays them out. A long
// value's table outlives the kernel: `RawByteView::new`'s contract.
unsafe impl<K: ByteKind> ConstantType for ByteView<K> {
    fn codegen_constant(value: &RawByteView, ctx: &mut CompilationContext<'_>) -> Value {
        let view = u128::codegen_constant(&value.view, ctx);
        let buffers = const_ptr::<ByteBuffer>(value.buffers).codegen(ctx);
        view_value(ctx, view, buffers)
    }
}

/// A `ByteView` from its two fields' values.
fn view_value(ctx: &mut CompilationContext, view: Value, buffers: Value) -> Value {
    ctx.struct_value::<RawByteView>(&[
        (
            Shape::of::<u128>(),
            RawByteViewType::__field_view::OFFSET,
            view,
        ),
        (
            Shape::of::<SPtr<ByteBuffer>>(),
            RawByteViewType::__field_buffers::OFFSET,
            buffers,
        ),
    ])
}

/// A [`ByteView`] from a view and the table it indexes. See [`ByteView::from_parts`].
#[derive(Clone, Copy)]
pub struct MakeByteView<V, B, K> {
    view: V,
    buffers: B,
    _k: PhantomData<fn() -> K>,
}

unsafe impl<V, B, K> Staged for MakeByteView<V, B, K>
where
    V: Staged<Out = u128>,
    B: Staged<Out = SPtr<ByteBuffer>>,
    K: ByteKind,
{
    type Out = ByteView<K>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let view = self.view.codegen(ctx);
        let buffers = self.buffers.codegen(ctx);
        view_value(ctx, view, buffers)
    }
}

impl<K: ByteKind> ByteView<K> {
    /// The value of `view` against `buffers`, the table its `buf_idx` indexes: an
    /// Arrow array's row and its array's data buffers, or a pool's view and its table.
    ///
    /// # Safety
    ///
    /// At execution, a long `view` names bytes inside `buffers[buf_idx]`, which the host
    /// keeps alive and unchanged while the kernel runs, and for `Utf8` they are valid
    /// UTF-8; an inline view's data past its length is zero, as Arrow requires.
    pub unsafe fn from_parts<V, B>(view: V, buffers: B) -> MakeByteView<V::Staged, B::Staged, K>
    where
        V: IntoStaged<u128>,
        B: IntoStaged<SPtr<ByteBuffer>>,
    {
        MakeByteView {
            view: view.into_staged(),
            buffers: buffers.into_staged(),
            _k: PhantomData,
        }
    }
}

// =============================================================================
// Staged operations
// =============================================================================

/// The low `u32` of a `u128` expression.
fn low_u32<E: Staged<Out = u128>>(e: E) -> impl Staged<Out = u32> {
    int_cast::<u32, u128, _>(e)
}

/// The sign of `a` against `b`: `-1`, `0` or `1`.
fn sign<T: crate::num::Ordered>(a: Var<T>, b: Var<T>) -> impl Staged<Out = i8> + use<T> {
    select(lt(a, b), -1i8, select(gt(a, b), 1i8, 0i8))
}

/// `ptr + i` bytes.
fn byte_at(ptr: Var<SPtr<u8>>, i: Var<u64>) -> impl Staged<Out = SPtr<u8>> {
    // SAFETY: callers only form addresses inside the value they read.
    unsafe { ptr_offset(ptr, int_cast::<i64, u64, _>(i)) }
}

/// The word of 8 bytes at `ptr + i`, big-endian: its first byte most significant, so
/// words order as their bytes.
fn word_be(ctx: &Ctx, ptr: Var<SPtr<u8>>, i: Var<u64>) -> Var<u64> {
    // SAFETY: callers read words inside the value: `i + 8 <= len`.
    ctx.bind(swap_bytes(unsafe {
        read_unaligned::<u64, _>(byte_at(ptr, i))
    }))
}

/// The byte at `ptr + i`.
fn byte(ctx: &Ctx, ptr: Var<SPtr<u8>>, i: Var<u64>) -> Var<u8> {
    // SAFETY: callers read bytes inside the value: `i < len`.
    ctx.bind(unsafe { read_unaligned::<u8, _>(byte_at(ptr, i)) })
}

/// The first index in `start..end` where the bytes at `a` and `b` differ, `end` if none
/// does: 8 at a time while a word fits, then one at a time (from inside the differing
/// word, if one was found).
fn first_difference(
    ctx: &Ctx,
    a: Var<SPtr<u8>>,
    b: Var<SPtr<u8>>,
    start: Var<u64>,
    end: Var<u64>,
) -> Var<u64> {
    let i = ctx.var(start);
    ctx.iterate(|ctx, done| {
        ctx.exit_if(gt(i + 8u64, end), done);
        ctx.exit_if(ne(word_be(ctx, a, i), word_be(ctx, b, i)), done);
        ctx.store(i, i + 8u64);
    });
    ctx.iterate(|ctx, done| {
        ctx.exit_if(ge(i, end), done);
        ctx.exit_if(ne(byte(ctx, a, i), byte(ctx, b, i)), done);
        ctx.store(i, i + 1u64);
    });
    i
}

/// Whether the bytes at `a` and `b` in `start..end` are equal.
fn bytes_eq(
    ctx: &Ctx,
    a: Var<SPtr<u8>>,
    b: Var<SPtr<u8>>,
    start: Var<u64>,
    end: Var<u64>,
) -> Var<bool> {
    ctx.bind(eq(first_difference(ctx, a, b, start, end), end))
}

impl<K: ByteKind> Var<ByteView<K>> {
    /// The Arrow view.
    pub fn view(self) -> impl Staged<Out = u128> {
        // SAFETY: a `ByteView`'s value is a `RawByteView`.
        unsafe { load_field_unchecked(self, RawByteViewType::view()) }
    }

    /// The buffer table the view's `buf_idx` indexes.
    pub fn buffers(self) -> impl Staged<Out = SPtr<ByteBuffer>> {
        // SAFETY: a `ByteView`'s value is a `RawByteView`.
        unsafe { load_field_unchecked(self, RawByteViewType::buffers()) }
    }

    /// The length in bytes.
    pub fn byte_len(self) -> impl Staged<Out = u64> {
        int_cast::<u64, u32, _>(low_u32(self.view()))
    }

    /// Whether the bytes are in the view itself.
    pub fn is_inline(self) -> impl Staged<Out = bool> {
        le(self.byte_len(), MAX_INLINE_LEN as u64)
    }

    /// The first 4 bytes, zero padded, as a little-endian `u32`.
    pub fn prefix(self) -> impl Staged<Out = u32> {
        low_u32(shr(self.view(), 32u128))
    }

    /// The address of the first byte. An inline value's bytes are copied to a stack slot
    /// of its own, so a value from a register is addressed like one in a buffer.
    fn data(self, ctx: &Ctx) -> Var<SPtr<u8>> {
        let view = ctx.bind(self.view());
        ctx.join(|ctx, out| {
            ctx.if_then(self.is_inline(), |ctx| {
                let slot = ctx.bind(stack_array::<u128>(1));
                // SAFETY: the slot holds one `u128`.
                ctx.emit(unsafe { store(slot, view) });
                let bytes = ptr_cast::<u8, u128, _>(ptr_as_const(slot));
                // SAFETY: the data starts 4 bytes into the 16-byte slot.
                ctx.goto(out, unsafe { ptr_offset(bytes, Const::new(4i64)) });
            });
            let buf_idx = int_cast::<i64, u32, _>(low_u32(view >> 64u128));
            let offset = int_cast::<i64, u32, _>(low_u32(view >> 96u128));
            // SAFETY: a long view's `buf_idx` is an entry of its table (`ByteView::from_parts`).
            let buffer = unsafe { ptr_offset(self.buffers(), buf_idx) };
            let base = unsafe { load_field_unchecked(buffer, ByteBufferType::ptr()) };
            // SAFETY: and its `offset` lies inside that entry's bytes.
            unsafe { ptr_offset(base, offset) }
        })
    }

    /// Its bytes, for an extern that reads them during the call. An inline value's
    /// bytes are a copy on this kernel's stack, so the slice is valid only while the
    /// kernel runs.
    pub fn as_slice(self, ctx: &Ctx) -> Var<RawSlice<u8>> {
        let (data, len) = (self.data(ctx), ctx.bind(self.byte_len()));
        // SAFETY: `len` bytes at `data`: the value's own (`ByteView::from_parts`).
        ctx.bind(unsafe { slice_from_raw_parts::<u8, _, _>(data, len) })
    }

    /// Whether the two hold the same bytes. Both are of one kind:
    ///
    /// ```compile_fail
    /// use rust_lms::prelude::*;
    /// fn mixed(ctx: &Ctx, s: Var<Str>, b: Var<Bin>) {
    ///     let _ = s.bytes_eq(ctx, b);
    /// }
    /// ```
    pub fn bytes_eq(self, ctx: &Ctx, other: Self) -> Var<bool> {
        let (a, b) = (ctx.bind(self.view()), ctx.bind(other.view()));
        ctx.join(|ctx, out| {
            // The length and prefix.
            let (lo_a, lo_b) = (int_cast::<u64, u128, _>(a), int_cast::<u64, u128, _>(b));
            ctx.if_then(ne(lo_a, lo_b), |ctx| ctx.goto(out, false));
            // Both inline: the rest of the view.
            ctx.if_then(self.is_inline(), |ctx| ctx.goto(out, eq(a, b)));
            // Both long, of one length: the bytes after the prefix.
            let len = ctx.bind(self.byte_len());
            let (da, db) = (self.data(ctx), other.data(ctx));
            bytes_eq(ctx, da, db, ctx.var(4u64), len)
        })
    }

    /// The order of the two's bytes, as `Ord for [u8]`: `-1`, `0` or `1`.
    pub fn compare(self, ctx: &Ctx, other: Self) -> Var<i8> {
        let (a, b) = (ctx.bind(self.view()), ctx.bind(other.view()));
        let (len_a, len_b) = (ctx.bind(self.byte_len()), ctx.bind(other.byte_len()));
        ctx.join(|ctx, out| {
            // Both inline: the data big-endian above the length, as one integer.
            let both_inline = bitand(self.is_inline(), other.is_inline());
            ctx.if_then(both_inline, |ctx| {
                let key =
                    |v: Var<u128>| (swap_bytes(v) << 32u128) | int_cast::<u128, u32, _>(low_u32(v));
                let (ka, kb) = (ctx.bind(key(a)), ctx.bind(key(b)));
                ctx.goto(out, sign(ka, kb));
            });
            // The prefixes, big-endian.
            let (pa, pb) = (
                ctx.bind(swap_bytes(self.prefix())),
                ctx.bind(swap_bytes(other.prefix())),
            );
            ctx.if_then(ne(pa, pb), |ctx| ctx.goto(out, sign(pa, pb)));
            // The first differing byte after the prefix, to the shorter's end; else the
            // lengths.
            let (da, db) = (self.data(ctx), other.data(ctx));
            let n = ctx.bind(min(len_a, len_b));
            let i = first_difference(ctx, da, db, ctx.var(4u64), n);
            ctx.if_then(lt(i, n), |ctx| {
                ctx.goto(out, sign(byte(ctx, da, i), byte(ctx, db, i)));
            });
            sign(len_a, len_b)
        })
    }

    /// Whether its bytes start with `prefix`'s.
    pub fn starts_with(self, ctx: &Ctx, prefix: Self) -> Var<bool> {
        let (len, n) = (ctx.bind(self.byte_len()), ctx.bind(prefix.byte_len()));
        ctx.join(|ctx, out| {
            ctx.if_then(gt(n, len), |ctx| ctx.goto(out, false));
            // The first `min(n, 4)` bytes, from the prefixes; then the rest.
            let checked = ctx.bind(min(n, 4u64));
            let mask = ctx.bind(int_cast::<u32, u64, _>(shl(1u64, checked * 8u64) - 1u64));
            let differ = bitand(bitxor(self.prefix(), prefix.prefix()), mask);
            ctx.if_then(ne(differ, 0u32), |ctx| ctx.goto(out, false));
            let (da, dp) = (self.data(ctx), prefix.data(ctx));
            bytes_eq(ctx, da, dp, checked, n)
        })
    }

    /// Whether its bytes end with `suffix`'s.
    pub fn ends_with(self, ctx: &Ctx, suffix: Self) -> Var<bool> {
        let (len, n) = (ctx.bind(self.byte_len()), ctx.bind(suffix.byte_len()));
        ctx.join(|ctx, out| {
            ctx.if_then(gt(n, len), |ctx| ctx.goto(out, false));
            let (da, ds) = (self.data(ctx), suffix.data(ctx));
            let tail = ctx.bind(byte_at(da, ctx.bind(len - n)));
            bytes_eq(ctx, tail, ds, ctx.var(0u64), n)
        })
    }

    /// Whether `needle`'s bytes occur in its bytes: each start in turn.
    pub fn contains(self, ctx: &Ctx, needle: Self) -> Var<bool> {
        let (len, n) = (ctx.bind(self.byte_len()), ctx.bind(needle.byte_len()));
        ctx.join(|ctx, out| {
            ctx.if_then(gt(n, len), |ctx| ctx.goto(out, false));
            let (dh, dn) = (self.data(ctx), needle.data(ctx));
            let last = ctx.bind(len - n);
            let i = ctx.var(0u64);
            ctx.iterate(|ctx, done| {
                ctx.exit_if(gt(i, last), done);
                let at = ctx.bind(byte_at(dh, i));
                let found = bytes_eq(ctx, at, dn, ctx.var(0u64), n);
                ctx.if_then(found, |ctx| ctx.goto(out, true));
                ctx.store(i, i + 1u64);
            });
            false
        })
    }
}

/// The UTF-8 continuation bytes (`0b10xx_xxxx`) of the 12 data bytes of an inline view:
/// bit 7 set and bit 6 clear, counted 12 at a time.
const DATA_HIGH_BITS: u128 = 0x8080_8080_8080_8080_8080_8080;

impl Var<Str> {
    /// The length in characters, as `str::chars().count()`. An inline string is counted
    /// in staged code; a longer one by the [`utf8_len`] extern.
    ///
    /// Only UTF-8 has characters; binary data has bytes:
    ///
    /// ```compile_fail
    /// use rust_lms::prelude::*;
    /// fn chars(ctx: &Ctx, b: Var<Bin>, utf8_len: ExternRef<Utf8LenExtern>) {
    ///     let _ = b.len(ctx, utf8_len);
    /// }
    /// ```
    pub fn len(self, ctx: &Ctx, utf8_len: ExternRef<Utf8LenExtern>) -> Var<u64> {
        let len = ctx.bind(self.byte_len());
        ctx.join(|ctx, out| {
            ctx.if_then(self.is_inline(), |ctx| {
                // Valid UTF-8 has one non-continuation byte per character. Padding
                // bytes are zero, which are not continuation bytes.
                let data = ctx.bind(shr(self.view(), 32u128));
                let continuation = bitand(
                    bitand(data, bitxor(data << 1u128, u128::MAX)),
                    DATA_HIGH_BITS,
                );
                let continuation = int_cast::<u64, u128, _>(count_ones(continuation));
                ctx.goto(out, len - continuation);
            });
            call_extern1(utf8_len, self.as_slice(ctx))
        })
    }
}

/// The number of characters of valid UTF-8: the bytes that are not continuation bytes.
/// For a [`Str`] longer than [`MAX_INLINE_LEN`]; see [`Var::len`].
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn utf8_len(bytes: FatSlice<u8>) -> u64 {
    // SAFETY: `bytes` is a valid `(ptr, len)` produced by staged code for the call.
    let bytes = unsafe { bytes.as_slice() };
    bytes.iter().filter(|&&b| (b as i8) >= -0x40).count() as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn views_round_trip_through_their_bytes() {
        let block = b"0123456789abcdefghijklmnopqrstuvwxyz".to_vec();
        let table = [ByteBuffer::new(b""), ByteBuffer::new(&block)];
        for (bytes, offset) in [(&block[..0], 0), (&block[3..15], 3), (&block[5..30], 5)] {
            // SAFETY: `table` and `block` outlive every read below.
            let view = unsafe { RawByteView::new(make_view(bytes, 1, offset), table.as_ptr()) };
            assert_eq!(unsafe { view.bytes() }, bytes);
            assert_eq!(view_len(view.view), bytes.len());
        }
    }

    #[test]
    fn utf8_len_counts_characters() {
        for s in ["", "abc", "café", "日本語のテキスト", "a😀b"] {
            let bytes = FatSlice {
                ptr: s.as_ptr(),
                len: s.len(),
            };
            assert_eq!(utf8_len(bytes), s.chars().count() as u64, "{s}");
        }
    }
}
