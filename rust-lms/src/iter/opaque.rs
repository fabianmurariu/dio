//! Opaque external iterators: drive a heap-allocated Rust iterator (e.g.
//! `Box<dyn Iterator<Item = u64>>`) from staged code via `extern "C"` calls.
//!
//! A producer extern fn returns an opaque handle ([`OpaqueHandle`] — a thin
//! `*mut ()`); a *kind* bundles the `next`/`drop` (and optionally
//! `len`/`next_value`) extern fns that operate on it. The resulting source is a
//! [`StagedIterator`], so every combinator/terminal applies.
//!
//! ```ignore
//! let all_nodes = compiler.extern_fn::<IterAllNodes>();       // producer
//! let nodes = compiler.opaque_iter_fns::<NodesKind>();        // next + drop
//! let f = compiler.fun1("count", move |ctx, g: Var<FFIGraph>| {
//!     nodes.iter(call_extern1(all_nodes, g)).count(ctx)
//! });
//! ```
//!
//! **Single-pass:** an external iterator is consumed once, so each source drives
//! exactly one terminal (no `zip`/replay). The handle is freed by `drop` on
//! every loop exit, including early `break` from `any`/`find`/`take_while`.

use std::marker::PhantomData;

use crate::ffi::{ExternFn, ExternRef, call_extern1_unchecked};
use crate::func::{Compiler, Ctx};
use crate::label::Label;
use crate::num::{add, ge};
use crate::option::{COption, COptionType, is_none, payload_unchecked};
use crate::refer::{SMutPtr, load_mut, ptr_cast_mut};
use crate::staged::{SigSpec, Staged, Value, ValueId, Var};
use crate::types::{CopyType, ScalarType, StagedType};

use super::traits::{Close, Cursor, StagedIterator};

/// The staged type of an opaque iterator handle: a thin `*mut ()` pointer.
///
/// Producer/`next`/`drop`/… extern fns all take or return `*mut ()`, which
/// `#[extern_fn]` maps to this type.
pub type OpaqueHandle = SMutPtr<()>;

// =============================================================================
// Kinds: bundle the extern fns that operate on a handle
// =============================================================================

/// An opaque-iterator kind: the `next`/`drop` extern fns over its handle.
///
/// - `Next`: `extern "C" fn(*mut ()) -> COption<Item>`
/// - `Drop`: `extern "C" fn(*mut ())`
/// # Safety
///
/// `Next` and `Drop` must operate on the same handle representation. `Next`
/// may only read or mutate the handle and `Drop` must consume it exactly once.
pub unsafe trait OpaqueIterKind: 'static {
    /// The element type returned by `next`.
    type Item: OpaqueIterItem;
    /// `next(it) -> COption<Item>`.
    type Next: ExternFn<Args = (OpaqueHandle,), Ret = COptionType<Self::Item>>;
    /// `drop(it)`.
    type Drop: ExternFn<Args = (OpaqueHandle,), Ret = ()>;
}

/// An [`OpaqueIterKind`] that also knows its length, enabling a tighter counted
/// loop and an O(1) `count`.
///
/// - `Len`: `extern "C" fn(*mut ()) -> u64`
/// - `NextValue`: `extern "C" fn(*mut ()) -> Item` (valid for `len` calls)
/// # Safety
///
/// `Len` and `NextValue` must use the same handle representation as
/// [`OpaqueIterKind`]. `NextValue` must return one initialized item for every
/// call up to the length reported by `Len`.
pub unsafe trait ExactSizeOpaqueIterKind: OpaqueIterKind {
    /// `len(it) -> u64`.
    type Len: ExternFn<Args = (OpaqueHandle,), Ret = u64>;
    /// `next_value(it) -> Item`, called exactly `len` times (no `Option`).
    type NextValue: ExternFn<Args = (OpaqueHandle,), Ret = Self::Item>;
}

// =============================================================================
// Resolved fn bundles (extern refs obtained from the Compiler, once)
// =============================================================================

/// `next`/`drop` extern refs for a kind, obtained via
/// [`Compiler::opaque_iter_fns`]. `Copy`, so capture it into a kernel closure.
pub struct OpaqueIterFns<K: OpaqueIterKind> {
    next: ExternRef<K::Next>,
    drop: ExternRef<K::Drop>,
}

impl<K: OpaqueIterKind> Clone for OpaqueIterFns<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: OpaqueIterKind> Copy for OpaqueIterFns<K> {}

impl<K: OpaqueIterKind> OpaqueIterFns<K> {
    /// Wrap a handle expression (the producer's result) into a source.
    ///
    /// # Safety
    ///
    /// The expression must evaluate to a fresh live handle for `K`. Ownership
    /// is transferred to the returned iterator, which will call `K::Drop`
    /// exactly once when its staged traversal exits.
    pub unsafe fn iter<H>(self, handle: H) -> OpaqueIter<K, H>
    where
        H: Staged<Out = OpaqueHandle> + 'static,
    {
        OpaqueIter {
            handle,
            next: self.next,
            drop: self.drop,
            _kind: PhantomData,
        }
    }
}

/// `len`/`next_value`/`next`/`drop` extern refs for an ExactSize kind, via
/// [`Compiler::exact_opaque_iter_fns`].
pub struct ExactSizeOpaqueIterFns<K: ExactSizeOpaqueIterKind> {
    len: ExternRef<K::Len>,
    next_value: ExternRef<K::NextValue>,
    drop: ExternRef<K::Drop>,
}

impl<K: ExactSizeOpaqueIterKind> Clone for ExactSizeOpaqueIterFns<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: ExactSizeOpaqueIterKind> Copy for ExactSizeOpaqueIterFns<K> {}

impl<K: ExactSizeOpaqueIterKind> ExactSizeOpaqueIterFns<K> {
    /// Wrap a handle expression into an ExactSize source (counted loop).
    ///
    /// # Safety
    ///
    /// The expression must evaluate to a fresh live handle for `K`. Ownership
    /// is transferred to the returned iterator, which will call `K::Drop`
    /// exactly once after its final use.
    pub unsafe fn iter<H>(self, handle: H) -> ExactSizeOpaqueIter<K, H>
    where
        H: Staged<Out = OpaqueHandle> + 'static,
    {
        ExactSizeOpaqueIter {
            handle,
            len: self.len,
            next_value: self.next_value,
            drop: self.drop,
            _kind: PhantomData,
        }
    }
}

impl Compiler {
    /// Register a kind's `next`/`drop` externs and bundle their refs.
    pub fn opaque_iter_fns<K: OpaqueIterKind>(&mut self) -> OpaqueIterFns<K> {
        OpaqueIterFns {
            next: self.extern_fn::<K::Next>(),
            drop: self.extern_fn::<K::Drop>(),
        }
    }

    /// Register an ExactSize kind's `len`/`next_value`/`drop` externs.
    pub fn exact_opaque_iter_fns<K: ExactSizeOpaqueIterKind>(
        &mut self,
    ) -> ExactSizeOpaqueIterFns<K> {
        ExactSizeOpaqueIterFns {
            len: self.extern_fn::<K::Len>(),
            next_value: self.extern_fn::<K::NextValue>(),
            drop: self.extern_fn::<K::Drop>(),
        }
    }
}

// =============================================================================
// OpaqueIter: next/drop source (storage-pointer loop)
// =============================================================================

/// A [`StagedIterator`] over an opaque external iterator, driven by `next`/`drop`.
pub struct OpaqueIter<K: OpaqueIterKind, H> {
    handle: H,
    next: ExternRef<K::Next>,
    drop: ExternRef<K::Drop>,
    _kind: PhantomData<K>,
}

impl<K, H> StagedIterator for OpaqueIter<K, H>
where
    K: OpaqueIterKind,
    H: Staged<Out = OpaqueHandle> + 'static,
{
    type Item = K::Item;
    type Cursor = OpaqueCursor<K>;

    fn open(self, ctx: &mut Ctx) -> OpaqueCursor<K> {
        // Bind the handle once; every step and the final drop use it.
        OpaqueCursor {
            handle: ctx.bind(self.handle),
            next: self.next,
            drop: self.drop,
        }
    }
}

/// The cursor of an [`OpaqueIter`]: one `next` extern call per step.
pub struct OpaqueCursor<K: OpaqueIterKind> {
    handle: Var<OpaqueHandle>,
    next: ExternRef<K::Next>,
    drop: ExternRef<K::Drop>,
}

impl<K: OpaqueIterKind> Cursor for OpaqueCursor<K> {
    type Item = K::Item;
    type Close = DropHandle<K::Drop>;

    /// ```text
    /// opt = next(h)              ; Rust thunk writes COption<Item>
    /// if opt is None → done
    /// elem = opt.payload
    /// ```
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<K::Item>, Self::Close) {
        // SAFETY: `handle` is the live handle this kind's producer returned; it
        // stays live until `DropHandle::close`, emitted after the loop.
        let mut opt = ctx.bind(unsafe {
            call_extern1_unchecked::<K::Next, _, OpaqueHandle>(self.next, self.handle)
        });
        ctx.exit_if(is_none(opt.reborrow()), done);
        // SAFETY: the exit above leaves only the `Some` path.
        let elem = ctx.bind(unsafe { payload_unchecked(opt) });
        (
            elem,
            DropHandle {
                handle: self.handle,
                drop: self.drop,
            },
        )
    }
}

/// Frees an opaque iterator handle: the `drop` extern, emitted once after the
/// loop — reached by exhaustion and by any early `break_loop` alike.
pub struct DropHandle<D: ExternFn> {
    handle: Var<OpaqueHandle>,
    drop: ExternRef<D>,
}

impl<D: ExternFn> Clone for DropHandle<D> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<D: ExternFn> Copy for DropHandle<D> {}

impl<D> Close for DropHandle<D>
where
    D: ExternFn<Args = (OpaqueHandle,), Ret = ()> + 'static,
{
    fn close(self, ctx: &mut Ctx) {
        // SAFETY: the matching drop for the live handle, after its final use.
        ctx.emit(unsafe { call_extern1_unchecked::<D, _, OpaqueHandle>(self.drop, self.handle) });
    }
}

// =============================================================================
// ExactSizeOpaqueIter: counted loop via len/next_value
// =============================================================================

/// A [`StagedIterator`] over an opaque external iterator with a known length:
/// a counted loop calling `next_value`, and an O(1) [`count`](Self::count).
pub struct ExactSizeOpaqueIter<K: ExactSizeOpaqueIterKind, H> {
    handle: H,
    len: ExternRef<K::Len>,
    next_value: ExternRef<K::NextValue>,
    drop: ExternRef<K::Drop>,
    _kind: PhantomData<K>,
}

impl<K, H> ExactSizeOpaqueIter<K, H>
where
    K: ExactSizeOpaqueIterKind,
    H: Staged<Out = OpaqueHandle> + 'static,
{
    /// O(1) element count: `len(it)`, then `drop(it)` — no iteration.
    pub fn count(self, ctx: &mut Ctx) -> Var<u64> {
        let handle = ctx.bind(self.handle);
        // SAFETY: `handle` came from this iterator kind's producer and remains
        // live until the matching drop call below.
        let n = ctx
            .bind(unsafe { call_extern1_unchecked::<K::Len, _, OpaqueHandle>(self.len, handle) });
        // SAFETY: this is the matching drop for the same live handle.
        ctx.emit(unsafe { call_extern1_unchecked::<K::Drop, _, OpaqueHandle>(self.drop, handle) });
        n
    }
}

impl<K, H> StagedIterator for ExactSizeOpaqueIter<K, H>
where
    K: ExactSizeOpaqueIterKind,
    H: Staged<Out = OpaqueHandle> + 'static,
{
    type Item = K::Item;
    type Cursor = ExactSizeOpaqueCursor<K>;

    fn open(self, ctx: &mut Ctx) -> ExactSizeOpaqueCursor<K> {
        let handle = ctx.bind(self.handle);
        // SAFETY: `handle` is the live handle produced for this iterator kind.
        let len = ctx
            .bind(unsafe { call_extern1_unchecked::<K::Len, _, OpaqueHandle>(self.len, handle) });
        ExactSizeOpaqueCursor {
            handle,
            len,
            pos: ctx.var(0u64),
            next_value: self.next_value,
            drop: self.drop,
        }
    }
}

/// The cursor of an [`ExactSizeOpaqueIter`]: a counted loop over `next_value`.
///
/// Not random access — `next_value` can only advance — so a `zip` pulls it.
pub struct ExactSizeOpaqueCursor<K: ExactSizeOpaqueIterKind> {
    handle: Var<OpaqueHandle>,
    len: Var<u64>,
    pos: Var<u64>,
    next_value: ExternRef<K::NextValue>,
    drop: ExternRef<K::Drop>,
}

impl<K: ExactSizeOpaqueIterKind> Cursor for ExactSizeOpaqueCursor<K> {
    type Item = K::Item;
    type Close = DropHandle<K::Drop>;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<K::Item>, Self::Close) {
        let pos = self.pos;
        ctx.exit_if(ge(pos, self.len), done);
        // SAFETY: `pos < len`, so `next_value` is never called past the end.
        let elem = ctx.bind(unsafe {
            call_extern1_unchecked::<K::NextValue, _, OpaqueHandle>(self.next_value, self.handle)
        });
        ctx.store(pos, add(pos, 1u64));
        (
            elem,
            DropHandle {
                handle: self.handle,
                drop: self.drop,
            },
        )
    }
}

// =============================================================================
// Generic kinds over `Box<dyn Iterator>` / `Box<dyn ExactSizeIterator>`
// =============================================================================
//
// The library supplies the `next`/`drop`/`len`/`next_value` extern fns
// generically (monomorphized per item type), so a user only writes the
// domain-specific producer that boxes their iterator. Structured values such as
// fat slices are reconstructed from the `COption<Item>` payload by the neutral layer.

/// A copyable staged item that can cross an opaque iterator's storage-pointer ABI.
pub trait OpaqueIterItem: StagedType + CopyType + Copy + 'static
where
    Self::RuntimeValue: Copy,
{
}

impl<T> OpaqueIterItem for T
where
    T: StagedType + CopyType + Copy + 'static,
    T::RuntimeValue: Copy,
{
}

/// RAII owner for a thin, double-boxed dynamic iterator handle.
///
/// Dropping this value releases an iterator that was never transferred to
/// generated code. Use [`into_raw`](Self::into_raw) only at the extern boundary
/// that returns the handle to a matching [`DynIter<T>`].
pub struct OpaqueIterOwner<'a, T: 'a> {
    raw: *mut Box<dyn Iterator<Item = T> + 'a>,
}

impl<T> Drop for OpaqueIterOwner<'_, T> {
    fn drop(&mut self) {
        // SAFETY: `raw` comes from `Box::into_raw` below and ownership has not
        // been transferred while this owner still exists.
        unsafe { drop(Box::from_raw(self.raw)) };
    }
}

impl<'a, T> OpaqueIterOwner<'a, T> {
    /// Transfer ownership into an erased handle returned by an extern producer.
    ///
    /// # Safety
    ///
    /// The handle must be consumed exactly once by the `next`/`drop` functions
    /// for [`DynIter<T>`], before any data borrowed by the iterator expires. It
    /// must not be reconstructed or freed through any other type.
    pub unsafe fn into_raw(self) -> *mut () {
        let raw = self.raw.cast();
        std::mem::forget(self);
        raw
    }
}

/// Double-box an iterator into an RAII-owned thin handle (the inner
/// `Box<dyn ..>` is a fat pointer; the outer box makes the handle one pointer).
///
/// The owner retains the iterator's borrow lifetime and frees it unless raw
/// ownership is explicitly transferred at an extern producer boundary. Pair
/// the transferred handle with [`DynIter<T>`].
pub fn box_dyn_iter<'a, T: 'a>(it: impl Iterator<Item = T> + 'a) -> OpaqueIterOwner<'a, T> {
    let inner: Box<dyn Iterator<Item = T> + 'a> = Box::new(it);
    OpaqueIterOwner {
        raw: Box::into_raw(Box::new(inner)),
    }
}

/// RAII owner for a thin, double-boxed exact-size iterator handle.
pub struct ExactOpaqueIterOwner<'a, T: 'a> {
    raw: *mut Box<dyn ExactSizeIterator<Item = T> + 'a>,
}

impl<T> Drop for ExactOpaqueIterOwner<'_, T> {
    fn drop(&mut self) {
        // SAFETY: `raw` comes from `Box::into_raw` below and ownership has not
        // been transferred while this owner still exists.
        unsafe { drop(Box::from_raw(self.raw)) };
    }
}

impl<'a, T> ExactOpaqueIterOwner<'a, T> {
    /// Transfer ownership into an erased handle returned by an extern producer.
    ///
    /// # Safety
    ///
    /// The handle must be consumed exactly once by the `len`/`next_value`/`drop`
    /// functions for [`DynExactIter<T>`], before any borrowed data expires. It
    /// must not be reconstructed or freed through any other type.
    pub unsafe fn into_raw(self) -> *mut () {
        let raw = self.raw.cast();
        std::mem::forget(self);
        raw
    }
}

/// Like [`box_dyn_iter`] but preserves `ExactSizeIterator` for the `len` /
/// counted-loop fast path. Pair a transferred handle with [`DynExactIter<T>`].
pub fn box_dyn_exact_iter<'a, T: 'a>(
    it: impl ExactSizeIterator<Item = T> + 'a,
) -> ExactOpaqueIterOwner<'a, T> {
    let inner: Box<dyn ExactSizeIterator<Item = T> + 'a> = Box::new(it);
    ExactOpaqueIterOwner {
        raw: Box::into_raw(Box::new(inner)),
    }
}

// SAFETY for all of the below: `it` is a handle from `box_dyn_iter` /
// `box_dyn_exact_iter` for the matching trait object and item type, used only
// within the call window where the boxed iterator is alive.

unsafe extern "C" fn dyn_next<T: Copy>(it: *mut ()) -> COption<T> {
    unsafe { (*(it as *mut Box<dyn Iterator<Item = T>>)).next().into() }
}
unsafe extern "C" fn dyn_drop<T>(it: *mut ()) {
    unsafe {
        drop(Box::from_raw(it as *mut Box<dyn Iterator<Item = T>>));
    }
}
unsafe extern "C" fn dyn_exact_next<T: Copy>(it: *mut ()) -> COption<T> {
    unsafe {
        (*(it as *mut Box<dyn ExactSizeIterator<Item = T>>))
            .next()
            .into()
    }
}
unsafe extern "C" fn dyn_len<T>(it: *mut ()) -> u64 {
    unsafe { (*(it as *mut Box<dyn ExactSizeIterator<Item = T>>)).len() as u64 }
}
unsafe extern "C" fn dyn_next_value<T>(it: *mut ()) -> T {
    unsafe {
        (*(it as *mut Box<dyn ExactSizeIterator<Item = T>>))
            .next()
            .expect("next_value called past len")
    }
}
unsafe extern "C" fn dyn_exact_drop<T>(it: *mut ()) {
    unsafe {
        drop(Box::from_raw(
            it as *mut Box<dyn ExactSizeIterator<Item = T>>,
        ));
    }
}

macro_rules! dyn_thunk {
    ($thunk:ident, $function:ident, $ret:ty) => {
        unsafe extern "C" fn $thunk<T: Copy>(input: *const u8, output: *mut u8) {
            let input = unsafe { input.cast::<*mut ()>().read() };
            let result: $ret = unsafe { $function::<T>(input) };
            if std::mem::size_of::<$ret>() != 0 {
                unsafe { output.cast::<$ret>().write(result) };
            }
        }
    };
}

dyn_thunk!(dyn_next_thunk, dyn_next, COption<T>);
dyn_thunk!(dyn_drop_thunk, dyn_drop, ());
dyn_thunk!(dyn_exact_next_thunk, dyn_exact_next, COption<T>);
dyn_thunk!(dyn_exact_drop_thunk, dyn_exact_drop, ());
dyn_thunk!(dyn_len_thunk, dyn_len, u64);
dyn_thunk!(dyn_next_value_thunk, dyn_next_value, T);

/// Generate a generic `ExternFn` marker over `T` for one of the library iterator
/// fns. The link name is disambiguated by the fn pointer at registration, so the
/// shared `NAME` across monomorphizations is fine. Every fn takes one `*mut ()`.
macro_rules! dyn_extern {
    ($Marker:ident, $name:ident, $thunk:ident, $ret_ty:ty) => {
        #[doc(hidden)]
        pub struct $Marker<T>(PhantomData<T>);
        unsafe impl<T> ExternFn for $Marker<T>
        where
            T: StagedType + Copy + 'static,
            T::RuntimeValue: Copy,
        {
            type Args = (OpaqueHandle,);
            type Ret = $ret_ty;
            const NAME: &'static str = stringify!($name);
            const FN_PTR: *const u8 = $thunk::<T::RuntimeValue> as *const u8;
        }
    };
}

dyn_extern!(DynNext, dyn_next, dyn_next_thunk, COptionType<T>);
dyn_extern!(DynDrop, dyn_drop, dyn_drop_thunk, ());
dyn_extern!(
    DynExactNext,
    dyn_exact_next,
    dyn_exact_next_thunk,
    COptionType<T>
);
dyn_extern!(DynExactDrop, dyn_exact_drop, dyn_exact_drop_thunk, ());
dyn_extern!(DynLen, dyn_len, dyn_len_thunk, u64);
dyn_extern!(DynNextValue, dyn_next_value, dyn_next_value_thunk, T);

/// Kind for a `Box<dyn Iterator<Item = T>>` (scalar `T`). Drive it with
/// [`Compiler::opaque_iter_fns`] over the producer's handle.
pub struct DynIter<T>(PhantomData<T>);
unsafe impl<T: OpaqueIterItem> OpaqueIterKind for DynIter<T>
where
    T::RuntimeValue: Copy,
{
    type Item = T;
    type Next = DynNext<T>;
    type Drop = DynDrop<T>;
}

/// Kind for a `Box<dyn ExactSizeIterator<Item = T>>` (scalar `T`): the counted
/// loop and O(1) `count` via [`Compiler::exact_opaque_iter_fns`] (also usable via
/// the plain next/drop path).
pub struct DynExactIter<T>(PhantomData<T>);
unsafe impl<T: OpaqueIterItem> OpaqueIterKind for DynExactIter<T>
where
    T::RuntimeValue: Copy,
{
    type Item = T;
    type Next = DynExactNext<T>;
    type Drop = DynExactDrop<T>;
}
unsafe impl<T: OpaqueIterItem> ExactSizeOpaqueIterKind for DynExactIter<T>
where
    T::RuntimeValue: Copy,
{
    type Len = DynLen<T>;
    type NextValue = DynNextValue<T>;
}

// =============================================================================
// Reused-storage opaque iterators (no per-call allocation in nested loops)
// =============================================================================
//
// For `for n in nodes { for dst in neighbours(g, n) { .. } }`, the inner
// iterator is created once *per outer element*. Instead of boxing each time, we
// reserve one slot per nesting level (a JIT stack slot) and reconstruct the
// iterator in place. Because the slot is type-erased, it carries its own
// 2-entry vtable (`next`/`drop` fn pointers) that the producer fills — see the
// module/`OpaqueIterSlot` docs.

use crate::staged::CompilationContext;
use std::mem::MaybeUninit;

/// Inline storage budget (bytes) for a reused-storage iterator. Iterators that
/// fit live in the slot (zero allocation); larger ones fall back to one heap box.
pub const OPAQUE_ITER_INLINE_CAP: usize = 256;

/// Reused per-nesting-level storage for one type-erased iterator.
///
/// Layout (`#[repr(C)]`, storage first so it gets the 16-byte alignment):
/// `[ storage: [u8; CAP] | next: fn | drop: fn | data: *mut u8 ]`.
/// `next`/`drop` are the hand-rolled mini-vtable filled by [`emplace_iter`];
/// `data` points into `storage` (inline) or at a heap box (fallback).
#[repr(C, align(16))]
pub struct OpaqueIterSlot<T> {
    storage: [MaybeUninit<u8>; OPAQUE_ITER_INLINE_CAP],
    next: Option<unsafe extern "C" fn(*const u8, *mut u8)>,
    drop: Option<unsafe extern "C" fn(*const u8, *mut u8)>,
    data: *mut u8,
    _item: PhantomData<T>,
}

/// Build `it` into `slot`: placed inline if it fits [`OPAQUE_ITER_INLINE_CAP`],
/// else one heap box. Fills the slot's mini-vtable with monomorphic `next`/`drop`
/// for `it`'s concrete type. Call this from a producer's `init` extern fn.
///
/// # Safety
///
/// `slot` must point to valid, properly aligned, exclusively writable storage
/// for an otherwise-uninitialized `OpaqueIterSlot<T>`. Once this function
/// returns, the caller must drive and drop the initialized iterator exactly
/// once before reusing the storage.
pub unsafe fn emplace_iter<T, I>(slot: *mut OpaqueIterSlot<T>, it: I)
where
    T: Copy,
    I: Iterator<Item = T>,
{
    unsafe {
        // Monomorphic mini-vtable thunks for the concrete `I` (known here in the
        // producer, type-erased on the staged side). Non-capturing → coerce to fn.
        unsafe extern "C" fn next_thunk<T: Copy, I: Iterator<Item = T>>(
            data: *const u8,
            output: *mut u8,
        ) {
            let data = unsafe { data.cast::<*mut u8>().read() };
            let result: COption<T> = unsafe { (*(data as *mut I)).next().into() };
            unsafe { output.cast::<COption<T>>().write(result) };
        }
        unsafe extern "C" fn drop_inline<I>(data: *const u8, _output: *mut u8) {
            let data = unsafe { data.cast::<*mut u8>().read() };
            unsafe { std::ptr::drop_in_place(data as *mut I) };
        }
        unsafe extern "C" fn drop_heap<I>(data: *const u8, _output: *mut u8) {
            let data = unsafe { data.cast::<*mut u8>().read() };
            unsafe { drop(Box::from_raw(data as *mut I)) };
        }

        // Treat the complete slot as uninitialized. Forming `&mut OpaqueIterSlot`
        // here would falsely claim that its function pointers and data pointer were
        // already initialized.
        let slot = &mut *slot.cast::<MaybeUninit<OpaqueIterSlot<T>>>();
        let slot = slot.as_mut_ptr();
        std::ptr::addr_of_mut!((*slot)._item).write(PhantomData);
        std::ptr::addr_of_mut!((*slot).next).write(Some(next_thunk::<T, I>));
        if std::mem::size_of::<I>() <= OPAQUE_ITER_INLINE_CAP
            && std::mem::align_of::<I>() <= std::mem::align_of::<OpaqueIterSlot<T>>()
        {
            let dst = std::ptr::addr_of_mut!((*slot).storage).cast::<I>();
            std::ptr::write(dst, it);
            std::ptr::addr_of_mut!((*slot).data).write(dst.cast());
            std::ptr::addr_of_mut!((*slot).drop).write(Some(drop_inline::<I>));
        } else {
            let data = Box::into_raw(Box::new(it)).cast();
            std::ptr::addr_of_mut!((*slot).data).write(data);
            std::ptr::addr_of_mut!((*slot).drop).write(Some(drop_heap::<I>));
        }
    }
}

/// A reused-storage opaque-iterator kind: just the `init` extern that builds the
/// iterator into a caller-provided slot (`init(args.., slot: *mut ())`).
/// # Safety
///
/// `Init` must initialize the final argument as exactly one
/// [`OpaqueIterSlot<Self::Item>`] using [`emplace_iter`]. If it returns, every
/// slot field must be initialized and the stored iterator must remain valid
/// until the generated traversal calls its drop thunk exactly once.
pub unsafe trait ReusedOpaqueIterKind: 'static {
    /// Copyable element type yielded by the erased iterator.
    type Item: OpaqueIterItem;
    /// `init(args.., slot: *mut ())` — calls [`emplace_iter`].
    type Init: ExternFn<Ret = ()>;
}

/// The resolved `init` ref for a kind, via [`Compiler::reused_opaque_iter_fns`].
pub struct ReusedOpaqueIterFns<K: ReusedOpaqueIterKind> {
    init: ExternRef<K::Init>,
}

impl<K: ReusedOpaqueIterKind> Clone for ReusedOpaqueIterFns<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: ReusedOpaqueIterKind> Copy for ReusedOpaqueIterFns<K> {}

impl<K: ReusedOpaqueIterKind> ReusedOpaqueIterFns<K> {
    /// One-argument producer, e.g. `nodes(g)`.
    pub fn iter1<A, AType>(self, a: A) -> ReusedOpaqueIter<K>
    where
        A: Staged<Out = AType> + 'static,
        AType: StagedType,
        K::Init: ExternFn<Args = (AType, OpaqueHandle), Ret = ()>,
    {
        ReusedOpaqueIter {
            init: self.init,
            args: Box::new(move |c| {
                let value = a.codegen(c);
                let mut args = Vec::with_capacity(1);
                let value = c.materialize_value(value);
                crate::ffi::push_extern_value::<AType>(c, &mut args, value);
                args
            }),
        }
    }

    /// Two-argument producer, e.g. `neighbours(g, n)`.
    pub fn iter2<A, B, AType, BType>(self, a: A, b: B) -> ReusedOpaqueIter<K>
    where
        A: Staged<Out = AType> + 'static,
        B: Staged<Out = BType> + 'static,
        AType: StagedType,
        BType: StagedType,
        K::Init: ExternFn<Args = (AType, BType, OpaqueHandle), Ret = ()>,
    {
        ReusedOpaqueIter {
            init: self.init,
            args: Box::new(move |c| {
                let a = a.codegen(c);
                let b = b.codegen(c);
                let mut args = Vec::with_capacity(2);
                let a = c.materialize_value(a);
                let b = c.materialize_value(b);
                crate::ffi::push_extern_value::<AType>(c, &mut args, a);
                crate::ffi::push_extern_value::<BType>(c, &mut args, b);
                args
            }),
        }
    }
}

impl Compiler {
    /// Register a reused-storage kind's `init` extern.
    pub fn reused_opaque_iter_fns<K: ReusedOpaqueIterKind>(&mut self) -> ReusedOpaqueIterFns<K> {
        ReusedOpaqueIterFns {
            init: self.extern_fn::<K::Init>(),
        }
    }
}

/// A [`StagedIterator`] backed by reused per-level storage. Each `for_each`
/// reserves one slot, so nested traversals allocate `O(depth)` (or zero, when
/// the iterators fit the inline budget) rather than once per inner set.
/// Codegens a producer's args (everything before the slot ptr) at compile time.
type ArgsCodegen = Box<dyn Fn(&mut CompilationContext) -> Vec<ValueId>>;

pub struct ReusedOpaqueIter<K: ReusedOpaqueIterKind> {
    init: ExternRef<K::Init>,
    args: ArgsCodegen,
}

impl<K: ReusedOpaqueIterKind> StagedIterator for ReusedOpaqueIter<K> {
    type Item = K::Item;
    type Cursor = ReusedOpaqueCursor<K>;

    /// Reserve this traversal's slot in the JIT frame (one per nesting level,
    /// reused every outer iteration) and let the producer build the iterator
    /// into it — which fills the slot's `next`/`drop` mini-vtable.
    fn open(self, ctx: &mut Ctx) -> ReusedOpaqueCursor<K> {
        let layout = SlotLayout::of::<K>();
        let init_id = self.init.extern_id;
        let args = self.args;
        let slot = ctx.bind_raw::<OpaqueHandle>(move |cctx| {
            let slot = cctx.alloc_stack_slot(layout.size, layout.align_shift);
            let slot_ptr = cctx.stack_addr(slot, 0);
            let mut a = (args)(cctx);
            crate::ffi::push_extern_value::<OpaqueHandle>(cctx, &mut a, slot_ptr);
            let init_ref = cctx.declare_extern_func(init_id);
            crate::ffi::emit_extern_call::<()>(cctx, init_ref, a);
            Value::scalar(slot_ptr)
        });
        // Scratch for the storage-pointer ABI of the indirect calls: the data
        // pointer argument, and the `COption<Item>` result.
        let scratch = ctx.bind_raw::<OpaqueHandle>(|cctx| {
            let s = cctx.alloc_stack_slot(8, 3);
            Value::scalar(cctx.stack_addr(s, 0))
        });
        let option = ctx.bind_raw::<OpaqueHandle>(|cctx| {
            let s = cctx.alloc_stack_slot(
                COptionType::<K::Item>::size_of() as u32,
                COptionType::<K::Item>::align_of().trailing_zeros() as u8,
            );
            Value::scalar(cctx.stack_addr(s, 0))
        });
        ReusedOpaqueCursor {
            slot: SlotVars {
                slot,
                scratch,
                option,
                layout,
            },
            _kind: PhantomData,
        }
    }
}

/// Offsets into an `OpaqueIterSlot<Item::RuntimeValue>`.
#[derive(Clone, Copy)]
struct SlotLayout {
    size: u32,
    align_shift: u8,
    next: i32,
    drop: i32,
    data: i32,
}

impl SlotLayout {
    fn of<K: ReusedOpaqueIterKind>() -> Self {
        type Slot<K> =
            OpaqueIterSlot<<<K as ReusedOpaqueIterKind>::Item as StagedType>::RuntimeValue>;
        SlotLayout {
            size: std::mem::size_of::<Slot<K>>() as u32,
            align_shift: std::mem::align_of::<Slot<K>>().trailing_zeros() as u8,
            next: std::mem::offset_of!(Slot<K>, next) as i32,
            drop: std::mem::offset_of!(Slot<K>, drop) as i32,
            data: std::mem::offset_of!(Slot<K>, data) as i32,
        }
    }
}

/// The per-traversal stack storage of a reused-slot iterator.
#[derive(Clone, Copy)]
pub struct SlotVars {
    slot: Var<OpaqueHandle>,
    scratch: Var<OpaqueHandle>,
    option: Var<OpaqueHandle>,
    layout: SlotLayout,
}

impl SlotVars {
    /// Call the slot's `next` (`at = next`) or `drop` (`at = drop`) through its
    /// mini-vtable: `f(&slot.data, &option)`.
    fn call_indirect(self, cctx: &mut CompilationContext, fn_off: i32) {
        let slot = cctx.resolve_var::<OpaqueHandle>(self.slot.id).leaf();
        let scratch = cctx.resolve_var::<OpaqueHandle>(self.scratch.id).leaf();
        let option = cctx.resolve_var::<OpaqueHandle>(self.option.id).leaf();
        // Canonical storage-pointer signature: (data slot ptr, output ptr) -> void.
        let sig = cctx.import_signature(&SigSpec {
            params: vec![ScalarType::Ptr, ScalarType::Ptr],
            ret: None,
        });
        let data = cctx.load(ScalarType::Ptr, slot, self.layout.data);
        let f = cctx.load(ScalarType::Ptr, slot, fn_off);
        cctx.store(data, scratch, 0);
        cctx.call_indirect(sig, f, &[scratch, option]);
    }
}

impl Close for SlotVars {
    /// Drop the iterator in place (and free it, if it was heap-boxed).
    fn close(self, ctx: &mut Ctx) {
        ctx.emit_raw(move |cctx| self.call_indirect(cctx, self.layout.drop));
    }
}

/// The cursor of a [`ReusedOpaqueIter`]: an indirect `next` per step through the
/// slot's mini-vtable.
///
/// ```text
/// (tag, val) = call_indirect slot.next (slot.data)   ; COption in the scratch
/// if tag == None → done
/// elem = val
/// ```
pub struct ReusedOpaqueCursor<K: ReusedOpaqueIterKind> {
    slot: SlotVars,
    _kind: PhantomData<K>,
}

impl<K: ReusedOpaqueIterKind> Cursor for ReusedOpaqueCursor<K> {
    type Item = K::Item;
    type Close = SlotVars;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<K::Item>, SlotVars) {
        let slot = self.slot;
        ctx.emit_raw(move |cctx| slot.call_indirect(cctx, slot.layout.next));
        // The COption the call wrote, read as a staged value.
        let opt = ptr_cast_mut::<COptionType<K::Item>, (), _>(slot.option);
        // SAFETY: `option` is this traversal's COption scratch, just written.
        let mut opt = ctx.bind(unsafe { load_mut(opt) });
        ctx.exit_if(is_none(opt.reborrow()), done);
        // SAFETY: the exit above leaves only the `Some` path.
        let elem = ctx.bind(unsafe { payload_unchecked(opt) });
        (elem, slot)
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;

    use super::*;

    struct DropIter<'a> {
        drops: &'a Cell<u32>,
    }

    impl Iterator for DropIter<'_> {
        type Item = u64;

        fn next(&mut self) -> Option<Self::Item> {
            None
        }
    }

    impl ExactSizeIterator for DropIter<'_> {
        fn len(&self) -> usize {
            0
        }
    }

    impl Drop for DropIter<'_> {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }

    #[test]
    fn iterator_owner_drops_an_untransferred_handle() {
        let drops = Cell::new(0);
        drop(box_dyn_iter(DropIter { drops: &drops }));
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn exact_iterator_owner_drops_an_untransferred_handle() {
        let drops = Cell::new(0);
        drop(box_dyn_exact_iter(DropIter { drops: &drops }));
        assert_eq!(drops.get(), 1);
    }

    #[test]
    fn transferred_handle_is_owned_by_matching_drop_function() {
        let drops = Cell::new(0);
        let owner = box_dyn_iter(DropIter { drops: &drops });
        // SAFETY: the handle is immediately consumed exactly once by the
        // matching `DynIter<u64>` drop implementation while `drops` is live.
        let raw = unsafe { owner.into_raw() };
        assert_eq!(drops.get(), 0);
        unsafe { dyn_drop::<u64>(raw) };
        assert_eq!(drops.get(), 1);
    }
}
