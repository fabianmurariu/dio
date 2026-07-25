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

use crate::ffi::{call_extern1, ExternFn, ExternRef};
use crate::func::{Compiler, Ctx};
use crate::num::{add, lt};
use crate::option::COption;
use crate::refer::SMutPtr;
use crate::staged::{Staged, Var};
use crate::types::StagedType;

use super::traits::StagedIterator;

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
pub trait OpaqueIterKind: 'static {
    /// The element type (an integer ≤ 64 bits for the `next` register path).
    type Item: StagedType + 'static;
    /// `next(it) -> COption<Item>`.
    type Next: ExternFn;
    /// `drop(it)`.
    type Drop: ExternFn;
}

/// An [`OpaqueIterKind`] that also knows its length, enabling a tighter counted
/// loop and an O(1) `count`.
///
/// - `Len`: `extern "C" fn(*mut ()) -> u64`
/// - `NextValue`: `extern "C" fn(*mut ()) -> Item` (valid for `len` calls)
pub trait ExactSizeOpaqueIterKind: OpaqueIterKind {
    /// `len(it) -> u64`.
    type Len: ExternFn;
    /// `next_value(it) -> Item`, called exactly `len` times (no `Option`).
    type NextValue: ExternFn;
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
    pub fn iter<H>(self, handle: H) -> OpaqueIter<K, H>
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
    pub fn iter<H>(self, handle: H) -> ExactSizeOpaqueIter<K, H>
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

impl Compiler<'_> {
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
// OpaqueIter: next/drop source (register-consume loop)
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

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<K::Item>) + 'static,
    {
        // Bind the handle once, then drive the register-consume loop.
        let handle = ctx.bind(self.handle);
        ctx.opaque_for_each::<K::Item, F>(
            handle,
            self.next.extern_id,
            self.drop.extern_id,
            consumer,
        );
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
        let n = ctx.bind(call_extern1::<K::Len, _, OpaqueHandle, u64>(
            self.len, handle,
        ));
        ctx.emit(call_extern1::<K::Drop, _, OpaqueHandle, ()>(
            self.drop, handle,
        ));
        n
    }
}

impl<K, H> StagedIterator for ExactSizeOpaqueIter<K, H>
where
    K: ExactSizeOpaqueIterKind,
    H: Staged<Out = OpaqueHandle> + 'static,
{
    type Item = K::Item;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<K::Item>) + 'static,
    {
        let handle = ctx.bind(self.handle);
        let n = ctx.bind(call_extern1::<K::Len, _, OpaqueHandle, u64>(
            self.len, handle,
        ));
        let i = ctx.var(0u64);
        let next_value = self.next_value;
        ctx.while_loop(lt(i, n), move |ctx| {
            let v = ctx.bind(call_extern1::<K::NextValue, _, OpaqueHandle, K::Item>(
                next_value, handle,
            ));
            consumer(ctx, v);
            ctx.store(i, add(i, 1u64));
        });
        ctx.emit(call_extern1::<K::Drop, _, OpaqueHandle, ()>(
            self.drop, handle,
        ));
    }
}

// =============================================================================
// Generic kinds over `Box<dyn Iterator>` / `Box<dyn ExactSizeIterator>`
// =============================================================================
//
// The library supplies the `next`/`drop`/`len`/`next_value` extern fns
// generically (monomorphized per item type), so a user only writes the
// domain-specific producer that boxes their iterator. Scalar items only — a
// `COption<Item>` must fit in registers for the register-consume `next`; compound
// items use a separate out-param path.

/// Staged scalar item whose `COption` fits in registers (≤ 16 bytes), so the
/// register-consume `next` path is valid.
pub trait RegisterScalar: StagedType<RuntimeValue: Copy> + Copy + 'static {}
impl RegisterScalar for u64 {}
impl RegisterScalar for i64 {}
impl RegisterScalar for u32 {}
impl RegisterScalar for i32 {}
impl RegisterScalar for u16 {}
impl RegisterScalar for i16 {}
impl RegisterScalar for u8 {}
impl RegisterScalar for i8 {}
impl RegisterScalar for f32 {}
impl RegisterScalar for f64 {}
impl RegisterScalar for bool {}

/// Double-box an iterator into a thin `*mut ()` handle (the inner `Box<dyn ..>`
/// is a fat pointer; the outer box makes the handle a single pointer).
///
/// The iterator may borrow its source: the handle is created and dropped within
/// one kernel call, where the borrow is live (the opaque-pointer contract), so
/// the lifetime is erased here. Pair with [`DynIter<T>`].
pub fn box_dyn_iter<'a, T: 'a>(it: impl Iterator<Item = T> + 'a) -> *mut () {
    let inner: Box<dyn Iterator<Item = T> + 'a> = Box::new(it);
    Box::into_raw(Box::new(inner)) as *mut ()
}

/// Like [`box_dyn_iter`] but preserves `ExactSizeIterator` (the `len` /
/// counted-loop fast path). Pair with [`DynExactIter<T>`].
pub fn box_dyn_exact_iter<'a, T: 'a>(it: impl ExactSizeIterator<Item = T> + 'a) -> *mut () {
    let inner: Box<dyn ExactSizeIterator<Item = T> + 'a> = Box::new(it);
    Box::into_raw(Box::new(inner)) as *mut ()
}

// SAFETY for all of the below: `it` is a handle from `box_dyn_iter` /
// `box_dyn_exact_iter` for the matching trait object and item type, used only
// within the call window where the boxed iterator is alive.

unsafe extern "C" fn dyn_next<T: Copy>(it: *mut ()) -> COption<T> {
    (*(it as *mut Box<dyn Iterator<Item = T>>)).next().into()
}
unsafe extern "C" fn dyn_drop<T>(it: *mut ()) {
    drop(Box::from_raw(it as *mut Box<dyn Iterator<Item = T>>));
}
unsafe extern "C" fn dyn_exact_next<T: Copy>(it: *mut ()) -> COption<T> {
    (*(it as *mut Box<dyn ExactSizeIterator<Item = T>>))
        .next()
        .into()
}
unsafe extern "C" fn dyn_len<T>(it: *mut ()) -> u64 {
    (*(it as *mut Box<dyn ExactSizeIterator<Item = T>>)).len() as u64
}
unsafe extern "C" fn dyn_next_value<T>(it: *mut ()) -> T {
    (*(it as *mut Box<dyn ExactSizeIterator<Item = T>>))
        .next()
        .expect("next_value called past len")
}
unsafe extern "C" fn dyn_exact_drop<T>(it: *mut ()) {
    drop(Box::from_raw(
        it as *mut Box<dyn ExactSizeIterator<Item = T>>,
    ));
}

/// Generate a generic `ExternFn` marker over `T` for one of the library iterator
/// fns. The link name is disambiguated by the fn pointer at registration, so the
/// shared `NAME` across monomorphizations is fine. Every fn takes one `*mut ()`.
macro_rules! dyn_extern {
    ($Marker:ident, $func:ident, $ret:expr) => {
        #[doc(hidden)]
        pub struct $Marker<T>(PhantomData<T>);
        unsafe impl<T> ExternFn for $Marker<T>
        where
            T: StagedType + Copy + 'static,
            T::RuntimeValue: Copy,
        {
            const NAME: &'static str = stringify!($func);
            const NUM_PARAMS: usize = 1;
            const FN_PTR: *const u8 = $func::<T::RuntimeValue> as *const u8;
            fn param_abi_types() -> Vec<Vec<cranelift_codegen::ir::Type>> {
                vec![vec![cranelift_codegen::ir::types::I64]]
            }
            fn return_abi_types() -> Vec<cranelift_codegen::ir::Type> {
                $ret
            }
        }
    };
}

/// The type-correct ABI of a `COption<T>` return: an i64 discriminant followed
/// by the payload's real ABI types, so a scalar payload (e.g. `f64`) is returned
/// in the right register class. (`COptionType::abi_types` is all-i64 for the
/// generic struct paths; here we need the precise classification.)
fn coption_return_abi<T: StagedType>() -> Vec<cranelift_codegen::ir::Type> {
    let mut v = vec![cranelift_codegen::ir::types::I64];
    v.extend(T::abi_types());
    v
}

dyn_extern!(DynNext, dyn_next, coption_return_abi::<T>());
dyn_extern!(DynDrop, dyn_drop, <() as StagedType>::abi_types());
dyn_extern!(DynExactNext, dyn_exact_next, coption_return_abi::<T>());
dyn_extern!(
    DynExactDrop,
    dyn_exact_drop,
    <() as StagedType>::abi_types()
);
dyn_extern!(DynLen, dyn_len, <u64 as StagedType>::abi_types());
dyn_extern!(DynNextValue, dyn_next_value, <T as StagedType>::abi_types());

/// Kind for a `Box<dyn Iterator<Item = T>>` (scalar `T`). Drive it with
/// [`Compiler::opaque_iter_fns`] over the producer's handle.
pub struct DynIter<T>(PhantomData<T>);
impl<T: RegisterScalar> OpaqueIterKind for DynIter<T> {
    type Item = T;
    type Next = DynNext<T>;
    type Drop = DynDrop<T>;
}

/// Kind for a `Box<dyn ExactSizeIterator<Item = T>>` (scalar `T`): the counted
/// loop and O(1) `count` via [`Compiler::exact_opaque_iter_fns`] (also usable via
/// the plain next/drop path).
pub struct DynExactIter<T>(PhantomData<T>);
impl<T: RegisterScalar> OpaqueIterKind for DynExactIter<T> {
    type Item = T;
    type Next = DynExactNext<T>;
    type Drop = DynExactDrop<T>;
}
impl<T: RegisterScalar> ExactSizeOpaqueIterKind for DynExactIter<T> {
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
use cranelift_codegen::ir::{InstBuilder, Value};
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
    next: Option<unsafe extern "C" fn(*mut u8) -> COption<T>>,
    drop: Option<unsafe extern "C" fn(*mut u8)>,
    data: *mut u8,
}

/// Build `it` into `slot`: placed inline if it fits [`OPAQUE_ITER_INLINE_CAP`],
/// else one heap box. Fills the slot's mini-vtable with monomorphic `next`/`drop`
/// for `it`'s concrete type. Call this from a producer's `init` extern fn.
///
/// # Safety
/// `slot` must point to a live, otherwise-uninitialized `OpaqueIterSlot<T>`.
pub unsafe fn emplace_iter<T, I>(slot: *mut OpaqueIterSlot<T>, it: I)
where
    T: Copy,
    I: Iterator<Item = T>,
{
    // Monomorphic mini-vtable thunks for the concrete `I` (known here in the
    // producer, type-erased on the staged side). Non-capturing → coerce to fn.
    unsafe extern "C" fn next_thunk<T: Copy, I: Iterator<Item = T>>(data: *mut u8) -> COption<T> {
        (*(data as *mut I)).next().into()
    }
    unsafe extern "C" fn drop_inline<I>(data: *mut u8) {
        std::ptr::drop_in_place(data as *mut I);
    }
    unsafe extern "C" fn drop_heap<I>(data: *mut u8) {
        drop(Box::from_raw(data as *mut I));
    }

    let slot = &mut *slot;
    slot.next = Some(next_thunk::<T, I>);
    if std::mem::size_of::<I>() <= OPAQUE_ITER_INLINE_CAP
        && std::mem::align_of::<I>() <= std::mem::align_of::<OpaqueIterSlot<T>>()
    {
        let dst = slot.storage.as_mut_ptr() as *mut I;
        std::ptr::write(dst, it);
        slot.data = dst as *mut u8;
        slot.drop = Some(drop_inline::<I>);
    } else {
        slot.data = Box::into_raw(Box::new(it)) as *mut u8;
        slot.drop = Some(drop_heap::<I>);
    }
}

/// A reused-storage opaque-iterator kind: just the `init` extern that builds the
/// iterator into a caller-provided slot (`init(args.., slot: *mut ())`).
pub trait ReusedOpaqueIterKind: 'static {
    /// Element type (scalar; `T == T::RuntimeValue`).
    type Item: StagedType + Copy + 'static;
    /// `init(args.., slot: *mut ())` — calls [`emplace_iter`].
    type Init: ExternFn;
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
    pub fn iter1<A>(self, a: A) -> ReusedOpaqueIter<K>
    where
        A: Staged + 'static,
    {
        ReusedOpaqueIter {
            init: self.init,
            args: Box::new(move |c| vec![a.codegen(c)]),
        }
    }

    /// Two-argument producer, e.g. `neighbours(g, n)`.
    pub fn iter2<A, B>(self, a: A, b: B) -> ReusedOpaqueIter<K>
    where
        A: Staged + 'static,
        B: Staged + 'static,
    {
        ReusedOpaqueIter {
            init: self.init,
            args: Box::new(move |c| vec![a.codegen(c), b.codegen(c)]),
        }
    }
}

impl Compiler<'_> {
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
type ArgsCodegen = Box<dyn Fn(&mut CompilationContext) -> Vec<Value>>;

pub struct ReusedOpaqueIter<K: ReusedOpaqueIterKind> {
    init: ExternRef<K::Init>,
    args: ArgsCodegen,
}

impl<K: ReusedOpaqueIterKind> StagedIterator for ReusedOpaqueIter<K> {
    type Item = K::Item;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<K::Item>) + 'static,
    {
        type Slot<K> =
            OpaqueIterSlot<<<K as ReusedOpaqueIterKind>::Item as StagedType>::RuntimeValue>;
        let slot_size = std::mem::size_of::<Slot<K>>() as u32;
        let align_shift = std::mem::align_of::<Slot<K>>().trailing_zeros() as u8;
        let next_off = std::mem::offset_of!(Slot<K>, next) as i32;
        let drop_off = std::mem::offset_of!(Slot<K>, drop) as i32;
        let data_off = std::mem::offset_of!(Slot<K>, data) as i32;

        let init_id = self.init.extern_id;
        let args = self.args;
        ctx.reused_opaque_for_each::<K::Item, _, F>(
            slot_size,
            align_shift,
            next_off,
            drop_off,
            data_off,
            move |cctx, slot_ptr| {
                let mut a = (args)(cctx);
                a.push(slot_ptr);
                let init_ref = cctx.get_extern_func_ref(init_id);
                cctx.builder.ins().call(init_ref, &a);
            },
            consumer,
        );
    }
}
