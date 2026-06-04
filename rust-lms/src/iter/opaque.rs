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
use crate::refer::SMutPtr;
use crate::staged::{Staged, Var};
use crate::types::{StagedType, U64Type, UnitType};

use super::traits::StagedIterator;

/// The staged type of an opaque iterator handle: a thin `*mut ()` pointer.
///
/// Producer/`next`/`drop`/… extern fns all take or return `*mut ()`, which
/// `#[extern_fn]` maps to this type.
pub type OpaqueHandle = SMutPtr<UnitType>;

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
    pub fn count(self, ctx: &mut Ctx) -> Var<U64Type> {
        let handle = ctx.bind(self.handle);
        let n = ctx.bind(call_extern1::<K::Len, _, OpaqueHandle, U64Type>(
            self.len, handle,
        ));
        ctx.emit(call_extern1::<K::Drop, _, OpaqueHandle, UnitType>(
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
        let n = ctx.bind(call_extern1::<K::Len, _, OpaqueHandle, U64Type>(
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
        ctx.emit(call_extern1::<K::Drop, _, OpaqueHandle, UnitType>(
            self.drop, handle,
        ));
    }
}
