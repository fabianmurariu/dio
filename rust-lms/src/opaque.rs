//! Opaque external handles.
//!
//! [`Opaque<T>`] marks a Rust value that staged code only ever holds *by
//! reference* and never inspects — a pointer it receives at the ABI boundary
//! and hands straight back to `extern "C"` functions. Unlike a `#[repr(C)]`
//! struct, `T` need **not** implement [`StagedType`] and has no staged field
//! access; the staged side sees only a pointer-sized value.
//!
//! Use it behind the existing reference types:
//! - `SRef<'a, Opaque<T>>`    — `&'a T`     at runtime
//! - `SRefMut<'a, Opaque<T>>` — `&'a mut T` at runtime
//!
//! ```ignore
//! type FFIGraph<'a> = SRef<'a, Opaque<Graph>>;   // &Graph, opaque to staged code
//!
//! // A kernel taking it compiles to `extern "C" fn(&Graph) -> _`:
//! compiler.fun1("nodes", |_ctx, g: Var<FFIGraph>| call_extern1(all_nodes, g));
//! ```
//!
//! `#[extern_fn]` maps a plain `&T` / `&mut T` parameter to
//! `SRef<Opaque<T>>` / `SRefMut<Opaque<T>>`, so opaque handles flow through
//! external calls with no annotation.

use std::marker::PhantomData;

use crate::refer::{ConstPtr, SRef, SRefMut};
use crate::staged::{CompilationContext, IntoStaged, Staged};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, Value};

/// Marker for an opaque, externally-owned value of type `T`, handled only
/// behind `SRef`/`SRefMut`. See the [module docs](self).
pub struct Opaque<T> {
    _phantom: PhantomData<T>,
}

impl<T> StagedType for Opaque<T> {
    /// Behind a reference this surfaces as `&'a T` / `&'a mut T`.
    type RuntimeValue = T;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        // Only ever materialized behind a reference, which is pointer-sized.
        types::I64
    }
}

/// Reinterpret a raw address (a staged `u64`) as `SRef<'static, Opaque<T>>` — an
/// opaque `&T` — for handing a stored pointer to an extern fn that takes `&T`.
/// Emits no code: a reference *is* the pointer value.
pub struct OpaqueRef<E, T> {
    addr: E,
    _t: PhantomData<T>,
}

impl<E: Clone, T> Clone for OpaqueRef<E, T> {
    fn clone(&self) -> Self {
        Self {
            addr: self.addr.clone(),
            _t: PhantomData,
        }
    }
}

impl<E: Copy, T> Copy for OpaqueRef<E, T> {}

impl<E, T> Staged for OpaqueRef<E, T>
where
    E: Staged<Out = u64>,
    T: 'static,
{
    type Out = SRef<'static, Opaque<T>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.addr.codegen(ctx)
    }
}

/// Reinterpret a staged raw address as an opaque `&T` (`SRef<Opaque<T>>`).
pub fn opaque_ref<T: 'static, E: IntoStaged<u64>>(addr: E) -> OpaqueRef<E::Staged, T> {
    OpaqueRef {
        addr: addr.into_staged(),
        _t: PhantomData,
    }
}

/// Mutable twin of [`OpaqueRef`]: reinterpret a raw address as
/// `SRefMut<'static, Opaque<T>>` — an opaque `&mut T` — for an extern fn taking
/// `&mut T`. Like `OpaqueRef`, emits no code (a reference *is* the pointer).
pub struct OpaqueRefMut<E, T> {
    addr: E,
    _t: PhantomData<T>,
}

impl<E: Clone, T> Clone for OpaqueRefMut<E, T> {
    fn clone(&self) -> Self {
        Self {
            addr: self.addr.clone(),
            _t: PhantomData,
        }
    }
}

impl<E: Copy, T> Copy for OpaqueRefMut<E, T> {}

impl<E, T> Staged for OpaqueRefMut<E, T>
where
    E: Staged<Out = u64>,
    T: 'static,
{
    type Out = SRefMut<'static, Opaque<T>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.addr.codegen(ctx)
    }
}

/// Reinterpret a staged raw address as an opaque `&mut T` (`SRefMut<Opaque<T>>`).
pub fn opaque_ref_mut<T: 'static, E: IntoStaged<u64>>(addr: E) -> OpaqueRefMut<E::Staged, T> {
    OpaqueRefMut {
        addr: addr.into_staged(),
        _t: PhantomData,
    }
}

/// Bake a host `*const T` as a staged opaque `&T` (`SRef<Opaque<T>>`) — the typed
/// counterpart of [`opaque_ref`] for a **baked** host struct (address known at
/// stage 0, owner outlives the run), e.g. a GROUP BY table handed to an extern.
pub fn const_opaque<T: 'static>(p: *const T) -> ConstPtr<SRef<'static, Opaque<T>>> {
    ConstPtr::from_addr(p as usize)
}

/// Bake a host `*mut T` as a staged opaque `&mut T` (`SRefMut<Opaque<T>>`).
pub fn const_opaque_mut<T: 'static>(p: *mut T) -> ConstPtr<SRefMut<'static, Opaque<T>>> {
    ConstPtr::from_addr(p as usize)
}
