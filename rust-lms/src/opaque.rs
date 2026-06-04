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

use crate::types::StagedType;
use cranelift_codegen::ir::types;

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
