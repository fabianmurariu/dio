//! Opaque external pointees.
//!
//! [`Opaque<T>`] marks a Rust value that generated code never inspects. It is
//! used behind staged references or raw pointers and passed back to external functions.
//! `T` therefore need not implement [`StagedType`].
//!
//! `#[extern_fn]` maps `&T` to `SRef<Opaque<T>>` and `&mut T` to
//! `SRefMut<Opaque<T>>`. Safe extern calls require those reference markers;
//! passing a raw staged pointer instead remains an explicit unchecked call.

use std::marker::PhantomData;

use crate::types::StagedType;
use cranelift_codegen::ir::types;

/// Marker for an externally-owned value of type `T` that staged code may only
/// address opaquely.
pub struct Opaque<T> {
    _phantom: PhantomData<T>,
}

unsafe impl<T> StagedType for Opaque<T> {
    type RuntimeValue = T;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }
}
