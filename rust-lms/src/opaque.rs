//! Opaque external pointees.
//!
//! [`Opaque<T>`] marks a Rust value that generated code never inspects. It is
//! used behind [`SPtr`](crate::refer::SPtr) or
//! [`SMutPtr`](crate::refer::SMutPtr) and passed back to external functions.
//! `T` therefore need not implement [`StagedType`].
//!
//! `#[extern_fn]` maps `&T` to `SPtr<Opaque<T>>` and `&mut T` to
//! `SMutPtr<Opaque<T>>`. Calls with those arguments require an unsafe extern
//! constructor because raw staged pointers cannot prove Rust reference
//! validity, lifetime, or exclusivity.

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
