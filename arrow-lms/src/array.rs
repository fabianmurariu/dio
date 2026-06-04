//! Staged views over an Arrow column.
//!
//! The values buffer of an [`FfiArray`] is stored inline as a [`FatSlice`], so
//! the *address* of that field is — bit for bit — a memory-resolved staged
//! `Slice<M>` (a pointer to a `(ptr, len)` pair). [`AsSlice`] is the tiny
//! adapter that re-types it as exactly that, after which **every** `rust-lms`
//! combinator and terminal applies for free:
//!
//! ```ignore
//! compiler.fun1("sum", |ctx, arr: Var<SRef<FfiArray<I32Type>>>| {
//!     arr.values().filter(|x| lt(0i32, x)).sum(ctx)
//! });
//! ```
//!
//! There is no bespoke iterator source here anymore — `ValuesI32`/`ArrowGetI32`
//! are gone; the work is done by `SliceIter` over the bridged slice.

use cranelift_codegen::ir::Value;
use rust_lms::prelude::*;
use std::marker::PhantomData;

use crate::ffi::{FfiArray, FfiArrayType};

// =============================================================================
// AsSlice: view an inline `FatSlice<M>` field as a staged `Slice<M>`
// =============================================================================

/// Re-types a reference-to-`FatSliceType<M>` as a reference-to-`Slice<M>`.
///
/// The two share a representation — a memory-resolved slice *is* "a pointer to a
/// `(ptr, len)` pair", which is exactly the address of an inline `FatSlice`
/// field. So this forwards `codegen` unchanged and only changes the staged
/// `Out` type; with no `var_id`, the slice ops take the memory-resolved path and
/// load `ptr`/`len` from offsets 0/8.
pub struct AsSlice<P, M> {
    inner: P,
    _elem: PhantomData<M>,
}

impl<P: Clone, M> Clone for AsSlice<P, M> {
    fn clone(&self) -> Self {
        AsSlice {
            inner: self.inner.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, M> Copy for AsSlice<P, M> {}

impl<'a, P, M> Staged for AsSlice<P, M>
where
    P: Staged<Out = SRef<'a, FatSliceType<M>>>,
    M: StagedType + 'a,
{
    type Out = SRef<'a, Slice<M>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.inner.codegen(ctx)
    }
}

// =============================================================================
// ArrowArrayOps: the staged column API
// =============================================================================

/// Staged operations on an Arrow column parameter (`Var<SRef<FfiArray<M>>>`).
///
/// [`values`](Self::values) is the entry point: a `Slice<M>` source that plugs
/// straight into `SliceIter` and every combinator/terminal. `'r` is the staged
/// reference's lifetime, `'a` the descriptor's borrow of the batch.
pub trait ArrowArrayOps<'r, 'a, M: StagedType + 'r>: Sized {
    /// The values buffer as a staged `Slice<M>` iterator source.
    fn values(self) -> impl Staged<Out = SRef<'r, Slice<M>>> + Clone + 'r;

    /// Number of null entries, as a staged value.
    fn null_count(self) -> impl Staged<Out = U64Type>;
}

impl<'r, 'a, M> ArrowArrayOps<'r, 'a, M> for Var<SRef<'r, FfiArray<'a, M>>>
where
    M: StagedType + 'r,
    M::RuntimeValue: 'a,
    'a: 'r,
{
    fn values(self) -> impl Staged<Out = SRef<'r, Slice<M>>> + Clone + 'r {
        AsSlice {
            inner: self.get_ref(FfiArrayType::values()),
            _elem: PhantomData,
        }
    }

    fn null_count(self) -> impl Staged<Out = U64Type> {
        self.get(FfiArrayType::null_count())
    }
}
