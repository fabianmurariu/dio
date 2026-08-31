//! Iterator over slice elements.
//!
//! Keyed on the *capability* ([`TrustedSliceType`]), not on one particular
//! marker, so iteration is an operation of a slice rather than an accident of
//! how the slice was obtained. A function parameter, a sub-slice, an `SVec`
//! view, and a promoted FFI descriptor all drive the same iterator.
//!
//! A raw descriptor deliberately does *not* iterate: it is not
//! `TrustedSliceType`, so nothing has vouched for the memory the loop would
//! read. Promote it with `assume_shared` first.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::num::{add, lt};
use crate::slice::{SliceGetUnchecked, SliceLen, SliceOps, TrustedSliceType};
use crate::staged::{LifetimeErased, Staged, Var, VarUse};
use crate::types::{ConstantType, CopyType, StagedType};

use super::traits::{IndexedSource, IndexedStagedIterator, IntoStagedIterator, StagedIterator};

/// Iterator over the elements of any trusted staged slice.
///
/// A raw descriptor cannot be iterated — it is not [`TrustedSliceType`], so
/// nothing has vouched for the memory the loop would read:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn raw_cannot_iterate(d: Var<RawSlice<i64>>, ctx: &mut Ctx) {
///     let _ = d.staged_iter().sum(ctx);
/// }
/// ```
///
/// Promote it first, and it iterates like any other slice:
///
/// ```ignore
/// unsafe { d.assume_shared() }.staged_iter().sum(ctx)
/// ```
pub struct SliceIter<T, S> {
    pub(crate) slice: S,
    _phantom: PhantomData<T>,
}

// Hand-written so the derive does not demand `T: Clone` — `T` is a staged
// *marker*, not a runtime value.
impl<T, S: Clone> Clone for SliceIter<T, S> {
    fn clone(&self) -> Self {
        SliceIter {
            slice: self.slice.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T, S> SliceIter<T, S>
where
    T: StagedType,
    S: LifetimeErased,
    S::Out: TrustedSliceType<Elem = T>,
{
    pub fn new(slice: S) -> Self {
        SliceIter {
            slice,
            _phantom: PhantomData,
        }
    }
}

impl<T, S> StagedIterator for SliceIter<T, S>
where
    T: StagedType + CopyType + ConstantType + 'static,
    S: LifetimeErased,
    S::Out: TrustedSliceType<Elem = T>,
    VarUse<S::Out>: LifetimeErased<Out = S::Out>,
    <VarUse<S::Out> as LifetimeErased>::ErasedOut: TrustedSliceType<Elem = T>,
    T::RuntimeValue: Default,
{
    type Item = T;

    /// Binds the slice once and reborrows per use rather than requiring
    /// `S: Clone`. A unique slice expression is deliberately not `Clone` — that
    /// *is* its uniqueness guarantee — so a `Clone` bound here would silently
    /// restrict iteration to shared origins.
    ///
    /// Bound with [`bind_lt`](Ctx::bind_lt), so a borrowing source — an `SVec`
    /// view — iterates through this same path. The borrow ends with the loop,
    /// because every value handed on is a borrow-free `Var`.
    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<T>),
    {
        let mut slice = ctx.bind_lt(self.slice);
        // Hoisted: the length is loop-invariant (the source's borrow forbids
        // growth while it is live), so this reads the descriptor once instead of
        // once per iteration.
        let n = ctx.bind_lt(slice.reborrow().len());
        let i = ctx.var(0u64);

        ctx.while_loop(lt(i, n), move |ctx| {
            // Bind the element *inside* the loop: no dead pre-loop init, and the
            // frontend resolves this single-def var to the loaded value with no
            // copy — so the emitted body matches a hand-written `while_loop`.
            // SAFETY: the loop condition proves `i < len`.
            let elem = ctx.bind_lt(unsafe { slice.reborrow().get_unchecked(i) });
            consumer(ctx, elem);
            ctx.store(i, add(i, 1u64));
        });
    }
}

// The indexed paths (`zip`, and anything needing the length without consuming
// the source) do keep `S: Clone`: `IndexedSource` takes `&self` and its
// supertrait requires `Clone`, so a unique origin cannot participate. Shared
// origins — which is every origin `zip` is used with — are unaffected.

impl<T, S> IndexedStagedIterator for SliceIter<T, S>
where
    T: StagedType + CopyType + ConstantType + 'static,
    S: Staged + Clone + 'static + LifetimeErased<Out = <S as Staged>::Out>,
    <S as Staged>::Out: TrustedSliceType<Elem = T>,
    VarUse<<S as Staged>::Out>: LifetimeErased<Out = <S as Staged>::Out>,
    <VarUse<<S as Staged>::Out> as LifetimeErased>::ErasedOut: TrustedSliceType<Elem = T>,
    T::RuntimeValue: Default,
{
    type LenExpr = SliceLen<S>;

    fn len(&self) -> Self::LenExpr {
        self.slice.clone().len()
    }
}

impl<T, S> IndexedSource for SliceIter<T, S>
where
    T: StagedType + CopyType + ConstantType + 'static,
    S: Staged + Clone + 'static + LifetimeErased<Out = <S as Staged>::Out>,
    <S as Staged>::Out: TrustedSliceType<Elem = T>,
    VarUse<<S as Staged>::Out>: LifetimeErased<Out = <S as Staged>::Out>,
    <VarUse<<S as Staged>::Out> as LifetimeErased>::ErasedOut: TrustedSliceType<Elem = T>,
    T::RuntimeValue: Default,
{
    type Item = T;
    type LenExpr = SliceLen<S>;
    type GetExpr = SliceGetUnchecked<S, Var<u64>>;

    fn count(&self) -> Self::LenExpr {
        self.slice.clone().len()
    }

    unsafe fn get_at(self, index: Var<u64>) -> Self::GetExpr {
        // SAFETY: forwarded from `IndexedSource::get_at`'s caller.
        unsafe { SliceOps::get_unchecked(self.slice, index) }
    }
}

// =============================================================================
// IndexedSource for slice variables (secondary source in zip)
// =============================================================================

/// Any `Copy` trusted slice variable is a random-access source. `CopyType`
/// bounds it to shared slices, which is what `IndexedSource: Clone` requires
/// anyway.
impl<T, R> IndexedSource for Var<R>
where
    T: StagedType + CopyType + ConstantType + 'static,
    R: TrustedSliceType<Elem = T> + CopyType + 'static,
{
    type Item = T;
    type LenExpr = SliceLen<Self>;
    type GetExpr = SliceGetUnchecked<Self, Var<u64>>;

    fn count(&self) -> Self::LenExpr {
        SliceOps::len(*self)
    }

    unsafe fn get_at(self, index: Var<u64>) -> Self::GetExpr {
        // SAFETY: forwarded from `IndexedSource::get_at`'s caller.
        unsafe { SliceOps::get_unchecked(self, index) }
    }
}

// =============================================================================
// IntoStagedIterator
// =============================================================================

/// Every trusted slice expression iterates — no `Clone` required, so a unique
/// slice works as well as a shared one.
impl<T, S> IntoStagedIterator for S
where
    T: StagedType + CopyType + ConstantType + 'static,
    S: LifetimeErased,
    S::Out: TrustedSliceType<Elem = T>,
    VarUse<S::Out>: LifetimeErased<Out = S::Out>,
    <VarUse<S::Out> as LifetimeErased>::ErasedOut: TrustedSliceType<Elem = T>,
    T::RuntimeValue: Default,
{
    type Iter = SliceIter<T, S>;

    fn staged_iter(self) -> Self::Iter {
        SliceIter::new(self)
    }
}
