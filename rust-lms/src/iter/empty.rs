//! The iterator with no elements.

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::staged::Var;
use crate::types::StagedType;

use super::traits::{Cursor, StagedIterator};

/// No elements: its step jumps straight to `done`. Built by [`empty`].
pub struct Empty<T>(PhantomData<fn() -> T>);

/// An iterator of `T` with no elements, as `std::iter::empty`: what an empty
/// `chain` of iterators chosen at stage 0 is.
pub fn empty<T: StagedType>() -> Empty<T> {
    Empty(PhantomData)
}

impl<T: StagedType + 'static> StagedIterator for Empty<T> {
    type Item = T;
    type Cursor = EmptyCursor<T>;

    fn open(self, _ctx: &Ctx) -> Self::Cursor {
        EmptyCursor(PhantomData)
    }

    fn count(self, ctx: &Ctx) -> Var<u64> {
        ctx.var(0u64)
    }
}

/// The cursor of an [`Empty`].
pub struct EmptyCursor<T>(PhantomData<fn() -> T>);

impl<T: StagedType + 'static> Cursor for EmptyCursor<T> {
    type Item = T;
    type Close = ();

    fn next(self, ctx: &Ctx, done: Label<'_>) -> (Var<T>, ()) {
        ctx.exit(done);
        (ctx.bind(dead::<T>()), ())
    }
}
