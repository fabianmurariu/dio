//! The positions of a word's set bits: [`set_bits`].

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::label::Label;
use crate::num::{IntNum, bitand, eq, sub, trailing_zeros};
use crate::staged::{Const, IntoStaged, Staged, Var};
use crate::types::{ConstantType, StagedType};

use super::range_iter::RangeStep;
use super::traits::{Cursor, StagedIterator};

/// Built by [`set_bits`].
pub struct SetBits<T, W> {
    word: W,
    _phantom: PhantomData<T>,
}

/// The indices of the set bits of `word`, lowest first.
///
/// Each step counts trailing zeros and clears the lowest set bit, so the loop
/// runs once per set bit and a zero word ends at once.
pub fn set_bits<T, W>(word: W) -> SetBits<T, W::Staged>
where
    T: IntNum + RangeStep + ConstantType,
    W: IntoStaged<T>,
{
    SetBits {
        word: word.into_staged(),
        _phantom: PhantomData,
    }
}

impl<T, W> StagedIterator for SetBits<T, W>
where
    T: IntNum + RangeStep + ConstantType + 'static,
    T::RuntimeValue: Default,
    W: Staged<Out = T> + 'static,
{
    type Item = T;
    type Cursor = SetBitsCursor<T>;

    fn open(self, ctx: &Ctx) -> Self::Cursor {
        SetBitsCursor {
            rest: ctx.var(self.word),
        }
    }
}

/// The cursor of a [`SetBits`]: the bits not yet visited.
pub struct SetBitsCursor<T: StagedType> {
    rest: Var<T>,
}

impl<T> Cursor for SetBitsCursor<T>
where
    T: IntNum + RangeStep + ConstantType + 'static,
    T::RuntimeValue: Default,
{
    type Item = T;
    type Close = ();

    fn next(self, ctx: &Ctx, done: Label<'_>) -> (Var<T>, ()) {
        let rest = self.rest;
        ctx.exit_if(eq(rest, Const::<T>::new(Default::default())), done);
        let bit = ctx.bind(trailing_zeros(rest));
        ctx.store(rest, bitand(rest, sub(rest, Const::<T>::new(T::one()))));
        (bit, ())
    }
}
