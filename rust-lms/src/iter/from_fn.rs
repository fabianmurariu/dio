//! An iterator driven by a producer rather than by a length: [`from_fn`].

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::staged::Var;
use crate::staged_opt::StagedOpt;

use crate::label::{Label, dead};

use super::traits::{Cursor, StagedIterator};

/// A source with no length, driven by a producer that reports exhaustion.
///
/// Built by [`from_fn`].
pub struct FromFn<F, O> {
    next: F,
    _phantom: PhantomData<fn() -> O>,
}

/// Iterate a stream whose end is only known by asking it.
///
/// `next` is called **once**, at staging time, to emit the pull; the
/// [`StagedOpt`] it returns becomes the loop's exit test, so `None` ends the
/// iteration. This is the shape a pull-based source has — an FFI callback
/// returning a null sentinel, a cursor, a channel — where no count is available
/// up front and `count`/`len`/`zip` therefore do not apply.
///
/// ```ignore
/// from_fn(|ctx| {
///     let item = ctx.bind(pull_next(source));
///     not(ptr_is_null(item)).then_some(item)
/// })
/// .for_each(ctx, |ctx, item| { /* ... */ });
/// ```
///
/// `next` may emit whatever it likes before yielding. To stop *before* pulling
/// — a guard a `take_while` on the produced item could not express — return
/// `None` from the guard (`guard.then_some(..)` is lazy, so the pull is only
/// emitted on the `Some` side). Don't `break_loop` from `next`: inside a
/// `chain` that would end the whole pipeline instead of this stream.
pub fn from_fn<F, O>(next: F) -> FromFn<F, O>
where
    F: FnOnce(&mut Ctx) -> O,
    O: StagedOpt,
{
    FromFn {
        next,
        _phantom: PhantomData,
    }
}

impl<F, O> StagedIterator for FromFn<F, O>
where
    F: FnOnce(&mut Ctx) -> O,
    O: StagedOpt,
    O::Item: 'static,
{
    type Item = O::Item;
    type Cursor = Self;

    fn open(self, _ctx: &mut Ctx) -> Self {
        self
    }
}

impl<F, O> Cursor for FromFn<F, O>
where
    F: FnOnce(&mut Ctx) -> O,
    O: StagedOpt,
    O::Item: 'static,
{
    type Item = O::Item;
    type Close = ();

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<O::Item>, ()) {
        let next = self.next;
        let item = ctx.join(|ctx, got| {
            next(ctx).eliminate(ctx, |ctx, v| ctx.goto(got, v), |ctx| ctx.exit(done));
            dead::<O::Item>()
        });
        (item, ())
    }
}
