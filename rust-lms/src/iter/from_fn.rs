//! An iterator driven by a producer rather than by a length: [`from_fn`].

use std::marker::PhantomData;

use crate::func::Ctx;
use crate::staged::Var;
use crate::staged_opt::StagedOpt;

use super::traits::StagedIterator;

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
/// `next` may emit whatever it likes before yielding, including an early
/// [`break_loop`](Ctx::break_loop) — a guard checked *before* the pull, which a
/// `take_while` on the produced item could not express.
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
{
    type Item = O::Item;

    fn for_each<C>(self, ctx: &mut Ctx, consumer: C)
    where
        C: FnOnce(&mut Ctx, Var<Self::Item>),
    {
        let next = self.next;
        ctx.while_loop(true, move |ctx| {
            let produced = next(ctx);
            produced.eliminate(ctx, consumer, |ctx| ctx.break_loop());
        });
    }
}
