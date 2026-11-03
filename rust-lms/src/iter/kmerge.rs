//! k-way merge of sorted streams: [`KMergeBy`].
//!
//! The number of parts is known at staging time, so the merge is compiled for it, with
//! each part's step staged exactly once.
//!
//! Up to [`CHAIN_MAX`] parts, the merge is a chain of compare branches, as a hand-written
//! two-way merge: part `j` is next when its key is at most every later part's, and its
//! successor is pulled right there, so the branch that picks a part is the only
//! data-dependent one. Above, the smallest key is found by a pass of `select`s over all
//! keys, and the winner's part is refilled behind a [`switch`](Ctx::switch) (a jump
//! table). Its heads live in a stack array, of which only the winner's is read.
//!
//! A part that runs dry gets the key type's maximum as its key, so the compare path has
//! no liveness flags; the merge ends when the smallest key is that maximum. Keys must
//! therefore never equal it (`u64::MAX` for a `Vid`, `i128::MAX` for a packed
//! `EventTime`).

use crate::ffi::stack_array;
use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::num::{Ordered, add, eq, int_cast, le, lt, min, select};
use crate::refer::{SMutPtr, load_mut, ptr_offset_mut, store};
use crate::staged::{Const, Staged, Var};
use crate::types::{CopyType, StagedType};

use super::traits::{Close, Cursor, IntoStagedIterator, MinMax, StagedIterator};

/// The elements of `parts`, each sorted by `key`, merged into one sorted stream. Stable:
/// on ties the earlier part's element comes first.
///
/// `key` must never return `K::max_sentinel()`, which marks a dry part.
pub fn kmerge_by<I, F, KE, K>(parts: Vec<I>, key: F) -> KMergeBy<I::Iter, F>
where
    I: IntoStagedIterator,
    F: Fn(&Ctx, Var<<I::Iter as StagedIterator>::Item>) -> KE,
    KE: Staged<Out = K>,
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
{
    let parts: Vec<_> = parts.into_iter().map(|p| p.staged_iter()).collect();
    let plan = if parts.len() <= CHAIN_MAX {
        Plan::Chain
    } else {
        Plan::Linear
    };
    KMergeBy { parts, key, plan }
}

/// The most parts merged by a chain of compare branches; more by a pass over all keys.
pub const CHAIN_MAX: usize = 4;

/// How the next part is picked.
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plan {
    Chain,
    Linear,
}

/// Built by [`kmerge_by`].
pub struct KMergeBy<I, F> {
    parts: Vec<I>,
    key: F,
    plan: Plan,
}

impl<I, F> KMergeBy<I, F> {
    /// Override the choice between a chain of branches and a linear pass.
    #[doc(hidden)]
    pub fn with_plan(mut self, plan: Plan) -> Self {
        self.plan = plan;
        self
    }
}

impl<I, F, KE, K> StagedIterator for KMergeBy<I, F>
where
    I: StagedIterator,
    I::Item: CopyType + 'static,
    F: Fn(&Ctx, Var<I::Item>) -> KE,
    KE: Staged<Out = K> + 'static,
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
{
    type Item = I::Item;
    type Cursor = KMergeCursor<I::Cursor, F, K>;

    fn open(self, ctx: &Ctx) -> Self::Cursor {
        let cursors: Vec<_> = self.parts.into_iter().map(|p| p.open(ctx)).collect();
        let k = cursors.len();
        let state = match self.plan {
            Plan::Chain => State::Chain {
                heads: (0..k)
                    .map(|_| {
                        let head = ctx.var(dead::<I::Item>());
                        (head, ctx.var(Const::new(K::min_sentinel())))
                    })
                    .collect(),
                filled: ctx.var(0u32),
            },
            Plan::Linear => State::Linear(Linear {
                heads: ctx.bind(stack_array::<I::Item>(k)),
                keys: (0..k)
                    .map(|_| ctx.var(Const::new(K::max_sentinel())))
                    .collect(),
                refill: ctx.var(0u32),
                priming: ctx.var(true),
                bound: ctx.var(Const::new(K::min_sentinel())),
            }),
        };
        KMergeCursor {
            cursors,
            key: self.key,
            state,
        }
    }
}

enum State<K: StagedType, T: StagedType> {
    /// Each part's head and key, and how many parts have been pulled for the first time.
    Chain {
        heads: Vec<(Var<T>, Var<K>)>,
        filled: Var<u32>,
    },
    Linear(Linear<K, T>),
}

/// The parts' heads in a stack array and their keys; the part to refill at the next step
/// (while `priming`, the part being pulled for the first time); and the key below which
/// that part keeps winning.
struct Linear<K: StagedType, T: StagedType> {
    heads: Var<SMutPtr<T>>,
    keys: Vec<Var<K>>,
    refill: Var<u32>,
    priming: Var<bool>,
    bound: Var<K>,
}

/// The cursor of a [`KMergeBy`].
pub struct KMergeCursor<C: Cursor, F, K: StagedType> {
    cursors: Vec<C>,
    key: F,
    state: State<K, C::Item>,
}

impl<C: Close> Close for Vec<C> {
    fn close(self, ctx: &Ctx) {
        for c in self {
            c.close(ctx);
        }
    }
}

impl<C, F, KE, K> Cursor for KMergeCursor<C, F, K>
where
    C: Cursor,
    C::Item: CopyType + 'static,
    F: Fn(&Ctx, Var<C::Item>) -> KE,
    KE: Staged<Out = K> + 'static,
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
{
    type Item = C::Item;
    type Close = Vec<C::Close>;

    fn next(self, ctx: &Ctx, done: Label<'_>) -> (Var<C::Item>, Self::Close) {
        if self.cursors.is_empty() {
            ctx.exit(done);
            return (ctx.bind(dead::<C::Item>()), Vec::new());
        }
        let parts = self.cursors.into_iter().map(Some).collect();
        match self.state {
            State::Chain { heads, filled } => chain(ctx, parts, &self.key, &heads, filled, done),
            State::Linear(s) => linear(ctx, parts, &self.key, &s, done),
        }
    }
}

/// Pull `part`'s next element `x` and store its key in `head_key`, then run `then(x)`;
/// when the part is dry, set the key to MAX instead. Returns the part's close.
fn pull<C, F, KE, K>(
    ctx: &Ctx,
    part: C,
    key: &F,
    head_key: Var<K>,
    then: impl FnOnce(&Ctx, Var<C::Item>),
) -> C::Close
where
    C: Cursor,
    C::Item: CopyType + 'static,
    F: Fn(&Ctx, Var<C::Item>) -> KE,
    KE: Staged<Out = K> + 'static,
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
{
    let mut close = None;
    ctx.block(|ctx, pulled| {
        ctx.block(|ctx, exhausted| {
            let (x, c) = part.next(ctx, exhausted);
            close = Some(c);
            ctx.store(head_key, key(ctx, x));
            then(ctx, x);
            ctx.exit(pulled);
        });
        ctx.store(head_key, Const::new(K::max_sentinel()));
    });
    close.expect("each part is staged once")
}

/// The step of a [`Plan::Chain`]:
///
/// ```text
/// again:
///   taken: {
///     arm 0: { arm 1: { … {
///       if filled < k { if filled == 0 → arm 0; …; → arm k-1 }
///       if key[0] <= min(key[1..]) → arm 0; …; if key[k-2] <= key[k-1] → arm k-2
///       → arm k-1
///     } … }
///     arm j:  j == 0: if key[0] == MAX → done
///             x = head[j]; head[j], key[j] = part[j].next() (exhausted → MAX)
///             taken(x)
///   }
///   if filled < k { filled += 1; → again }
///   x
/// ```
/// The first part to pass is the first with the smallest key, so when every key is MAX
/// it is part 0, which is where the end is checked. Keys start at the minimum, so part
/// 0 is not taken for ended while priming.
fn chain<C, F, KE, K>(
    ctx: &Ctx,
    mut parts: Vec<Option<C>>,
    key: &F,
    heads: &[(Var<C::Item>, Var<K>)],
    filled: Var<u32>,
    done: Label<'_>,
) -> (Var<C::Item>, Vec<C::Close>)
where
    C: Cursor,
    C::Item: CopyType + 'static,
    F: Fn(&Ctx, Var<C::Item>) -> KE,
    KE: Staged<Out = K> + 'static,
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
{
    let k = parts.len();
    let mut closes = Vec::with_capacity(k);
    let item = ctx.repeat(|ctx, again| {
        let x = ctx.join(|ctx, taken| {
            let mut dispatch = |ctx: &Ctx, to: &To| {
                ctx.if_then(lt(filled, k as u32), |ctx| {
                    for j in 0..k - 1 {
                        to(ctx, j, Some(ctx.bind(eq(filled, j as u32))));
                    }
                    to(ctx, k - 1, None);
                });
                let mut later = heads[k - 1].1;
                let mut firsts = Vec::with_capacity(k);
                for j in (0..k - 1).rev() {
                    firsts.push(ctx.bind(le(heads[j].1, later)));
                    if j > 0 {
                        later = ctx.bind(min(heads[j].1, later));
                    }
                }
                for (j, first) in firsts.into_iter().rev().enumerate() {
                    to(ctx, j, Some(first));
                }
                to(ctx, k - 1, None);
            };
            let mut arm = |ctx: &Ctx, j: usize| {
                let (head, head_key) = heads[j];
                if j == 0 {
                    ctx.exit_if(eq(head_key, Const::new(K::max_sentinel())), done);
                }
                let x = ctx.bind(head);
                let part = parts[j].take().expect("each part is staged once");
                closes.push(pull(ctx, part, key, head_key, |ctx, y| ctx.store(head, y)));
                ctx.goto(taken, x);
            };
            arms(
                ctx,
                k - 1,
                &|_, _, _| unreachable!(),
                &mut dispatch,
                &mut arm,
            );
            dead::<C::Item>()
        });
        ctx.if_then(lt(filled, k as u32), |ctx| {
            ctx.store(filled, add(filled, 1u32));
            ctx.again(again);
        });
        x
    });
    (item, closes)
}

/// Jump to arm `j`, if the condition holds (always, without one).
type To<'a> = dyn Fn(&Ctx, usize, Option<Var<bool>>) + 'a;

/// Arms `0..=j`, each staged once after a block whose exit enters it, nested so that
/// `dispatch`, staged innermost, can jump to any of them.
fn arms(
    ctx: &Ctx,
    j: usize,
    outer: &To,
    dispatch: &mut dyn FnMut(&Ctx, &To),
    arm: &mut dyn FnMut(&Ctx, usize),
) {
    ctx.block(|ctx, enter| {
        let to = |ctx: &Ctx, i: usize, cond: Option<Var<bool>>| match (i == j, cond) {
            (true, Some(c)) => ctx.exit_if(c, enter),
            (true, None) => ctx.exit(enter),
            (false, _) => outer(ctx, i, cond),
        };
        if j == 0 {
            dispatch(ctx, &to);
        } else {
            arms(ctx, j - 1, &to, dispatch, arm);
        }
    });
    arm(ctx, j);
}

/// The step of a [`Plan::Linear`]:
///
/// ```text
/// got: {
///   again:
///     switch refill {
///       j: key[j] = key(x = part[j].next()) (exhausted → MAX)
///          if key[j] < bound → got(x); head[j] = x
///     }
///     if priming { if refill < k - 1 { refill += 1; → again }; priming = false }
///     (w, m, bound) = argmin, min and runner-up of key[..]
///     if m == MAX → done
///     refill = w; got(head[w])
/// }
/// ```
/// `bound` starts at the key's minimum, so no head passes it before the first pass. A
/// part that keeps winning (parts that don't interleave) costs one compare per element.
fn linear<C, F, KE, K>(
    ctx: &Ctx,
    mut parts: Vec<Option<C>>,
    key: &F,
    s: &Linear<K, C::Item>,
    done: Label<'_>,
) -> (Var<C::Item>, Vec<C::Close>)
where
    C: Cursor,
    C::Item: CopyType + 'static,
    F: Fn(&Ctx, Var<C::Item>) -> KE,
    KE: Staged<Out = K> + 'static,
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
{
    let k = parts.len() as u32;
    let mut closes = Vec::with_capacity(parts.len());
    let item = ctx.join(|ctx, got| {
        ctx.repeat(|ctx, again| {
            ctx.switch(s.refill, k, |ctx, j| {
                let part = parts[j as usize].take().expect("each part is staged once");
                let head_key = s.keys[j as usize];
                closes.push(pull(ctx, part, key, head_key, |ctx, x| {
                    ctx.if_then(lt(head_key, s.bound), |ctx| ctx.goto(got, x));
                    // SAFETY: `j < k`, within the array.
                    unsafe { ctx.emit(store(ptr_offset_mut(s.heads, Const::new(j as i64)), x)) };
                }));
            });
            ctx.if_then(s.priming, |ctx| {
                ctx.if_then(lt(s.refill, k - 1), |ctx| {
                    ctx.store(s.refill, add(s.refill, 1u32));
                    ctx.again(again);
                });
                ctx.store(s.priming, false);
            });
            let (least, winner) = argmin(ctx, &s.keys, s.bound);
            ctx.exit_if(eq(least, Const::new(K::max_sentinel())), done);
            ctx.store(s.refill, winner);
            // SAFETY: `winner < k`, a part whose head was stored: its key is not MAX.
            let head =
                unsafe { load_mut(ptr_offset_mut(s.heads, int_cast::<i64, u32, _>(winner))) };
            ctx.goto(got, head);
        });
        dead::<C::Item>()
    });
    (item, closes)
}

/// The smallest of `keys` and its part (the lowest on ties); `bound` becomes the
/// runner-up key.
fn argmin<K>(ctx: &Ctx, keys: &[Var<K>], bound: Var<K>) -> (Var<K>, Var<u32>)
where
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
{
    let mut best = (keys[0], ctx.bind(Const::new(0u32)));
    let mut runner_up: Var<K> = ctx.bind(Const::new(K::max_sentinel()));
    for (j, &key) in keys.iter().enumerate().skip(1) {
        let better = ctx.bind(lt(key, best.0));
        runner_up = ctx.bind(select(better, best.0, min(runner_up, key)));
        best = (
            ctx.bind(select(better, key, best.0)),
            ctx.bind(select(better, Const::new(j as u32), best.1)),
        );
    }
    ctx.store(bound, runner_up);
    best
}
