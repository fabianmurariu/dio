//! k-way merge of sorted streams: [`KMergeBy`].
//!
//! The number of parts is known at staging time, so the merge is compiled for it, with
//! each part's step staged exactly once behind a [`switch`](Ctx::switch) (a jump table)
//! on the part to refill.
//!
//! Up to [`LINEAR_MAX`] parts, each part's head and key are variables, and the smallest
//! key is found by a pass of `select`s over all of them. Above, by a loser tree in stack
//! arrays: each internal node holds the key, part and head of the loser of its match, so
//! refilling part `j` replays the path from leaf `j` to the root, `log2 k` loads,
//! compare-and-selects and stores. The path is fixed per part, so it is staged inside
//! part `j`'s refill with constant offsets. Keeping the tree in memory keeps it out of
//! the registers the loop carries.
//!
//! A part that runs dry gets the key type's maximum as its key, so the compare path has
//! no liveness flags; the merge ends when the smallest key is that maximum. Keys must
//! therefore never equal it (`u64::MAX` for a `Vid`, `i128::MAX` for a packed
//! `EventTime`).
//!
//! With runs on, the merge remembers a bound below which the winner's part keeps winning:
//! the runner-up key of the last pass, or the least loser on the winner's path in the
//! tree. A refilled head whose key is below it is the next element, so parts that don't
//! interleave cost one compare per element.

use crate::ffi::stack_array;
use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::num::{Ordered, add, bitand, bitor, eq, lt, min, select};
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
    let plan = if parts.len() <= LINEAR_MAX {
        Plan::Linear
    } else {
        Plan::Tree
    };
    KMergeBy {
        parts,
        key,
        runs: true,
        plan,
    }
}

/// The most parts merged by a pass over all keys; more use a loser tree.
pub const LINEAR_MAX: usize = 8;

/// How the smallest key is found.
#[doc(hidden)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Plan {
    Linear,
    Tree,
}

/// Built by [`kmerge_by`].
pub struct KMergeBy<I, F> {
    parts: Vec<I>,
    key: F,
    runs: bool,
    plan: Plan,
}

impl<I, F> KMergeBy<I, F> {
    /// Turn the runner-up shortcut off: every element takes a full pass over the keys.
    #[doc(hidden)]
    pub fn without_runs(mut self) -> Self {
        self.runs = false;
        self
    }

    /// Override the choice between a linear pass and a loser tree.
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
        let heads = match self.plan {
            Plan::Linear => Heads::Vars(
                (0..k)
                    .map(|_| {
                        let head = ctx.var(dead::<I::Item>());
                        (head, ctx.var(Const::new(K::max_sentinel())))
                    })
                    .collect(),
            ),
            Plan::Tree => Heads::Tree(Tree {
                keys: ctx.bind(stack_array::<K>(2 * k)),
                parts: ctx.bind(stack_array::<u32>(2 * k)),
                heads: ctx.bind(stack_array::<I::Item>(2 * k)),
                winner: Entry::declare(ctx),
            }),
        };
        KMergeCursor {
            cursors,
            key: self.key,
            refill: ctx.var(0u32),
            priming: ctx.var(true),
            bound: self.runs.then(|| ctx.var(Const::new(K::min_sentinel()))),
            heads,
        }
    }
}

/// A contender: a part's head, its key, and the part's index for ties.
struct Entry<K: StagedType, T: StagedType> {
    key: Var<K>,
    part: Var<u32>,
    head: Var<T>,
}

impl<K: CopyType, T: CopyType> Clone for Entry<K, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: CopyType, T: CopyType> Copy for Entry<K, T> {}

impl<K, T> Entry<K, T>
where
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
    T: CopyType + 'static,
{
    fn declare(ctx: &Ctx) -> Self {
        Entry {
            key: ctx.var(Const::new(K::max_sentinel())),
            part: ctx.var(0u32),
            head: ctx.var(dead::<T>()),
        }
    }

    /// Whether `self` comes before `other`: a smaller key, or an equal key from an
    /// earlier part.
    fn beats(self, other: Self) -> impl Staged<Out = bool> + 'static {
        bitor(
            lt(self.key, other.key),
            bitand(eq(self.key, other.key), lt(self.part, other.part)),
        )
    }

    /// `a` if `c`, else `b`.
    fn pick(ctx: &Ctx, c: Var<bool>, a: Self, b: Self) -> Self {
        Entry {
            key: ctx.bind(select(c, a.key, b.key)),
            part: ctx.bind(select(c, a.part, b.part)),
            head: ctx.bind(select(c, a.head, b.head)),
        }
    }

    fn store(self, ctx: &Ctx, to: Self) {
        ctx.store(to.key, self.key);
        ctx.store(to.part, self.part);
        ctx.store(to.head, self.head);
    }
}

/// A loser tree in heap order, in three stack arrays of `2k` slots: internal node `i` in
/// `1..k` holds the loser of its match, and leaf `j` (slot `k + j`) a part's first head
/// until the tree is built. `winner` is the root's winner.
struct Tree<K: StagedType, T: StagedType> {
    keys: Var<SMutPtr<K>>,
    parts: Var<SMutPtr<u32>>,
    heads: Var<SMutPtr<T>>,
    winner: Entry<K, T>,
}

impl<K, T> Tree<K, T>
where
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
    T: CopyType + 'static,
{
    fn load(&self, ctx: &Ctx, slot: usize) -> Entry<K, T> {
        let i = Const::new(slot as i64);
        // SAFETY: `slot < 2k`, within the arrays; each slot read was written before (a
        // leaf while priming, a node when the tree was built).
        unsafe {
            Entry {
                key: ctx.bind(load_mut(ptr_offset_mut(self.keys, i))),
                part: ctx.bind(load_mut(ptr_offset_mut(self.parts, i))),
                head: ctx.bind(load_mut(ptr_offset_mut(self.heads, i))),
            }
        }
    }

    fn save(&self, ctx: &Ctx, slot: usize, e: Entry<K, T>) {
        let i = Const::new(slot as i64);
        // SAFETY: `slot < 2k`, within the arrays.
        unsafe {
            ctx.emit(store(ptr_offset_mut(self.keys, i), e.key));
            ctx.emit(store(ptr_offset_mut(self.parts, i), e.part));
            ctx.emit(store(ptr_offset_mut(self.heads, i), e.head));
        }
    }
}

/// Where the parts' heads live: a head and key variable per part ([`Plan::Linear`]), or
/// a loser tree ([`Plan::Tree`]).
enum Heads<K: StagedType, T: StagedType> {
    Vars(Vec<(Var<T>, Var<K>)>),
    Tree(Tree<K, T>),
}

/// The cursor of a [`KMergeBy`].
pub struct KMergeCursor<C: Cursor, F, K: StagedType> {
    cursors: Vec<C>,
    key: F,
    /// The part whose head was yielded last, refilled at the next step. While
    /// `priming`, the part being filled for the first time.
    refill: Var<u32>,
    priming: Var<bool>,
    /// The key below which the last winner's part keeps winning; `None` with runs off.
    bound: Option<Var<K>>,
    heads: Heads<K, C::Item>,
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

    /// ```text
    /// got: {
    ///   again:
    ///     switch refill {
    ///       j: pulled: { x = part[j].next() (exhausted → key = MAX); head = x
    ///                    key = key(x); if key < bound → got(x) }
    ///          linear: head[j], key[j] = head, key
    ///          tree: if priming { leaf j = (head, key) } else { replay it from leaf j to
    ///                the root; if j still wins, bound = least loser on its path, else MIN }
    ///     }
    ///     if priming { if refill < k - 1 { refill += 1; → again }; priming = false
    ///                  tree: build it from the leaves }
    ///     linear: (w, m, m2) = argmin, min and runner-up of key[..]; bound = m2
    ///     tree: (w, m) = the root's winner
    ///     if m == MAX → done
    ///     refill = w; got(head of w)
    /// }
    /// ```
    /// `bound` starts at the key's minimum, so no head passes it before the first pass.
    fn next(self, ctx: &Ctx, done: Label<'_>) -> (Var<C::Item>, Self::Close) {
        let KMergeCursor {
            cursors,
            key,
            refill,
            priming,
            bound,
            heads,
        } = self;
        let k = cursors.len() as u32;
        if k == 0 {
            ctx.exit(done);
            return (ctx.bind(dead::<C::Item>()), Vec::new());
        }
        let mut cursors: Vec<_> = cursors.into_iter().map(Some).collect();
        let mut closes: Vec<Option<C::Close>> = (0..k).map(|_| None).collect();
        let sentinel = || Const::new(K::max_sentinel());
        let part = |ctx: &Ctx, j: u32| ctx.bind(Const::new(j));

        let item = ctx.join(|ctx, got| {
            ctx.repeat(|ctx, again| {
                ctx.switch(refill, k, |ctx, j| {
                    let (head, head_key) = match &heads {
                        Heads::Vars(vars) => vars[j as usize],
                        Heads::Tree(_) => (ctx.var(dead::<C::Item>()), ctx.var(sentinel())),
                    };
                    let cursor = cursors[j as usize]
                        .take()
                        .expect("each part is staged once");
                    ctx.block(|ctx, pulled| {
                        ctx.block(|ctx, exhausted| {
                            let (x, c) = cursor.next(ctx, exhausted);
                            closes[j as usize] = Some(c);
                            ctx.store(head, x);
                            ctx.store(head_key, key(ctx, x));
                            if let Some(bound) = bound {
                                ctx.if_then(lt(head_key, bound), |ctx| ctx.goto(got, x));
                            }
                            ctx.exit(pulled);
                        });
                        ctx.store(head_key, sentinel());
                    });
                    if let Heads::Tree(tree) = &heads {
                        let entry = Entry {
                            key: head_key,
                            part: part(ctx, j),
                            head,
                        };
                        let leaf = (k + j) as usize;
                        ctx.if_then_else(
                            priming,
                            |ctx| tree.save(ctx, leaf, entry),
                            |ctx| replay(ctx, tree, entry, leaf / 2, bound),
                        );
                    }
                });
                ctx.if_then(priming, |ctx| {
                    ctx.if_then(lt(refill, k - 1), |ctx| {
                        ctx.store(refill, add(refill, 1u32));
                        ctx.again(again);
                    });
                    ctx.store(priming, false);
                    if let Heads::Tree(tree) = &heads {
                        build(ctx, tree, k as usize).store(ctx, tree.winner);
                    }
                });

                let winner = match &heads {
                    Heads::Tree(tree) => tree.winner,
                    Heads::Vars(vars) => {
                        let entries: Vec<_> = (0..k)
                            .map(|j| {
                                let (head, key) = vars[j as usize];
                                Entry {
                                    key,
                                    part: part(ctx, j),
                                    head,
                                }
                            })
                            .collect();
                        linear(ctx, &entries, bound)
                    }
                };
                ctx.exit_if(eq(winner.key, sentinel()), done);
                ctx.store(refill, winner.part);
                ctx.goto(got, winner.head);
            });
            dead::<C::Item>()
        });
        let closes = closes
            .into_iter()
            .map(|c| c.expect("each part is staged once"))
            .collect();
        (item, closes)
    }
}

/// The first of `entries` (the lowest part on ties); `bound` becomes the runner-up key.
fn linear<K, T>(ctx: &Ctx, entries: &[Entry<K, T>], bound: Option<Var<K>>) -> Entry<K, T>
where
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
    T: CopyType + 'static,
{
    let mut best = entries[0];
    let mut runner_up: Var<K> = ctx.bind(Const::new(K::max_sentinel()));
    for &e in &entries[1..] {
        let better = ctx.bind(lt(e.key, best.key));
        if bound.is_some() {
            runner_up = ctx.bind(select(better, best.key, min(runner_up, e.key)));
        }
        best = Entry::pick(ctx, better, e, best);
    }
    if let Some(bound) = bound {
        ctx.store(bound, runner_up);
    }
    best
}

/// Play the matches of a fresh tree bottom-up from its `k` leaves, saving each loser;
/// returns the winner.
fn build<K, T>(ctx: &Ctx, tree: &Tree<K, T>, k: usize) -> Entry<K, T>
where
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
    T: CopyType + 'static,
{
    let mut win: Vec<Option<Entry<K, T>>> = vec![None; k];
    win.extend((k..2 * k).map(|slot| Some(tree.load(ctx, slot))));
    for i in (1..k).rev() {
        let (a, b) = (win[2 * i].expect("child"), win[2 * i + 1].expect("child"));
        let a_wins = ctx.bind(a.beats(b));
        win[i] = Some(Entry::pick(ctx, a_wins, a, b));
        tree.save(ctx, i, Entry::pick(ctx, a_wins, b, a));
    }
    win[1].expect("root")
}

/// Replay `entry` from `node` up to the root: at each node the loser stays and the
/// winner goes on. If `entry` wins, `bound` becomes the least loser key on its path;
/// otherwise the winner's path was not all replayed, and `bound` turns the shortcut off.
fn replay<K, T>(
    ctx: &Ctx,
    tree: &Tree<K, T>,
    mut entry: Entry<K, T>,
    mut node: usize,
    bound: Option<Var<K>>,
) where
    K: Ordered + MinMax + StagedType<RuntimeValue = K>,
    T: CopyType + 'static,
{
    let part = entry.part;
    let mut least: Var<K> = ctx.bind(Const::new(K::max_sentinel()));
    while node >= 1 {
        let held = tree.load(ctx, node);
        let held_wins = ctx.bind(held.beats(entry));
        let loser = Entry::pick(ctx, held_wins, entry, held);
        entry = Entry::pick(ctx, held_wins, held, entry);
        tree.save(ctx, node, loser);
        if bound.is_some() {
            least = ctx.bind(min(least, loser.key));
        }
        node /= 2;
    }
    entry.store(ctx, tree.winner);
    if let Some(bound) = bound {
        let kept = eq(entry.part, part);
        ctx.store(bound, select(kept, least, Const::new(K::min_sentinel())));
    }
}
