//! k-way merge of sorted rows whose number is only known at run time:
//! [`kmerge_rows`].
//!
//! [`kmerge_by`](super::kmerge_by) fixes its parts at staging time, so its code grows
//! with them. Here the parts are rows, each a slice of keys sorted ascending and a
//! slice of values of the same length, pushed while the merge opens by a staged loop
//! of the caller's. Their cursors live in stack arrays of a capacity fixed at
//! compile time and are ordered as a binary heap on their head keys, so the code is
//! the same whatever the number of rows, and each element costs O(log P).
//!
//! A row's cursor is its head key, pointers to its head key and value, and how many
//! elements it has left. Rows that are empty when pushed are never stored.
//!
//! While at most two rows are live (from the start, or once the others have run
//! dry), their cursors move to variables, [`Head`]s, and each element is a
//! two-way step instead of a pop and a sift through the stack arrays.
//! A merge of one or two rows, the common case of a node over few layers, never
//! touches the heap after it opens.

use std::marker::PhantomData;

use crate::control::not;
use crate::ffi::stack_array;
use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::num::{Ordered, add, eq, ge, int_cast, le, lt, shl, shr, sub};
use crate::refer::{SMutPtr, SPtr, SRef, load, load_mut, ptr_offset, ptr_offset_mut, store};
use crate::slice::{Slice, SliceOps};
use crate::staged::{Const, Staged, Var};
use crate::types::{CopyType, StagedType};

use super::traits::{Cursor, StagedIterator};
use super::zip::{Pair, ZipItem};

/// The rows `fill` pushes, merged into one stream of `(key, value)` pairs sorted by
/// key. Not stable: equal keys come in any order.
///
/// `fill` is staged once, when the merge opens, and pushes rows with
/// [`Heap::push`], typically from loops of its own.
///
/// # Safety
///
/// At execution, `fill` pushes at most `capacity` non-empty rows; each row's keys are
/// sorted ascending, its values are as many as its keys, and both stay live and
/// unchanged until the merge ends.
pub unsafe fn kmerge_rows<K, V, F>(capacity: usize, fill: F) -> KMergeRows<K, V, F>
where
    F: FnOnce(&Ctx, &Heap<K, V>),
{
    KMergeRows {
        capacity,
        fill,
        _kv: PhantomData,
    }
}

/// Built by [`kmerge_rows`].
pub struct KMergeRows<K, V, F> {
    capacity: usize,
    fill: F,
    _kv: PhantomData<fn() -> (K, V)>,
}

/// The cursors of a [`kmerge_rows`]: parallel stack arrays, a binary heap on `keys`
/// over their first `len` entries; or, once `small`, the `len <= 2` cursors in `a`
/// and `b`, `a` the smaller head.
pub struct Heap<K: StagedType, V: StagedType> {
    keys: Array<K>,
    key_ptrs: Array<SPtr<K>>,
    value_ptrs: Array<SPtr<V>>,
    left: Array<u64>,
    len: Var<u64>,
    small: Var<bool>,
    a: Head<K, V>,
    b: Head<K, V>,
}

/// A row's cursor held in variables.
struct Head<K: StagedType, V: StagedType> {
    key: Var<K>,
    key_ptr: Var<SPtr<K>>,
    value_ptr: Var<SPtr<V>>,
    left: Var<u64>,
}

/// A [`Head`]'s fields, copied.
struct Snapshot<K: StagedType, V: StagedType> {
    key: Var<K>,
    key_ptr: Var<SPtr<K>>,
    value_ptr: Var<SPtr<V>>,
    left: Var<u64>,
}

impl<K, V> Head<K, V>
where
    K: Ordered + StagedType + CopyType + 'static,
    V: StagedType + CopyType + 'static,
{
    fn new(ctx: &Ctx) -> Self {
        Self {
            key: ctx.var(dead::<K>()),
            key_ptr: ctx.var(dead::<SPtr<K>>()),
            value_ptr: ctx.var(dead::<SPtr<V>>()),
            left: ctx.var(dead::<u64>()),
        }
    }

    /// Copy the stack arrays' entry `i` here.
    fn load(&self, ctx: &Ctx, heap: &Heap<K, V>, i: Var<u64>) {
        ctx.store(self.key, heap.keys.get(ctx, i));
        ctx.store(self.key_ptr, heap.key_ptrs.get(ctx, i));
        ctx.store(self.value_ptr, heap.value_ptrs.get(ctx, i));
        ctx.store(self.left, heap.left.get(ctx, i));
    }

    fn snapshot(&self, ctx: &Ctx) -> Snapshot<K, V> {
        Snapshot {
            key: ctx.bind(self.key),
            key_ptr: ctx.bind(self.key_ptr),
            value_ptr: ctx.bind(self.value_ptr),
            left: ctx.bind(self.left),
        }
    }

    fn assign(&self, ctx: &Ctx, from: &Snapshot<K, V>) {
        ctx.store(self.key, from.key);
        ctx.store(self.key_ptr, from.key_ptr);
        ctx.store(self.value_ptr, from.value_ptr);
        ctx.store(self.left, from.left);
    }
}

impl<K, V> Heap<K, V>
where
    K: Ordered + StagedType + CopyType + 'static,
    V: StagedType + CopyType + 'static,
{
    /// Add a row: `keys` sorted ascending, and as many `values`. An empty row is
    /// skipped.
    pub fn push(&self, ctx: &Ctx, keys: Var<SRef<Slice<K>>>, values: Var<SRef<Slice<V>>>) {
        let n = ctx.bind(keys.len());
        ctx.if_then(lt(0u64, n), |ctx| {
            let at = ctx.bind(self.len);
            let key_ptr = ctx.bind(keys.into_ptr());
            // SAFETY: the row is non-empty, so its first key is in it.
            self.keys.set(ctx, at, unsafe { load(key_ptr) });
            self.key_ptrs.set(ctx, at, key_ptr);
            self.value_ptrs.set(ctx, at, values.into_ptr());
            self.left.set(ctx, at, n);
            ctx.store(self.len, add(self.len, 1u64));
            self.sift_up(ctx, at);
        });
    }

    /// Move entry `i` up while its key is below its parent's.
    fn sift_up(&self, ctx: &Ctx, i: Var<u64>) {
        let i = ctx.var(i);
        ctx.while_loop(lt(0u64, i), |ctx| {
            let parent = ctx.bind(shr(sub(i, 1u64), 1u64));
            let (key, parent_key) = (self.keys.get(ctx, i), self.keys.get(ctx, parent));
            ctx.if_then(le(parent_key, key), |ctx| ctx.break_loop());
            self.swap(ctx, i, parent);
            ctx.store(i, parent);
        });
    }

    /// Move entry 0 down while a child's key is below its own.
    fn sift_down(&self, ctx: &Ctx) {
        let i = ctx.var(0u64);
        ctx.while_loop(true, |ctx| {
            let left = ctx.bind(add(shl(i, 1u64), 1u64));
            ctx.if_then(ge(left, self.len), |ctx| ctx.break_loop());
            let right = ctx.bind(add(left, 1u64));
            let child = ctx.var(left);
            ctx.if_then(lt(right, self.len), |ctx| {
                let (left_key, right_key) = (self.keys.get(ctx, left), self.keys.get(ctx, right));
                ctx.if_then(lt(right_key, left_key), |ctx| ctx.store(child, right));
            });
            let (key, child_key) = (self.keys.get(ctx, i), self.keys.get(ctx, child));
            ctx.if_then(le(key, child_key), |ctx| ctx.break_loop());
            self.swap(ctx, i, child);
            ctx.store(i, child);
        });
    }

    fn swap(&self, ctx: &Ctx, i: Var<u64>, j: Var<u64>) {
        self.keys.swap(ctx, i, j);
        self.key_ptrs.swap(ctx, i, j);
        self.value_ptrs.swap(ctx, i, j);
        self.left.swap(ctx, i, j);
    }

    /// With `len <= 2`, move the live cursors from the arrays to `a` and `b`, in
    /// order, and merge from those on.
    fn enter_small(&self, ctx: &Ctx) {
        ctx.store(self.small, true);
        self.a.load(ctx, self, ctx.var(0u64));
        ctx.if_then(eq(self.len, 2u64), |ctx| {
            self.b.load(ctx, self, ctx.var(1u64));
            self.order(ctx);
        });
    }

    /// Swap `a` and `b` if `b`'s head is smaller. Only with `len == 2`. A branch, not
    /// selects: a select makes each element wait on the last one's load, while the
    /// branch is predicted whenever one row runs ahead of the other.
    fn order(&self, ctx: &Ctx) {
        ctx.if_then(lt(self.b.key, self.a.key), |ctx| {
            let (a, b) = (self.a.snapshot(ctx), self.b.snapshot(ctx));
            self.a.assign(ctx, &b);
            self.b.assign(ctx, &a);
        });
    }

    /// `a`'s head into `key` and `value`, then `a` advanced, or replaced by `b`
    /// when dry.
    fn next_small(&self, ctx: &Ctx, key: Var<K>, value: Var<V>) {
        let a = &self.a;
        ctx.store(key, a.key);
        // SAFETY: `a`'s row has an element left, at these pointers.
        ctx.store(value, unsafe { load(a.value_ptr) });
        let left = ctx.bind(sub(a.left, 1u64));
        ctx.if_then_else(
            eq(left, 0u64),
            |ctx| {
                ctx.store(self.len, sub(self.len, 1u64));
                a.assign(ctx, &self.b.snapshot(ctx));
            },
            |ctx| {
                // SAFETY: the row has `left > 0` elements after these.
                let (key_ptr, value_ptr) = unsafe {
                    (
                        ctx.bind(ptr_offset(a.key_ptr, Const::new(1i64))),
                        ctx.bind(ptr_offset(a.value_ptr, Const::new(1i64))),
                    )
                };
                ctx.store(a.key, unsafe { load(key_ptr) });
                ctx.store(a.key_ptr, key_ptr);
                ctx.store(a.value_ptr, value_ptr);
                ctx.store(a.left, left);
                ctx.if_then(eq(self.len, 2u64), |ctx| self.order(ctx));
            },
        );
    }

    /// The smallest head off the heap into `key` and `value`, its row advanced (or
    /// dropped when dry) and the heap restored; or, once two rows or fewer are left,
    /// moved to `a` and `b`.
    fn next_heap(&self, ctx: &Ctx, key: Var<K>, value: Var<V>) {
        let zero = ctx.var(0u64);
        let key_ptr = self.key_ptrs.get(ctx, zero);
        let value_ptr = self.value_ptrs.get(ctx, zero);
        ctx.store(key, self.keys.get(ctx, zero));
        // SAFETY: entry 0's row has an element left, at these pointers.
        ctx.store(value, unsafe { load(value_ptr) });
        let left = ctx.bind(sub(self.left.get(ctx, zero), 1u64));
        ctx.if_then_else(
            eq(left, 0u64),
            |ctx| {
                let last = ctx.bind(sub(self.len, 1u64));
                ctx.store(self.len, last);
                self.move_entry(ctx, last, zero);
                ctx.if_then(le(self.len, 2u64), |ctx| self.enter_small(ctx));
            },
            |ctx| {
                // SAFETY: the row has `left > 0` elements after these.
                let (key_ptr, value_ptr) = unsafe {
                    (
                        ctx.bind(ptr_offset(key_ptr, Const::new(1i64))),
                        ctx.bind(ptr_offset(value_ptr, Const::new(1i64))),
                    )
                };
                self.keys.set(ctx, zero, unsafe { load(key_ptr) });
                self.key_ptrs.set(ctx, zero, key_ptr);
                self.value_ptrs.set(ctx, zero, value_ptr);
                self.left.set(ctx, zero, left);
            },
        );
        ctx.if_then(not(self.small), |ctx| self.sift_down(ctx));
    }
}

impl<K, V, F> StagedIterator for KMergeRows<K, V, F>
where
    K: Ordered + StagedType + CopyType + 'static,
    V: StagedType + CopyType + 'static,
    F: FnOnce(&Ctx, &Heap<K, V>),
{
    type Item = ZipItem<K, V>;
    type Cursor = Heap<K, V>;

    fn open(self, ctx: &Ctx) -> Heap<K, V> {
        let heap = Heap {
            keys: Array::new(ctx, self.capacity),
            key_ptrs: Array::new(ctx, self.capacity),
            value_ptrs: Array::new(ctx, self.capacity),
            left: Array::new(ctx, self.capacity),
            len: ctx.var(0u64),
            small: ctx.var(false),
            a: Head::new(ctx),
            b: Head::new(ctx),
        };
        (self.fill)(ctx, &heap);
        ctx.if_then(le(heap.len, 2u64), |ctx| heap.enter_small(ctx));
        heap
    }
}

impl<K, V> Cursor for Heap<K, V>
where
    K: Ordered + StagedType + CopyType + 'static,
    V: StagedType + CopyType + 'static,
{
    type Item = ZipItem<K, V>;
    type Close = ();

    fn next(self, ctx: &Ctx, done: Label<'_>) -> (Var<ZipItem<K, V>>, ()) {
        ctx.exit_if(eq(self.len, 0u64), done);
        let (key, value) = (ctx.var(dead::<K>()), ctx.var(dead::<V>()));
        ctx.if_then_else(
            self.small,
            |ctx| self.next_small(ctx, key, value),
            |ctx| self.next_heap(ctx, key, value),
        );
        (ctx.bind(Pair::new(key, value)), ())
    }
}

impl<K, V> Heap<K, V>
where
    K: Ordered + StagedType + CopyType + 'static,
    V: StagedType + CopyType + 'static,
{
    fn move_entry(&self, ctx: &Ctx, from: Var<u64>, to: Var<u64>) {
        self.keys.set(ctx, to, self.keys.get(ctx, from));
        self.key_ptrs.set(ctx, to, self.key_ptrs.get(ctx, from));
        self.value_ptrs.set(ctx, to, self.value_ptrs.get(ctx, from));
        self.left.set(ctx, to, self.left.get(ctx, from));
    }
}

/// A stack array of `capacity` values of `T`, indexed by staged `u64`s. Indices are
/// the heap's, always below its length, which is at most the capacity.
struct Array<T: StagedType> {
    base: Var<SMutPtr<T>>,
}

impl<T: StagedType + CopyType + 'static> Array<T> {
    fn new(ctx: &Ctx, capacity: usize) -> Self {
        Self {
            base: ctx.bind(stack_array::<T>(capacity)),
        }
    }

    fn at(&self, i: Var<u64>) -> impl Staged<Out = SMutPtr<T>> + 'static {
        // SAFETY: `i` is below the capacity (see the type).
        unsafe { ptr_offset_mut(self.base, int_cast::<i64, u64, _>(i)) }
    }

    fn get(&self, ctx: &Ctx, i: Var<u64>) -> Var<T> {
        // SAFETY: entries below the heap's length were stored.
        ctx.bind(unsafe { load_mut(self.at(i)) })
    }

    fn set(&self, ctx: &Ctx, i: Var<u64>, value: impl Staged<Out = T> + 'static) {
        // SAFETY: `i` is within the array.
        ctx.emit(unsafe { store(self.at(i), value) });
    }

    fn swap(&self, ctx: &Ctx, i: Var<u64>, j: Var<u64>) {
        let (a, b) = (self.get(ctx, i), self.get(ctx, j));
        self.set(ctx, i, b);
        self.set(ctx, j, a);
    }
}
