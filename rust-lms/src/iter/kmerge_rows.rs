//! k-way merge of sorted rows whose number is only known at run time:
//! [`kmerge_rows`].
//!
//! [`kmerge_by`](super::kmerge_by) fixes its parts at staging time, so its code grows
//! with them. Here the parts are rows, each a slice of keys sorted ascending and a
//! slice of values of the same length, pushed while the merge opens by a staged loop
//! of the caller's. The code is the same whatever the number of rows, and each
//! element costs O(log P).
//!
//! A row's cursor is its head key, pointers to the ends of its keys and values, and
//! how many elements it has left: a [`Row`]. The row with the smallest head, `top`, is held in
//! variables; the others are records in one stack array of a capacity fixed at
//! compile time, ordered as a binary heap on their head keys. Each element comes
//! from `top`, which then advances, and the heap is touched only when its smallest
//! head drops below `top`'s or `top` runs dry. So a row that runs ahead, and a merge
//! of one row, never sift. Sifts move a hole, each row written once. With two rows
//! `top` and the root are ordered by selects instead, as their heads interleave
//! unpredictably more often than not.
//!
//! Rows that are empty when pushed are never stored.

use std::marker::PhantomData;

use crate::ffi::stack_array;
use crate::func::Ctx;
use crate::label::{Label, dead};
use crate::num::{Ordered, add, eq, ge, int_cast, le, lt, select, shl, shr, sub};
use crate::refer::{
    SMutPtr, SPtr, SRef, load, load_mut, ptr_cast_mut, ptr_offset, ptr_offset_mut, store,
};
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

/// The cursors of a [`kmerge_rows`]: `top`, the smallest head, and a binary heap of
/// the others over the first `len` records. `top.left == 0` once every row is dry.
pub struct Heap<K: StagedType, V: StagedType> {
    records: Records<K, V>,
    len: Var<u64>,
    top: Row<K, V>,
}

/// A row's cursor: its head key, the ends of its keys and values, and how many
/// elements it has left, the head `left` elements before the ends. As variables
/// (`top`), or as values bound from them or from a record. Advancing is a
/// decrement; the ends never change.
struct Row<K: StagedType, V: StagedType> {
    key: Var<K>,
    key_end: Var<SPtr<K>>,
    value_end: Var<SPtr<V>>,
    left: Var<u64>,
}

impl<K, V> Row<K, V>
where
    K: Ordered + StagedType + CopyType + 'static,
    V: StagedType + CopyType + 'static,
{
    fn vars(ctx: &Ctx) -> Self {
        Self {
            key: ctx.var(dead::<K>()),
            key_end: ctx.var(dead::<SPtr<K>>()),
            value_end: ctx.var(dead::<SPtr<V>>()),
            left: ctx.var(0u64),
        }
    }

    /// The head's address in the row ending at `end`: `left` elements before it.
    fn head<T: StagedType + 'static>(
        &self,
        end: Var<SPtr<T>>,
    ) -> impl Staged<Out = SPtr<T>> + 'static {
        let back = sub(Const::new(0i64), int_cast::<i64, u64, _>(self.left));
        // SAFETY: `left` is at most the row's length.
        unsafe { ptr_offset(end, back) }
    }

    fn bind(&self, ctx: &Ctx) -> Self {
        Self {
            key: ctx.bind(self.key),
            key_end: ctx.bind(self.key_end),
            value_end: ctx.bind(self.value_end),
            left: ctx.bind(self.left),
        }
    }

    /// `if_true` if `cond`, else `if_false`, field by field.
    fn select(ctx: &Ctx, cond: Var<bool>, if_true: &Self, if_false: &Self) -> Self {
        Self {
            key: ctx.bind(select(cond, if_true.key, if_false.key)),
            key_end: ctx.bind(select(cond, if_true.key_end, if_false.key_end)),
            value_end: ctx.bind(select(cond, if_true.value_end, if_false.value_end)),
            left: ctx.bind(select(cond, if_true.left, if_false.left)),
        }
    }

    fn assign(&self, ctx: &Ctx, from: &Self) {
        ctx.store(self.key, from.key);
        ctx.store(self.key_end, from.key_end);
        ctx.store(self.value_end, from.value_end);
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
            let key_ptr = ctx.bind(keys.into_ptr());
            let len = int_cast::<i64, u64, _>(n);
            // SAFETY: the ends are one past the last element of each slice.
            let row = Row {
                // SAFETY: the row is non-empty, so its first key is in it.
                key: ctx.bind(unsafe { load(key_ptr) }),
                key_end: ctx.bind(unsafe { ptr_offset(key_ptr, len) }),
                value_end: ctx.bind(unsafe { ptr_offset(values.into_ptr(), len) }),
                left: n,
            };
            self.sift_up(ctx, &row);
        });
    }

    /// Store `row` at the end of the heap and restore it: parents above `row`'s key
    /// move down into the hole until `row` fits.
    fn sift_up(&self, ctx: &Ctx, row: &Row<K, V>) {
        let i = ctx.var(self.len);
        ctx.store(self.len, add(self.len, 1u64));
        ctx.while_loop(lt(0u64, i), |ctx| {
            let parent = ctx.bind(shr(sub(i, 1u64), 1u64));
            ctx.if_then(le(self.records.key(ctx, parent), row.key), |ctx| {
                ctx.break_loop()
            });
            self.records.copy(ctx, parent, i);
            ctx.store(i, parent);
        });
        self.records.set(ctx, i, row);
    }

    /// Store `row` at the root of the heap and restore it: the smaller child moves up
    /// into the hole while its key is below `row`'s.
    fn sift_down(&self, ctx: &Ctx, row: &Row<K, V>) {
        let i = ctx.var(0u64);
        ctx.while_loop(true, |ctx| {
            let left = ctx.bind(add(shl(i, 1u64), 1u64));
            ctx.if_then(ge(left, self.len), |ctx| ctx.break_loop());
            let right = ctx.bind(add(left, 1u64));
            let child = ctx.var(left);
            let child_key = ctx.var(self.records.key(ctx, left));
            ctx.if_then(lt(right, self.len), |ctx| {
                let right_key = self.records.key(ctx, right);
                ctx.if_then(lt(right_key, child_key), |ctx| {
                    ctx.store(child, right);
                    ctx.store(child_key, right_key);
                });
            });
            ctx.if_then(le(row.key, child_key), |ctx| ctx.break_loop());
            self.records.copy(ctx, child, i);
            ctx.store(i, child);
        });
        self.records.set(ctx, i, row);
    }

    /// `top` replaced by the heap's smallest row, the heap's last row sifted into the
    /// root's place; or marked dry if the heap is empty.
    fn pop(&self, ctx: &Ctx) {
        ctx.if_then_else(
            eq(self.len, 0u64),
            |ctx| ctx.store(self.top.left, 0u64),
            |ctx| {
                let zero = ctx.var(0u64);
                self.top.assign(ctx, &self.records.get(ctx, zero));
                ctx.store(self.len, sub(self.len, 1u64));
                // With `len` now 0 the last row is the root, stored back unread.
                self.sift_down(ctx, &self.records.get(ctx, self.len));
            },
        );
    }

    /// `top` advanced past its head, then swapped with the heap's root if the root's
    /// head is below it; or, when dry, replaced by the root.
    fn advance(&self, ctx: &Ctx) {
        let top = &self.top;
        let left = ctx.bind(sub(top.left, 1u64));
        ctx.if_then_else(
            eq(left, 0u64),
            |ctx| self.pop(ctx),
            |ctx| {
                ctx.store(top.left, left);
                // SAFETY: the row has `left > 0` elements after its head.
                ctx.store(top.key, unsafe { load(top.head(top.key_end)) });
                let zero = ctx.var(0u64);
                ctx.if_then_else(
                    eq(self.len, 1u64),
                    |ctx| {
                        // Two rows, `top` and the root: ordered by selects, as their
                        // heads interleave unpredictably more often than not.
                        let (row, root) = (top.bind(ctx), self.records.get(ctx, zero));
                        let swap = ctx.bind(lt(root.key, row.key));
                        top.assign(ctx, &Row::select(ctx, swap, &root, &row));
                        self.records
                            .set(ctx, zero, &Row::select(ctx, swap, &row, &root));
                    },
                    |ctx| {
                        // A branch: predicted whenever a row runs ahead.
                        ctx.if_then(lt(0u64, self.len), |ctx| {
                            ctx.if_then(lt(self.records.key(ctx, zero), top.key), |ctx| {
                                let old = top.bind(ctx);
                                top.assign(ctx, &self.records.get(ctx, zero));
                                self.sift_down(ctx, &old);
                            });
                        });
                    },
                );
            },
        );
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
            records: Records::new(ctx, self.capacity),
            len: ctx.var(0u64),
            top: Row::vars(ctx),
        };
        (self.fill)(ctx, &heap);
        heap.pop(ctx);
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
        ctx.exit_if(eq(self.top.left, 0u64), done);
        let key = ctx.bind(self.top.key);
        // SAFETY: `top`'s row has an element left, at these pointers.
        let value = ctx.bind(unsafe { load(self.top.head(self.top.value_end)) });
        self.advance(ctx);
        (ctx.bind(Pair::new(key, value)), ())
    }
}

/// The heap's rows: a stack array of `capacity` records of four words, a [`Row`]'s
/// fields. Indices are the heap's, always below the capacity.
struct Records<K: StagedType, V: StagedType> {
    base: Var<SMutPtr<u64>>,
    _kv: PhantomData<fn() -> (K, V)>,
}

// `left` between the two ends: no two neighbouring words share a type, so LLVM's SLP
// vectoriser does not pair the ends into a vector register, which then costs a
// move back to an integer register for each element read.
const KEY: u64 = 0;
const KEY_END: u64 = 1;
const LEFT: u64 = 2;
const VALUE_END: u64 = 3;

impl<K, V> Records<K, V>
where
    K: Ordered + StagedType + CopyType + 'static,
    V: StagedType + CopyType + 'static,
{
    fn new(ctx: &Ctx, capacity: usize) -> Self {
        assert!(
            K::size_of() <= 8 && K::align_of() <= 8,
            "kmerge_rows: a key must fit a word"
        );
        Self {
            base: ctx.bind(stack_array::<u64>(4 * capacity)),
            _kv: PhantomData,
        }
    }

    /// Field `field` of record `i`, as a pointer to `T`.
    fn field<T: StagedType + 'static>(
        &self,
        i: Var<u64>,
        field: u64,
    ) -> impl Staged<Out = SMutPtr<T>> + 'static {
        let word = int_cast::<i64, u64, _>(add(shl(i, 2u64), field));
        // SAFETY: `i` is below the capacity (see the type); every field fits a word,
        // whose alignment is 8.
        ptr_cast_mut::<T, u64, _>(unsafe { ptr_offset_mut(self.base, word) })
    }

    fn key(&self, ctx: &Ctx, i: Var<u64>) -> Var<K> {
        // SAFETY: records below the heap's length were stored.
        ctx.bind(unsafe { load_mut(self.field::<K>(i, KEY)) })
    }

    fn get(&self, ctx: &Ctx, i: Var<u64>) -> Row<K, V> {
        // SAFETY: records below the heap's length were stored.
        unsafe {
            Row {
                key: self.key(ctx, i),
                key_end: ctx.bind(load_mut(self.field::<SPtr<K>>(i, KEY_END))),
                value_end: ctx.bind(load_mut(self.field::<SPtr<V>>(i, VALUE_END))),
                left: ctx.bind(load_mut(self.field::<u64>(i, LEFT))),
            }
        }
    }

    fn set(&self, ctx: &Ctx, i: Var<u64>, row: &Row<K, V>) {
        // SAFETY: `i` is within the array.
        unsafe {
            ctx.emit(store(self.field::<K>(i, KEY), row.key));
            ctx.emit(store(self.field::<SPtr<K>>(i, KEY_END), row.key_end));
            ctx.emit(store(self.field::<SPtr<V>>(i, VALUE_END), row.value_end));
            ctx.emit(store(self.field::<u64>(i, LEFT), row.left));
        }
    }

    fn copy(&self, ctx: &Ctx, from: Var<u64>, to: Var<u64>) {
        self.set(ctx, to, &self.get(ctx, from));
    }
}
