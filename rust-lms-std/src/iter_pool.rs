//! [`IterPool`] — chunked, pooled external iterators. A Rust iterator is driven
//! from staged code a **chunk** at a time: one `refill` extern fills a reused
//! buffer with up to `chunk` items, and the kernel reads them back as plain typed
//! loads. The per-element FFI round-trip of [`OpaqueIter`](rust_lms::prelude::OpaqueIter)
//! becomes one call per chunk.
//!
//! # The pool is resolved at stage 0
//!
//! A staged `for_each` builds its consumer immediately, so staging walks the
//! iterator nesting in exactly the shape the loops will run. [`StagedIterPool`]
//! exploits that with a stage-0 stack: each pooled source takes the next slot
//! index when its `for_each` begins and gives it back when it returns.
//!
//! ```text
//! nodes(&pool).for_each(ctx, |ctx, n| {            // slot 0
//!     neighbours(&pool, n).for_each(ctx, |ctx, b| { // slot 1
//!         ..
//!     });                                          // release 1
//!     edges(&pool, n).for_each(ctx, |ctx, e| {      // slot 1 again: siblings share
//!         ..
//!     });
//! });                                              // release 0
//! ```
//!
//! Nested loops get distinct slots and sibling loops reuse one. The slot index
//! is a constant in the emitted code, so there is no runtime pool bookkeeping.
//! Each slot keeps its iterator storage and its chunk buffer across uses and
//! across kernel calls: after warm-up, nested traversal allocates nothing.
//!
//! # The loop is flattened
//!
//! Every pooled source emits a **single** loop with a cold refill branch, never
//! an outer "refill" loop around an inner "chunk" loop:
//!
//! ```text
//! slot = iter_pool_slot(pool, K)       ; K fixed at stage 0
//! init(args.., slot)                   ; producer emplaces its iterator
//! i = n = 0
//! loop:
//!   if i == n { (data, n) = refill(slot, CHUNK); i = 0; if n == 0 break }
//!   elem = data[i]; i += 1
//!   <consumer>
//! exit:
//!   drop(slot)                         ; on exhaustion and on break_loop
//! ```
//!
//! The combinators rely on this: `any`/`find_map`/`take_while` short-circuit with
//! `break_loop`, which targets the innermost loop. With nested loops the break
//! would leave only the chunk loop and iteration would silently resume on the
//! next chunk.
//!
//! # Items are by value
//!
//! Items are [`OpaqueIterItem`]s: `Copy` values delivered in registers. Nothing
//! handed to the consumer points into the chunk buffer, so nothing can dangle
//! when the buffer is refilled or reused by a sibling loop — write results out
//! by pushing them into an output buffer.
//!
//! # Ownership
//!
//! The host owns the [`IterPool`] and passes it to each kernel call as
//! `&mut IterPool` (staged `SRefMut<Opaque<IterPool>>`). One pool per thread:
//! the `&mut` is what keeps concurrent calls from sharing buffers. Kernels must
//! not recurse (`fun_rec`) while a pooled loop is live, because a recursive call
//! would reuse the caller's slot indices.
//!
//! # Where the unsafe is
//!
//! The host side is safe Rust: the iterator is a trait object held inline by
//! [`SmallBox`], each slot's chunk buffers are `Vec<R>`s found by type, and the
//! externs are ordinary `#[extern_fn]`s whose ABI thunks the derive generates.
//! What remains cannot be expressed safely, and each point is marked `unsafe`:
//!
//! 1. [`ChunkSlot::emplace_borrowed`] — storing an iterator that borrows host data
//!    in a slot that outlives the borrow. The slot lifetime cannot be named
//!    across `extern "C"`; the caller promises the data outlives the kernel call.
//! 2. The slot lookup in [`PooledIter`] — `iter_pool_slot` returns a
//!    `&mut PoolSlot` whose exclusivity the staged result cannot carry (the
//!    derive leaves reference-returning externs unchecked). Exclusive because
//!    stage 0 gives each live traversal its own slot index.
//! 3. The producer call in [`PooledIterFns::iter1`]/[`iter2`](PooledIterFns::iter2)
//!    — handing the slot to `init` as `&mut ChunkSlot<R>`, a typed view of the
//!    `PoolSlot` the pool returned. Sound because `ChunkSlot` is
//!    `repr(transparent)` over `PoolSlot`.
//! 4. The element load in [`PooledIter`]'s loop — reading `data[i]` for `i < n`,
//!    where `(data, n)` is the chunk the last refill returned. The chunk pointer
//!    is relabelled from `*const u8` to `*const R` (like [`SVec`](crate::SVec)'s
//!    buffer); `R` is fixed by the `ChunkSlot<R>` the producer filled.

use std::any::Any;
use std::cell::{Cell, RefCell};
use std::marker::PhantomData;
use std::ptr::NonNull;

use rust_lms::prelude::*;
use smallbox::space::S32;
use smallbox::{SmallBox, smallbox};

// =============================================================================
// Host side: the pool and its slots (safe Rust)
// =============================================================================

/// What a slot's iterator can do once its concrete type is erased.
trait ChunkSource {
    /// Refill this iterator's chunk buffer in `bufs` with up to `chunk` items.
    fn refill(&mut self, bufs: &mut ChunkBufs, chunk: usize) -> Chunk;
}

/// An iterator plus its end-of-stream latch.
struct Filler<I> {
    it: I,
    /// Set once a refill comes back short, so a non-fused iterator is never
    /// polled again after reporting its end.
    exhausted: bool,
}

impl<I> ChunkSource for Filler<I>
where
    I: Iterator,
    I::Item: Copy + 'static,
{
    fn refill(&mut self, bufs: &mut ChunkBufs, chunk: usize) -> Chunk {
        let buf = bufs.get::<I::Item>();
        buf.clear();
        if !self.exhausted {
            // Within capacity after the first chunk, so the buffer is reused.
            buf.extend(self.it.by_ref().take(chunk));
            self.exhausted = buf.len() < chunk;
        }
        Chunk::of(buf)
    }
}

/// A slot's chunk buffers, one `Vec<R>` per item type the slot has served.
/// Sibling loops of different item types can share a slot, so each keeps its
/// own typed buffer (and capacity) instead of reinterpreting bytes.
#[derive(Default)]
struct ChunkBufs(Vec<Box<dyn Any>>);

impl ChunkBufs {
    fn get<R: 'static>(&mut self) -> &mut Vec<R> {
        let i = match self.0.iter().position(|b| b.is::<Vec<R>>()) {
            Some(i) => i,
            None => {
                self.0.push(Box::new(Vec::<R>::new()));
                self.0.len() - 1
            }
        };
        self.0[i]
            .downcast_mut()
            .expect("the buffer at `i` was found or pushed as a `Vec<R>`")
    }
}

/// A chunk as the kernel sees it: the items the last refill wrote. The pointer
/// is untyped here because the refill extern serves every item type; staged code
/// relabels it to the slot's item type.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct Chunk {
    #[staged(SPtr<u8>)]
    ptr: *const u8,
    #[staged(u64)]
    len: usize,
}

impl Chunk {
    fn of<R>(buf: &[R]) -> Self {
        Chunk {
            ptr: buf.as_ptr().cast(),
            len: buf.len(),
        }
    }

    const EMPTY: Chunk = Chunk {
        ptr: NonNull::dangling().as_ptr(),
        len: 0,
    };
}

/// A type-erased iterator, stored inline when it fits 256 bytes.
type Source = SmallBox<dyn ChunkSource, S32>;

/// One reusable slot: the current iterator plus the chunk buffers. The pool
/// boxes each slot, so its address is stable while the kernel holds it.
#[derive(Default)]
pub struct PoolSlot {
    source: Option<Source>,
    bufs: ChunkBufs,
}

impl PoolSlot {
    fn refill(&mut self, chunk: usize) -> Chunk {
        match &mut self.source {
            Some(source) => source.refill(&mut self.bufs, chunk),
            None => Chunk::EMPTY,
        }
    }
}

/// Host owner of a kernel's pooled-iterator slots. Create one per thread, keep
/// it across calls, and pass it to each call as `&mut IterPool`.
///
/// Slots are created lazily the first time a kernel reaches a nesting depth, and
/// keep their buffers (capacity) until the pool is dropped.
#[derive(Default)]
pub struct IterPool {
    /// Boxed so a slot stays put while the list grows: the kernel holds slot `k`
    /// while it asks for slot `k + 1`.
    #[allow(clippy::vec_box)]
    slots: Vec<Box<PoolSlot>>,
}

impl IterPool {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of slots created so far (the deepest nesting any kernel reached).
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    fn slot(&mut self, k: usize) -> &mut PoolSlot {
        if self.slots.len() <= k {
            self.slots.resize_with(k + 1, Box::default);
        }
        &mut self.slots[k]
    }
}

/// A pool slot typed by the runtime item type `R` its iterator yields. This is
/// what a producer's `init` extern receives (`slot: &mut ChunkSlot<R>`), so the
/// item type the staged kind declares is the item type the producer can emplace.
#[repr(transparent)]
pub struct ChunkSlot<R> {
    slot: PoolSlot,
    _item: PhantomData<fn() -> R>,
}

impl<R: Copy + 'static> ChunkSlot<R> {
    /// Install `it` as this slot's iterator, replacing any earlier one. The
    /// chunk buffers are kept.
    ///
    /// The slot's type fixes the item type, so a producer cannot emplace an
    /// iterator the kernel would read as something else:
    ///
    /// ```compile_fail
    /// # use rust_lms_std::ChunkSlot;
    /// fn producer(slot: &mut ChunkSlot<u64>) {
    ///     slot.emplace([1.5f64, 2.5].into_iter()); // error: expected `u64`
    /// }
    /// ```
    ///
    /// And an iterator that borrows needs [`emplace_borrowed`](Self::emplace_borrowed):
    ///
    /// ```compile_fail
    /// # use rust_lms_std::ChunkSlot;
    /// fn producer(data: &[u64], slot: &mut ChunkSlot<u64>) {
    ///     slot.emplace(data.iter().copied()); // error: `data` must be 'static
    /// }
    /// ```
    pub fn emplace<I: Iterator<Item = R> + 'static>(&mut self, it: I) {
        let source: Source = smallbox!(Filler {
            it,
            exhausted: false
        });
        self.slot.source = Some(source);
    }

    /// [`emplace`](Self::emplace) for an iterator that borrows host data — the
    /// common case, e.g. neighbours read out of a graph passed to the producer.
    ///
    /// # Safety
    ///
    /// Everything `it` borrows must outlive the kernel call that runs this
    /// traversal. Generated code drops the iterator when its loop exits, which
    /// always happens before the call returns, but the slot itself outlives the
    /// call, and the borrow cannot be named across `extern "C"`.
    pub unsafe fn emplace_borrowed<'a, I: Iterator<Item = R> + 'a>(&mut self, it: I) {
        let source: SmallBox<dyn ChunkSource + 'a, S32> = smallbox!(Filler {
            it,
            exhausted: false
        });
        // SAFETY: only the trait object's lifetime bound changes, so the layout
        // is identical; the caller guarantees the borrowed data outlives every
        // use, which ends when generated code drops the iterator.
        let source: Source = unsafe { std::mem::transmute(source) };
        self.slot.source = Some(source);
    }
}

// =============================================================================
// Library externs: slot lookup, refill, drop
// =============================================================================

/// Staged type of the pool parameter a kernel receives.
pub type IterPoolRef = SRefMut<Opaque<IterPool>>;
/// Staged type of a slot handle.
pub type PoolSlotRef = SRefMut<Opaque<PoolSlot>>;
/// Staged type of the slot parameter of a producer's `init` extern.
pub type ChunkSlotRef<T> = SRefMut<Opaque<ChunkSlot<<T as StagedType>::RuntimeValue>>>;
type ChunkSlotPtr<T> = SMutPtr<Opaque<ChunkSlot<<T as StagedType>::RuntimeValue>>>;

/// Slot `k` of the pool, created on first use.
#[extern_fn]
pub extern "C" fn iter_pool_slot(pool: &mut IterPool, k: u64) -> &mut PoolSlot {
    pool.slot(k as usize)
}

/// The slot iterator's next chunk of up to `chunk` items; empty once it is
/// exhausted (or if the producer installed nothing).
#[extern_fn]
pub extern "C" fn chunk_refill(slot: &mut PoolSlot, chunk: u64) -> Chunk {
    slot.refill(chunk as usize)
}

/// Drop the slot's iterator. The chunk buffers keep their capacity.
#[extern_fn]
pub extern "C" fn chunk_drop(slot: &mut PoolSlot) {
    slot.source = None;
}

// =============================================================================
// Kinds and resolved fn bundles
// =============================================================================

/// A pooled-iterator kind: the item type and the producer's `init` extern,
/// `init(args.., slot: &mut ChunkSlot<R>)`, which calls [`ChunkSlot::emplace`].
///
/// Safe to implement: a producer that emplaces nothing yields an empty
/// iteration, and the slot's type fixes the item type it may emplace.
pub trait PooledIterKind: 'static {
    type Item: OpaqueIterItem;
    type Init: ExternFn<Ret = ()>;
}

/// The registered externs for a kind. `Copy`, so capture it into kernel closures.
pub struct PooledIterFns<K: PooledIterKind> {
    slot: ExternRef<IterPoolSlotExtern>,
    refill: ExternRef<ChunkRefillExtern>,
    drop: ExternRef<ChunkDropExtern>,
    init: ExternRef<K::Init>,
}

impl<K: PooledIterKind> Clone for PooledIterFns<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: PooledIterKind> Copy for PooledIterFns<K> {}

impl<K: PooledIterKind> PooledIterFns<K> {
    /// Register the kind's producer and the library's slot/refill/drop externs.
    pub fn register(compiler: &mut Compiler) -> Self {
        PooledIterFns {
            slot: compiler.extern_fn(),
            refill: compiler.extern_fn(),
            drop: compiler.extern_fn(),
            init: compiler.extern_fn(),
        }
    }

    /// A source over the producer `init(a, slot)`, reading `chunk` items per
    /// refill.
    pub fn iter1<'p, A, AType>(
        self,
        pool: &'p StagedIterPool,
        a: A,
        chunk: u64,
    ) -> PooledIter<'p, K>
    where
        A: Staged<Out = AType> + 'static,
        AType: StagedType,
        K::Init: ExternFn<Args = (AType, ChunkSlotRef<K::Item>)>,
    {
        let init = self.init;
        PooledIter::new(
            self,
            pool,
            chunk,
            Box::new(move |slot| {
                // SAFETY: `slot` addresses the `PoolSlot` this traversal owns, and
                // `ChunkSlot<R>` is `repr(transparent)` over `PoolSlot`, so it is
                // a valid, exclusive `&mut ChunkSlot<R>` for the call.
                Box::new(unsafe { call_extern2_unchecked(init, a, slot) })
            }),
        )
    }

    /// A source over the producer `init(a, b, slot)`, reading `chunk` items per
    /// refill.
    pub fn iter2<'p, A, B, AType, BType>(
        self,
        pool: &'p StagedIterPool,
        a: A,
        b: B,
        chunk: u64,
    ) -> PooledIter<'p, K>
    where
        A: Staged<Out = AType> + 'static,
        B: Staged<Out = BType> + 'static,
        AType: StagedType,
        BType: StagedType,
        K::Init: ExternFn<Args = (AType, BType, ChunkSlotRef<K::Item>)>,
    {
        let init = self.init;
        PooledIter::new(
            self,
            pool,
            chunk,
            Box::new(move |slot| {
                // SAFETY: as in `iter1`.
                Box::new(unsafe { call_extern3_unchecked(init, a, b, slot) })
            }),
        )
    }
}

// =============================================================================
// Stage 0: slot assignment
// =============================================================================

/// The kernel's view of its [`IterPool`] argument, plus the stage-0 stack that
/// assigns slot indices by nesting depth. Borrow it into nested consumers
/// (`let pool = &pool;`) — sources take `&StagedIterPool`.
pub struct StagedIterPool {
    /// A unique reference, reborrowed once per slot lookup. Behind a `RefCell`
    /// because nested sources share the pool by `&`.
    pool: RefCell<Var<IterPoolRef>>,
    depth: Cell<u64>,
    max_depth: Cell<u64>,
}

impl StagedIterPool {
    /// Wrap the kernel's `&mut IterPool` parameter.
    pub fn new(pool: Var<IterPoolRef>) -> Self {
        StagedIterPool {
            pool: RefCell::new(pool),
            depth: Cell::new(0),
            max_depth: Cell::new(0),
        }
    }

    /// Slots this kernel needs: its deepest pooled-loop nesting so far.
    pub fn slots_used(&self) -> u64 {
        self.max_depth.get()
    }

    fn acquire(&self) -> u64 {
        let k = self.depth.get();
        self.depth.set(k + 1);
        self.max_depth.set(self.max_depth.get().max(k + 1));
        k
    }

    fn release(&self, k: u64) {
        debug_assert_eq!(
            self.depth.get(),
            k + 1,
            "pooled loops released out of order"
        );
        self.depth.set(k);
    }
}

// =============================================================================
// The source
// =============================================================================

/// A single-use reborrow of a unique staged reference, leaving `v` usable for
/// later, sequenced uses (the staged `&mut *v`).
fn reborrow<T: StagedType + 'static>(
    v: &mut Var<SRefMut<T>>,
) -> impl Staged<Out = SRefMut<T>> + 'static {
    IntoExternArg::<SRefMut<T>>::into_extern_arg(v)
}

/// Emits the producer's `init(args.., slot)` call for a given slot handle.
type InitCall<T> = Box<dyn FnOnce(Var<ChunkSlotPtr<T>>) -> Box<dyn Staged<Out = ()>>>;

/// A [`StagedIterator`] over a pooled, chunked external iterator. Built by
/// [`PooledIterFns::iter1`]/[`iter2`](PooledIterFns::iter2).
pub struct PooledIter<'p, K: PooledIterKind> {
    fns: PooledIterFns<K>,
    pool: &'p StagedIterPool,
    chunk: u64,
    init: InitCall<K::Item>,
}

impl<'p, K: PooledIterKind> PooledIter<'p, K> {
    fn new(
        fns: PooledIterFns<K>,
        pool: &'p StagedIterPool,
        chunk: u64,
        init: InitCall<K::Item>,
    ) -> Self {
        assert!(chunk > 0, "pooled iterator chunk size must be positive");
        PooledIter {
            fns,
            pool,
            chunk,
            init,
        }
    }
}

impl<K: PooledIterKind> StagedIterator for PooledIter<'_, K>
where
    <K::Item as StagedType>::RuntimeValue: Copy,
{
    type Item = K::Item;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<K::Item>),
    {
        let PooledIter {
            fns,
            pool,
            chunk,
            init,
        } = self;
        let k = pool.acquire();
        let chunk = Const::<u64>::new(chunk);

        let pool_ref = reborrow(&mut pool.pool.borrow_mut());
        // SAFETY: the returned `&mut PoolSlot` is exclusive for this traversal —
        // stage 0 hands each live nesting depth its own index `k` — and it lives
        // in the kernel's `IterPool`, which outlives the call. Each call below
        // reborrows the unique `slot`.
        let mut slot: Var<PoolSlotRef> =
            ctx.bind(unsafe { call_extern2_unchecked(fns.slot, pool_ref, Const::<u64>::new(k)) });
        let typed_slot = ctx.bind(ptr_cast_mut::<Opaque<ChunkSlot<_>>, Opaque<PoolSlot>, _>(
            ref_mut_as_ptr(reborrow(&mut slot)),
        ));
        ctx.emit(init(typed_slot));

        let first = ctx.bind(call_extern2(fns.refill, &mut slot, chunk));
        let data = ctx.var(ptr_cast::<K::Item, u8, _>(first.get(ChunkType::ptr())));
        let n = ctx.var(first.get(ChunkType::len()));
        let i = ctx.var(0u64);
        let refill = fns.refill;
        let slot_in_loop = &mut slot;
        ctx.while_loop(true, move |ctx| {
            ctx.if_then(eq(i, n), move |ctx| {
                let next = ctx.bind(call_extern2(refill, slot_in_loop, chunk));
                ctx.store(data, ptr_cast::<K::Item, u8, _>(next.get(ChunkType::ptr())));
                ctx.store(n, next.get(ChunkType::len()));
                ctx.store(i, 0u64);
                ctx.if_then(eq(n, 0u64), |ctx| ctx.break_loop());
            });
            let idx = ctx.bind(int_cast::<i64, u64, _>(i));
            // SAFETY: `i < n`, and `data` points at the `n` items of the slot's
            // item type that the last refill wrote; they stay put until the
            // next refill.
            let elem = ctx.bind(unsafe { load(ptr_offset(data, idx)) });
            ctx.store(i, add(i, 1u64));
            consumer(ctx, elem);
        });
        ctx.emit(call_extern1(fns.drop, &mut slot));

        pool.release(k);
    }
}
