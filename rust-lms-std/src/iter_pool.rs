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

use std::cell::Cell;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use rust_lms::prelude::*;

// =============================================================================
// Host side: the pool and its slots
// =============================================================================

/// Inline storage budget (bytes) for a slot's iterator. Iterators that fit live
/// in the slot; larger ones fall back to one heap box per `init`.
pub const POOLED_ITER_INLINE_CAP: usize = 256;

/// Alignment of a slot's inline iterator storage and of its chunk buffer.
/// Item types must not need more (checked at compile time, per item type).
const POOL_ALIGN: usize = 16;

#[repr(C, align(16))]
struct InlineStorage([MaybeUninit<u8>; POOLED_ITER_INLINE_CAP]);

/// One aligned unit of a chunk buffer. Buffers are `Vec`s of these, so the
/// allocation is managed by `Vec` and always `POOL_ALIGN`-aligned.
#[derive(Clone, Copy)]
#[repr(C, align(16))]
struct Block([MaybeUninit<u8>; POOL_ALIGN]);

/// Where a slot's iterator lives.
///
/// An inline iterator's address is recomputed from the slot borrow in hand on
/// every use, never stored: each extern call reborrows the slot afresh, which
/// would invalidate a pointer derived from an earlier borrow.
#[derive(Clone, Copy)]
enum Place {
    Inline,
    Heap(*mut u8),
}

/// The type-erased iterator currently emplaced in a slot: where it lives, plus
/// monomorphic refill/drop functions for its concrete type.
struct ErasedIter {
    place: Place,
    /// Write up to `chunk` items into the buffer; returns the data pointer and
    /// the number written.
    refill: unsafe fn(*mut u8, &mut Vec<Block>, usize) -> (*const u8, usize),
    drop: unsafe fn(*mut u8),
}

/// One reusable slot: iterator storage plus a chunk buffer. Boxed by the pool so
/// its address is stable.
struct PoolSlot {
    storage: InlineStorage,
    iter: Option<ErasedIter>,
    /// Set once a refill comes back short, so a non-fused iterator is never
    /// polled again after reporting its end.
    exhausted: bool,
    buf: Vec<Block>,
}

impl PoolSlot {
    fn new() -> Self {
        PoolSlot {
            storage: InlineStorage([MaybeUninit::uninit(); POOLED_ITER_INLINE_CAP]),
            iter: None,
            exhausted: false,
            buf: Vec::new(),
        }
    }

    /// Drop the emplaced iterator, if any. The buffer and its capacity stay.
    fn clear(&mut self) {
        if let Some(it) = self.iter.take() {
            let data = iter_data(&mut self.storage, it.place);
            // SAFETY: `it` was produced by `ChunkSlot::emplace` for the value at
            // `data`, and `take` guarantees it is dropped only once.
            unsafe { (it.drop)(data) };
        }
        self.exhausted = false;
    }

    /// Write the next chunk into the buffer; returns its data pointer and length.
    fn refill(&mut self, chunk: usize) -> (*const u8, usize) {
        let PoolSlot {
            storage,
            iter,
            exhausted,
            buf,
        } = self;
        let (ptr, n) = match iter {
            Some(it) if !*exhausted => {
                let data = iter_data(storage, it.place);
                // SAFETY: `data` holds the live iterator `it.refill` was
                // monomorphized for (see `ChunkSlot::emplace`).
                unsafe { (it.refill)(data, buf, chunk) }
            }
            // Items are never read from an empty chunk; any aligned pointer does.
            _ => (buf.as_ptr().cast(), 0),
        };
        if n < chunk {
            *exhausted = true;
        }
        (ptr, n)
    }
}

fn iter_data(storage: &mut InlineStorage, place: Place) -> *mut u8 {
    match place {
        Place::Inline => storage.0.as_mut_ptr().cast(),
        Place::Heap(p) => p,
    }
}

/// Host owner of a kernel's pooled-iterator slots. Create one per thread, keep
/// it across calls, and pass it to each call as `&mut IterPool`.
///
/// Slots are created lazily the first time a kernel reaches a nesting depth, and
/// keep their buffers (capacity) until the pool is dropped.
pub struct IterPool {
    /// Individually boxed (via `Box::into_raw`), so slot addresses never move
    /// while the list grows.
    slots: Vec<NonNull<PoolSlot>>,
}

impl IterPool {
    pub fn new() -> Self {
        IterPool { slots: Vec::new() }
    }

    /// Number of slots created so far (the deepest nesting any kernel reached).
    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    fn slot(&mut self, k: usize) -> NonNull<PoolSlot> {
        while self.slots.len() <= k {
            let slot = Box::into_raw(Box::new(PoolSlot::new()));
            // SAFETY: `Box::into_raw` never returns null.
            self.slots.push(unsafe { NonNull::new_unchecked(slot) });
        }
        self.slots[k]
    }
}

impl Default for IterPool {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for IterPool {
    fn drop(&mut self) {
        for slot in self.slots.drain(..) {
            // SAFETY: every entry came from `Box::into_raw` in `slot` and is
            // released exactly once, here.
            let mut slot = unsafe { Box::from_raw(slot.as_ptr()) };
            slot.clear();
        }
    }
}

/// A pool slot typed by the runtime item type `R` its iterator yields. This is
/// what a producer's `init` extern receives (`slot: &mut ChunkSlot<R>`), so the
/// item type the staged kind declares is the item type the producer must emplace.
#[repr(transparent)]
pub struct ChunkSlot<R> {
    slot: PoolSlot,
    _item: PhantomData<fn() -> R>,
}

impl<R: Copy> ChunkSlot<R> {
    /// Install `it` as this slot's iterator: inline if it fits
    /// [`POOLED_ITER_INLINE_CAP`], else one heap box. Any iterator left from an
    /// earlier use is dropped first; the chunk buffer is kept.
    ///
    /// # Safety
    ///
    /// Anything `it` borrows must outlive the kernel call that runs this
    /// traversal: generated code keeps the iterator after this function returns
    /// and drops it when the loop exits.
    pub unsafe fn emplace<I: Iterator<Item = R>>(&mut self, it: I) {
        unsafe fn refill<R: Copy, I: Iterator<Item = R>>(
            data: *mut u8,
            buf: &mut Vec<Block>,
            chunk: usize,
        ) -> (*const u8, usize) {
            const {
                assert!(
                    std::mem::align_of::<R>() <= POOL_ALIGN,
                    "pooled iterator items must not need more than 16-byte alignment"
                )
            };
            let Some(bytes) = chunk.checked_mul(std::mem::size_of::<R>()) else {
                std::process::abort();
            };
            let blocks = bytes.div_ceil(POOL_ALIGN);
            if buf.len() < blocks {
                buf.resize(blocks, Block([MaybeUninit::uninit(); POOL_ALIGN]));
            }
            let out = buf.as_mut_ptr().cast::<R>();
            // SAFETY: `data` holds a live `I` (see `ChunkSlot::emplace`).
            let it = unsafe { &mut *data.cast::<I>() };
            let mut n = 0;
            while n < chunk {
                let Some(x) = it.next() else { break };
                // SAFETY: the buffer holds at least `chunk` items of `R`, and is
                // aligned for `R` (checked above).
                unsafe { out.add(n).write(x) };
                n += 1;
            }
            (out.cast_const().cast(), n)
        }
        unsafe fn drop_inline<I>(data: *mut u8) {
            // SAFETY: `data` holds a live `I` written in place by `emplace`.
            unsafe { std::ptr::drop_in_place(data.cast::<I>()) };
        }
        unsafe fn drop_heap<I>(data: *mut u8) {
            // SAFETY: `data` came from `Box::into_raw` in `emplace`.
            unsafe { drop(Box::from_raw(data.cast::<I>())) };
        }

        let slot = &mut self.slot;
        slot.clear();
        let (place, drop): (Place, unsafe fn(*mut u8)) = if std::mem::size_of::<I>()
            <= POOLED_ITER_INLINE_CAP
            && std::mem::align_of::<I>() <= POOL_ALIGN
        {
            let dst = slot.storage.0.as_mut_ptr().cast::<I>();
            // SAFETY: the storage is large and aligned enough for `I` (checked
            // above), and holds no live value after `clear`.
            unsafe { dst.write(it) };
            (Place::Inline, drop_inline::<I>)
        } else {
            (
                Place::Heap(Box::into_raw(Box::new(it)).cast()),
                drop_heap::<I>,
            )
        };
        slot.iter = Some(ErasedIter {
            place,
            refill: refill::<R, I>,
            drop,
        });
    }
}

// =============================================================================
// Library externs: slot lookup, refill, drop
// =============================================================================
//
// Generic over the staged item type `T`, so each is monomorphized per item type
// and its signature carries `ChunkSlot<T::RuntimeValue>`. The thunks follow the
// canonical storage-pointer ABI: each argument arrives as a pointer to its
// storage, and the result is written through the trailing output pointer.

/// Staged type of a slot handle in generated code: `*mut ChunkSlot<R>`.
pub type ChunkSlotPtr<T> = SMutPtr<Opaque<ChunkSlot<<T as StagedType>::RuntimeValue>>>;
/// Staged type of the slot parameter of a producer's `init` extern.
pub type ChunkSlotRef<T> = SRefMut<Opaque<ChunkSlot<<T as StagedType>::RuntimeValue>>>;
/// Staged type of the pool parameter a kernel receives.
pub type IterPoolRef = SRefMut<Opaque<IterPool>>;

unsafe extern "C" fn iter_pool_slot_thunk<R>(pool: *const u8, k: *const u8, out: *mut u8) {
    // SAFETY: storage-pointer ABI — `pool` holds a `&mut IterPool`, `k` a `u64`,
    // and `out` has room for one pointer.
    unsafe {
        let pool = pool.cast::<*mut IterPool>().read();
        let k = k.cast::<u64>().read();
        let slot = (*pool).slot(k as usize);
        out.cast::<*mut ChunkSlot<R>>().write(slot.as_ptr().cast());
    }
}

unsafe extern "C" fn chunk_refill_thunk<R>(slot: *const u8, chunk: *const u8, out: *mut u8) {
    // SAFETY: storage-pointer ABI — `slot` holds a pointer to a live slot, `chunk`
    // a `u64`, and `out` has room for one `FatSlice<R>`.
    unsafe {
        let slot = &mut *slot.cast::<*mut PoolSlot>().read();
        let chunk = chunk.cast::<u64>().read() as usize;
        let (ptr, n) = slot.refill(chunk);
        out.cast::<FatSlice<R>>()
            .write(FatSlice::from_raw_parts(ptr.cast(), n));
    }
}

unsafe extern "C" fn chunk_drop_thunk(slot: *const u8, _out: *mut u8) {
    // SAFETY: storage-pointer ABI — `slot` holds a pointer to a live slot.
    unsafe { (*slot.cast::<*mut PoolSlot>().read()).clear() };
}

/// `iter_pool_slot(pool: &mut IterPool, k: u64) -> *mut ChunkSlot<R>`: slot `k`,
/// created on first use.
#[doc(hidden)]
pub struct IterPoolSlotExtern<T>(PhantomData<T>);
unsafe impl<T: StagedType + 'static> ExternFn for IterPoolSlotExtern<T> {
    type Args = (IterPoolRef, u64);
    type Ret = ChunkSlotPtr<T>;
    const NAME: &'static str = "iter_pool_slot";
    const FN_PTR: *const u8 = iter_pool_slot_thunk::<T::RuntimeValue> as *const u8;
}

/// `chunk_refill(slot: &mut ChunkSlot<R>, chunk: u64) -> FatSlice<R>`: the next
/// chunk, empty once the iterator is exhausted.
#[doc(hidden)]
pub struct ChunkRefillExtern<T>(PhantomData<T>);
unsafe impl<T: StagedType + 'static> ExternFn for ChunkRefillExtern<T> {
    type Args = (ChunkSlotRef<T>, u64);
    type Ret = RawSlice<T>;
    const NAME: &'static str = "chunk_refill";
    const FN_PTR: *const u8 = chunk_refill_thunk::<T::RuntimeValue> as *const u8;
}

/// `chunk_drop(slot: &mut ChunkSlot<R>)`: drop the slot's iterator, keep the
/// buffer.
#[doc(hidden)]
pub struct ChunkDropExtern<T>(PhantomData<T>);
unsafe impl<T: StagedType + 'static> ExternFn for ChunkDropExtern<T> {
    type Args = (ChunkSlotRef<T>,);
    type Ret = ();
    const NAME: &'static str = "chunk_drop";
    const FN_PTR: *const u8 = chunk_drop_thunk as *const u8;
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
    slot: ExternRef<IterPoolSlotExtern<K::Item>>,
    refill: ExternRef<ChunkRefillExtern<K::Item>>,
    drop: ExternRef<ChunkDropExtern<K::Item>>,
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
                // SAFETY: `slot` comes from `iter_pool_slot` for this kernel's
                // live pool and is used by this traversal alone (see `PooledIter`).
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
    pool: Var<SMutPtr<Opaque<IterPool>>>,
    depth: Cell<u64>,
    max_depth: Cell<u64>,
}

impl StagedIterPool {
    /// Wrap the kernel's `&mut IterPool` parameter.
    pub fn new(ctx: &mut Ctx, pool: Var<IterPoolRef>) -> Self {
        StagedIterPool {
            pool: ctx.bind(ref_mut_as_ptr(pool)),
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

        // SAFETY (all extern calls below): `pool.pool` is the kernel's live
        // `&mut IterPool`; slot `k` is used by no other live traversal, because
        // stage 0 hands each nesting depth its own index.
        let slot =
            ctx.bind(unsafe { call_extern2_unchecked(fns.slot, pool.pool, Const::<u64>::new(k)) });
        ctx.emit(init(slot));

        let first =
            ctx.bind(unsafe { call_extern2_unchecked(fns.refill, slot, Const::<u64>::new(chunk)) });
        let data = ctx.var(first.into_ptr());
        let n = ctx.var(first.len());
        let i = ctx.var(0u64);
        let refill = fns.refill;
        ctx.while_loop(true, move |ctx| {
            ctx.if_then(eq(i, n), move |ctx| {
                let next = ctx.bind(unsafe {
                    call_extern2_unchecked(refill, slot, Const::<u64>::new(chunk))
                });
                ctx.store(data, next.into_ptr());
                ctx.store(n, next.len());
                ctx.store(i, 0u64);
                ctx.if_then(eq(n, 0u64), |ctx| ctx.break_loop());
            });
            let idx = ctx.bind(int_cast::<i64, u64, _>(i));
            // SAFETY: `i < n`, and `data` points at the `n` items the last refill
            // wrote, which stay put until the next refill.
            let elem = ctx.bind(unsafe { load(ptr_offset(data, idx)) });
            ctx.store(i, add(i, 1u64));
            consumer(ctx, elem);
        });
        ctx.emit(unsafe { call_extern1_unchecked(fns.drop, slot) });

        pool.release(k);
    }
}
