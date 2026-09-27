//! [`ChunkedIter`] — an external Rust iterator driven from staged code a
//! **chunk** at a time, through a buffer in the kernel's own stack frame.
//!
//! The cost that matters for graph-shaped data is per *list*, not per item:
//! most adjacency lists are short, so a design is judged by the extern calls it
//! makes per list. Here a list of up to [`CHUNK`] items costs **one** call: the
//! producer builds its iterator and fills the first chunk in the same call, and
//! drops the iterator right there if it is already exhausted.
//!
//! ```text
//! slot = stack slot (one per loop in the kernel frame; reentrant, allocation-free)
//! slot.head = { len: 0, done: 1 }        ; a producer that starts nothing is empty
//! init(args.., slot)                     ; producer: slot.start(iter) — fills chunk 1
//! n = head.len; done = head.done; i = 0
//! loop:
//!   if i == n {
//!     if done { break }                  ; short list: no further call
//!     fill(slot); n = head.len; done = head.done; i = 0
//!     if n == 0 { break }
//!   }
//!   elem = buf[i]; i += 1
//!   <consumer>
//! exit:
//!   if !done { drop(slot) }              ; only after an early `break_loop`
//! ```
//!
//! The loop is flattened — one loop with a cold refill branch — so the
//! combinators' `break_loop` (`any`, `find_map`, `take_while`) leaves the whole
//! iteration, never just one chunk.
//!
//! # When to use it
//!
//! Measured by `benches/graph_iter.rs` (1M-node graphs, mean out-degree ≈ 10,
//! power-law and uniform):
//!
//! - about **1.4–1.9× faster** than [`ReusedOpaqueIter`],
//!   which makes one indirect extern call per item;
//! - about **1.4–1.9× slower** than one extern call per list that returns the
//!   list as a slice, and slower again than handing the kernel the storage itself.
//!
//! So when storage can expose a list as a slice, pass the slice. Use this when
//! only an iterator can produce the items — computed, filtered, or spread across
//! structures a slice cannot describe.
//!
//! # Where the unsafe is
//!
//! The host side is safe Rust: the slot is an ordinary `#[repr(C)]` struct, the
//! iterator a [`SmallBox`] trait object, the chunk a `[MaybeUninit<R>; CHUNK]`
//! that Rust only ever writes. The externs are plain `#[extern_fn]`s. What cannot
//! be expressed safely is confined to:
//!
//! 1. [`ChunkStart::start_borrowed`] — storing an iterator that borrows host data
//!    in a slot the borrow cannot be named for across `extern "C"`.
//! 2. `SlotView` — the kernel's typed view of its stack slot: resetting the
//!    head, reading it back, handing the slot to the externs, and reading
//!    `buf[i]`. Each operation states the protocol step that makes it valid.

use std::marker::PhantomData;
use std::mem::MaybeUninit;

use rust_lms::prelude::*;
use smallbox::space::S32;
use smallbox::{SmallBox, smallbox};

/// Items per chunk. Most adjacency lists fit in one, so they cost one call.
pub const CHUNK: usize = 64;

// =============================================================================
// Host side (safe Rust)
// =============================================================================

/// The part of a slot the kernel reads: how many items the last fill wrote, and
/// whether the iterator is finished (and already dropped). Protocol internal.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct ChunkHead {
    #[staged(u64)]
    len: usize,
    #[staged(u64)]
    done: usize,
}

/// A type-erased iterator's one operation: fill a chunk.
trait Fill<R> {
    /// Write up to `CHUNK` items; returns the count and whether the iterator is
    /// finished.
    fn fill(&mut self, buf: &mut [MaybeUninit<R>; CHUNK]) -> (usize, bool);
}

impl<I: Iterator> Fill<I::Item> for I {
    fn fill(&mut self, buf: &mut [MaybeUninit<I::Item>; CHUNK]) -> (usize, bool) {
        for (n, slot) in buf.iter_mut().enumerate() {
            match self.next() {
                Some(x) => {
                    slot.write(x);
                }
                None => return (n, true),
            }
        }
        // A full chunk: finished only if the iterator says so (exact for slice
        // iterators), which saves a call that would return nothing.
        (CHUNK, self.size_hint().1 == Some(0))
    }
}

/// A kernel stack slot holding one iterator and its current chunk. Protocol
/// internal: producers see it only as [`ChunkStart`].
///
/// `head` comes first (`#[repr(C)]`), so the kernel reads it at offset 0.
#[repr(C)]
pub struct ChunkedSlot<R> {
    head: ChunkHead,
    buf: [MaybeUninit<R>; CHUNK],
    source: Option<SmallBox<dyn Fill<R>, S32>>,
}

impl<R> ChunkedSlot<R> {
    /// Refill from the iterator; drop it as soon as it reports the end.
    fn refill(&mut self) {
        let (len, done) = match &mut self.source {
            Some(source) => source.fill(&mut self.buf),
            None => (0, true),
        };
        if done {
            self.source = None;
        }
        self.head = ChunkHead {
            len,
            done: done as usize,
        };
    }
}

/// What a producer's `init` extern receives: the uninitialized slot, typed by
/// the item type `R` the kernel will read.
#[repr(transparent)]
pub struct ChunkStart<R>(MaybeUninit<ChunkedSlot<R>>);

impl<R: Copy + 'static> ChunkStart<R> {
    /// Build the slot around `it` and fill its first chunk.
    ///
    /// Call it at most once per `init`: a second call replaces the first
    /// iterator without dropping it (a leak, not undefined behavior).
    ///
    /// The slot's type fixes the item type, so a producer cannot start an
    /// iterator the kernel would read as something else:
    ///
    /// ```compile_fail
    /// # use rust_lms_std::ChunkStart;
    /// fn producer(slot: &mut ChunkStart<u64>) {
    ///     slot.start([1.5f64, 2.5].into_iter()); // error: expected `u64`
    /// }
    /// ```
    ///
    /// And an iterator that borrows needs [`start_borrowed`](Self::start_borrowed):
    ///
    /// ```compile_fail
    /// # use rust_lms_std::ChunkStart;
    /// fn producer(data: &[u64], slot: &mut ChunkStart<u64>) {
    ///     slot.start(data.iter().copied()); // error: `data` must be 'static
    /// }
    /// ```
    pub fn start<I: Iterator<Item = R> + 'static>(&mut self, it: I) {
        self.start_source(smallbox!(it));
    }

    /// [`start`](Self::start) for an iterator that borrows host data — the
    /// common case, e.g. neighbours read out of a graph passed to the producer.
    ///
    /// # Safety
    ///
    /// Everything `it` borrows must outlive the kernel call that runs this
    /// traversal. Generated code drops the iterator before the call returns,
    /// but the borrow cannot be named across `extern "C"`.
    pub unsafe fn start_borrowed<'a, I: Iterator<Item = R> + 'a>(&mut self, it: I) {
        let source: SmallBox<dyn Fill<R> + 'a, S32> = smallbox!(it);
        // SAFETY: only the trait object's lifetime bound changes, so the layout
        // is identical; the caller guarantees the borrowed data outlives every
        // use, which ends when generated code drops the iterator.
        self.start_source(unsafe {
            std::mem::transmute::<SmallBox<dyn Fill<R> + 'a, S32>, SmallBox<dyn Fill<R>, S32>>(
                source,
            )
        });
    }

    fn start_source(&mut self, source: SmallBox<dyn Fill<R>, S32>) {
        let slot = self.0.write(ChunkedSlot {
            head: ChunkHead { len: 0, done: 0 },
            buf: [const { MaybeUninit::uninit() }; CHUNK],
            source: Some(source),
        });
        slot.refill();
    }
}

/// The next chunk. Called only while `head.done == 0`.
#[extern_fn]
pub extern "C" fn chunk_fill<R: Copy + 'static>(slot: &mut ChunkedSlot<R>) {
    slot.refill();
}

/// Drop the iterator after an early exit. Called only while `head.done == 0`.
#[extern_fn]
pub extern "C" fn chunk_drop<R: Copy + 'static>(slot: &mut ChunkedSlot<R>) {
    slot.source = None;
}

// =============================================================================
// Kinds and resolved fn bundles
// =============================================================================

type Rt<T> = <T as StagedType>::RuntimeValue;

/// Staged type of the slot parameter of a producer's `init` extern.
pub type ChunkStartRef<T> = SRefMut<Opaque<ChunkStart<Rt<T>>>>;

/// A chunked-iterator kind: the item type and the producer's `init` extern,
/// `init(args.., slot: &mut ChunkStart<R>)`, which calls [`ChunkStart::start`].
///
/// Safe to implement: a producer that starts nothing yields an empty iteration,
/// and the slot's type fixes the item type it may start.
pub trait ChunkedIterKind: 'static {
    type Item: OpaqueIterItem;
    type Init: ExternFn<Ret = ()>;
}

/// The registered externs for a kind. `Copy`, so capture it into kernel closures.
pub struct ChunkedIterFns<K: ChunkedIterKind> {
    init: ExternRef<K::Init>,
    fill: ExternRef<ChunkFillExtern<Rt<K::Item>>>,
    drop: ExternRef<ChunkDropExtern<Rt<K::Item>>>,
}

impl<K: ChunkedIterKind> Clone for ChunkedIterFns<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: ChunkedIterKind> Copy for ChunkedIterFns<K> {}

impl<K: ChunkedIterKind> ChunkedIterFns<K>
where
    Rt<K::Item>: Copy,
{
    /// Register the kind's producer and the library's fill/drop externs.
    pub fn register(compiler: &mut Compiler) -> Self {
        ChunkedIterFns {
            init: compiler.extern_fn(),
            fill: compiler.extern_fn(),
            drop: compiler.extern_fn(),
        }
    }

    /// A source over the producer `init(a, slot)`.
    pub fn iter1<A, AType>(self, a: A) -> ChunkedIter<K>
    where
        A: Staged<Out = AType> + 'static,
        AType: StagedType,
        K::Init: ExternFn<Args = (AType, ChunkStartRef<K::Item>)>,
    {
        let init = self.init;
        ChunkedIter {
            fns: self,
            // SAFETY: `slot` is this traversal's reset stack slot (see `SlotView`).
            init: Box::new(move |slot| Box::new(unsafe { call_extern2_unchecked(init, a, slot) })),
        }
    }

    /// A source over the producer `init(a, b, slot)`.
    pub fn iter2<A, B, AType, BType>(self, a: A, b: B) -> ChunkedIter<K>
    where
        A: Staged<Out = AType> + 'static,
        B: Staged<Out = BType> + 'static,
        AType: StagedType,
        BType: StagedType,
        K::Init: ExternFn<Args = (AType, BType, ChunkStartRef<K::Item>)>,
    {
        let init = self.init;
        ChunkedIter {
            fns: self,
            // SAFETY: as in `iter1`.
            init: Box::new(move |slot| {
                Box::new(unsafe { call_extern3_unchecked(init, a, b, slot) })
            }),
        }
    }
}

// =============================================================================
// Stage 0: the kernel's view of its slot
// =============================================================================

/// The kernel's typed view of one stack slot. Every raw access to the slot goes
/// through here, so each can state the protocol step that makes it valid.
#[derive(Clone, Copy)]
struct SlotView<T: StagedType> {
    base: Var<SMutPtr<u8>>,
    _item: PhantomData<T>,
}

impl<T> SlotView<T>
where
    T: OpaqueIterItem,
    Rt<T>: Copy,
{
    fn reserve(ctx: &mut Ctx) -> Self {
        const {
            assert!(
                std::mem::align_of::<ChunkedSlot<Rt<T>>>() <= 8,
                "chunked iterator slots are 8-byte aligned stack scratch"
            )
        };
        SlotView {
            base: ctx.bind(stack_alloc(std::mem::size_of::<ChunkedSlot<Rt<T>>>())),
            _item: PhantomData,
        }
    }

    fn head(self) -> impl Staged<Out = SMutPtr<ChunkHead>> + Copy {
        ptr_cast_mut::<ChunkHead, u8, _>(self.base)
    }

    /// Mark the slot empty-and-done before the producer runs, so a producer that
    /// starts nothing reads as an empty iteration.
    fn reset(self, ctx: &mut Ctx) {
        // SAFETY: `head` is at offset 0 of this live, 8-aligned stack slot.
        unsafe {
            ctx.emit(store(
                field_addr(self.head(), ChunkHeadType::len()),
                Const::<u64>::new(0),
            ));
            ctx.emit(store(
                field_addr(self.head(), ChunkHeadType::done()),
                Const::<u64>::new(1),
            ));
        }
    }

    /// `(len, done)` as the last reset, start or fill left them.
    fn read_head(self) -> (impl Staged<Out = u64>, impl Staged<Out = u64>) {
        // SAFETY: the head is always initialized — by `reset`, then by the
        // producer's `start` and every `fill`.
        unsafe {
            (
                load_field_unchecked(self.head(), ChunkHeadType::len()),
                load_field_unchecked(self.head(), ChunkHeadType::done()),
            )
        }
    }

    /// The slot typed for the producer: `&mut ChunkStart<R>` (uninitialized).
    fn start_arg(self) -> impl Staged<Out = StartPtr<T>> + Copy + 'static {
        ptr_cast_mut::<Opaque<ChunkStart<Rt<T>>>, u8, _>(self.base)
    }

    /// The slot typed for `fill`/`drop`: `&mut ChunkedSlot<R>`.
    ///
    /// Valid only while `done == 0`: that state is written only by a `start`,
    /// which initialized the whole slot.
    fn slot_arg(self) -> impl Staged<Out = SMutPtr<Opaque<ChunkedSlot<Rt<T>>>>> + Copy {
        ptr_cast_mut::<Opaque<ChunkedSlot<Rt<T>>>, u8, _>(self.base)
    }

    /// Item `i` of the current chunk.
    ///
    /// # Safety
    ///
    /// `i < len` for the current head: the last start/fill wrote items `0..len`.
    unsafe fn item(self, ctx: &mut Ctx, i: Var<u64>) -> Var<T> {
        let buf = std::mem::offset_of!(ChunkedSlot<Rt<T>>, buf) as i64;
        // SAFETY: `buf` is within the slot; `ChunkedSlot`'s layout aligns it for `T`.
        let data = ptr_cast::<T, u8, _>(ptr_as_const(unsafe {
            ptr_offset_mut(self.base, Const::<i64>::new(buf))
        }));
        let idx = ctx.bind(int_cast::<i64, u64, _>(i));
        // SAFETY: forwarded from the caller: item `i` was written by the last fill.
        ctx.bind(unsafe { load(ptr_offset(data, idx)) })
    }
}

// =============================================================================
// The source
// =============================================================================

type StartPtr<T> = SMutPtr<Opaque<ChunkStart<Rt<T>>>>;
type InitCall<T> = Box<dyn FnOnce(Var<StartPtr<T>>) -> Box<dyn Staged<Out = ()>>>;

/// A [`StagedIterator`] over a chunked external iterator. Built by
/// [`ChunkedIterFns::iter1`]/[`iter2`](ChunkedIterFns::iter2).
pub struct ChunkedIter<K: ChunkedIterKind> {
    fns: ChunkedIterFns<K>,
    init: InitCall<K::Item>,
}

impl<K: ChunkedIterKind> StagedIterator for ChunkedIter<K>
where
    Rt<K::Item>: Copy,
{
    type Item = K::Item;
    type Cursor = ChunkedCursor<K>;

    fn open(self, ctx: &mut Ctx) -> ChunkedCursor<K> {
        let ChunkedIter { fns, init } = self;
        let slot = SlotView::<K::Item>::reserve(ctx);
        slot.reset(ctx);
        let start = ctx.bind(slot.start_arg());
        ctx.emit(init(start));

        let (len, done) = slot.read_head();
        ChunkedCursor {
            slot,
            n: ctx.var(len),
            finished: ctx.var(done),
            i: ctx.var(0u64),
            fns,
        }
    }
}

/// The cursor of a [`ChunkedIter`]: an index into the current chunk, with a
/// cold refill branch. See the module docs for the loop it emits.
pub struct ChunkedCursor<K: ChunkedIterKind> {
    slot: SlotView<K::Item>,
    n: Var<u64>,
    finished: Var<u64>,
    i: Var<u64>,
    fns: ChunkedIterFns<K>,
}

impl<K: ChunkedIterKind> Cursor for ChunkedCursor<K>
where
    Rt<K::Item>: Copy,
{
    type Item = K::Item;
    type Close = ChunkedClose<K>;

    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<K::Item>, ChunkedClose<K>) {
        let ChunkedCursor {
            slot,
            n,
            finished,
            i,
            fns,
        } = self;
        let fill = fns.fill;
        ctx.if_then(eq(i, n), move |ctx| {
            ctx.exit_if(eq(finished, 1u64), done);
            // SAFETY: `finished == 0`, so a `start` initialized the slot.
            ctx.emit(unsafe { call_extern1_unchecked(fill, slot.slot_arg()) });
            let (len, is_done) = slot.read_head();
            ctx.store(n, len);
            ctx.store(finished, is_done);
            ctx.store(i, 0u64);
            ctx.exit_if(eq(n, 0u64), done);
        });
        // SAFETY: `i < n`, the item count of the last start/fill.
        let elem = unsafe { slot.item(ctx, i) };
        ctx.store(i, add(i, 1u64));
        (
            elem,
            ChunkedClose {
                slot,
                finished,
                drop: fns.drop,
            },
        )
    }
}

/// Drops the iterator after an early exit: when the traversal stopped before
/// the source reported its end, the iterator is still live in the slot.
pub struct ChunkedClose<K: ChunkedIterKind> {
    slot: SlotView<K::Item>,
    finished: Var<u64>,
    drop: ExternRef<ChunkDropExtern<Rt<K::Item>>>,
}

impl<K: ChunkedIterKind> Clone for ChunkedClose<K> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<K: ChunkedIterKind> Copy for ChunkedClose<K> {}

impl<K: ChunkedIterKind> Close for ChunkedClose<K>
where
    Rt<K::Item>: Copy,
{
    fn close(self, ctx: &mut Ctx) {
        let (slot, drop) = (self.slot, self.drop);
        ctx.if_then(eq(self.finished, 0u64), move |ctx| {
            // SAFETY: `finished == 0`, so a `start` initialized the slot and the
            // iterator is still live.
            ctx.emit(unsafe { call_extern1_unchecked(drop, slot.slot_arg()) });
        });
    }
}
