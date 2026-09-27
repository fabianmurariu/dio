# Pull iterators for rust-lms — making `zip` and `chain` work on every source

Status: analysis / design proposal. Nothing here is implemented yet.
Scope: `rust-lms/src/iter/`, plus the `Ctx` control-flow primitives it would need
(`rust-lms/src/func.rs`), and how a Raphtory query engine would use the result.

---

## 0. Summary

- **Every iterator is push-based today.** A source owns the loop and calls its
  consumer once, at staging time, to emit the loop body. Combinators wrap the
  consumer. This fuses perfectly, but only when a single loop is involved.
- **`zip` needs two sources advanced in lock-step.** With push that means
  interleaving two loops, which can't be done without coroutines. The current
  `zip` gets around this by requiring random access (`IndexedSource`): one loop,
  one counter, `get_at(i)` on both sides. External (Raphtory B-tree) iterators
  have no `get_at`, so they can't be zipped.
- **`chain` needs one consumer fed by two loops.** Push only allows that if the
  consumer is duplicated (it's `FnOnce`, so it can't be) or if the consumer is
  jumped to and "returns" to the right loop. On top of that, `break_loop` in
  `take_while`/`any`/`find_map` would leave the first loop and carry on into the
  second, which is wrong.
- **Every source in the code base is already a pull iterator underneath.** The
  slice, range, opaque (`next`/`drop`), exact-size opaque, reused-slot opaque,
  chunked (rust-lms-std) and `from_fn` sources all run a `step → test → body`
  loop. Push is how the loop is *written*, not a property of the data. The one
  genuinely push-only construct is a nested-loop `flat_map`, which doesn't exist
  on master.
- **Recommendation (option D below):** make pull the required protocol of
  `StagedIterator` and derive `for_each` from it as a *provided* method, which
  any iterator can override (`flat_map`'s nested loops, for example). The pull
  step is written in *direct style*: it emits code that either falls through
  with the item bound, or jumps to a typed `done` label. With that shape:
  - `zip` is two `next` calls in a row, both jumping to the same `done`.
  - `chain` is a phase flag plus a join block, and the consumer is emitted once.
  - `filter`/`filter_map`/`skip_while` use a local backward label, not a loop,
    so `break_loop` in a terminal still targets the one driver loop.
  - `merge_by`, `take`, `skip`, `peekable` and sorted intersection become easy.
    Raphtory needs these (time-ordered merges of the memory and disk layers,
    neighbour intersection).
  - All existing combinators and terminals keep their semantics and, for the
    single-source pipelines we have today, produce the same CFG.
- **One new codegen primitive is needed:** Wasm-style labels (`join`/`repeat`/
  `goto`) that are typed, lifetime-branded, and sealed at scope end. They reuse
  the typed merge-block machinery that `IfThenElse` already uses
  (`append_value_block_params`, `jump_value`, `block_value`), so both backends
  already support what they emit.

The rest of this document covers what exists today and the problems found in
review (§1), why push can't do `zip`/`chain` (§2), the options (§3), the proposed
design in detail (§4–§7), performance (§8), how Raphtory maps onto it (§9), a
migration plan (§10), and risks and open questions (§11).

---

## Implementation status (branch `pull-iter`)

| Increment (§10) | State |
|---|---|
| 1. Labels: `block`/`join`/`repeat`/`iterate` + `exit`/`goto`/`again` (+ `_if`/`_unless`) | **done** — `rust-lms/src/label.rs`, `func.rs`; `tests/test_labels.rs` |
| 2–5. Pull protocol as the required method of `StagedIterator`; every source and combinator a cursor; `for_each` provided | **done** — went straight to the merged trait (no temporary parallel trait); `opaque_for_each`/`reused_opaque_for_each` deleted from `Ctx` |
| 4. `zip` for any two iterators (shared counter when both indexed), `chain`, `take`, `skip` | **done** — `tests/test_pull_iter.rs` |
| 6a. `flat_map` (nested loops when consumed, state machine when pulled), `merge_by` / `merge` | **done** — `iter/flat_map.rs`, `iter/merge.rs`; tests in `tests/test_pull_iter.rs` |
| 6b. `peekable`, `intersect_sorted` | next |
| `flat_map` + `break_loop` | The push `for_each` `for_each` fixes the §5.7 caveat: a consumer's `break_loop` exits the inner loop, the open inner iterator is closed, and the break is re-raised on the outer loop, so `find_map`/`position`/`any` over a `flat_map` stop the whole traversal (tested 3 levels deep) |
| 7. Item representation (3-arg consumers, register tuples) | open (the 3-arg `Zip::for_each`/`Enumerate::for_each` remain) |
| §6.3 stage-0 `size_hint` | open |

Correctness gaps from §1.3 closed so far: `map`/`enumerate` keep random access
(`a.map(f).zip(b)` compiles and shares one counter); `i64` ranges zip;
`take_while` inside a `chain` ends only its half, and a downstream early exit
leaves the whole chain; every extern handle opened by a pipeline is dropped
exactly once on every exit; the `u64` range length no longer wraps for an empty
range (`range(5, 3).rev()` used to run ~2^64 times).

Where the implementation differs from the sketches below: value-less labels
are `ctx.block`/`ctx.exit` (not `join::<()>`/`goto(done, ())`); the random-access
probe is `Cursor::indexed_len`/`next_at` rather than a boxed `Indexed<T>` (§6.2
explains why); `IndexedStagedIterator` stays as the typed capability behind
`rev`, with `zip` moved off it.

Verified: whole workspace green with `--features llvm`, where every test body
runs on both Cranelift and LLVM (rust-lms 337 tests, sql-gen 113, rust-lms-std
28); emitted IR for slice sum and indexed zip
has the same loop as before, and filter adds only the one trivial jump predicted
in §8.1.

**Benchmarks** (`rust-lms-std/benches/graph_iter.rs`, Cranelift, 1M-node graphs,
master vs `pull-iter`; the untouched `native` baseline moves ±6% run to run, so
that's the noise floor):

- Two-level traversals (`sum`, `dst_gt_src`, `any_mod16`, `window`) are the
  same or faster everywhere: `csr`/`slice_ffi` −10% to −47%, `chunked` −5% to
  −23%, `unbuffered` −9% to +4%. The old loops' dead blocks after `break_loop`
  left 8–13 block parameters on loop headers; the cursor loops carry 3–4.
- Three-level `two_hop`: `csr`/`slice_ffi` −4% to −17%, `unbuffered` within
  noise. **Open regression:** `two_hop/uniform/chunked` is +14–15%
  (44.2 → 50.5 ms, three alternating paired runs). The branch's IR for that
  kernel is structurally simpler than master's, so the cause is below the IR
  (register allocation or block layout). The next step is a disassembly hook
  (Cranelift `set_disasm`) to compare machine code.

---

## 1. What's there today (review of `rust-lms/src/iter/`)

### 1.1 The model

```rust
pub trait StagedIterator: Sized {
    type Item: StagedType;
    fn for_each<F: FnOnce(&mut Ctx, Var<Self::Item>)>(self, ctx: &mut Ctx, consumer: F);
    // combinators: map, filter, scan, filter_map, take_while, skip_while, enumerate
    // terminals:   sum, count, count_if, sum_if, min, max, fold, any, all, position, find_map
}
pub trait IndexedStagedIterator: StagedIterator { type LenExpr; fn len(&self); fn rev(); fn zip(); }
pub trait IndexedSource: Clone + 'static { type Item; type LenExpr; type GetExpr;
                                           fn count(&self); unsafe fn get_at(self, Var<u64>); }
```

- **Sources own the loop.** `SliceIter::for_each` emits
  `while i < n { elem = s[i]; <consumer>; i += 1 }` and the other sources emit
  the same shape. `consumer` is an `FnOnce` called exactly once at staging time,
  so code size is linear in pipeline length.
- **Combinators wrap the consumer:** `Filter` wraps it in `if_then`, `Map` binds
  `f(x)` first, `Scan`/`Enumerate`/`SkipWhile` declare their state `Var` *before*
  calling `inner.for_each`, which hoists it out of the loop for free.
- **Invariant, written down in `traits.rs`:** combinators only add `if_then`s and
  never loops, so `break_loop` (in `take_while`, `any`, `all`, `position`,
  `find_map`) always targets the source's single loop. The whole design depends
  on this invariant, and it's the one that `chain` and a pull-style `filter`
  would break.
- **`zip` avoids the problem by being index-driven.** `Zip<I, S>` requires both
  sides to be `IndexedSource`. It computes `min(len_a, len_b)` once and runs one
  counter, with `get_at(i)` on each side. The inherent 3-argument
  `Zip::for_each(|ctx, a, b|)` passes both elements as separate `Var`s. The
  `StagedIterator` path goes through `ZipGetAt`/`Pair`, which writes a `ZipItem`
  to a stack slot.
- **`rev` is also index-driven:** it counts down over `IndexedSource::get_at`.

### 1.2 The sources, seen as step functions

| Source | What its loop header does | Pull step (`next`) it implies |
|---|---|---|
| `SliceIter` | `i < n` | `if i ≥ n → done; x = s[i]; i += 1` |
| `RangeIter` | `i < end` | `if i ≥ end → done; x = i; i += step` |
| `OpaqueIter` (`next`/`drop`) | `next(h) → COption`, `brif tag` | `o = next(h); if !tag → done; x = payload` |
| `ExactSizeOpaqueIter` | `i < len(h)` | `if i ≥ n → done; x = next_value(h); i += 1` |
| `ReusedOpaqueIter` | indirect `slot.next` → `COption` | the same, through the slot's mini-vtable |
| `ChunkedIter` (rust-lms-std) | `i == n` → refill; `break` if empty | `if i == n { if done → done; fill; if n == 0 → done }; x = buf[i]; i += 1` |
| `FromFn` | `next(ctx) → StagedOpt`, `None → break` | `next(ctx)` *is* the step |

All seven are pull iterators whose loop happens to be written inside
`for_each`. `ChunkedIter`'s own comment says the loop is "flattened — one loop
with a cold refill branch", which is exactly what the step shape looks like.

### 1.3 Problems found in review (independent of pull)

1. **`map` loses zip-ability.** `Map` implements `IndexedStagedIterator` but not
   `IndexedSource`, and `zip` requires `Self: IndexedSource`. So
   `a.staged_iter().map(f).zip(b)` and `a.zip(b.map(f))` don't compile, although
   `get_at(i) = f(inner.get_at(i))` would be trivial. The only thing
   `IndexedStagedIterator` adds over `IndexedSource` is a second `len`. Two
   traits carry the same capability, and only one of them is preserved by `map`.
   *Remove before you add:* fold `IndexedStagedIterator` into `IndexedSource`
   (or the reverse) and have `Map` and `Enumerate` implement random access.
2. **`zip` through the trait path materialises every pair.** `ZipGetAt`/`Pair`
   store both fields to a stack slot on each iteration and hand out the slot
   address. Cranelift has no SROA, so the stores stay. The inherent 3-argument
   `for_each` avoids this, but as soon as anything is chained after `zip`
   (`.map(|p| p.first() * p.second())`) the round-trip comes back. This matters
   more under pull, where `zip` becomes more common (§7).
3. **`Var<ZipItem>` aliases its slot.** Each `Pair` codegen site owns one stack
   slot, reused on every iteration. If a `ZipItem` is kept across iterations
   (a `scan` state, a "previous element"), it gets overwritten. This is latent
   today and would be hit by `peekable`/`merge` under pull. See §7.
4. **Missing combinators that pull makes trivial:** `take(n)`, `skip(n)`,
   `step_by`, `chain`, `peekable`, `flat_map`, `nth`, `last`, `dedup`, pairwise
   windows. `take(n)` and `skip(n)` are easy even in push and could be added now.
5. **`FromFn` documents `break_loop` inside `next` as a supported guard.** Under
   pull that becomes a bug if the `from_fn` is the left side of a `chain` (§5.4).
   Whatever design is chosen, the guard should jump to the iterator's own
   "exhausted" target, not the innermost loop.
6. **`ExactSizeOpaqueIter::count` is an inherent method that shadows the trait
   `count`.** That works, but only when called directly on the source; after
   `.map(..)` it falls back to the loop. A stage-0 `size_hint` (§6.3) would make
   O(1) `count` survive `map`/`enumerate`/`zip`.

---

## 2. Why push can't do `zip` and `chain`

### 2.1 `zip`

A push source *is* its loop. Zipping `a` and `b` means taking one element from
each on every iteration, so either:

- one loop runs inside the other (a cross product, not a zip), or
- both loops run as coroutines that yield to each other. There are no
  coroutines at stage 1: the output is a flat Cranelift CFG.

The only way out is to **invert all but one side into pull**: drive `a`'s loop,
and inside the body *pull* one element from `b`. The current `zip` does exactly
this, with the pull restricted to "random access at the shared index".
Generalising that pull from `get_at(i)` to `next()` is the whole idea. Zip needs
at least N−1 of its N inputs to be pull; nothing more.

### 2.2 `chain`

`chain(a, b).for_each(consumer)` looks easy in push:
`a.for_each(consumer); b.for_each(consumer)`. It has three problems:

1. **The consumer is `FnOnce` and is emitted at staging time.** Emitting it
   twice needs `F: Clone` all the way down the pipeline, including every
   combinator's closure, and it doubles the emitted body per `chain` (linear in
   the number of chained leaves, so tolerable, but it compounds with nesting).
2. **`break_loop` targets the innermost loop.** `chain(a, b).take_while(p)`:
   `take_while` breaks out of `a`'s loop, then `b`'s loop runs anyway, so the
   result is wrong. A fix needs an "iteration scope" separate from structural
   loops, i.e. labelled breaks.
3. **The alternative to duplication is a join-and-return:** emit the consumer
   once in a block, jump there from both loops, and jump back. "Jump back to the
   right loop" is a continuation index, so either a `switch` or a phase flag.
   That's a pull-style state machine under another name.

`chain` is therefore awkward in push even for two slices, and it's needed
constantly in Raphtory (memory layer ++ disk layer).

### 2.3 What push is genuinely better at

**Nested loops.** `outer.flat_map(|o| inner(o))` in push is
`outer.for_each(|o| inner(o).for_each(consumer))`: two natural loops, where the
inner one can be a tight, possibly vectorisable slice loop. In pull, `flat_map`
becomes a state machine: one loop, an "inner active" flag, and a retry edge
(§5.7). That costs a little per inner element and loses the tight inner loop.
Any design should keep the push `flat_map` path for pipelines that don't zip or
chain the flattened stream. This is also what strymonas (§3) found.

---

## 3. The options

This design space is well studied. Two papers are directly relevant:

- **Kiselyov, Biboudis, Palladinos, Smaragdakis, "Stream Fusion, to
  Completeness" (POPL 2017, the *strymonas* library).** A staged stream library
  in MetaOCaml/Scala LMS. Producers are either `For` (indexed: length + index)
  or `Unfold` (pull: step + termination test), and consumers are push. `zip`
  case-splits on the producer kinds: For×For uses a shared index, anything with
  an Unfold is converted to pull. Nested (`flat_map`) streams inside `zip` are
  handled by turning them into state machines. The "linear" (at most one
  element per step) versus "non-linear" (filter, flat_map) distinction is what
  decides whether a stream can be zipped cheaply. This is almost exactly the
  design recommended below, translated to Rust traits.
- **Shaikhha, Dashti, Koch, "Push versus pull-based loop fusion in query
  engines" (JFP 2018).** Shows that with staging, pull pipelines fuse as well as
  push pipelines. The only differences are `filter` (a pull filter needs an
  inner retry loop) and early termination. It also shows that pull is the one
  that supports merge-joins and `zip`. Relevant because the goal is a query
  engine.

### Option A: materialise external data, keep the current model

Collect every extern iterator into a buffer (an `SVec`, an Arrow array, or a
`ChunkedIter` chunk), then use indexed `zip`. `chain` becomes "concatenate into a
buffer".

- ✅ No library changes. Indexed zip stays optimal.
- ❌ An extra pass plus memory per list, which defeats the point of fusing
  Raphtory's lazy B-tree range scans (time windows). `chain` still isn't
  expressible over live streams. A merge of two sorted streams needs both fully
  materialised.
- Verdict: fine as a stopgap for small adjacency lists; not a framework.

### Option B: patch push (push `chain`, push × pull `zip`)

- Push `Chain`: `F: Clone` consumers, emitted once per leaf, plus a new
  "iteration scope" so `break_loop` leaves the whole chain.
- Mixed `zip`: generalise `IndexedSource` into `PullSource` (`next`-style), keep
  the left side push: `a.for_each(|x| { y = b.pull(done=loop exit); consumer(x, y) })`.
- ✅ Incremental; indexed paths untouched.
- ❌ Two different iterator concepts (push `StagedIterator` and pull
  `PullSource`), each with half the operations. `Clone` bounds leak into every
  combinator. `chain` of chains duplicates bodies. You can't `zip` two chains or
  two filtered streams (a filtered stream isn't a `PullSource`). It keeps growing
  special cases, which goes against "fewer, cleaner pieces".

### Option C: a second, parallel pull hierarchy

`PullIterator` with its own `PMap`, `PFilter`, `PZip`, … next to the push ones,
plus a blanket `impl<P: PullIterator> StagedIterator for P`.

- ❌ The blanket impl conflicts with `impl StagedIterator for Map<I, …>` under
  coherence as soon as `Map` also implements `PullIterator`. So the pull types
  would have to be separate structs: every combinator written twice, two sets of
  names, and conversions between them.
- Verdict: rejected. It's the duplication the project principles tell us to
  remove.

### Option D (recommended): pull is the protocol, push is the default driver

A single `StagedIterator` trait whose required method is the pull `open`/`next`,
with `for_each` as a **provided method** implemented once as "open, loop over
`next`, close". Every combinator is written once, in pull form, so `zip`,
`chain`, `merge_by`, … are available on every iterator. An iterator that has a
better push loop overrides `for_each`: `FlatMap` with nested loops, and
optionally the slice source. Default methods plus overriding give us exactly the
specialisation we need, with no nightly features. Random access (`IndexedSource`)
stays as an optional capability that `zip` and `rev` exploit through a stage-0
check (§6.2).

- ✅ One trait, one implementation per combinator, and every operation on every
  iterator. Terminals are untouched: they're all built on `for_each`.
- ✅ Single-source pipelines emit the same CFG as today (§8.1).
- ✅ Extern and Arrow sources compose freely. That's the Raphtory requirement.
- ❌ Needs new `Ctx` primitives (labels; §4). Every source and combinator is
  rewritten, but they're 10–60 lines each and there's one external implementor
  (`ChunkedIter`).
- ❌ Pull `flat_map`, `chain` and `zip` of non-indexed streams carry a small
  per-element cost (a flag test, or a second counter). §8 quantifies it and
  gives the mitigations.

The rest of this document designs option D.

---

## 4. The codegen primitive: typed, branded labels

Direct-style pull needs to jump *forward* (to "done", or to a join point with a
value) and *backward* (filter's retry) without opening a loop scope that
`break_loop` would bind to. `Ctx` only has structured `while_loop`, `if_then`,
`if_then_else` and `break_loop` today. The proposal is the WebAssembly structured
model, `block` + `loop` + `br`, typed and tied to the closure scope:

```rust
// rust-lms/src/label.rs
/// A forward jump target that receives a `T` (the merge-block parameters — a
/// phi). `Label<'s>` (`T = ()`) is a plain target. `'s` is an invariant brand
/// tying it to the scope that created it.
pub struct Label<'s, T = ()> { id: usize, … }   // Copy
/// A backward target: jumping to it re-runs the `repeat` body.
pub struct Again<'s> { id: usize, … }           // Copy

// rust-lms/src/func.rs
impl Ctx {
    /// The iteration loop: `'done: loop { body }`. `break_loop` inside targets
    /// the same exit as `exit(done)`.
    pub fn iterate<F>(&mut self, body: F)
    where F: for<'s> FnOnce(&mut Ctx, Label<'s>);

    /// Forward label, no value (Wasm `block`): `'out: { body }`.
    pub fn block<F>(&mut self, body: F)
    where F: for<'s> FnOnce(&mut Ctx, Label<'s>);

    /// Forward label carrying a value: every `goto(out, v)` inside `body`, plus
    /// `body`'s own fall-through result, lands on one merge block.
    pub fn join<T, E, F>(&mut self, body: F) -> Var<T>
    where
        T: StagedType + 'static,
        E: IntoStaged<T>, E::Staged: 'static,
        F: for<'s> FnOnce(&mut Ctx, Label<'s, T>) -> E;

    /// Backward label (Wasm `loop`, exiting by default). *Not* an iteration
    /// scope: `break_loop` inside passes through to the enclosing loop.
    pub fn repeat<R, F>(&mut self, body: F) -> R
    where F: for<'s> FnOnce(&mut Ctx, Again<'s>) -> R;

    pub fn goto<T, E>(&mut self, label: Label<'_, T>, value: E);   // break 'l value
    pub fn exit(&mut self, label: Label<'_>);                      // break 'l
    pub fn again(&mut self, again: Again<'_>);                     // continue 'l

    // One `brif` each, no merge block (what sources and filters use):
    pub fn exit_if<C>(&mut self, cond: C, label: Label<'_>);       // if c { break 'l }
    pub fn exit_unless<C>(&mut self, cond: C, label: Label<'_>);   // if !c { break 'l }
    pub fn again_if<C>(&mut self, cond: C, again: Again<'_>);      // if c { continue 'l }
    pub fn again_unless<C>(&mut self, cond: C, again: Again<'_>);  // if !c { continue 'l }
}
```

### 4.1 Understanding the primitives by example

#### They are Rust's labelled `break` and `continue`

The primitives look foreign, but each one is a Rust construct you already use:

| `Ctx` primitive | Plain-Rust equivalent | Meaning |
|---|---|---|
| `ctx.join(\|ctx, out\| … v)` | `'out: { … v }` (labelled block) | A block that produces a value |
| `ctx.goto(out, x)` | `break 'out x` | Leave the block early with `x` |
| `ctx.repeat(\|ctx, again\| … v)` | `'again: loop { …; break v }` | Code you can jump back to the start of |
| `ctx.again(again)` | `continue 'again` | Jump back to the start |
| `ctx.iterate(\|ctx, done\| …)` | `'done: loop { … }` | The one real iteration loop |

There are two differences from Rust, and both are the reason these primitives
exist:

- **`repeat` exits by default.** If the body reaches its end, execution
  continues after it; it only runs again when you call `again`. It's a "retry
  point", not a loop.
- **`join` and `repeat` are invisible to `break_loop`.** Only `iterate` (and
  `while_loop`) catch it.

#### `join` + `goto`: a block with several exits that share one result

```rust
// stage 0
let sign = ctx.join::<i64, _, _>(|ctx, out| {
    ctx.if_then(lt(x, 0i64), |ctx| ctx.goto(out, -1i64));
    ctx.if_then(gt(x, 0i64), |ctx| ctx.goto(out,  1i64));
    0i64                                   // the fall-through value also goes to `out`
});
// `sign: Var<i64>` here
```

The same thing in plain Rust:

```rust
let sign = 'out: {
    if x < 0 { break 'out -1 }
    if x > 0 { break 'out 1 }
    0
};
```

The Cranelift IR it emits (dead blocks omitted):

```
block0:  v1 = icmp slt x, 0 ; brif v1, block1, block2
block1:  jump out(-1)
block2:  v2 = icmp sgt x, 0 ; brif v2, block3, block4
block3:  jump out(1)
block4:  jump out(0)
out(sign: i64):                            ; block parameter = the phi
         ...
```

`join` creates one block (`out`) with a typed parameter. Each `goto` and the
fall-through jump there with a value, and after the closure returns the block is
sealed and code generation continues inside it. This is exactly what
`IfThenElse` does with its merge block today
(`append_value_block_params`/`jump_value`), except that the number of incoming
edges isn't fixed at two.

`Label<()>` carries no value; it's just a jump target. The iterator's `done`
label is one of these.

#### `repeat` + `again`: a retry point

Suppose you need to skip tombstones in an extern stream:

```rust
let x = ctx.repeat(|ctx, again| {
    let x = ctx.bind(unsafe { call_extern1_unchecked(next_raw, h) });
    ctx.if_then(is_tombstone(x), |ctx| ctx.again(again));   // retry
    x                                                         // otherwise continue with x
});
```

```rust
let x = 'again: loop {
    let x = next_raw(h);
    if is_tombstone(x) { continue 'again }
    break x;
};
```

```
block0:  jump head
head:    v1 = call next_raw(h) ; v2 = is_tombstone(v1) ; brif v2, block1, block2
block1:  jump head                         ; again
block2:  ...continue with x = v1           ; fall-through leaves the repeat
```

`head` is sealed when the closure ends, because every `again` back-edge has been
emitted by then.

**Why not `while_loop`?** `while_loop` is an iteration scope, so a `break_loop`
staged inside its body jumps to *its* exit. If `filter` used a `while_loop` to
retry, `slice.filter(p).any(q)` would break out of filter's retry loop instead of
the whole iteration, and `any` would keep scanning. `repeat` is transparent, so
`break_loop` goes past it to the real loop. This is exactly the bug that forces
the current push code to follow the rule "combinators never add loops".

#### `iterate` + `done`: the driver

```rust
ctx.iterate(|ctx, done| {
    let x = cursor.next(ctx, done);   // source jumps to `done` when exhausted
    consumer(ctx, x);                 // any/all/find_map may break_loop → the same exit
});
close.close(ctx);                     // reached by both exits → handles dropped once
```

`iterate` is `loop { … }` whose exit is available both as the `done` label and as
the `break_loop` target. Exhaustion and early termination leave through the same
block, which is what makes "drop the handle exactly once" structural.

#### All three together: `slice.filter(p).any(q)`

This is what the pieces stage into, with each part labelled by the iterator that
emitted it:

```rust
// driver (any's for_each)              // Rust equivalent
ctx.iterate(|ctx, done| {               // 'done: loop {
  // Filter::next
  let x = ctx.repeat(|ctx, again| {     //   let x = 'again: loop {
    // SliceCursor::next
    ctx.if_then(ge(i, n), |ctx|         //     if i >= n { break 'done }
        ctx.exit(done));
    let x = ctx.bind(s[i]);             //     let x = s[i];
    ctx.store(i, i + 1);                //     i += 1;
    ctx.if_then(not(p(x)), |ctx|        //     if !p(x) { continue 'again }
        ctx.again(again));
    x                                   //     break x };
  });
  // any's consumer
  ctx.if_then(q(x), |ctx| {             //   if q(x) {
    ctx.store(found, true);             //     found = true;
    ctx.break_loop();                   //     break 'done  ← skips 'again, as intended
  });                                   //   }
});                                     // }
```

`goto(done)` sits three closures deep, inside the slice's step, inside filter's
`repeat`, but it still reaches the driver's exit. That's the ability push
iterators lack: a source can say "I'm exhausted" from anywhere, and each
combinator decides what that means by choosing which label to pass down.

#### Why labels matter: `chain` in about 10 lines

```rust
let x = ctx.join(|ctx, got| {                        // 'got: {
    ctx.if_then(not(in_b), |ctx| {                   //   if !in_b {
        ctx.block(|ctx, a_done| {         //     'a_done: {
            let x = a.next(ctx, a_done);             //       let x = a.next() else break 'a_done;
            ctx.goto(got, x);                        //       break 'got x;
        });                                          //     }
        ctx.store(in_b, true);                       //     in_b = true;   // only via a_done
    });                                              //   }
    b.next(ctx, done)                                //   b.next() else break 'done  → falls to 'got
});                                                  // }
```

`chain` passes `a` a *private* done label (`a_done`) that means "switch to `b`",
and passes `b` the real `done`. Neither source knows it's inside a chain, and the
consumer is emitted once, after `got`.

#### The lifetime brand: why a label can't outlive its scope

The closure is `for<'s> FnOnce(&mut Ctx, Label<'s, T>)`, so the label's lifetime
is tied to that closure and can't be named outside it:

```rust
let mut leaked = None;
ctx.block(|_ctx, l| { leaked = Some(l); });   // error[E0521]: borrowed data escapes the closure
ctx.goto(leaked.unwrap(), ());
```

This matters because `join` seals its block when the closure returns. A `goto`
staged afterwards would add a predecessor to a sealed block, which is the #1
Cranelift panic in the CLAUDE.md invariants. The brand turns that runtime panic
into a compile error. Generated code only ever sees the label's plain id, as with
`bind_lt`.

#### Rules and edge cases

- **Code after `goto`/`again` is dead.** It's emitted into a fresh unreachable
  block, as `break_loop` does today, so a closure can still return its
  fall-through value afterwards.
- **Values leave a `join` only through the label.** `goto(out, v)` is typed
  (`v: IntoStaged<T>`), so every exit must supply a `T`. Don't define a var
  inside the join and read it afterwards on the assumption that all paths set it;
  route it through the label.
- **`break_loop` rule after the change:** it targets the innermost
  `iterate`/`while_loop` and passes through `join` and `repeat`.
- **No new `unsafe`, no backend changes.** Everything lowers to `create_block`,
  `append_block_param`, `jump`, `brif` and `seal_block`, which both backends
  already implement.

### 4.2 Lowering

**Codegen for each primitive** (stage 1, replaying actions in order, as
`while_loop` does):

- `join`: `merge = create_block(); append_value_block_params::<T>(merge);`
  register `id → merge` in a `labels` map on `CompilationContext` (next to
  `loop_exit_stack`); replay the body; `jump_value(merge, fallthrough)`;
  `switch_to_block(merge)`; `seal_block(merge)`. **Sealing happens after the
  body, so every predecessor has already been emitted.** The brand makes that a
  type error to violate: no `Label<'s, _>` can escape the `for<'s>` closure, so no
  `goto` to it can be staged after the block is sealed. This is the "seal only
  after all predecessors" invariant (the #1 cause of Cranelift panics) enforced
  by the Rust type system instead of by care.
- `goto`: `jump_value(labels[id], v)`, then switch to a fresh dead block, exactly
  as `break_loop` does now.
- `repeat`: `head = create_block(); jump(head); switch_to_block(head)`; replay
  the body; continue in the current block; `seal_block(head)` at the end (its
  back-edges are all inside the body).
- `block`: as `join`, with no block parameters.
- `exit_if`/`exit_unless`/`again_if`/`again_unless`: `brif cond, target, cont`
  (or the targets swapped), then continue in `cont`. One branch, no merge block
  and no negation instruction.
- `iterate` and `while_loop` share one lowering (`emit_loop` in `func.rs`):
  `while_loop` adds the header test; `iterate` has none and registers its exit
  block as the `done` label. Both push the exit on `loop_exit_stack`, so
  `break_loop` targets them and only them. (Re-expressing `while_loop` *as*
  `iterate` + `exit_unless` was rejected: it adds a block per loop.)

**The brand and the `'static` action queue.** Ctx actions are `'static` boxed
closures. The brand only exists on the stage-0 handle; the recorded action
captures the plain `usize` id, which is `'static`. This is the same split
`bind_lt` already uses (the checked type at stage 0, the erased value in the
queue).

**Both backends already support it.** Typed merge blocks exist today
(`IfThenElse` uses `append_value_block_params`/`jump_value`/`block_value`, fat
values included). Multiple jumps into one block and back-edges already exist
(`while_loop`). Nothing new is needed below `CompilationContext`.

**Why not keep "yield" as a continuation closure** (a
`next(ctx, on_some, on_none)` CPS step, like `StagedOpt::eliminate`)? Because
then `zip` calls `on_none` twice (a's exhaustion and b's) and `chain` calls
`on_some` twice. Both would duplicate code, or need `Clone` continuations.
Direct style with a label solves both problems: jumping to a label is cheap and
can be done from any number of staging sites, and the item comes back as a plain
`Var` in straight-line code. It's the same move `StagedOpt` made for
`filter_map`, one level up.

---

## 5. The trait design

### 5.1 Core traits

```rust
pub trait StagedIterator: Sized {
    type Item: StagedType;
    type Cursor: Cursor<Item = Self::Item>;

    /// Emit setup: bind handles, call producers, declare state vars. Runs once,
    /// before the loop that drives the cursor.
    fn open(self, ctx: &mut Ctx) -> Self::Cursor;

    /// (Not yet implemented: a stage-0 `size_hint`, §6.3.)

    /// Provided: the one push driver. Override for a better loop (FlatMap).
    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where F: FnOnce(&mut Ctx, Var<Self::Item>)
    {
        let cursor = self.open(ctx);
        let mut fin = None;
        ctx.iterate(|ctx, done| {
            let (x, close) = cursor.next(ctx, done);
            fin = Some(close);
            consumer(ctx, x);
        });
        fin.expect("next is staged exactly once").close(ctx);
    }

    // … all current combinators and terminals, unchanged in signature …
    // `zip`/`chain` take `IntoStagedIterator`, so a bare slice var still works:
    fn zip<B: IntoStagedIterator>(self, other: B) -> Zip<Self, B::Iter> { … }
    fn chain<B: IntoStagedIterator>(self, other: B) -> Chain<Self, B::Iter>
        where B::Iter: StagedIterator<Item = Self::Item> { … }
    fn take<N: IntoStaged<u64>>(self, n: N) -> Take<Self, N::Staged> { … }
    fn skip<N: IntoStaged<u64>>(self, n: N) -> Skip<Self, N::Staged> { … }
    // Planned (increment 6): flat_map, peekable, merge_by.
}

/// An opened iterator: one staged advance.
pub trait Cursor: Sized {
    type Item: StagedType;
    type Close: Close;

    /// Emit one advance. Either jumps to `done` (exhausted), or falls through
    /// with the next item bound. Consumes `self`, so it can be staged at most
    /// once, which keeps code size linear.
    ///
    /// Contract (fused): once `done` has been taken, the caller never runs this
    /// step again. Callers (driver, chain, zip, merge) guarantee it, so sources
    /// don't need a "fused" flag.
    fn next(self, ctx: &mut Ctx, done: Label<'_>) -> (Var<Self::Item>, Self::Close);

    /// Stage-0 random-access probe (§6.2). Asked by `zip` while it opens,
    /// before the loop, so any length code (a range's count) lands there.
    fn indexed_len(&mut self, ctx: &mut Ctx) -> Option<Var<u64>> { None }

    /// Element at an external index — only after `indexed_len` said `Some`.
    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<Self::Item>, Self::Close);
}

/// Releases resources. `Copy` (vars and extern refs only), because some
/// combinators emit it at more than one site (flat_map: on inner exhaustion
/// and at final close).
pub trait Close: Copy { fn close(self, ctx: &mut Ctx); }
impl Close for () { fn close(self, _: &mut Ctx) {} }
impl<A: Close, B: Close> Close for (A, B) { … }
```

Design decisions and the reasons for them:

- **`next` consumes the cursor.** "Staged at most once" is enforced by the type
  system, not by convention. Each combinator's `next` calls its inner `next`
  exactly once, so the emitted code is linear in pipeline size. That's the same
  guarantee `for_each(consumer: FnOnce)` gives today.
- **`next` returns the `Close`, `open` doesn't.** `flat_map` only learns its
  inner iterator's `Close` when it stages the inner `open`, and that happens
  inside `next` (the inner producer runs per outer element). Returning `Close`
  from `next` avoids stage-0 `Rc<Cell<…>>` plumbing. Sources that know their
  `Close` at `open` just carry it in the cursor.
- **The done label is passed in, not owned.** The *caller* decides what
  exhaustion means: end the loop (driver), switch to the right-hand side
  (`chain`), mark one side dead (`merge_by`). That's the difference that makes
  every combinator composable.
- **Items stay `Var<Item>`.** That keeps every current closure signature
  (`map(|x: Var<T>| …)`). §7 covers the representation of compound items such as
  zip pairs.

### 5.2 Why the consumer never ends up inside a combinator's structure

In push, `filter` puts the consumer *inside* its `if_then`. A pull `filter` puts
its retry logic *inside* `next`, and the consumer is emitted by the driver
*after* `next` returns, directly in the `iterate` body. So a terminal's
`break_loop` (`any`/`all`/`position`/`find_map`) always lands in the driver
loop, whatever the pipeline looks like. The "combinators never add loops"
invariant from `traits.rs` is no longer needed: combinators may use `repeat`
freely, because `repeat` isn't an iteration scope and the consumer is never
inside one.

### 5.3 Sources

```rust
// SliceIter
fn open(self, ctx) -> SliceCursor { let s = ctx.bind_lt(self.slice);
    let n = ctx.bind_lt(s.reborrow().len()); let i = ctx.var(0u64); SliceCursor { s, n, i } }
fn next(self, ctx, done) {
    ctx.if_then(not(lt(self.i, self.n)), |ctx| ctx.exit(done));
    let x = ctx.bind_lt(unsafe { self.s.reborrow().get_unchecked(self.i) }); // SAFETY: i < n
    ctx.store(self.i, add(self.i, 1u64));
    (x, ())
}

// OpaqueIter (next/drop)
fn open(self, ctx) -> OpaqueCursor { let h = ctx.bind(self.handle); OpaqueCursor { h, next, drop } }
fn next(self, ctx, done) {
    let opt = ctx.bind(unsafe { call_extern1_unchecked(self.next, self.h) }); // COption<Item>
    ctx.if_then(not(opt.is_some()), |ctx| ctx.exit(done));
    (ctx.bind(opt.payload_unchecked()), DropHandle { h: self.h, drop: self.drop })
}
// Close for DropHandle: emit drop(h). The driver emits it after the loop exit,
// which both `done` and `break_loop` reach. That's today's `opaque_for_each`
// guarantee ("drop on every exit"), now structural rather than special-cased.

// ChunkedIter: the body of today's for_each becomes next() almost line-for-line:
//   if i == n { if finished → done; fill; reload n/finished; i = 0; if n == 0 → done }
//   x = buf[i]; i += 1
// Close = "if !finished { drop(slot) }", exactly today's epilogue.

// FromFn: `from_fn(|ctx| -> impl StagedOpt)` stays; next = eliminate(Some → fall
// through with value via a join, None → goto done). The documented "break_loop
// inside next as a guard" changes to "return None" (or receive the done label).
```

`ExactSizeOpaqueIter` and `RangeIter` follow the slice pattern.
`ReusedOpaqueIter` reserves its slot in `open` (as it does now) and its indirect
`next` goes in `next`. `opaque_for_each` and `reused_opaque_for_each` in
`func.rs` become about 15-line cursors: they are hand-written `iterate` +
`next` + `close` today.

### 5.4 Combinators

Each of these is `next` only. `open` just opens the inner iterator and declares
state, and state vars are declared in `open`, i.e. hoisted, as today.

```rust
// map
fn next(self, ctx, done) { let (x, c) = self.inner.next(ctx, done); (ctx.bind((self.f)(x)), c) }

// filter: local retry, NOT a loop scope
fn next(self, ctx, done) {
    ctx.repeat(|ctx, again| {
        let (x, c) = self.inner.next(ctx, done);
        ctx.if_then(not((self.p)(x)), |ctx| ctx.again(again));
        (x, c)
    })
}
```

That `filter` has a subtlety: `self.inner.next` is staged **once**, inside the
`repeat`. The backward jump re-runs the same code at runtime, not at staging
time.

```rust
// filter_map: StagedOpt eliminated into a join
fn next(self, ctx, done) {
    let mut close = None;
    let v = ctx.join(|ctx, got| ctx.repeat(|ctx, again| {
        let (x, c) = self.inner.next(ctx, done); close = Some(c);
        (self.f)(x).eliminate(ctx, |ctx, v| ctx.goto(got, v), |ctx| ctx.again(again));
        unreachable_value()                  // fall-through is dead
    }));
    (v, close.unwrap())
}

// take_while: exhaustion IS the stop; no break_loop
fn next(self, ctx, done) {
    let (x, c) = self.inner.next(ctx, done);
    ctx.if_then(not((self.p)(x)), |ctx| ctx.exit(done));
    (x, c)
}

// skip_while: latch declared in open
fn next(self, ctx, done) { ctx.repeat(|ctx, again| {
    let (x, c) = self.inner.next(ctx, done);
    ctx.if_then(self.skipping, |ctx| ctx.if_then_else((self.p)(x),
        |ctx| ctx.again(again), |ctx| ctx.store(self.skipping, false)));
    (x, c) }) }

// scan / enumerate: state declared in open, updated after the inner step
// take(n):  if k >= n { goto done }; (x,c) = inner.next(done); k += 1
// skip(n):  repeat { (x,c) = inner.next(done); if k < n { k += 1; again } }
//           (on an indexed source, into_indexed turns skip/take into bound tweaks, §6.2)
// step_by(s): repeat { x = inner.next(done); j += 1; if (j-1) % s != 0 { again } }
```

`take_while` is simpler than today's (no `break_loop`), and it's now correct
under `zip` and `chain`: `chain(a.take_while(p), b)` stops `a` and moves on to
`b`, which a push `break_loop` could never do.

### 5.5 `zip`

```rust
fn next(self, ctx, done) {
    let (a, ca) = self.a.next(ctx, done);
    let (b, cb) = self.b.next(ctx, done);
    (ctx.bind(Pair::new(a, b)), (ca, cb))
}
// Close = (ca, cb): both are dropped after the loop. If `a` produced an element
// and `b` was then exhausted, `a`'s element is discarded, as in std::iter::Zip.
```

Any combination now works: extern × extern, extern × slice, `filter(..)` ×
`chain(..)`, `zip(zip(a, b), c)`. Indexed × indexed goes through the stage-0
fast path (§6.2) so it keeps today's single counter.

### 5.6 `chain`

```rust
// open: open BOTH sides (like std's Chain, which holds both iterators);
//       in_b = ctx.var(false).
fn next(self, ctx, done) {
    let (mut ca, mut cb) = (None, None);
    let x = ctx.join(|ctx, got| {
        ctx.if_then(not(self.in_b), |ctx| {
            ctx.block(|ctx, a_done| {
                let (x, c) = self.a.next(ctx, a_done); ca = Some(c);
                ctx.goto(got, x);
            });
            ctx.store(self.in_b, true);           // lands here only via a_done
        });
        let (y, c) = self.b.next(ctx, done); cb = Some(c);
        y                                           // fall-through → got
    });
    (x, (ca.unwrap(), cb.unwrap()))
}
```

- The consumer is emitted **once** (in the driver, after `got`). `a.next` and
  `b.next` are each staged once.
- Per element there's one well-predicted branch on `in_b`, plus the merge-block
  parameter (the phi) for `x`.
- Eager versus lazy open of `b`: opening both in `open` costs `b`'s producer
  call even when `a` short-circuits the whole stream. A lazy variant (open `b`
  under `a_done`, as `flat_map` does) avoids that. The default should be eager
  because it's simpler, with a lazy variant for expensive producers (such as a
  disk-layer scan).
- Close: close both unconditionally at the end. That's correct because each
  close is "drop the handle" and each handle was opened exactly once. Freeing
  `a` early at `a_done` is an optimisation that needs an `a_live` guard.

### 5.7 `flat_map`: pull form plus an overridden push `for_each`

The pull form is used whenever the flattened stream is zipped, chained or merged:

```rust
// open: outer cursor; o = uninitialised Var<O::Item>; active = ctx.var(false)
fn next(self, ctx, done) {
    let x = ctx.join(|ctx, got| ctx.repeat(|ctx, again| {
        let mut inner = None;
        ctx.if_then(not(self.active), |ctx| {
            let (ov, oc) = self.outer.next(ctx, done);   // outer exhausted → stream done
            ctx.store(self.o, ov);
            inner = Some((self.f)(self.o).open(ctx));    // producer runs per outer element
            ctx.store(self.active, true);
        });
        let inner = inner.unwrap();
        ctx.block(|ctx, inner_done| {
            let (y, ic) = inner.next(ctx, inner_done);
            ctx.goto(got, y);
        });
        ic.close(ctx); ctx.store(self.active, false);   // inner exhausted
        ctx.again(again);
    }));
    (x, FlatClose { outer: oc, inner: ic, active: self.active })
    // FlatClose::close = outer.close; if active { inner.close }
}
```

The push form overrides `for_each`:
`outer.for_each(ctx, |ctx, o| (f)(o).for_each(ctx, consumer))`. That gives
natural nested loops, and it's what today's `ReusedOpaqueIter` nested traversals
already emit. Because `for_each` is a provided method, `FlatMap` just
implements it. A pipeline that only *consumes* a `flat_map` gets nested loops,
and a pipeline that zips or chains it gets the state machine. That's strymonas'
linear/non-linear split, decided by Rust method dispatch.

**A caveat on the push override:** `break_loop` inside the consumer would then
only leave the inner loop. The override must run the inner loop inside an
iteration scope whose exit is the *outer* loop's exit, i.e. `iterate` needs to
expose its exit label, and `break_loop` must target the innermost `iterate`,
not any structural loop. With labels this is a one-line rule. Today it's the
reason `flat_map` couldn't be added safely.

### 5.8 `peekable`, `merge_by` and intersection (the query-engine operators)

```rust
// merge_by(a, b, le): stable merge of two sorted streams. State (in open):
//   ha, hb: Var<T>; need_a, need_b = true; a_live, b_live = true
fn next(self, ctx, done) {
    ctx.if_then(and(self.need_a, self.a_live), |ctx| {
        ctx.block(|ctx, pulled| {
            ctx.block(|ctx, a_done| {
                let (x, _) = a.next(ctx, a_done);
                ctx.store(self.ha, x); ctx.store(self.need_a, false);
                ctx.exit(pulled);            // got a head: skip the a_done tail
            });
            ctx.store(self.a_live, false);        // reached only via a_done
        });
    });
    /* same for b */
    ctx.join(|ctx, got| {
        ctx.if_then(and(self.a_live, or(not(self.b_live), le(self.ha, self.hb))), |ctx| {
            ctx.store(self.need_a, true); ctx.goto(got, self.ha) });
        ctx.if_then(self.b_live, |ctx| { ctx.store(self.need_b, true); ctx.goto(got, self.hb) });
        ctx.exit(done); unreachable_value()
    })
}
```

- Each input's `next` is still staged exactly once: the "need" flags defer the
  pull to the top of the next step, which avoids a priming call in `open` (a
  priming call would duplicate `next`).
- `ha`/`hb` are loop-carried item vars. For scalar items (timestamps, ids) they
  stay in registers. For `ZipItem`-style compound items they hit problem 3 from
  §1.3 (slot aliasing), so compound heads must be copied into owned storage. §7
  covers this.
- `peekable` is the one-sided version (`head`, `has_head`). `intersect_sorted`
  (for triangle counting and common neighbours) is a `repeat` that advances the
  smaller side until the heads are equal. `merge_join` for time-aligned property
  lookups is the same pattern.

None of these are possible in push. For Raphtory they're the core operators
(§9).

### 5.9 Terminals

All of them (`sum`, `count`, `count_if`, `sum_if`, `min`, `max`, `fold`, `any`,
`all`, `position`, `find_map`) stay as they are, because they're built on
`for_each`, and the provided `for_each` gives them the driver loop. Pull adds
one-shot terminals that need no loop: `first(ctx) -> (Var<T>, Var<bool>)`
(open, one `next` into a join, close) and `nth`/`last` via `skip`.

### 5.10 `rev`

`rev` still needs random access (or a double-ended source). It stays keyed on
`IndexedSource`. A new capability, `DoubleEndedCursor` with `next_back`, fits
Raphtory well: `BTreeMap::range` iterators are double-ended, so "latest N
updates before t" could reverse an extern stream through a `next_back` extern.
This is optional; see §11.

---

## 6. Keeping indexed sources fast

### 6.1 The cost to avoid

A generic pull `zip` over two `SliceIter`s runs two counters and two bound
checks (`i < n1`, `j < n2`), where today's indexed `zip` runs one counter against
a hoisted `min(n1, n2)`. LLVM's IndVarSimplify usually merges two identical
induction variables. Cranelift won't, which leaves one extra `add` and one extra
`cmp`/branch per element. That's small, but it's on the hottest Arrow paths, and
it also blocks the vectoriser on the LLVM backend when the bound checks aren't
merged.

### 6.2 Specialise at stage 0 (the LMS way)

Rust can't specialise `impl StagedIterator for Zip<A, B>` for indexed `A, B`.
**At stage 0 it doesn't need to**: the choice is ordinary Rust code running
while the kernel is built, and it costs nothing at runtime.

*As implemented*, the probe lives on the **cursor**, not on the iterator:

```rust
pub trait Cursor {
    …
    fn indexed_len(&mut self, ctx: &mut Ctx) -> Option<Var<u64>> { None }
    unsafe fn next_at(self, ctx: &mut Ctx, index: Var<u64>) -> (Var<Self::Item>, Self::Close);
}

// Zip::open
let mut a = self.iter.open(ctx);
let mut b = self.other.open(ctx);
let shared = match (a.indexed_len(ctx), b.indexed_len(ctx)) {
    (Some(len_a), Some(len_b)) => Some(SharedIndex {
        len: ctx.bind(ZipLen::new(len_a, len_b)),   // min, hoisted
        pos: ctx.var(0u64),
    }),
    _ => None,                                      // each side pulls
};
// ZipCursor::next: shared → exit_if(pos ≥ len); a.next_at(pos); b.next_at(pos); pos += 1
//                  none   → a.next(done); b.next(done)
```

Why the cursor rather than the `Indexed<T>` box first sketched here: the boxed
`get` closure would have to own the source, but a slice source is bound with
`bind_lt` and may carry a borrow (an `SVec` view), which a `'static` box can't
hold. The opened cursor already holds the bound slice, so `next_at` just loads
from it. No boxes, no lifetime erasure.

- **Random access, as implemented:** `SliceCursor` (its hoisted `len`),
  `RangeCursor` for `u64` *and* `i64` (the `RangeStep::count`/`nth` hooks; the
  count is only emitted when a zip asks), `RevCursor`, and the combinators that
  preserve it: `map`, `enumerate`, `take` (`min(len, n)`), `skip` (`index + n`)
  and `zip` itself (so nested zips share one counter). `Filter`, `FilterMap`,
  `TakeWhile`, `SkipWhile`, `Scan`, `Chain` and every extern source answer
  `None`.
- This fixes review problem 1: `a.map(f).zip(b)` compiles, and it is indexed.
- `IndexedStagedIterator` + `IndexedSource` remain as the *typed* capability
  behind `rev` (only `zip` moved off them).

### 6.3 `size_hint` at stage 0

A stage-0 `SizeHint::{Exact(Box<dyn Staged<Out = u64>>), Unknown}`, propagated by
`map`/`enumerate`/`zip` (min)/`chain` (sum)/`take`, gives O(1) `count` through
combinators (review problem 6). It also lets `collect`-style terminals
pre-size an `SVec`, and lets `ExactSizeOpaqueIter` keep its counted loop.

---

## 7. Item representation (zip pairs and loop-carried heads)

Pull makes compound items common: every `zip` yields a pair, and `merge` and
`peekable` keep items across iterations. Today a `ZipItem` is a `Var` holding a
**stack-slot address**, with one slot per staging site. That causes two problems:

1. **A store/load round-trip per element** (review problem 2). Cranelift keeps
   the stores.
2. **Aliasing:** a `ZipItem` stored into a loop-carried var (`merge_by`'s `ha`,
   `peekable`'s head, a `scan` state) points at a slot the next iteration
   overwrites.

Options, cheapest first:

- **(a) Keep the 3-argument consumers.** `Zip::for_each(|ctx, a, b|)` already
  exists; add the equivalent for `zip(..).map(|a, b| …)` and
  `enumerate(..).map(|i, x| …)` so that the common "zip then combine" pipelines
  never build a pair. It's cheap and removes most of the cost.
- **(b) Copy compound heads into owned slots.** `merge_by`/`peekable` allocate
  their own slot in `open` and copy into it. That fixes aliasing locally.
- **(c) Register tuples (the right fix).** The `value_refactor` branch is
  turning `Value` into a structured value (fat ptr+len first). Extending it with
  a `Tuple`/`Struct` of leaves would let a `ZipItem` (and any small `repr(C)`
  struct) live in registers across `Var`, `join` block params and `ctx.store`:
  no slot, no aliasing, and `first()`/`second()` become leaf projections. This
  is the long-term answer and the natural next step after `Fat`. Pull iterators
  don't *require* it, but they'll make it pay off.

Recommendation: ship pull with (a) and (b), and plan (c) as the `value_refactor`
increment after `Fat`.

---

## 8. Performance

### 8.1 Single-source pipelines: same CFG

- **Slice sum:** `iterate { if i ≥ n goto exit; x = s[i]; i += 1; acc += x }`.
  That's today's loop with the increment moved before the consumer (irrelevant
  for codegen) and `while (cond)` rotated into `if !cond goto exit`. Both lower to
  the same header branch. Checking with `RUST_LMS_DEBUG_IR=1` is part of
  increment 2 (§10).
- **Filter:** push emits `header → load → if p { body } → latch`. Pull emits
  `header → R: load → if !p goto R → body → latch`. It's the same set of blocks
  and edges, except that the rejection edge targets `R` instead of the header.
  Both are reducible natural loops. The only possible difference is one trivial
  `header → R` jump block, which LLVM's simplifycfg folds and Cranelift leaves
  as a fall-through.
- **Extern sources:** the cost is the per-item extern call in both models. Pull
  adds nothing.

### 8.2 Where pull costs

| Construct | Extra per-element cost vs ideal | Mitigation |
|---|---|---|
| `zip`, indexed × indexed | none (stage-0 fast path) | §6.2 |
| `zip`, stream × stream | none beyond the two steps | none needed; inherent |
| `chain` | one predictable branch on `in_b` + a phi | none needed. The push alternative duplicates the body. |
| `flat_map`, consumed | none (push override) | §5.7 |
| `flat_map`, zipped/chained | one `active` flag test per inner element; the inner loop isn't a tight natural loop | none needed; it's inherent to zipping a non-linear stream (strymonas says the same) |
| `merge_by` | two flag tests + a compare | this is the work a merge has to do anyway |
| compound items | slot store/load (today's problem too) | §7 |

### 8.3 Extern-bound paths

For Raphtory's B-tree layers the dominant cost is one extern call per element.
The existing measurements (see the `ChunkedIter` docs: chunked ≈ 1.4–1.9× faster
than per-item `next`) carry over unchanged, because `ChunkedIter` becomes a
cursor with identical code. With pull, **zip of two chunked streams** is
possible for the first time, costing two cold refill branches per element.

---

## 9. Mapping Raphtory onto this

Raphtory's storage splits into mutable in-memory structures (B-tree / sorted-vec
adjacency, `BTreeMap<TimeIndexEntry, _>` temporal props and edge additions) and
immutable Arrow-backed disk storage. A query engine has to combine them
constantly. Each case, with its operator:

| Query need | Sources | Operator (pull) |
|---|---|---|
| All edges of a node across layers or storage tiers | mem adjacency (extern) ++ disk adjacency (Arrow slice) | `chain` |
| Edge updates in time order across tiers | mem `BTreeMap` range (extern, sorted by t) and disk timestamps (Arrow, sorted) | `merge_by(|a, b| a.t <= b.t)` |
| Property value per update | timestamp column ⨝ value column (both Arrow) | indexed `zip` (fast path) |
| Property at an extern-produced time | extern time stream × Arrow value column | `zip` (mixed: one counter + one pull) |
| Time window `[t0, t1)` on a sorted stream | any sorted stream | `skip_while(t < t0).take_while(t < t1)`, or better, push the window into the extern producer (`BTreeMap::range`) |
| Common neighbours / triangles | two sorted adjacency streams (mem or disk) | `intersect_sorted` (a `repeat` step, §5.8) |
| Nested traversal (neighbours of neighbours) | extern × extern | `flat_map` (push override when consumed; pull when merged) |
| Latest-before-t | double-ended B-tree range | `DoubleEndedCursor::next_back` (optional, §5.10) |

Things to keep true while building the framework:

- **Push filters into producers.** A window or layer filter on the Rust side of
  an extern (`BTreeMap::range(t0..t1)`) beats any staged `skip_while`. The staged
  operators are for combining streams, not for scanning past data you could
  have skipped.
- **Items should be scalars or small structs.** Carry ids and timestamps through
  the pipeline and look up properties late (by index into Arrow). That keeps
  `merge`/`zip` heads in registers until §7(c) lands.
- **Prefer chunked externs for short, numerous lists** (adjacency): one call per
  list, as the existing benches show.
- **Resource safety comes from `Close`:** every extern handle opened in a
  pipeline is dropped exactly once on every exit path (exhaustion, `break_loop`
  from a terminal, `take_while`), including handles inside `chain`, `zip` and
  `flat_map`. This is structural, via `Close` after the driver loop, not
  something each source re-implements.

A sketch of what a query looks like:

```rust
// Updates of edge e in [t0, t1), merged across memory and disk, summed by value.
let mem  = mem_updates.iter2(g, e);                        // extern, sorted by t
let disk = disk_t.staged_iter().zip(disk_v)                // Arrow cols, indexed fast path
               .skip_while(move |p| lt(p.first(), t0));
let total = mem.merge_by(disk, |a, b| le(a.first(), b.first()))
               .take_while(move |p| lt(p.first(), t1))
               .map(|p| p.second())
               .sum(ctx);
```

---

## 10. Migration plan

Each increment is green on both backends (`tests/common` `for_each_backend`),
has integration tests in `rust-lms/tests/` in the existing style, and has its IR
checked with `RUST_LMS_DEBUG_IR=1`.

1. **Labels:** `Ctx::join`/`repeat`/`goto`/`again`/`iterate`, plus the `labels`
   map in `CompilationContext`. Re-express `while_loop` via `iterate`. Add a
   `compile_fail` test for label escape (the brand). Also test typed and fat join
   params.
2. **`PullIterator` alongside `StagedIterator`** (temporarily a separate trait,
   so there's no coherence clash): cursors for all seven sources, and a
   `drive(ctx, consumer)` helper. Compare the IR against the current `for_each`
   for slice, range, opaque, exact and chunked.
3. **Pull combinators:** `map`, `filter`, `filter_map`, `scan`, `enumerate`,
   `take_while`, `skip_while`, then the new `take`, `skip`, `step_by`. Property
   tests (proptest is already used in the workspace) comparing against
   `std::iter` on random inputs, on both backends.
4. **`zip` (general + the stage-0 indexed path), `chain`, `Close` composition.**
   Drop-count tests with extern iterators under `any`/`take_while`/`zip`/`chain`,
   extending `opaque.rs`'s `DropIter` tests to generated code.
5. **Merge the traits:** `StagedIterator` gets `open` as its required method and
   `for_each` as a provided one. Delete the push `for_each` bodies,
   `opaque_for_each`/`reused_opaque_for_each` in `func.rs`, and
   `IndexedStagedIterator` (folded into `into_indexed` + `IndexedSource`).
   Port `ChunkedIter` (the only external implementor; sql-gen doesn't use
   `StagedIterator`).
6. **`flat_map` (pull + push override, with `break_loop` bound to `iterate`),
   `peekable`, `merge_by`, `intersect_sorted`.** Graph-shaped tests and benches
   in `rust-lms-std/benches/graph_iter.rs`.
7. **Items:** 3-argument consumer variants now, and register tuples with
   `value_refactor` later (§7).

Increments 1–4 add code, and increment 5 removes more than 1–4 added. The end
state has fewer concepts than today: one iterator trait, one random-access
capability, no special-cased opaque loops in `Ctx`.

---

## 11. Risks and open questions

- **Lexical def-before-use of vars.** Codegen resolves staged vars by *replay
  order* (`CompilationContext::variables`), and the SSA backends
  (Cranelift `Variable`s, LLVM `alloca`s) handle the phis. `flat_map` opens its
  inner cursor inside an `if_then` and uses it after the merge. That's fine
  because the definition comes first lexically, and on the runtime path where it
  wasn't executed this iteration the value is loop-carried, guarded by `active`.
  On the entry path Cranelift's SSA builder materialises a zero and LLVM reads an
  uninitialised `alloca`. Both are harmless *because they're guarded*, but the
  rule should be written down: a cursor's state vars must be declared (or
  defined) lexically before every use, and join vars always get values through
  block parameters, never through a "declared but unassigned" var.
- **`break_loop` and scopes.** After the change, `break_loop` must mean "leave
  the innermost `iterate`", with `repeat` and `join` transparent to it. Audit
  user code that calls `break_loop` inside `from_fn` or `scan` closures:
  inside `chain` those should exhaust the sub-stream, not the whole pipeline.
  The fix is to give those closures the `done` label, or to have them return
  `None`.
- **Code size with `chain` of many streams and `merge` trees.** It's linear:
  every `next` is staged once, and joins deduplicate the tails. Still worth a
  test with, say, a 16-way merge to make sure compile time stays sane.
- **Eager versus lazy open in `chain`.** Default eager, with a lazy variant.
  Decide per use-case (disk scans probably want lazy).
- **Double-ended externs.** Worth adding only if Raphtory's "latest before t"
  shows up in hot queries; otherwise use the reverse-ordered producer.
- **Alternative considered for the step signature:**
  `next(ctx) -> impl StagedOpt` (the current `from_fn` shape). It's equivalent
  in power, but every consumer of the `StagedOpt` then has to eliminate it with
  two continuations, which brings back the duplication problem of §4. The label
  form is strictly simpler.
- **Soundness surface:** labels add no new `unsafe`. The brand prevents jumps
  into sealed blocks, and a jump to a label that isn't in scope can't be named.
  Unchecked element access stays where it is today, in sources behind
  `// SAFETY: i < n` comments, and moves from `for_each` bodies into `next`
  bodies without changing its argument.

---

### References

- O. Kiselyov, A. Biboudis, N. Palladinos, Y. Smaragdakis. *Stream Fusion, to
  Completeness.* POPL 2017. (strymonas; the For/Unfold producers, linear versus
  non-linear streams, zip case analysis.)
- A. Shaikhha, M. Dashti, C. Koch. *Push versus pull-based loop fusion in query
  engines.* Journal of Functional Programming, 2018.
- D. Coutts, R. Leshchinskiy, D. Stewart. *Stream Fusion: From Lists to Streams to
  Nothing at All.* ICFP 2007. (The step-function representation this pull design
  stages.)
- In this repo: `rust-lms/docs/staged_iterator_api.md` (the original push
  design), `rust-lms-std/src/chunked.rs` (a source already written in step
  shape), and `docs/Tidy Tuples and Flying Start…pdf` (Umbra's push
  produce/consume model, which is what sql-gen follows and what this proposal
  keeps as the default driver).
