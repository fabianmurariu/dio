# Path to an Umbra-style GROUP BY

A staged, **typed** plan to evolve our GROUP BY from today's columnar slot buffers
into Umbra's design — row-wise packed aggregate records in a growable hash table,
with the hot loop generated and only allocation/growth behind an extern. Along the
way we grow a small **`rust-lms-std`** toolbox (dynamic array, arena, hash map) that
joins, sorts, and `DISTINCT` will reuse.

Guiding rule, unchanged: **simple, beautiful, typed APIs.** The record shape is
query-dependent, so it can't be a Rust `struct` — but every *leaf* access stays a
typed pointer. No `u64`-as-pointer anywhere.

---

## 0. Where we are → where we're going

| | Today (`docs/group_by.md`) | Target (this doc) |
|---|---|---|
| Aggregate storage | columnar: one `Vec<i64>` **per slot**, indexed by `gidx` | **row-wise**: one packed record per group |
| Key → group | `group_intern` (Rust `HashMap<i64,u32>`), fixed key type | host `hashbrown::HashTable<Entry<K>>` via a per-row proxy, **key-generic** (`GroupKey`) — the paper's model, faster to extend to composite/string keys |
| Memory | `O(rows)` — every buffer sized to the row count | `O(groups)` — records grow with the group count |
| Layout typing | `AccTy` + hand-indexed `i64`/`f64` cells | `RecordLayout` + `FieldHandle<T>` typed field accessors |
| Reusability | group-by-specific | `SVec` / `BytesPool`(→`SArena`) reused by join/sort/distinct |

> **Direction note (Phase 3):** we deliberately *keep the hash table host-side* (a
> per-row proxy `find_or_insert`), rather than generating a staged inline probe.
> That is what the "Tidy Tuples" paper actually does (Fig 5's `insert` is a proxy),
> it's far simpler, and `hashbrown::HashTable`'s raw hash/`eq` hooks are exactly the
> German "specialize to the key" mechanism — so complex keys are new `GroupKey`
> impls, not a rewrite. Inlining the probe stays a possible *later* optimisation,
> not a goal.

The performance rationale (fold-loop cache locality, `O(groups)` memory) is in
`docs/group_by.md §9`. This doc is the *how*.

---

## 1. Two axes — keep them separate

We conflated these once; the plan depends on holding them apart.

- **Axis A — where work runs:** *generated* staged code vs. *precompiled* host code
  called through an extern. Umbra generates **hash, key-compare, pack, payload
  update**; it calls a **proxy** for **insert, allocation, growth** (Fig 5, Line 25:
  `insert(ht, h, size) -> Ptr<UInt8>`).
- **Axis B — how typed the authoring is:** raw `u64`/`Ptr<u8>` vs. typed handles.

`rust-lms` already gives us both ingredients: `#[extern_fn]` **is** Umbra's proxy
system (Axis A), and `SPtr`/`SMutPtr`/`SRef<Opaque<T>>` **are** its "fully typed
view of C++ classes" (Axis B). **`rust-lms-std` is the Axis-B toolbox.** It does not
change *where* work runs — `SVec::push` still calls an extern to grow — it changes
*how typed our code is* when we wire the pieces together.

This maps our stack onto Umbra's layers (their Fig 4):

```
Operators        sql-gen: plan.rs / codegen.rs
Data Structures  rust-lms-std: SVec, SArena, SHashMap        <-- NEW
Tuples           RecordLayout / FieldHandle (pack + hash)    <-- NEW
SQL Values       sql-gen: ColVal / Nullness
Codegen          rust-lms core: Var, Ctx, typed pointers, control flow
```

---

## 2. The one hard idea: handle indirection

Everything works today because we **bake the buffer pointer** as a Cranelift
constant. That is exactly why it can't grow: reallocation moves the buffer and every
baked constant dangles.

The fix — the enabler for *all* growable staged structures — is to bake a pointer to
a **stable control block**, never to the volatile buffer:

```rust
#[derive(StagedType)]                 // #[repr(C)]; lives at a fixed host address
struct RawVec<T> { ptr: *mut T, len: u64, cap: u64 }
```

- The kernel holds `SRefMut<Opaque<RawVec<T>>>` (or a baked `*mut RawVec<T>`).
- Every access **reloads `ptr` from the block** (`load_field`) — one extra load.
- Growth is an `#[extern_fn]` that reallocs and writes `ptr`/`cap` back.
- The block never moves, so its baked pointer is valid for the whole run.

That single indirection is the entire trick. Nothing else below is conceptually
hard.

---

## 3. `rust-lms-std` — the toolbox

Three types, in dependency order. Each is independently useful and independently
testable.

### 3a. `SVec<T>` — reallocating dynamic array (typed, homogeneous)

The `RawVec<T>` handle above, plus staged ops:

```rust
impl<T: StagedType> SVec<T> {
    fn len(&self, ctx) -> Var<u64>;
    fn get_unchecked(&self, ctx, i: Var<u64>) -> Var<T>;            // load(ptr + i)
    fn set_unchecked(&self, ctx, i: Var<u64>, v: Var<T>);          // store(ptr + i, v)
    fn push(&self, ctx, v: Var<T>);   // if len==cap { grow ext }; reload ptr; ptr[len]=v; len++
}
```

Use it wherever data is **homogeneous** and you hand out **indices, not long-lived
pointers**: the hash **directory** (`SVec<u32>`), index/selection vectors, a column
you're materializing. Genuinely generic over `T`.

### 3b. `SArena<T>` — chunked, append-only, stable element pointers

A typed generalization of the existing `BytesPool` (`rust-lms/src/pool.rs` — a
"stable-pointer bump arena": grows by adding chunks, **never moves** existing
elements). Use it wherever you take a pointer *into* the structure and keep using it:
hash-table **entries**, string data.

```rust
impl<T: StagedType> SArena<T> {
    fn push(&self, ctx, v: Var<T>) -> SMutPtr<T>;   // stable pointer to the slot
    fn get(&self, ctx, i: Var<u64>) -> SMutPtr<T>;  // by append index
    fn len(&self, ctx) -> Var<u64>;
}
```

The stability contract is the difference that matters: `SVec` reallocates (fast
indexing, no pointer stability), `SArena` never moves an element (pointer stability,
chunked). A hash map needs **both**.

### 3c. `SHashMap<K, V>` / `SAggTable` — open, chained, growable

Composed from the two above:

```rust
struct SHashMap<K, V> {
    dir:     SVec<u32>,           // power-of-two buckets, head index (NIL = empty)
    entries: SArena<Entry>,       // stable; chained by INDEX so rehash never moves payloads
}
```

`Entry` header is `[hash: u64][next: u32][key…][payload…]`. The **one primitive**
group-by needs:

```rust
fn find_or_insert(&self, ctx, key: /*K*/) -> SMutPtr<u8>;   // -> the entry's payload region
```

Inline: `h = hash(key)`; `head = dir[h & mask]`; staged `while` walking `next`,
comparing `(hash, key)` (`while_loop` + `break_loop` + `eq`, all present); on hit
return `&mut payload`; on miss append an `Entry` (arena `push`) and link it. Check the
load factor **at entry** and rehash the **directory only** first (entries never move),
so growth never happens mid-walk — and the returned payload pointer stays valid.

Note `find_or_insert` returns `SMutPtr<u8>` (Umbra's `insert -> Ptr<UInt8>`), **not**
a typed payload struct — because the payload is query-shaped. That's §4.

> **Reuse:** `SHashMap` with *multiple* entries per key (don't dedup) is a hash-join
> build side; probe is the same `hash + walk`. `SVec`/`SArena` are the storage for
> sort runs, `DISTINCT`, and set ops. This toolbox is why the investment pays off
> beyond GROUP BY.

---

## 4. The query-shaped record: `RecordLayout` + `FieldHandle<T>`

The payload can't be a Rust `struct` — `sum(a)` vs `min(a),max(b),avg(c)` are
different shapes, known only at query time. Umbra doesn't use a struct either; its
Tuples layer emits `store(target + layout[slot].offset, value)` from a
**query-computed layout descriptor** (Fig 5, Lines 30-52). We already have the seed
of this in `group_layout`.

The typed version — **the set of fields is dynamic, each field is typed**:

```rust
struct RecordLayout { stride: usize }
struct FieldHandle<T> { offset: usize, _t: PhantomData<T> }   // T known in Rust, offset per-query

impl RecordLayout {
    fn field<T: StagedType>(&mut self) -> FieldHandle<T>;      // reserve size_of::<T>(), bump stride
}

impl<T: StagedType> FieldHandle<T> {
    fn at(&self, entry: impl Staged<Out = SMutPtr<u8>>) -> SMutPtr<T>;  // entry + offset, typed
}
```

Built per query by the loop we already run (a stage-0 dispatch on the arrow
`DataType` picks the `T` — this is today's `dispatch_prim!` / `AccTy`):

```rust
let mut layout = RecordLayout::new();
for agg in aggs {
    let field = match acc_ty(agg) {                 // dynamic, per query
        AccTy::I64 => AnyField::I64(layout.field::<i64>()),
        AccTy::F64 => AnyField::F64(layout.field::<f64>()),
    };
    // keep `field` for this agg's fold/finalize
}
```

Then fold/finalize read and write **typed leaves** off a type-erased record:

```rust
let entry: SMutPtr<u8> = table.find_or_insert(ctx, key);   // payload region
let cur = ctx.bind(load(sum.at(entry)));                    // sum.at(entry): SMutPtr<f64>
ctx.emit(store(sum.at(entry), ctx.bind(add(cur, v))));
```

The **record as a whole** is type-erased (it must be — the shape is the query); the
**leaves stay typed**, the same discipline as today's `acc_get_i64`/`acc_get_f64`,
just at `entry + offset` instead of `base[gidx]`. `offset` is an integer; the pointer
is `SMutPtr<T>`. This is the "dynamic struct" — and it's the API we like.

`RecordLayout` becomes the single owner of layout (offsets, stride, later the null
bitmap offset), the way `CompilationContext::slice_data_ptr` is the single owner of
slice layout. One place, no re-derivation.

---

## 5. GROUP BY on the new stack

Fold (per input row, all generated):

```
h     = hash(key)                       // inline: mul/xor/shr
entry = table.find_or_insert(h, key)    // inline walk; extern only to append/grow a NEW group
for each agg:                           // typed field updates into the packed record
    fold agg into <field>.at(entry)     // combine_i64 / combine_f64 as today
```

Emit (once per group): iterate the entry arena, read the key + each `<field>.at(entry)`,
`finalize` into a manifested `Row`, `yld` downstream. `avg` still divides at emit;
per-group nullability rides in a `seen`/`count` field of the record (a packed null
bitmap can come later, like Umbra's `nullIndicator`).

What deletes: `group_intern`/`group_len` externs, the `Vec<Vec<i64>>` columnar
buffers, and the `num_rows`-sized allocation. What replaces them: one `SAggTable`
sized to groups.

---

## 6. What `rust-lms` core needs (small)

Most of the hot path already exists (`add/sub/mul`, `bitand/bitor/bitxor`, `shl/shr`,
`eq/lt/gt`, `select`, `min/max`, `int_cast`, `while_loop`/`break_loop`, `load`/`store`,
`ptr_offset`/`array_index`, `load_field`, `#[derive(StagedType)]`). Genuinely new:

| Need | Why | Size |
|---|---|---|
| `ptr_cast::<T>(SMutPtr<u8>) -> SMutPtr<T>` on a **runtime** pointer (no-op) | `FieldHandle::at` — reinterpret a runtime entry ptr; the typed-pointer analogue of `opaque_ref` | tiny |
| `bitcast` `f64 ↔ u64` | hashing / storing `Float64` keys | small |
| `alloc`/`realloc`/`free` extern convention | `SVec`/`SArena` growth (the cold path) | trivial |
| staged hash (`mul`/`xor`/`shr` finalizer; byte loop for strings) | inline hashing | small; ops already exist |
| atomics (`cmpxchg`) | **later** — parallel partial/final build | deferred |

`rotate`/`crc32` (Umbra's exact hash, their Table 1) are **not** needed — a
multiply-shift/murmur finalizer over existing ops is enough.

---

## 7. Phased plan

Each phase builds, tests, and ships green on its own.

1. **✅ DONE — `SVec<T>` + control-block + grow extern.** New `rust-lms-std` crate
   (`RawVec` control block, `HostVec<R>` owner, `SVec<T>` typed kernel handle,
   `svec_grow` extern). Round-trip tests push N, force grows (`cap` 0→4→8→…), and
   read back through reallocation; IR confirms the buffer pointer is reloaded from
   the control block after the grow branch. Core additions this phase:
   `ptr_cast`/`ptr_cast_mut` (reinterpret a runtime pointer's element type — the
   typed analogue of `opaque_ref`); `FieldRefOf`/`PointerLike` for the `RustPtr`
   tag (so `field_addr`, hence field *writes*, work on baked pointers); `CopyType`
   for `SPtr`/`SMutPtr` (so a pointer field can be loaded/bound to a `Var`).
2. **✅ DONE — `RecordLayout` + `FieldHandle<T>`.** The dynamic-but-typed record API
   (`rust-lms-std::record`): `RecordLayout::field::<T>()` reserves a typed field,
   `record(ctx, base, i)` gives `base + i*stride`, `FieldHandle::{at,get,set}` are
   typed leaf accesses on a `*mut u8` record. sql-gen's GROUP BY state is now a
   **single packed byte buffer** (`GroupState.records: Vec<u64>`, 8-aligned, one
   `[key | per-agg value (+count)]` record per group), pre-filled with an identity
   template (`group_template`); `codegen::group_record` builds the layout, fold/emit
   go through `layout.record` + field handles. Still pre-sized to rows and still
   using `group_intern` — row-wise packing landed *without* touching the hash map.
   IR shows `record = gidx * stride` then per-field offsets; the columnar
   `Vec<Vec<i64>>` slot buffers, `acc_get/set_*`, and `group_slot_inits` are gone.
3. **✅ DONE — host `hashbrown::HashTable` group table (reroute).** We reconsidered
   the "staged inline hash map" (old phases 3–5) and chose the paper's *actual*
   design: the table stays **host-side, behind a per-row proxy** (`insert` is Umbra's
   Fig-5 proxy call), and we generate the key + the fold. `GroupTable<K>` is now
   `hashbrown::HashTable<Entry<K>>` (the *raw* Swiss-table API where we supply
   hash + `eq` per op — the German "specialize to the key" hook), keyed via a small
   `GroupKey` trait; the key is stored **in the entry** (decoupled from the records
   buffer), value = the dense `gidx`. `group_find_or_insert(&mut GroupTable<u64>,
   u64) -> gidx` replaces `group_intern`. `K = u64` only for now (int columns,
   reinterpreted from the signed key — a no-op cast). No staged inline table is
   built; `SArena` is **deferred to string keys** (its honest use). Records buffer,
   `RecordLayout`, and everything downstream are unchanged.

   > **Why the reroute:** an open-addressing/`hashbrown` table keyed on `gidx` needs
   > no stable-pointer arena, and per-row proxy `find_or_insert` is exactly what the
   > paper does for the build/aggregation side. Simpler, faithful, and key-generic —
   > composite/string keys become new `GroupKey` impls, not a rewrite.

4. **O(groups) records.** Make the records buffer grow with the group count instead
   of pre-sizing to `num_rows` — a byte-stride variant of `SVec`'s control-block/grow
   (base reloaded per row, so the baked pointer survives a grow). Removes the last
   O(rows) allocation. `group_find_or_insert` grows the buffer when it mints a new
   `gidx`.
5. **Typed / complex keys via `GroupKey`:** `Float64` (bitcast to `u64` bits),
   composite (`[u64; 2]` inline fast path, else a bundled key-pool), then **string
   keys** — a `BytesPool` **bundled with the table**, entry storing a stable
   `(offset, len)`, `hash`/`eq` on *content* (long-string views differ per
   occurrence, so hashing the view is wrong; the ≤12-byte inline view is a fast
   path). This is where `SArena`/`BytesPool` earns its place.
6. **Nullable/null keys**, then the **parallel partial/final split** (the
   `[keys | payloads]` state is already the mergeable unit).

Done in order 1 → 2 → 3. Next: Phase 4 (O(groups) records) or jump to Phase 5
(string keys) — both build only on what's already green.

---

## 8. Invariants to keep (so it stays simple)

- **Typed leaves always.** The query-shaped record is type-erased; every field access
  is a `SMutPtr<T>`. Never thread a bare address as a pointer.
- **One owner of layout.** `RecordLayout` owns offsets/stride/null-offset; nobody
  re-derives them (cf. the slice ptr/len invariant).
- **Proxy only on the cold path.** Generate hash, walk, compare, pack, update; call an
  extern only for allocation/growth/append — exactly where Umbra draws the line.
- **Stability contract is explicit.** `SVec` reallocates (indices only); `SArena`
  never moves (pointers OK). Pick per use; a hash map needs both.
- **Remove before adding.** When Phase 5 lands, the columnar buffers and interning
  externs *go* — the new path replaces them, it doesn't sit beside them.
