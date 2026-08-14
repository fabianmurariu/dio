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
| Layout typing | `AccTy` + hand-indexed `i64`/`f64` cells | `RecordLayout` + `FieldId<T>` tokens + `DynamicRecord` (pointer hidden) |
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
Tuples           RecordLayout / FieldId / DynamicRecord      <-- NEW
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

## 4. The query-shaped record: `RecordLayout` + `FieldId<T>` + `DynamicRecord` ✅

The payload can't be a Rust `struct` — `sum(a)` vs `min(a),max(b),avg(c)` are
different shapes, known only at query time. Umbra doesn't use a struct either; its
Tuples layer emits `store(target + layout[slot].offset, value)` from a
**query-computed layout descriptor** (Fig 5, Lines 30-52).

Our version (built, `rust-lms-std::record`) goes one better than a raw offset: the
**set of fields is dynamic, but each field is a typed, layout-bound token**, and the
raw `*mut u8` never surfaces:

```rust
impl RecordLayout {
    fn field<T>(&mut self) -> FieldId<T>;                 // reserve aligned slot; typed token (carries layout brand + offset)
    fn record(&self, ctx, base, index) -> DynamicRecord;  // base + index*stride, branded
    fn wrap(&self, ptr: Var<SMutPtr<u8>>) -> DynamicRecord;// wrap an extern-returned record ptr, branded
}
impl DynamicRecord {                                       // hides the pointer entirely
    fn get<T>(&self, ctx, FieldId<T>) -> Var<T>;           // T inferred from token
    fn set<T>(&self, ctx, FieldId<T>, Var<T>);
}
```

Built per query by the loop we already run — a stage-0 dispatch on the arrow
`DataType` picks the `T`, and the heterogeneous tokens live in an enum:

```rust
let mut layout = RecordLayout::new();
let key = layout.field::<i64>();
for agg in aggs {
    let value = match acc_ty(agg) {                        // dynamic, per query
        AccTy::I64 => AggValueField::I64(layout.field::<i64>()),
        AccTy::F64 => AggValueField::F64(layout.field::<f64>()),
    };
    // keep the token(s) for this agg's fold/finalize
}
```

Then fold/finalize never touch a pointer — just tokens on a `DynamicRecord`:

```rust
let rec = layout.wrap(record_ptr);       // record_ptr came from group_upsert
let cur = rec.get(ctx, sum);             // sum: FieldId<f64> → Var<f64>
rec.set(ctx, sum, ctx.bind(add(cur, v)));
// rec.get::<i64>(ctx, sum)  -- won't compile (sum is FieldId<f64>)
// a token from another layout -> panics at stage 0 (a layout brand), never an OOB pointer op
```

Two guardrails, both free at runtime: **wrong type is a compile error**, and **wrong
layout is a stage-0 panic** (each `RecordLayout` carries a unique brand its tokens
check). The pointer math (`ptr_offset_mut` + `ptr_cast_mut`) is hidden inside
`DynamicRecord::get`/`set`. `RecordLayout` is the single owner of layout (offsets,
stride, later the null-bitmap offset).

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
| `ptr_cast::<T>(SMutPtr<u8>) -> SMutPtr<T>` on a **runtime** pointer (no-op) | `DynamicRecord::get`/`set` — reinterpret a runtime record ptr at a field's offset; the typed-pointer analogue of `opaque_ref` | tiny |
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
2. **✅ DONE — `RecordLayout` + `FieldId<T>` tokens + `DynamicRecord`.** The
   dynamic-but-typed record API (`rust-lms-std::record`, see §4): `field::<T>()`
   hands back a typed layout-bound token; `DynamicRecord::get/set(token)` hide the
   `*mut u8` (wrong type = compile error, wrong layout = stage-0 panic). sql-gen's
   GROUP BY state is a **single packed byte buffer** (one `[key | per-agg value
   (+count)]` record per group); `codegen::group_record` builds the tokens,
   fold/finalize call `rec.get/set` and never touch a pointer. IR shows
   `record = gidx * stride` then per-field offsets; the columnar `Vec<Vec<i64>>`
   slot buffers, `acc_get/set_*`, `group_slot_inits`, and `FieldHandle` are gone.
   (Row-wise packing landed without touching the hash map; keys/growth came in 3–4.)
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

4. **✅ DONE — O(groups) records.** The records buffer now grows with the group count
   (starts empty), not pre-sized to `num_rows` — the last O(rows) allocation is gone.
   Simpler than the "byte-stride SVec" first sketched: since the table is host-side
   (Phase 3), growth folds into the proxy. `group_upsert(&mut GroupState, u64) ->
   *mut u8` interns the key, appends one identity record when it mints a new group,
   and **returns that group's record pointer** — so the fold never bakes a records
   base that a grow could dangle (the pointer is used immediately, valid until the
   next `upsert`). Emit fetches the now-stable base once (`group_records_base`) and
   indexes with `layout.record`. `GroupHandle` is now a single baked `*mut GroupState`
   (the whole host state — table + growable records — like Umbra's table owning its
   tuples). IR: the fold does `rec = call group_upsert(...)` then field writes at
   offsets off `rec`; no baked base, no `gidx*stride` in the hot loop. Stress test
   `group_by_many_groups_grows_records` (1000 groups) exercises the realloc path.
5. **✅ DONE — string keys via `GroupKey`.** `GroupKey` gained `matches` (content
   equality) and `store(&mut BytesPool) -> Self` (copy variable data into the table's
   pool). `StrKey { ptr, len }` hashes/compares on **content** (long-string views
   differ per occurrence, so hashing the view is wrong) and `store` copies the bytes
   into a `BytesPool` **bundled with the `GroupTable`** — so the group state (and the
   result) survive the input batch being dropped, the contract a *stream* of batches
   needs (the user's reason for copying, verified by `group_by_string_key_survives_input_drop`).
   `GroupState` holds a non-generic `KeyTable = Int(GroupTable<u64>) | Str(GroupTable<StrKey>)`;
   only `group_upsert` vs `group_upsert_str(state, ptr, len)` differ, and the upsert
   externs now write the key into the record's leading field(s) themselves (so the
   fold only folds aggregates). The record's key is typed — `FieldId<i64>` for int,
   `FieldId<SPtr<u8>> + FieldId<u64>` for a string — and emit reads it back into a
   `ColVal::Str` through the existing output path. Codegen picks the int/string path
   from the key column's `DataType`; nullable keys still `NotImplemented`.
   **`Float64` keys — ✅ also done:** the kernel canonicalizes (`-0.0`→`+0.0`, any NaN→
   one canonical NaN — two `fcmp`+`select`) then `bitcast`s the f64 to `u64` bits and
   reuses the `Int` table (`KeyKind::Float` → `KeyTable::Int`); `KeyFields::Float(FieldId<f64>)`
   reads the bits back as f64 at emit. Needed one new rust-lms primitive `bitcast<TO,FROM>`
   (Cranelift `bitcast`, same-sized reinterpret). **Deferred:** composite (`[u64;2]` inline
   + pool spill), computed string keys (`upper(x)` — needs string functions).
6. **✅ DONE — nullable/null keys** (int/float/string, uniformly). A NULL key forms
   its own group (SQL semantics), kept **out of the hash table**: `GroupState` tracks a
   lazily-minted `null_gidx`, and `group_upsert_null` routes null-key rows to it (still
   a normal record slot, so the single emit loop covers it — required because our
   downstream continuation is single-use). The fold branches only when the key column
   is nullable (`if_then_else(valid, <type-specific upsert>, group_upsert_null)`), stores
   the validity in a **key-valid cell** (added to the record only for nullable keys —
   zero IR for non-nullable), and emit reads it to produce a NULL key. `intern` was
   refactored to `find_or_insert(probe, next_gidx)` so the null group's slot doesn't
   collide with hash gidxs (both draw from the records count).
7. **✅ DONE — composite (multi-column) keys** (fixed-width, approach A). A composite
   key is a **packed byte key** — it reuses the string path entirely (`Str` table,
   `group_upsert_str`, pool, the record's `(ptr, len)` key). The fold packs each key
   column's canonicalized bits (float: `-0.0`/NaN canonicalize + bitcast; int: bits) +
   a `u64` null bitmap into a **stack scratch** (`PackedKey` = a `RecordLayout` of
   8-byte cells + bitmap, written via `DynamicRecord`), and hands `(ptr, len)` to
   `group_upsert_str` (which copies to the pool on a miss). Emit **unpacks** the pooled
   packed key back into N typed `ColVal`s; nulls come from the bitmap, so each
   `(a, b, …)` combination (nulls included) is its own group — no `null_gidx` for
   composite. Two new rust-lms primitives: `stack_alloc(size) -> SMutPtr<u8>` (runtime
   stack scratch — broadly useful) and `ptr_as_const` (`*mut`→`*const` for the extern).
8. **✅ DONE — string columns inside a composite key** (variable-length). Fixed-only
   composites keep the fast stack path (7); a composite with **any string column** uses
   a **host key-builder**: the kernel pushes each column's bytes into a reusable scratch
   (`group_key_reset`, `group_key_push_u64` for the bitmap + fixed cols, `group_key_push_bytes`
   for a string's `[len | content]`), then `group_upsert_composite` interns the assembled
   flat byte key (still the `Str` table's content hash/eq + pool). Emit **unpacks** the
   pooled flat key with a **running byte offset** (a string advances the cursor by
   `8 + len` at runtime). **Deferred:** the `[u64; 2]` inline fast path (skip the pool for
   ≤2 small fixed-width columns).

Done 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8. Keys are now Int32/Int64, Float64, Utf8View (each
nullable), and composites of any mix (fixed fast path, strings via the builder). Next:
the **parallel partial/final split** (the `[keys | payloads]` state is already the
mergeable unit), or computed keys once we have string functions.

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
