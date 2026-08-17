# Joins

Status: **plan**. Builds on the streaming/multi-table foundation
(`docs/table_scan.md`, Phases 1–5) and reuses the GROUP BY host-table machinery.
Single-partition, no parallelism.

> **Prerequisite:** revert the ouroboros `CodegenCtx` experiment back to the
> working `*const u8` baseline first — a borrowed pool can't live in the `'static`
> clonable push closures, and `cx` drops at codegen while the baked literal
> addresses must survive to `as_fn`. The join work builds on `codegen/`, which
> doesn't compile until that's reverted.

---

## 1. The shape: hash join — build the LEFT, probe the RIGHT

A hash join has a **build** side (fully materialized — a pipeline breaker) and a
**probe** side (streamed). Decisions (locked with the user):

- **Build the LEFT input, probe the RIGHT.** In `A JOIN B` we materialize `A` into
  a **build relation** and stream `B` past it.
- **Output stays `[left | right]`** (the join's schema order): per probe (right)
  row we look up the matching left rows and emit `[left cols (lifted from the
  located build row) | right cols (the live probe row)]`.
- The build relation is a **materialized relation kept as Arrow `RecordBatch`es**
  (not a bespoke packed buffer), so a future disk/mmap-backed store can slot in
  (§2). The hash table indexes *rows* of it, it does not copy values out.

### The hash index

```
key  ->  Vec<(batch_idx: u32, row_idx: u32)>          // a MULTIMAP
```

The key is the equi-join column value; the value is every build-relation row
(located by `(batch, row)`) that has that key. This is the GROUP BY table
(`GroupTable<u64>` over `hashbrown` + `ahash`) generalized from "key → one dense
gidx" to "key → a run of row locators." Phase 1 uses a `u64` key (Int columns);
later key types reuse `GroupKey`/`StrKey` verbatim (§ Phases).

SQL null semantics: a `NULL` key never matches (even `NULL = NULL` is false in an
equijoin), so null-keyed build rows are **not indexed** and null-keyed probe rows
find nothing — no null group, unlike GROUP BY.

---

## 2. The build relation: a trait, with an in-memory `Vec<RecordBatch>` today

```rust
/// A materialized relation the probe reads located rows out of. Behind a trait so a
/// future spill-to-disk / mmap-backed store slots in without touching codegen.
pub trait BuildRelation {
    fn num_batches(&self) -> usize;
    /// FFI descriptors for batch `i` (stable for the whole probe). The probe rebuilds
    /// a `&[FfiArray]` from this and `gen_read`s the located row's columns.
    fn batch_descriptors(&self, i: usize) -> *const FfiArray;
}

/// Phase-1 impl: build side is a bare `Scan`, so we just clone its input RBs
/// (Arc bump — cheap) and precompute their descriptors.
pub struct InMemoryRelation {
    batches: Vec<RecordBatch>,        // cloned build-input batches (or materialized output)
    descs: Vec<Vec<FfiArray>>,        // per-batch descriptors, valid while `batches` is retained
}
```

Locator `(batch_idx, row_idx)` — two `u32`s packed into a `u64`. Caps: 4 B batches,
4 B rows/batch. The build relation is retained for the whole probe (the pipeline
breaker); the *probe* side still streams and drops one batch at a time.

**Future (not now):** an mmap/`BuildRelation` impl backed by Arrow IPC files on disk,
for build sides too large for memory. Keeping the relation as RBs behind the trait
is what makes that a drop-in later.

---

## 3. Building the relation

Two build modes, split across phases:

- **Bare-`Scan` build (Phase 1) — host-side, no JIT.** When the build (left)
  subtree is a plain `Scan`, drain its stream host-side: **clone each RB** into
  `InMemoryRelation.batches`, read the key column (Int) row by row, and insert
  `(batch_idx, row_idx)` under each non-null key. No build kernel — the whole build
  is Rust reading Arrow arrays. Fast and simple; this is why Phase 1 requires a bare
  scan on the left.
- **General build subtree (later) — JIT.** When the left subtree has a
  `Filter`/`Project` (or is itself a join/aggregate), its *output* must be computed
  and materialized. That means a compiled **build kernel** (or an in-kernel
  build-loop before the probe-loop) that runs the subtree and appends surviving/
  projected rows into fresh RBs, indexing keys as it goes. The `BuildRelation` +
  hash-index interface is identical; only how the relation gets filled changes.

Because the build happens before the probe, and (Phase 1) entirely host-side,
`gen_op(Join)` in Phase 1 **codegens only the probe** over the right subtree, with
the finished `JoinState` baked in.

---

## 4. Probe: streaming, codegen (`codegen/join.rs`)

`gen_op(Join { left, right, on, .. }, …, yld)` (Phase 1) = build the left host-side,
then:

```text
gen_op(right, …, |ctx, right_row| {                 // ordinary streaming scan of the right
    let key = eval(right_key_expr, right_row);       // Phase 1: a column read
    let (base, count) = join_probe(state, key);      // key's locator run (count 0 = no match)
    let mut i = 0;
    while i < count {
        let (b, r) = unpack_locator(base + i);
        let descs = join_left_batch(state, b);        // extern: BuildRelation descriptors for batch b
        let left_batch = slice(descs, ncols_left);
        let left_cols = (0..ncols_left).map(|c| gen_read(left_batch, c, r));   // REUSE gen_read
        yld(ctx, concat(left_cols, right_row));       // [left | right]
        i += 1;
    }
});
```

Value lifting reuses `gen_read` straight against the located batch's descriptors —
no pack/unpack layer. The match loop is a staged `while i < count` (we have
`while_loop`/`break_loop`/`ptr_is_null`). Right (probe) columns are the live row's
`ColVal`s; left columns are lifted from the located build row.

---

## 5. Host state & externs (`sql-gen/src/join.rs`)

```rust
pub struct JoinState {
    relation: InMemoryRelation,                 // the build relation (§2)
    index: GroupTable<u64>,                      // reused; entry value = Vec<u64> locators (multimap)
    // outer-join bookkeeping (Phase 3): a "matched" bit per indexed build row
}
```

Externs (`#[extern_fn]`, mirroring the `group_*` family; all pointers typed):
- Phase 1 build is host-side (a plain Rust method on `JoinState`, called from
  `run_operator` — no extern needed).
- `join_probe(&JoinState, key…) -> *const u64` + its `count` — the located run for a
  key (null/0 = no match).
- `join_left_batch(&JoinState, batch_idx) -> *const FfiArray` — descriptors for a
  build-relation batch, for the probe's `gen_read`.

`CodegenCtx` gets a baked `JoinState` pointer (like `group: Option<Rc<GroupHandle>>`);
a `Vec` of handles once we allow more than one join per query (Phase 4).

---

## 6. Lowering (`sql.rs`)

`LogicalPlan::Join` → `Operator::Join { left, right, on: Vec<(Expr, Expr)>,
join_type: JoinType, schema }`:
- Phase 1: `Inner` only; exactly one equijoin pair `(left_col, right_col)` of Int
  type; left subtree must be a bare `Scan`. Reject the rest with `NotImplemented`.
- A post-join `WHERE` already lowers to `Operator::Filter` above the join, so
  `… ON a.k=b.k WHERE …` works for free.
- `plan.rs`: `Operator::Join { left: Box<Operator>, right: Box<Operator>, on, join_type, schema }`; `output_schema` = the carried `schema`.

---

## 6a. Optimizer (predicate pushdown)

`sql.rs` runs a curated pair of datafusion optimizer rules (54.1.0, our version)
between `SqlToRel` and `lower()`: `ExtractEquijoinPredicate` (populates `join.on`)
and `PushDownFilter` (pushes a `WHERE` into a join's inputs). So the natural
`SELECT … FROM a JOIN b ON a.k=b.k WHERE a.x > 1` pushes `a.x > 1` into the **build**
side (it becomes `Filter(Scan)` → materialized), instead of filtering the join
output — no derived-table contortion needed. `PushDownFilter` may move predicates
into `TableScan.filters`; since we don't execute pushed-down filters natively,
`lower(TableScan)` re-applies them as a `Filter` above the scan. (A full **optd**
Cascades integration is a later milestone — `optd-core` is datafusion-free, but its
df connector is pinned to 53.1.0, so it needs its own df54→optd IR + optd→Operator
lowering.)

---

## 7. Phased plan

1. **Inner equi-join, single Int key, bare-scan left build.** ✅ **Done.**
   Host-side build (`join.rs`): clone left RBs into `InMemoryRelation` (behind the
   `BuildRelation` trait), index the Int key column into a `HashMap<u64,
   Vec<locator>>` multimap (locator = `(batch_idx<<32)|row_idx`). Probe (right) is a
   JIT kernel (`codegen/join.rs`): per right row compute the u64 key → `join_probe_
   count`/`join_probe_base` externs → `while i<count` match loop → `join_left_batch`
   descriptors + `gen_read` the located left row → emit `[left | right]`. One join
   per query; `JoinState` baked in `CodegenCtx.join`. Lowering extracts the equijoin
   pair from `join.filter` (raw `SqlToRel` leaves `on` empty). Null keys never
   indexed / probe skipped on a null key. Tests (`tests/joins.rs`, 7): multiplicity,
   no-match/empty-build → empty, negative keys, multi-batch build+probe, null keys
   never match, join-then-filter.
2. **All key types + composite keys.** Reuse GROUP BY wholesale — `Float64`
   (canonicalize → bitcast → u64), `Utf8View` (`StrKey`, content hash/eq, bytes in a
   pool), and multi-column `ON` (packed byte key). Still inner, still bare-scan
   build.
3. **General build subtree (materialized build).** ✅ **Done.** `run_operator` was
   split into `run_kernel` (compile+run+materialize one kernel over `&mut Inputs`) +
   a join orchestrator: a bare-`Scan` left still clones its input RBs (cheap), but a
   `Filter`/`Project`/derived-table left is *run* via `run_kernel(left, …)` and its
   output materialized into one RB, which `JoinState::build_int` indexes. Both feed
   the same `BuildRelation`. `SubqueryAlias` lowers transparently (derived tables).
   Tests: `build_side_filter_is_materialized`, `_multi_batch`, `_empty_result`.
   (Still Int key / inner — keys are Phase 2.)
4. **Outer / semi / anti joins.** `LEFT`/`RIGHT`/`FULL` (a `matched` bit per build
   row, swept after the probe to emit unmatched build rows with NULL-padded probe
   columns; NULL-pad the other way for unmatched probe rows), `SEMI`/`ANTI`.
5. **Plan generality.** More than one join (a `Vec` of baked `JoinState` handles),
   join → GROUP BY, join → join.
6. **Later / optional.** Build-side selection (materialize the smaller input),
   spill/mmap `BuildRelation`, sort-merge join, parallelism.

---

## 8. Explicitly deferred

- Non-equi / range joins (`ON a.k > b.k`) and `ON` residual filters.
- Build-side cost/size selection — always build the left for now.
- Spilling / mmap of the build relation (the trait is the seam for it).
- Parallelism — single-partition build and probe.

---

## 9. Reuse summary

| Need | Reused from |
| --- | --- |
| Key hash / compare / interning | `group::{GroupKey, StrKey, GroupTable}` (hashbrown + ahash) |
| String-key byte copies (Phase 2) | `BytesPool` + the composite-key bytes builder |
| Two input streams + table ids | `Inputs` / `ScanStream` / `exec_jit_multi` |
| Reading located build rows | `codegen::gen_read` against `FfiArray` descriptors |
| Host table behind a baked pointer + externs | `group::GroupState` pattern + `#[extern_fn]` |
| Probe / match-loop primitives | `while_loop` / `break_loop` / `ptr_is_null` |
| Streaming probe | the push model (`gen_op` + `for_each_batch`) |

New pieces are small: the `BuildRelation` trait + in-memory impl, the multimap
(key → run of locators), the host-side bare-scan build, and the per-match probe loop
that `gen_read`s located rows.
