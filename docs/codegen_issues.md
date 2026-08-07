# Codegen issues (deferred)

Known inefficiencies in the emitted Cranelift IR. These are **not correctness
bugs** — results are verified correct — but are worth revisiting when we optimize.

## 1. Redundant loop-invariant accumulator-pointer reloads

**Where:** grouped-aggregate fold loop (`codegen::gen_grouped` / `GroupedAgg::fold`),
via `PrimitiveArrayView::get` + `set`.

**Symptom (IR):** for each accumulator fold, `get` and `set` independently
re-derive the column base — the output descriptor pointer is loaded twice per fold
(e.g. `load v48+64` for both the read and the write), and the `col * size_of(FfiArray)`
offset is recomputed each time. The accumulator column's `values.ptr` is
**loop-invariant**, but it stays inside the per-row loop: conservative alias
analysis assumes the data store to `out[col][gidx]` may alias the descriptor load,
so Cranelift does not hoist it (LICM) or CSE the duplicate loads.

**Cost:** ~2 avoidable loads per row × per accumulator, plus a recomputed offset.

**Fix ideas:** compute each accumulator's base pointer once per row (share between
`get`/`set`), or better, hoist the per-column base pointers out of the fold loop
(they are established before the loop and never change). Relatedly, the group-key
column is stored on **every** row (`out[0][gidx] = key`) even for a repeat key —
idempotent but a redundant store; `group_intern` could return an "is-new" flag so
the key is written only for new groups.

## 2. Dead `iconst.i8 0` in the grouped fold block

**Where:** the fold block (`block4`) of the grouped-aggregate kernel.

**Symptom (IR):** an unused `v.. = iconst.i8 0` (a `bool`/`i8` constant) is emitted
between the group-key store and the first aggregate fold. It is dead code
(Cranelift DCEs it), so it costs nothing at runtime, but nothing in
`gen_grouped`/`fold`/`get`/`set` should emit an `i8` there — the legitimate `i8 0`
is the `AND` predicate's `select` false-arm, which lives in the filter block.

**Why it matters:** an unexplained constant usually means some staged op is
materializing a value it shouldn't. Worth tracing to its source (candidates: the
`if_then` branch staging, or a `column_mut`/`get`/`set` path) to confirm it's
benign and not a symptom of a larger redundancy.
