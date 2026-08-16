# Table scan, streaming input, and multi-table execution

Status: **plan**. This is the foundation the JOIN milestone builds on. It replaces
the "one borrowed `RecordBatch` in, one pre-sized `RecordBatch` out" model with a
**host-driven stream of batches** that the JIT kernel pulls from, and a **growable
output** it pushes into — so a scan over 1 Bn rows never lifts the whole table into
memory (except where a pipeline breaker like a JOIN build side deliberately does).

Parallelism, partial/final aggregation splits, JOIN hashing, and pipeline breakers
are **out of scope here** — noted where they attach, deferred to the parallelism
milestone. Everything below is single-partition.

---

## 1. Where we are today

The whole query is one kernel over one batch:

```rust
// run.rs
let prepared_in = prepare_record_batch(rb);          // Vec<FfiArray> borrowing rb
let mut out = PreparedOutput::alloc(schema, capacity); // capacity = max_output_rows(rb.num_rows())
let f = compiler.fun2("query",
    |ctx, batch: Var<SRef<Slice<FfiArray>>>, sink: Var<SRefMut<Slice<FfiArray>>>|
        gen_collect(ctx, batch, sink, &op, &out_schema, &cx));
let n = compiled.as_fn()(prepared_in.arrays(), out.as_ffi_mut());
```

```rust
// codegen.rs — gen_scan
fn gen_scan<B: BatchSource>(ctx, batch, schema, yld) {
    let len = gen_len(ctx, batch, schema.field(0).data_type()); // column 0's element count
    let i = ctx.var(0u64);
    ctx.while_loop(lt(i, len), move |ctx| {
        let row = /* read every column at i */;
        yld(ctx, row);                                          // push upward
        ctx.store(i, add(i, 1u64));
    });
}
```

Three things break for streaming + joins:

1. **One batch.** `batch` is a single `&[FfiArray]`. A table is a *stream* of them.
2. **One table.** There is exactly one input param. A JOIN needs two sources.
3. **Pre-sized output.** `capacity = max_output_rows(rb.num_rows())` needs the total
   input row count up front. A stream doesn't have one until it's exhausted.

What already works in our favour:

- **The push model is untouched by any of this.** `Filter`/`Project`/`Aggregate`
  receive rows through `yld` and don't care where the loop came from. Only
  `gen_scan` changes shape.
- **Cross-batch state is already register-scoped.** Scalar-agg accumulators are
  `ctx.var`s created in the `Aggregate` arm *before* `gen_op(input)` emits the scan
  loop, so they already live *outside* the loop. Wrapping the row loop in an outer
  "next batch" loop keeps them exactly where they are — in registers, spanning the
  whole stream. `GroupState` is host-side and baked; folding across batches is just
  more `upsert` calls. **No operator above the scan needs to change.**
- **`GroupState` is the precedent** for a host object the kernel drives through
  `#[extern_fn]`s. The input stream and the output builders are the same idea.

---

## 2. Target model: the kernel drives the batch loop

The decision (over driving batches one-by-one from the host into a re-entrant
kernel): **the `while next_batch` loop lives inside the JIT kernel.** The host owns
the batch *iterator* and the *ownership/lifecycle*; the kernel owns the *control
flow*. One `as_fn()` call runs the entire query over the entire stream.

Why this shape:

- **State stays in registers across batches.** An unfiltered `count(*)` becomes
  `count += batch_len` at the top of each batch iteration — no per-row loop at all.
  A `sum`/`min`/`max` accumulator is a register threaded through every batch. The
  host never round-trips per batch, so there's nothing to spill and reload.
- **Bulk shortcuts are expressible.** Because the scan sees `batch_len` before the
  row loop, whole-batch fast paths (count, `SELECT count(*)`, later SIMD-friendly
  column ops) are a codegen choice, not an ABI change.
- **The operator tree above the scan is unchanged.** The outer loop wraps the inner
  row loop; `yld` still fires once per surviving row.

Kernel skeleton after the change (single table shown; §5 generalizes):

```text
fn query(inputs: &mut Inputs, out: &mut OutCols) {
    // ── accumulators / group-table pointer: registers, live across ALL batches ──
    let mut i;
    loop {
        let descs = scan_next(inputs, TABLE_0);   // extern: &[FfiArray] or null-sentinel
        if descs.is_null() { break }              // stream exhausted
        let len = descs[0].len();                 // this batch's row count
        // ── whole-batch fast path OR per-row loop ──
        i = 0;
        while i < len { /* read row i, push downstream → out.push_* */ ; i += 1 }
        // `descs` (and the prev RecordBatch) is dropped by the NEXT scan_next
    }
    // scalar/group emit happens here, after the stream — appends to `out`
}
```

---

## 3. Host side: `ScanStream` and batch ownership

Input is `Box<dyn Iterator<Item = RecordBatch>>` per table (a `Vec<RecordBatch>`
is the trivial impl; a file/exec reader is the real one). One partition per stream
for now.

```rust
/// One table's batch stream + the single live batch's FFI descriptors.
/// The kernel pulls from it via `scan_next`; it owns the drop lifecycle.
pub struct ScanStream {
    iter: Box<dyn Iterator<Item = RecordBatch>>,
    /// The batch currently exposed to the kernel. Dropping it frees that
    /// RecordBatch's Arrow buffers. Exactly ONE input batch is alive at a time
    /// (per stream) on the streaming path.
    current: Option<RecordBatch>,
    /// Stable, reused descriptor buffer refilled in place each `next`. Its address
    /// is handed to the kernel; the kernel reads columns out of it.
    descs: Vec<FfiArray>,
    /// Pipeline-breaker hook (JOIN build side, §7). When set, consumed batches are
    /// moved here instead of dropped, so their buffers outlive the probe.
    retained: Option<Vec<RecordBatch>>,
}

impl ScanStream {
    /// Advance to the next batch. Returns the descriptor slice, or `None` at end.
    /// Drops (or retains) the previous batch first — so on the streaming path only
    /// one input RecordBatch is resident per stream.
    fn next(&mut self) -> Option<&[FfiArray]> {
        // release / hand off the previous batch
        if let (Some(prev), Some(keep)) = (self.current.take(), self.retained.as_mut()) {
            keep.push(prev);            // JOIN build side: keep buffers alive
        } // else: prev dropped here → Arrow buffers freed
        let rb = self.iter.next()?;     // None → stream exhausted
        refill(&mut self.descs, &rb);   // prepare_record_batch, in place, no realloc
        self.current = Some(rb);        // keep this batch's buffers alive for the kernel
        Some(&self.descs)
    }
}
```

The drop point is the single most important line: **overwriting `self.current`
frees the previous batch.** A filter or group-by over a billion-row table holds at
most one input batch per stream at any instant. A JOIN sets `retained = Some(_)`
on its build side (only), so those buffers survive into the probe.

`Inputs` holds one `ScanStream` per table (indexed by a small table id assigned at
lowering time):

```rust
pub struct Inputs { streams: Vec<ScanStream> }
```

### FFI contract

`scan_next` returns the descriptor slice as an FFI-safe fat pointer, with a
null pointer as the "exhausted" sentinel (`FatSlice<FfiArray> { ptr: null, len: 0 }`):

```rust
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn scan_next(inputs: &mut Inputs, table: u64) -> FatSlice<FfiArray> {
    match inputs.streams[table as usize].next() {
        Some(descs) => FatSlice::from_slice(descs),   // ptr + len (= n columns)
        None => FatSlice::from_raw_parts(null(), 0),  // exhausted
    }
}
```

Staged view:
- `&mut Inputs` → `SRefMut<Opaque<Inputs>>` (a real `&mut`, pointer-sized).
- `table` → a **stage-0 constant** (the scan's table id), so multi-table dispatch
  costs nothing at runtime.
- return `FatSlice<FfiArray>` → a staged fat slice; `ptr == null` ends the loop,
  otherwise it reconstructs `SRef<Slice<FfiArray>>` (the existing `BatchSource`),
  and the *rest of the read path is unchanged*. `len` is the column count; the row
  count is still `gen_len` over column 0 (an empty mid-stream batch → `len` rows =
  0, loop body runs zero times — **not** confused with end-of-stream).

No bare `u64`-as-pointer crosses the boundary: `Inputs` rides as a typed opaque
`&mut`, the descriptors ride as a typed `FatSlice<FfiArray>`.

---

## 4. Output side: `SVec`-backed growable columns

Pre-sized output is gone (no total row count up front). Output becomes one growable
column per output field. The **fixed-width columns are `SVec`s** (`rust-lms-std`),
so the append is *inline JIT code* — no FFI per value. Only the variable-length and
cold cases (strings, growth) touch an extern.

### Why `SVec`, not an extern-per-value

The naive design appends every value through an extern
(`out_push_i64(out, col, v)`). That is a real C call **per value, per column, per
row** — caller-saved registers spilled, args marshaled, and the call is an
optimization barrier (Cranelift can't keep the cursor in registers across it, can't
hoist, can't vectorize). A projection of M columns over N rows pays N·M
non-inlinable calls.

`SVec::push` (`rust-lms-std/src/svec.rs`) is all inline Cranelift IR instead:

```text
load len; load cap; if len==cap { call svec_grow }   // branch usually NOT taken
data = *ctrl.ptr; store data[len] = v; len += 1       // ~6 instrs, no call
```

The only FFI call is `svec_grow`, and it fires **only when `len==cap`** — amortized
O(1), effectively never on the hot path. So `SVec` turns a *guaranteed* per-value C
call into ~6 inline stores plus an *amortized-rare* grow call.

This is exactly what `SVec`'s **handle indirection** was built for: the kernel bakes
a pointer to the stable `RawVec` control block (not the movable buffer), and `push`
reloads `data` from the block after a possible grow. So the output buffer can
realloc mid-stream **without dangling the baked pointer** — which is precisely the
"we don't know the total row count up front" problem this section exists to solve.
`SVec` fixes the growable-output problem *and* removes the per-value FFI in one move.

### Shape

```rust
/// One growable column per output field, in output-schema order. Host-owned; the
/// kernel holds a baked `SVec<T>` handle per fixed-width column (control-block ptr
/// + the `svec_grow` extern) and a `StringViewBuilder` handle per string column.
pub struct OutCols { cols: Vec<OutCol> }

enum OutCol {
    /// Fixed-width value column: a typed `HostVec<T>` (i32/i64/f64). Nullable
    /// columns pair it with a validity column (byte-per-row first, bit-pack later).
    Fixed { values: HostVec, validity: Option<HostVec /* u8 */> },
    /// Variable-length: the existing arrow-lms sink. Strings stay on the extern.
    Str(StringViewBuilder),
}
```

- **Fixed-width value append → `SVec::push`** (inline, per §"Why `SVec`"). Codegen
  already dispatches per `Prim` type in `write_col`, so each column maps cleanly to
  its monomorphized `SVec<i32|i64|f64>`.
- **Validity append → a parallel `SVec<u8>`** (1 byte/row: `push 1` valid / `0`
  null). Simplest first cut; bit-pack later without touching the value path.
- **Strings → `StringViewBuilder` append extern** (variable-length view + byte pool,
  not a fixed-stride `SVec`). Less hot, already host-side — keep it on the extern.

`write_col` (today: store into a pre-sized Arrow buffer at row `n`) becomes: on the
column's `OutCol`, branch on `Nullness` as now — non-null → `values.push(v)` (and
`validity.push(1)` if nullable); nullable-null → `values.push(⊥)` +
`validity.push(0)`; string → the append extern.

### Finalize (near zero-copy)

Each `HostVec<T>` owns a plain `Vec<T>` of primitives, so
`arrow::Buffer::from_vec` / `ScalarBuffer::from(vec)` **adopts it without a copy**
for numeric columns — no per-row copy at the end either, if `HostVec` exposes its
`Vec`. The validity `SVec<u8>` finalizes to a `NullBuffer` (packing bytes→bits
here, once, O(rows)). The host then assembles the result `RecordBatch`.

This **unifies every emit**: scan/filter/project rows, the single scalar-agg row,
and the O(groups) GROUP BY rows all append to `OutCols`. `PreparedOutput`,
`max_output_rows`, and the `capacity` computation are deleted.

### Where the win lands

- **Scan / Filter / Project pass-through** (high output volume): the big win — this
  is the unbounded-output case this section worried about, and every emitted value
  drops from a C call to ~6 inline instructions.
- **GROUP BY** (O(groups), emitted once after the fold) and **scalar agg** (1 row):
  negligible — those emits aren't hot, but they share the same `OutCols` path for
  free.

> **Chunking (near-term, not blocking):** `OutCols` can flush a fixed-size result
> `RecordBatch` every N rows to a host collector instead of growing unbounded, so a
> huge filtered result streams out too. The kernel side is identical (still `SVec`
> pushes into a control block the host swaps at the flush boundary); only the host
> growth policy changes. Start with grow-only; add the flush when a result is large
> enough to matter.

---

## 5. Multiple tables

Each `Scan` is lowered against a specific table, so it carries a **table id** (index
into `Inputs.streams`):

```rust
// plan.rs
Operator::Scan { table: usize, schema: SchemaRef }
```

`gen_scan` bakes that id as the `table` constant in its `scan_next` call. With a
single input the kernel drives one stream; a JOIN (§7) drives two. The kernel ABI is
the two typed params `fun2(inputs: &mut Inputs, out: &mut OutCols)` regardless of
table count — table arity lives inside `Inputs`, not in the ABI.

Lowering (`sql.rs`) assigns table ids while walking the plan: each
`LogicalPlan::TableScan` gets the next id and registers its schema; the host builds
`Inputs` with a `ScanStream` per registered table, in id order.

---

## 6. The new `gen_scan`

Only this function changes; the operator tree above it is unaffected.

```rust
fn gen_scan(ctx, inputs: Var<SRefMut<Opaque<Inputs>>>, table: u64, schema, yld) {
    let i = ctx.var(0u64);
    ctx.while_loop(true, move |ctx| {                 // outer: batches
        let descs = ctx.bind(call_extern2(scan_next, inputs, Const::new(table)));
        ctx.if_then(is_null(descs.ptr()), |ctx| ctx.break_loop());   // exhausted
        let batch = slice_of(descs);                  // SRef<Slice<FfiArray>>
        let len = gen_len(ctx, batch, schema.field(0).data_type());
        ctx.store(i, 0u64);
        ctx.while_loop(lt(i, len), move |ctx| {       // inner: rows (unchanged body)
            let row = /* read every column at i, via `batch` */;
            yld(ctx, row);
            ctx.store(i, add(i, 1u64));
        });
    });
}
```

`inputs` reaches `gen_scan` the same way `batch` does today — as a kernel parameter
threaded through `gen_op` (the `BatchSource` param generalizes from
`SRef<Slice<FfiArray>>` to `SRefMut<Opaque<Inputs>>` + a per-scan `table` const).

### The `count(*)` shortcut (the payoff)

Because `len` is in hand before the inner loop and the accumulator is a register
above the outer loop, an unfiltered `count(*)` collapses to:

```text
count = 0
loop { descs = scan_next(...); if null break; count += descs[0].len() }
out.push_i64(0, count)
```

No row loop. This is a codegen recognition (`Aggregate{count(*)}` directly over
`Scan`, no `Filter` between) enabled *by construction* by driving the loop in-kernel.
Implement after the base path works; the architecture already permits it.

---

## 7. How JOIN consumes this (sketch — build in the JOIN milestone)

A hash join has a **build** side (materialized, a pipeline breaker) and a **probe**
side (streamed). This foundation gives both:

- **Build side:** its `ScanStream` runs with `retained = Some(_)`, so every consumed
  batch's Arrow buffers stay alive. The build drains the stream fully, hashing each
  row's join key into a host hash table (the same `hashbrown::HashTable` proxy model
  as GROUP BY) whose entries point into those retained buffers. **This is the only
  place we deliberately hold the whole side in memory** — exactly the JOIN case the
  milestone calls out.
- **Probe side:** an ordinary streaming scan (`retained = None`, one batch resident).
  For each probe row, generate the key, look up the build table, and `yld` joined
  rows downstream. Filter/Project/Aggregate above the join are unchanged.

So JOIN = one retained `ScanStream` + a host hash table (build) + one streaming
`ScanStream` (probe). No new streaming/ownership machinery — only the join operator
and its key hashing/compare codegen, which reuse the GROUP BY host-table code.
Deferred here.

---

## 8. What is explicitly deferred

- **Parallelism / partitions.** One `ScanStream` = one partition today. Multiple
  partitions, partial/final aggregation splits, and repartitioning are the
  parallelism milestone.
- **Pipeline breakers in general** (sort, distinct, join build). Only the JOIN
  build-side retain hook is designed above; the operators come later.
- **Output chunking / flush.** Grow-only `OutCols` first; fixed-size flush later
  (§4). Same kernel code.
- **`count(*)` and other whole-batch fast paths** (§6). Enabled by construction,
  implemented after the base path is green.

---

## 9. Phased plan

1. **`OutCols` `SVec`-backed output.** ✅ **Done.** Host `OutCols` with an `SVec`-backed
   `HostVec` per fixed-width column (+ a validity `SVec<u8>` for nullable ones) and a
   `StringViewBuilder` per string column; bake each column's `SVec` handle
   (control-block ptr + `svec_grow`). Reroute `write_col` to inline `SVec::push` for
   fixed-width, the append extern for strings; delete
   `PreparedOutput`/`max_output_rows`/`capacity`. Finalize each `HostVec` to an Arrow
   array (zero-copy via `Buffer::from_vec` for numerics; pack validity bytes→bits).
   Keep a single in-memory batch as input for now. All existing tests must stay green
   (output shape unchanged; only how it's built changes).
2. **`ScanStream` + `Inputs` + `scan_next`.** ✅ **Done.** `scan.rs` holds
   `ScanStream`/`Inputs` + the `scan_next(&mut Inputs, table) -> *const FfiArray`
   extern (null = exhausted). `gen_scan` is now the two-level loop: outer
   `while true { descs = scan_next(inputs, 0); if ptr_is_null(descs) break;
   batch = slice_ref_from_raw_parts(descs, ncols); <inner row loop> }`. Output stays
   baked, so the kernel is `fun1(inputs: &mut Inputs)`. `Inputs::single(rb.clone())`
   wraps the single batch as a one-element stream. NEW rust-lms prims: `ptr_is_null`,
   `slice_ref_from_raw_parts`. All existing single-batch tests green; IR-verified
   two-level loop; `ScanStream` host unit test (refill/drop/null). Kernel-level
   multi-batch execution is validated in step 3.
3. **Real streaming.** ✅ **Done.** `exec_jit_stream(sql, table, schema, batches:
   impl IntoIterator<Item=RecordBatch>)` runs a whole query over a batch stream in
   one kernel; `run_operator(op, inputs: Inputs)` now takes an owned `Inputs`, and
   `exec_jit(sql, table, &rb)` is the `Inputs::single(rb.clone())` wrapper. Tests
   (`tests/streaming.rs`, 7): passthrough / filter / scalar `sum`+`count` / GROUP BY
   all across 3 batches; a split-vs-whole oracle equality; string output across
   batches surviving input drop; and `only_one_input_batch_resident_at_a_time` — a
   `Weak`-based iterator that asserts each batch is dropped before the next is
   pulled. (A multi-table streaming API — `HashMap<table, stream>` — comes with the
   table ids in step 4.)
4. **Table ids / multi-table plumbing.** ✅ **Done.** `Operator::Scan { table:
   usize, schema }`; `sql.rs` lowering resolves each `TableScan`'s name → id via an
   ordered registry (`sql_to_operator_multi(sql, &[(name, schema)])`, id = position);
   `gen_scan` bakes the id as the `scan_next` table constant. `exec_jit_multi(sql,
   Vec<StreamTable{name,schema,batches}>)` builds one `ScanStream` per table in id
   order. Tests (`tests/multi_table.rs`, 6): scanning the id-1 table reads *its*
   stream (not id 0), routing follows registration order (reversed → still correct),
   filter + GROUP-BY-across-batches over a non-zero id, and an unregistered-table
   error. No JOIN yet (a query scans one table); this is the routing a JOIN threads.
5. **`count(*)` / `count(col)` whole-batch shortcut** (§6). ✅ **Done.** When a
   scalar `Aggregate` is a lone `count(*)` or `count(col)` directly over a `Scan`
   (no filter between), `gen_op` drives only the outer batch loop — `count(*)` adds
   `batch_len`, `count(col)` adds `batch_len − null_count` (read type-agnostically
   from the column's validity) — no per-row loop. Shared via `for_each_batch` (the
   batch-loop core extracted from `gen_scan`); gated by `count_fast` (rejects
   `DISTINCT`, `FILTER (WHERE …)`, and computed args, so those keep the row path).
   IR-verified: no inner `icmp ult` for `count(*)`/`count(col)`, present for the
   filtered variant. Tests: `count_star_across_batches_uses_shortcut`,
   `count_star_empty_stream_is_zero`, `count_col_across_batches_sums_nonnull_counts`,
   plus `count_nonnull_col_is_row_count` / `count_all_null_col_is_zero`.

Then the JOIN milestone (§7) sits on top: retain hook + host build table + probe.
