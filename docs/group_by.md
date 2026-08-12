# GROUP BY code generation

How `sql-gen` executes `GROUP BY` today. The short version: a grouped aggregate is
an ordinary **push operator** inside the JIT kernel. It folds its input into
**Rust-hosted accumulator buffers**, then **emits one manifested row per group**
downstream — so any projection/filter above it (including `HAVING`) runs in the
*same* JIT unit. There is no second compiler and no second pass.

Scope today: a single `Int64` (non-null) group key; `count(*)`, `count(col)`,
`sum`, `min`, `max`, `avg` over **integer *or* `Float64`** inputs, with full null
handling. The accumulator cell type (`i64` vs `f64`) is chosen per aggregate from its
datafusion output type — see [§3](#3-the-slot-layout).

---

## 1. The shape of the plan

`SELECT key, sum(value) + 2 FROM t GROUP BY key` lowers (via datafusion → our
`Operator` tree) to:

```
Project([ key, sum(value) + 2 ])
  └─ Aggregate(group=[key], aggr=[sum(value)])
       └─ Scan(t)
```

The aggregate produces `[key, sum(value)]`; the projection above it computes
`sum(value) + 2`. Both run in one kernel: the aggregate emits group rows, the
projection consumes them. This is the same push model used everywhere else —
`Aggregate` just happens to emit *N* rows (one per group) instead of streaming its
input through.

The only thing special about a grouped aggregate is that its accumulators are
**arrays indexed by a runtime group index**, which can't be Cranelift `Var`s
(those are fixed registers). So the accumulators live in host memory.

---

## 2. The three pieces of state

### 2a. `GroupState` — the host-owned buffers (`group.rs`)

Allocated *before* the kernel is compiled and kept alive across the whole run.

```rust
pub struct GroupState {
    pub table: GroupTable,      // key -> dense group index
    buffers: Vec<Vec<i64>>,     // one accumulator buffer per "slot"
}

impl GroupState {
    /// One `capacity`-row buffer per slot, filled with the slot's identity.
    pub fn new(slot_inits: &[i64], capacity: usize) -> Self { /* ... */ }

    /// Baked addresses handed to the kernel:
    pub fn table_ptr(&mut self) -> u64 { &mut self.table as *mut GroupTable as u64 }
    pub fn base_ptrs(&self) -> Vec<u64> { self.buffers.iter().map(|b| b.as_ptr() as u64).collect() }
}
```

Key facts:

- **Buffers are `i64` 8-byte slots.** `avg`'s running sum is an `f64`, but it reuses
  the same 8 bytes via a bit-reinterpreting pointer (and `0.0` is the `0` bit
  pattern, so a zero-init works for it too).
- **Sized to the input row count.** The number of groups is `≤` the number of rows,
  and group indices are dense (`0..num_groups`), so a buffer of `capacity = rows`
  never overflows and **never grows** — its base pointer stays valid for the run.

### 2b. `GroupTable` — just the hash map (`group.rs`)

The table owns *only* `key -> group index`. Accumulation is not its job.

```rust
#[derive(Default)]
pub struct GroupTable { index: HashMap<i64, u32> }

/// Return `key`'s group index — existing, or the next free one (= current count).
#[extern_fn]
pub extern "C" fn group_intern(table: &mut GroupTable, key: i64) -> u64 {
    let next = table.index.len() as u32;
    *table.index.entry(key).or_insert(next) as u64
}

#[extern_fn]
pub extern "C" fn group_len(table: &GroupTable) -> u64 { table.index.len() as u64 }
```

`group_intern` assigns dense, insertion-ordered indices — so the index doubles as
the row cursor into the accumulator buffers. `group_len` gives the group count for
the emit loop's bound.

### 2c. `GroupHandle` — the baked pointers (`codegen.rs`)

The bridge from host state to the kernel. It carries only addresses, captured
before compilation and constant-folded into the generated code — exactly the
pattern the string-literal pool uses.

```rust
#[derive(Clone)]
pub struct Cx {
    pub rt: Runtime,                       // extern handles
    pub lits: Arc<HashMap<String, u64>>,   // interned string literals
    pub group: Option<Arc<GroupHandle>>,   // the GROUP BY's baked state, if any
}

pub struct GroupHandle {
    pub table_ptr: u64,   // address of the GroupTable
    pub bases: Vec<u64>,  // base address of each slot buffer
}
```

`Cx.group` is `Option` because most plans have no group-by; it's a single handle
for now, but a `Vec` indexed by plan order generalizes to multiple group-bys
(CTEs) later.

---

## 3. The slot layout

`group_layout` maps the aggregate list to a flat list of accumulator slots, and it
is the single source of truth shared by the host allocator and codegen (so they
agree on buffer indices).

```rust
struct Slot { init: i64 }                                   // identity fill
struct AggSlots { kind: AggKind, value: usize, count: Option<usize> }
struct GroupLayout { key: usize, slots: Vec<Slot>, aggs: Vec<AggSlots> }
```

Each aggregate also carries an **`AccTy`** (`I64` | `F64`) — the physical cell it
folds in — chosen from its datafusion output `DataType` (`agg_output_types`): `count`
is always `I64`, `avg` always `F64`, and `sum`/`min`/`max` are `F64` iff the output is
`Float64` (so `min(f64_col)` folds in `f64`, `sum(i32_col)` widens to `i64`). The cell
type also picks `min`/`max`'s identity fill.

Layout rules (`init` is the raw `i64` bits stored in the 8-byte cell):

| slot                       | who                        | init                        |
|----------------------------|----------------------------|-----------------------------|
| `0` (key)                  | always                     | `0`                         |
| `value` per aggregate      | `count`, `sum`, `avg`      | `0` (`f64 0.0` == `0` bits)  |
|                            | `min` (`i64` / `f64`)      | `i64::MAX` / `+∞` bits       |
|                            | `max` (`i64` / `f64`)      | `i64::MIN` / `−∞` bits       |
| `count` per aggregate      | `sum`/`min`/`max`/`avg`    | `0`                         |

`min`/`max` over a float column seed the cell with `±∞` (not `i64::MAX/MIN`
reinterpreted, which would be a bogus float) so the first real value always wins. The
fold dispatches on `AccTy` — `combine_i64` (`add`/`min`/`max` → `iadd`/int compare) vs
`combine_f64` (→ `fadd`/`fcmp`); `finalize` emits `ColVal::I64` or `ColVal::F64`
accordingly, and `write_col` narrows to the output column type.

The per-aggregate **`count` slot** is the non-null input count. It serves two
purposes: it's the "seen a non-null value?" bit (`count > 0`) for the NULL
semantics of `sum`/`min`/`max`/`avg`, and it's the divisor for `avg`. `count(*)` /
`count(col)` need no count slot — they *are* a count and are never null.

The host only needs the identity fills to allocate:

```rust
/// `group_slot_inits(aggs).len()` == number of buffers; each entry is that slot's fill.
pub fn group_slot_inits(aggs: &[Expr]) -> Vec<i64> { /* ... */ }
```

For `aggr=[sum(value)]` the layout is 3 slots: `[key=0, sum.value=0, sum.count=0]`.

---

## 4. Reaching a baked buffer from the kernel

Indexed reads/writes of `buffers[slot][gidx]` go through the baked base address and
raw-pointer ops. `raw_ptr`/`raw_mut_ptr` are new rust-lms helpers that reinterpret
a staged `u64` as a typed `*const T`/`*mut T` (emitting no code — a pointer *is*
its address, just like `opaque_ref`):

```rust
fn acc_get_i64(ctx: &mut Ctx, base: u64, gidx: Var<u64>) -> Var<i64> {
    let i = ctx.bind(int_cast::<i64, u64, _>(gidx));
    ctx.bind(array_index(raw_ptr::<i64, _>(base), i))          // *(base as *const i64).add(i)
}

fn acc_set_i64(ctx: &mut Ctx, base: u64, gidx: Var<u64>, v: Var<i64>) {
    let i = ctx.bind(int_cast::<i64, u64, _>(gidx));
    ctx.emit(store(ptr_offset_mut(raw_mut_ptr::<i64, _>(base), i), v));  // *(base as *mut i64).add(i) = v
}
```

`acc_get_f64` / `acc_set_f64` are identical but reinterpret the same bytes as
`f64` (used only by `avg`'s running sum).

---

## 5. The kernel: fold, then emit

`gen_grouped` is the whole operator. It receives the downstream continuation `yld`
(the projection above it), folds the input, then emits one row per group into
`yld`.

```rust
fn gen_grouped<B: BatchSource>(
    ctx: &mut Ctx, batch: B,
    group_exprs: &[Expr], aggs: &[Expr], input: &Operator,
    cx: &Cx, yld: Yld,
) {
    let handle = cx.group.clone().expect("grouped aggregate without a baked GroupState");
    let layout = group_layout(aggs);
    let key_base = handle.bases[layout.key];
    let table_ptr = handle.table_ptr;

    // Resolve each aggregate's slots to their baked base addresses.
    let resolved: Vec<ResolvedAgg> = layout.aggs.iter().zip(aggs).map(|(a, e)| ResolvedAgg {
        kind: a.kind,
        value_base: handle.bases[a.value],
        count_base: a.count.map(|c| handle.bases[c]),
        arg: GroupedAgg::parse(e).arg,
    }).collect();

    // --- FOLD: reuse gen_op for the input; the closure is the per-row body ---
    gen_op(input, ctx, batch, cx, Box::new(move |ctx, row| {
        let key = to_i64(ctx, gen_expr(ctx, &group_expr, &input_schema, &row, &cx_c));
        let gidx = ctx.bind(call_extern2(
            cx_c.rt.group_intern,
            opaque_ref_mut::<GroupTable, _>(table_ptr),   // baked &mut GroupTable
            key,
        ));
        acc_set_i64(ctx, key_base, gidx, key);            // remember the key for emit
        for agg in &resolved_f { agg.fold(ctx, gidx, &row, &input_schema, &cx_c); }
    }));

    // --- EMIT: one manifested Row per group, pushed downstream ---
    let num_groups = ctx.bind(call_extern1(cx.rt.group_len, opaque_ref::<GroupTable, _>(table_ptr)));
    let g = ctx.var(0u64);
    ctx.while_loop(lt(g, num_groups), move |ctx| {
        let key = acc_get_i64(ctx, key_base, g);
        let mut row: Row = vec![ColVal::I64(key, Nullness::NonNull)];
        for agg in &resolved { row.push(agg.finalize(ctx, g)); }
        yld(ctx, row);                                    // -> projection/filter above
        ctx.store(g, add(g, 1u64));
    });
}
```

Two things worth noticing:

- **The fold reuses `gen_op(input, …)`.** The input can be a `Scan`, or a `Filter`
  (a `WHERE`), or anything else — it's the normal push machinery, and the fold
  closure is just its consumer. So `GROUP BY` under a `WHERE` composes for free.
- **`yld` is called once, at codegen, inside the `while_loop` body.** The loop
  *runs* per group at runtime, but the body — including the projection's code — is
  *emitted* once. This is exactly how `gen_scan` calls `yld` inside its row loop.

### The fold, per aggregate

Only non-null inputs contribute (`if_then(valid, …)`); nullable aggregates also
bump their count slot (the "seen" bit / divisor):

```rust
impl ResolvedAgg {
    fn fold(&self, ctx, gidx, row, schema, cx) {
        let (val_cv, valid) = agg_arg_value(&self.arg, ctx, row, schema, cx);  // value + validity bit
        match self.kind {
            AggKind::Count => ctx.if_then(valid, move |ctx| {
                let cur = acc_get_i64(ctx, value_base, gidx);
                acc_set_i64(ctx, value_base, gidx, ctx.bind(add(cur, 1i64)));
            }),
            AggKind::Sum => {
                let v = to_i64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    bump_count(ctx, count_base, gidx);                 // seen++
                    let cur = acc_get_i64(ctx, value_base, gidx);
                    acc_set_i64(ctx, value_base, gidx, ctx.bind(add(cur, v)));
                });
            }
            AggKind::Min => { /* … min(cur, v), bump_count … */ }
            AggKind::Max => { /* … max(cur, v), bump_count … */ }
            AggKind::Avg => {                                          // f64 sum + i64 count
                let v = to_f64(ctx, val_cv.unwrap());
                ctx.if_then(valid, move |ctx| {
                    bump_count(ctx, count_base, gidx);
                    let cur = acc_get_f64(ctx, value_base, gidx);
                    acc_set_f64(ctx, value_base, gidx, ctx.bind(add(cur, v)));
                });
            }
        }
    }
}
```

The `valid` bit comes from the input value's static nullness: `true` for a
non-nullable column or `count(*)` (no argument), otherwise the column's `is_valid`
bit. For a non-nullable input, `if_then(true, …)` is a constant-true branch that
Cranelift collapses.

### The emit, per aggregate — where `avg` divides and NULLs are set

`finalize` manifests each aggregate's **final `ColVal`** for group `g`, carrying its
nullability. Because the value is fully materialized here, everything above (a
projection, a `HAVING` filter) just reads it as a normal column.

```rust
impl ResolvedAgg {
    fn finalize(&self, ctx, g) -> ColVal {
        match self.kind {
            AggKind::Count => ColVal::I64(acc_get_i64(ctx, self.value_base, g), Nullness::NonNull),

            AggKind::Sum | AggKind::Min | AggKind::Max => {
                let v = acc_get_i64(ctx, self.value_base, g);
                let seen = self.seen(ctx, g);                          // count > 0
                ColVal::I64(v, Nullness::Nullable(seen))               // NULL if the group had only nulls
            }

            AggKind::Avg => {
                let sum = acc_get_f64(ctx, self.value_base, g);
                let count = acc_get_i64(ctx, self.count_base.unwrap(), g);
                let avg = ctx.bind(div(sum, ctx.bind(int_to_float::<f64, i64, _>(count))));
                let seen = ctx.bind(gt(count, 0i64));
                ColVal::F64(avg, Nullness::Nullable(seen))             // NULL if count == 0
            }
        }
    }
}
```

**Why `avg` divides here and not as a projection.** `HAVING avg(v) > 10` lowers to
`Aggregate → Filter(avg(v) > 10) → Projection`. The `HAVING` filter sits *below* the
final projection and references `avg(v)` as an aggregate output column. So `avg`
must be a manifested aggregate value that any downstream operator can read — not a
value that only exists after some projection. Dividing at emit makes `avg` behave
like every other aggregate: `HAVING` is "just a `Filter` on the aggregate's rows."

---

## 6. Wiring it up (`run.rs`)

`exec_jit` is uniform — no special grouped path. `run_operator` allocates the group
state up front, bakes the pointers, and compiles the whole plan into one kernel:

```rust
fn run_operator(op: Operator, rb: &RecordBatch) -> Result<RecordBatch> {
    // Allocate the GROUP BY state (if the plan has one), sized to the row count.
    let mut group_state = match find_grouped(&op) {
        Some(Operator::Aggregate { aggs, .. }) => Some(GroupState::new(&group_slot_inits(aggs), rb.num_rows())),
        _ => None,
    };
    let group = group_state.as_mut().map(|gs| Arc::new(GroupHandle {
        table_ptr: gs.table_ptr(),
        bases: gs.base_ptrs(),
    }));

    let cx = Cx { rt, lits: Arc::new(lits), group };

    // ONE kernel for the whole plan. The grouped Aggregate is a push operator in it.
    let f = compiler.fun2("query", move |ctx, batch, sink| {
        gen_collect(ctx, batch, sink, &op, &out_schema, &cx)
    });
    let compiled = compiler.compile(f)?;

    let n = compiled.as_fn()(prepared_in.arrays(), out.as_ffi_mut());
    let result = out.into_record_batch(n as usize);
    drop(pool);         // interned literal bytes stay alive across the run…
    drop(group_state);  // …and so does the group state (baked pointers reference it)
    Ok(result)
}
```

`find_grouped` locates the (single) grouped `Aggregate` node. The kernel signature
is the ordinary `fun2(input, out)` — the group state is *baked*, not a parameter.

### Lifetime / safety contract

The baked pointers are raw addresses into `group_state`, a local that lives until
the end of `run_operator`. The contract — the same one `Compiled`/`as_fn` and the
literal pool already rely on — is upheld by construction:

1. `group_state` is allocated **before** `compiler.compile`.
2. Its `Vec` buffers are never resized after allocation (sized to `rows`), so their
   base pointers stay valid.
3. It is dropped **after** `as_fn` returns.

During `as_fn`, the kernel mutates `group_state.table` (through the baked
`table_ptr`) and the buffers — and we touch nothing else, so the raw aliasing is
sound, exactly like an `extern "C"` call.

---

## 7. Worked example, end to end

`SELECT key, sum(value) + 2 FROM t GROUP BY key`

**Layout** (`aggr = [sum(value)]`): 3 slots — `key`=0, `sum.value`=1, `sum.count`=2,
all identity `0`. `GroupState` allocates three `Vec<i64>` of length `rows`.

**Emitted kernel** (`fn(input, out) -> row_count`), in pseudocode:

```
// ── FOLD (drive the Scan; body per input row i) ──
for i in 0 .. len(input):
    key  = input.key[i]
    gidx = group_intern(&table, key)            // extern; dense index
    key_buf[gidx] = key                          // baked buffer 0
    if input.value is valid at i:
        sum_count[gidx] += 1                      // baked buffer 2 (seen / count)
        sum_val[gidx]   += input.value[i]         // baked buffer 1

// ── EMIT (one row per group) ──
n = 0
for g in 0 .. group_len(&table):
    key = key_buf[g]
    sum = sum_val[g]
    seen = sum_count[g] > 0
    // projection above the aggregate: [ key, sum + 2 ]
    out.col0[n] = key
    if seen { out.col1[n] = sum + 2 } else { out.col1 null at n }
    n += 1
return n
```

The `sum + 2` and the null-when-unseen live in the **projection's** code, emitted
into the emit loop's body via `yld`. One kernel; no second compile.

For `avg`, buffer 1 holds the `f64` running sum and buffer 2 the count; the emit
loop computes `sum / count` (NULL when `count == 0`) before handing the row up.

---

## 8. Limitations & extension points

- **Single, non-null `Int32`/`Int64`, `Float64`, or `Utf8View` key.** A nullable key
  is rejected (`NotImplemented`). Int keys are widened to `u64` bits; `Float64` keys are
  canonicalized (`-0.0`/NaN) and bit-keyed on the same `u64` table; string keys hash/
  compare on content and are copied into a `BytesPool` bundled with the table (so the
  result survives the input batch — needed once inputs stream). Composite keys are the
  next key-typing step; computed string keys (`upper(x)`) wait on string fns.
- **Aggregate value types: `i64` and `Float64` done** (`sum`/`min`/`max` pick their
  cell from the output type; `avg` always `f64`). `Decimal`/other numeric inputs
  still fall through to the `to_i64` panic.
- **Utf8View / composite keys** need a richer `GroupTable` (byte-hash via the
  `BytesPool`, or multi-column interning) — future work.
- **Single group-by per plan.** `Cx.group` is one handle; a `Vec` indexed by plan
  order extends it to CTEs with multiple group-bys, each baking its own state.
- **Parallel merge.** `GroupState`'s `[keys | accumulators]` (with `avg` kept as
  `(sum, count)` until emit) is already the mergeable partial-aggregate state, so a
  two-phase partial/final split drops in without redesign.
- Two deferred codegen inefficiencies (redundant loop-invariant pointer reloads; a
  dead `iconst.i8 0`) are tracked in `docs/codegen_issues.md`.

---

## 9. Storage layout & future work (Umbra comparison)

> **Update (Phase 2, done):** the state is now **row-wise packed**, not columnar.
> `GroupState.records` is a single `u64`-backed buffer of one `[key | per-agg value
> (+count)]` record per group; `codegen::group_record` builds a `RecordLayout`
> (rust-lms-std), fold/emit go through `layout.record(gidx)` + typed `FieldHandle`s.
> This delivered item (3) below. The full roadmap to Umbra lives in
> `docs/path_to_umbra_group_by.md`; what remains here is items (1)–(2).

Umbra packs each group **row-wise**, a contiguous tuple `[hash | key(s) | aggregate
payload]` in one open-addressing arena (the "Tidy Tuples" paper's Tuples layer:
*"packing tuples into a memory efficient format"*) — which is now what we do too, for
the *payload*. Two first-order gaps remain, in priority order:

1. **Per-row proxy call.** `group_upsert` is one host call per input row (find-or-
   insert on `hashbrown` + return the record pointer). This is *exactly* Umbra's
   model (Fig 5's `insert` is a proxy), so it's a deliberate design point, not a
   wart — inlining the probe as staged code stays a possible later optimisation, not
   a goal.
2. ✅ **O(groups) memory** (Phase 4) — the records buffer starts empty and grows one
   record per new group; the last O(rows) allocation is gone. Full roadmap and the
   key-generic table are in `docs/path_to_umbra_group_by.md`.
3. ✅ **Row-wise packing** (Phase 2) — a group's cells share a cache line in the fold
   loop.
