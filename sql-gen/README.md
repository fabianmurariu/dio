# sql-gen

A compiling SQL executor: each query is lowered to a **staged program**, JIT-compiled
to native code, and run over Apache Arrow columns. No interpreter, no expression
trees at runtime — the query becomes machine code.

```
SQL ──datafusion──▶ LogicalPlan ──sql.rs──▶ Operator tree ──codegen/──▶ staged rust-lms program
                                             (push-based)                        │
                                                                     Cranelift ──┴── LLVM/MLIR
                                                                          │
                                                                    machine code ──▶ RecordBatch
```

It is two things at once: a working query engine, and the flagship worked example
for [`rust-lms`](../rust-lms), the multi-stage programming library underneath it.

---

## The idea it is built on

sql-gen is a direct transposition of Rompf & Amin, **"A SQL to C Compiler in 500
Lines of Code"** (`docs/sql_to_c.pdf`), from Scala/LMS into Rust/rust-lms.

The paper's argument, in one line: *write a clean definitional interpreter for
relational algebra, then change one type, and you have a compiler.* That is the
first Futamura projection — specializing an interpreter to a program yields a
target program:

```
target = staged-interpreter(source)
```

The interpreter looks like this (paper §2) — a push model where each operator gets
a callback it invokes per emitted record:

```scala
type Semant = (Record => Unit) => Unit

def execOp(o: Operator)(yld: Record => Unit): Unit = o match {
  case Filter(pred, parent) =>
    execOp(parent) { rec => if (evalPred(pred)(rec)) yld(rec) }
  case Project(newSchema, parentSchema, parent) =>
    execOp(parent) { rec => yld(Record(rec(parentSchema), newSchema)) }
  ...
}
```

The single creative change (§3) is to make a record's *fields* future-stage while
its *schema* stays present-stage — a **mixed-stage data structure**:

```scala
case class Record(fields: Vector[Rep[String]], schema: Vector[String])
```

Everything else follows from fixing the resulting type errors. Records can then no
longer exist in the generated code; the fields become local variables in the
emitted program.

Here is sql-gen's `Filter` and `Project`, against the same two cases above:

```rust
type Yld = Box<dyn FnOnce(&mut Ctx, Row) + 'static>;

fn gen_op(op: &Operator, ctx: &mut Ctx, inputs: I, cx: &CodegenCtx, yld: Yld) {
    match op {
        Operator::Filter { predicate, input } => gen_op(input, ctx, inputs, cx,
            Box::new(move |ctx, row| {
                let keep = gen_predicate(ctx, &predicate, &schema, &row, &cx_c);
                ctx.if_then(keep, move |ctx| yld(ctx, row));
            })),

        Operator::Project { exprs, input, .. } => gen_op(input, ctx, inputs, cx,
            Box::new(move |ctx, row| {
                let projected: Row = exprs.iter()
                    .map(|e| gen_expr(ctx, e, &schema, &row, &cx_c)).collect();
                yld(ctx, projected);
            })),
        ...
    }
}
```

Same shape, same push model, same recursion. `if` became `ctx.if_then` because the
condition is now a stage-1 value — that is the whole difference, and it is exactly
the difference the paper describes.

---

## The correspondence

| "SQL to C in 500 Lines" | sql-gen |
|---|---|
| `Rep[T]` | `Var<T>` — a handle to a stage-1 value (`rust-lms`) |
| `type Semant = (Record => Unit) => Unit` | `type Yld = Box<dyn FnOnce(&mut Ctx, Row)>` |
| `def execOp(o: Operator)(yld: …)` | `fn gen_op(op: &Operator, ctx, inputs, cx, yld: Yld)` |
| `Operator` ADT — `Scan`/`Project`/`Filter`/`HashJoin`/`Group` | `plan::Operator` — the same set |
| `Record(fields: Vector[Rep[String]], schema: Schema)` | `value::Row = Vec<ColVal>` + an Arrow `SchemaRef` |
| `Value` / `StringValue` / `IntValue` hierarchy (§5) | `value::ColVal` — `I32`/`I64`/`F64`/`Bool`/`Str` |
| `StringValue(Rep[Pointer[Char]], Rep[Int])` | `StrVal::Bytes { ptr: Var<SPtr<u8>>, len: Var<u64> }` |
| `ColBuffer` / `ArrayBuffer` — column storage (§5) | `output::OutCols` — one growable `SVec<T>` per column |
| `HashMapAgg`, `HashMapBuffer` (§4) | `group::GroupState`, `join::JoinState` |
| `processCSV(file)(yld)` | `gen_scan` / `for_each_batch` over Arrow batches |
| parser combinators → `Operator` | datafusion parser → `LogicalPlan` → `sql.rs` → `Operator` |
| `printf`, `Scanner` as runtime primitives | `runtime.rs`, via `#[extern_fn]` |
| `LMS_Driver` → Scala/C source → `scalac`/`gcc` | `Compiler::compile` → Cranelift or LLVM, in-process |

The two that carry the most weight:

**`Row` is the mixed-stage record.** Static Arrow schema, `Vec` of stage-1
handles. It never materializes in generated code — the fields are registers. The
only place a tuple is genuinely packed is the GROUP BY record, and that is packed
deliberately, because it is host-owned state that outlives the row.

**`ColVal` is the `Value` hierarchy.** The paper arrives there in §5, when
generating C forces it to stop treating every field as `Rep[String]` and introduce
typed field representations with their own `compare`/`hash`. sql-gen starts there,
because Arrow columns are typed on arrival.

---

## Where sql-gen goes further

The paper is deliberately a 500-line teaching artifact. sql-gen is ~4,800 lines,
and most of the difference is the things a working engine cannot skip.

### NULLs, decided at stage 0

The paper has no NULL semantics at all. sql-gen carries nullness as a *static* tag
next to each value:

```rust
enum Nullness { NonNull, Nullable(Var<bool>) }
```

For a column the Arrow schema declares non-nullable there is no validity `Var` at
all — not a branch that folds away later, but code that was never emitted, because
the decision was made in Rust before anything reached the backend. Nullable values
carry an `is_valid` bit that propagates through `gen_expr` with SQL's three-valued
semantics.

This is the paper's own staging discipline applied to a question it never asks:
nullability is static data, so it belongs in the present stage.

### IR, not source text

The paper's LMS driver emits **Scala or C source**, then shells out to `scalac` or
`gcc` and loads the result — a real compiler invocation per query.

rust-lms emits into an in-process JIT — Cranelift's function builder, or MLIR —
with no source text, no temporary files, and no external toolchain. It also means
control flow is generated **directly in SSA form**: `ctx.if_then_else` emits a
merge block whose block parameter *is* the phi node.

Two backends are available, mirroring the paper's Scala-vs-C choice as a
fast-compile-vs-fast-code one:

```rust
exec_jit(sql, table, &batch)                          // Cranelift (default)
exec_jit_with(JitBackend::Llvm, sql, table, &batch)   // LLVM/MLIR
```

On rust-lms's own loop kernels (`cargo bench -p rust-lms --bench iter`), LLVM
produces code **~3–6× faster** on flat `i64` reductions — essentially all of it
auto-vectorization, which Cranelift does not do for scalar loops by design. The
backend is chosen by the caller and never revisited; there is no adaptive policy.

### The staged type system carries more than `Rep[T]`

`Rep[Pointer[Char]]` is the paper's only pointer type. rust-lms distinguishes:

- **provenance** — `SPtr<T>`/`SMutPtr<T>` are raw addresses; `SRef<T>`/`SRefMut<T>`
  are known-valid references. A descriptor arriving from FFI is a `RawSlice<T>`
  and reaches no safe accessor until an explicit `assume_shared`/`assume_unique`;
- **mutability and aliasing** — the slice taxonomy (`SliceType` →
  `TrustedSliceType` → `MutSliceType`, with `RawSliceType` as a sibling) is sealed,
  and sub-slicing preserves capability rather than laundering it;
- **borrows of host state** — a view into a growable host buffer carries a Rust
  lifetime, so growing that buffer while a derived slice is live is a *compile*
  error rather than a dangling pointer baked into a kernel.

Which is why sql-gen bakes host addresses as real `*const T`/`*mut T` and
reinterprets them through typed constructors, rather than threading `u64`s around.

### Arrow in, Arrow out

The paper reads CSV, and much of §5's performance work is about *getting out of
CSV's way* — `mmap`, pointers into the mapped file, avoiding copies into string
objects.

sql-gen skips that problem: input is already columnar, already typed, and already
carries validity bitmaps. `arrow-lms` hands the kernel lifetime-free `FfiArray`
descriptors, and output goes into growable Arrow builders. The scan is a batch-pull
loop over a stream, not a tokenizer.

### Errors from generated code

The paper does not address what happens when a runtime callback fails. sql-gen
needs an answer, because `DataFusionError` is not FFI-safe: fallible callbacks
record the error in a Rust-owned `RuntimeStatus` and return only an ABI-safe
sentinel (a null pointer, or a `bool`), and the kernel carries a **poison flag**
that stops the scan pulling further batches. See `status.rs`.

### The plan comes from datafusion

The paper writes its own parser-combinator SQL grammar. sql-gen reuses
datafusion's parser, its `LogicalPlan`, and `datafusion_expr::Expr` verbatim as the
scalar expression language. `sql.rs` lowers datafusion's pull-based plan into our
push-based `Operator` tree; everything downstream is ours.

---

## Scope

- `Scan` / `Filter` / `Project`
- scalar aggregates and `GROUP BY` — `count`/`sum`/`min`/`max`/`avg` with NULL
  semantics; single, composite, and string keys
- **inner** equi-joins (hash join; left input builds, right probes) — one key pair
- primitive columns (`i32`/`i64`/`f64`/`bool`) and `Utf8View` strings
- multi-batch streaming input, multi-table queries

Not yet: sorting, window functions, outer/semi/anti joins, multi-key joins,
`HAVING`/`ORDER BY`/`LIMIT`, parallelism (the GROUP BY state is designed to be
mergeable, but nothing merges it yet), overflow checking.

The paper's §7 sketches what growing this into a full system takes — index
structures, parallel execution, TPC-H — via the Flare work. The design notes under
`docs/` track the same ground.

## Running

```bash
cargo test -p sql-gen                     # 113 tests, Cranelift
cargo test -p sql-gen --features llvm     # every test on both backends

# Both backends need llvm-config on PATH:
env MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm LLVM_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
    PATH="/opt/homebrew/opt/llvm/bin:$PATH" DYLD_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib \
    cargo test -p sql-gen --features llvm
```

```rust
use sql_gen::exec_jit;
let out = exec_jit("SELECT key, count(*) FROM t GROUP BY key", "t", &batch)?;
```

`RUST_LMS_DEBUG_IR=1` dumps the generated IR — the first debugging tool to reach
for when a kernel misbehaves.

## Further reading

- [`docs/group_by.md`](../docs/group_by.md) — GROUP BY design and the packed record layout
- [`docs/joins.md`](../docs/joins.md) — hash join build/probe
- [`docs/table_scan.md`](../docs/table_scan.md) — the scan and batch-pull protocol
- [`docs/codegen_issues.md`](../docs/codegen_issues.md) — known rough edges
- [`docs/path_to_umbra_group_by.md`](../docs/path_to_umbra_group_by.md) — where the aggregation design is headed
- [`rust-lms/docs/deep_dive.md`](../rust-lms/docs/deep_dive.md) — the staging library underneath
