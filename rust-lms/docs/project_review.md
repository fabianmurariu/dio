# rust-lms Project Review

_Snapshot: 2026-05-27, master @ f03ef74_

## TL;DR

`rust-lms` is a type-safe staged-computation DSL on top of Cranelift, modelled
on Scala LMS. The compiler infrastructure (variables, sequencing via tuples,
control flow, slices, struct ABI handling, extern functions, fun0..fun8 +
recursion) is solid and battle-tested by integration tests. Benchmarks land
within ~1.5x of native Rust on a filter + reduce loop.

The project is currently in a transitional state: a push-based staged-iterator
API (`iter/`) has just been added but its key combinators (`zip`, `fold`,
`sum`, `count`, `min`, `max`) are `TODO`-disabled, so users still have to write
`while_loop` + `let_var` boilerplate by hand. That gap is the single biggest
thing standing between this library and "obviously the right way to write a
fused loop in Rust."

The codebase is ~9.2 KLOC across `src/` (excluding tests) with significant
macro-expanded boilerplate in `func.rs` (1.5 KLOC) and `option.rs` (1.5 KLOC).

---

## What Currently Works

### Type system (`types.rs`, ~380 LOC)

- Primitive markers: `i64`, `U64Type`, `I32Type`, `U32Type`, `F64Type`,
  `BoolType`, `UnitType`.
- `Owned<T>` wrapper for by-value struct ABI.
- `StagedType` is the open trait for "things that can flow through codegen";
  `ConstantType` and `CopyType` are refinement traits.
- ABI metadata baked into the trait (`num_abi_values`, `abi_types`,
  `is_fat_pointer`, `should_pass_by_pointer`) — sufficient to express
  pass-by-value structs and fat-pointer slices across the ABI boundary.

### Core staging primitives (`staged.rs`, ~500 LOC)

- `Var<T>` — Copy-when-T-Copy, just an integer ID.
- `Const<T>` — embedded literal.
- `LetVar<T, EXPR>` — bundles a `Var<T>` with its initializer, derefs to the
  underlying `Var<T>`. This is the user-facing "let-binding" abstraction.
- `Assign<V, EXPR>` — side-effecting variable update, returns `UnitType`.
- `IntoStaged<T>` — coerces primitive literals (`42i64`, `0u64`, `3.14f64`)
  into `Const<T>` for ergonomic call sites.
- A reusable `CompilationContext` carries the FunctionBuilder, JITModule, var
  map, extern func refs, and a per-function `unit_value` cache.

### Control flow (`control.rs`, `tuple.rs`)

- `IfThenElse`, `IfThen`, `While` — direct mapping onto Cranelift blocks.
- Tuples `(a, b, c, ...)` sequence side-effects: the tuple's `Out` is the
  last element's `Out` (up to arity 12).

### Slice & reference support (`slice.rs`, `refer.rs`, ~1300 LOC together)

- `SRef<'a, T>` / `SRefMut<'a, T>` — single-value references.
- `Slice<T>` DST with `SRef<'a, Slice<T>>` (`&[T]`) and
  `SRefMut<'a, Slice<T>>` (`&mut [T]`) wrappers.
- Optimised fat-pointer ABI path: slice (ptr, len) lives in two register
  variables rather than a stack slot, with all `SliceLen` / `SliceAsPtr` /
  `SliceGetUnchecked` ops checking `ctx.slice_vars` first and only falling
  back to memory loads when the slice came from inside the function.
- `SPtr<T>` / `SMutPtr<T>` raw-pointer variants for criterion-friendly
  signatures.

### Functions (`func.rs`, `func_def.rs`, `func_impl.rs`, ~2 KLOC together)

- `Compiler::fun0..fun8` (+ `_rec` variants) for arities 0–8.
- `extern_fn::<S: ExternFn>` for safe-ish FFI with auto-generated ABI types
  (via `rust-lms-derive`).
- `compile(expr) -> Compiled<T>` — multi-pass (declare → define → finalize),
  per-platform call conv (SystemV / WindowsFastcall).
- Cranelift opt level is `speed` and libcalls are colocated.

### Other

- `r#struct.rs` (~580 LOC) — field path traversal, copy struct ABI flattening.
- `option.rs` (~1.5 KLOC) — `COption<T>`, `OptRef<T>`, `OptMutRef<T>` and
  pattern matching combinators.
- Iterators (`iter/`) — see "Issue 1" below.

### Test coverage

- `tests/test_slices.rs`, `tests/test_structs.rs`, `tests/test_extern_fn.rs`,
  `tests/test_ergonomic_api.rs`, `tests/type_safety.rs`, `tests/programs.rs`.
- Examples: `basic_usage.rs`, `cranelift_functions.rs`.
- Benches: `jit_comparison{,_simple}.rs`, `minimal_comparison.rs`,
  `copy_overhead.rs`. README claims native parity within ~1.5–1.6x.

---

## Top 5 Fundamental Issues

### 1. Staged iterator API exists but is half-disabled

`src/iter/` advertises a push-based / CPS-fused iterator design, but the
combinators that matter are stubs:

| Combinator                                | Status                            |
| ----------------------------------------- | --------------------------------- |
| `StagedIterator::map`                     | works                             |
| `StagedIterator::filter`                  | works                             |
| `StagedIterator::for_each`                | works                             |
| `StagedIterator::fold`                    | **disabled** — `traits.rs:89`     |
| `StagedIterator::sum / count / min / max` | **disabled** — `traits.rs:97-106` |
| `IndexedStagedIterator::zip`              | **disabled** — `traits.rs:174`    |
| `Zip::consume_indexed`                    | **disabled** — `zip.rs:30`        |

The root cause is a Rust type-system limitation: terminal operations need the
body expression to be `Clone` (so the loop body can be reused), but the
`IntoAccumulatorUpdate::apply_update` return type is `impl Staged<Out =
UnitType>` which doesn't prove `Clone`. The CPS fusion design is right in
principle; the implementation hits the wall the comments call out.

**Impact:** the very example in `src/iter/mod.rs` rustdoc doesn't compile.
Users must still write `while_loop` + `let_var` + manual indexing. The whole
point of the iterator API was to make `slice.staged_iter().zip(other).filter()
.fold()` work, and it can't.

**Why this is _fundamental_:** every other improvement (loop unrolling, SIMD,
multi-source fusion) is supposed to slot into the iterator pipeline. Without a
working `fold`/`zip`, the iterator layer is dead weight.

**Suggested fix:** make `IntoAccumulatorUpdate::apply_update` return a
concrete `Clone` type (parameterised on the accumulator's `Vars`), or expose
an `AccumulatorUpdate: Clone` associated type. The macros in
`iter/accumulator.rs` already specialise per primitive, so a concrete return
type is reachable. Re-enabling `fold` unblocks `sum`, `count`, `min`, `max`,
`zip` (whose blocker is the same `Clone` constraint).

### 2. Sequencing via tuples is a footgun

Side-effects are composed by tuple construction; `(a, b, c)` runs `a`, then
`b`, then yields `c`'s value. This means user code routinely looks like:

```rust
((i, total), while_loop(...), *total)
```

…where `(i, total)` is a tuple-of-`LetVar`-wrappers that exists _purely_ to
trigger initialization, and the outermost expression's type is a 3-tuple of
`Staged` values. Three failure modes follow from this design:

1. **Silent omission.** Forgetting to thread a `LetVar` into the body tuple
   doesn't fail at staging — it fails inside Cranelift codegen with `Variable
N not found in var_map` (see `staged.rs:160`). The borrow checker thinks
   the code is fine; the JIT panics at run time.
2. **Spaghetti types.** A modestly nested loop body produces a return type
   spanning a screen-and-a-half. Type errors from a missing comma in a body
   tuple are unreadable.
3. **Encoded scope leakage.** The benches define `let i = compiler.let_var(...)`
   _outside_ the function body and then include `i` in the body tuple, which
   blurs the line between "variable I want inside this function" and
   "variable I want at module scope." (See `test_slices.rs:69`.) Recent code
   has migrated to `ctx.let_var(...)` inside the closure, which is better;
   the older pattern still compiles silently.

**Why this is _fundamental_:** the DSL's entire control-flow story is built
on overloaded `Tuple` impls. Fixing it cleanly requires a real
`Seq<A, B>`-style sequencing primitive (or a `do!`-style macro) — a breaking
API change.

**Suggested fix:** add an explicit `seq!(stmt; stmt; expr)` macro that desugars
to the existing tuple impls but offers (a) per-statement type checking, (b) a
clear "statements vs. final expression" distinction, (c) a single error
location when initialization is missing. Deprecate raw tuple sequencing once
the macro lands.

### 3. JIT performance is ~1.5x slower than native, with no clear plan to close the gap

The README publishes the current state honestly:

| Size | Native  | rust-lms warm | Ratio |
| ---- | ------- | ------------- | ----- |
| 1K   | 782 ns  | 1.25 µs       | 1.6x  |
| 10K  | 9.36 µs | 15.2 µs       | 1.6x  |
| 100K | 234 µs  | 347 µs        | 1.5x  |

For a _JIT_, being slower than `rustc -O3` on a tight loop is awkward. Several
suspects in the current code:

- **No loop unrolling.** `docs/loop_unrolling_design.md` lays out a clean
  proposal, but nothing in `src/` implements it.
- **No SIMD.** Cranelift supports `simd` via flags / lane types, but rust-lms
  emits scalar ops only. A `sum`/`zip` reduction is exactly the case where 4–8
  lanes would close most of the gap to native (which auto-vectorises).
- **Length recomputed inside the loop predicate.** `slice_iter.rs:135` emits
  `lt(i_var, self.slice.clone().len())` — relying on Cranelift LICM to hoist
  the load. Worth verifying in CLIF dumps; if it doesn't hoist, an explicit
  `let_var` for length would help.
- **`opt_level = "speed"` only.** Cranelift also has `speed_and_size`. Worth
  exploring whether the egraph optimiser (introduced upstream) helps.

**Why this is _fundamental_:** the project's reason to exist over hand-written
Rust is _runtime code generation_ — schema-shaped queries, branchless
selection, fused operator pipelines. If a JITted loop costs more than the
equivalent native loop, the runtime-compile pivot has to pay for itself in
ways that aren't visible in current benches (e.g. dynamic shapes).

**Suggested fix:** establish a perf budget — pick a target ratio (1.05–1.15x
of native on simple reductions) and prove out one or two of: (i) loop unrolling
with split accumulators, (ii) emitting Cranelift `simd` lanes, (iii) explicit
LICM in the staged layer. Pick the cheapest one first and re-measure.

### 4. Massive boilerplate around function arity

`func.rs:246-562` is 18 nearly-identical `fun{N}` / `fun{N}_rec` methods, one
per arity (0..8), each ~17 lines. `func.rs:983-1020` is the matching
`impl_compiled_as_fn!` macro applied 9 times. The accumulator system in
`iter/accumulator.rs:32-214` is the same pattern for tuple arities. `option.rs`
has ~1.5 KLOC of similar shape.

This is partly inherent — Rust doesn't have variadic generics — but the
project leans on it heavily, which compounds:

- **Compile-time cost.** Each `fun{N}` body instantiates Cranelift code-gen
  trampolines; the test suite compiles all of them whether or not your test
  uses arity 8.
- **API discoverability.** Adding a feature (e.g., async functions, or a new
  parameter category) means editing 9 + 9 sites; the chances of forgetting one
  are high.
- **Refactor fragility.** Renaming `FunRef` to `FunRef1` mid-stream (which
  appears half-done — see the prelude re-exports in `lib.rs:132` listing
  both `FunRef` and `FunRef1`) leaves callers inconsistent.

**Why this is _fundamental_:** more of a code-health issue, but it shapes how
quickly the rest of the issues here can be addressed. A `macro_rules!` rewrite
(or `seq_macro` crate) shrinking this to one definition unblocks routine
maintenance.

**Suggested fix:** consolidate `fun{N}` with `macro_rules!` (or
`seq_macro::seq!`). Apply the same treatment to `iter/accumulator.rs` and the
prelude re-exports. Decide between `FunRef` and `FunRef1` and pick one.

### 5. Compile-time errors and panics blur the type-safety promise

The README's headline is "Compile-time type safety." But the type system
guarantees less than it claims:

- `Compiler::var_unchecked` exists and is `unsafe`, with a comment that says
  "you MUST assign before reading or codegen will panic." This is a
  _runtime_ panic dressed up as `unsafe` — the compiler can't help.
- `var_map.get(&self.id).expect(...)` (`staged.rs:160`) — every lookup is a
  panic on miss. Forgetting to sequence a `LetVar` produces this.
- `extern_func_ids.get(&extern_id).expect(...)` (`staged.rs:65`) — same shape
  for extern functions.
- `compile().expect("compile failed")` is the convention in every test and
  bench, so users never see Cranelift errors typed back to them.

The type-level constraints on operations (`Add<L, R>` requires `Out = T` on
both sides; `Lt` always returns `BoolType`; etc.) _do_ work — and they're
genuinely nice. But the surrounding scaffolding has too many panic paths for
the "type-safe staged computation" pitch to hold up.

**Why this is _fundamental_:** the gap between "type-safe at the leaves" and
"type-safe end-to-end" is exactly what a new user trips on when adopting an
embedded DSL.

**Suggested fix:** distinguish staging errors (recoverable, returned as
`Result<_, StagingError>` from a top-level `build()` step) from codegen errors
(real bugs, panics OK). Replace `var_unchecked` with a typestate that proves
initialization at compile time, or remove it and direct users to `let_var` only.

---

## Honorable Mentions (Not in the Top 5)

- **Duplicate cranelift in dependency graph.** The workspace pins
  cranelift 0.127.1; `wasmtime = "25"` (used only behind a feature) bundles
  its own 0.112.3. Both versions are in `Cargo.lock`. Not load-bearing yet but
  inflates build time.
- **Benchmark file proliferation.** `jit_comparison.rs` and
  `jit_comparison_simple.rs` are ~70% the same code; `copy_overhead.rs` is a
  one-trick file. Worth consolidating.
- **Tracked dead file.** `examples/dump_wasm_clif.rs` shows `AD` in git status
  (added then deleted). Either restore or clean.
- **Doc-test format drift.** Most module rustdoc uses ` ```ignore ` —
  reasonable given the codegen-time machinery, but means rustdoc never catches
  drift in the example snippets that the prelude / lib.rs lean on.

---

## Recommended Next Steps (rough priority order)

1. **Unblock `fold` and `zip` in the iterator layer** (Issue 1). Without this,
   any user-facing API improvement runs into a stub.
2. **Re-run the perf comparison after enabling loop unrolling** (Issue 3).
   The design doc is already written; implementing it as an `unroll(N)`
   combinator on `IndexedStagedIterator` aligns with the iterator strategy.
3. **Introduce `seq!`** (Issue 2). Lands the type-safety promise more
   convincingly than the current tuple-sequencing trick.
4. **Macro-collapse `fun{N}` and accumulator impls** (Issue 4). Quality of
   life, but cheap.
5. **Audit panic paths and pick a `Result` story** (Issue 5).
