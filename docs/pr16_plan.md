# PR-16 plan — no-unwind FFI + SQL validation errors

Started: 2026-08-20
Covers PR-16 (and the remaining PR-10 no-unwind work). Follows the execution
contract from the external reviewer's spec: **generated code only ever sees
ABI-safe statuses/values; all error ownership and reporting stays on the Rust
side.** `DataFusionError` never crosses the JIT ABI.

## The contract we're implementing

```
SQL validation error   -> Err(DataFusionError)                     (before JIT runs)
Runtime callback error -> record in query runtime state
                       -> return an ABI-safe sentinel/status to generated code
                       -> generated code terminates the loop normally
                       -> host calls check() after the kernel returns
                       -> Err(DataFusionError)
Internal unexpected    -> contained at the Rust boundary (catch_unwind INSIDE the
panic                     extern "C" fn), query marked failed, never unwinds
                          through JIT frames
```

Key facts that shape the design:

- On current rustc a panic in a plain `extern "C" fn` is a **defined abort**, not
  UB. So the live defect is **availability** (a whole process dies on bad input),
  not memory unsafety. Still worth fixing; framed honestly.
- `catch_unwind` is a **containment layer, not the primary mechanism**. The primary
  path is *checked operations* returning `Result` + `record_error` + sentinel.
- The `catch_unwind` must live **inside** each `extern "C"` callback, wrapping a
  `*_impl` that returns `Result`. Wrapping the Phase-3 thunk's *call* is too late —
  the abort fires the instant a panic tries to leave the callback's own
  non-unwinding frame.
- **We are NOT using `extern "C-unwind"`** — unwinding through JIT frames imposes
  platform unwind requirements and breaks cleanup/invariant reasoning.

## What the inventory found (the important part)

Two distinct surfaces, and the runtime one is *smaller and less reachable* than the
raw panic count suggests; the reachable-from-bad-input panics are mostly stage-0.

### A. Runtime `extern "C"` callbacks (called from the JIT kernel)

None of these panic on *valid* input. Their real failure modes are (1) **allocation
failure** (table/records/scratch growth) and (2) **capacity overflow** — the same
`as u32` truncation as PR-17. The `unreachable!` arms are **true post-validation
invariants** (codegen only emits the int callback for int-keyed tables, the str
callback for str-keyed), not user-reachable.

| Callback | Ret | Real failure modes | Free error sentinel? | Codegen change |
| --- | --- | --- | --- | --- |
| `scan_next` | `*const FfiArray` | batch prepare/schema | **null (already done)** | already checks null → break |
| `group_upsert` | `*mut u8` | alloc grow; `num_records as u32` overflow; str-arm = invariant | **yes, null** (a record ptr is never null) | add null-check → break fold loop |
| `group_upsert_null` | `*mut u8` | alloc grow | yes, null | add null-check |
| `group_upsert_str` | `*mut u8` | alloc grow + pool copy; int-arm = invariant | yes, null | add null-check |
| `group_upsert_composite` | `*mut u8` | alloc; int-arm = invariant | yes, null | add null-check |
| `group_key_reset` / `group_key_push_u64` / `group_key_push_bytes` | `()` | scratch alloc | **no** (unit) → needs status or defer to next upsert's check | prefer: fail lazily, caught at the following upsert |
| `group_records_base` / `group_len` | `*mut u8` / `u64` | none (reads) | n/a | none |
| `join_rel_count` | `u64` | none (`num_batches as u64`) | n/a | none |
| `join_insert` | `()` | `entry().or_default().push` alloc; `(rb_pos,row) as u32` overflow (PR-17) | **no** (unit) | check state error after build loop |
| `join_probe_count` | `u64` | none (lookup) | **no** (0 = valid "no match") | n/a |
| `join_probe_base` | `*const Locator` | none | **NO — null is a valid "no match"** (`join.rs:150`) | must NOT reuse null as error; count already gates the loop |
| `join_left_batch` | `*const FfiArray` | out-of-range `batch_descriptors` index | null ambiguous | bounds-check in impl; report via state |
| `runtime.rs` `str_ptr` | `*const u8` | bad row index | null ambiguous | bounds-check in impl |
| `runtime.rs` `bytes_eq` | `bool` | none | **no** (both valid) | n/a |
| `runtime.rs` `strview_append_bytes` / `_null` | `()` | builder alloc | **no** (unit) | check sink error after output loop |

**Consequence for the design:** a single "null pointer = stop" convention is *not*
enough. The upsert family gets a free null sentinel and codegen adds a null-check.
The unit-returning and count/bool-returning callbacks (`join_insert`, the key
pushers, the strview appends) have **no free sentinel**; for those we do **not**
invent a per-call status — we let them fail into the shared runtime error slot and
have the driver `check()` **after the bounded phase** (build loop, fold loop, output
loop), which is where a first-error-wins sink shines. Codegen currently checks
**none** of these returns, so every "add a check" above is a real codegen edit.

### B. Stage-0 SQL validation / codegen panics (before any JIT code runs)

~24 reachable `panic!`/`unreachable!`/`expect` sites in `codegen/expr.rs`,
`codegen/aggregate.rs`, `codegen/numeric.rs`, `output.rs` (`panic!("unsupported
output column type")`), and `sql.rs`. These unwind normally back to `exec_jit`'s
caller — **not** an FFI crossing, so no abort risk — but they are where *unsupported
SQL* turns into a crash instead of `Err(DataFusionError::NotImplemented/Plan)`.
**This is the bigger user-facing win** and the lower-risk half.

## Standardized runtime error sink

Generalize the `Inputs::error` pattern (already in `scan.rs`) into one carrier
reused by `GroupState`, `JoinState`, and the output builder:

```rust
#[derive(Default)]
pub struct RuntimeStatus { error: Option<DataFusionError> }
impl RuntimeStatus {
    pub fn record_error(&mut self, e: DataFusionError) { if self.error.is_none() { self.error = Some(e); } }
    pub fn check(&mut self) -> Result<()> { self.error.take().map_or(Ok(()), Err) }
}
```

First-error-wins; once failed, callbacks return neutral values and stop mutating
partial state.

## Implementation order (maps to the reviewer's 7 steps)

1. **[done] Inventory** — this document.
2. **Runtime error sink** — add `RuntimeStatus` (a shared module), embed in
   `GroupState`/`JoinState`/output builder; `run.rs` calls `check()?` after each
   bounded phase. (Refactor `Inputs::error` onto it.)
3. **Checked callback impls** — split each fallible callback into a safe `*_impl`
   returning `Result`, using `usize::try_from`, `slice::get(..).ok_or_else`,
   `try_reserve`; the `extern "C"` wrapper records the error + returns the sentinel
   (null for the upsert family; neutral value + recorded error otherwise). This is
   also where PR-17's `as u32` truncations become checked conversions.
4. **Codegen stops safely** — emit a null-check-and-break after `group_upsert*`;
   confirm the join probe already terminates on `count == 0` and gate the build/
   fold/output loops on the post-phase `check()`.
5. **SQL validation → structured errors** — convert the ~24 Category-B panics to
   `DataFusionError`; reserve `unreachable!` for genuine post-validation bugs.
6. **Containment wrappers** — wrap each `extern "C"` callback body in
   `catch_unwind(AssertUnwindSafe(..))` as the final net for the invariant arms and
   anything unexpected; a caught panic records an internal execution error + returns
   the sentinel.
7. **Regression tests** — malformed input → `Err` (no abort, no partial output);
   forced allocation/overflow → recorded error; each converted Category-B case.

## Proposed first increment (for review before coding)

Steps 2 + 4 for **GROUP BY only**, plus the tests for it: land `RuntimeStatus`, wire
it through `GroupState` and `run.rs`, make the `group_upsert*` family checked with a
null sentinel, add the codegen null-check-and-break, and a test that a forced group
overflow/alloc failure returns `Err` instead of aborting. GROUP BY is the cleanest
vertical slice and establishes the pattern the join/output/scan paths then reuse.

## Resolved decisions

- **Both mechanisms (decided).** `RuntimeStatus` retains the detailed
  `DataFusionError`; a `#[repr(u8)] CallbackStatus { Ok = 0, Failed = 1 }` lets
  generated code **stop immediately** instead of deferring. Deferring until the JIT
  returns is only safe if every later callback notices the poison and no-ops — a
  fragile contract that wastes work and risks later ops observing partial mutation.
  So:
  - **Fallible unit callbacks** (`join_insert`, key pushers, `strview_append*`)
    return `CallbackStatus`; on failure they `record_error` + return `Failed`, and
    generated code branches to a **common error epilogue**.
  - **Fallible pointer callbacks** (the `group_upsert*` family) keep returning
    `*mut u8` with **null = Failed** — null is *unambiguous* there (a record pointer
    is never null), so no out-param is needed; still `record_error` first.
  - **Genuinely infallible callbacks** keep returning `()` / `u64` / pointer with
    their existing semantics (e.g. `join_probe_base`'s null = "no match").
  - A staged helper **`ctx.try_call(...)`** hides the repetitive bind/check/branch:
    it emits the call, compares the result to its sentinel (null, or
    `CallbackStatus::Failed`), and `brif`s to the shared error epilogue.
- **PR-17 folded into Phase 4 (decided).** The `as u32` conversions live in exactly
  the callbacks we're rewriting; leaving them means touching the same code twice and
  keeping a silent-corruption path meanwhile. Step 3 replaces them with checked
  `u32::try_from(..).map_err(|_| DataFusionError::Execution("… exceeds u32 range"))`.
  Record PR-17 as completed by Phase 4, with dedicated overflow-boundary tests.
- **Allocation failure policy:** recover capacity overflow and `try_reserve`
  failures into `Err`; do **not** promise recovery from every OS-level OOM (an
  aborting allocator is out of scope). Matches the reviewer's caveat.

## Implementation notes (grounded in the code)

- **`CallbackStatus` lowers to `bool` at the JIT ABI.** rust-lms has no `u8` staged
  marker, and a hand-written `StagedType` over a `#[repr(u8)]` enum would risk
  invalid-discriminant UB if codegen ever produced a non-{0,1} byte. rust-lms
  already models "a 1-byte status" as `bool` (its `I8` control type), so a fallible
  unit callback's `extern "C"` return is `bool` (`true` = failed). The Rust-side
  `CallbackStatus` enum is kept for wrapper/impl readability and converted at the
  boundary. Only fallible *unit* callbacks need it; the `group_upsert*` family uses
  the unambiguous null-pointer sentinel, so the GROUP-BY-first increment needs only
  `RuntimeStatus`.
- **"Stop immediately" mechanism = a poison flag the generated code checks.**
  `break_loop()` only exits the innermost loop (there is no non-local jump to a
  function epilogue). So `ctx.try_call` sets a kernel-wide `poison: Var<bool>` and
  `break_loop`s the innermost loop; each enclosing loop AND-ins `!poison` (or checks
  it at body top) so the nest unwinds to the kernel return, where the driver runs
  `check()?`. This is *generated code* enforcing the stop — not the fragile
  "later callbacks notice poison" contract the reviewer warned against.

## Increment staging (each keeps `cargo test` green)

### Increment 1 — GROUP BY vertical slice — **DONE** ✅

Full workspace green: **343 passed, 0 failed, 2 ignored**; no new clippy lints from
the changed files.

- `sql-gen/src/status.rs` (new): `RuntimeStatus` (first-error-wins `record_error` /
  `is_failed` / `check`) + `CallbackStatus` (declared; first *used* in increment 2).
- `GroupState` embeds a `RuntimeStatus`; `group_upsert*` rewritten as a shared
  `upsert_guard` + checked bodies: on failure they `record_error` and return the
  **null** sentinel; when already poisoned they no-op to null. `group_next_index`
  replaces `num_records() as u32` with a checked `u32::try_from` (**PR-17**, group
  side).
- Codegen: `CodegenCtx.poison: Rc<Cell<Option<Var<bool>>>>`, created once in
  `gen_collect`; `for_each_batch` breaks when poisoned; `stop_if_null` after each
  `group_upsert*` sets poison + `break_loop` so the fold never dereferences null.
- `run.rs` calls `group_state.check()?` after the kernel (alongside
  `inputs.take_error()?`).
- Tests (`group.rs` lib, via a `#[cfg(test)]` `GROUP_LIMIT` seam): an induced
  overflow returns `Err` (no abort / null-deref / partial result); the happy path is
  unaffected.

### Remaining increments

2. **Join / output / strview callbacks** — the fallible *unit* callbacks
   (`join_insert`, `group_key_*`, `strview_append*`): first use of `CallbackStatus`
   (lowered to `bool`) + `ctx.try_call`'s status form; finish **PR-17** on the join
   side (`Locator`/`join_insert` `rb_pos,row` as u32). Embed `RuntimeStatus` in
   `JoinState` / output builder; refactor `Inputs::error` onto the shared type.
3. **Category B — SQL validation** — convert the ~24 stage-0 `panic!`/`unreachable!`
   in `codegen/{expr,aggregate,numeric}.rs`, `output.rs`, `sql.rs` to
   `DataFusionError::{NotImplemented,Plan,Execution}`; reserve `unreachable!` for
   genuine post-validation bugs.
4. **Containment wrappers** — wrap each `extern "C"` callback body in
   `catch_unwind(AssertUnwindSafe(..))` as the final net (records an internal error +
   returns the sentinel), so the invariant `unreachable!` arms can never unwind
   through JIT frames.
5. **Regression tests** for each of the above; mark **PR-17** complete with the join
   overflow-boundary test.
