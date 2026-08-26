# Post-LLVM review — remediation plan

Companion to [`post_llvm_review.md`](post_llvm_review.md). This plan first records what
I independently verified against the code, then lays out an ordered, green-at-every-step
remediation. Each milestone is a self-contained, committable unit; the full Cranelift
suite + `--features llvm` suite + clippy must stay green after each.

## 1. Verification of the review

I read the cited code for every P0/P1 finding, the two load-bearing factual claims, and
the benchmark. **The review is accurate.** Summary:

| # | Finding | Verdict | Evidence checked |
|---|---------|---------|------------------|
| 1 | Safe `div`/`rem`/`shl`/`shr` lower to poison/UB | **Confirmed** | `num/ops.rs:429/442/494/507` are safe ctors with no preconditions; `backend.rs` `sdiv/udiv/srem/urem` + `shli/shrsi/shrui` are bare `arith` ops; no `trap` exists |
| 2 | Niche-option `None` emits `I64`, not `Ptr` | **Confirmed (latent bug)** | `option.rs:358,403` emit `iconst(I64,0)`; `OptRefType/OptMutRefType::scalar_type()==Ptr`; `MatchOptRef` `brif`s the raw ptr (`:800`) and binds it into `declare_var(I64)` (`:807`). `coerce_to_bool` would build `IntegerAttribute::new(ptr_ty,0)` → invalid IR. Untested on LLVM (see #6). |
| 3 | `sum_above_median` hides `get_unchecked` behind safe `Box<dyn Fn>` | **Confirmed** | `benches/backends.rs` reads `a[n/2]`/`b[i]` unchecked; wrapper enforces neither non-empty `a` nor `b.len()>=a.len()`; native baseline uses `min(len)` — not equivalent |
| 4 | Silent size/offset narrowing | **Confirmed** | `backend.rs:616` `0..size as i32`; `:659` `bytes as i32`; `func_impl.rs:39/40` `size_of()/align_of() as u32` |
| 5 | `compile()` promises `Result`, LLVM path panics | **Confirmed** | `llvm/mod.rs` `jit_lookup` uses `assert!` (`:121,133,140,166,188,202`) + `.expect` (`:132,187`) for verify/lower/lookup |
| 6 | Backend-sensitive unit tests run Cranelift-only | **Confirmed** | 20 `Compiler::new()` in `src/option.rs`; same in `refer.rs`/`tuple.rs`/`ffi.rs`/`func.rs`. `--features llvm` does not change `Compiler::new()`'s default |
| 7 | `call_indirect` discards `SigSpec.params` | **Confirmed** | `backend.rs` `let (_, ret) = self.sigs[..]` — params dropped; no arg-count/type check before `llvm.call` |
| 8 | User names become linker symbols; `__main__` fixed | **Confirmed** | `mod.rs:385` emits `def.name`; `:399` hard-codes `__main__`; duplicate/`__main__`/extern collision → the panic path from #5 |
| 9 | 64-bit contract implicit; Windows LLVM untested | **Confirmed** | `Ptr` fixed 8 bytes → Cranelift `I64`; no `target_arch` guard; CI runs LLVM on 4 Linux/macOS, Cranelift-only on 2 Windows |
| 10–13 | Prototype public drivers, per-fn metadata cloning, dedup, fixed opt policy | **Confirmed** | 5 `pub fn jit_*` drivers; `assemble`/`build_function` clone decls per function; `JIT_OPT_LEVEL=2` hard-coded |
| doc | melior lacks `mem2reg`; header "no code yet" | **Confirmed the doc is wrong** | melior 0.27.4 `pass::transform` exposes `mlirCreateTransformsMem2Reg` → `create_mem2reg()`. `llvm.md:4` still says "design study … no code yet" |
| bench | `as_fn()` per iteration | **Confirmed** | `backends.rs:93` `move |a,z| compiled.as_fn().call(a,z)` re-resolves the trampoline each call |

**Nuances to carry into the plan** (things the review is right about, worth flagging):

- The `ptr_to_int` helper the review's #2 fix leans on **already exists** (added when
  sql-gen's pointer-compare gap was fixed). #2's remediation is mostly wiring it in +
  a `null_ptr` op, not new infrastructure.
- Several findings **overlap**: #12 ("one null-pointer op", "one memcpy op") is the same
  work as #2 and #4. #10's "one lowering pipeline" subsumes part of #5. The plan below
  merges them so nothing is done twice.
- #1 is a **contract decision**, not just a code fix — SQL `a/b` can divide by zero at
  runtime, so this is user-visible, not hypothetical. It needs a decision before code.
- This session has itself been **appending** to `llvm.md` — the "diary" problem in the
  doc section is real and partly self-inflicted; the rewrite is warranted.

## 2. Guiding constraints

- **Green at every step.** After each milestone: `cargo test --workspace`,
  `cargo test -p rust-lms --features llvm` (+ `-p sql-gen --features llvm`), `cargo clippy`.
- **Both backends or neither.** Every semantic fix lands on the neutral `Backend` trait
  with a Cranelift impl and an MLIR impl, and is differential-tested. No Cranelift type
  leaks into the AST.
- **Bounded, committable increments.** Milestones are ordered by the review's priority
  (soundness → hardening → cleanup → docs → bench) and sized to a single review.

## 3. Milestones

### M1 — Integer edge semantics (P0 #1) ⚠️ decision first

**Decision required** (pick one contract, applied on both backends):

- **(A) Checked + trap** — a neutral `Backend::trap()` (`trap` on Cranelift, `llvm.intr.trap`
  on MLIR); guard `div`/`rem` on zero divisor and signed `MIN/-1`; mask shift counts to
  `width-1`. Rust-like, safe-by-default. **Recommended.**
- **(B) Wrapping/masking** — mask shifts; define division-by-zero result explicitly (e.g.
  return 0 or a sentinel). Cheaper, less surprising to remove, but divergent from Rust.
- **(C) Unsafe split** — keep checked safe ops; add clearly-named `*_unchecked` ops used
  only where the SQL planner has proven the precondition. Most work, most control.

Recommendation: **(A)** for the safe constructors, with **(C)** as a follow-up only if the
trap guards show up in `sum`/`GROUP BY` hot loops. Shifts: mask in all cases (cheap, no UB).

Steps: add `Backend::trap()` (+ both impls); add guard helpers in `num/ops.rs` codegen (or
a lowering pass) for `Div`/`Rem`/`Shl`/`Shr`; mask shift counts. Add runtime-argument
**differential tests**: zero divisor, `i64::MIN / -1`, shift counts `width-1`, `width`,
`width+1`, for every integer width. Trap-path tests run in a subprocess (they abort).

### M2 — `null_ptr` op + pointer-shaped options (P0 #2, folds in #12's null-op)

- Add `Backend::null_ptr() -> ValueId`: Cranelift `iconst(I64,0)`; MLIR
  `llvm.mlir.zero : !llvm.ptr` (verify op name against LLVM 22 at impl time).
- `OptRefNone`/`OptMutRefNone::codegen` → `ctx.null_ptr()` (was `iconst(I64,0)`).
- `MatchOptRef`/`MatchOptMutRef`: bind with `declare_var(ScalarType::Ptr)` (was `I64`).
- Pointer branch conditions: extend `coerce_to_bool` to `ptr_to_int` a pointer condition
  first (or emit an `llvm.icmp ne null`), so `brif(ptr,...)` is valid MLIR.
- Delete the stale "single i64" comments in `option.rs`.
- Tests land with M4 (they need the crate-internal harness).

### M3 — Memcpy + de-narrowing at the memory boundary (P1 #4, folds in #12's memcpy)

- Replace the unrolled byte loop in `copy_nonoverlapping` with `llvm.intr.memcpy` (via
  `llvm.call_intrinsic` or `OperationBuilder`), i64 length + known alignment. Removes the
  `size as i32` narrowing and the "hundreds of byte ops" from #12.
- `ptr_offset_const`: use a dynamic i64 GEP when the constant doesn't fit MLIR's static
  i32 GEP; keep the static path for the common small case.
- `func_impl.rs`: validate `size_of`/`align_of` against the Cranelift `u32` stack-slot
  limit once while building `TypeInfo`; return `CompileError` on overflow instead of
  silent `as u32`. Audit remaining arena/operand `as u32`/`as i32` (use `try_into`).

### M4 — Crate-internal both-backends harness + close the test gap (P1 #6)

- Add a `#[cfg(test)] pub(crate) fn for_each_backend(...)` (mirrors the integration
  harness) so `src/` unit tests can run on both backends; convert the option/refer/tuple
  backend-sensitive tests, or relocate them to `tests/`.
- Add LLVM cases for: `OptRefType`/`OptMutRefType` `Some`/`None`/mutation (validates M2);
  pointer-valued `if_then_else` merges and pointer conditions; every scalar width; bool
  args/results; aggregates + nested aggregates; the M1 arithmetic edge policy; and
  deliberately-invalid duplicate symbols / indirect-call signatures (validates M5–M7).

### M5 — LLVM compile failures return `CompileError` (P1 #5, folds in #10's single pipeline)

- Split `jit_lookup` into `lower_module(...) -> Result<Module, CompileError>` and
  `create_engine_and_lookup(...) -> Result<_, CompileError>`; thread through `assemble`
  and `compile_llvm`.
- Replace `assert!`/`expect` on **module / pass / lookup / symbol** with `CompileError`
  carrying the function + pass stage. Install an MLIR diagnostic handler so verification
  messages are captured, not just printed to stderr. Keep genuinely-internal
  one-result-builder invariants as `debug_assert`.

### M6 — Stable internal symbol names (P1 #8)

- Emit internal symbols as `__rust_lms_fn_{id}`; keep the user name as debug metadata.
  Reserve namespaces for the `__main__` trampoline and extern thunks. Mark helper
  functions `private` in the module (stop exporting every user name). This removes the
  duplicate-name / user-`__main__` / extern-collision panic surfaced by #5/#8.

### M7 — Validate indirect-call signatures (P1 #7)

- Before emitting the indirect `llvm.call`, check `args.len() == SigSpec.params.len()` and
  each MLIR operand type against the expected param type; use checked conversion for
  `operandSegmentSizes`. Fail as a `CompileError` (compiler-internal invariant → also a
  `debug_assert` is acceptable, but a clean error is better given #5). Stretch: store the
  expected `ScalarType` alongside each arena value so all ops can diagnose neutral-IR type
  mistakes (bigger change — separate follow-up).

### M8 — Target contract guard (P1 #9)

- Add a compile-time guard: `compile_error!` unless `target_arch` is `x86_64`/`aarch64`,
  plus `const _: () = assert!(size_of::<usize>() == 8)`. Document the six supported
  targets in one place.
- CI/docs: either add a provisioned **x86_64-Windows MLIR** job, or explicitly state that
  Windows LLVM is unqualified and **stop claiming six-target LLVM**. Given Windows MLIR
  provisioning cost, recommend the honest-scope route now, x86_64-Windows LLVM as a
  stretch.

### M9 — Prototype-driver + metadata cleanup (P2 #10, #11, #12, #13)

- Make `llvm` and `MlirBackend` crate-private; move `jit_return_i64_const`,
  `jit_run_i64_unary`, `jit_eval_*` under `#[cfg(test)]`. One production lowering/JIT path
  (the M5 split), tested through `Compiler::with_backend(Llvm)`.
- Build one immutable module symbol table keyed by internal/extern ID; give each
  per-function backend a reference; drop `internal_funcs`/`externs`/most `FuncDecl`
  cloning (fixes the ~quadratic per-function copy in #11).
- Small dedups (#12): one append+intern-optional-result helper (direct/indirect calls);
  one checked arena-ID allocator; reuse the M2 `null_ptr` and M3 memcpy ops. **Keep** the
  explicit one-line opcode methods — they are deliberate audit points.
- Fix the false "opt-level 2 required for correctness" comment (mem2reg is an
  optimization; entry-block alloca load/store is valid unoptimized). If opt level ever
  becomes public, use a `CompilerOptions` value, not another builder method.

### M10 — Rewrite `docs/llvm.md` as current architecture

- Collapse to four short sections: **current architecture**, **build/toolchain setup**,
  **lowering & ABI invariants**, **supported targets & known semantic gaps**. Move the
  phase history to an archived implementation log (or delete).
- Fix the factual errors: drop "no code yet" header; correct the `mem2reg` claim (melior
  *does* expose it; using it is a pipeline choice); document the implemented
  `func.func → lower` pipeline (not the rejected `llvm.func`-direct alternative); state
  that `Compiled` owns the `Module` and `MlirExecutable` owns engine+context; make the CI
  description match reality (Linux LLVM jobs exist, Windows LLVM does not); fold in the M1
  arithmetic contract and the M8 target scope.
- Pin `melior = "=0.27.4"` and `mlir-sys` exactly together. Update `src/llvm` comments to
  describe current invariants, not milestones/"future" work that has shipped.

### M11 — Benchmark correctness, then re-assess conclusions (bench + #3)

- Resolve the JIT entry once (`as_fn()` outside the closure), or route native through
  equivalent dispatch, so the timed loop isn't paying trampoline overhead native skips.
- Fix `sum_above_median` (#3): assert non-empty `a` and `b.len() >= a.len()` in the safe
  wrapper (or encode `min(len)`/empty handling in the staged graph), and make the native
  baseline semantically equivalent.
- Add a **cold compile/lower/JIT** timing group; add shuffled-selectivity and
  larger-than-LLC data cases; label `Throughput` as rows/s; record CPU/OS/Rust/LLVM
  version/commit/CI with saved Criterion results.
- **Re-open the "slice representation is not a perf lever" conclusion**: inspect
  post-optimization LLVM IR/asm and benchmark representative sql-gen plans before letting
  that claim stand in the (rewritten) doc.

## 4. Sequencing & checkpoints

```
P0 soundness:   M1 (decision) → M2 → M3
P1 hardening:   M4 (harness+tests, validates M2) → M5 → M6 → M7 → M8
P2 cleanup:     M9
Docs & bench:   M10 → M11
```

M4 is placed right after the P0 fixes because its harness is what lets M2's pointer-option
fix (and M5–M7's error/validation paths) be differential-tested at the unit level — the
gap that let #2 hide in the first place. M1 is gated on a **contract decision**; I'll bring
the three options to you before writing code. Everything else is mechanical-but-careful and
proceeds under the green-at-every-step rule.

## 5. Not doing (yet), with rationale

- **Six-target LLVM CI including Windows** — deferred to M8's stretch; provisioning an
  MLIR 22 toolchain on Windows runners is high-cost and low-value versus documenting the
  honest scope now.
- **Per-arena-value type tracking** (#7 stretch) — larger refactor; the arg-count/type
  check at the call site closes the immediate hole; full type-carrying arena is a separate
  proposal.
- **Registering only required dialects** (#13) — measure cold-compile impact first; only
  worth it if it moves the needle.

## 6. A deeper simplification (recommended over piecemeal patching)

Stepping back from the individual findings, most of them are three root causes wearing
different hats. Fixing the root causes makes whole classes of these bugs *unrepresentable*
rather than patched one at a time — which is the "remove before you add" / "make invalid
stage-1 code impossible to express" bar the project already sets for itself.

### Root cause A — the neutral IR is untyped (dissolves #2, #4, #7; shrinks #6)

`ValueId` is a bare arena index (`u32`) with **no associated type**. Every consequence
below flows from that one fact:

- A pointer can be *born* as `I64` (`OptRefNone`) and nothing objects — that's #2.
- `icmp` / `coerce_to_bool` / `copy_nonoverlapping` can't know an operand's type from the
  neutral IR, so the MLIR backend **interrogates the backend** (`val.r#type()`) while
  Cranelift has no type at all. The `ptrtoint`/`inttoptr`/`coerce_to_bool` dance exists
  purely to recover types the IR threw away.
- `call_indirect` *can't* validate its args because there is nothing to validate against —
  that's #7, and the review's own suggested fix is "store the expected type alongside each
  arena value."
- Sizes/offsets flow as untyped ints that get silently narrowed — that's #4.
- And because the neutral IR never type-checks, a wrong lowering only fails at *MLIR
  verification of an untested path* — which is exactly how #2 stayed hidden (#6).

**Proposal:** make the arena value carry its neutral type — `ValueId → (index, ScalarType)`
(or a small `Type` for fat pointers/aggregates). Then:

- `null_ptr` yields a `Ptr`; an `I64` zero at a `Ptr` merge is a **construction-time error**,
  not invalid MLIR discovered later. #2 becomes unrepresentable.
- `icmp`/branch/copy dispatch on the *neutral* type identically on both backends; the
  `ptr_to_int`/`coerce_to_bool` special-casing collapses into "the op already knows it's a
  pointer." The two backends stop diverging in how they recover types.
- `call_indirect` checks args against `SigSpec.params` for free (#7).
- This also retires the `u64`-as-pointer smuggling in the opaque-iterator vtable — which
  CLAUDE.md explicitly forbids ("a `u64` smuggled around to avoid [a clean type] is not
  [a good trade]"). The vtable would hold typed pointers end-to-end.

**Migration is incremental and low-risk**, mirroring the original Phase 0 `ValueId` flip:
add the type as an ignored field first (everything compiles, behaviour identical), then turn
on assertions op-by-op, then delete the now-dead `ptrtoint`/`coerce_to_bool` recovery code.
If this lands, **M2, M4, and M7 mostly fall out of it** rather than being separate work.

#### A′ — the higher-ambition version: a *structured* value (also fixes the slice side-channel)

A `ScalarType` on `ValueId` catches type errors but still can't let one neutral value *be*
two machine values. That single-value contract (`codegen -> ValueId`) is why slices need a
**side-channel**: a Cranelift slice is already `(ptr, len)` in two variables, but they're
smuggled through `CompilationContext::slice_vars` keyed by the staged var-id
(`staged.rs:200,353-383`) because `codegen` can only return one `ValueId`. The opaque-iter
vtable smuggles function pointers as `u64` for the same reason.

The more fundamental simplification is to make a neutral value a **small typed tree of
backend SSA leaves**:

```
enum Value { Scalar(Leaf), Fat(Leaf /*data ptr*/, Leaf /*len*/), /* aggregates stay by-ptr */ }
```

where each `Leaf` is a single backend SSA value carrying its `ScalarType`. Then:

- a slice is genuinely `Fat(ptr, len)` — two registers on Cranelift, two SSA values on
  LLVM — and the whole slice side-channel (`slice_vars`, `slice_data_ptr`/`slice_len`, the
  memory-vs-register resolution "invariant") **collapses into one abstraction**;
- the opaque-iter `u64` vtable becomes typed pointer leaves — retiring the `u64`-as-pointer
  smuggling CLAUDE.md forbids (the same win claimed for A, now structural);
- multi-value returns (the "multi-register opaque iter items" gap in the project memory)
  are expressible;
- backends stay **scalar-only** — they only ever see leaves; the neutral layer owns shape.

`A′` subsumes `A` (leaves are typed). Its cost is higher: it changes the `Staged::codegen`
return type, touching every `Staged` impl — but it migrates the same incremental way (`Value`
starts as a scalar-only newtype around one leaf → everything compiles unchanged → add `Fat`
→ move slices off `slice_vars` → move opaque-iters off the `u64` vtable). At ABI boundaries a
`Fat` still materializes to a memory `{ptr,len}` under the storage-pointer ABI, exactly as
LLVM lowers an SSA `{ptr,len}` at call edges.

**Caveat — this is a simplicity/correctness play, not a perf one.** The benchmark showed
Cranelift's gap is autovectorization, and slice *params* are already register-resolved, so
`A′` is unlikely to move throughput. Its payoff is deleting three bespoke mechanisms and one
CLAUDE.md violation, and making `A`'s type-safety fall out of the same value type. Choose `A`
if the goal is just to close the correctness findings cheaply; choose `A′` if the goal is the
"fewer, cleaner pieces" simplification — it is the one that makes slices first-class `ptr+len`.

### Root cause B — "safe" ops carry undocumented runtime preconditions (dissolves #1, #3)

The type system encodes staged *types* but not staged *safety*. Today `div`, `rem`, `shl`,
`shr`, and `get_unchecked`-behind-a-safe-closure are all expressible through safe APIs that
can hit UB. That contradicts the project's defining claim that invalid stage-1 code is
unrepresentable.

**Proposal:** make the checked/unchecked boundary a *type-level* distinction, not a naming
convention. Safe ops are always UB-free (M1's guards); unchecked ops are separate, named,
and only reachable where a precondition has been established (e.g. the SQL planner proved
`divisor != 0`, or a bounds check dominates the read). This is #1's option (C) generalized
into a principle, and it's what makes #3 (unchecked read behind a safe `Box<dyn Fn>`) a
type error instead of a latent OOB.

### Root cause C — the LLVM path re-derives what the driver already computed (dissolves #5, #8, #10, #11)

The Cranelift path is structured through `func.rs`; the MLIR path is a parallel assembler
that panics, re-registers all dialects per compile, and **clones the whole symbol/decl set
into every per-function backend** (~quadratic, #11). Responsibilities between `func.rs`,
`llvm/mod.rs`, and `MlirBackend` overlap.

**Proposal:** a clean three-layer split, computed once and shared:

1. neutral AST → neutral op-stream (unchanged);
2. a backend-agnostic **module plan** — the symbol table (stable internal names, #8), the
   `SigSpec`s, the ABI/aggregate decisions — built **once**;
3. a backend is a pure consumer of (op-stream + module plan) that returns
   `Result<_, CompileError>` uniformly (#5).

This makes #5/#8 structural rather than patched, deletes the per-function cloning (#11),
and gives one lowering pipeline (#10) into which the prototype `jit_*` drivers fold as
`#[cfg(test)]` helpers.

### How this changes the plan

The three redesigns are **independent** and each stands alone, but A is the highest-leverage
(it touches the most findings and aligns with the project's core principle). Two ways to
sequence:

- **Tactical-first** (M1–M11 as written): faster to green on each individual finding, but
  M2/M4/M7 get done twice if A lands later.
- **Foundation-first** (recommended): M1's contract decision → **Root cause A (typed IR)** →
  then M2/M4/M7 largely fall out → then C (M5/M8/M10/M11 restructure) → B as the arithmetic
  contract from M1 hardens into a type-level split. Slightly more up-front, but several
  milestones shrink or disappear, and the result is *simpler than today*, not just patched.

Recommendation: do the **urgent P0 soundness** parts of M1 (guards) immediately regardless —
they are user-visible UB — but adopt **foundation-first** for the rest, leading with the
typed neutral IR. This is a genuine fork worth deciding before writing code.

### Spike result (branch `value_refactor`, increment 1 — DONE, green)

Ran the first step of `A′`: introduced `Value` as a scalar-only newtype
(`struct Value(ValueId)` with `scalar()`/`leaf()`), flipped `Staged::codegen -> Value` and
the capability layer (`Num`/`IntNum`/`FloatNum::codegen_*`, `ConstantType::codegen_constant`)
to speak `Value`, and kept slices on the existing `slice_vars` side-channel (the `Fat` variant
is increment 2). Result: **green on both backends, behaviour bit-identical** — rust-lms 236
Cranelift / 255 LLVM, workspace 366, sql-gen 113 on both, clippy 27 (baseline).

Findings, i.e. "is it as clean as the `ValueId` flip?":

- **The seam held.** The ~87 AST composition impls (`Add`, `Sub`, …) needed **zero** changes —
  they pass `Value`s through capability methods, which absorbed the wrap/unwrap. Confirms the
  capability layer is the right concentration point.
- **The churn is real but purely mechanical and compiler-verifiable.** ~184 wrap/unwrap sites
  at the `Value`↔leaf boundary; **156 were auto-fixed** by a script driving off `cargo`'s own
  error spans (`.leaf()` where a leaf is expected, `Value::scalar(...)` where a `Value` is),
  the rest were a handful of multi-arg backend ops + the `call` arg arrays + the capability
  macros (one `perl` each). No behavioural judgement was needed anywhere.
- **It is more invasive than the `ValueId` flip** (which touched only the backend) because the
  AST genuinely constructs and consumes values — but every edit was at a leaf boundary the
  compiler pinpointed, and the two backends were untouched (`Backend` still speaks leaves).
- External `Staged` impls (in `tests/`, `arrow-lms`) needed only their `-> ValueId` return type
  renamed to `-> Value`; they delegate, so no wrapping.

**Verdict: green-light `A′`.** The rails are in. Increment 2 — add `Value::Fat(ptr, len)`, move
slices off `slice_vars`, then opaque-iters off the `u64` vtable — is where the payoff (deleting
the side-channels) lands; this increment proved the contract change is safe and mechanical.

### A′ increment 2 — DONE, green

Completed the structured-value migration started by `d05e6ad`:

- `Value::Fat { ptr, len }` is now the canonical in-kernel slice representation. Slice
  parameters, locals, sub-slices, block merges, internal calls, extern calls, option
  payloads, fields, and memory loads/stores preserve that shape.
- The separate `var_map` / `slice_vars` stores are one shape-aware variable map. The unused
  `Staged::var_id` escape hatch and the old register-vs-memory slice resolution path are gone.
- Generic value transport is centralized in `load_value`, `store_value`, block-value helpers,
  and ABI materialization. This closes scalar-only assumptions that were latent in function
  returns, `if_then_else`, option matching, references, slice elements, and zip items.
- Opaque iterator vtable/data fields are loaded as `Ptr`, not `I64`. Opaque iterators can now
  carry structured copy items; `RegisterScalar` remains only as a deprecated source-compatible
  bound over the broader `OpaqueIterItem` contract.
- Regression coverage includes internal and extern fat-slice returns, fat values through
  conditionals and `COption`, slices containing fat-slice elements, and opaque iterators
  yielding fat slices.

Verification: `cargo test --workspace`, `cargo test -p rust-lms --features llvm`, and
`cargo test -p sql-gen --features llvm` all pass. Plain workspace clippy completes with the
existing warning baseline; `-D warnings` remains blocked by pre-existing lints.

**Still pending in Root A:** replace bare `ValueId` leaves with `(backend id, ScalarType)` and
enable neutral-layer type assertions operation by operation. Increment 2 makes value *shape*
explicit; typed leaves are the remaining step that makes pointer-vs-integer mistakes
construction-time errors on both backends.
