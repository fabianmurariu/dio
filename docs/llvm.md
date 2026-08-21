# Pluggable codegen backend: Cranelift or LLVM (MLIR via melior)

Written: 2026-08-21
Status: design study — feasibility + plan, no code yet.

## The question

Today every staged construct lowers itself directly to Cranelift IR:
`Staged::codegen(&self, ctx: &mut CompilationContext) -> cranelift_codegen::ir::Value`.
Can we introduce an abstraction so a caller chooses the backend — Cranelift (current)
or **LLVM through [melior](https://mlir-rs.github.io/melior/melior/) (MLIR)** — and is it
even worth it? Short answer: **yes, it is possible, and the current boundary is
unusually well-placed for it, but it is a medium-to-large effort whose real cost is
concentrated in one place (the SSA/variable model) and one non-code implication (a
heavyweight system LLVM dependency).**

## TL;DR verdict

- **Feasible.** ~90% of the coupling is a mechanical opcode/type/JIT mapping that any
  SSA IR builder exposes near-identically.
- **One hard part:** rust-lms leans on Cranelift's *automatic mutable-variable SSA*
  (`declare_var`/`def_var`/`use_var` + `seal_block` phi insertion). MLIR/LLVM have no
  such abstraction. The portable answer is **`llvm.alloca` + load/store and let LLVM's
  `mem2reg` promote to SSA** — the same thing Clang does for locals. That dissolves
  both the variable problem *and* the `seal_block` protocol.
- **Recommended shape:** a **trait object `dyn Backend` + an opaque, lifetime-free
  `ValueId`** that `codegen` returns instead of a raw `cranelift ... Value`. Not a
  generic `B: Backend`. Rationale below — the deciding factor is melior's
  `Value<'c,'a>` lifetimes.
- **Biggest non-code implication:** melior needs a **system install of LLVM/MLIR 22**
  (`llvm-config` / `MLIR_SYS_220_PREFIX`), versus Cranelift's pure-Rust build. This
  reshapes the build, CI, and contributor onboarding. Put the whole thing behind a
  `--features llvm` gate.
- **Payoff:** LLVM's optimizer (far better generated code, autovectorization, more
  mature targets) at the cost of much slower compile/JIT times. Cranelift stays the
  fast-compile default.

## 1. What is coupled to Cranelift today (measured)

From an inventory of `rust-lms/src`:

- **16 files** reference `cranelift`; **142** `.ins().<op>()` call sites across **39
  distinct instruction builders**. Densest: `num/traits.rs` (40), `func.rs` (21),
  `slice.rs`/`option.rs` (15 each), `refer.rs` (10), `types.rs` (9).
- **The one pervasive coupling** is the return type of `Staged::codegen` —
  `cranelift_codegen::ir::Value` (`staged.rs:243`). Everything else hangs off that.
- **`CompilationContext`** (`staged.rs:65`) is the backend surface. Every field is
  already `pub(crate)` and downstream `Staged` impls are *compile-fail-forbidden* from
  touching `builder` (`staged.rs:58`). Fields worth naming: `builder:
  &mut FunctionBuilder`, `module: &mut JITModule`, `var_map: HashMap<usize, Variable>`,
  `slice_vars: HashMap<usize, SliceVars{ptr_var, len_var}>`, `loop_exit_stack:
  Vec<Block>`, plus the single-source-of-truth slice helpers `slice_data_ptr`/
  `slice_len`/`slice_parts`.
- **`dyn Staged` is re-exported (`BoxableStaged`) but never actually used.** The real
  type erasure is over *closures*: `FunDef.body: Box<dyn FnOnce(&mut
  CompilationContext) -> Value>` and `CodegenAction = Box<dyn FnOnce(&mut
  CompilationContext)>`. **This matters for the generic-vs-trait decision (§4).**
- **Instruction selection already lives behind Rust trait methods** (`Num`/`IntNum` in
  `num/traits.rs` pick `sdiv`/`udiv`/`fdiv`; casts pick `sextend`/`uextend`/`ireduce`
  by comparing sizes). A second backend mostly re-implements those method bodies.
- **Types** are already flattened: `cranelift_type()` returns `I64` ×79, `I8` ×12,
  `I32` ×5, `I16` ×3, `F64`/`F32` ×2. **All pointers, slices, and struct handles are
  `I64`; `bool` and `()` are `I8`.** There are no reference IR types.
- **ABI** is one uniform convention (Phase 3): every logical parameter is one `I64`
  storage pointer, plus a trailing `I64` output pointer, call conv =
  `isa().default_call_conv()`. This is the crucial detail for the JIT mapping (§7).
- **JIT** (`func.rs:1086`): `cranelift_native` ISA → `JITBuilder`/`JITModule` →
  declare/define/`finalize_definitions`/`get_finalized_function` → raw `*const u8`.
  `Compiled` owns the module and frees code memory on `Drop`.

**Conclusion:** the abstraction seam is already drawn. The hard content is §5.

## 2. The two backends are different in three ways that matter

| Concern | Cranelift (today) | MLIR via melior |
|---|---|---|
| Value handle | `ir::Value` — **lifetime-free, `Copy`, integer-like** | **`Value<'c,'a>`** — two lifetimes (Context + block/region), `Copy` |
| Mutable variables | native `declare_var`/`def_var`/`use_var`; auto-SSA | **none** — build block-arg phis, or `llvm.alloca`+load/store + `mem2reg` |
| SSA sealing | explicit `seal_block` (Braun algorithm) | **no sealing**; block args wired at branch sites |
| Control flow | blocks + `brif`/`jump` + block params | `cf` dialect: `cf.br`/`cf.cond_br` + block arguments (direct analogue); or `scf` structured |
| Types | `types::I64/I8/...` | `IntegerType`, `FloatType`, `llvm.ptr` (via `arith`/`llvm` dialects) |
| Aggregate copy | `emit_small_memory_copy` | `llvm.intr.memcpy` / explicit loads+stores |
| JIT | `JITModule` → `get_finalized_function` → `*const u8` | `ExecutionEngine::new(module, opt, libs, …)` → `invoke_packed(name, &mut [*mut ()])` |
| Build | pure Rust | **system LLVM/MLIR 22** (`llvm-config` / `MLIR_SYS_220_PREFIX`) |

The **value-lifetime** row is the one that dictates the whole design.

## 3. The central problem: representing "a value" across backends

`Staged::codegen` must return *something* both backends can produce. Cranelift's
`Value` is a lifetime-free `Copy` handle — trivial to thread through a recursive AST
and store in maps. melior's `Value<'c,'a>` borrows from the `Context` (`'c`) and the
block/region it lives in (`'a`). If `codegen` returned that, the two lifetimes would
infect **every** signature in the crate, and a backend that owns its `Context`/blocks
*and* stores `Value<'c,'a>` handles in an arena is self-referential — it will not
borrow-check.

**Solution: an opaque, lifetime-free value handle.**

```rust
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct ValueId(u32);        // index into the backend's own arena
```

`codegen` returns `ValueId`. Each backend keeps its real values in an internal arena
keyed by `ValueId`:

- Cranelift backend: `Vec<cranelift ... Value>`.
- MLIR backend: `Vec<mlir_sys::MlirValue>`. **`MlirValue` (the mlir-sys C handle) has
  no lifetime — it is a plain `Copy` struct wrapping a raw pointer.** melior's
  `Value<'c,'a>` only *adds* those two lifetimes as a wrapper convenience to model
  MLIR's ownership in the type system. So storing the raw `MlirValue` in an arena is
  **not self-referential** and needs no lifetime gymnastics: the backend owns the
  `Context`/`Module` for the whole compile, and we wrap a raw handle back into a
  `Value<'c,'a>` (with `'c`/`'a` reborrowed from the backend) only at the moment we
  pass it as an operand to `melior`'s op builders. The single invariant is "the
  `Context` outlives its handles," which the backend guarantees — a runtime validity
  contract exactly like Cranelift's `Value` being an unchecked arena index, **not** a
  borrow the compiler must track.

This indirection is the keystone. It makes the AST backend-agnostic and dissolves the
melior lifetime infection at one well-defined boundary. `VarHandle`, `BlockHandle`,
`FuncHandle` get the same treatment (opaque `u32` ids into backend arenas).

## 4. Generic (`B: Backend`) vs trait object (`dyn Backend`)?

Both are technically open now that we know `dyn Staged` is unused. They trade off:

**Generic `B: Backend` (monomorphized).** `CompilationContext<B>`, `Compiler<B>`, and
`Staged::codegen<B>(…) -> B::Value`. Zero-cost dispatch. But: (a) it makes `<B>` viral
across the whole library and the closure types (`Box<dyn FnOnce(&mut
CompilationContext<B>)>`); (b) it monomorphizes the *entire* AST + iterator machinery
per backend (compile time, binary size); (c) `B::Value = Value<'c,'a>` still drags
melior's lifetimes into `CompilationContext<B>` unless we *also* adopt the `ValueId`
indirection — at which point the generic buys us nothing over a trait object.

**Trait object `&mut dyn Backend` + `ValueId` (recommended).**
`Staged::codegen(&self, ctx: &mut CompilationContext) -> ValueId`, where
`CompilationContext` holds `&mut dyn Backend`. One AST, no viral generics, and — the
decisive point — **the dynamic dispatch is a stage-1 (compile-time) cost only**. It
happens once while *emitting* IR; the generated native code contains no dispatch. IR
emission is not a hot path, so the vtable calls are irrelevant. Meanwhile `ValueId`
keeps melior's lifetimes fully contained.

**Recommendation: trait object + `ValueId`.** The only thing the generic form wins
(monomorphized emit) is worthless because emit runs once, and it loses on compile
time, binary size, and lifetime containment.

## 5. The hard part: mutable variables, phis, and loops

This is where Cranelift semantics genuinely differ and most of the design work lives.

**What rust-lms does today.** `Var::codegen` is `use_var`; `assign`/`Ctx::var`/`store`
do `declare_var`+`def_var`; loop induction variables mutated across a back-edge rely
entirely on Cranelift to insert the header phi. `While::codegen` performs the delayed
`seal_block` dance (seal the loop header only after the back-edge jump exists).
`IfThenElse` is the one explicit phi: it `append_block_param(merge, T)` and reads it
back.

**MLIR has no `Variable` and no sealing.** Two ways to bridge:

- **(a) `llvm.alloca` + load/store + `mem2reg` (recommended).** Map the Backend
  variable API onto memory: `declare_var` → `llvm.alloca`; `def_var` → `llvm.store`;
  `use_var` → `llvm.load`; **`seal_block` → no-op**. Then rely on LLVM's `mem2reg`/SROA
  (which the ExecutionEngine's optimization level runs) to promote every alloca to real
  SSA registers with correct phis — *exactly* how Clang lowers C locals. This makes the
  entire imperative-variable and loop-phi story disappear on the MLIR side with no
  hand-threaded phis and no sealing protocol. `IfThenElse`'s merge value can also be an
  alloca (store in each arm, load after) and get promoted, so even the one explicit phi
  becomes uniform.
- **(b) Hand-built block-argument phis.** MLIR `cf` blocks take arguments (true phis),
  so in principle we could thread loop-carried values as block args. This is a much
  larger rewrite (every mutated variable must be discovered and threaded through every
  branch) and essentially re-implements Cranelift's SSA construction by hand. Reject it
  for variables; optionally use `cf` block args only for the `IfThenElse` merge if we
  want tighter pre-optimization IR.

**Net:** the Backend trait keeps Cranelift's variable vocabulary
(`declare_var`/`def_var`/`use_var`/`seal_block`) as the common denominator. The
Cranelift impl forwards to native calls; the MLIR impl implements them as
alloca/store/load with a no-op seal. Control-flow blocks + branches map to `cf` dialect
directly (MLIR blocks natively support the block-parameter phi that `IfThenElse` uses).

## 6. The `Backend` trait (sketch)

Object-safe, ~45–55 methods, grouped. All handles are opaque ids.

```rust
pub trait Backend {
    // ---- types ----
    fn ty_int(&self, bits: u16) -> TypeId;      // I8/I16/I32/I64
    fn ty_float(&self, bits: u16) -> TypeId;    // F32/F64
    fn ty_ptr(&self) -> TypeId;                 // Cranelift: I64; MLIR: llvm.ptr

    // ---- constants ----
    fn iconst(&mut self, ty: TypeId, v: i64) -> ValueId;
    fn fconst(&mut self, ty: TypeId, v: f64) -> ValueId;

    // ---- arithmetic / bitwise / shift (39 ops collapse to ~20 methods) ----
    fn iadd(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn imul(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn idiv(&mut self, signed: bool, a: ValueId, b: ValueId) -> ValueId;   // sdiv/udiv
    fn irem(&mut self, signed: bool, a: ValueId, b: ValueId) -> ValueId;
    fn fadd(&mut self, a: ValueId, b: ValueId) -> ValueId; // …fsub/fmul/fdiv
    fn band(&mut self, a: ValueId, b: ValueId) -> ValueId; // …bor/bxor
    fn ishl(&mut self, a: ValueId, b: ValueId) -> ValueId; // …sshr/ushr

    // ---- compare / select ----
    fn icmp(&mut self, cc: IntCc, a: ValueId, b: ValueId) -> ValueId;
    fn fcmp(&mut self, cc: FloatCc, a: ValueId, b: ValueId) -> ValueId;
    fn select(&mut self, c: ValueId, a: ValueId, b: ValueId) -> ValueId;

    // ---- casts ----
    fn iextend(&mut self, signed: bool, to: TypeId, v: ValueId) -> ValueId; // sextend/uextend
    fn ireduce(&mut self, to: TypeId, v: ValueId) -> ValueId;
    fn fcvt_from_int(&mut self, signed: bool, to: TypeId, v: ValueId) -> ValueId;
    fn bitcast(&mut self, to: TypeId, v: ValueId) -> ValueId;

    // ---- memory ----
    fn load(&mut self, ty: TypeId, ptr: ValueId, offset: i32) -> ValueId;
    fn store(&mut self, val: ValueId, ptr: ValueId, offset: i32);
    fn stack_alloc(&mut self, size: u32, align: u8) -> ValueId;     // stack_addr / llvm.alloca
    fn memcpy(&mut self, dst: ValueId, src: ValueId, size: u32, align: u8);

    // ---- variables (see §5) ----
    fn declare_var(&mut self, ty: TypeId) -> VarHandle;
    fn def_var(&mut self, v: VarHandle, val: ValueId);
    fn use_var(&mut self, v: VarHandle) -> ValueId;

    // ---- blocks / control flow ----
    fn create_block(&mut self) -> BlockHandle;
    fn append_block_param(&mut self, b: BlockHandle, ty: TypeId) -> ValueId;
    fn switch_to_block(&mut self, b: BlockHandle);
    fn seal_block(&mut self, b: BlockHandle);          // MLIR: no-op
    fn brif(&mut self, c: ValueId, then_: BlockHandle, else_: BlockHandle);
    fn jump(&mut self, target: BlockHandle, args: &[ValueId]);
    fn block_param(&mut self, b: BlockHandle, i: usize) -> ValueId;

    // ---- calls / functions ----
    fn call(&mut self, f: FuncHandle, args: &[ValueId]) -> Option<ValueId>;
    fn call_indirect(&mut self, sig: SigId, callee: ValueId, args: &[ValueId]) -> Option<ValueId>;
    fn func_addr(&mut self, f: FuncHandle) -> ValueId;
    fn ret(&mut self, v: Option<ValueId>);
}

// A separate object for the compile lifecycle (declare/define/finalize/JIT):
pub trait Module {
    fn declare_function(&mut self, name: &str, sig: &SigSpec, linkage: Linkage) -> FuncHandle;
    fn define_function(&mut self, f: FuncHandle, body: impl FnOnce(&mut dyn Backend));
    fn register_extern(&mut self, name: &str, ptr: *const u8);
    fn finalize(self) -> Box<dyn Compiled>;    // owns exec memory; frees on Drop
}
```

`CompilationContext` keeps its backend-neutral maps (`var_map: HashMap<usize,
VarHandle>`, `loop_exit_stack: Vec<BlockHandle>`, `slice_vars`, the slice helpers) and
holds a `&mut dyn Backend`. The ~hundreds of `Staged::codegen` bodies change only
mechanically (`ctx.builder.ins().iadd(a,b)` → `ctx.backend.iadd(a, b)`); the `Num`/
`StagedType` trait methods absorb the signed/unsigned/float selection they already do.

## 7. ABI & JIT — the favorable part

rust-lms's Phase-3 ABI is **one uniform convention: N storage pointers + one output
pointer, void return.** MLIR's JIT entry point is
`ExecutionEngine::invoke_packed(name, &mut [*mut ()])`, whose "arguments" are *pointers
to arguments and results*. **These are the same convention.** So the MLIR backend:

1. Emits each JIT function as an `llvm.func` taking `llvm.ptr` params + an output
   `llvm.ptr`, returning void — mirroring the existing signature builder.
2. Lowers to the LLVM dialect with a `PassManager` (`convert-scf-to-cf` if used,
   `convert-cf-to-llvm`, `convert-arith-to-llvm`, `convert-func-to-llvm`,
   `finalize-memref-to-llvm`, plus `mem2reg`/canonicalize for §5), then
   `register_all_llvm_translations`.
3. Builds `ExecutionEngine::new(&module, opt_level, &[], false, false)`, registers each
   host extern with `engine.register_symbol(name, ptr)` (the analogue of
   `JITBuilder::symbol`), and invokes via `invoke_packed` with a `[*mut ()]` slice
   built from the same storage pointers `Compiled::run`/`CompiledFn::call` already
   marshal.
4. `Compiled` for MLIR owns the `ExecutionEngine` (+ `Context`/`Module`) and drops
   them together — the same "owns executable memory, frees on Drop" contract as the
   Cranelift `JITModule`.

Because `as_fn_unchecked`'s by-pointer signature is exactly this shape, most of
`func.rs`'s `Compiled`/`CompiledFn` glue is backend-neutral already; only the module
finalize + entry-lookup differ.

## 8. Type mapping

| rust-lms | Cranelift | MLIR |
|---|---|---|
| `I8` (bool, unit) | `types::I8` | `IntegerType::new(ctx, 8)` |
| `I16/I32/I64` | `types::I16/I32/I64` | `IntegerType::new(ctx, 16/32/64)` |
| `F32/F64` | `types::F32/F64` | `FloatType` (`Float32Type`/`Float64Type`) |
| pointer / slice ptr / struct handle (all `I64`) | `types::I64` | `llvm.ptr` (opaque pointer) — or keep `i64` and `inttoptr` at loads |

Decision to make in Phase 1: keep pointers as `i64` end-to-end (closest to today, one
`inttoptr` per memory op) **or** adopt `llvm.ptr` (more idiomatic, better alias info).
Recommend `llvm.ptr` for loads/stores/calls and `i64` only where an address is
arithmetic — but this is a local choice inside the MLIR backend, invisible to the AST.

## 9. Implications (the honest list)

- **Build & CI (biggest).** melior 0.27 / mlir-sys 220 require **LLVM/MLIR 22 installed
  on the machine** and discoverable via `llvm-config` or `MLIR_SYS_220_PREFIX`. That is
  a multi-hundred-MB system dependency, a new CI provisioning step on all six target
  runners, and a real onboarding cost. **Mitigation:** gate the entire MLIR backend
  behind `--features llvm` (off by default). The pure-Rust Cranelift build stays the
  default; `cargo build`/`test` are unaffected unless you opt in.
- **Compile/JIT time.** LLVM produces much better code but JITs far slower than
  Cranelift (seconds vs milliseconds for large kernels). This flips rust-lms's "fast
  staging" value prop, so LLVM is the "optimize hard, run many times" mode, Cranelift
  the "compile fast" default. Worth exposing the choice per-`Compiler`, not globally.
- **`unsafe` containment.** Because `MlirValue` carries no lifetime, the
  `ValueId`→raw-`MlirValue` arena is an ordinary (safe) `Vec`; the only obligation is
  the runtime invariant "the backend's `Context` outlives its handles," which the
  backend owns for the compile's duration — the same shape as Cranelift's `Value`
  being an unchecked arena index. No lifetime leaks into the AST, and there is no
  self-referential-struct problem to work around.
- **Debugging.** `RUST_LMS_DEBUG_IR` becomes backend-specific: Cranelift CLIF vs MLIR
  (pre- and post-lowering) textual IR. Keep the env var; print whichever backend is
  active.
- **Correctness/testing.** The entire existing test suite becomes a **differential
  oracle**: run `tests/` against both backends and assert identical results. This is
  the single most valuable safety net and should gate the MLIR backend's maturity. The
  Phase-3 `test_abi.rs` shapes are especially important across backends.
- **The `seal_block` invariant** (CLAUDE.md's "#1 cause of Cranelift panics") becomes a
  no-op on MLIR — the Backend trait keeps the call so the Cranelift impl stays correct,
  but the invariant simply doesn't exist for MLIR (block args are explicit).
- **Scope of the mechanical edit.** ~142 `.ins()` sites across 16 files get rewritten
  to `ctx.backend.*`. Large but rote, and mostly concentrated in `num/traits.rs`,
  `func.rs`, `slice.rs`, `option.rs`, `refer.rs`.
- **Not in scope / won't-port cleanly:** nothing fundamental. `emit_small_memory_copy`
  becomes an explicit `memcpy` method. `stack_addr` becomes `llvm.alloca`.

## 10. Phased plan

**Phase 0 — introduce the seam, Cranelift-only (pure refactor, tests stay green).**
Define `Backend`/`Module`/`Compiled` traits + `ValueId`/`VarHandle`/`BlockHandle`.
Implement them as a thin `CraneliftBackend` wrapping today's code. Change
`Staged::codegen -> ValueId` and rewrite the 142 sites to `ctx.backend.*`. No behavior
change; the whole suite must stay green. **This is the bulk of the work and it is
backend-count-independent — worth doing even if LLVM never ships**, because it also
makes the codegen unit-testable and documents the real backend contract.

**Phase 1 — MLIR backend skeleton behind `--features llvm`.** Context/Module setup,
type mapping (§8), constants, arithmetic/bitwise/compare/select/casts, memory
(alloca/load/store), and the arena/`ValueId` plumbing. Prove a nullary `fun0`
returning a constant JITs and runs via `ExecutionEngine`.

**Phase 2 — variables & control flow (§5).** `declare/def/use_var` as alloca+load/store,
`seal_block` no-op, blocks/`brif`/`jump`/block-params via `cf`, plus the lowering
passes incl. `mem2reg`. Land `if_then_else`, `while_loop`, `break_loop`. Differential
test against Cranelift on `tests/programs.rs`, `p99.rs`, `euler.rs`.

**Phase 3 — calls, externs, ABI, JIT.** `call`/`call_indirect`/`func_addr`, extern
symbol registration, the storage-pointer signatures, and `invoke_packed`. Land
`test_extern_fn.rs`, `test_abi.rs` across both backends.

**Phase 4 — parity & choice.** Differential-test the full suite on both backends; expose
`Compiler::with_backend(Backend::Cranelift | Backend::Llvm)`; decide the six-target CI
story (Cranelift on all six; LLVM where an MLIR 22 toolchain is provisioned).

**Rough effort:** Phase 0 is the large one (mechanical but wide, touches the whole
codegen surface + the SSA-var redesign of the trait). Phases 1–3 are a few focused
weeks each for someone comfortable with MLIR. Phase 0 delivers standalone value
(testable, documented backend boundary) even before any LLVM code exists.

## 11. Open questions / risks

- **`mem2reg` reliability.** The whole variable story bets on LLVM promoting our allocas.
  It will for straight-line and structured loops (Clang relies on it), but pathological
  IR could leave stack traffic. Verify with `-O2` on the benches early in Phase 2.
- **`llvm.ptr` vs `i64` pointers** (§8) — pick once, early; affects every memory op in
  the MLIR backend (but not the AST).
- **melior/MLIR version churn.** melior tracks LLVM major versions aggressively (0.27 =
  LLVM 22). Pin it; expect periodic bumps.
- **Two-backend maintenance tax.** Every new staged op now needs a method on `Backend`
  and two impls. The differential test suite is what keeps them honest.
- **Is it worth it?** If the goal is *faster* compilation or more portability with a
  pure-Rust build, Cranelift already wins and this is not worth it. If the goal is
  *maximum generated-code quality* (vectorization, LLVM's optimizer) or reusing the MLIR
  ecosystem (custom dialects, GPU/accelerator lowering), the LLVM backend is the way,
  and Phase 0 is a good investment regardless.

## Sources

- melior crate docs — <https://mlir-rs.github.io/melior/melior/> (`Value<'c,'a>`,
  `ExecutionEngine::{new, register_symbol, invoke_packed}`, dialects `arith/func/llvm/
  scf/cf/memref/index`, "no Cranelift-style variable abstraction").
- mlir-sys / LLVM requirement — <https://github.com/mlir-rs/mlir-sys>,
  <https://crates.io/crates/mlir-sys> (LLVM/MLIR 22, `llvm-config` / `MLIR_SYS_220_PREFIX`).
- Internal inventory of `rust-lms/src` (this study): 16 Cranelift files, 142 `.ins()`
  sites, 39 ops; `Staged::codegen -> cranelift ... Value`; `CompilationContext`
  fields; Phase-3 storage-pointer ABI.
