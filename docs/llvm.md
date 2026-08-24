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
- **Payoff (hypothesis, to be measured):** LLVM's optimizer *may* produce better code
  (autovectorization, stronger opts) and covers more targets, likely at the cost of
  slower compile/JIT. This is **not** a proven claim — SQL kernels dominated by extern
  calls, storage-pointer ABI slots, and weak alias info may give LLVM little to work
  with. Benchmark before treating it as the rationale. Cranelift stays the fast-compile
  default.

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
| JIT | `JITModule` → `get_finalized_function` → `*const u8` | `ExecutionEngine::new(module, opt, libs, …)` → **`lookup(name)` → native fn ptr** (not `invoke_packed`; §7) |
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

**But "the `Context` outlives the handles" is not the only validity invariant.** melior
explicitly warns that references can be invalidated when operations move ownership, are
erased, or when **passes rewrite the module**. A raw `MlirValue` captured during
emission may dangle after `mem2reg`/lowering runs. So the lifecycle must be
**structural**, not just lifetime-scoped:

```
MlirEmitter (owns the ValueId/VarHandle/BlockHandle arenas)
  → emit all functions        (handles valid here, and ONLY here)
  → finish()                  → drops every arena; no handle survives
  → verify module
  → run passes (mem2reg, lowering, reconcile-unrealized-casts)
  → verify module
  → build ExecutionEngine
```

**No `ValueId`/handle may outlive `finish()`.** Everything the runtime needs afterward
(the entry point, extern registrations) is keyed by *name/`FuncHandle` metadata*, not by
IR-value handles. This makes the raw-handle window a closed emission phase rather than a
standing invariant — the safe way to keep the handles lifetime-free without pretending
they are eternally valid.

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

- **(a) `llvm.alloca` + load/store + explicit `mem2reg` (recommended).** Map the
  Backend variable API onto memory: `declare_var` → `llvm.alloca`; `def_var` →
  `llvm.store`; `use_var` → `llvm.load`; **`seal_block` → no-op**. `llvm.alloca`
  implements MLIR's `PromotableAllocationOpInterface`, and the `-mem2reg` pass promotes
  it to SSA, inserting block-argument phis where needed. This makes the imperative-
  variable and loop-phi story disappear with no hand-threaded phis and no sealing.
  Strict rules that make it actually work:
  - **`declare_var` inserts the `alloca` in the function *entry block*, always** —
    never at the current position. rust-lms declares variables lazily (on first
    assignment), which can be *inside a loop*; an alloca emitted there would allocate
    every iteration and typically block promotion. Entry-block placement is mandatory.
  - **Promote the allocas to SSA.** The Phase -1 spike found that **`mlir-sys 220`
    exposes no `mlirCreateTransformsMem2Reg`** and melior's `pass::transform` has no
    mem2reg constructor, so an explicit *MLIR-level* mem2reg pass is unavailable through
    these bindings today. In the spike, LLVM's own mem2reg (run by `ExecutionEngine` at
    opt level ≥ 2) promotes the allocas and the mutable loop runs correctly — so relying
    on the LLVM opt pipeline is a valid baseline. If MLIR-level promotion is later
    wanted (e.g. to optimize before translation), wrap a raw pass with
    `melior::pass::Pass::from_raw_fn` once a suitable symbol is available. Do **not**
    assume opt level `O0` promotes — the engine must run at ≥ 2, or add the pass.
  - **Keep ABI/output stack slots distinct from variable slots.** ABI slots hold the
    storage pointers that escape to calls and normally *cannot* be promoted; only the
    variable slots should be fed to `mem2reg`.
- **(b) Hand-built block-argument phis for *variables* — rejected.** Threading every
  mutated variable through every branch as `cf` block args re-implements Cranelift's SSA
  construction by hand; too large. Use (a) for variables.

**Keep `IfThenElse`'s explicit block-argument phi as-is — do not turn it into memory.**
MLIR `cf` blocks natively support block arguments (true phis), which is a direct
analogue of Cranelift's merge block param. The existing explicit phi is already optimal
IR; routing it through an alloca would only create work for `mem2reg` to undo. So:
variables → alloca+`mem2reg`; the `IfThenElse` merge → native block argument.

**Net:** the Backend trait keeps Cranelift's variable vocabulary
(`declare_var`/`def_var`/`use_var`/`seal_block`) as the common denominator. Cranelift
forwards to native calls; MLIR implements them as entry-block alloca / store / load with
a no-op seal and an explicit `mem2reg` pass. Control-flow blocks + branches map to `cf`
directly, and `IfThenElse` uses real `cf` block arguments.

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

    // ---- pointers (semantic; NOT raw iadd — see §8b) ----
    fn ptr_offset_bytes(&mut self, ptr: ValueId, offset: ValueId) -> ValueId; // gep / iadd
    fn ptr_offset_const(&mut self, ptr: ValueId, bytes: i64) -> ValueId;
    fn ptr_to_addr(&mut self, ptr: ValueId) -> ValueId;   // ptrtoint / no-op
    fn addr_to_ptr(&mut self, addr: ValueId) -> ValueId;  // inttoptr / no-op

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

// Module lifecycle, kept SEPARATE from per-function emission because Cranelift's
// FunctionBuilder and MLIR's block/region builder have different ownership rules.
// Written to be object-safe: `body` is a boxed FnOnce (not `impl FnOnce`, which would
// make the method generic), and `finalize` takes `self: Box<Self>` (not `self` by
// value, which cannot be called through `dyn Module`).
pub trait Module {
    fn declare_function(&mut self, name: &str, sig: &SigSpec, linkage: Linkage) -> FuncHandle;
    fn define_function(&mut self, f: FuncHandle, body: Box<dyn FnOnce(&mut dyn Backend)>);
    fn register_extern(&mut self, name: &str, ptr: *const u8);
    fn finalize(self: Box<Self>) -> Result<Box<dyn Executable>, CompileError>;
}
// `Executable` owns the JIT resources (Cranelift JITModule, or MLIR
// ExecutionEngine+Context+Module) and frees them on Drop. NOTE: the MLIR
// ExecutionEngine is `!Send + !Sync` (§9), so the LLVM `Executable` is thread-affine.
```

`CompilationContext` keeps its backend-neutral maps (`var_map: HashMap<usize,
VarHandle>`, `loop_exit_stack: Vec<BlockHandle>`, `slice_vars`, the slice helpers) and
holds a `&mut dyn Backend`. The ~hundreds of `Staged::codegen` bodies change only
mechanically (`ctx.builder.ins().iadd(a,b)` → `ctx.backend.iadd(a, b)`); the `Num`/
`StagedType` trait methods absorb the signed/unsigned/float selection they already do.

## 7. ABI & JIT — use `lookup()`, not `invoke_packed`

rust-lms's Phase-3 ABI is one uniform convention: **N storage pointers + one output
pointer, void return**, and `CompiledFn`/`as_fn_unchecked` already `transmute` the JIT
entry to `unsafe extern "C" fn(*const u8, …, *mut u8)` and call it directly
(`func.rs:1584`).

**Correction to an earlier draft:** MLIR's `invoke_packed(name, &mut [*mut ()])` is
**not** this convention. `invoke_packed` calls a generated `void(void**)` trampoline
that `llvm.emit_c_interface` emits, whose slots are *pointers to* the argument storage.
Since our arguments are *already* pointers, that adds a **second** level of indirection
and requires the `emit_c_interface` attribute — a different (and slower) calling path.

The right move preserves the Phase-3 ABI and all of `Compiled`/`CompiledFn` unchanged:

1. Emit each JIT function with the existing storage-pointer signature (`llvm.func`
   taking `llvm.ptr` params + an output `llvm.ptr`, `void` return).
2. Verify the module, lower to the LLVM dialect and translate (§ pass pipeline below).
3. Build `ExecutionEngine::new(&module, opt_level, &[], false, false)`, register each
   host extern with `engine.register_symbol(name, ptr)` (the analogue of
   `JITBuilder::symbol`).
4. Get the **native function pointer** via `ExecutionEngine::lookup(name)` and
   `transmute` it to the same `extern "C" fn(*const u8, …, *mut u8)` that
   `as_fn_unchecked` already uses. **No packed wrapper, no extra indirection, ABI
   identical to Cranelift.** Reserve `invoke_packed` only for tests or an explicit
   adapter.
5. The MLIR `Compiled` owns the `ExecutionEngine` (+ `Context`/`Module`) and drops them
   together — the same "owns executable memory, frees on `Drop`" contract as the
   Cranelift `JITModule`. **Caveat (§9): melior's `ExecutionEngine` is `!Send + !Sync`**,
   so the LLVM `Compiled`/`CompiledFn` are thread-affine — an auto-trait change from the
   Cranelift path that the ownership design must account for.

Because the entry point is a plain function pointer with the by-pointer signature,
`Compiled`/`CompiledFn` glue stays backend-neutral; only module finalize +
`lookup` differ.

**Pass pipeline (corrected — pick one source level).** The earlier draft mixed
"emit `llvm.func`" *and* `convert-func-to-llvm`, which is contradictory. Choose:
either (a) emit `func.func` + `arith` + `cf` + LLVM memory ops, then convert *all*
remaining dialects; **or (b, recommended) emit `llvm.func` directly and only convert
nested `arith`/`cf`.** We use `llvm.alloca`, not `memref`, so
`finalize-memref-to-llvm` is unnecessary. Run **`mem2reg` explicitly** (§5) rather
than trusting the engine's opt level, and add **`reconcile-unrealized-casts`** after
progressive conversion (per the LLVM lowering docs). Then
`register_all_llvm_translations` before building the `ExecutionEngine`.

## 8. Types, pointers, and booleans

### 8a. A backend-neutral type is needed (coupling is broader than `codegen`)

Cranelift types leak through more than `Staged::codegen`. `StagedType::cranelift_type`
(`types.rs:27`), `ConstantType::codegen_constant`, and `TypeInfo::value_type`
(`func_impl.rs:20`) all name Cranelift types, and there is a **downstream `Staged` impl
in `arrow-lms/src/array.rs:264`** returning a Cranelift value. All of these must move to
a backend-neutral representation. Introduce:

```rust
pub enum ScalarType { Bool, I8, I16, I32, I64, F32, F64, Ptr }
```

`TypeInfo` should carry **representation + layout (size/align)**, not a backend
`TypeId`; each backend maps `ScalarType` → its own type at emit time.

| `ScalarType` | Cranelift | MLIR |
|---|---|---|
| `I8/I16/I32/I64` | `types::I8/…/I64` | `IntegerType::new(ctx, 8/…/64)` |
| `F32/F64` | `types::F32/F64` | `Float32Type`/`Float64Type` |
| `Bool` | `types::I8` | **`i1`** internally, `i8` at storage (§8c) |
| `Ptr` (all pointers/slices/struct handles) | `types::I64` | `llvm.ptr` |

### 8b. Pointers need *semantic* ops — this is **not** local to the AST

The earlier draft claimed the `i64`-vs-`llvm.ptr` choice was invisible to the AST. It
is not: pointer arithmetic today lowers to integer `iadd` (e.g. `refer.rs:454`), and an
`llvm.ptr` **cannot** be an operand to `llvm.add`. So the `Backend` trait must expose
*semantic* pointer operations, and the AST must call them instead of `iadd`:

```rust
fn ptr_offset_bytes(&mut self, ptr: ValueId, offset: ValueId) -> ValueId;
fn ptr_offset_const(&mut self, ptr: ValueId, bytes: i64) -> ValueId;
fn ptr_to_addr(&mut self, ptr: ValueId) -> ValueId;   // ptrtoint
fn addr_to_ptr(&mut self, addr: ValueId) -> ValueId;  // inttoptr
```

Cranelift implements these with integer arithmetic (as today); MLIR uses
`llvm.getelementptr` + explicit `ptrtoint`/`inttoptr` casts. This means `refer.rs`'s
`ptr_offset`/`array_index` codegen must be rewritten from raw `iadd` to these semantic
methods — a real (if small) change on the Cranelift side too. Note also that `llvm.ptr`
**does not by itself** give better alias information; that needs `noalias`, alias
scopes, and alignment metadata/attributes, which are out of scope for a first cut.

### 8c. Boolean representation needs an explicit policy

rust-lms stores `bool` as `I8`, but MLIR comparisons (`arith.cmpi`) produce `i1` and
conditional branches/`select` consume `i1`. Policy: **`Bool` is logical internally**,
lowered to Cranelift `I8` and MLIR `i1`, with **explicit `i1`↔`i8` conversion at
storage boundaries** — loads, stores, calls, block arguments, and any place a bool is
materialized into memory or an ABI slot. The `Backend` trait hides this: `icmp` returns
a `Bool` value, `brif`/`select` take one, and load/store of a `Bool` field insert the
`zext`/`trunc` on the MLIR side (no-op on Cranelift).

## 9. Implications (the honest list)

- **Build & CI (biggest).** melior 0.27 / mlir-sys 220 require **LLVM/MLIR 22 installed
  on the machine** and discoverable via `llvm-config` or `MLIR_SYS_220_PREFIX`. That is
  a multi-hundred-MB system dependency, a new CI provisioning step on all six target
  runners, and a real onboarding cost. **Mitigation:** gate the entire MLIR backend
  behind `--features llvm` (off by default). The pure-Rust Cranelift build stays the
  default; `cargo build`/`test` are unaffected unless you opt in.
- **Compile/JIT time (hypothesis).** LLVM is expected to JIT slower than Cranelift and
  to optimize harder, positioning LLVM as the "optimize hard, run many times" mode and
  Cranelift as the "compile fast" default — but the magnitude ("seconds vs
  milliseconds") is a guess until measured on real kernels. Expose the choice
  per-`Compiler`, not globally, and benchmark both compile latency and steady-state
  execution before publishing any performance rationale.
- **`ExecutionEngine` is `!Send + !Sync`.** melior documents this. An LLVM `Compiled`/
  `CompiledFn` that owns the engine therefore loses the `Send`/`Sync` the Cranelift
  path may have. Decide deliberately: either LLVM-compiled functions are thread-affine
  (document it), or the engine is isolated behind a different ownership design. This is
  an auto-trait change on a public type and must be a conscious decision, not a
  surprise.
- **Backend edge-case semantics are not automatically shared.** Division by zero,
  oversized/negative shifts, signed-division overflow, and float NaN handling differ:
  LLVM may produce **poison** where Cranelift **traps** or defines a result, and shift
  semantics differ past the bit width. rust-lms already treats slice/pointer ops as
  unchecked (author's contract), but the arithmetic edge cases need a **defined,
  backend-independent policy** and **differential tests that include them** — not only
  successful SQL workloads.
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

**Phase -1 — MLIR spike — DONE ✅** (`mlir-spike/`, excluded from the workspace). A
standalone melior binary that proves the load-bearing MLIR mechanics work end to end,
so we don't commit to the `dyn Backend` + `ValueId` refactor on faith. **All five
checks pass** against Homebrew `llvm@22` (LLVM/MLIR 22.1.7), with `melior 0.27.4` /
`mlir-sys 220.0.2` and `MLIR_SYS_220_PREFIX`:

1. ✅ JIT a function and call it via **`ExecutionEngine::lookup`** — the native function
   pointer, cast to `extern "C" fn(...)` and called directly (the Phase-3 ABI, **not**
   `invoke_packed`). Confirms §7.
2. ✅ A mutable loop from **entry-block `llvm.alloca` + load/store + `cf` blocks** JITs
   and runs correctly (`sum_to(5) == 10`). Promotion is done by **LLVM's own mem2reg at
   `ExecutionEngine` opt level 2** — see the finding below.
3. ✅ **Load through a real `llvm.ptr`** (`deref(&42) == 42`).
4. ✅ Call a **registered Rust `extern "C"`** through a pointer arg (`register_symbol` +
   `call_read(&41) == 41`).
5. ✅ **Module verified before and after lowering** (`create_to_llvm`).

**Findings that update this plan:**
- The known-good pipeline is `Module::parse` → verify → `PassManager` with
  `pass::conversion::create_to_llvm()` → verify → `ExecutionEngine::new(&m, 2, &[],
  false, false)` → `lookup(name)`. `register_all_llvm_translations(&context)` is
  required at context setup.
- **`mlir-sys 220` exposes no `mlirCreateTransformsMem2Reg`** (and melior's
  `pass::transform` has no mem2reg constructor). So §5's "explicit `mem2reg`" is not
  available through these bindings today: either rely on LLVM's opt-level mem2reg (which
  works — check 2 proves it), or wrap a raw pass via `melior::pass::Pass::from_raw_fn`
  once/if a symbol exists. Adjust §5's "run mem2reg explicitly" accordingly.
- The spike builds IR from **textual MLIR** (`Module::parse`) on purpose — it de-risks
  the JIT/ABI/lowering pipeline (the real risk), not melior's alpha op-builder helpers
  (Phase 1's job).

Verdict: the `dyn Backend` + `ValueId` direction is sound, and the two architectural
risks (pointer representation §8b, and the corrected native-vs-packed ABI §7) are
cleared. Proceed to Phase 0.

**Phase 0 — introduce the seam, Cranelift-only (pure refactor, tests stay green).**
The bulk of the whole effort, and **backend-count-independent — worth doing even if
LLVM never ships** (it gives the codegen a documented, unit-testable boundary with no
Cranelift types leaking into the AST). The art is keeping the suite green at *every*
step. Two linchpins make that possible:

1. **Transparent `ValueId` alias until the very end.** `pub type ValueId =
   cranelift_codegen::ir::Value;` (likewise `VarHandle = Variable`, `BlockHandle =
   Block`). While it's an alias every rewrite is behavior-identical and compiles. Only
   the last sub-phase flips it to an opaque `struct ValueId(u32)` arena index — and
   since the AST only *names* `ValueId`, that flip touches only the backend internals,
   not the ~142 call sites.
2. **Inherent op-methods on `CompilationContext` first; extract the trait later.** Don't
   move `builder` out on day one (that breaks all 142 sites at once). First add thin
   `ctx.iadd(a,b)` wrappers over `self.builder.ins()`, migrate the sites onto them, and
   only then extract the wrappers into the `Backend` trait.

Sub-phases (each a green, committable unit; `cargo test -p rust-lms` after each file in
0b/0c, full `--workspace --all-targets` at every boundary, clippy at 0f):

- **0a — Backend-neutral types. DONE (scaffolding) ✅.** Added `ScalarType { Bool, I8,
  I16, I32, I64, F32, F64, Ptr }` with `to_cranelift()`/`from_cranelift()`
  (`types.rs`), and a *provided* `StagedType::scalar_type()` defaulting to
  `from_cranelift(cranelift_type())` (backward-compatible — downstream impls unchanged),
  with a precise `bool → Bool` override. Exported from the prelude. Full workspace green
  (343/0/2). **Deferred to 0b (deliberately, so each is consumer-driven and testable):**
  the precise per-impl overrides (`Ptr` for the pointer/handle markers in
  `refer.rs`/`option.rs`/`slice.rs`/`ffi.rs`/`opaque.rs` + the derive), and folding
  `TypeInfo` (`func_impl.rs`) and `ConstantType::codegen_constant` onto `ScalarType` —
  these all touch builder/ABI code the 0b op-sweep already rewrites, so they land there
  where a real backend method reads the value.
- **0b — Op-method sweep. Value-ops DONE ✅ (except `func.rs`).** Added the inherent
  `CompilationContext` funnel (`iconst`/`f64const`/`f32const`, all int/float arithmetic,
  bitwise/shift, `icmp`/`icmp_imm`/`fcmp`/`select`, casts, `load`/`store`/`stack_addr`),
  `ValueId`/`ScalarType` in the signatures, `MemFlags` hidden. Migrated every value-op
  site through it in `num/traits.rs` (threaded `&mut CompilationContext` through the
  `Num`/`IntNum` methods), `num/ops.rs`, `types.rs` (`codegen_constant` too),
  `struct.rs`, `refer.rs`, `slice.rs`, `option.rs`, `ffi.rs`, `iter/zip.rs`, and
  `staged.rs` (`Const`/slice-helpers). Full workspace green (343/0/2); clippy unchanged
  at the 13-lint baseline; unused Cranelift imports cleaned. The staged.rs funnel
  wrapper *bodies* still call `self.builder.ins()` by design (extracted in 0d).
  **Deferred:** `func.rs`'s 7 remaining value-op sites (`stack_addr`/`load`/`iconst`) are
  interleaved with control-flow, the `item_cty`/`declare_var` var machinery, and the
  `compile()`/ABI code that uses a bare `builder` — so `func.rs` is migrated
  *holistically* in 0c/0d rather than piecemeal in the crate's most delicate file.
  Control-flow (`jump`/`brif`/`return_`), calls (`call`/`func_addr`), and vars
  (`declare_var`/`def_var`/`use_var`) across all files remain for 0c/0d.
- **0c — Control-flow, vars, calls, pointer ops. DONE ✅.** Added the rest of the
  funnel: blocks (`create_block`/`append_block_param`/`block_param`/`switch_to_block`/
  `seal_block`), terminators (`jump`/`brif` — taking `&[ValueId]`, wrapping `BlockArg`
  internally), variables (`declare_var(ScalarType)`/`def_var`/`use_var`), calls
  (`call`/`call_indirect`/`func_addr`, returning `Option<ValueId>`), and the semantic
  pointer ops (`ptr_offset_bytes`/`ptr_offset_const`/`addr_to_ptr`, §8b — Cranelift
  no-ops today, `getelementptr`/`inttoptr` under MLIR). Migrated `control.rs`
  (`if_then_else` block-param phi, `while`, `if_then`, `not`), `func.rs` (the delicate
  opaque-iterator SSA loops + `let_var`/`assign` — renaming only, so the `seal_block`
  ordering is preserved), and the block/var/call sites in `option.rs`/`slice.rs`/
  `refer.rs`/`ffi.rs`/`func_impl.rs`/`staged.rs`. Pointer arithmetic (`refer.rs`
  `element_addr`, `struct` field offsets, `COption` payload, fat-slice writes) now goes
  through the semantic ptr ops, not raw `iadd` on a pointer value. Full workspace green
  (343/0/2); clippy unchanged at the 13-lint baseline; unused imports cleaned.
  **Everything AST-facing now routes through the funnel.** The only remaining
  `ctx.builder`/`builder` access is the funnel's own wrapper bodies (extracted in 0d)
  and the `compile()`/ABI/`Module` machinery: `create_sized_stack_slot` (×13),
  `import_signature`/`declare_func_in_func`, `append_block_params_for_function_params`,
  the entry-block param loading, `return_`, `finalize`, `symbol`, `emit_small_memory_copy`
  — all 0d.
- **0d — Extract `Backend` trait + `CraneliftBackend` (riskiest; ownership reshuffle).**
  Promote the funnel methods into an object-safe `Backend` trait; `CraneliftBackend`
  owns `&mut FunctionBuilder` + `&mut JITModule`; `CompilationContext` holds `&mut dyn
  Backend` and delegates. Sites don't move; only builder ownership changes.

  **Concrete design (mapped from the field/access audit — ready to execute):**
  - **Moves into the backend:** only `builder` + `module`. Everything else in
    `CompilationContext` is neutral our-id→handle bookkeeping and **stays**: `var_map`
    (`usize→VarHandle`), `slice_vars`, `loop_exit_stack` (`Vec<BlockHandle>`),
    `unit_value` (`Option<ValueId>`), `func_map`, `extern_func_ids/refs`. (These are
    accessed *outside* `staged.rs` — `func.rs` break-loop + var binding, `option.rs` var
    binding — so they must remain on `CompilationContext`.)
  - **`Backend` trait surface:** the ~30 funnel ops already defined (value/ptr/control/
    var/call/memory/`stack_addr`), plus the builder/module-coupled ones that are still
    direct today — `create_stack_slot(size, align)` (wrapping `StackSlotData`),
    `import_signature`, `declare_func_in_func(FuncId)->FuncRef`, `memcpy`
    (`emit_small_memory_copy`), and `target_frontend_config()`/`default_call_conv()` for
    the two external `ctx.module.isa()` uses (`func.rs:403`, `iter/zip.rs:224`).
  - **`CompilationContext` inherent methods stay** (`get_extern_func_ref`,
    `get_unit_value`, `slice_data_ptr/len/parts`) but compose `self.backend.<op>` + the
    kept maps.
  - **Lifetimes:** `CompilationContext<'a,'b>` gets `backend: &'b mut (dyn Backend +
    'a)`; `CraneliftBackend<'a,'b>` coerces in. This is the one fiddly bit.
  - **`compile()` + `Module`/`Executable`:** the entry-block setup,
    `append_block_params_for_function_params`, param loading (`declare_var`/`def_var`/
    `block_params`), `return_`, ABI stack slots, `finalize`, `symbol`, and the
    `TypeInfo.value_type`→`ScalarType` fold move behind object-safe `Module`/`Executable`
    traits (`Box<dyn FnOnce>` body, `finalize(self: Box<Self>)`). This is the second half
    of 0d and where `func.rs`'s deferred bare-`builder` code lands.

  Rename `Staged::codegen -> ValueId`. **Do this as a fresh, focused effort** — it is a
  lifetime-sensitive change that should not be rushed; verify with the full suite at each
  bounded move (backend struct → delegation → module.isa methods → compile()/Module).

  **Ownership reshuffle DONE ✅.** `Backend` trait (the ~50 primitive ops incl.
  `create_stack_slot`/`import_signature`/`declare_func_in_func`/`copy_nonoverlapping`/
  `default_call_conv`) + `CraneliftBackend { builder, module }` implementing it;
  `CompilationContext` collapsed to a single lifetime `<'c>`, holds `backend: &'c mut
  dyn Backend` + the neutral bookkeeping maps, and **`Deref`/`DerefMut` to `dyn
  Backend`** so `ctx.<op>()` routes to the backend with zero per-op delegators. All
  coupled call sites (`create_sized_stack_slot`→`create_stack_slot`, `import_signature`,
  `declare_func_in_func`, `module.isa().default_call_conv()`) migrated; `zip`'s manual
  memcpy now `ctx.copy_nonoverlapping`; `func_impl`'s ABI return load bridges
  `TypeInfo.value_type` via `ScalarType::from_cranelift`. `compile()` builds a scoped
  `CraneliftBackend` per function. `Backend` is `pub #[doc(hidden)]` (only because the
  public `CompilationContext` derefs to it). Full workspace green (343/0/2); clippy
  unchanged at the 13-lint baseline; no `ctx.builder`/`ctx.module` leaks remain.
  **0d part 2 — DONE ✅ (the AST-facing items):** `TypeInfo.value_type` (a Cranelift
  `Type`) → `TypeInfo.repr: ScalarType` (compile()'s Cranelift driver calls
  `.to_cranelift()` at its two bare-builder param-load sites); and `Staged::codegen`
  renamed to return **`ValueId`** — every `Staged` impl across the crate now returns the
  neutral handle, and the AST modules import `crate::staged::ValueId` instead of
  `cranelift ... Value`. Full workspace green (343/0/2); clippy at the 13-lint baseline.
  **Deferred to the LLVM-backend phase (not Phase 0):** the `Module`/`Executable`
  lifecycle traits abstracting `compile()`'s driver (entry block,
  `append_block_params_for_function_params`, param loading, `return_`, `finalize`,
  `symbol`). Rationale — that abstraction can only be designed correctly against a
  *second* lifecycle (MLIR's `ExecutionEngine::{new,lookup}`, which the Phase -1 spike
  showed is nothing like Cranelift's declare/define/finalize); building it now with only
  Cranelift in hand would be a speculative guess. `compile()` is the backend driver and
  is legitimately Cranelift-specific for a Cranelift-only refactor.
- **0e — Flip `ValueId` to opaque** — **DONE.** `ValueId`/`BlockHandle`/`VarHandle` are
  now opaque `struct _(u32)` newtypes; the AST never names a Cranelift value type. The
  `u32` is the Cranelift entity's own `as_u32()` index, so `CraneliftBackend` needs **no
  arena** — conversion is stateless (`from_cranelift`/`cranelift` via `as_u32`/`from_u32`);
  MLIR later reuses the same `u32` as an index into its own value `Vec`. Downstream `Staged`
  impls (arrow-lms `ValidityIsValid`, test_slices) and the prelude export were updated.
  *Green: full workspace, no new clippy lints.*
- **0f — Cleanup + boundary — DONE.** `cranelift_type()` is deleted; `scalar_type()` is
  the required source of truth on `StagedType` (each impl states its `ScalarType`
  directly — pointers/handles are `Ptr`, `size_of`'s default reads
  `ScalarType::size_bytes()`). Also neutralized the last two Cranelift *types* the AST
  named in op calls, both universal (not speculative second-backend design):
  - **comparison predicates** → `IntCmp`/`FloatCmp` enums with `to_cranelift()`
    (replacing `IntCC`/`FloatCC` at ~20 call sites in num/control/option/refer/slice/zip);
  - **stack slots** → opaque `StackSlotId` handle + neutral
    `Backend::alloc_stack_slot(size, align_shift)` (replacing `StackSlotData`/
    `StackSlotKind`/`StackSlot` at every AST site; `TypeInfo::slot_dims()` owns the dims).

  Result: **no `cranelift::*` type appears in any AST module.** The sole remaining
  Cranelift reference outside the backend/driver modules is `func_ref: FuncRef` on the
  `emit_extern_call` helper in `ffi.rs`. The `Backend` trait's only non-neutral surface
  is the **calls & signatures** cluster (`FuncRef`/`SigRef`/`Signature`/`FuncId`/
  `CallConv`) — the JIT function-reference/symbol machinery, deliberately left with
  `Module`/`Executable` because neutralizing it needs a second backend's call/symbol
  model (MLIR's `ExecutionEngine`) to design against. `compile_fail` doctests still guard
  `ctx.builder`; full workspace green (366 passed / 25 ignored), clippy unchanged.

**Phase 1 — MLIR backend skeleton behind `--features llvm` — IN PROGRESS.**
Context/Module setup, type mapping (§8), constants, arithmetic/bitwise/compare/select/casts,
memory (alloca/load/store), and the arena/`ValueId` plumbing. Prove a nullary `fun0`
returning a constant JITs and runs via `ExecutionEngine`.

- **Milestone 1 — DONE.** `rust-lms/src/llvm/mod.rs` (feature-gated), `melior 0.27`
  wired as an optional dep behind `llvm = ["dep:melior"]` (default build unaffected —
  366 tests / clippy baseline unchanged, no melior pulled). `make_context` (from the
  spike) + `jit_return_i64_const` build a nullary `() -> i64` `func.func`
  **programmatically** through melior's op builders (`arith.constant` + `func.return`) —
  the alpha-API risk the spike skipped — verify, lower via `create_to_llvm`, JIT, and run
  the native `ExecutionEngine::lookup` pointer. Feature-gated test passes against Homebrew
  `llvm@22`.
  - **Build env (gotcha):** the `llvm` feature needs `MLIR_SYS_220_PREFIX`,
    `LLVM_SYS_220_PREFIX`, `DYLD_LIBRARY_PATH=$P/lib`, **and `PATH="$P/bin:$PATH"`** — the
    last is required because melior-macro's `tblgen` build script shells out to
    `llvm-config`, which must be on `PATH` (not just discoverable via the `*_PREFIX` vars).
  - **Arena finding:** the `ValueId(u32)` → MLIR-value mapping is confirmed viable exactly
    as §9 predicted — melior's `Value<'c,'a>` is `#[repr(transparent)]` over a lifetime-free
    `mlir_sys::MlirValue`, reachable via the **public** `ValueLike::to_raw` and
    reconstructable via `Value::from_raw`, so a `Vec<MlirValue>` is the MLIR analogue of
    Cranelift's entity arena. It lands with the multi-value op layer next.
- **Milestone 2 — DONE.** The value-op layer: `scalar_to_mlir` (the `ScalarType`→MLIR
  type map, §8 — `Bool`→`i1`, `Ptr`→`llvm.ptr`), the `ValueId`→`Vec<MlirValue>` arena
  (§9 confirmed — `Value` is `#[repr(transparent)]` over a lifetime-free `MlirValue` via
  the public `ValueLike::to_raw`, so the arena is a plain safe `Vec`; `mlir-sys` added as
  an optional dep for the raw type), and `MlirBackend` with constants, integer/float
  arithmetic, bitwise, shifts, `IntCmp`/`FloatCmp` compares, branchless `select`,
  sign/zero-extend + truncate + int→float casts, and `alloca`/`load`/`store`. Driven by
  `jit_run_i64_unary`; 5 feature-gated tests pass (arithmetic, branchless abs via
  cmp+select, alloca/store/load roundtrip, narrow→widen cast), default build still 366 /
  clippy 28. `MlirBackend` stays an **inherent** API (not the shared `Backend` trait) —
  the trait's opaque handles line up, but its calls/signatures cluster is still
  Cranelift-typed, so unification waits on Phase 3/4.
**Phase 2 — variables & control flow (§5) — MLIR PRIMITIVES DONE (in `MlirBackend`).**
`declare_var`/`def_var`/`use_var` lower to entry-block `llvm.alloca` + `store`/`load`;
`seal_block` is a no-op; `create_block`/`append_block_param`/`switch_to_block`/`jump`/
`brif` build `cf` blocks with block-argument phis. Block model: a dedicated **entry block**
holds the params + *all* allocas and is terminated (branch to body block 0) at
`into_module`, so lazily-declared variables always precede the entry terminator — the
"entry-block alloca" rule §5 requires for `mem2reg`. Promotion is LLVM's own `mem2reg` at
`ExecutionEngine` opt level 2 (no MLIR-level pass; §5). 7 feature-gated tests pass, incl. a
**mutable `sum_to` loop** (two alloca vars mutated across a `cf` back-edge, promoted to SSA)
and an **if/else merged through a real `cf` block-argument phi**; default build still 366 /
clippy 28.
- **Remaining for Phase 2:** these are backend *primitives*; the AST constructs
  `if_then_else`/`while_loop`/`break_loop` route through them only once `MlirBackend`
  implements the shared `Backend` trait, and the differential test against Cranelift on
  `tests/{programs,p99,euler}.rs` likewise needs that wiring — both gated on the
  calls/signatures neutralization (Phase 3/4). Until then, correctness is checked against
  scalar oracles in the feature-gated tests.

**Phase 3 — calls, externs, ABI, JIT — MLIR PRIMITIVES DONE (in `MlirBackend`).**
`declare_extern` records an external function and emits it as a module-level `func.func
private` declaration at `into_module`; `call` emits a `func.call` (returning a value, or
void); the driver binds each extern to its host address via
`ExecutionEngine::register_symbol` before `lookup`. 9 feature-gated tests pass, incl.
**calling a registered Rust `extern "C"` returning i64 through a `*const i64`** and a
**void two-arg extern writing through a `*mut i64`** — the Phase-3 storage-pointer ABI
shape, now proven *programmatically* (the spike proved it with textual MLIR). Native
`lookup` path, no `invoke_packed`. Default build still 366 / clippy 28.
- **Remaining for Phase 3:** `call_indirect`/`func_addr` (function-pointer values) and the
  full storage-pointer signature ABI arrive with the shared-trait wiring; landing
  `test_extern_fn.rs`/`test_abi.rs` across both backends needs the AST routed through
  `MlirBackend`, which is the neutralization step below.

**Calls/signatures neutralization — DONE.** The `Backend` trait is now **fully
backend-neutral: no Cranelift type appears in any method signature.** The calls & signatures
cluster was replaced with opaque handles [`FuncRefId`]/[`SigRefId`] and the neutral `SigSpec`
(params + optional result); `call`/`call_indirect`/`func_addr`/`import_signature` take those,
and `declare_func`/`declare_extern_func` resolve *our* `usize` ids — the `usize → Cranelift
FuncId` maps and the per-function `FuncRef` caches moved out of `CompilationContext` into
`CraneliftBackend`, so the neutral context no longer names a Cranelift type either.
`default_call_conv` was dropped (the backend picks its own when lowering a `SigSpec`).
Net effect: **the AST is now 100% cranelift-free** (the last leak, `emit_extern_call`'s
`FuncRef` param in `ffi.rs`, is gone), and only the `compile()` driver + the not-yet-abstracted
`Module`/`Executable` lifecycle remain Cranelift-specific. Behavior-preserving: 366 tests
green, clippy 27 (one fewer — dropped a `format!`-in-`expect`), 9 LLVM tests still green.

**Module layout (post-neutralization).** With the trait neutral, the two backends are now
symmetric sibling modules: `cranelift/mod.rs` holds `CraneliftBackend` (the default `impl
Backend`) + the memcpy helper; `llvm/mod.rs` holds the MLIR context/JIT drivers with
`llvm/backend.rs` holding `MlirBackend`. `staged.rs` is the neutral core — handle types (with
their `pub(crate)` Cranelift/`llvm` conversions, kept there because they touch the handles'
private field), `SigSpec`, `CompilationContext`, the `Backend` trait, and the AST value types
— and no longer contains any backend `impl`. The `compile()` driver in `func.rs` remains the
Cranelift-specific module/JIT lifecycle (the `Module`/`Executable` boundary).

**`impl Backend for MlirBackend` — DONE.** `MlirBackend` now implements the *entire* neutral
`Backend` trait (the previously-inherent ops became the trait impl, and the missing ones were
filled): `f64const`/`f32const`, `icmp_imm`, `bitcast` (`arith.bitcast`), the `StackSlotId`
memory model (`alloc_stack_slot` → aligned entry-block `llvm.alloca` of `size × i8`;
`stack_addr` + offset-carrying `load`/`store` via byte-indexed `llvm.getelementptr`),
`copy_nonoverlapping` (unrolled byte copy), the pointer ops (`ptr_offset_bytes`/`_const` via
GEP, `addr_to_ptr` via a raw `llvm.inttoptr`), and the id-based calls
(`declare_func`/`declare_extern_func` resolve `usize` ids against inherent
`declare_internal_func`/`declare_extern` setup into a `FuncRefId` table; `call`/`func_addr`/
`call_indirect`/`import_signature` use it). 11 feature-gated tests pass, incl. new coverage for
f64 arithmetic + bitcast, a stack slot with a non-zero byte offset, and `copy_nonoverlapping`
between slots; default build still 366 / clippy 27. The internal-function-call path
(`declare_func`/`func_addr` + multi-function module assembly) is implemented but only exercised
once the driver below populates `internal_funcs` — no test drives JIT-to-JIT calls yet.

**First differential proof — DONE.** `jit_eval_nullary_i64` runs the **neutral AST**
`Staged::codegen` over a `CompilationContext` backed by `MlirBackend` (the identical code path
Cranelift takes, just a different `dyn Backend`). A differential test compiles the *same*
nullary `Staged` graph through Cranelift's public `compile().run()` **and** through MLIR and
asserts equal results. Coverage spans the operator/cast surface where backend semantics could
silently diverge (the §9 "edge-case semantics" risk): signed div/rem incl. negatives, bitwise,
`shl`/arithmetic-`shr`, branchless `min`/`max`, `eq`/signed-`gt`, a **signedness discriminator**
(`u64::MAX > 1` unsigned), truncate→sign-extend `int_cast`, and `if_then_else` (`cf` blocks +
block-argument phi *through the AST*). 12 feature-gated tests green; default build still 366 /
clippy 27. This is the project's thesis made executable: one neutral AST, two backends,
identical output.

**Differential proof extended to imperative bodies + loops — DONE.** `jit_eval_ctx_i64` runs
a `fun0`-style imperative [`Ctx`] body (via `Ctx::new`/`into_body`) through `MlirBackend`, and a
second differential test asserts Cranelift == MLIR for bodies using **mutable locals + `while`
loops** (`sum 0..10 == 45`, `6! == 720`). This exercises the §5 machinery *through the real AST*:
entry-block `alloca` variables, `cf` loop blocks, and mem2reg promotion — the part the spike
flagged as the hardest. 13 feature-gated tests green; default 366 / clippy 27.

**Differential proof extended to parameterized functions — DONE.** `jit_eval_ctx_unary_i64`
runs a `fun1`-style body through MLIR, mirroring `make_fun1`'s var-id assignment (param = id 0,
locals from id 1) so the body's `Var`s resolve identically to the Cranelift path; the parameter
arrives as a direct scalar block-arg stored into its variable slot (a direct-scalar ABI that
sidesteps the storage-pointer *parameter* ABI while still driving the parameterized AST end to
end). A third differential test asserts Cranelift (`compile(f).as_fn().call(x)`) == MLIR across
arguments for a pure expression, a polynomial, and a `while` loop whose **trip count is the
runtime argument** (data-dependent control flow). 14 feature-gated tests green; default 366 /
clippy 27.

**Step 1 — `compile()`'s per-function assembly is now backend-neutral — DONE.** Extracted
`emit_function_body(ctx, params, param_infos, param_var_ids, body, return_info)` in `func.rs`:
it unpacks each storage-pointer argument into a variable (slices into their `slice_vars` pair),
runs the body, and writes the result back through the output pointer — entirely through neutral
`Backend` ops on `ctx` (zero `builder`/`.ins()`/cranelift references in its body). Both the
per-function loop *and* `__main__` (a zero-arg function under the same ABI) now route through it;
only the Cranelift-specific *shell* remains inline in `compile()` — module creation, per-function
`FunctionBuilder` entry setup, the terminating `return`, finalize, and extern registration. This
is a behavior-preserving refactor gated by the full 366-test suite (still green, clippy 27); it's
the seam the MLIR module lifecycle plugs into.

**Step 2 — multi-function MLIR modules — DONE.** `MlirBackend::into_function_op` finalizes a
backend into a single `func.func` op (plus the externs it references), and `assemble_module`
collects several ops + their extern declarations into one `Module`. A test builds `double(x)=2x`
and `quad(x)=double(double(x))` — the latter resolving its callee via `declare_func`/`call` — into
one module and runs `quad` through the JIT. This is the last missing *capability* for routing
`compile()` (whose output is helper functions alongside `__main__`, some calling others). 15
feature-gated tests green; default 366 / clippy 27.

**Next up — the MLIR assembler + `Compiler::with_backend`.** All the pieces now exist: the neutral
`emit_function_body` (per-function assembly, reused verbatim), `into_function_op`/`assemble_module`
(multi-function), `declare_func`/`call` (internal calls), and the storage-pointer ABI expressible
through the trait. The remaining glue is one function — `crate::llvm::assemble(functions, externs,
entry) -> Executable`: for each `FunDef` build an `MlirBackend` with `N+1` `llvm.ptr` params, run
`emit_function_body`, `into_function_op`; `assemble_module`; register extern symbols; lower; JIT;
resolve the entry by symbol (sidestepping the `func_addr`/`__main__` trampoline). Then make
`Compiled` backend-neutral (enum over `JITModule` | MLIR engine, the latter `!Send+!Sync`) and add
`Compiler::with_backend(Cranelift | Llvm)`. That collapses the bespoke `jit_eval_*` drivers into one
path and extends differential coverage to the full `tests/{programs,p99,euler,extern_fn,abi}.rs`
(Phase 4); slices/aggregates come after scalars.

**Phase 4 — parity & choice.** Differential-test the full suite on both backends; expose
`Compiler::with_backend(Backend::Cranelift | Backend::Llvm)`; decide the six-target CI
story (Cranelift on all six; LLVM where an MLIR 22 toolchain is provisioned).

**Rough effort:** Phase 0 is the large one (mechanical but wide, touches the whole
codegen surface + the SSA-var redesign of the trait). Phases 1–3 are a few focused
weeks each for someone comfortable with MLIR. Phase 0 delivers standalone value
(testable, documented backend boundary) even before any LLVM code exists.

## 11. Open questions / risks

Ranked — the two biggest are *not* the mutable-variable model (which has a practical
answer, §5) but pointer representation and getting the JIT ABI right:

- **(greatest) Pointer representation is a cross-cutting decision, not a local one**
  (§8b). It changes the `Backend` trait surface and forces rewriting `refer.rs`'s
  pointer arithmetic on *both* backends. Settle the semantic pointer ops in Phase 0 and
  validate `llvm.ptr` loads in the Phase -1 spike. Note `llvm.ptr` alone does not
  improve alias analysis.
- **(corrected) JIT ABI — use `lookup()`, not `invoke_packed`** (§7). The earlier
  "same convention" claim was wrong; `invoke_packed` adds a wrapper + indirection.
  Lock this down in the Phase -1 spike by calling the native pointer directly.
- **`mem2reg` reliability + placement.** The variable story bets on promotion. It holds
  for entry-block allocas over straight-line/structured loops (Clang relies on it), but
  **allocas must be entry-block** (§5) and `mem2reg` must be run **explicitly**
  (`mlirCreateTransformsMem2Reg` via `Pass::from_raw`). Verify promotion on the benches
  early in Phase 2 (check no residual stack traffic).
- **Handle invalidation** (§3). Raw `MlirValue` handles die when passes rewrite the
  module; enforce the structural lifecycle (no `ValueId` survives `finish()`).
- **melior is alpha & incomplete.** 0.27.4 still describes itself as API-unstable —
  **pin it exactly**, not with a semver range. `dialect::llvm` exposes only a subset of
  LLVM ops; some calls, indirect calls, address ops, and intrinsics will need
  `OperationBuilder` or the generated ODS APIs rather than a helper function.
- **Backend edge-case semantics** (§9) — define and differential-test div-by-zero,
  shift overflow, signed-division overflow, NaN, and pointer offsets; don't assume the
  backends agree.
- **`!Send`/`!Sync`** (§9) — decide the thread-affinity policy for the LLVM `Executable`.
- **Two-backend maintenance tax.** Every new staged op needs a `Backend` method and two
  impls; the differential suite keeps them honest.
- **Is it worth it?** If the goal is *faster* compilation or a pure-Rust build,
  Cranelift already wins and this is not worth it. If the goal is *maximum
  generated-code quality* or reusing the MLIR ecosystem (custom dialects,
  GPU/accelerator lowering), the LLVM backend is the way — and Phase 0 (+ the Phase -1
  spike) is a good investment regardless.

## Sources

- melior crate docs — <https://mlir-rs.github.io/melior/melior/> (`Value<'c,'a>`,
  dialects `arith/func/llvm/scf/cf/memref/index`, "no Cranelift-style variable
  abstraction", reference-invalidation safety notes, alpha/API-unstable).
- melior `ExecutionEngine` — <https://mlir-rs.github.io/melior/melior/struct.ExecutionEngine.html>
  (`new`, `register_symbol`, `lookup`, `invoke_packed`, `!Send`/`!Sync`).
- MLIR `ExecutionEngine` / packed invocation & `emit_c_interface` —
  <https://mlir.llvm.org/doxygen/classmlir_1_1ExecutionEngine.html>.
- LLVM dialect (`llvm.alloca` `PromotableAllocationOpInterface`, `llvm.ptr`, `getelementptr`)
  — <https://mlir.llvm.org/docs/Dialects/LLVM/>; `mem2reg` pass —
  <https://mlir.llvm.org/docs/Passes/#mem2reg>; LLVM lowering &
  `reconcile-unrealized-casts` — <https://mlir.llvm.org/docs/TargetLLVMIR/>.
- mlir-sys (raw C API incl. `mlirCreateTransformsMem2Reg`, lifetime-free `MlirValue`) /
  LLVM requirement — <https://mlir-rs.github.io/melior/mlir_sys/index.html>,
  <https://github.com/mlir-rs/mlir-sys>, <https://crates.io/crates/mlir-sys>
  (LLVM/MLIR 22, `llvm-config` / `MLIR_SYS_220_PREFIX`).
- Internal inventory of `rust-lms/src` (this study): 16 Cranelift files, 142 `.ins()`
  sites, 39 ops; `Staged::codegen -> cranelift ... Value`; `CompilationContext`
  fields; type coupling in `types.rs`/`func_impl.rs` + `arrow-lms/src/array.rs:264`;
  pointer arithmetic in `refer.rs`; Phase-3 storage-pointer ABI (`func.rs:1584`).
