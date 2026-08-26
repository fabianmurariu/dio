# Post-LLVM backend review

Reviewed at commit `8515545` (`sql tests now run against llvm`). Scope: `docs/llvm.md`,
`rust-lms/src/llvm`, the shared contracts exercised by that backend, and
`rust-lms/benches/backends.rs`.

## Executive summary

The overall backend boundary is sound: the `ValueId` arena is consumed before MLIR passes run,
the `Context` outlives the `ExecutionEngine`, functions use one backend-neutral storage-pointer
ABI, and modules are verified on both sides of lowering. I did not find a dangling-MLIR-handle or
execution-engine ownership defect.

The backend is nevertheless not ready to be treated as fully safe/parity-complete. Two safe API
paths still lower to LLVM semantics that can become poison or invalid IR, the benchmark exposes an
unchecked kernel through a safe closure, and several unchecked narrowing conversions can turn a
large aggregate copy into a partial copy. The differential test claim in `docs/llvm.md` is also
broader than the actual harness, which is why the pointer-option defect was not caught.

Recommended order:

1. Define safe integer edge semantics and fix pointer-option lowering.
2. Close the LLVM test gaps around those semantics and pointer-shaped values.
3. Remove unchecked size/offset narrowing and replace byte-copy expansion with an LLVM memcpy.
4. Make LLVM compilation failures return `CompileError` and validate indirect-call signatures and
   module symbol names.
5. Correct the benchmark, then reassess its performance conclusions.
6. Collapse the prototype JIT paths and rewrite `docs/llvm.md` as current architecture rather than
   an implementation diary.

## P0: soundness and correctness

### 1. Safe integer operations can acquire LLVM poison/undefined semantics

The safe `div`, `rem`, `shl`, and `shr` constructors have no documented runtime preconditions, but
the LLVM backend lowers them directly to `arith.divsi`/`divui`, `remsi`/`remui`, and the `arith`
shift operations ([backend.rs:435](../rust-lms/src/llvm/backend.rs#L435),
[backend.rs:483](../rust-lms/src/llvm/backend.rs#L483)). Division by zero, signed `MIN / -1`, and a
shift count greater than or equal to the bit width are poison/undefined cases in this lowering.
Cranelift can trap or otherwise differ. The existing differential tests exercise only valid inputs.

This is more than a result mismatch: LLVM is allowed to optimize a function containing reachable
poison in ways that do not preserve local, predictable failure. A safe generated function should
not make callers responsible for an undocumented LLVM IR precondition.

Choose one backend-independent contract before adding more arithmetic:

- Rust-like checked behavior: add a neutral `trap` operation, guard zero divisors and signed
  overflow, and guard or normalize shift counts.
- Explicit wrapping/masking behavior: mask shift counts and define division failure separately.
- An unsafe split: keep checked safe operations and expose clearly named unchecked operations only
  where SQL planning has already proved the preconditions.

Add runtime-argument differential tests for zero divisors, signed overflow, and counts of
`width - 1`, `width`, and `width + 1`. Trap tests should run in subprocesses. The open-risk note in
[llvm.md:450](llvm.md#L450) correctly identified this issue, but the implementation and merge-ready
claims did not resolve it.

### 2. Niche-option references are not consistently represented as `Ptr`

`OptRefType` and `OptMutRefType` correctly declare `ScalarType::Ptr`, but their `None` constructors
still emit an `I64` zero ([option.rs:353](../rust-lms/src/option.rs#L353),
[option.rs:399](../rust-lms/src/option.rs#L399)). Their match operations then branch directly on a
pointer and bind the pointer into an `I64` variable ([option.rs:789](../rust-lms/src/option.rs#L789),
[option.rs:880](../rust-lms/src/option.rs#L880)).

This conflicts with the LLVM backend in three places:

- `coerce_to_bool` constructs an `IntegerAttribute` using the condition's type. That is invalid
  when the type is `!llvm.ptr` ([backend.rs:343](../rust-lms/src/llvm/backend.rs#L343)).
- A branch merge whose result is declared `Ptr` cannot receive the `I64` produced by `None`.
- Storing a pointer in an `I64` variable and loading it back as `I64` loses the staged pointer type;
  a later LLVM load/GEP expects `!llvm.ptr`.

Introduce a semantic `Backend::null_ptr()` operation. Lower it to `llvm.mlir.zero : !llvm.ptr` on
LLVM and an integer zero on Cranelift. Make pointer conditions compare through the existing
`ptr_to_int` path (or an `llvm.icmp` against a null pointer), and declare both bound variables as
`ScalarType::Ptr`. Also update the comments that still call these values "single i64".

The current LLVM integration suite does not exercise these unit tests; see finding 6.

### 3. `sum_above_median` hides `get_unchecked` preconditions behind a safe closure

The staged kernel uses `n = a.len()`, reads `a[n / 2]`, and reads `b[i]` through
`get_unchecked` ([backends.rs:96](../rust-lms/benches/backends.rs#L96)). The returned type is a safe
`Box<dyn Fn(&[i64], &[i64]) -> i64>`, but it neither rejects an empty `a` nor checks
`b.len() >= a.len()`. Calling it with either invalid shape can execute an out-of-bounds read. The
benchmark data happens to satisfy the comment, but the safe wrapper does not enforce it.

The native baseline is also not semantically equivalent: it uses `min(a.len(), b.len())` and
panics on an empty input, whereas the staged version uses only `a.len()` and can read out of bounds.
Either implement the same empty/min-length behavior in the staged graph or assert both
preconditions in the safe wrapper before entering JIT code. Keep the unchecked operations inside
the staged loop only after those conditions are established.

## P1: safety hardening and API correctness

### 4. Aggregate sizes and pointer offsets are silently narrowed

`ptr_offset_const` accepts the neutral trait's `i64` offset and silently casts it to `i32`
([backend.rs:658](../rust-lms/src/llvm/backend.rs#L658)). `copy_nonoverlapping` casts `usize` to
`i32` in the range bound ([backend.rs:612](../rust-lms/src/llvm/backend.rs#L612)). Above
`i32::MAX`, the copy can become empty or copy only a wrapped prefix. If that copy initializes a
return slot, `CompiledFn::call` can subsequently `assume_init` an incompletely written Rust value.

The shared ABI metadata has the same class of unchecked limit: `StagedType::size_of()` and
`align_of()` are cast to `u32` ([func_impl.rs:34](../rust-lms/src/func_impl.rs#L34)). Even if values
that large are not expected, silent wrap is the wrong failure mode at a trusted memory boundary.

Use a dynamic i64 GEP for constant offsets that do not fit MLIR's static i32 GEP encoding. Replace
the byte loop with `llvm.intr.memcpy` (or `llvm.call_intrinsic`) using a checked length and the known
alignment. Where Cranelift imposes a `u32` stack-slot limit, validate it once while building
`TypeInfo` and return a compilation error. Audit the remaining arena/operand-count `as u32`/`as i32`
casts at the same time.

### 5. `compile()` promises recoverable errors but the LLVM path panics

`Compiler::compile` returns `Result<_, CompileError>`, and `llvm::assemble` also returns `Result`,
but `jit_lookup` uses `assert!` for both verifications and lookup and `expect` for pass execution
([mod.rs:157](../rust-lms/src/llvm/mod.rs#L157)). In practice, `assemble` returns `Ok` or unwinds;
duplicate symbols, malformed backend IR, a missing extern, and lowering failures do not reach the
caller's error handling.

Split this into `lower_module(...) -> Result<Module, CompileError>` and
`create_engine_and_lookup(...) -> Result<_, CompileError>`. Include the function/pass stage in the
error and install a diagnostic handler so verification errors are retained rather than only printed
to stderr. Internal one-result builder assumptions can remain assertions, but module-, pass-, and
symbol-level failures are normal compiler errors.

### 6. "Whole suite on both backends" excludes backend-sensitive unit tests

The integration tests under `rust-lms/tests` use `for_each_backend`, but many tests inside
`rust-lms/src` instantiate only `Compiler::new()`. In particular, all the reference-option tests
at [option.rs:1242](../rust-lms/src/option.rs#L1242) run only Cranelift. Enabling the `llvm` Cargo
feature does not change `Compiler::new()`'s default. This contradicts the broader statements at
[llvm.md:860](llvm.md#L860) and in the CI workflow comments.

Move backend-sensitive unit tests to integration tests or provide a crate-internal equivalent of
`for_each_backend`. Add LLVM cases for:

- `OptRefType` and `OptMutRefType`, both `Some` and `None`, including mutation;
- pointer-valued `if_then_else` merges and pointer conditions;
- every scalar width, bool arguments/results, aggregates, and nested aggregates;
- the arithmetic edge policy from finding 1;
- deliberately invalid duplicate symbols and indirect-call signatures.

The existing differential coverage remains valuable; the claim and CI gate just need to match its
actual boundary.

### 7. Indirect calls discard the parameter half of `SigSpec`

`call_indirect` clones `self.sigs[sig]` and immediately discards the parameter types
([backend.rs:793](../rust-lms/src/llvm/backend.rs#L793)). Because the callee is an opaque pointer,
there is no referenced symbol declaration for MLIR verification to compare against. A wrong
argument count/type in the compiler can therefore become a wrong native calling convention rather
than a clean compile error.

At minimum, check argument count and each MLIR operand type against `SigSpec.params` before emitting
`llvm.call`. Prefer storing the expected type alongside each arena value so all backend operations
can diagnose neutral-IR type mistakes consistently. Also use checked conversion for
`operandSegmentSizes`.

### 8. User-visible names are used as linker identities without validation

Every `funN(name, ...)` name becomes its MLIR symbol, while the generated top level is always
`__main__` ([mod.rs:381](../rust-lms/src/llvm/mod.rs#L381)). Duplicate user names, a user function
named `__main__`, or an internal name colliding with an extern produces duplicate module symbols
and currently triggers the panic path from finding 5.

Use stable internal names derived from function IDs, such as `__rust_lms_fn_12`; retain the supplied
name as debug metadata. Reserve separate namespaces for the top-level trampoline and extern thunks.
This also allows helper definitions to be marked `private` instead of exporting every user name
from the JIT module.

### 9. The 64-bit/supported-target contract is implicit, and LLVM is not tested on Windows

`ScalarType::Ptr` is fixed at eight bytes and lowers to Cranelift `I64`
([types.rs:39](../rust-lms/src/types.rs#L39)); LLVM pointer/function addresses are likewise converted
through `i64` ([backend.rs:383](../rust-lms/src/llvm/backend.rs#L383)). That is consistent with the
intended x86_64/AArch64-only scope, but the crate does not enforce the scope.

Add a compile-time target guard for 64-bit `x86_64`/`aarch64` on Windows, Linux, and macOS, or derive
pointer width from a target layout. Since the stated project decision is to support only those six
targets, an explicit guard is simpler and prevents accidental 32-bit codegen.

CI currently exercises LLVM on the four Linux/macOS combinations and only Cranelift on the two
Windows combinations ([ci.yml:17](../.github/workflows/ci.yml#L17),
[ci.yml:78](../.github/workflows/ci.yml#L78)). Do not claim six-target LLVM support until at least
x86_64 Windows has a provisioned MLIR job; AArch64 Windows should remain separately qualified if a
native LLVM/MLIR toolchain is unavailable.

## P2: deduplication and simplification

### 10. Prototype drivers are public production API and duplicate the real pipeline

`jit_return_i64_const`, `jit_run_i64_unary`, `jit_eval_nullary_i64`, `jit_eval_ctx_i64`, and
`jit_eval_ctx_unary_i64` are public, although they are milestone/test drivers
([mod.rs:81](../rust-lms/src/llvm/mod.rs#L81), [mod.rs:209](../rust-lms/src/llvm/mod.rs#L209)). The
first also duplicates verification, pass setup, engine creation, and lookup instead of using
`jit_lookup`, and its pass list has already diverged by omitting `reconcile-unrealized-casts`.

Unless low-level MLIR construction is an intentional supported API, make `llvm` and `MlirBackend`
crate-private and move these drivers under `#[cfg(test)]`. Keep one production lowering/JIT path and
test it through `Compiler::with_backend(JitBackend::Llvm)`. If a low-level API is intentional, put
it in a clearly named experimental module and return errors instead of panicking.

### 11. Function/extern metadata is rebuilt and cloned for every function

`assemble` builds module metadata, then every `build_function` copies all internal and external
declarations into a fresh backend ([mod.rs:370](../rust-lms/src/llvm/mod.rs#L370),
[mod.rs:443](../rust-lms/src/llvm/mod.rs#L443)). `declare_func`/`declare_extern_func` then clone a
declaration again for each resolved call. Finally, only `__main__`'s copy of the extern list is used
to emit declarations. This is roughly quadratic in helper count and obscures the actual ownership
model.

Build one immutable module symbol table keyed by internal/extern ID. Give each per-function backend
a reference to it, let `FuncRefId` identify an entry without cloning, and emit extern declarations
once from the module plan. This removes `internal_funcs`, `externs`, and most `FuncDecl` cloning from
`MlirBackend`.

### 12. Keep explicit opcode mappings; deduplicate lifecycle and result plumbing instead

The repetitive one-line arithmetic methods are useful audit points between Cranelift, neutral
semantics, and MLIR. A macro that hides all mappings would save lines but make semantic review
harder. Better small deduplications are:

- one helper for appending and interning an optional call result (shared by direct/indirect calls);
- one checked arena-ID allocator instead of repeated `len() as u32`;
- one null-pointer operation instead of backend-leaking integer constants;
- one lowering pipeline, as described in finding 10;
- one memcpy operation instead of hundreds or thousands of generated byte operations.

### 13. Context and optimization policy are fixed rather than configured

Each compile registers and loads every available dialect ([mod.rs:69](../rust-lms/src/llvm/mod.rs#L69))
although emitted IR uses a small set, and `JIT_OPT_LEVEL` is hard-coded to 2. Registering only the
required dialects may reduce cold compile latency; measure before retaining the extra setup.

More importantly, the comment that optimization level 2 is required for correctness is wrong.
Loads/stores through entry-block allocas are semantically valid without promotion; mem2reg is an
optimization. If optimization selection becomes public, use a `CompilerOptions`/backend-options
value instead of another builder method per backend.

## `docs/llvm.md` accuracy

The document is now an append-only design log rather than current documentation. Specific factual
problems:

- The header still says "no code yet" and the opening inventory describes the pre-abstraction
  codebase ([llvm.md:4](llvm.md#L4)). Later sections simultaneously mark implementation complete.
- melior 0.27.4 does expose the `mlirCreateTransformsMem2Reg` binding in
  `pass::transform`; the claims at [llvm.md:198](llvm.md#L198), [llvm.md:504](llvm.md#L504), and
  [mod.rs:64](../rust-lms/src/llvm/mod.rs#L64) are false. An explicit MLIR mem2reg pass is available.
  Whether to use it is a performance/pipeline decision, not a correctness requirement.
- The pass recommendation says to emit `llvm.func` directly, while the implementation emits
  `func.func` and lowers it. Document the implemented pipeline, not the rejected alternative.
- It says `Compiled` owns the MLIR `Module`; `MlirExecutable` owns only the engine and context.
  That is valid because the MLIR C API explicitly permits destroying the module after engine
  creation, but the documentation should match the code.
- "Whole suite green on both backends" omits the crate unit-test gap from finding 6.
- The CI description at [llvm.md:911](llvm.md#L911) is stale in both directions: Linux LLVM jobs
  now exist, while Windows LLVM jobs do not.
- The risk list still describes already-completed pointer/ABI phases as open, and Phase 4 repeats
  work the preceding paragraphs call done.
- melior is declared as `version = "0.27"` despite the document recommending an exact
  `0.27.4` pin. Pin melior and the direct `mlir-sys` dependency together if reproducibility across
  this unstable binding surface is required.

Rewrite `docs/llvm.md` into four short sections: current architecture, build/toolchain setup,
lowering and ABI invariants, and supported targets/known semantic gaps. Move the phase history to a
separate archived implementation log or remove it. Comments in `rust-lms/src/llvm` should describe
current invariants rather than milestones and "future" work that has shipped.

## Benchmark review

The benchmark is useful as a warm, cache-resident kernel comparison, but its current results do not
support the broad architectural conclusions in `docs/llvm.md`.

1. JIT calls pay wrapper overhead that native does not. Both JITs go through `Box<dyn Fn>`, and the
   closure calls `compiled.as_fn()` on every iteration ([backends.rs:92](../rust-lms/benches/backends.rs#L92)).
   `as_fn()` invokes the generated `__main__` trampoline to fetch the helper pointer before calling
   the helper ([func.rs:1467](../rust-lms/src/func.rs#L1467),
   [func.rs:1560](../rust-lms/src/func.rs#L1560)). The native path is a direct call. Resolve the JIT
   entry once for the benchmark, or route native through equivalent dispatch.
2. Compilation is correctly outside the timed loop, but compile latency is not measured. Add a
   separate cold compile/lower/JIT group before describing the backend tradeoff.
3. Each iteration reuses the same sorted/random arrays. This measures warmed data and a very
   predictable branch distribution, not a general SQL scan. Keep it, but label it; add shuffled
   selectivity cases and data larger than the target's last-level cache.
4. Report CPU, OS, Rust version/profile, LLVM version, commit, command, and Criterion confidence
   interval with saved results. Exact Gelem/s values without that metadata should not live as
   durable architecture facts.
5. `Throughput::Elements(size)` means rows for both kernels, even though the second reads two
   columns. That is acceptable if labeled rows/s, but not directly comparable as memory bandwidth.
6. Two kernels showing native parity do not prove that stack-slot slice representation cannot be a
   performance lever. They show that LLVM optimized these two shapes on one system. Inspect the
   post-optimization LLVM IR/assembly and benchmark representative sql-gen plans before retaining
   the conclusion at [llvm.md:438](llvm.md#L438).

## Findings that appear sound

- The raw `MlirValue` arena does not survive pass execution: `MlirBackend` is consumed into
  operations before lowering, so rewritten operations cannot leave live `ValueId`s behind.
- Entry-block placement of variable allocas is correct. Promotion affects performance, not their
  load/store semantics.
- The native `ExecutionEngine::lookup` ABI is the right path; `invoke_packed` would add a different
  wrapper convention.
- Dropping the temporary MLIR module after `ExecutionEngine::new` is supported by the MLIR C API.
  Keeping the context in `MlirExecutable` and dropping the engine first is conservative and sound.
- External addresses enter through the unsafe `ExternFn` contract/generated thunk, and the engine
  is retained for the lifetime of every safe callable wrapper.
