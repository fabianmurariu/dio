# mlir-spike — Phase -1 MLIR de-risking spike

Proves the load-bearing mechanics of a future LLVM/MLIR codegen backend
(see [`docs/llvm.md`](../docs/llvm.md)) work end to end via [melior](https://mlir-rs.github.io/melior/)
+ a system LLVM/MLIR 22, **before** committing to the wide `dyn Backend` refactor.

This crate is **excluded from the workspace** (`exclude = ["mlir-spike"]` in the root
`Cargo.toml`) because it needs a system LLVM/MLIR install. The normal
`cargo build`/`cargo test` at the repo root is completely unaffected and stays pure-Rust.

## Prerequisite

A system LLVM/MLIR **22** with MLIR libraries and the `mlir-c` C API headers. On macOS:

```sh
brew install llvm@22           # ships libMLIR, mlir-opt, and include/mlir-c
```

(Homebrew's `llvm@22` does include MLIR — verified: `libMLIR.dylib`, the static
`libMLIR*.a`, `bin/mlir-opt`, and `include/mlir-c/IR.h` are all present.)

## Build & run

melior/mlir-sys discover LLVM via `MLIR_SYS_220_PREFIX` (and `LLVM_SYS_220_PREFIX`).
At runtime the dynamic `libMLIR` must be findable. With Homebrew's keg-only `llvm@22`
at `/opt/homebrew/opt/llvm`:

```sh
P=/opt/homebrew/opt/llvm
MLIR_SYS_220_PREFIX=$P LLVM_SYS_220_PREFIX=$P PATH="$P/bin:$PATH" \
DYLD_LIBRARY_PATH="$P/lib" \
cargo run --manifest-path mlir-spike/Cargo.toml
```

Expected output: all five checks report `OK`.

## What the spike establishes (findings)

- **`melior 0.27.4` / `mlir-sys 220.0.2`** build and link against LLVM/MLIR 22.1.7; the
  correct env var is `MLIR_SYS_220_PREFIX`.
- **Native call path works:** `ExecutionEngine::lookup(name)` returns the native
  function pointer, callable directly as `extern "C" fn(...)` — the Phase-3 ABI,
  **not** the `invoke_packed` `void(void**)` wrapper. (Confirms `docs/llvm.md` §7.)
- **Mutable loops via entry-block `llvm.alloca` + `cf` blocks** JIT and run correctly;
  LLVM's own mem2reg (at `ExecutionEngine` opt level 2) promotes the allocas. **Finding:
  `mlir-sys 220` exposes no `mlirCreateTransformsMem2Reg`**, so an explicit MLIR-level
  mem2reg pass is not available through these bindings; rely on LLVM's opt pipeline, or
  wrap a raw pass via `melior::pass::Pass::from_raw_fn` if a symbol is added upstream.
- **`llvm.ptr` loads** and **registered Rust externs** (`register_symbol` + a pointer
  argument) both work — the storage-pointer ABI shape.
- **Module verification** passes before and after lowering (`create_to_llvm`).

## Note on IR construction

The spike writes IR as **textual MLIR** and `Module::parse`s it. That is deliberate: the
spike's job is to de-risk the JIT / ABI / lowering **pipeline**, not to exercise melior's
alpha op-builder helpers — building the IR programmatically through the `Backend` trait
is Phase 1's work.
