# dio — staged JIT compilation in Rust

This workspace is built around **[`rust-lms`](rust-lms/)**, a type-safe
**multi-stage programming** library in the spirit of Scala
[LMS](https://scala-lms.github.io/) (Lightweight Modular Staging). You build a
description of a computation out of ordinary, strongly-typed Rust values; the
library lowers it to [Cranelift](https://cranelift.dev/) IR, JIT-compiles it to
native machine code, and hands you back a callable function pointer.

The key property is that **a value's Rust type encodes its staged type**, so the
Rust compiler *is* the staged type checker: invalid computations (adding an `i64`
to a `bool`, writing through a `&[T]`, returning a dangling field reference) don't
type-check at `cargo build` — there is no separate, runtime type system to get
wrong.

```rust
use rust_lms::prelude::*;

let mut compiler = Compiler::new();

// Define an iterative factorial as a staged function.
let factorial = compiler.fun1("factorial", |ctx, n: Var<i64>| {
    let i      = ctx.var(1i64);
    let result = ctx.var(1i64);
    ctx.while_loop(lt(i, n + 1i64), move |ctx| {
        ctx.store(result, result * i);
        ctx.store(i, i + 1i64);
    });
    result
});

// Compile to native code and get a typed function pointer.
let compiled = compiler.compile(factorial).expect("compile");
let factorial = compiled.as_fn();           // extern "C" fn(i64) -> i64

assert_eq!(factorial(5), 120);
assert_eq!(factorial(10), 3_628_800);
```

## Two phases

```text
   Rust source (stage 0)          Cranelift + JIT            native code (stage 1)
  ┌────────────────────┐  compile  ┌──────────────┐ finalize ┌──────────────────┐
  │ build a typed tree │ ────────► │  IR + ABI    │ ───────► │ fn(...) -> ...    │
  │  of Staged values  │           │  lowering    │          │ .as_fn()/.run()   │
  └────────────────────┘           └──────────────┘          └──────────────────┘
```

- **Stage 0 ("now"):** plain Rust that *builds* a computation — `Var<i64>`,
  `add(x, y)`, `slice.staged_iter().filter(..).sum(ctx)`. Nothing runs on data.
- **Stage 1 ("later"):** `Compiler::compile` JIT-compiles that description into a
  native function you call as many times as you like.

## What's in the box

- **Numbers & operators** — `i64`, `u64`, `i32`, `u32`, `f64`, `bool`, `()`, with
  `+ - * / %`, comparisons, and branchless `select`/`min`/`max`.
- **Functions** — `fun0`..`fun8` (and recursive `_rec` variants), first-class
  `FunRef` handles, direct calls.
- **Control flow** — value-producing `if_then_else` (real phi nodes), `if_then`,
  `while_loop`, `break_loop`, `not`.
- **Memory** — typed references and raw pointers (`SRef`/`SRefMut`/`SPtr`/
  `SMutPtr`), load/store/offset/index.
- **Slices** — `&[T]`/`&mut [T]` as 16-byte fat pointers, closed under
  sub-slicing, with register-resident `(ptr, len)` for hot loops.
- **Structs** — `#[derive(StagedType)]` on `#[repr(C)] Copy` structs, with
  lifetime-aware field access and correct by-value / by-reference ABI.
- **Staged iterators** — push-based, fully fused: `map`/`filter`/`filter_map`/
  `scan`/`take_while`/`skip_while`/`zip`, plus `sum`/`count`/`min`/`max`/`fold`,
  branchless `sum_if`/`count_if`, and short-circuiting `any`/`all`/`position`/
  `find_map`.
- **Optionals** — staging-time `StagedOpt` (never materializes; becomes control
  flow) and FFI-safe `COption<T>`.
- **FFI** — call ordinary Rust `extern "C"` functions from JIT code via
  `#[extern_fn]` and `FatSlice`.

## Workspace layout

| Crate | Role |
|-------|------|
| [`rust-lms`](rust-lms/) | the staged-computation library and Cranelift backend |
| [`rust-lms-derive`](rust-lms-derive/) | `#[derive(StagedType)]` and `#[extern_fn]` proc macros |
| [`sql-gen`](sql-gen/) | (early) SQL → staged-code generation over `datafusion-sql` + `arrow` |

## Build & test

```bash
cargo build                                   # build the workspace
cargo test                                    # run all tests
cargo test -p rust-lms                        # just the library
cargo test -p rust-lms <name>                 # a single test

# Inspect the generated Cranelift IR for any test:
RUST_LMS_DEBUG_IR=1 cargo test -p rust-lms test_while_loop_factorial -- --nocapture
```

## Documentation

- **[rust-lms/docs/deep_dive.md](rust-lms/docs/deep_dive.md)** — the architecture
  deep dive: the mental model, every subsystem, the ABI, and the invariants to
  respect. **Start here** if you're new (human or AI agent).
- **[rust-lms/README.md](rust-lms/README.md)** — library-focused overview.
- **[rust-lms/docs/staged_iterator_api.md](rust-lms/docs/staged_iterator_api.md)**
  and **[rust-lms/docs/loop_unrolling_design.md](rust-lms/docs/loop_unrolling_design.md)**
  — iterator design and the unrolling/SIMD roadmap.
- The integration tests under `rust-lms/tests/` (`programs.rs`, `p99.rs`,
  `euler.rs`, `test_iter.rs`, …) are the best worked examples of every feature.
