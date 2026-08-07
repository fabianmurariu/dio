# CLAUDE.md

Guidance for Claude Code (claude.ai/code) when working in this repository.

## What this project is

This workspace is built around **`rust-lms`**, a type-safe **multi-stage
programming** (staging / LMS-style) library. You build a description of a
computation out of strongly-typed Rust values, and the library lowers it to
[Cranelift](https://cranelift.dev/) IR and JIT-compiles it to native machine code.

There are two phases, and keeping them straight is essential:

- **Stage 0 ("now"):** ordinary Rust that *builds* an expression graph of `Staged`
  values (`Var<i64>`, `Add<L, R>`, `SliceIter<…>`). No computation on data happens
  here.
- **Stage 1 ("later"):** `Compiler::compile` turns that graph into native code.

> **You are writing a code generator, not an evaluator.** Every construct's job is
> a single `Staged::codegen` method that emits Cranelift instructions. A feature is
> "correct" when the IR it emits is correct *and* its Rust trait bounds make
> invalid stage-1 code impossible to express.

The library's defining idea: **a value's Rust type encodes its staged type**, so
the Rust type checker is the staged type checker. Put real constraints on `Staged`
impls (`T: Num`, `S::Out: MutSliceType`, lifetime bounds) — that's the type system.

## Guiding principles

> **"Simplicity is the ultimate sophistication."**

- **Fewer, cleaner pieces.** Prefer the smallest set of well-typed constructs.
  *Remove before you add* — if a new feature reveals a duplication, collapse it
  first. A `Box` (or an `Arc`) to get a clean type is a good trade; a `u64` smuggled
  around to avoid one is not.
- **Type the pointers.** Never thread a bare address (`u64`, `usize`) as if it were a
  pointer. A pointer is `SPtr<T>` / `SMutPtr<T>` (`*const T` / `*mut T` at runtime),
  and an opaque host struct handed to an extern is `SRef` / `SRefMut<Opaque<T>>`
  (`&T` / `&mut T`). Host addresses you bake into a kernel are real Rust pointers
  (`*const T` / `*mut T`) at stage 0, not integers. If you find yourself writing
  `u64` for something that is a pointer, rust-lms has a type for it — use it. Anyone
  can throw integers around and pretend they're pointers; the whole point of this
  project is that the compiler you build keeps its types.
- **`sql-gen` is a worked example, not just an app.** It must be both a good SQL
  engine *and* a demonstration of how rust-lms lets a compiler author track types
  end-to-end. Held to the same bar as the library: no `u64`-as-pointer, real trait
  bounds, clean enums over tag-and-cast.

## Workspace layout

- `rust-lms/` — the library and Cranelift backend (the heart of the project).
- `rust-lms-derive/` — `#[derive(StagedType)]` and `#[extern_fn]` proc macros.
- `arrow-lms/` — staged Apache Arrow interop: lifetime-free `#[repr(C)]` column
  descriptors (`FfiArray`) extracted from a `RecordBatch`, plus the host-side output
  side (`PreparedOutput`, `StringViewBuilder` sinks). The data layer the SQL engine
  codegens against.
- `sql-gen/` — the SQL engine: datafusion parses SQL → we lower its `LogicalPlan`
  into a push-based `Operator` tree → `codegen.rs` emits one rust-lms kernel per
  query (`exec_jit`). Supports `Scan`/`Filter`/`Project`, scalar + `GROUP BY`
  aggregates (`count`/`sum`/`min`/`max`/`avg`, with nulls), and `Utf8View` strings.
  Also the flagship example of typed compiler construction on rust-lms (see
  principles above). Design docs: `docs/group_by.md`, `docs/codegen_issues.md`.

## Build & test

```bash
cargo build                                   # build the workspace
cargo check                                   # type-check only
cargo test                                    # all tests
cargo test -p rust-lms                        # just the library
cargo test -p rust-lms <name>                 # a single test
cargo fmt        cargo clippy                  # format / lint

# Print the generated Cranelift IR for a test (the first debugging tool to reach for):
RUST_LMS_DEBUG_IR=1 cargo test -p rust-lms <name> -- --nocapture
```

## Architecture (rust-lms)

- **`Staged` trait** (`src/staged.rs`) — the contract: `type Out: StagedType` plus
  `fn codegen(&self, ctx) -> Value`. Everything that becomes stage-1 code
  implements it.
- **Type system** (`src/types.rs`, `src/num/traits.rs`) — `StagedType` (with ABI
  methods), `ConstantType`, `CopyType`, and the capability traits `Num`/`IntNum`/
  `FloatNum` that carry instruction selection (signed vs unsigned vs float). Type
  markers are mostly the real primitives (`i64`, `u64=U64Type`, `f64=F64Type`,
  `bool=BoolType`, `()=UnitType`). **`bool` is not `Num`** — keep it on the
  control-flow/`select` path.
- **Values** — `Var<T>` (Copy handle to a Cranelift variable), `Const<T>`,
  `IntoStaged<T>` (lets APIs accept bare literals like `5i64`).
- **Operations as types** (`src/num/ops.rs`) — `Add<L,R>`, `Lt<L,R>`, … are generic
  structs, not enum variants; bounds enforce type safety; comparisons are
  heterogeneous (`Out = BoolType`). `select`/`min`/`max` are branchless.
- **Compiler & functions** (`src/func.rs`, `func_impl.rs`, `func_def.rs`) —
  `Compiler` owns definitions; `fun0..8`(`_rec`) define functions; `compile`
  performs ABI lowering and JIT; `Compiled::run()`/`as_fn()` execute.
- **Authoring styles** — imperative `Ctx`/`VarBuilder` (`ctx.var`/`store`/`if_then`/
  `while_loop`/`break_loop`, preferred) and expression-tree (tuples + `assign` +
  `while_loop`/`if_then_else`).
- **Control flow** (`src/control.rs`) — `IfThenElse` (merge block param = phi),
  `IfThen`, `While`, `Not`.
- **Memory** — `src/refer.rs` (`SRef`/`SRefMut`/`SPtr`/`SMutPtr`), `src/slice.rs`
  (`Slice<T>` fat pointers, `SliceType`, closed under sub-slicing), `src/struct.rs`
  (`#[derive(StagedType)]`, `Field`, four field-access traits, `Owned<T>`).
- **Iterators** (`src/iter/`) — push-based, fused: sources + combinators +
  terminals. Prefer branchless terminals (`count_if`/`sum_if`) in hot loops.
- **Optionals** — `StagedOpt` (`src/staged_opt.rs`, never materializes → control
  flow) vs `COption` (`src/option.rs`, FFI-safe, stored value).
- **FFI** (`src/ffi.rs`, `#[extern_fn]`) — call Rust `extern "C"` fns; `FatSlice`.

**For a full, current walkthrough see [`rust-lms/docs/deep_dive.md`](rust-lms/docs/deep_dive.md).**

## Invariants to respect

- Leave the Cranelift builder valid: terminate the current block correctly and
  `seal_block` only after all predecessors are emitted (mis-ordered sealing is the
  #1 cause of Cranelift panics).
- Slice ptr/len layout lives in exactly one place — `CompilationContext::
  slice_data_ptr`/`slice_len`. Don't re-derive it elsewhere.
- Slice/pointer ops are **unchecked** (no bounds checks); safety is the staged
  author's contract, surfaced through Rust types (mutability, lifetimes) wherever
  possible.
- Baked host pointers must stay typed. When a kernel holds a host buffer whose owner
  outlives the run (the string pool, a GROUP BY's state), bake it as a real
  `*const T` / `*mut T`, not a `u64`. rust-lms provides the reinterpret from a typed
  host pointer to a staged `SPtr`/`SMutPtr`/`SRef`/`SRefMut<Opaque<T>>` — the address
  is the only thing that reaches Cranelift, but the Rust type is checked at stage 0.
- `Compiled` owns the JIT module's executable memory — it must outlive any `as_fn`
  pointer.

## Testing

The integration tests under `rust-lms/tests/` (`programs.rs`, `p99.rs`,
`euler.rs`, `test_iter.rs`, `test_slices.rs`, `test_structs.rs`,
`test_extern_fn.rs`, `type_safety.rs`) are the spec and the best worked examples of
the current, idiomatic API. When adding a feature, add a test there in the same
style, and verify the emitted IR with `RUST_LMS_DEBUG_IR=1` when codegen is
involved.
