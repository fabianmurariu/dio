# dio — staged JIT compilation in Rust

This workspace is built around **[`rust-lms`](rust-lms/)**, a type-safe
**multi-stage programming** library in the spirit of Scala
[LMS](https://scala-lms.github.io/) (Lightweight Modular Staging). You build a
description of a computation out of ordinary, strongly-typed Rust values; the
library lowers it to [Cranelift](https://cranelift.dev/) IR, JIT-compiles it to
native machine code, and hands you back a callable function pointer.

The key property is that **a value's Rust type encodes its staged type**, so the
Rust compiler _is_ the staged type checker: invalid computations (adding an `i64`
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

// Compile to native code and get a typed, owner-checked callable.
let compiled = compiler.compile(factorial).expect("compile");
let factorial = compiled.as_fn();           // a CompiledFn: typed, owner-checked

assert_eq!(factorial.call(5), 120);
assert_eq!(factorial.call(10), 3_628_800);
```

## Two phases

```text
   Rust source (stage 0)          Cranelift + JIT            native code (stage 1)
  ┌────────────────────┐  compile  ┌──────────────┐ finalize ┌──────────────────┐
  │ build a typed tree │ ────────► │  IR + ABI    │ ───────► │ fn(...) -> ...    │
  │  of Staged values  │           │  lowering    │          │ .as_fn()/.run()   │
  └────────────────────┘           └──────────────┘          └──────────────────┘
```

- **Stage 0 ("now"):** plain Rust that _builds_ a computation — `Var<i64>`,
  `add(x, y)`, `slice.staged_iter().filter(..).sum(ctx)`. Nothing runs on data.
- **Stage 1 ("later"):** `Compiler::compile` JIT-compiles that description into a
  native function you call as many times as you like.

## Examples at a glance

Each snippet assumes `use rust_lms::prelude::*;` and a `let mut compiler = Compiler::new();`.

**Iterators** — push-based and fully fused: `filter` + `map` + `sum` compile to _one_
native loop with no allocation and no intermediate collection.

```rust
// Sum the squares of the positive elements of a slice.
let sum_pos_sq = compiler.fun1("sum_pos_sq", |ctx, xs: Var<SRef<Slice<i64>>>| {
    xs.staged_iter()
        .filter(|x| lt(0i64, x)) // keep positives
        .map(|x| x * x)          // square them
        .sum(ctx)
});
let f = compiler.compile(sum_pos_sq).expect("compile");
assert_eq!(f.as_fn().call(&[-2i64, 3, -1, 4][..]), 25); // 3² + 4²
```

**Slices** — `&mut [T]` is a 16-byte fat pointer; reads/writes are unchecked and
in-place, with `len()` register-resident in the loop.

```rust
// Double every element of a mutable slice in place.
let double = compiler.fun1("double", |ctx, mut xs: Var<SRefMut<Slice<i64>>>| {
    let i = ctx.var(0u64);
    ctx.while_loop(lt(i, xs.len()), move |ctx| {
        // SAFETY: the loop condition proves `i < xs.len()`.
        let v = ctx.bind(unsafe { xs.get_unchecked(i) });
        ctx.emit(unsafe { xs.set_unchecked(i, v * 2i64) });
        ctx.store(i, i + 1u64);
    });
    unit()
});
let mut data = [1i64, 2, 3, 4];
compiler.compile(double).expect("compile").as_fn().call(&mut data[..]);
assert_eq!(data, [2, 4, 6, 8]);
```

**Structs** — `#[derive(StagedType)]` gives typed, ABI-correct field access, including
_disjoint_ mutable borrows that are checked at `cargo build` (asking for the same
field twice is a type error).

```rust
#[derive(StagedType, Copy, Clone)]
#[repr(C)]
struct Point { #[staged(i64)] x: i64, #[staged(f64)] y: f64 }

let reset = compiler.fun1("reset", |ctx, p: Var<SRefMut<Point>>| {
    let (x, y) = split_fields_mut(p, PointType::x(), PointType::y());
    ctx.emit(store_ref(x, Const::<i64>::new(17)));
    ctx.emit(store_ref(y, Const::<f64>::new(2.5)));
    unit()
});
let mut p = Point { x: 0, y: 0.0 };
compiler.compile(reset).expect("compile").as_fn().call(&mut p);
assert_eq!((p.x, p.y), (17, 2.5));
```

**External pointers & FFI** — call ordinary Rust `extern "C"` functions from a JIT
kernel; `#[extern_fn]` generates the typed handle, so arguments and results keep
their staged types across the boundary.

```rust
#[extern_fn]
pub extern "C" fn ext_square(x: i64) -> i64 { x * x }

let square = compiler.extern_fn::<ExtSquareExtern>();
let quad = compiler.fun1("quad", |_ctx, x: Var<i64>| {
    call_extern1(square, call_extern1(square, x)) // ((x²)²) = x⁴
});
assert_eq!(compiler.compile(quad).expect("compile").as_fn().call(2), 16);
```

## What's in the box

- **Numbers & operators** — `i64`, `u64`, `i32`, `u32`, `f64`, `bool`, `()`, with
  `+ - * / %`, comparisons, bit ops, `int_cast`/`int_to_float`/`bitcast`, and
  branchless `select`/`min`/`max`.
- **Functions** — `fun0`..`fun8` (and recursive `_rec` variants), first-class
  `FunRef` handles, direct calls, and both safe (`Compiled::run`/`as_fn`) and
  owner-checked calling.
- **Control flow** — value-producing `if_then_else` (real phi nodes), `if_then`,
  `while_loop`, `break_loop`, `not`.
- **Memory** — typed references and raw pointers (`SRef`/`SRefMut`/`SPtr`/
  `SMutPtr`), load/store/offset/index, `ptr_is_null`, and ref↔ptr conversions.
- **Slices** — `&[T]`/`&mut [T]` as 16-byte fat pointers, closed under
  sub-slicing, with register-resident `(ptr, len)` for hot loops.
- **Structs & tuples** — `#[derive(StagedType)]` on `#[repr(C)] Copy` structs, with
  lifetime-aware field access (read, write, and _disjoint_ mutable borrows via
  `split_fields_mut`) and correct by-value / by-reference ABI; plus `Staged` tuples
  for the expression-tree authoring style.
- **Staged iterators** — push-based, fully fused: `map`/`filter`/`filter_map`/
  `scan`/`take_while`/`skip_while`/`zip`, plus `sum`/`count`/`min`/`max`/`fold`,
  branchless `sum_if`/`count_if`, short-circuiting `any`/`all`/`position`/
  `find_map`, and opaque external iterators driven from staged code.
- **Optionals** — staging-time `StagedOpt` (never materializes; becomes control
  flow) and FFI-safe `COption<T>`.
- **Host memory** — a runtime-writable `stack_alloc` scratch slot and a host-owned
  append-only `BytesPool` a kernel can grow via an extern.
- **FFI** — call ordinary Rust `extern "C"` functions from JIT code via
  `#[extern_fn]` and `FatSlice`, with a checked calling API plus `_unchecked`
  variants for hot paths.

## Workspace layout

| Crate                                 | Role                                                                                                                                                                                                                                                                          |
| ------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [`rust-lms`](rust-lms/)               | the staged-computation library and Cranelift backend                                                                                                                                                                                                                          |
| [`rust-lms-derive`](rust-lms-derive/) | `#[derive(StagedType)]` and `#[extern_fn]` proc macros                                                                                                                                                                                                                        |
| [`rust-lms-std`](rust-lms-std/)       | typed staged data structures built on `rust-lms` — a growable `SVec` (handle indirection: a JIT structure grows without dangling baked pointers) and query-shaped packed records (`RecordLayout` / `FieldId`)                                                                 |
| [`arrow-lms`](arrow-lms/)             | staged Apache Arrow interop — read a `RecordBatch`'s columns inside a JIT kernel via lifetime-free `FfiArray` descriptors, with first-class validity bitmaps                                                                                                                  |
| [`sql-gen`](sql-gen/)                 | a SQL → JIT engine: datafusion parses SQL, a small optimizer (predicate pushdown) runs, and the plan lowers to one rust-lms kernel per query — `Scan`/`Filter`/`Project`, scalar + `GROUP BY` aggregates, `Utf8View` strings, streaming multi-batch input, and hash **joins** |

## Build & test

The JIT is intentionally restricted to six 64-bit native targets:
`x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`,
`x86_64-apple-darwin`, `aarch64-apple-darwin`,
`x86_64-pc-windows-msvc`, and `aarch64-pc-windows-msvc`. Musl, MinGW,
Arm64EC, mobile, 32-bit, and big-endian targets are not supported.

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
- **[docs/](docs/)** — the SQL engine's design notes: `group_by.md`,
  `table_scan.md` (streaming input), and `joins.md`.
- The integration tests under `rust-lms/tests/` (`test_func.rs`, `test_iter.rs`,
  `programs.rs`, `p99.rs`, `euler.rs`, …) are the best worked examples of every
  feature.
