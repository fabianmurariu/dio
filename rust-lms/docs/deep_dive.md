# rust-lms Deep Dive

A guided tour of the `rust-lms` design for someone starting on the project — human
or AI agent. It explains the *mental model* first, then walks the codebase module
by module, and finishes with the ABI/codegen details and the invariants you must
not break.

If you only read one section, read [The mental model](#1-the-mental-model).

---

## 1. The mental model

`rust-lms` is a Rust implementation of **multi-stage programming** (staging), in
the spirit of Scala [LMS](https://scala-lms.github.io/) (Lightweight Modular
Staging). There are two phases:

- **Stage 0 — "now", the host program.** You run ordinary Rust that *builds a
  description* of a computation. Building blocks are normal Rust values
  (`Var<i64>`, `Add<L, R>`, `SliceIter<…>`, …). Composing them is just calling
  functions and constructors — no code runs on data yet.
- **Stage 1 — "later", the generated program.** When you call
  `Compiler::compile`, that description is lowered to [Cranelift] IR, JIT-compiled
  to native machine code, and handed back as a callable function pointer. *This*
  is what actually crunches numbers.

The key trick: **a value's Rust type encodes its staged type, and the Rust type
checker enforces the staged type system.** `add(x, y)` only compiles in stage 0 if
`x` and `y` agree on a numeric output type; you cannot add an `i64` to a `bool`
because there is no `Add` impl for that combination. So stage-0 type errors *are*
stage-1 type errors, caught at `cargo build` with no separate type checker.

```text
   Rust source (stage 0)            Cranelift + JIT            native code (stage 1)
  ┌────────────────────┐   compile  ┌──────────────┐  finalize ┌──────────────────┐
  │ add(mul(x,2), y)   │ ─────────► │   IR + ABI   │ ────────► │  fn(i64,i64)->i64 │
  │  (a tree of types) │            │  lowering    │           │  .as_fn() / .run()│
  └────────────────────┘            └──────────────┘           └──────────────────┘
```

Nothing in `rust-lms` interprets the expression tree at runtime. Every construct's
job is a single method — `Staged::codegen` — that emits Cranelift instructions.

> **Why this matters for contributors:** when you add a feature you are writing a
> *code generator*, not an evaluator. A new combinator is "correct" when the IR it
> emits is correct, and the Rust type bounds on its impl are what keep callers from
> emitting nonsense.

[Cranelift]: https://cranelift.dev/

---

## 2. The core trait: `Staged`

Everything that can become stage-1 code implements `Staged` (`src/staged.rs`):

```rust
pub trait Staged {
    type Out: StagedType;                              // the runtime type it yields
    fn codegen(&self, ctx: &mut CompilationContext) -> Value;
    fn var_id(&self) -> Option<usize> { None }         // Some(id) iff a bare Var
}
```

- `Out` is the *staged* type the expression produces at runtime (`i64`, `bool`,
  `SRef<Slice<f64>>`, …). It is an associated type, so it is inferred and checked
  by Rust.
- `codegen` emits the IR that computes the value and returns the Cranelift `Value`
  holding it. Side-effect-only constructs (assignment, stores, loops) return the
  cached **unit value** and have `Out = UnitType`.
- `var_id` is an optimization hook: a bare `Var` reports its id so slice
  operations can find register-resident `(ptr, len)` pairs instead of reloading
  from memory (see [§10 ABI](#10-the-abi-how-values-cross-the-boundary)).

This trait is the whole contract. Read a few impls back-to-back and the pattern is
obvious: pull operands' `Value`s via `operand.codegen(ctx)`, emit one or two
`ctx.builder.ins()` instructions, return the result.

---

## 3. The type system

`src/types.rs` defines what types may appear in staged computations.

### `StagedType` — the universe of staged types

```rust
pub trait StagedType {
    type RuntimeValue;                          // the real Rust type at runtime
    fn cranelift_type() -> Type;                // I64 / F64 / I8 / …
    fn size_of() -> usize;        fn align_of() -> usize;
    fn is_copy_struct() -> bool;  fn is_fat_pointer() -> bool;
    fn num_abi_values() -> usize; fn abi_types() -> Vec<Type>;
    fn should_pass_by_pointer() -> bool;
}
```

The ABI-related methods (`num_abi_values`, `abi_types`, `should_pass_by_pointer`,
`is_fat_pointer`) drive how a value is passed across the function boundary — see
[§10](#10-the-abi-how-values-cross-the-boundary). Primitives override only the
basics; structs and slices override the ABI methods.

### Type markers

The "marker" types are mostly just the real Rust primitives, so the API reads
naturally:

| Staged type | `RuntimeValue` | Cranelift | Notes |
|-------------|----------------|-----------|-------|
| `i64`       | `i64`          | `I64`     | signed; `Num` + `IntNum` |
| `U64Type` (`= u64`) | `u64`  | `I64`     | unsigned; `Num` + `IntNum` |
| `I32Type` / `U32Type` | `i32`/`u32` | `I32` | `Num` + `IntNum` |
| `F64Type` (`= f64`) | `f64`  | `F64`     | `Num` + `FloatNum` |
| `BoolType` (`= bool`) | `bool` | `I8`    | **not** `Num`; control flow only |
| `UnitType` (`= ()`) | `()`     | `I8`(0)   | result of side effects |

Because `i64`/`u64`/`f64`/`bool`/`()` are the markers themselves, a literal like
`5i64` *is* a runtime value and the From/IntoStaged impls turn it into `Const`.

### Refining traits

- **`ConstantType`** — can be embedded as a compile-time constant
  (`codegen_constant`). All primitives qualify.
- **`CopyType`** — semantically `Copy` (so it can live in a register / be reused).
  All primitives, plus structs whose fields are all `CopyType`.
- **`Num` / `IntNum` / `FloatNum`** (`src/num/traits.rs`) — capability traits that
  carry the actual instruction selection. `Num` provides `codegen_add/sub/mul/
  div/lt/gt/eq`; `IntNum` adds `codegen_rem`; `FloatNum` is a marker for future
  float-only ops. Signed vs unsigned vs float pick `sdiv`/`udiv`/`fdiv`,
  `SignedLessThan`/`UnsignedLessThan`/`FloatCC::LessThan`, etc. — this is the one
  place where signedness lives.

> **Invariant:** `bool` is deliberately not `Num`. Booleans flow through
> `if_then`, `select`, `not`, comparisons — never arithmetic.

---

## 4. Values: `Var`, `Const`, and `IntoStaged`

Two leaf node kinds, both in `src/staged.rs`:

- **`Const<T>`** — a literal embedded into the code (`Const::<i64>::new(5)`).
- **`Var<T>`** — a *lightweight handle* (`{ id: usize }`) to a Cranelift variable.
  It is `Copy` (when `T: Copy`), which is the whole point: you can use the same
  `x` twice in `add(x, x)` without clones or borrow gymnastics. The id is resolved
  to a real `cranelift_frontend::Variable` through `ctx.var_map` at codegen time.

**`IntoStaged<T>`** is the ergonomics glue. It lets any API accept either a staged
expression *or* a bare Rust literal: `add(x, 5i64)` works because `5i64:
IntoStaged<i64>` converts to `Const`. There is a blanket impl so anything already
`Staged<Out = T>` is its own `IntoStaged<T>`. Almost every builder function
(`add`, `assign`, `if_then_else`, `get_unchecked`, …) takes `IntoStaged` rather
than `Staged`.

`BoxableStaged::boxed()` exists for the rare case you need dynamic dispatch
(`Box<dyn Staged<Out = T>>`), e.g. to unify branches of differing concrete types.

---

## 5. Operations as types

Arithmetic and comparison live in `src/num/ops.rs`. The defining idea of the whole
library: **operations are separate generic structs, not enum variants.**

```rust
pub struct Add<L, R> { left: L, right: R }

impl<L, R, T> Staged for Add<L, R>
where L: Staged<Out = T>, R: Staged<Out = T>, T: Num {
    type Out = T;
    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let l = self.left.codegen(ctx);
        let r = self.right.codegen(ctx);
        T::codegen_add(l, r, ctx.builder)
    }
}
```

Consequences:

- **Type constraints are trait bounds.** `Add` requires both sides to share `T:
  Num`. `Lt`/`Gt`/`Eq` take `T: Num` but report `Out = BoolType` — a *heterogeneous*
  operation that changes the type. There is no runtime tag to get wrong.
- **No `Box` in the common path.** The expression "tree" is a nest of zero-cost
  generic structs (`Mul<Add<Var<i64>, Const<i64>>, Const<i64>>`), monomorphized
  and usually `Copy`.
- **Operator sugar.** `impl_num_ops_for!` wires up `core::ops::{Add,Sub,Mul,Div,
  Rem}` for every staged carrier (`Var`, `Const`, the op structs, `LetVar`,
  slice/ref/field accessors), so you can write `x * 2 + y` or `n % d` and get the
  same `Add`/`Mul`/`Rem` structs the free functions build.

`select(cond, a, b)` is a **branchless** conditional (Cranelift `select`/cmov);
`min`/`max` are built from `select` + `Lt`/`Gt`, so they stay vectorizable.

---

## 6. The `Compiler`, functions, and `Compiled`

`src/func.rs` is the coordinator. A `Compiler` owns all function definitions and
hands out variable ids.

### Defining functions

`fun0` … `fun8` (and `funN_rec` variants) define named staged functions. Each
takes a closure that receives an imperative builder (`Ctx`/`VarBuilder`) and one
`Var<A>` per parameter, and returns the body expression:

```rust
let square = compiler.fun1("square", |_ctx, x: Var<i64>| mul(x, x));
let gcd = compiler.fun2_rec("gcd", |f, _ctx, a: Var<i64>, b: Var<i64>| {
    if_then_else(eq(b, 0i64), a, call2(f, b, a % b))
});
```

`funN_rec` passes the function's own `FunRefN` in as the first closure argument so
the body can call itself.

### Calling and compiling

- `FunRefN<…, OUT>` is a type-safe handle. `call0`…`call3` (`src/func_impl.rs`)
  build call expressions; the arities above 3 are reachable through the generic
  call machinery.
- `compiler.compile(expr)` lowers everything: it declares all internal + extern +
  the `__main__` function, then defines each one by walking the stored bodies,
  finalizes the JIT module, and returns `Compiled<OUT>`.
- `Compiled::run()` executes a nullary computation and returns its value;
  `Compiled::as_fn()` returns a typed `extern "C" fn(...)` pointer when you
  compiled a *function reference* (so you can call it many times with different
  inputs). Note: `as_fn` works by compiling `__main__` to *return the function
  pointer*, then transmuting it.

### `Compiled` owns the JIT module

`Compiled<T>` holds the `JITModule`; dropping it frees the executable memory, so
keep it alive as long as you hold function pointers into it.

---

## 7. Two authoring styles

There are two ways to write a body. They interoperate freely.

### (a) Imperative — the `Ctx` builder (preferred)

`Ctx` (aka `VarBuilder`) records *actions* (closures that emit IR) in declaration
order. You call side-effecting methods and return the result expression:

```rust
let factorial = compiler.fun1("factorial", |ctx, n: Var<i64>| {
    let i      = ctx.var(1i64);     // declare & init a local -> Var<i64>
    let result = ctx.var(1i64);
    ctx.while_loop(lt(i, n + 1i64), move |ctx| {
        ctx.store(result, result * i);
        ctx.store(i, i + 1i64);
    });
    result                          // returned value
});
```

`Ctx` methods: `var`/`let_var` (declare+init), `bind` (evaluate an expression once
into a fresh var), `store`/`emit` (statements), `if_then`/`if_then_else`,
`while_loop`, and `break_loop` (jump to the innermost loop's exit). No `Clone`
bounds, no tuple threading — this is the ergonomic path and what most tests use.

### (b) Expression-tree — tuples + combinators

Tuples implement `Staged` (`src/tuple.rs`): a tuple runs each element for side
effects and yields the **last** element's value. Combined with `assign` and the
`while_loop`/`if_then_else` *expression* constructs (`src/control.rs`), you can
write a whole body as one value:

```rust
( (i, sum),                                   // init LetVars
  while_loop(lt(*i, arr.len()), (
      assign(*sum, *sum + arr.get_unchecked(*i)),
      assign(*i, *i + 1u64),
  )),
  *sum )                                       // result
```

The imperative `Ctx` style (a) covers the same ground with less ceremony, so new
code should prefer it; style (b) is mostly of interest when you want the whole body
to be a single value expression.

---

## 8. Control flow

`src/control.rs`:

- **`IfThenElse`** — a value-producing conditional. It creates `then`/`else`/`merge`
  blocks and adds a **block parameter** to `merge` (a phi node) that both branches
  jump to with their value. Both branches must have the same `Out = T`.
- **`IfThen`** — one-sided, side effects only, `Out = UnitType`.
- **`While`** — pre-checked loop, `Out = UnitType`. Note the sealing dance: the
  loop header is sealed *after* the back-edge is emitted (it has two predecessors).
- **`Not`** — boolean negation via `icmp_imm eq 0`.

The `Ctx` methods `if_then`/`if_then_else`/`while_loop`/`break_loop` are the
imperative mirrors of these and are usually what you reach for.

> **SSA note:** `rust-lms` leans on Cranelift's `FunctionBuilder` variable/SSA
> construction. Mutable locals are Cranelift `Variable`s (`declare_var`/`def_var`/
> `use_var`); the only hand-written phi is the `IfThenElse` merge block parameter.
> Get the block *seal* order right or Cranelift will panic.

---

## 9. Memory: references, slices, and structs

### References & pointers — `src/refer.rs`

Four pointer flavors, all i64 at the IR level, distinguished by type tags so the
*runtime* type and mutability are tracked:

| Staged type | Runtime | Tag |
|-------------|---------|-----|
| `SRef<'a, T>`    | `&T`       | `RustRef` |
| `SRefMut<'a, T>` | `&mut T`   | `RustRef` |
| `SPtr<T>`        | `*const T` | `RustPtr` |
| `SMutPtr<T>`     | `*mut T`   | `RustPtr` |

Operations: `load`/`load_ref`/`load_mut` (deref-read), `store`/`store_ref`
(deref-write, mutable only), `ptr_offset`(`_mut`) (scaled pointer arithmetic), and
`array_index` (offset + load). Mutability is enforced by the type: `store_ref`
only accepts `SRefMut`.

### Slices — `src/slice.rs`

`Slice<T>` is a DST marker that **never** implements `StagedType` on its own —
exactly like `[T]` in Rust, it must sit behind a reference: `SRef<Slice<T>>`
(`&[T]`) or `SRefMut<Slice<T>>` (`&mut [T]`). These are **16-byte fat pointers**
`(ptr, len)`.

Within the staged graph a slice's `codegen` value is a single i64, resolved two
ways (the *only* code that knows this is `CompilationContext::slice_data_ptr` /
`slice_len`):

- **register-resolved** — slice *parameters* are split at the ABI boundary into
  two Cranelift variables (`ptr_var`, `len_var`) kept in `ctx.slice_vars`. Slice
  ops read those registers directly — no memory traffic, the fast path for loops.
- **memory-resolved** — sub-slices (and anything without a `var_id`) are a pointer
  to a `(ptr, len)` pair on a stack slot (`ptr` at +0, `len` at +8).

The `SliceType` trait unifies `&[T]` and `&mut [T]` so a *single* op impl serves
both, with mutability carried in the associated `ElemRef`. Crucially `slice_unchecked`
reports `Out = S::Out`, so **slices are closed under sub-slicing**: a sub-slice of
a `&mut [T]` is still a `&mut [T]` supporting `len`/`get`/`set`/`slice` again. Ops:
`len`, `get_unchecked`/`get_ref_unchecked`, `set_unchecked` (mutable),
`swap_unchecked` (mutable), `slice_unchecked`/`slice_mut_unchecked`. All are
**unchecked** — bounds safety is the caller's responsibility.

### Structs — `src/struct.rs` + `#[derive(StagedType)]`

`#[derive(StagedType)]` on a `#[repr(C)] Copy` struct (in `rust-lms-derive`)
generates a `…Type` module of field tokens (each implementing `Field` with a
`memoffset`-computed `OFFSET`), the `StagedType` ABI impl, and a `CopyType` impl
gated on all fields being `CopyType`. Field access is split across four traits so
the *type* decides what's legal:

- `CopyFieldAccess::get` — load a Copy field (any struct-like input).
- `RefFieldAccess::get_ref` — `&field` (pointer inputs only; preserves lifetime).
- `MutRefFieldAccess::get_ref_mut` — `&mut field` (mutable pointer inputs only).
- `OwnedFieldAccess::field`/`get_ptr` — navigate/raw-pointer into by-value structs
  (`Var<Owned<T>>`); **no** `get_ref` (would dangle into a stack slot).

`Owned<T>` (`src/types.rs`) is the marker that says "this struct came in by value";
it forwards all ABI methods to `T` but distinguishes by-value from by-reference in
the type system so lifetimes stay sound.

---

## 10. The ABI: how values cross the boundary

The trickiest part of `compile` (`src/func.rs`) is matching the platform C ABI
(SystemV on Unix, WindowsFastcall on Windows) so JIT'd functions interop with real
Rust `extern "C"` fns. Four cases, driven by the `StagedType` ABI methods:

1. **Primitive** — one ABI value, bound straight to a Cranelift variable.
2. **Small struct (≤16 bytes)** — passed in registers (N×i64). On entry the values
   are stored to a fresh stack slot and the variable holds the slot pointer (so
   field access is uniform pointer+offset). On return they are reloaded.
3. **Large struct (>16 bytes)** — `should_pass_by_pointer()` is true; passed as a
   single pointer the caller already owns; used in place, no copy.
4. **Fat pointer (slice)** — `is_fat_pointer()` is true; the `(ptr, len)` pair is
   bound to **two** Cranelift variables recorded in `slice_vars` (the
   register-resolved encoding above).

`as_fn()` returns an `extern "C" fn(...)` precisely so these conventions line up
with Rust's view of the same signature.

### Set `RUST_LMS_DEBUG_IR=1` to see the generated IR

```bash
RUST_LMS_DEBUG_IR=1 cargo test -p rust-lms test_while_loop_factorial -- --nocapture
```

prints each function's Cranelift IR (including `__main__`) before it is compiled —
the first thing to reach for when codegen looks wrong.

---

## 11. Staged iterators

`src/iter/` is a **push-based**, fully-fused iterator framework. There is no
runtime iterator object: a `StagedIterator` emits an imperative loop, and the
consumer closure runs **once at staging time** to build the loop body.

```rust
let total = slice.staged_iter()
    .filter(move |x| gt(x, 0.0f64))
    .map(move |x| x * x)
    .sum(ctx);                    // -> Var<f64>
```

- **Sources:** `SliceIter` (from `arr.staged_iter()`), `RangeIter`/`RangeStep`
  (`range(lo, hi)`, `range_step(lo, hi, step)`). Sources that know their length
  implement `IndexedStagedIterator`/`IndexedSource`, which unlocks `zip`.
- **Combinators** wrap the consumer before handing it upstream — they introduce
  `if_then`s, never new loops: `map`, `filter`, `filter_map`, `scan` (stateful),
  `take_while`, `skip_while`, `zip`.
- **Terminals** drive `for_each`: eager reductions `sum`/`count`/`min`/`max`/`fold`,
  the branchless `sum_if`/`count_if` (predicated add via `select` instead of a
  data-dependent branch — keeps the loop vectorizable), and the short-circuiting
  `any`/`all`/`position`/`find_map` (which `break_loop` out of the source's single
  loop — safe because combinators add no loops of their own).

> **Design principle:** prefer branchless terminals in hot loops. `count_if`/
> `sum_if` emit a cmov per element; `filter(...).count()` emits a branch. Both are
> correct; the former stays SIMD-friendly.

See `docs/loop_unrolling_design.md` and `docs/staged_iterator_api.md` for the
design rationale and the roadmap toward unrolling/SIMD.

---

## 12. Staging-time optionals vs FFI optionals

Two distinct "maybe a value" tools — do not confuse them:

- **`StagedOpt`** (`src/staged_opt.rs`) is **not** a `StagedType` and never
  materializes: no discriminant, no memory. You *eliminate* it with `eliminate(ctx,
  on_some, on_none)`, which emits a branch and binds the payload in a register only
  inside the `Some` arm. `cond.then_some(value)` is the workhorse constructor;
  `s_some`/`s_none` are the static (branchless) ones. This is what powers fused
  `filter_map`/`find_map` — the `Some`/`None` becomes *control flow*.
- **`COption<T>`** (`src/option.rs`) is a real `#[repr(C, u64)]` value with a
  discriminant in memory — use it when an optional must be **stored**, returned
  across a function boundary, or passed via FFI. `OptRefType`/`OptMutRefType` are
  niche-optimized (`null` = `None`) single-i64 reference options.

Rule of thumb: control flow inside a body → `StagedOpt`; a value that must exist in
memory or cross FFI → `COption`.

---

## 13. FFI: calling back into Rust

`src/ffi.rs` + `#[extern_fn]` (in `rust-lms-derive`) let JIT code call ordinary
Rust functions:

```rust
#[extern_fn]
#[no_mangle]
pub extern "C" fn sum_array(data: FatSlice<i64>) -> i64 {
    unsafe { data.as_slice().iter().sum() }
}

let f = compiler.extern_fn::<SumArrayExtern>();          // ExternRef
let r = call_extern1::<_, _, FatSliceType<i64>, i64>(f, slice_arg);
```

`#[extern_fn]` generates a `…Extern` zero-sized type implementing the `ExternFn`
trait, carrying the symbol name, function pointer, and ABI types (mapping Rust
param/return types to staged types). `compiler.extern_fn::<S>()` registers the
symbol with the JIT and returns a typed `ExternRef`. `FatSlice`/`FatSliceMut` are
the `#[repr(C)]` slice representations that survive the C ABI by value.

---

## 14. A worked example, end to end

```rust
use rust_lms::prelude::*;

let mut compiler = Compiler::new();

// sum of squares of the positive elements of a slice
let f = compiler.fun1("sum_pos_sq", |ctx, arr: Var<SRef<Slice<f64>>>| {
    arr.staged_iter()
       .filter(move |x| gt(x, 0.0f64))
       .map(move |x| x * x)
       .sum(ctx)                       // Var<f64>
});

let compiled = compiler.compile(f).expect("compile");
let sum_pos_sq = compiled.as_fn();     // extern "C" fn(&[f64]) -> f64

let data = [1.0, -2.0, 3.0, -4.0];
assert_eq!(sum_pos_sq(&data[..]), 1.0 + 9.0);
```

What happened: stage 0 built `SliceIter → Filter → Map → sum`, which emitted a
single Cranelift loop that loads each `f64`, branches on `> 0`, squares, and
accumulates; `compile` JIT'd it; `as_fn` gave a native function pointer.

---

## 15. Invariants & gotchas for contributors

- **You are emitting code, not interpreting.** Every `Staged::codegen` must leave
  the builder in a valid state (current block terminated appropriately, blocks
  sealed once predecessors are known). Mis-ordered `seal_block` is the #1 cause of
  Cranelift panics.
- **Type bounds are the type system.** When adding an op, put the real constraints
  on the `Staged` impl (`T: Num`, `S::Out: MutSliceType`, …). If invalid stage-1
  code can be expressed, the bound is too loose.
- **Slice layout lives in exactly one place** — `slice_data_ptr`/`slice_len`. Don't
  re-derive ptr/len anywhere else; go through those helpers.
- **`bool` is not `Num`.** Keep booleans on the control-flow/`select` path.
- **Prefer branchless** (`select`, `min`/`max`, `count_if`/`sum_if`) in loop bodies
  when correctness allows — it preserves vectorizability.
- **Unchecked means unchecked.** Slice/pointer ops do no bounds checks; safety is
  the staged-program author's contract, surfaced through Rust types where possible
  (mutability, lifetimes via `SRef`/`Owned`).
- **`Compiled` owns the executable memory.** Keep it alive while using `as_fn`
  pointers.
- **Debug with `RUST_LMS_DEBUG_IR=1`.**

---

## 16. Module map

| Path | Responsibility |
|------|----------------|
| `src/lib.rs` | crate docs, module wiring, `prelude` |
| `src/staged.rs` | `Staged`, `Var`, `Const`, `IntoStaged`, `Assign`, `LetVar`, `CompilationContext`, slice/unit helpers |
| `src/types.rs` | `StagedType`, `ConstantType`, `CopyType`, type markers, `Owned<T>` |
| `src/num/` | `Num`/`IntNum`/`FloatNum` traits (`traits.rs`); `Add`…`Eq`, `Select`, `min`/`max`, operator sugar (`ops.rs`) |
| `src/control.rs` | `IfThenElse`, `IfThen`, `While`, `Not` |
| `src/func.rs` | `Compiler`, `Ctx`/`VarBuilder`, `fun0..8`(`_rec`), `compile`, `Compiled`, ABI lowering |
| `src/func_impl.rs` | `TypeInfo`, `FunTypeN`/`FunRefN`/`CallN`, `call0..3`, shared call codegen |
| `src/func_def.rs` | `FunDef::make_funN` construction helpers |
| `src/tuple.rs` | `Staged` for tuples (sequencing) |
| `src/refer.rs` | `SRef`/`SRefMut`/`SPtr`/`SMutPtr`, load/store/offset/index |
| `src/slice.rs` | `Slice<T>`, `SliceType`, fat-pointer ops |
| `src/struct.rs` | `Field`, field accessors, the four field-access traits |
| `src/iter/` | push-based staged iterators: sources, combinators, terminals (`traits.rs`) |
| `src/staged_opt.rs` | `StagedOpt`, `When`/`then_some`, `s_some`/`s_none` |
| `src/option.rs` | `COption<T>`, `OptRefType`/`OptMutRefType`, match/unwrap ops |
| `src/ffi.rs` | `FatSlice`/`FatSliceMut`, `ExternFn`, `ExternRef`, `call_externN` |
| `rust-lms-derive/` | `#[derive(StagedType)]`, `#[extern_fn]` proc macros |
| `sql-gen/` | (early) SQL → staged-code generation over `datafusion-sql` + `arrow` |

---

## 17. Where to read next

- **Tests are the spec.** `tests/programs.rs`, `tests/p99.rs`, `tests/euler.rs`,
  `tests/test_iter.rs`, `tests/test_slices.rs`, `tests/test_structs.rs`,
  `tests/test_extern_fn.rs`, `tests/type_safety.rs` are the best worked examples of
  every feature, in idiomatic modern style.
- `benches/filtered_sum_two_slices.rs` for performance-shaped code.
- `docs/staged_iterator_api.md` and `docs/loop_unrolling_design.md` for the
  iterator design and the unrolling/SIMD roadmap.
