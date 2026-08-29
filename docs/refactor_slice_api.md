# Slice API refactor plan

**Status:** proposed; **lifetime spike (row 1) executed — see "Spike results"**  
**Written:** 2026-08-26  
**Spike run:** 2026-08-27

## Purpose

Rethink the staged slice API around one representation and one operation
vocabulary while preserving the safety differences between:

- function parameters represented by `SRef<Slice<T>>` and
  `SRefMut<Slice<T>>`;
- shared and mutable views borrowed from `SVec<T>`;
- raw `(ptr, len)` descriptors received from FFI calls or descriptor fields;
- sub-slices derived from any of the above.

The desired user experience is that, once a value is established as a valid
shared or mutable slice, its storage origin is irrelevant. It should support the
same `len`, checked access, unchecked access, sub-slicing, pointer access, and
staged iteration operations as every other slice with the same capability.

This is an API and ownership refactor, not a backend representation refactor.
Both backends already use the canonical internal form:

```text
Value::Fat { ptr: Ptr, len: I64 }
```

## Executive decisions

1. **One representation does not mean one Rust type.** Shared, unique mutable,
   and raw descriptors have the same two-word representation but different
   validity and aliasing contracts. The type system must retain those contracts.
2. **Operations are unified by capability, not origin.** A function parameter
   and an `SVec` borrow should use the same trusted slice traits. An FFI result
   joins those traits only after an explicit provenance/validity conversion.
3. **`SVec` view validity is enforced by ordinary Rust borrows.** `as_slice(&self)`
   / `as_mut_slice(&mut self)` return views that borrow the handle; `push`/`set`
   take `&mut self`; the handle is neither `Copy` nor `Clone`. The one rule that
   makes this possible: **the lifetime belongs to the view, never to the staged
   expression it hands out** — `Ctx` retains expressions under `'static`, so a
   view that implemented `Staged` would pin its own borrow to `'static`. See §3
   and the spike Correction.
4. **Sub-slicing is closed.** A shared slice produces a shared slice; a mutable
   slice produces a mutable slice without duplicating the unique capability;
   a raw slice produces a raw slice until it is validated.
5. **Do not expose Rust slice references directly as a C ABI.** `FfiSlice<T>`
   and `FfiSliceMut<T>` remain explicit ABI transport structs. The staged API
   decides separately whether their contents have trusted provenance.
6. ~~**Prove the lifetime design before migrating the API.**~~ **DONE
   2026-08-27.** The spike showed borrowed staged expressions *can* be retained
   (S1-S5, S7, S8, S10), but that S6 is unreachable under any retained-graph
   design. Decision 3 supersedes it: the `'static` graph stays and validity is
   dynamic. See "Lifetime viability spike" for the recorded evidence.

## Current state

| Origin or role | Current staged type/API | Current issue |
|---|---|---|
| Shared function parameter | `Var<SRef<Slice<T>>>`, `SliceRefOps` | Good capability, but operation names and traits differ from raw and mutable slices. |
| Mutable function parameter | `Var<SRefMut<Slice<T>>>`, inherent methods plus `SliceMutOps` | Correctly non-`Copy`, but behavior is split between inherent and extension methods. |
| FFI descriptor value | `FatSliceType<T>` / `FatSliceMutType<T>` | ABI transport and slice capability are conflated; mutable descriptors do not participate fully in `SliceType`. |
| Raw parts | `slice_from_raw_parts`, producing `FatSliceType<T>` | Immutable-only constructor and a separate operation surface. |
| Descriptor reinterpretation | `AsSlice`, `AsMutSlice`, `AsRawSlice` and three conversion traits | Necessary safety witnesses are spread over several similarly named wrappers. |
| `SVec<T>` | Direct `len`, `get`, `set`, and `push`; no staged slice view | `SVec` is `Copy`, and `push(&self)` can reallocate, so a safe persistent slice view cannot currently exist. |
| Sub-slice | `SliceSliceUnchecked<S, ...>` | Correctly preserves `S::Out`, but public construction is split between three operation traits and is unchecked only. |
| Staged iteration | `SliceIter` hard-coded to `SRef<Slice<T>>` and `'static` | Raw, mutable-read, and future `SVec` views cannot use the same iterator entry point. |
| Backend lowering | `Value::Fat { ptr, len }` | Already canonical; no replacement is needed. |

## Target model

### 1. Representation umbrella

Keep a sealed staged-output trait describing the two-leaf representation. It can
evolve from the current `SliceType` rather than introducing another parallel
abstraction:

**As built (row 3):**

```rust
pub trait SliceType: StagedType + sealed::Sealed {
    type Elem: StagedType;
    type DataPtr: StagedType;      // SPtr<T> shared, SMutPtr<T> unique
}

pub trait TrustedSliceType: SliceType + sealed::TrustedSealed {
    type ElemRef: StagedType;      // SRef<T> shared, SRefMut<T> unique
}

pub trait MutSliceType: TrustedSliceType + sealed::MutableSealed {}
pub trait RawSliceType: SliceType + sealed::RawSealed {}
```

Two deviations from the sketch, both deliberate:

- **`ElemRef` moved from `SliceType` to `TrustedSliceType`.** Only a trusted
  slice can yield a *reference* to an element; a raw descriptor can offer no more
  than a pointer. The old shape forced `FatSliceType` to declare
  `ElemRef = SPtr<T>`, calling a pointer a reference. `ElemRef` had exactly one
  consumer (`SliceGetRefUnchecked`), so the move cost one tightened bound.
- **`MutSliceType`, not `MutableSliceType`.** The crate already spells this axis
  `Mut` everywhere — `SRefMut`, `SMutPtr`, `MutSliceRepr`, `SliceMutOps`,
  `AsMutSlice` — so `MutableSliceType` would have been the odd one out.

`len` and the data pointer sit on `SliceType`: reading a descriptor you already
hold dereferences nothing. Everything that touches memory needs
`TrustedSliceType`; writing additionally needs `MutSliceType`.

Planned classification:

| Staged type | Representation | Capability |
|---|---|---|
| `SRef<Slice<T>>` | `(ptr, len)` | trusted shared read |
| `SRefMut<Slice<T>>` | `(ptr, len)` | trusted unique read/write |
| `SVecSlice<T>` deref target | `(ptr, len)`, reloading | trusted shared read while the guard is held |
| `SVecSliceMut<T>` deref target | `(ptr, len)`, reloading | trusted unique read/write while the guard is held |
| FFI/raw shared descriptor | `(ptr, len)` | raw until explicitly validated |
| FFI/raw mutable descriptor | `(ptr, len)` | raw and unique only under an explicit unsafe contract |

`FatSlice<T>` and `FatSliceMut<T>` remain the runtime `#[repr(C)]` transport
types. During the naming pass, their staged markers should receive names that
make raw provenance obvious, for example `RawSlice<T>` and `RawSliceMut<T>`,
with compatibility aliases for `FatSliceType<T>` and `FatSliceMutType<T>`.

### 2. Operation umbrella

Expose one read API for every trusted slice expression and one extension for
trusted mutable expressions:

**As built (row 4).** Three traits, mirroring the three capability traits
one-for-one, each adding exactly what its capability licenses:

```rust
pub trait SliceOps: Staged + Sized where Self::Out: SliceType {
    fn len(self) -> ...;                    // touches no memory
    fn into_ptr(self) -> ...;               // DataPtr: SPtr or SMutPtr
    unsafe fn get_unchecked(self, i) -> ...;
    unsafe fn subslice_unchecked(self, start, end) -> ...;   // Out = Self::Out
}

pub trait TrustedSliceOps: SliceOps where Self::Out: TrustedSliceType {
    fn get_or(self, i, default) -> ...;     // SAFE
    unsafe fn get_ref_unchecked(self, i) -> ...;             // ElemRef
}

pub trait SliceMutOps: TrustedSliceOps where Self::Out: MutSliceType {
    fn set(self, i, value) -> ...;          // SAFE
    unsafe fn set_unchecked(self, i, value) -> ...;
    unsafe fn swap_unchecked(self, i, j) -> ...;
}
```

The dividing line is **safety, not the type you hold**: the two safe,
bounds-checked operations (`get_or`, `set`) are exactly the ones needing
established provenance, so they sit above `TrustedSliceType`. Everything on
`SliceOps` either touches no memory or is already `unsafe` — which is why
**`RawSliceOps` was deleted outright**: a raw descriptor simply stops at
`SliceOps` and needs no trait of its own.

Consolidations the taxonomy paid for: `get_ref_unchecked` replaces
`get_ref_unchecked`/`get_mut_unchecked` (mutability rides on `ElemRef`);
`subslice_unchecked` replaces `slice_unchecked`/`slice_mut_unchecked`
(`Out = Self::Out`); `into_ptr` replaces `into_ptr`/`into_mut_ptr` (`DataPtr`);
`len` replaces four spellings across two names.

Two findings from doing it:

- **A by-value blanket trait silently shadows inherent `&self` methods.** Rust
  probes the receiver type by value *before* autoref, so `SliceOps::len(self)`
  beat the inherent `Var::<SRefMut<_>>::len(&self)` and moved the variable. This
  is why the pre-refactor code had *no* blanket impl for the mutable trait — the
  duplication was load-bearing. Resolved by deleting the inherent family entirely
  and exposing `Var::reborrow(&mut self) -> VarUse<T>`, making the distinction
  explicit exactly as `&mut *x` does in Rust: `arr.reborrow().len()` keeps the
  variable, `arr.subslice_unchecked(..)` consumes it. That is also what §2's own
  rule asks for — no second family of convenience methods on `Var<T>`.
- **`as_ptr` was the wrong name.** Every op here consumes, and `as_*`
  conventionally borrows (clippy's `wrong_self_convention` flags it). Kept
  `into_ptr`.

The final receiver choices must preserve unique capabilities. In particular,
mutable projection may need to consume a value or borrow it through `&mut self`;
it must never clone or recreate the same `Var<SRefMut<_>>` ID.

Raw descriptors get a deliberately small API:

- inspect `len` and the raw data pointer;
- derive another raw descriptor through an unsafe range operation;
- cross an explicit unsafe `assume_valid`/`borrowed_from` boundary into a
  trusted shared or mutable view.

Rust cannot make one trait method safe for trusted implementations and unsafe
for raw implementations. Keeping the raw promotion explicit is therefore
simpler and safer than either making every slice access unsafe or pretending
that every FFI-returned pointer is a reference.

### 3. `SVec` borrowing — **DECIDED: ordinary Rust borrows** (revised)

> **Revised 2026-08-28.** This section previously specified stage-0 dynamic
> tracking with an `Rc<Cell<i64>>`, on the grounds that S6 was unreachable with
> type-level borrows. That premise was false (see the spike Correction). The
> tracker has been removed; a plain Rust lifetime does the whole job, with
> compile-time errors instead of build-time panics and ~35 fewer lines.

`SVec<T>` is a unique capability: neither `Copy` nor `Clone`, with `push`/`set`
taking `&mut self`. `as_slice(&self)` and `as_mut_slice(&mut self)` return views
that borrow the handle, so the borrow checker enforces the whole discipline.

Two properties make it work, and both already exist in the codebase:

1. **Reloading views.** The view's `codegen` reloads `{ptr, len}` from the control
   block, so a *retained* view expression is never stale (`AsRawSlice` already
   does exactly this, and `SVec::data()` already follows the discipline).
2. **The lifetime is on the guard, never on the staged expression.** This is the
   load-bearing detail. `Ctx` retains staged expressions under a `'static`
   bound, so a view that *itself* implemented `Staged` would have its borrow
   pinned to `'static` and could never be released. The guard carries `'a`; the
   expression it hands out through `Deref` is lifetime-free and `Copy`.

```rust
/// Stage-0 only; never reaches codegen. >0 = shared views, -1 = exclusive.
#[derive(Clone, Default)]
struct StageBorrow(Rc<Cell<i64>>);

/// A `Copy`, lifetime-free, reloading slice expression.
pub struct SliceExpr<T> { ctl: *mut RawVec, _t: PhantomData<T> }

/// The borrow guard. Not `Copy`; `Drop` releases the borrow.
pub struct SVecSlice<T>    { expr: SliceExpr<T>,    guard: SharedGuard }
pub struct SVecSliceMut<T> { expr: SliceExprMut<T>, guard: ExclusiveGuard }

impl<T> Deref for SVecSlice<T> {
    type Target = SliceExpr<T>;          // Copy -> by-value ops reach it
    fn deref(&self) -> &SliceExpr<T> { &self.expr }
}

impl<T> SVec<T> {
    pub fn as_slice(&self) -> SVecSlice<T>;
    pub fn as_mut_slice(&mut self) -> SVecSliceMut<T>;
    pub fn push(&mut self, ctx: &mut Ctx, value: Var<T>);  // asserts unborrowed
}
```

Because `SliceExpr<T>` is `Copy`, method resolution on `slice.get_unchecked(i)`
autoderefs and *copies* the inner expression out; the guard stays in the view.
Verified: a by-value trait method reached through `Deref` does not move the
guard, and the count drops only at scope end. So:

- one value at the call site, no tuple;
- retained nodes hold only the `Copy` expression, which reloads — never stale;
- `drop(view)` (or end of scope) releases the borrow, so **S6 works**;
- the `Deref` target is an ordinary staged slice, so the view gets the whole
  common trait family with no lifetime parameter of its own.

Requirements:

- `SVec` is not `Copy`; cloning must not duplicate the reallocation capability;
- shared views may be cloned (the `Rc` count rises), matching `&[T]`;
- mutable views are neither `Copy` nor `Clone`;
- `push` and every capacity-changing operation require `&mut SVec` **and** assert
  the borrow count is zero, panicking at kernel-build time otherwise;
- the raw SQL construction path stays unsafe and must not manufacture a second
  handle to the same control block while a view is live.

**Open (D2): the snapshot hole.** `ctx.bind` of a view's fat `(ptr, len)` yields a
`Var` that outlives the guard and goes stale across a growth. `Var` is `Copy`
with no `Drop`, so it cannot carry a guard. Either withhold the safe API that
materialises a view's fat value into a `Var`, or mark that one path `unsafe`.
Every other op reloads, so this is the sole residual hazard.

### 4. FFI results and provenance

An arbitrary safe Rust function can return a dangling `FatSlice<T>` without
using unsafe code because `FatSlice` itself is only a pair of public fields and
does not carry a lifetime. Therefore an FFI result cannot automatically become
`SRef<Slice<T>>`.

The initial contract is:

```text
extern result -> raw staged slice -> explicit unsafe provenance conversion
              -> ordinary trusted SliceOps
```

The unsafe conversion must state:

- which owner keeps the allocation alive;
- the maximum lifetime of the view;
- alignment and initialized element count;
- whether mutation is excluded or exclusive;
- whether the producer may reallocate the storage.

Future work may add annotated extern return relationships such as "borrowed
from argument 0" or an owned FFI buffer with drop glue. Those are not required
to complete this refactor and must not be guessed by the macro.

Rust `&[T]` and `&mut [T]` remain unsupported on the safe `extern "C"` path
because Rust slice references do not have a stable C ABI. The thunk may know how
to transport them internally, but that is not a public ABI guarantee.

### 5. Sub-slicing

`SliceSliceUnchecked` already has the essential rule `Out = S::Out`. Preserve
that rule in the consolidated API:

| Source | Sub-slice result |
|---|---|
| trusted shared | trusted shared, same borrow lifetime |
| trusted mutable | trusted mutable, with the parent consumed or reborrowed |
| raw shared | raw shared |
| raw mutable | raw mutable without duplicating uniqueness |

Two range operations, **both taking `(start, end)` rather than a range**:

1. `get_range(start, end)` returns a [`StagedOpt`] slice, `Some` when
   `start <= end && end <= len`.
2. `subslice_unchecked(start, end)` remains the proof-carrying fast path.

**Range syntax was checked and rejected.** The naming pass said to adopt
`start..end` "after checking inference ergonomics"; the check fails.
`std::ops::Range<Idx>` has a *single* index type, so the common mixed form —
a literal start with a staged end — is a type error at the `..` itself:

```text
error[E0308]: mismatched types
  let _r = 0u64..len;      // len: Var<u64>
                  ^^^ expected `u64`, found `Var<u64>`
```

That is 5 of the 25 sub-slice call sites in this workspace, and they are the
dynamic ones (`0u64, len`, `1u64, a.len()`, …). Two arguments accept mixed
`IntoStaged<u64>` types; a range cannot. Consolidation is therefore in the
*vocabulary*, achieved in row 4 (`slice_unchecked`/`slice_mut_unchecked` ->
one `subslice_unchecked`), with `get_range` sharing its argument shape.

`get_range` is **safe and lives on `SliceOps`**, not on the trusted traits:
bounds-checking and adjusting a `(ptr, len)` pair dereferences nothing, so a raw
descriptor gets a checked *raw* sub-slice exactly as a trusted slice gets a
trusted one. The bounds test lowers to one branchless `select` (there is no
`le`/`and` op, so it reads as "if `start > end` then false, else `end <= len`"),
and the sub-slice is only built on the taken arm.

Implementation note: the slice expression is needed twice (once for `len`, once
for the sub-slice), which a unique origin cannot supply by cloning. `get_range`
therefore binds once and reborrows — the same idiom row 5 found for generic
helpers — which is what lets one implementation serve every capability instead
of only shared ones.

A Rust-indexing-style trapping operation can be added only after the project
defines one backend-independent trap/runtime-failure contract. Clamping is not
acceptable because it does not match Rust semantics.

## Lifetime viability spike (executed; route not adopted)

The lifetime proposal must be tested before production types or downstream
crates are migrated. The spike should change the smallest possible internal
surface and may be discarded after its findings are recorded here.

### Hypothesis

Replace the hard-coded `'static` deferred graph with a graph lifetime:

```rust
type CodegenAction<'stage> =
    Box<dyn FnOnce(&mut CompilationContext) + 'stage>;

pub struct Ctx<'stage> {
    actions: Vec<CodegenAction<'stage>>,
}

struct FunDef<'stage> { /* body: ... + 'stage */ }
```

Propagate the same lifetime through `VarBuilder`, `Compiler`, function
definition helpers, and only the combinators that retain staged expressions.
Do not change `Value::Fat`, either backend, or the private storage-pointer ABI.

`Compiler<'a>` and `Compiled<'a, T>` already contain lifetime parameters, but
they are not evidence that this works: `FunDef` and `CodegenAction` currently
remain `'static`. The spike must determine what those lifetimes actually own
and whether build-time borrows and runtime storage lifetimes need to be
separate.

### Spike experiments — RESULTS

Executed 2026-08-27 against a `'stage`-parameterised deferred graph
(`CodegenAction<'stage>`, `Ctx<'stage>`, `FunDef<'stage>`,
`Compiler.functions: Vec<Option<FunDef<'a>>>`, `BODY: … + 'a`).
**93 lines across 3 files** (`func.rs`, `func_def.rs`, `llvm/mod.rs`) and
**zero changes** in `rust-lms-std`, `arrow-lms`, or `sql-gen` — all 73 downstream
`&mut Ctx` signatures elide to `&mut Ctx<'_>` unchanged.

Evidence lives in `rust-lms/tests/spike_lifetime.rs` and
`rust-lms/tests/spike_dyn_borrow.rs`, plus the `compile_fail` doctest on
[`Compiled`] in `func.rs` (in `src/`, so cargo actually enforces it — a
`compile_fail` block inside `tests/` is never run).

| # | Experiment | Result | Finding |
|---:|---|:---:|---|
| S1 | Retain a borrow of non-`'static` host storage, compile, run | **PASS** | Works on Cranelift and LLVM, no `transmute`, no erasure. |
| S2 | The borrow through `bind`/`var`/`store`/`while_loop`/`if_then_else` | **PASS** | The graph lifetime propagates through every retention point. |
| S3 | Two simultaneous shared views | **PASS** | Shared views are `Copy` and coexist. |
| S4 | `push` while a shared view is used later | **PASS** (rejected) | Ordinary `E0502`. |
| S5 | Second mutable view / read while a mutable view is live | **PASS** (rejected) | Ordinary borrow conflict. |
| S6 | Stop using a view, then `push` | **PASSES** (after correction) | The original run was wrong — see "Correction" below. |
| S7 | Derive a sub-slice, including a mutable sub-slice | inherits S1/S2 | No new lifetime machinery required. |
| S8 | Drop host storage before invoking compiled code | **PASS** (rejected) | Falls out for free; see "One lifetime, not two". |
| S9 | Return a captured `SVec` slice from a generated function | not run | Deferred; rejected in the first API. |
| S10 | Two functions sharing one host owner | **PASS** | Both retained in the same `Compiler<'a>`. |

#### One lifetime, not two — Risk #2 dissolves

The document worried that build-time and runtime lifetimes are different, and
that `Compiled` might need a separate owner lifetime or an unsafe constructor.
It does not. Threading **one** `'stage` lifetime through
`Ctx → FunDef → Compiler.functions → Compiled` makes `Compiled<'a, T>`'s
already-present `PhantomData<&'a T>` load-bearing for the first time, and S8
follows automatically:

```text
error[E0597]: `host` does not live long enough
   |         -------- borrow later stored here
   |         `host` dropped here while still borrowed
```

Before the spike, `Compiler<'a>`/`Compiled<'a, T>` were **pure decoration**: `'a`
was an unconstrained phantom, so `Compiler::new()` inferred whatever the caller
wanted (which is why `tests/common/mod.rs` can write `Compiler<'static>`).

This answers the open question in §3: `'borrow` and `'host` collapse into one
lifetime, and it does not appear in common signatures.

#### Correction (2026-08-28): S6 does pass — the original spike was flawed

The first spike concluded S6 was unreachable. That conclusion was **wrong**, and
the error was in the spike, not in Rust.

The spike made the *view itself* implement `Staged`:

```rust
struct View<'b> { ctrl: *mut i64, _b: PhantomData<&'b MockSVec> }
unsafe impl<'b> Staged for View<'b> { ... }   // <-- the mistake
let bound = ctx.bind(view);                    // requires Staged + 'static
```

`Ctx::bind` requires `'static`, so `'b` was forced to `'static`, which pinned the
borrow of the `SVec` to `'static` — hence "cannot borrow as mutable" forever.
The mechanism was `ctx.bind`'s bound leaking onto the view's lifetime parameter,
not the retained graph as such.

Put the lifetime on a **guard** that hands out a *lifetime-free* staged
expression — the `Deref` shape this document had already settled on for
ergonomics — and nothing forces `'static` onto the borrow. NLL then ends it at
the view's last use, exactly as it would for `&[T]`:

```rust
let n = { let view = svec.as_slice(); view.len(ctx) };  // borrow ends here
svec.push(ctx, v);                                       // accepted
```

Verified end to end: `view_released_then_grow` passes, and the three aliasing
violations become **compile errors** rather than build-time panics.

**Consequence: read-xor-grow was never a real constraint,** and the stage-0
dynamic tracker is unnecessary. See the revised §3.

#### The stage lifetime on a slice *parameter* is inert

`RuntimeParam::Arg<'call>` and `RuntimeResult::Output<'call>` decouple the staged
type's lifetime from the invocation lifetime, so a parameter declared
`Var<SRef<Slice<i64>>>` still accepts a short-lived `&data[..]` at call
time (verified). **Only slices baked at staging time** — `SVec` views, descriptor
fields — ever need a non-`'static` stage lifetime.

Consequence for row 10: generalising the iterator family is *not* required to
make slice parameters work. It is required only if stage-time-baked slices are
given type-level lifetimes. If they instead use the dynamic scheme below, the
`'static` bounds in `iter/traits.rs` cost nothing.

#### Alternative validated: stage-0 dynamic borrow tracking

Because staging *is* an ordinary Rust program, and it runs in exactly emission
order — the timeline the reallocation hazard actually lives on — a
`RefCell`-style counter checked at staging time is **more precise** than the
borrow checker, which can only see the whole-`'stage` over-approximation.
Violations panic while the kernel is being *built*, which for a staging library
is a build-time failure, not a production one.

Two properties make it work, and both are already available:

1. **Reloading views.** `AsRawSlice::codegen` reloads `{ptr, len}` from the
   control block at codegen, so a *retained* view expression is never stale.
   The existing `into_raw_slice` → `FatSliceType<T>` path is exactly this.
2. **Lifetime-free views.** With validity enforced dynamically, the view needs no
   `'stage` parameter, so it joins the existing slice trait family untouched.

Measured against the same experiments (`spike_dyn_borrow.rs`):

| Experiment | Type-level borrows | Stage-0 dynamic tracking |
|---|:---:|:---:|
| S3 two shared views | pass | pass |
| S4/S5 alias rejection | compile error | staging-time panic |
| **S6 grow after view released** | **impossible** | **pass** |
| View joins existing slice API | needs `'stage` on iterator traits | **no change needed** |
| Failure surfaces at | compile time | kernel-build time |

Residual hazard to close if this route is taken: *snapshotting*. A
`ctx.bind` of a fat `(ptr, len)` value produces a `Var` that survives a later
growth. The tracker must gate snapshot creation, not just view creation.

### Revised go/no-go criteria

Proceed with borrowed `SVec` views only if:

- S1-S5, S7, S8, S10 pass without `transmute`, leaked allocations, global borrow
  registries, or reverting borrowed slice markers to `'static` — **met**;
- the error messages identify an ordinary Rust borrow conflict — **met**;
- both backends still consume exactly the same neutral value graph — **met**
  (361 Cranelift / 379 LLVM, clippy `-D warnings` and `fmt --check` clean);
- common APIs do not require users to write lifetime parameters routinely —
  **met** (zero downstream signature changes);
- the lifetime of runtime storage used by compiled code is explicit and
  enforceable — **met** via the single `'stage` lifetime.

**S6 is struck from the criteria**: it is unachievable under any retained-graph
design, and the original list would have produced a spurious no-go. The
read-xor-grow consequence replaces it as an accepted design constraint of the
type-level route.

## Open decisions

Everything else in this plan is settled. These four are not, and D1 is large
enough to reorder the plan.

### D1 — Do staged reference types keep their lifetime parameter?

**Evidence.** With the `'static` deferred graph (now confirmed as the design),
every staged lifetime is pinned to `'static`. `ctx.bind`/`var`/`store`/`emit` and
`BODY: Staged + 'static` force it; the first spike reproduced the exact error
(`argument requires that 'a must outlive 'static`). And the `'a: 'static` bounds
in `slice_iter.rs` are not independently removable — they are consequences of
`IndexedSource: Clone + 'static` and its `'static` associated-type bounds, which
exist precisely because retained expressions must be `'static`.

Separately, `RuntimeParam::Arg<'call>` / `RuntimeResult::Output<'call>` already
carry the *real* invocation lifetime, and are independent of the staged type's
`'a` — a `Var<SRef<'static, Slice<i64>>>` parameter accepts a short-lived
`&data[..]` (verified).

So `'a` in `SRef<'a, T>` / `SRefMut<'a, T>` / `SRef<'a, Slice<T>>` can only ever
be `'static`, across roughly **137 mention sites and 68 `impl<'a>` blocks** in
`rust-lms` alone. It is decoration that reads like a guarantee.

**The one place it does work:** `StagedType::RuntimeValue = &'a T::RuntimeValue`,
consumed by `Compiled::run() -> T::RuntimeValue`. Dropping `'a` means references
need a lifetime-free `RuntimeValue` — the honest choice being a typed raw pointer
(`*const T::RuntimeValue`), which pushes reference-returning kernels onto
`as_fn().call()`, where `Output<'call>` gives a properly bounded `&'call T`.
Note `Compiled<'a, T>`'s `'a` is an unconstrained phantom today, so `run()`'s
current reference guarantee is not real anyway.

**DECIDED 2026-08-27: (a) strip it everywhere. DONE — row 0.5.**

Landed: `SRef<T>`, `SRefMut<T>`, `SRef<Slice<T>>`, `SRefMut<Slice<T>>`,
`OptRefType<T>`, `OptMutRefType<T>` and its four constructors, plus the
`LoadRef`/`LoadMutRef`/`StoreRef`/`IntoMutRef` helpers (which now match the
already-lifetime-free `LoadPtr`/`LoadMutPtr` beside them). The four slice op
traits (`SliceRefOps`, `SliceMutOps`, `ReprSliceOps`, `ReprSliceMutOps`) lose
their `'a`. Reference `RuntimeValue` becomes a typed raw pointer — `*const T` /
`*mut T`, and `*const [T]` / `*mut [T]` for slices, `*const T` (null = None) for
the optional forms. Nothing referenced these through `run()`, so no call site
changed. Safe reference results still go through `as_fn().call()`, where
`Output<'call>` supplies the real bound.

Corroboration found while doing it: `rust-lms-derive` was already hard-coding
`SRef<'static, ..>` / `SRefMut<'static, ..>` in all four places it emits them —
the macro had been writing the only lifetime that was ever possible.

Measured: **137 -> 0** explicit lifetime arguments on staged reference types,
**6 -> 0** vacuous `'a: 'static` bounds, `impl<'a>` blocks **68 -> 21** in
`rust-lms`. 353 Cranelift / 372 LLVM tests, 23 doctests, clippy `-D warnings` and
`fmt --check` clean on both backends.

**Still decoration, not yet stripped:** `Compiler<'a>` and `Compiled<'a, T>` carry
an unconstrained phantom `'a` (this is why `tests/common` can write
`Compiler<'static>`). Same defect, but it is the *function* API rather than the
slice API — left for a separate decision.

### D2 — The snapshot hole in `SVec` views

See §3. `ctx.bind` of a view's fat `(ptr, len)` produces a `Var` that outlives the
guard. **Proposed: withhold the safe API** rather than add an `unsafe` escape —
every other op reloads, so nothing else needs it. Open until contradicted.

### D3 — Compatibility aliases during the rename

**DECIDED 2026-08-27: clean break, no aliases.** Pre-1.0 with three in-tree
consumers; rename outright and fix the call sites. Nothing carries two names.
Strike the "deprecate then remove" half of row 12.

### D4 — Optional type for `get_range`

**DECIDED 2026-08-27: `StagedOpt`.** A checked sub-slice is branched on at the
use site, so it should lower to control flow and never materialise a tag. If a
stored/returned checked range is ever needed, add it then as a separate
constructor rather than widening `get_range`.

## Ordered implementation plan

Every row is intended to be independently reviewable and green before the next
row begins.

| Order | Deliverable | Goal | Role in the bigger picture | Exit criterion |
|---:|---|---|---|---|
| 0 | Characterization matrix | ~~Capture current behavior~~ **DONE 2026-08-27** — `rust-lms/tests/slice_characterization.rs`, 15 cells over 4 origins, both backends. | Prevents the redesign from losing ABI or typed-leaf behavior that already works; surfaced three gaps (G6a/G6b/G10) that now have explicit exit criteria. | **Met.** 368 Cranelift / LLVM green, clippy `-D warnings` clean. |
| 0.5 | **Staged lifetime removal (D1)** | ~~Strip `'a` from staged reference types~~ **DONE 2026-08-27.** `SRef<T>`, `SRefMut<T>`, `SRef<Slice<T>>`, `OptRefType<T>`, `OptMutRefType<T>`, and the `LoadRef`/`LoadMutRef`/`StoreRef`/`IntoMutRef` helpers; the four slice op traits lose their `'a`; reference `RuntimeValue` becomes a typed raw pointer. | Stops the taxonomy being written with lifetimes and then stripped. | **Met.** 137 -> 0 explicit lifetime args, 6 -> 0 vacuous `'a: 'static` bounds, `impl<'a>` 68 -> 21; 353 Cranelift / 372 LLVM, clippy `-D warnings` clean. |
| 1 | Lifetime viability spike | ~~Run S1-S10~~ **DONE 2026-08-27.** Results and revised criteria recorded above. | Determined that Rust borrows *can* found `SVec` views, with read-xor-grow; and that a stage-0 dynamic alternative exists. | **Met.** Spike code is in the tree, uncommitted, green on both backends. |
| 2 | ~~Lifetime-aware deferred graph~~ | **STRUCK.** Superseded by the §3 decision: validity is dynamic, so the `'static` graph stays and no `'stage` parameter is introduced. | — | Spike code removed; tree back to baseline. |
| 3 | Capability taxonomy and names | ~~Finalize the sealed traits~~ **DONE 2026-08-27.** Four traits: `SliceType` (representation) / `TrustedSliceType` (validity, owns `ElemRef`) / `MutSliceType` (trusted + writable) / `RawSliceType` (unknown provenance). `FatSliceMutType` classified for the first time (half of G6b). Aliases: none, per D3. | Gives all later APIs one vocabulary and makes provenance visible in signatures. | **Met.** `tests/slice_taxonomy.rs` asserts all 16 positive cells + associated-type projections + sub-slice closure; the four negative cells are `compile_fail` doctests that genuinely fail. |
| 4 | Consolidated core operations | ~~Replace the overlapping op surfaces~~ **DONE 2026-08-27.** Four surfaces (`SliceRefOps`, `RawSliceOps`, `SliceMutOps`, the `Var<SRefMut<..>>` inherent family) collapse to three blanket-implemented traits mirroring the capability traits. Names standardized on `len`/`subslice_unchecked`/`into_ptr`. Closes the rest of G6b. | Removes the largest source of slice API duplication while retaining safety distinctions. | **Met.** 369 Cranelift / 388 LLVM, 25 doctests, clippy `-D warnings` clean. |
| 5 | Function parameter migration | ~~Move call sites to the consolidated traits~~ **DONE 2026-08-28.** Largely pre-paid by row 4's fallout; the remaining work was collapsing `MutField`'s three bespoke `slice_*` methods onto the common traits behind one `as_mut_slice`, and adding origin-independence proofs. ABI untouched. | Establishes the simplest trusted origin as the reference implementation. | **Met.** 372 Cranelift / LLVM, 26 doctests, clippy `-D warnings` clean. |
| 6 | Raw/FFI descriptor boundary | ~~Rename markers, add constructors, consolidate witnesses, add promotion~~ **DONE 2026-08-28.** `FatSliceType`/`FatSliceMutType` -> `RawSlice`/`RawSliceMut`; four dead `Ffi*` aliases deleted; one raw-parts node now serves both `slice_from_raw_parts` and the new `slice_from_raw_parts_mut`; every descriptor conversion yields **raw** with trust arriving only via `RawSliceOps::assume_shared`/`assume_unique`; G6a closed. | Lets FFI-returned and descriptor-backed slices join the common API without laundering raw pointers into safe references. | **Met.** Two `compile_fail` doctests prove a raw value reaches no safe accessor before promotion and that a shared raw cannot promote to a unique view. |
| 7 | `SVec` unique capability | ~~Remove `Copy`, `&mut self` mutation, add a tracker~~ **DONE 2026-08-28.** `SVec<T>` is neither `Copy` nor `Clone`; `push`/`set` take `&mut self`; `as_slice`/`as_mut_slice` return lifetime-carrying views, so ordinary Rust borrows enforce the discipline; `from_raw_unchecked` stays as the one documented escape hatch. No dynamic tracker — see the §3 revision. | Creates the owner whose borrows prevent generated reallocation. | **Met, and exceeded:** the three alias violations are *compile* errors (`compile_fail` doctests) rather than panics; `view_released_then_grow` (S6) passes; sql-gen 113/113 green. |
| 8 | `SVec` shared and mutable views | ~~Add the `Deref` guards~~ **DONE 2026-08-28.** `SVecSlice<'a,T>`/`SVecSliceMut<'a,T>` deref to `Copy`, lifetime-free, reloading `SVecSliceExpr<T>`/`SVecSliceExprMut<T>` whose `Out` is `SRef<Slice<T>>`/`SRefMut<Slice<T>>`. Built by composing row 6's `slice_from_raw_parts_mut` + `assume_shared`/`assume_unique`. `rust-lms-std` gained an `llvm` feature and a `for_each_backend` harness. | Makes growable output storage readable/writable through the same API as function parameters. | **Met.** One generic helper drives a parameter slice and an `SVec` view unchanged; `view_released_then_grow` passes; all 9 SVec tests run on both backends. |
| 9 | Closed and checked sub-slicing | ~~Consolidate range syntax, add `get_range`~~ **DONE 2026-08-29.** `get_range(start, end)` yields a `StagedOpt` (D4), safe and available on every origin; `subslice_unchecked` stays the proof-carrying primitive. Range syntax **rejected on the ergonomics check the doc asked for** — see below. | Makes "a slice of a slice is a slice" true across the entire public API. | **Met.** `SliceGetRange::Item = S::Out` asserted; range tests cover shared/unique/raw origins, nesting, and writing through a checked sub-slice. |
| 10 | Slice iteration adapter | ~~Generalize `SliceIter`~~ **DONE 2026-08-29.** Keyed on `TrustedSliceType<Elem = T>`; `for_each` binds-once-and-reborrows instead of requiring `S: Clone`, which is what lets *unique* origins iterate. `IndexedSource` (zip) keeps `Clone`, so it stays shared-only. No `'stage` parameter; the `'static` bounds stay. | Ensures iteration is an operation of a slice rather than an accident of parameter type. | **Met.** Mutable parameter, `SVec` view, promoted FFI, sub-slice and checked sub-slice all run the same iterator; raw stays non-iterable (`compile_fail`). **G10 closed — no gaps remain.** |
| 11 | Downstream migration | ~~Migrate downstream; isolate unsafe promotion~~ **DONE 2026-08-29.** `arrow-lms` promotes at the descriptor boundary (`values`/`bytes` now yield trusted slices); three duplicated batch-descriptor constructions folded into `batch_from_descs`; two duplicated byte-descriptor constructions folded into `resolved_bytes`; four inlined bitmap promotions folded into `bitmap_bytes`. | Proves the umbrella works outside `rust-lms` and reduces repeated raw-parts plumbing. | **Met.** 391 Cranelift / 410 LLVM; sql-gen 113/113, arrow-lms 10/10; audit below. |
| 12 | Compatibility removal and documentation | ~~Remove redundant wrappers/traits, update the prelude, document the contracts~~ **DONE 2026-08-29.** Deleted `RegisterScalar` (deprecated, zero users) and the three `#[allow(deprecated)]` sites it needed, the `VarBuilder` alias, and the free `slice_get_ptr_unchecked` (now a `SliceOps` method). Stripped refactor-history and stale claims from rustdoc. | Leaves an API explainable without knowing its refactor history. | **Met.** No deprecated APIs remain; rustdoc **0 warnings** (was 11); 391 Cranelift / 410 LLVM, 32 doctests, clippy `-D warnings` + `fmt` clean. |

## Detailed milestone guidance

### Characterization before refactoring — **DONE**

`rust-lms/tests/slice_characterization.rs` holds one matrix rather than isolated
tests added during each rename. Every runtime cell runs through
`for_each_backend`.

| Origin | shared read | mut write | sub-slice | nested sub-slice | internal call/ret | FFI call/ret | iterator |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| shared parameter `SRef<Slice<T>>`     | OK | n/a | OK | OK | OK | OK / **G6a** | OK |
| mutable parameter `SRefMut<Slice<T>>` | OK | OK  | OK | OK | OK | **G6a**      | **G10** |
| raw descriptor `FatSliceType<T>`      | OK | **G6b** | OK | OK | OK | OK      | **G10** |
| descriptor field (`SliceRepr`)        | OK | OK  | OK | OK | OK | OK          | OK |

`OK` = a passing test locks the behaviour in; `n/a` = not meaningful; `Gn` = not
expressible today, closed by the plan row named below.

#### Origin independence (row 5)

`slice_characterization.rs` carries two helpers written *once* against the
capability traits, each driven by every origin that qualifies:

- `total` (bounded on `TrustedSliceType<Elem = i64>`) is fed a function
  parameter, a sub-slice of it, a witnessed descriptor field, *and* a unique
  slice — one helper across both origins and both capabilities.
- `fill` (bounded on `MutSliceType<Elem = i64>`) is fed a mutable parameter and
  a mutable sub-slice, and a `compile_fail` doctest on `SliceMutOps` proves a
  shared slice cannot reach it (the failure is "trait bounds were not
  satisfied", not a missing method).

**A generic slice helper must bind once and reborrow, not clone per use.**
Unique slice expressions are deliberately not `Clone` — that *is* the uniqueness
guarantee — so an `S: Clone` bound silently restricts a helper to shared
origins. `ctx.bind` costs one use of the expression and every later use is a
reborrow, which serves both capabilities; that is what let `total` drop its
`Clone` bound and become capability-independent.

Also folded in here: `MutField`'s `slice_len` / `slice_get_unchecked` /
`slice_set_unchecked` are gone. They restated the same validity contract three
times; one `unsafe fn as_mut_slice<E>(&mut self)` states it once and hands back
an ordinary `MutSliceType` expression carrying the whole common op surface.
(`slice_len` turned out to be dead code — nothing called it.)

#### `SVec` views as built (row 8)

The views are guards that `Deref` to a `Copy`, **lifetime-free**, **reloading**
staged expression whose `Out` is `SRef<Slice<T>>` / `SRefMut<Slice<T>>`. Because
`Out` is an ordinary trusted marker, the views inherit the entire op surface from
rows 3-6 with no new impls: `view.len()`, `view.get_or(..)`,
`view.subslice_unchecked(..)`, `view.set_unchecked(..)` are the *same* methods a
function parameter uses, reached through the deref.

The expression is composed rather than hand-lowered — it is row 6's machinery
applied to the control block:

```text
load ctrl.ptr / ctrl.len  ->  slice_from_raw_parts_mut  ->  RawSliceMut<T>
                          ->  assume_shared / assume_unique  ->  trusted slice
```

Two properties carry over from the design notes and both matter:

- **Lifetime-free**, so `ctx.bind`'s `'static` bound never reaches the guard's
  borrow (the mistake behind the erroneous S6 result).
- **Reloading**, so an expression copied out of a guard and used after a growth
  has been emitted observes the *new* buffer. `view_expression_reloads_after_growth`
  pushes past two reallocations and reads the full contents through a copy taken
  before them.

`rust-lms-std` had no `llvm` feature, so `SVec` codegen had never run on MLIR.
It now has one plus a `for_each_backend` harness, and all nine SVec tests run on
both backends.

#### `SVec` ownership as built (row 7)

`SVec<T>` is now a unique capability: neither `Copy` nor `Clone`, with `push`
and `set` taking `&mut self`. Alongside the handle sits
`StageBorrows(Rc<Cell<i64>>)` — `> 0` shared views, `-1` a unique view, `0`
unborrowed. `as_slice`/`as_mut_slice` hand out `Drop` guards; `push` asserts the
count is zero, because growth may move the buffer.

The migration was two mechanical shapes:

- `sql-gen` mints a fresh handle per output column and pushes immediately — only
  needed `let mut`.
- The `rust-lms-std` tests relied on `Copy` to use `svec` inside a `move` closure
  *and* after it. Dropping the spurious `move` fixes them: `while_loop` calls its
  body synchronously at staging time, so a borrowing closure's borrow ends when
  `while_loop` returns. Arguably better code than before.

**The tracker is per-handle**, which is the one aliasing rule stage-0 tracking
cannot enforce: two handles over the same control block carry independent
counters. That is now written into `from_raw_unchecked`'s safety contract —
reconstructing a handle *to push* is fine (what `sql-gen` does); reconstructing
one while a view from another is live is not.

`view_released_then_grow` is the payoff and the reason the type-level route was
rejected: take a view, finish with it, drop it, then grow. Under type-level
borrows that is permanently rejected (the view's borrow is pinned to the whole
staging region); under stage-0 tracking it passes, because the guard's `Drop`
runs in emission order.

#### The provenance boundary as built (row 6)

The pipeline the doc specified is now the *only* path from a descriptor to a
trusted slice:

```text
extern result / descriptor field
  -> raw staged slice          (SAFE: claims nothing)
  -> assume_shared / assume_unique   (UNSAFE: states the contract, once)
  -> ordinary TrustedSliceOps / SliceMutOps
```

The consolidation that made this work was removing the *second* promotion path.
Previously `into_slice` / `into_mut_slice` produced a **trusted** slice directly,
bypassing the boundary. Now every conversion (`into_raw_slice`,
`into_raw_slice_mut`, `MutField::as_mut_slice`) yields a **raw** descriptor, and
the trust claim is made exactly once, at `assume_*`, where the contract is
written down.

> **Correction (2026-08-29).** Those three conversions were briefly made *safe*,
> on the reasoning that a raw descriptor claims nothing and every op that
> dereferences one is itself `unsafe`. **That reasoning is wrong and the change
> was a soundness hole.** An extern declared with a `FatSlice<T>` parameter is a
> `SafeExternFn` — the derive marks any non-`unsafe` extern without reference
> parameters safe, and a `#[repr(C)]` struct is not a reference — and such an
> extern dereferences its argument. So a safe constructor completed a fully safe
> path from an arbitrary descriptor to a dereference:
>
> ```rust,ignore
> let raw = d.into_raw_slice::<i64>();   // was safe
> call_extern1(ext, raw)                 // safe; the extern dereferences
> ```
>
> The invariant is: **producing a `RawSlice` value is itself the unsafe act.**
> That is why `slice_from_raw_parts` has always been `unsafe`, and the three
> conversions are `unsafe` again. A `compile_fail` doctest on `ReprSliceOps`
> holds the line. Found by asking why `batch_from_descs` needed `unsafe`.

`assume_unique` is gated on `DataPtr = SMutPtr<T>`, so a shared `RawSlice`
cannot launder itself into a writable view — the error is a concrete type
mismatch (`expected SMutPtr<i64>, found SPtr<i64>`), not a missing method.

**What did *not* consolidate:** the three conversion traits remain three, because
they are keyed on the *addressing form* of the receiver (`SRef<R>`,
`SRefMut<R>`, `SPtr<R>`) and Rust's coherence rules reject overlapping blanket
impls over distinct `Staged::Out` types even though the three are disjoint. The
duplication the doc was actually pointing at — three different *notions of
trust* — is gone: all three now produce raw and share one promotion.

#### Downstream after migration (row 11)

**Promotion is now isolated at descriptor-construction boundaries.** Every
`assume_shared`/`assume_unique` in the workspace sits at one:
`PrimitiveArrayView::values`, `ValidityView::bytes`,
`ValidityView::bitmap_bytes`, and the two `SVec` view expressions. Nothing
promotes mid-computation.

`PrimitiveArrayView::values` is **safe** and returns a trusted slice. The
obligation is not gone, it moved: both constructors
(`ArrayBatchOps::primitive`, `FfiArrayOps::into_primitive`) are already
`unsafe fn` requiring the Arrow buffer to be represented by `M` for every
generated use — holding a `PrimitiveArrayView` *is* that proof, so re-asserting
it per read was redundant. Consumers now get the whole slice surface (`len`,
`get_or`, `get_range`, `staged_iter`, `subslice_unchecked`) where they
previously had an unsafe read per element.

**Repeated raw-parts plumbing collapsed**, three duplications in total:

| was | now |
|---|---|
| `slice_from_raw_parts::<FfiArray,…>` x3 (scan, join build, join probe) | `batch_from_descs` |
| `slice_from_raw_parts::<u8,…>` x2 (string equality, string append) | `resolved_bytes` |
| `field_mut(..).as_mut_slice::<u8>()` x4 (bitmap read/write x2) | `bitmap_bytes` |

Each carried its own near-identical SAFETY comment; each now states the contract
once. Deduplicating the batch case removed the last direct `FfiArray` reference
from `join.rs` — the descriptor layout is no longer a detail that module knows.

**`batch_from_descs` returns a trusted slice.** Its safety contract — the array
is retained for every generated use and holds exactly `ncols` entries — *is* the
`assume_shared` contract, so it promotes there rather than handing back a proof
it already has. Reading a *column* out of the batch stays `unsafe` via
`ArrayBatchOps::primitive`, but for an unrelated obligation (the element type
must match the Arrow buffer); those are two separate facts and conflating them
was an error in the first draft of this row.

**`resolved_bytes` stays raw — and the reason is the call path, not the contract.**
The *safe* `call_externN` is bounded on `IntoExternArg`, which demands the
extern's **exact** staged argument type; for a `FatSlice<u8>` parameter that is
`RawSlice<u8>`. Only `call_externN_unchecked` accepts the `UncheckedExternArg`
representation witness. So promoting `resolved_bytes` would force its call sites
onto the unchecked path for no gain.

> This also sharpens row 6's G6a claim. The witnesses added there let a slice
> *parameter* reach a `FatSlice`-declared extern **via `call_externN_unchecked`**
> — which is what the G6a tests use. The safe call path still requires an exact
> type match, by design: it is what keeps `slice_from_raw_parts` being `unsafe`
> sufficient to guard the safe extern path.

No downstream code selects an operation by slice origin; the benchmarks needed no
changes beyond the row-4 renames.

#### Iteration as built (row 10)

`SliceIter` is keyed on the **capability** (`S::Out: TrustedSliceType<Elem = T>`)
rather than on `SRef<Slice<T>>`, so a parameter, a sub-slice, a checked
sub-slice, an `SVec` view and a promoted FFI descriptor all drive one iterator.

Relaxing the bound was not enough on its own. `for_each` required `S: Clone`
(it cloned the slice for the loop condition and again per element), and a unique
slice expression deliberately is not `Clone` — so a mutable parameter would still
not have iterated. It now uses the same **bind-once-and-reborrow** idiom rows 5
and 9 needed, which as a side effect hoists the length out of the loop: one
descriptor read instead of one per iteration.

`IndexedSource`/`IndexedStagedIterator` (the `zip` path) keep `S: Clone` — the
trait takes `&self` and its supertrait requires `Clone`, so unique origins cannot
participate. Every origin `zip` is actually used with is shared, so nothing is
lost. `IndexedSource` for slice *variables* generalized to any
`R: TrustedSliceType + CopyType`, `CopyType` being exactly the shared-only
restriction the supertrait already implies.

Raw descriptors still do not iterate, by design: they are not
`TrustedSliceType`, so nothing has vouched for the memory a loop would read.
Promotion first, then iteration — a `compile_fail` doctest on `SliceIter` holds
the line.

#### Gaps the matrix surfaced

These are now the concrete exit criteria for rows 6 and 10.

- **G6a — a slice *parameter* cannot reach a `FatSlice`-declared extern.**
  `UncheckedExternArg` witnesses `FatSliceType<T> -> SRef<Slice<T>>` and
  `FatSliceMutType<T> -> SRefMut<Slice<T>>`, but neither reverse direction, even
  though both lower to the identical `(ptr, len)` argument pair. Today a kernel
  must declare its parameter as `Var<FatSliceType<T>>` to call such an extern.
  Corroboration: `ext_double_slice` in `test_extern_fn.rs` is declared but called
  by no test — the mutable half was never reachable. Row 6 must add the missing
  witnesses (and a test that actually calls a mutable-slice extern).
- **G6b — a mutable raw descriptor supports no slice operation at all,** not even
  `len`. **Half closed by row 3:** `FatSliceMutType<T>` is now
  `SliceType + RawSliceType`, so every op *node* accepts it. The remaining half is
  row 4 — the public `RawSliceOps<T>` trait is still hard-bound to
  `Staged<Out = FatSliceType<T>>`, so no call site can reach those nodes.
- **G10 — `SliceIter` is bound to `SRef<Slice<T>>`,** so neither a mutable
  parameter nor a raw descriptor can be iterated. Row 10.

Cells for `SVec` views are deliberately absent: they do not exist until rows 7/8,
so there is no current behaviour to characterize. The matrix gains that row when
they land.

### API naming pass

Prefer conventional names and make safety visible in suffixes:

| Current name | Direction |
|---|---|
| `count()` for slice length | standardize on `len()`; reserve `count()` for iterators |
| `slice_unchecked(start, end)` | `subslice_unchecked(start..end)` or `get_unchecked(start..end)` after checking inference ergonomics |
| `slice_mut_unchecked` | use the same range vocabulary with mutability expressed by the receiver capability |
| `into_ptr` / `into_mut_ptr` | use `as_ptr` / `as_mut_ptr` only when the receiver is borrowed; use `into_*` only for consuming capability conversion |
| `FatSliceType` | compatibility alias to a name that says raw/FFI descriptor if the migration remains readable |
| `AsSlice`, `AsMutSlice`, `AsRawSlice` | consolidate behind provenance-oriented constructors; retain separate internal nodes only if lowering actually differs |

Do not add a second family of convenience methods on `Var<T>`. Inherent methods
should exist only where Rust borrowing of a non-`Copy` variable needs behavior
that a blanket by-value trait cannot express.

### FFI and `arrow-lms`

`SliceRepr` and `MutSliceRepr` remain unsafe downstream extension points because
`arrow-lms` implements them for descriptor layouts. Consolidation must preserve
that external implementation capability. The plan should reduce three public
conversion traits to one representation witness plus explicit shared/mutable
promotion, not seal the witness inside `rust-lms`.

The FFI macro must continue to distinguish:

- runtime ABI carriers (`FfiSlice`, `FfiSliceMut`);
- Rust reference parameters (`&[T]`, `&mut [T]`), which are not safe C ABI;
- reference returns, whose provenance must not be invented;
- raw descriptor returns, which remain raw in staged code.

### `SVec` migration impact

The main downstream changes should be mechanical:

- bind reconstructed SQL output handles as mutable locals before `push`;
- replace direct `SVec::get`/`set` loops with borrowed views where the buffer is
  not growing;
- keep `from_raw_unchecked` only at the runtime-selected SQL type boundary;
- ensure no second handle is reconstructed while a shared or mutable view is
  retained;
- add a regression that grows first, creates a view second, and reads the
  reallocated buffer through both backends.

## Verification rules

After every implementation row:

```text
cargo test --workspace --all-targets
cargo test --workspace --features llvm
```

At phase boundaries also run:

```text
cargo test --workspace --doc
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy --workspace --all-targets --features llvm -- -D warnings
cargo fmt --all -- --check
```

Negative lifetime and capability claims require compile-fail tests. Runtime
tests alone cannot prove that aliasing or escape is impossible.

## Risks and limits

- ~~**Lifetime propagation may be invasive.**~~ **RESOLVED (spike).** 93 lines
  across three `rust-lms` files; zero downstream changes. The risk was the
  iterator family instead — see below.
- ~~**Build-time and runtime lifetimes are not automatically the same.**~~
  **RESOLVED (spike).** One `'stage` lifetime threaded
  `Ctx → FunDef → Compiler → Compiled` covers both, because `Compiled<'a, T>`'s
  existing `PhantomData<&'a T>` becomes constrained. No separate owner lifetime
  and no unsafe constructor are required (S8).
- **The iterator trait family, not `Ctx`, is where `'static` is load-bearing.**
  `iter/traits.rs` bakes `'static` into the *traits* (`fn for_each(self, ctx:
  &mut Ctx, …) where F: … + 'static`, and `IndexedSource: Clone + 'static`), and
  `SliceIter` carries explicit `'a: 'static` bounds. Relaxing them fails with
  `E0308: method not compatible with trait` until the traits themselves take a
  `'stage` parameter — roughly 90 `'static` occurrences over 11 files. Note this
  work is only needed for *stage-time-baked* slices; slice parameters do not
  need it.
- **A staged type must not carry a borrow it wants released.** `Ctx` retains
  expressions under a `'static` bound, so any type that implements `Staged` *and*
  carries a lifetime has that lifetime forced to `'static`. Keep borrows on
  guards/handles; keep the staged expressions they yield lifetime-free. This was
  the single mistake behind the erroneous S6 result.
- **FFI cannot supply provenance by layout alone.** `(ptr, len)` compatibility
  proves neither validity nor lifetime. Raw results need an unsafe witness or a
  future owned/annotated return protocol.
- ~~**Rust-like mutable reborrowing is permanently conservative.**~~ **WITHDRAWN
  2026-08-28.** This rested on the flawed S6 result. Lexical reborrow recovery
  works fine as long as the lifetime sits on the view rather than on the staged
  expression. Consuming a mutable parent to produce a sub-slice remains the rule
  for *expressions* (they are values), but a borrowed **variable** recovers
  normally via `Var::reborrow` / a dropped view.
- **Safe indexing needs failure semantics.** `get_or` and optional range access
  can land now. Exact Rust panic behavior waits for the neutral trap/runtime
  status work.
- **The project is intentionally 64-bit.** The canonical layout assumes a
  pointer and `usize` are each eight bytes. The target contract must enforce
  x86_64/aarch64 rather than letting this slice refactor imply portability to
  other widths.

## Row 12 notes

Removed: `RegisterScalar` (a `#[deprecated]` compatibility bound with no users,
which was keeping three `#[allow(deprecated)]` attributes alive), the
`VarBuilder` type alias for `Ctx`, and the free `slice_get_ptr_unchecked` —
folded into `SliceOps::get_ptr_unchecked` so every slice operation is reached the
same way.

Kept: `let_var`/`LetVar`. Its doc called it a back-compat shim, but it has 23 live
call sites and serves the expression-tree authoring style, which is a supported
style rather than a legacy one. The doc comment was the stale part, not the API.

Documentation swept for claims that had gone stale as later rows landed — the op
umbrella still said "there is no `RawSliceOps`" (row 6 added one) and described
inherent `Var<SRefMut<_>>` methods that row 4 deleted; `ScalarType` still
described the MLIR backend as future work and referenced a removed
`cranelift_type`. Refactor-history references (row numbers, gap identifiers) were
removed from `src` and left in the tests, where the characterization matrix is the
point.

## Definition of done

The refactor is complete when:

- one backend-neutral `(ptr, len)` implementation serves all slice operations;
- trusted shared and mutable slices have one origin-independent public API;
- function parameters, `SVec` borrows, promoted FFI results, and sub-slices are
  accepted by the same generic helpers;
- raw FFI values cannot perform safe memory access before an explicit validity
  and provenance boundary;
- `SVec` cannot be reallocated through safe staged code while a derived view is
  live;
- mutable slice capabilities cannot be copied or aliased through safe APIs;
- sub-slicing preserves lifetime, mutability, and raw/trusted status;
- redundant operation traits and wrapper nodes are removed or private;
- both backend suites, downstream workspace tests, compile-fail tests, rustdoc,
  formatting, and warning-denying Clippy pass.
