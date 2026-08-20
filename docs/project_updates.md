# Project Updates — review of the `joins → issues` remediation

Written: 2026-08-20
Reviewer scope: the delta `git diff joins..issues` (HEAD), read against
`docs/project_review.md` and `docs/project_review_plan.md`.

This is my take on the changes that have landed since the external review, and on
the fixes the two planning docs propose. It is a second opinion, not a restatement
of the plan: where the docs and the code agree I say so briefly, and I spend the
words on what I verified independently and on the risks I think are still live.

## What this branch actually is

`joins` is the merge-base. Its tip (`154a793 external review of safety and
correctness`) is where `project_review.md` was authored. The `issues` branch is
`joins` **plus ten remediation commits** (`7369857 … fc4ca11`) that implement the
review's plan:

```
7369857 fixes to soundness              0724b3d moving closer to safer API
9e9de5c fixes to slices                 c24a12f nicer cleaner API, less unsafe
a2cd67f document and make Staged unsafe  20eb985 finalized the mut API
4c610d9 callX with correct lifetimes     fc4ca11 ABI across all targets + GH action
662e225 mark things that are unsafe
11427f4 SRef/Var/SRefMut better
```

It is a large, coherent delta: ~6,700 insertions / ~2,800 deletions across 65
files — the whole rust-lms core, `arrow-lms`, `rust-lms-std`, `sql-gen`, two new
test files (`test_abi.rs`, `test_func.rs`), a `build.rs` target allowlist, and a
CI workflow. Phases 1, 2, 2A and 3 of the plan are here; Phases 4 and 5 are not,
and the files they touch (`codegen/expr.rs`, `iter/range_iter.rs`, `grouping.rs`,
`sql.rs`) are untouched in this range — the "pending" labels are honest.

## Verification I ran

- **Tests:** `cargo test --workspace --all-targets` → **340 passed, 0 failed, 2
  ignored** across 67 binaries, on `aarch64-apple-darwin`. Green.
- **Clippy:** `cargo clippy --workspace --all-targets -- -D warnings` → **still
  fails** (~13 rust-lms lib lints: `wrong_self_convention`, `type_complexity`,
  `too_many_arguments`, `len_without_is_empty`, `expect_fun_call`,
  `bool_assert_comparison`). Every one is a Phase-5 cosmetic/naming lint — exactly
  the baseline the plan records as deferred. Nothing new, nothing load-bearing.
- **Diff audit:** four focused passes over the code (Phase 1 + ABI; pointer/
  ownership; the data layer + sql-gen; and a pending-work reality check), each
  reading the actual unified diff and current sources rather than the docs.

## Assessment by phase

### Phase 1 — soundness containment · **delivered**

The trusted-computing-base is now actually trusted-by-the-compiler, not by
convention:

- **PR-01** `Compiled::as_fn` returns `CompiledFn<'compiled, F>` with a *private*
  code pointer and a `PhantomData<&'compiled JITModule>` — the entry point cannot
  outlive the module. `Compiled` gained a real `Drop` that frees the JIT memory,
  so the "unbounded code-memory leak" the review called out is closed, not just
  the escape. `as_fn_unchecked` is the `unsafe` hatch. Pinned by `compile_fail`
  doctests.
- **PR-02** `StagedType`, `Staged`, `CopyType`, `ConstantType`, `RuntimeParam`,
  `RuntimeResult` are all `unsafe trait`; `CompilationContext` and its fields
  dropped to `pub(crate)`; `DirectValue` is sealed. Crucially the derive now emits
  a **forced** compile-time layout check — `LAYOUT_VALID` asserts `size_of`/
  `align_of` of each field against its `#[staged(T)]` marker, and it is *reached*
  (evaluated) through the `OFFSET` const, plus a `RuntimeValue = FieldTy`
  where-bound on non-erased fields. A lying marker fails to compile. This is the
  single most important structural change in the branch.
- **PR-04** `ExternFn` carries a sealed typed `Args` tuple; `call_externN` is
  type-checked for arity *and* per-slot Rust type; a `SafeExternFn` marker gates
  the safe path and `_unchecked` variants carry unsafe/pointer/slice ABIs. The
  derive single-sources the Rust signature and the ABI. Wrong-arity, same-ABI/
  wrong-type, unsafe-callback, raw-ptr-for-`&T`, and `&mut`-aliasing are all
  `compile_fail`-tested.
- **PR-06** slice reinterpretation is gated behind an `unsafe` `SliceRepr<T>`
  witness implemented only for the two audited Arrow descriptors; `as_slice` is
  `unsafe`. An arbitrary `SRef<i64>` can no longer become a slice header.
- **PR-08** each `ScanStream` stores the declared `SchemaRef` and validates every
  batch (full schema equality, stronger than the "count + physical type" the
  review asked for) before producing descriptors; errors travel through an
  `Inputs::error` channel surfaced after the kernel returns, so there is no
  `expect`/panic at the `extern "C"` boundary.

### Phase 2 / 2A — pointer and ownership model · **delivered**

I could not construct a safe-Rust path that forges a reference from an integer,
duplicates a staged `&mut`, or reads a slice element out of the checked API
without writing `unsafe`. Concretely:

- **PR-03** `Var<T>` is `Copy` only for `T: CopyType`; `SRefMut` is move-only;
  raw `load`/`store`/`ptr_offset`/`array_index` are `unsafe`. The old
  `opaque_ref`/`const_opaque` integer→`SRef<'static>` fabricators are **deleted**
  (opaque.rs went from ~150 lines to 29), and `ConstPtr::from_addr` is now
  `from_addr_unchecked`. `Num` is sealed.
- **PR-07 / PR-09 / PR-13** (data layer) Arrow descriptors split into read-only
  `*const` and writable `*mut` types, writable constructible only from `&mut [T]`
  behind an owner, wire fields made private, pointer fields typed as `SPtr<u8>`
  not `u64`. `SVec` gets a typed `HostVecHandle` issued only by `HostVec<T>`,
  `RawVec` private, `Layout::array` + checked capacity, ZSTs rejected, and
  allocation failure `abort`s instead of unwinding across FFI. Validity mutation
  keeps `null_count` truthful (with a dedup-guarded test).
- **PR-11 / PR-12** opaque iterators return RAII owners that drop unconsumed
  handles (with `MaybeUninit`-correct slot init and sealed/`unsafe` kind traits);
  unchecked slice/index ops are `unsafe`, checked `get_or`/`set` are the safe
  default, and `Zip::len` is now the *minimum* of both sources — matching
  `std::iter::Zip` and closing the OOB read.
- **Phase 2A projections** the strongest piece: `field_mut` yields a scoped
  `MutField` token exposing only terminal ops; `split_fields_mut` consumes the
  parent and returns two non-`Copy` projections gated on a derive-generated
  `DisjointField` witness, so **asking for the same field twice is a compile
  error**; mutable element/sub-slice projections consume their parent; `get_or`/
  `set` evaluate the slice expression exactly once (enforced in `slice_parts`,
  one place); slice data-pointer access returns honest `SPtr`/`SMutPtr`; and the
  three temporary `*_unchecked` projection APIs are gone. Each is pinned by a
  `compile_fail` doctest.

### Phase 3 — ABI correctness · **delivered (the substantive win)**

This is the change I'd most want a second set of eyes on, and it holds up. The
old approach — round `size_of` up to 8-byte chunks, declare each `I64`, a `>16`
indirect rule, and a `cfg!(windows)` SysV/Fastcall fork — is **entirely removed**
(`num_abi_values`, `abi_types`, `should_pass_by_pointer`, `StructInfo` all grep
clean). In its place:

> Every JIT boundary is **pointer args (one per logical param) + one caller-owned
> output pointer + a void return**, using the target ISA's `default_call_conv()`.
> No float or aggregate ever crosses the boundary in a register, so per-target
> classification is *impossible to get wrong*. All real platform-ABI work is
> pushed into Rust-compiled `extern "C"` thunks (`#[extern_fn]`, opaque-iterator
> callbacks) that own the true ABI.

Aggregate copies use exact size + alignment (`emit_small_memory_copy`, not rounded
word loads); `COption` uses its real `align_up(8, align_of::<T>())` payload offset;
unit `()` writes no result. **The transmute the review flagged as "not guaranteed
interchangeable" is now sound by construction** — every surviving `transmute` in
`func.rs` targets `unsafe extern "C" fn(*const u8…, *mut u8)`, a shape on which
Rust `extern "C"` and Cranelift's default convention provably agree on all six
triples. `build.rs` hard-fails off the six-target allowlist.

## My take on the proposed fixes

**The plan's ordering is right and the execution matches it.** Soundness boundary
before API cleanup was the correct call, and — unusually — the "Remediation
status" claims in `project_review.md` are *not* overstated. Every PR I checked
that is marked Fixed is actually fixed in code, with a test behind it; the items
marked Pending are genuinely untouched. That alignment between the docs and the
tree is itself a good signal about how this work was run.

Two design choices deserve explicit endorsement, because they went beyond
patching the symptom:

1. **The by-pointer ABI is the right architecture, not a workaround.** Collapsing
   the boundary and relocating classification into Rust thunks removes the entire
   bug class (by-value struct / `COption` transmutes that only happened to work on
   a SysV dev box) instead of trying to reimplement three platform ABIs in the
   lowering code. This is the "generate a Rust `extern "C"` trampoline" option the
   review floated as *most robust*, and it was taken.
2. **Forced compile-time layout checks make `unsafe trait StagedType` honest.** An
   `unsafe trait` whose invariants are only prose is a liability; here the derive
   proves the layout claim at compile time and refuses to build a lying marker.
   That is what turns PR-02 from a label into a guarantee.

### Residual risks I'd track (my additions, not in the docs)

> **Update (2026-08-20): the concrete hardening items below are now done** on the
> `issues` branch — see "Residual-risk hardening — applied" at the end of this
> document. The list is kept here as the rationale; each item is annotated with its
> outcome.

None of these are fatal; all sit behind `unsafe` or a fail-safe compile error. But
they are the places I'd expect a future regression to hide:

- **ABI test coverage trails the design's sharpest edge.** `test_abi.rs` exercises
  12-byte partial words, `{f64,f64}`, over-aligned and nested aggregates, and unit
  through both internal and extern paths — but there is **no end-to-end compiled
  round-trip of an over-aligned `COption`** (Some/None/match through the JIT),
  which is the one place the payload-offset arithmetic actually feeds codegen, and
  **no small mixed int/float aggregate** (`{u64, f64}` ≤16B) — the classic
  split-class trap. The by-pointer ABI makes these moot *today*; they are exactly
  the cases where an accidental reintroduction of word-rounding or a hard-coded `8`
  would silently pass. I'd add all three before the "general `extern "C"`
  compatibility" claim is retired. **Outcome: done** — `test_abi.rs` now covers
  1-byte (`Byte1`), 4-byte (`Word4`), and `{u64,f64}` (`MixedSmall`) through both
  the internal and extern-thunk paths, plus an over-aligned `COption<Aligned16>`
  (payload at offset 16) driven Some and None through the JIT.
- **`payload_offset` is computed in two places.** `func.rs`'s opaque-iterator loop
  re-derives `align_up(8, align)` inline rather than calling
  `COptionType::payload_offset()` (which is private). They agree now; this is
  precisely the "slice/layout math lives in exactly one place" invariant the
  project sets for itself. Make the method `pub(crate)` and call it. **Outcome:
  done** — `payload_offset()` is `pub(crate)` with a "single source of truth" doc,
  and both opaque-iterator loop sites in `func.rs` call it instead of re-deriving
  `align_up(8, align)`.
- **An undocumented 64-bit interlock.** The derive's allowlist that lets
  `usize`/`isize`/`*T` erase to `u64`/`i64` skips the `RuntimeValue` where-bound
  and is only correct because `build.rs` forbids 32-bit targets. It fails *safe*
  (a size mismatch becomes a `LAYOUT_VALID` compile error), but nothing in the
  code says the two are linked. One comment at the allowlist would close it.
  **Outcome: done** — a "64-bit interlock" doc block on `is_supported_erased_field`
  now states the 8-byte assumption, points at `build.rs` as its enforcer, and
  explains the `LAYOUT_VALID` fail-safe.
- **`SliceRepr` is `unsafe` but not sealed.** *(Corrected — my original take was
  wrong.)* I claimed sealing "would cost nothing." It would not: the only
  implementors are `arrow-lms`'s `FfiBuffer`/`FfiBufferMut`, and **`arrow-lms` is a
  separate crate depending on `rust-lms`.** Rust's sealed-trait pattern confines
  impls to the defining crate, so sealing `SliceRepr` would break `arrow-lms` and
  the intended extension model (a data-layer crate defines its own `#[repr(C)]`
  descriptor and witnesses its layout). Keeping it an open `unsafe trait` is the
  *correct* design here; the safety boundary is the `unsafe impl` obligation plus
  the `unsafe fn as_slice` and the `compile_fail` proof that safe code cannot
  reinterpret an arbitrary `SRef<R>`. **Outcome:** left open by design; documented
  the contract and an explicit "do not seal — arrow-lms is cross-crate" warning on
  the trait so a future maintainer doesn't regress it.
- **`ptr_cast`/`ptr_cast_mut` retype a pointer's pointee in *safe* code.** No IR is
  emitted and every deref is `unsafe`, so this is consistent with "a pointer is
  just its typed address" — but the typed-pointer guarantee is then redeemed only
  at the unsafe deref, not by a witness at the cast. Worth a doc note so nobody
  reads the safe cast as a proof. **Outcome: done** — a "Why this is safe despite
  reinterpreting the pointee" section on `PtrCast` now spells out that the cast
  emits nothing and the type obligation is redeemed at the (unsafe) deref.
- **`is_copy_struct()` is overloaded** to mean "indirect aggregate representation"
  and returns `true` for `SRefMut<Slice<T>>`, which is emphatically *not* Copy.
  Documented, but a naming trap for the next maintainer.
- **`MemFlags::trusted()` + `non_overlapping = true`** are used at every copy on
  the assumption that in/out storage is disjoint and aligned. It holds by
  construction (typed stack slots, fresh `MaybeUninit` outputs) but is not
  asserted; a future caller that reuses one buffer as both input and output would
  violate it silently.
- **sql-gen's honest compromises.** Two are worth naming: (a) Arrow descriptors
  still let the caller choose the element type `M` (`primitive::<M>`), recovered to
  safety by the SQL layer's schema validation + `dispatch_prim!` rather than by the
  descriptor type itself — end-to-end safe, but the library type alone is not; and
  (b) the scan *read* path deliberately moved off lifetime-checked `SRef` borrows
  onto raw lifetime-free pointers guarded by documented "owner retains the batch"
  contracts. Both are defensible given the lifetime-free `#[repr(C)]` wire-struct
  goal, but they move weight from the borrow checker onto prose.

### The one claim not yet demonstrated

PR-05 is verified **locally on `aarch64-apple-darwin` only**. The six-target CI
matrix exists and is correct in shape, but its first remote run is still pending,
and two runner labels it depends on (`macos-15-intel`, `windows-11-arm`) are
newish/limited-availability; there is no toolchain pin or build cache. Until that
matrix goes green, "target-correct on all six triples" is a well-argued claim, not
a demonstrated one. It is the single biggest open verification item, and the docs
do flag it honestly.

## The pending work (Phases 4–5), and how I'd prioritise it

All confirmed genuinely deferred. My ranking differs slightly from the review's
severities:

1. **PR-16 — no-unwind FFI error propagation (elevate this).** ~62
   `panic!`/`unreachable!`/`unwrap`/`expect` sites remain across `sql-gen`, many
   inside `extern "C"` callbacks (`group.rs`, `codegen/grouping.rs`, `output.rs`,
   `codegen/strings.rs`). A panic across a non-unwinding `extern "C"` boundary is
   an abort at best and UB-adjacent, and it is reachable from safe public SQL
   input. The review lists it High/partially-fixed; I'd make it the *first* Phase-4
   item — it is closer to the soundness boundary than to "runtime correctness."
2. **PR-14 — three-valued boolean logic.** Silent wrong query results
   (`TRUE OR NULL` etc.). Pure correctness, user-visible, self-contained.
3. **PR-17 — `u32` row/group identifiers.** Real truncation
   (`Locator { rb_pos: u32, row: u32 }`, `num_records() as u32`) but only bites
   past ~4B rows/groups; a checked conversion returning a capacity error is enough
   for now.
4. **PR-15 / PR-18 / PR-19 / PR-20** — range-step validation, `min`/`max` double
   evaluation of effectful operands, empty-iterator extrema sentinels, and
   `let_var` duplicate initialization. Local, Medium, do together.
5. **Phase 5** — clippy `-D warnings` (currently ~13 rust-lms lib lints), export
   narrowing, dedup of arity/layout machinery, structured error sources. None
   block correctness; the clippy gate can't be turned on in CI until this lands.

## Bottom line

This branch does what the review asked, and it does it at the level the review
demanded — types and `unsafe` boundaries, not comments. The ABI rewrite is the
real prize and is architecturally correct; the reference/ownership model is now
enforced by the compiler and pinned by ~20 `compile_fail` doctests; the data layer
and `sql-gen` keep the project's "type the pointers" principle with no surviving
`u64`-as-pointer. The gaps are narrow and mostly *test-coverage and
one-place-for-layout* hygiene, plus the still-pending remote CI run that would turn
the ABI claim from argued to demonstrated. I'd (a) land the three missing ABI
round-trip tests and de-duplicate `payload_offset` while the design is fresh, (b)
get the six-target matrix green, and (c) open Phase 4 with the FFI no-unwind work
rather than the boolean logic.

## Residual-risk hardening — applied (2026-08-20)

Acting on the list above, the concrete, bounded items are done on `issues`. These
are hygiene/coverage fixes that tighten the branch before Phase 4 — none change the
runtime behavior of correct programs.

| Item | Change | Location |
| --- | --- | --- |
| De-duplicate `payload_offset` | Made `COptionType::payload_offset()` `pub(crate)` with a "single source of truth" doc; both opaque-iterator loop sites now call it instead of re-deriving `align_up(8, align)` inline. | `rust-lms/src/option.rs`, `rust-lms/src/func.rs` (2 sites) |
| Missing ABI round-trips | Added `Byte1` (1B), `Word4` (4B), and `MixedSmall` (`{u64,f64}`, the split-class case) through both internal and extern-thunk paths; added an end-to-end `COption<Aligned16>` test driving Some (store+load at offset 16) and None through the JIT. | `rust-lms/tests/test_abi.rs` |
| Document 64-bit interlock | Added a "64-bit interlock" doc block explaining the 8-byte assumption, naming `build.rs` as its enforcer, and the `LAYOUT_VALID` fail-safe. | `rust-lms-derive/src/lib.rs` (`is_supported_erased_field`) |
| `ptr_cast` safety note | Added a "why this is safe despite reinterpreting the pointee" section: the cast emits nothing; the type obligation is redeemed at the (unsafe) deref, not the cast. | `rust-lms/src/refer.rs` (`PtrCast`) |
| `SliceRepr` sealing | **Reversed my recommendation** after finding the only implementors are in `arrow-lms` (a separate crate) — sealing would break it. Left the trait an open `unsafe trait` (the correct design) and documented the contract + a "do not seal, arrow-lms is cross-crate" warning. | `rust-lms/src/slice.rs` (`SliceRepr`) |

**Verification:** `cargo test --workspace --all-targets` → 341 passed, 0 failed, 2
ignored (the new `COption` test is the +1). `cargo test -p rust-lms --doc` → 23
passed. `cargo clippy -p rust-lms --lib` unchanged at the pre-existing 13-lint
Phase-5 baseline (no new lint). Run natively on `aarch64-apple-darwin`.

**Not addressed here** (deliberately, larger than "hardening"): the `is_copy_struct()`
naming overload and the unasserted `MemFlags::trusted()`/`non_overlapping` copy
invariants are left for the ABI/Phase-5 cleanup, and the six-target CI matrix run is
external. The next substantive step remains **Phase 4, PR-16 first** (no-unwind FFI).
