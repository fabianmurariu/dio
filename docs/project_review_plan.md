# Project Review Remediation Plan

Started: 2026-08-18

This plan orders the findings in `docs/project_review.md` by dependency and
blast radius. Soundness boundaries come before API cleanup because later work
needs trustworthy types, function signatures, and runtime inputs.

## Working rules

- Land one bounded fix at a time where practical.
- Add a regression, compile-fail, or adversarial test for every defect.
- Run `cargo test --workspace --all-targets` after every fix.
- Run focused tests while developing, but do not substitute them for the full
  workspace test run.
- Run Clippy at phase boundaries and record pre-existing versus new findings.
- Keep compatibility shims only when they do not preserve the unsafe behavior.

## Phase 1: Soundness containment

Goal: prevent safe callers from creating invalid JIT/FFI operations before the
larger ownership and ABI redesigns.

1. **Validate streamed schemas and surface scan failures.** Store the declared
   schema in each stream, reject every mismatched batch before producing raw
   descriptors, and propagate callback failures after the kernel returns.
   Covers PR-08. **Status: complete.**
2. **Type external function arguments.** Add an associated argument tuple to
   `ExternFn`; make each `call_externN` available only for its exact generated
   signature. Add compile-fail tests for wrong arity and same-ABI/wrong-Rust-type
   calls. Covers PR-04. **Status: complete.**
3. **Restrict representation-to-slice conversions.** Replace blanket safe
   reinterpretation with unsafe representation witnesses and prove that
   ordinary `SRef<T>` cannot become a slice. Covers PR-06. **Status: complete.**
4. **Mark the trusted trait boundary.** Seal internal layout/codegen traits and
   make supported external implementations unsafe with documented invariants.
   Reduce `CompilationContext` visibility. Covers PR-02. **Status: complete.**
5. **Tie compiled entry points to ownership.** Add borrowing `call` wrappers,
   make detached function-pointer extraction an unsafe escape hatch, and define
   an executable-memory reclamation policy. Covers PR-01. **Status: complete.**

Exit criteria: invalid external signatures and arbitrary slice reinterpretation
fail at compile time; streamed schema changes return an error; unsafe extension
points and detached entry points are explicit.

**Phase 1 verification (2026-08-18):** `cargo test --workspace --all-targets`
and `cargo test --workspace --doc` pass. Workspace Clippy with warnings denied
initially reported 17 findings outside the PR-01 diff and no new PR-01
findings. The subsequent `CodegenAction` alias reduced the current baseline to
16: two `type_complexity`, two generated `too_many_arguments`, three
`len_without_is_empty`, four `wrong_self_convention`, two `expect_fun_call`, one
`clone_on_copy`, and two `bool_assert_comparison` diagnostics. These remain
tracked for the API-consolidation phase.

## Phase 2: Pointer and ownership model

1. Replace address-as-`u64` paths with `SPtr<T>`/`SMutPtr<T>`.
   **Status: complete.** Raw pointer operations are separate from staged Rust
   references and require explicit unsafe dereference/offset/store operations.
2. Stop fabricating `'static` staged references from raw addresses.
   **Status: complete.** Opaque extern arguments and SQL scan/join batches now
   remain raw pointers or lifetime-free raw slice descriptors internally.
3. Split Arrow descriptors into typed read-only and writable forms with
   lifetimes or scoped owners. **Status: complete.** Prepared host owners retain
   source borrows; read-only and writable wire descriptors are distinct and
   their fields are private. Validity mutation also keeps `null_count` in sync,
   pulling PR-13 forward from Phase 4.
4. Connect `HostVec<T>` and `SVec<T>` through an opaque typed handle; make raw
   control blocks private and harden allocation arithmetic. **Status:
   complete.** Dynamic SQL dispatch uses the documented unsafe raw escape hatch.
5. Add RAII ownership and `MaybeUninit`-correct construction for opaque
   iterators. **Status: complete.** Raw ownership transfer and iterator-kind
   contracts are unsafe; untransferred owners drop normally.
6. Make unchecked pointer, slice, and indexed operations unsafe; make zip stop
   at the shorter input. **Status: complete.** Unequal-length zip regressions
   cover both iterator paths.

Covers PR-03, PR-07, PR-09, the ownership portion of PR-10, PR-11, PR-12,
and PR-13.

**Phase 2 verification (2026-08-18):** Every bounded fix was followed by
`cargo test --workspace --all-targets` and `cargo test --workspace --doc`; the
final runs pass. This includes all `sql-gen` grouping, join, streaming, output,
string, and property tests. Workspace Clippy with warnings denied reports only
the existing 16-diagnostic Phase 5 baseline and no new Phase 2 diagnostics.

## Phase 2A: Staged reference ownership

This phase was inserted before ABI work after the `SRefMut`/extern audit. Raw
pointer separation alone does not make staged Rust references honest if their
AST handles can still be copied or if extern metadata immediately erases them
back to raw pointers.

1. **Make mutable staged variables unique.** Make `Var<T>` `Copy` only when
   `T: CopyType`; keep `SRef` copyable and make `SRefMut` non-`Copy`. Mutable
   loads, stores, and direct slice operations borrow the root handle and store
   a crate-controlled single-use variable occurrence in the deferred AST.
   Consuming conversion to `SMutPtr` remains available, after which repeated
   use is explicitly raw-pointer code. **Status: complete.**
2. **Separate staging scope from invocation lifetime.** Replace
   `SRefMut<'static, T>` as the function-parameter convention with a parameter
   abstraction whose runtime argument is generic over each call, for example
   `RuntimeParam::Arg<'call> = &'call mut T`. Keep `Compiled::call` safe and
   prevent reference parameters or results from escaping their invocation.
   `CompiledFn` now stores a private code address plus the staged `FunTypeN`
   signature instead of a monomorphic Rust function pointer. `RuntimeParam`
   and `RuntimeResult` select fresh argument/result lifetimes at every safe
   call; the marker lifetime remains only as AST provenance. **Status:
   complete.**
3. **Preserve references in extern metadata.** Derive `&T` as a staged shared
   reference and `&mut T` as a staged unique reference rather than as `SPtr` or
   `SMutPtr`. Add reference-aware call constructors that borrow unique
   arguments simultaneously, so Rust rejects passing the same mutable handle
   twice. Raw pointer arguments cannot substitute for reference parameters on
   the safe path; forgeable descriptor hardening remains tracked by PR-10.
   Thin references now map to `SRef<Opaque<T>>` / `SRefMut<Opaque<T>>`, and
   `IntoExternArg` turns `&mut Var<SRefMut<_>>` into a controlled single-use
   occurrence. Raw-pointer/reference ABI compatibility is available only to
   the unchecked constructors. Reference returns and Rust slice-reference C
   ABIs are deliberately not classified as safe. **Status: complete.**
4. **Add scoped projections and splitting.** Make mutable field, element, and
   sub-slice projections borrow their parent capability. Provide explicit
   operations for statically disjoint fields and checked dynamic indices.
   Remove `field_addr_mut_unchecked`, `load_ref_mut_unchecked`, and
   `store_ref_unchecked` where a scoped operation can express the proof.
   `MutField` now borrows a parent for terminal field/descriptor operations;
   the derive macro generates disjoint-field witnesses for consuming
   `split_fields_mut`. Mutable element and sub-slice reference projections
   consume their parent. `get_or` and `set` provide bounds-checked scalar
   access without producing references or duplicating evaluation of their
   slice expression, while slice data-pointer access now returns honest
   raw-pointer markers. The three temporary unchecked projection APIs have
   been removed. **Status: complete.**
5. **Audit higher-level owners.** Migrate mutable options, Arrow validity,
   opaque inputs, pools, and SQL callbacks. Safe APIs must not clone a mutable
   root, manufacture a second handle with the same provenance, or hide a raw
   descriptor validity requirement. Mutable options remain consuming and
   non-`Copy`; Arrow validity uses scoped fields; pools and opaque iterators use
   safe extern reborrows. SQL, dynamic records, and `SVec` retain explicit raw
   paths after consuming an owner-backed capability. **Status: complete.**

**Phase 2A step 1 verification (2026-08-19):**
`cargo test --workspace --all-targets` and `cargo test --workspace --doc` pass.
The latter includes a compile-fail regression proving that
`Var<SRefMut<'static, T>>` cannot be duplicated; a positive test confirms that
`Var<SRef<'static, T>>` remains `Copy`. The complete `sql-gen` integration and
property suite passes after binding its consumed `&mut Inputs` parameter into
an explicit copyable raw-pointer variable.

**Phase 2A step 2 verification (2026-08-19):**
`cargo test --workspace --all-targets` and `cargo test --workspace --doc` pass,
including the complete `sql-gen` integration and property suites. The
regressions prove that one `CompiledFn` accepts fresh sequential mutable borrows
and that a staged reference result cannot escape a shorter-lived input.

**Phase 2A step 3 verification (2026-08-19):**
`cargo test --workspace --all-targets` and `cargo test --workspace --doc` pass,
including the complete `sql-gen` integration and property suites. Runtime tests
cover shared and slice reference calls, sequential mutable reborrows, pool
callbacks, and opaque iterators. Compile-fail regressions prove that one unique
staged root cannot satisfy two mutable parameters in the same extern call and
that Rust slice-reference C ABIs are excluded from safe extern calls. The
rust-lms library Clippy count remains at its existing 13-warning baseline.

**Phase 2A steps 4 and 5 verification (2026-08-19):** Each bounded projection,
splitting, checked-indexing, and raw-slice-pointer fix was followed by
`cargo test --workspace --all-targets`; every run passed, including all
`sql-gen` integration and property suites. Runtime regressions cover scoped
field mutation, two disjoint mutable fields, checked in/out-of-bounds slice
access, consuming mutable element/sub-slice projections, Arrow validity
updates, single evaluation of checked slice operands, and empty-slice raw
pointers. Compile-fail regressions reject a live parent alongside a scoped
field, duplicate field splitting, parent reuse after mutable sub-slicing, and
duplicated optional mutable references. Workspace doctests pass, and the
rust-lms library Clippy count remains at its existing 13-warning baseline.

## Phase 3: ABI correctness

1. Introduce one target-aware ABI classification and copy-lowering service.
2. Remove rounded eight-byte aggregate loads/stores and fix aligned option
   payload layout.
3. Correct unit and trampoline calling conventions.
4. Add ABI tests for partial-word, floating-point, aligned, nested, and indirect
   aggregates on every supported target.

Covers PR-05 and removes duplicated lowering described in the review.

## Phase 4: Runtime and semantic correctness

1. Finish no-unwind FFI error propagation for allocation, indexing, join,
   grouping, and output callbacks. Covers the remaining PR-10 and PR-16 work.
2. Keep validity bitmaps and `null_count` consistent. Covers PR-13. **Status:
   complete in Phase 2.**
3. Implement SQL three-valued truth tables. Covers PR-14.
4. Validate range steps and overflow-safe lengths. Covers PR-15.
5. Replace truncating row/group identifiers. Covers PR-17.
6. Bind effectful `min`/`max` operands once, return optional extrema for empty
   iterators, and remove duplicate `let_var` initialization. Covers PR-18,
   PR-19, and PR-20.

## Phase 5: API consolidation

1. Unify function/call arity machinery around one typed argument abstraction.
2. Remove stale iterator bounds, aliases, and duplicate initialization APIs.
3. Narrow `sql-gen` and backend exports.
4. Reject duplicate catalog names and preserve structured error sources.
5. Standardize consuming conversion names and collection conventions.
6. Clear Clippy with `-D warnings` and update crate-level feature documentation.

Exit criteria: the public surface has one vocabulary for each concept, the full
test suite passes, and workspace Clippy passes with warnings denied.
