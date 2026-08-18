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
   reinterpretation with sealed/unsafe representation witnesses and prove that
   ordinary `SRef<T>` cannot become a slice. Covers PR-06.
4. **Mark the trusted trait boundary.** Seal internal layout/codegen traits and
   make supported external implementations unsafe with documented invariants.
   Reduce `CompilationContext` visibility. Covers PR-02.
5. **Tie compiled entry points to ownership.** Add borrowing `call` wrappers,
   make detached function-pointer extraction an unsafe escape hatch, and define
   an executable-memory reclamation policy. Covers PR-01.

Exit criteria: invalid external signatures and arbitrary slice reinterpretation
fail at compile time; streamed schema changes return an error; unsafe extension
points and detached entry points are explicit.

## Phase 2: Pointer and ownership model

1. Replace address-as-`u64` paths with `SPtr<T>`/`SMutPtr<T>`.
2. Stop fabricating `'static` staged references from raw addresses.
3. Split Arrow descriptors into typed read-only and writable forms with
   lifetimes or scoped owners.
4. Connect `HostVec<T>` and `SVec<T>` through an opaque typed handle; make raw
   control blocks private and harden allocation arithmetic.
5. Add RAII ownership and `MaybeUninit`-correct construction for opaque
   iterators.
6. Make unchecked pointer, slice, and indexed operations unsafe; make zip stop
   at the shorter input.

Covers PR-03, PR-07, PR-09, PR-10, PR-11, and PR-12.

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
2. Keep validity bitmaps and `null_count` consistent. Covers PR-13.
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
