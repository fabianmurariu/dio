# Project Review

Reviewed: 2026-08-17

## Remediation status

Last updated: 2026-08-19

Phases 1 and 2 are complete. **Phase 2A: staged reference ownership** now blocks
Phase 3 because the generated API must represent `&mut T` as a unique
capability before reference-taking extern calls can be classified safely. The
detailed implementation order and verification record are in
`docs/project_review_plan.md`.

Phase 2A step 1 is complete: `Var<T>` is `Copy` only for `CopyType` values,
`SRef` remains copyable, and `SRefMut` is non-`Copy`. Direct mutable loads,
stores, and slice operations now borrow the unique staged handle for one AST
operation. Consuming `SRefMut` into `SMutPtr` makes the raw-pointer boundary
explicit, and repeated raw use requires binding the result to a copyable staged
pointer variable. Mutable validity views no longer clone their root reference.

Still pending in Phase 2A:

- Replace the `'static` staged-parameter marker with an invocation-lifetime
  abstraction such as `RuntimeParam::Arg<'call>`.
- Preserve `SRef`/`SRefMut` in derived extern signatures and add call
  constructors that use Rust reborrows instead of raw-pointer demotion.
- Replace temporary unchecked mutable field/reference projections with scoped
  projection and disjoint-splitting APIs.
- Audit mutable options, slices, structs, and SQL callbacks, then remove escape
  hatches that no longer have a justified caller-side proof.

Phase 2 replaced integer-address/reference fabrication with explicit staged raw
pointers, split Arrow read/write ownership, typed the normal `HostVec`/`SVec`
path, added opaque-iterator ownership, and made unchecked access unsafe. Scan,
join, grouping, string, and output code in `sql-gen` now use those raw-pointer
boundaries; its complete integration suite remains green.

Cross-cutting Phase 5 cleanup is also pending. The current workspace Clippy
baseline with warnings denied is 16 diagnostics; the `CodegenAction` alias
removed the former `Ctx::actions` type-complexity warning.

| Finding | Status | Planned phase |
| --- | --- | --- |
| PR-01: Compiled entry-point ownership and JIT memory | Fixed | Phase 1 |
| PR-02: Trusted layout and codegen traits | Fixed | Phase 1 |
| PR-03: Raw addresses, fabricated lifetimes, and copyable mutable references | Partially fixed | Phases 2 and 2A |
| PR-04: Typed external function signatures | Fixed | Phase 1 |
| PR-05: Target-correct aggregate ABI lowering | Pending | Phase 3 |
| PR-06: Unrestricted slice reinterpretation | Fixed | Phase 1 |
| PR-07: Typed, owned Arrow buffer descriptors | Fixed | Phase 2 |
| PR-08: Stream schema validation and scan failures | Fixed | Phase 1 |
| PR-09: Typed `SVec<T>` allocation ownership | Fixed | Phase 2 |
| PR-10: Forgeable raw FFI descriptors and failure paths | Partially fixed | Phases 2 and 4 |
| PR-11: Opaque iterator ownership and initialization | Fixed | Phase 2 |
| PR-12: Safe unchecked pointer, slice, and iterator operations | Fixed | Phase 2 |
| PR-13: Validity bitmap `null_count` consistency | Fixed | Phase 2 (pulled forward) |
| PR-14: SQL three-valued boolean logic | Pending | Phase 4 |
| PR-15: Range step and length validation | Pending | Phase 4 |
| PR-16: Panics at public SQL boundaries | Partially fixed | Phase 4 |
| PR-17: Truncating row and group identifiers | Pending | Phase 4 |
| PR-18: Repeated evaluation in `min` and `max` | Pending | Phase 4 |
| PR-19: Empty iterator extrema | Pending | Phase 4 |
| PR-20: Duplicate effects in `Ctx::let_var` | Pending | Phase 4 |

## Scope

This review covers the root Cargo workspace: `rust-lms`, `rust-lms-derive`,
`rust-lms-std`, `arrow-lms`, and `sql-gen`. It excludes the vendored/submodule
`sql-gen/optd` tree and the unrelated untracked `mantis/` directory.

The review focused on memory safety, generated-code safety, correctness, public
API design, duplicated mechanisms, and places where Rust ownership or typed
values can replace raw pointers and integer addresses.

## Executive summary

The project has a clear core idea and a substantial passing test suite, but its
safe public API currently exposes several operations whose soundness depends on
unstated caller invariants. The most important problem is that types such as
`StagedType`, `Staged`, `Field`, `SRef`, and `ExternFn` communicate stronger Rust
layout, lifetime, aliasing, and function-signature guarantees than the compiler
enforces. As a result, downstream safe Rust can construct JIT programs that
dereference invalid pointers, call Rust functions with the wrong argument types,
or outlive host data referenced by executable code.

The first milestone should be a soundness boundary rather than new operators:

1. Tie compiled entry points and embedded data to explicit lifetimes.
2. Make trusted layout/codegen traits sealed or `unsafe trait`s.
3. Give external functions a typed argument tuple, not only an untyped ABI list.
4. Separate staged raw pointers from staged Rust references.
5. Validate every streamed Arrow batch against the compiled schema.
6. Replace the current aggregate ABI approximation with target-correct lowering
   or Rust-generated trampolines.

## Findings

Severity meanings:

- **Critical**: safe Rust can reach undefined behavior, or a core generated-code
  contract is not defensible on supported targets.
- **High**: a concrete correctness or availability defect at a public boundary.
- **Medium**: an API or implementation issue that makes defects likely, restricts
  valid use unnecessarily, or duplicates behavior.

### PR-01: Compiled code is leaked and entry points escape ownership policy

**Severity: High**

`Compiled` owns the `JITModule` and raw code address, but `Compiled::as_fn`
returns a plain `extern "C" fn` whose type is not tied to the borrow of
`Compiled` (`rust-lms/src/func.rs:1460-1508`). Safe code can therefore write:

```rust
let function = compiler.compile(expr).unwrap().as_fn();
// The temporary Compiled owner has been dropped here.
let value = function();
```

This pattern already occurs in tests, for example
`arrow-lms/tests/validity.rs:32` and `arrow-lms/tests/validity.rs:53`. The pinned
Cranelift memory provider deliberately leaks finalized mappings when the module
is dropped, so this is not an immediate code-memory use-after-free in the
current build. It is instead an unbounded code-memory leak for workloads that
compile repeatedly. The escaping pointer also prevents `Compiled` from safely
adopting a reclaiming memory provider later and does not carry lifetimes for
host data addresses embedded in the generated code (see PR-03).

**Recommendation:** Return a `CompiledFn<'compiled, Signature>` wrapper that
borrows `Compiled` and exposes `call`, or put typed `call` methods directly on
`Compiled`. Keep extraction of a bare function pointer as an explicitly unsafe
escape hatch such as `as_fn_unchecked`. Once entry points cannot escape, give
`Compiled` an explicit reclamation policy. The same owner should retain all
data whose address was embedded into the generated function.

### PR-02: Safe extension traits form the trusted computing base

**Severity: Critical**

Several public, safe traits let downstream code assert facts that are later
trusted by raw loads, aggregate copies, ABI lowering, or `transmute`:

- `StagedType` controls the runtime Rust type, Cranelift type, size, alignment,
  copy behavior, and ABI (`rust-lms/src/types.rs:21-88`).
- `Staged` can emit arbitrary IR through a public `CompilationContext`
  (`rust-lms/src/staged.rs:33-53`, `rust-lms/src/staged.rs:142-153`).
- `Field` supplies field offsets (`rust-lms/src/struct.rs:33-45`).
- `SliceType` and `MutSliceType` claim fat-pointer representations
  (`rust-lms/src/slice.rs:295-317`).

A safe implementation can lie about any of these facts. The compiler then uses
the claim in unchecked memory operations and presents the resulting function as
safe Rust. This is the standard condition for an `unsafe trait`, not a safe
extension trait.

**Recommendation:** Seal traits that are only intended for workspace-owned
types. Make genuinely extensible layout traits `unsafe trait`s and document all
size, alignment, valid-value, ABI, and codegen invariants next to the trait.
Generated implementations must use `unsafe impl` and emit compile-time layout
checks. A stronger long-term design separates a safe expression AST from a
small, crate-private unsafe lowering layer.

### PR-03: Raw addresses are presented as Rust references with fabricated lifetimes

**Severity: Critical**

The pointer API makes unchecked operations safe to construct:

- `load`, `store`, pointer arithmetic, and unchecked array access are safe
  functions in `rust-lms/src/refer.rs:138-432`.
- `opaque_ref` and `opaque_ref_mut` convert any staged `u64` into an
  `SRef<'static, Opaque<T>>` or mutable equivalent
  (`rust-lms/src/opaque.rs:47-123`).
- `ConstPtr::from_addr` is safe (`rust-lms/src/refer.rs:465-475`), as are
  `const_opaque` and `const_opaque_mut` from raw pointers
  (`rust-lms/src/opaque.rs:125-134`).
- The lifetime on `Compiler<'a>` is not derived from those raw pointer inputs
  (`rust-lms/src/func.rs:615-623`).

This permits null, dangling, misaligned, or short-lived addresses to become
staged Rust references and ultimately arguments to safe Rust `extern "C"`
functions. `SRefMut` also flows through copyable staged variables, so the same
nominal `&mut T` can be supplied more than once in a call, violating exclusivity.

**Recommendation:** Model arbitrary addresses as `SPtr<T>`/`SMutPtr<T>` only.
Creating a staged `SRef<'a, T>` should require an actual `&'a T`; creating
`SRefMut` should require `&'a mut T` and preserve exclusive ownership. Raw
pointer dereference, offsetting, and unchecked indexing should be unsafe staging
operations. Prefer handles borrowing a resource arena over storing addresses as
`u64`. If staged handles must remain copyable for SSA construction, do not claim
that they are Rust references at the generated function boundary.

### PR-04: External calls check ABI shapes, not Rust function signatures

**Severity: Critical**

`ExternFn` has an associated return type but represents its parameters as a
runtime `Vec<AbiParam>` (`rust-lms/src/ffi.rs:466-506`). `call_extern1` and
`call_extern2` accept caller-selected staged argument types with no type-level
relationship to the registered function (`rust-lms/src/ffi.rs:572-646`). The
derive macro likewise records only ABI entries (`rust-lms-derive/src/lib.rs:519-555`).

Consequences include calling a two-argument function through a one-argument
builder, or passing an integer where the callee expects `&T` because both lower
to one integer register. A Cranelift verifier may reject some arity errors, but
it cannot prove Rust reference validity or distinguish same-shaped Rust types.

**Recommendation:** Give `ExternFn` an associated typed `Args` tuple and define
`ExternRef<extern "C" fn(A, B) -> R>` or an equivalent signature marker. Generate
call methods only for the exact signature. The macro should derive both the Rust
signature and its ABI from one source, eliminating the caller's independent
`AType`/`BType` choice. Mark raw-address or reference-taking functions unsafe and
carry that unsafety through the staged call API.

### PR-05: Aggregate ABI lowering is not target-correct and can overrun objects

**Severity: Critical**

The struct derive rounds `size_of::<T>()` up to eight-byte chunks and declares
every chunk as `I64` (`rust-lms-derive/src/lib.rs:199-228`). Function argument,
return, call, and copy paths then load or store complete eight-byte chunks
(`rust-lms/src/func_impl.rs:113-173`, `rust-lms/src/func.rs:1253-1328`).

For a 4-byte or 12-byte aggregate, the last operation accesses bytes beyond the
object. The all-`I64` model is also not the platform ABI for aggregates such as
homogeneous floating-point structs. The `>16` indirect-result rule in
`rust-lms/src/types.rs:55-65` is target-specific, while Cranelift can target more
than one ABI. `COptionType` similarly assumes eight-byte payload alignment and
copies only `size / 8` whole chunks (`rust-lms/src/option.rs:111-141`,
`rust-lms/src/option.rs:194-229`). The unit type is exposed as an `I8` result even
though a Rust/C `()` return has no result (`rust-lms/src/types.rs:322-344`).

The derive also does not prove that a field annotated `#[staged(T)]` has a Rust
layout compatible with `T::RuntimeValue`. Arrow currently uses this escape hatch
to declare pointer fields as `u64`, which hides rather than validates the ABI.
At the outermost boundary, `Compiled::run` and the getter inside `as_fn` also
transmute a System V/Windows Fastcall entry point to a Rust-ABI `fn` pointer
(`rust-lms/src/func.rs:1470-1508`). Those calling conventions are not guaranteed
to be interchangeable even when today's scalar tests happen to pass.

**Recommendation:** Centralize ABI classification per target. The most robust
option is to generate an ordinary Rust `extern "C"` trampoline for each compiled
signature and keep the JIT's internal ABI private. Otherwise implement the
platform ABI completely. Copy exact byte counts with alignment-aware operations,
never rounded-up word loads. Add compile-time assertions for derived runtime
layouts and tests for 1-, 4-, and 12-byte aggregates, `{ f64, f64 }`, aligned
payload options, unit returns, and every supported target.

### PR-06: Any staged reference can be reinterpreted as a slice descriptor

**Severity: Critical**

The blanket `ReprSliceOps` implementation lets any `SRef<R>` or `SRefMut<R>` use
`as_slice::<T>()`/`as_mut_slice::<T>()` (`rust-lms/src/slice.rs:90-173`). It then
reads the referenced bytes as a pointer-length pair without proving that `R` has
that representation. For example, a staged reference to an `i64` can be treated
as a slice header, causing an out-of-bounds descriptor read followed by arbitrary
memory access.

**Recommendation:** Replace the blanket implementation with an unsafe sealed
marker such as `SliceRepr<T>`, implemented only for audited descriptor types, or
make reinterpretation explicitly unsafe. Normal code should receive an
`SSlice<'a, T>` directly rather than reconstructing one from representation.

### PR-07: Arrow buffer descriptors erase element type, lifetime, and mutability

**Severity: Critical**

`FfiBuffer` stores a `*mut u8`, is `Copy`, has no lifetime, and its safe
`from_bytes(&[u8])`/`from_typed_slice(&[T])` constructors cast immutable storage
to mutable (`arrow-lms/src/ffi.rs:29-68`). Pointer fields in `FfiBuffer` and
`FfiArray` are staged as `u64` (`arrow-lms/src/ffi.rs:31-37`,
`arrow-lms/src/ffi.rs:120-129`). `ArrayBatchOps::primitive<M>` chooses the element
type independently of the descriptor and uses an unchecked column index
(`arrow-lms/src/array.rs:25-43`). `value_unchecked` is also a safe builder
(`arrow-lms/src/array.rs:96-107`).

Safe code can therefore describe a 12-byte `i32` buffer as three `f64` values and
read 24 bytes. It can also place a descriptor built from `&[T]` behind the mutable
validity API and write through an immutable borrow.

**Recommendation:** Split descriptors into read-only
`FfiBuffer<'a, T> { ptr: *const T, ... }` and writable descriptors constructed
only from `&'a mut [T]`. Keep raw `repr(C)` wire structs internal and unsafe;
expose validated typed column handles to staged code. Column access should use a
schema-derived typed index/witness, and unchecked access should be marked unsafe.

### PR-08: Streamed batches are not checked against the compiled schema

**Severity: Critical**

`exec_jit_stream` documents that all batches have one schema but accepts an
arbitrary iterator (`sql-gen/src/run.rs:56-74`). `ScanStream::next_batch` prepares
each batch independently and does not compare its column count or physical types
with the schema used for code generation (`sql-gen/src/scan.rs:44-56`). Generated
scan code reconstructs a descriptor slice using the compile-time column count
and then performs unchecked typed reads (`sql-gen/src/codegen/mod.rs:369-390`,
`sql-gen/src/codegen/mod.rs:433-469`).

A later batch with fewer columns or a different physical type can therefore
produce out-of-bounds or type-punned memory access through the safe public API.

**Recommendation:** Store the expected `SchemaRef` in every `ScanStream` and
validate every incoming `RecordBatch` before exposing descriptors. The iterator
should yield `Result<RecordBatch>` or store a recoverable error for the driver;
`next_batch` must not `expect` at an `extern "C"` boundary. A validated batch
token should own the descriptors for exactly the duration of the kernel call.

### PR-09: `SVec<T>` is not connected to the element type of its allocation

**Severity: Critical**

`RawVec` is a public, untyped control block (`rust-lms-std/src/svec.rs:18-31`).
`SVec<T>::new` safely accepts any `*mut RawVec`, regardless of the `HostVec<R>`
that allocated it (`rust-lms-std/src/svec.rs:149-168`). Pairing a
`HostVec<i32>` with `SVec<f64>` causes eight-byte writes into storage allocated in
four-byte elements. Public safe `svec_grow` trusts caller-provided pointer,
length, capacity, and element size (`rust-lms-std/src/svec.rs:111-129`). Growth
math is unchecked, zero-sized types are not handled, and allocation failure
crosses an `extern "C"` function as a panic/abort.

**Recommendation:** Use a private `RawVec<T>` or an opaque typed handle issued
only by `HostVec<T>`. Make direct control-block construction crate-private or
unsafe. Use `Layout::array`, checked capacity arithmetic, and
`handle_alloc_error`; either support zero-sized types explicitly or reject them
at construction. Prefer an owning/borrowing handle to passing the control block
as an integer-like raw address.

### PR-10: Safe public FFI functions trust forgeable raw descriptors

**Severity: Critical**

`FatSlice` and `FatSliceMut` expose public raw fields, so invalid descriptors can
be constructed without `unsafe` (`rust-lms/src/ffi.rs:49-110`). Public safe
`extern "C"` functions immediately turn these values or other raw inputs into
Rust slices/references:

- `pool_append` (`rust-lms/src/pool.rs:96-105`)
- `bytes_eq` and `strview_append_bytes` (`sql-gen/src/runtime.rs:36-50`)
- `group_upsert_str` (`sql-gen/src/group.rs:190-210`,
  `sql-gen/src/group.rs:325-338`)
- `svec_grow` (`rust-lms-std/src/svec.rs:111-129`)

Safe callers can supply null, dangling, misaligned, or inconsistent descriptors,
which makes the internal `from_raw_parts` or dereference undefined behavior.
Other FFI callbacks use `expect`, indexing, `assert`, or `unreachable!`; a panic
in a non-unwinding `extern "C"` callback aborts the process.

**Recommendation:** Make fields private, provide lifetime-aware safe constructors,
and mark boundary functions `unsafe extern "C" fn` with precise contracts. The
derive macro must preserve that unsafety in generated call markers. Prefer
no-unwind shims returning a status code and recording an error in runtime state;
convert that status to the workspace's normal `Result` after the JIT returns.

### PR-11: Opaque iterator helpers erase ownership and initialization state

**Severity: High**

`box_dyn_iter` and `box_dyn_exact_iter` safely convert a borrowed iterator into a
raw `*mut ()`, erasing its lifetime and leaving no RAII owner
(`rust-lms/src/iter/opaque.rs:266-281`). The pointer leaks when it is not consumed
and can dangle if generated code outlives its captures. `ReusedOpaqueIterKind` is
a safe trait whose implementor controls initialization behavior
(`rust-lms/src/iter/opaque.rs:465-507`). `emplace_iter` writes a slot through a
reference formed before the complete `OpaqueIterSlot<T>` is initialized
(`rust-lms/src/iter/opaque.rs:433-463`).

**Recommendation:** Return an RAII `OpaqueIterOwner<'a, T>` that owns the box,
exposes only a borrowed staged handle, and drops unconsumed iterators. Make raw
extraction explicit and unsafe. Use `MaybeUninit<OpaqueIterSlot<T>>` for the whole
slot plus raw field writes and an initialized-state drop guard. Seal or make the
kind trait unsafe and type its initializer arguments.

### PR-12: Safe "unchecked" iterators and slices can read out of bounds

**Severity: High**

`SliceRefOps` and `SliceMutOps` expose safe unchecked `get`/`slice` operations
(`rust-lms/src/slice.rs:568-733`). `Zip` uses the first source's length and reads
the second source unchecked; its documentation transfers the equal-length
obligation to callers (`rust-lms/src/iter/zip.rs:198-223`,
`rust-lms/src/iter/zip.rs:264-293`). That differs from `std::iter::Zip`, which
stops at the shorter iterator, and allows a safe staged program to read beyond
the second source.

**Recommendation:** Make `Zip::len` the minimum of both lengths. Provide checked
slice/index operations as the safe defaults and mark unchecked variants unsafe.
If bounds checks are intentionally omitted for performance, validation must
happen once in a typed constructor that returns a proof-carrying range/slice.

### PR-13: Validity mutation leaves `null_count` stale

**Severity: High**

`FfiValidityMut::set_null` and `set_valid` only update bitmap bits
(`arrow-lms/src/ffi_mut.rs:18-42`). `ValidityView::is_valid` skips reading the
bitmap when `null_count == 0` (`arrow-lms/src/array.rs:226-246`). After
`set_null`, the same descriptor can still report every row valid. Current tests
assert raw bitmap bytes but do not read the mutated value through `is_valid`
(`arrow-lms/tests/validity.rs:8-57`).

**Recommendation:** Update `null_count` only when a bit changes state, or remove
the fast path from mutable descriptors and reconcile the count at finalization.
Add round-trip tests for valid-to-null, null-to-valid, repeated writes, and
`is_valid` after each mutation.

### PR-14: SQL boolean `AND`/`OR` do not implement three-valued logic

**Severity: High**

All binary expressions combine operand validity with a simple validity `AND`
(`sql-gen/src/codegen/expr.rs:108-151`). SQL requires value-dependent validity:
`TRUE OR NULL` is valid `TRUE`, and `FALSE AND NULL` is valid `FALSE`. The current
implementation marks both results null. This is observable in filters, for
example `(a = 1) OR (nullable_b = 2)` incorrectly rejects a row where `a = 1`
and `b` is null.

**Recommendation:** Implement explicit SQL truth tables over `(is_valid, value)`
for `AND`, `OR`, and `NOT`. Keep null propagation for strict arithmetic and
comparison operators separate from boolean connectives. Test all nine
`TRUE`/`FALSE`/`NULL` input pairs for both binary operators.

### PR-15: Runtime ranges accept invalid steps and compute unsafe lengths

**Severity: High**

`RuntimeRangeIter` documents `step >= 1` but accepts any runtime value
(`rust-lms/src/iter/range_iter.rs:61-78`). A zero step can make iteration
non-terminating and divides by zero in indexed length calculation. Negative or
wrapping steps can also fail to progress. The indexed formula uses unsigned
subtraction and addition that underflow or overflow when `start > end` or near
the integer limit (`rust-lms/src/iter/range_iter.rs:94-125`). A bogus length then
feeds unchecked zipped access.

**Recommendation:** Use `NonZeroU64` for static steps and a checked runtime
constructor that can report an invalid step. Compute an empty length when
`start >= end`, and calculate the ceiling quotient without `span + step - 1`
overflow. Model descending ranges as a separate operation with clear semantics.

### PR-16: Several public SQL paths panic instead of returning `DataFusionError`

**Severity: High**

The SQL entry point returns `datafusion_common::Result`, but parsing and codegen
contain `panic!`, unchecked argument indexing, and `unreachable!` for unsupported
or malformed inputs (`sql-gen/src/codegen/expr.rs:40-53`,
`sql-gen/src/codegen/expr.rs:66-99`, `sql-gen/src/grouping.rs:862-880`,
`sql-gen/src/output.rs:33-40`). `sql_to_operator` also pops the first parsed
statement without rejecting additional statements (`sql-gen/src/sql.rs:53-59`).

**Recommendation:** Require exactly one statement and add a validation/typecheck
pass that converts the planner output into a supported typed IR. Validation
should return structured errors; codegen over the validated IR can then treat
unreachable states as internal bugs. FFI callbacks should never rely on panics
for ordinary bad input.

### PR-17: Row and group identifiers silently truncate to `u32`

**Severity: High**

Group indices use `state.num_records() as u32` in several insertion paths
(`sql-gen/src/group.rs:291`, `sql-gen/src/group.rs:311`,
`sql-gen/src/group.rs:331`, `sql-gen/src/group.rs:384`). Join locators also store
batch and row positions as `u32` (`sql-gen/src/join.rs:18-25`). Long-running
streams can wrap these identifiers, causing collisions or references to the
wrong row without an error.

**Recommendation:** Use `u64`/`usize` end to end, or perform a checked conversion
and return a capacity error before insertion. Put the limit in a typed index
constructor rather than repeating casts at call sites.

### PR-18: `min`/`max` can evaluate staged operands more than once

**Severity: Medium**

Numeric `min` and `max` clone their operands for the condition and then evaluate
them again in the selected arms (`rust-lms/src/num/ops.rs:679-718`). Staged
expressions are not required to be pure, so an operand containing an external
call can execute twice. The underlying select is eager as well
(`rust-lms/src/num/ops.rs:644-658`), which is surprising for users expecting
branch semantics.

**Recommendation:** Bind each operand once through `Ctx` before comparison. Name
the eager primitive explicitly (`select_eager`) and provide a control-flow form
for lazy branches. Alternatively, introduce a sealed `PureStaged` marker and
only allow unbound duplication for expressions proven pure.

### PR-19: Empty iterator `min`/`max` return ambiguous sentinels

**Severity: Medium**

Staged iterator extrema return numeric sentinel values on empty input rather
than an option (`rust-lms/src/iter/traits.rs:264-293`). A sentinel can also be a
valid data value, so callers cannot distinguish empty input from a real result.
This diverges from Rust's `Iterator::min`/`max` contract.

**Recommendation:** Return `StagedOpt<T>` or a `(seen, value)` aggregate and make
the empty case explicit at the API boundary.

### PR-20: `Ctx::let_var` can duplicate side effects

**Severity: Medium**

`Ctx::let_var` registers initialization as an action and also returns a
`LetVar` whose `Staged` implementation performs the same initialization
(`rust-lms/src/func.rs:143-167`, `rust-lms/src/staged.rs:535-610`). Its comment
calls tuple-driven double initialization harmless, but the initializer may
contain external calls or other effects. Such effects run twice.

**Recommendation:** Deprecate `let_var` in favor of the existing `Ctx::var` and
remove the staged initialization behavior. If legacy tuple sequencing must be
supported temporarily, give it a separate constructor that does not also
register an action.

## API and design improvements

These are lower-risk improvements after the soundness boundary is fixed.

1. **Make compilation internals private.** `CompilationContext` exposes the
   Cranelift builder and internal maps publicly. Keep backend mutation
   crate-private and expose a small checked lowering interface.
2. **Use one function-argument abstraction.** `fun0` through `fun8`, generated
   `call0` through `call8`, and a shorter set of external-call builders duplicate
   signature logic and have already drifted in capability. A tuple-based sealed
   `Args` trait generated once for supported arities would centralize ABI,
   runtime types, and call lowering without changing call-site ergonomics.
3. **Remove stale iterator bounds.** `SliceIter` and `Map` require
   `ConstantType` and `RuntimeValue: Default` although their current bind-based
   implementations do not use those capabilities
   (`rust-lms/src/iter/slice_iter.rs:25-44`,
   `rust-lms/src/iter/map.rs:30-59`). Removing them permits staged structs and
   pointer-like values and reduces trait noise.
4. **Choose one slice vocabulary.** `FfiSlice`/`FfiSliceMut` are aliases for
   `FatSlice`/`FatSliceMut`. Retain one public name and use migration aliases only
   behind a deprecation period.
5. **Retire compatibility names.** `VarBuilder` is only an alias for `Ctx`
   (`rust-lms/src/func.rs:593-594`), while comments and examples use both names.
   Deprecating the old term will make documentation and diagnostics consistent.
6. **Narrow `sql-gen` exports.** `CodegenCtx`, group/join state, output builders,
   and runtime modules are exposed even though `exec_jit` is the intended product
   API. Prefer `pub(crate)` for implementation modules and expose a deliberate
   extension layer only where external implementations are supported.
7. **Validate catalog construction.** `Catalog::with_table` and multi-table SQL
   setup silently overwrite duplicate names. Return a duplicate-name error to
   prevent queries from resolving to an unexpected stream.
8. **Preserve error sources.** Compilation errors currently flatten backend
   errors into strings. Use structured variants with `source()` so callers can
   distinguish verifier, module, schema, unsupported-feature, and runtime errors.
9. **Clarify consuming conversion names.** Clippy flags `as_slice`,
   `as_mut_slice`, `as_ptr`, and `as_mut_ptr` because they consume `self`.
   `into_slice`/`into_ptr`, or `reinterpret_as_*` for an unsafe conversion, better
   communicates staged-expression ownership.
10. **Add `is_empty`.** Public traits that expose `len` should also expose
    `is_empty`, following Rust collection conventions.
11. **Update feature documentation.** `sql-gen/src/lib.rs:11-13` still describes
    only scan/filter/project over primitive types, while grouping, joins, and
    strings are now implemented. Keep the crate-level contract aligned with the
    supported validation pass proposed above.

## Redundancy and maintenance observations

- Aggregate argument and return copying is repeated across function definitions,
  calls, external calls, and options. A single layout/ABI lowering service would
  remove duplicated unsafe arithmetic and prevent the paths from diverging.
- Pointer-like functionality is spread among `SRef`, `SRefMut`, `SPtr`,
  `SMutPtr`, `ConstPtr`, opaque references, fat slices, Arrow descriptors, and
  integer `u64` fields. Consolidating on typed raw pointers plus separately
  proven borrows would make the safety model reviewable.
- `let_var` and `var` are two initialization mechanisms, with the former also
  retaining legacy tuple sequencing. One action model is enough.
- External call arities and compiled function arities are generated independently.
  They should share one signature/tuple implementation and one test matrix.
- Schema facts are reconstructed independently in SQL planning, Arrow descriptor
  preparation, and staged column access. A typed validated schema object should
  be produced once and carried through all three phases.

## Recommended remediation order

1. Add compile-fail and adversarial tests demonstrating entry-point lifetime,
   wrong extern signatures, invalid slice reinterpretation, streamed schema
   changes, typed `SVec` mismatch, and Arrow mutable-alias rejection.
2. Introduce typed compiled-function and external-function signature wrappers.
   Mark existing raw entry points unsafe and deprecate them.
3. Seal or mark unsafe every trait and constructor whose implementation is
   trusted by loads, stores, layout, ABI, or reference creation.
4. Replace address-as-`u64` APIs with typed raw pointers and lifetime-carrying
   resource handles. Split immutable and mutable Arrow descriptors.
5. Add per-batch schema validation and no-unwind runtime error propagation.
6. Correct aggregate ABI lowering and add cross-target ABI tests before claiming
   general `extern "C"` compatibility.
7. Fix validity counts, SQL three-valued boolean logic, range validation, and
   identifier truncation.
8. Consolidate duplicated arity/layout code and narrow public exports.

## Verification performed

- `cargo test --workspace --all-targets` passed: 343 tests passed, 2 ignored.
  Benchmark smoke executables also completed successfully.
- `cargo clippy --workspace --all-targets -- -D warnings` did not pass. Clippy
  stopped in `rust-lms` with 15 errors, including type complexity, too many
  arguments, `len` without `is_empty`, consuming `as_*` naming, eager formatting
  inside `expect`, cloning a `Copy` value, and boolean assertions written as
  equality comparisons.

The passing tests are valuable regression coverage, but they mostly exercise
well-formed generated programs. They do not establish the unsafe contracts
identified above, and a few tests currently rely on the detached `as_fn`
entry point and Cranelift's deliberately leaked mapping behavior.
