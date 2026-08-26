//! Core staged computation traits and abstractions.
//!
//! This module defines the foundation for type-safe staged computations:
//! - `Staged`: Trait for anything that can generate runtime code
//! - `VarRef<T>`: Typed variable references (just indices, Copy-able)
//! - `Const<T>`: Typed constants (Copy-able)

use std::collections::HashMap;
use std::ops::{Deref, DerefMut};

// Entity types named only by the opaque handles' Cranelift conversions below. Cranelift's
// `Value` is aliased so the neutral AST-level `Value` (defined below) owns the bare name.
use cranelift_codegen::ir::{Block, FuncRef, SigRef, StackSlot, Value as CraneliftValue};
use cranelift_frontend::Variable;

use crate::types::{ConstantType, CopyType, FloatCmp, IntCmp, ScalarType, StagedType};

/// An opaque handle to a value produced during codegen.
///
/// Phase 0e of docs/llvm.md: the AST-facing value handle is now **opaque** — a bare
/// `u32` the AST cannot inspect. The active backend interprets it: `CraneliftBackend`
/// treats it as a Cranelift `Value` index (`as_u32`/`from_u32`, stateless — Cranelift
/// values *are* `u32` entities); an MLIR backend would use the same `u32` as an index
/// into its own `Vec<MlirValue>`. The AST never names a backend value type.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId(u32);

/// A neutral staged value — the result of [`Staged::codegen`] and the AST-level currency.
///
/// A `Value` is a value's *shape*, tracked by the neutral layer itself so the AST can stop
/// smuggling multi-value shapes past a single-value contract (docs/post_llvm_plan.md, A′):
///
/// - [`Scalar`](Value::Scalar) — one backend SSA leaf ([`ValueId`]); every non-slice value.
/// - [`Fat`](Value::Fat) — a slice as **two** SSA leaves `(data ptr, len)` directly, instead
///   of a pointer to a `{ptr,len}` pair threaded through a side-channel. On Cranelift these
///   are two registers; on LLVM two SSA values. Materialized as `{ptr,len}` when canonical
///   memory storage is required (for example at an ABI boundary or inside an aggregate).
///
/// Backends still speak only in leaves — a `Value` is composed/destructured entirely in the
/// neutral layer.
#[derive(Clone, Copy, Debug)]
pub enum Value {
    Scalar(ValueId),
    Fat { ptr: ValueId, len: ValueId },
}

impl Value {
    /// Wrap a single backend leaf as a scalar value.
    pub(crate) fn scalar(leaf: ValueId) -> Self {
        Value::Scalar(leaf)
    }

    /// A fat (slice) value: data pointer + length, each a backend leaf.
    pub(crate) fn fat(ptr: ValueId, len: ValueId) -> Self {
        Value::Fat { ptr, len }
    }

    /// The backend leaf of a scalar value. Panics on a `Fat` value to report a neutral-IR
    /// shape violation at the first scalar-only boundary.
    pub(crate) fn leaf(self) -> ValueId {
        match self {
            Value::Scalar(v) => v,
            Value::Fat { .. } => panic!("expected a scalar value, found a fat (slice) value"),
        }
    }

    /// The `(data ptr, len)` leaves of a fat (slice) value. Panics on a `Scalar`.
    pub(crate) fn parts(self) -> (ValueId, ValueId) {
        match self {
            Value::Fat { ptr, len } => (ptr, len),
            Value::Scalar(_) => panic!("expected a fat (slice) value, found a scalar value"),
        }
    }
}

impl From<ValueId> for Value {
    fn from(leaf: ValueId) -> Self {
        Value::Scalar(leaf)
    }
}

/// Opaque handle to a basic block during codegen (see [`ValueId`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BlockHandle(u32);

/// Opaque handle to a mutable variable during codegen (see [`ValueId`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarHandle(u32);

/// Opaque handle to a stack allocation during codegen (see [`ValueId`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StackSlotId(u32);

/// Opaque handle to a function reference usable by `call`/`func_addr` in the function
/// currently being built (see [`ValueId`]). Cranelift interprets it as a `FuncRef` index;
/// an MLIR backend as an index into its own symbol table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FuncRefId(u32);

/// Opaque handle to a signature imported for `call_indirect` (see [`ValueId`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SigRefId(u32);

/// A backend-neutral function signature: parameter types and an optional single result.
/// Backends lower it to their own signature (Cranelift `Signature`, MLIR function type).
pub struct SigSpec {
    pub params: Vec<ScalarType>,
    pub ret: Option<ScalarType>,
}

// Backend/driver-only conversions between the opaque handles and Cranelift entities.
// These stay here (not in the `cranelift` module) because they touch the handles' private
// `.0` field; that privacy is exactly what stops an AST module from fabricating a handle.
impl ValueId {
    pub(crate) fn from_cranelift(v: CraneliftValue) -> Self {
        Self(v.as_u32())
    }
    pub(crate) fn cranelift(self) -> CraneliftValue {
        CraneliftValue::from_u32(self.0)
    }
}
impl BlockHandle {
    pub(crate) fn from_cranelift(b: Block) -> Self {
        Self(b.as_u32())
    }
    pub(crate) fn cranelift(self) -> Block {
        Block::from_u32(self.0)
    }
}
impl VarHandle {
    pub(crate) fn from_cranelift(v: Variable) -> Self {
        Self(v.as_u32())
    }
    pub(crate) fn cranelift(self) -> Variable {
        Variable::from_u32(self.0)
    }
}
impl StackSlotId {
    pub(crate) fn from_cranelift(s: StackSlot) -> Self {
        Self(s.as_u32())
    }
    pub(crate) fn cranelift(self) -> StackSlot {
        StackSlot::from_u32(self.0)
    }
}
impl FuncRefId {
    pub(crate) fn from_cranelift(f: FuncRef) -> Self {
        Self(f.as_u32())
    }
    pub(crate) fn cranelift(self) -> FuncRef {
        FuncRef::from_u32(self.0)
    }
}
impl SigRefId {
    pub(crate) fn from_cranelift(s: SigRef) -> Self {
        Self(s.as_u32())
    }
    pub(crate) fn cranelift(self) -> SigRef {
        SigRef::from_u32(self.0)
    }
}

// The LLVM/MLIR backend interprets these handles as indices into its own arenas
// (docs/llvm.md §9) — the same "u32 the active backend interprets" contract the
// Cranelift path uses, just a different encoding: `ValueId` → value arena slot,
// `BlockHandle` → body-block index, `VarHandle` → variable (alloca) index.
#[cfg(feature = "llvm")]
impl ValueId {
    pub(crate) fn from_u32(index: u32) -> Self {
        Self(index)
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }
}
#[cfg(feature = "llvm")]
impl BlockHandle {
    pub(crate) fn from_u32(index: u32) -> Self {
        Self(index)
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }
}
#[cfg(feature = "llvm")]
impl VarHandle {
    pub(crate) fn from_u32(index: u32) -> Self {
        Self(index)
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }
}
#[cfg(feature = "llvm")]
impl StackSlotId {
    pub(crate) fn from_u32(index: u32) -> Self {
        Self(index)
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }
}
#[cfg(feature = "llvm")]
impl FuncRefId {
    pub(crate) fn from_u32(index: u32) -> Self {
        Self(index)
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }
}
#[cfg(feature = "llvm")]
impl SigRefId {
    pub(crate) fn from_u32(index: u32) -> Self {
        Self(index)
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.0
    }
}

// =============================================================================
// Compilation Context
// =============================================================================

/// Backend variables holding one neutral [`Value`]. Keeping the shape in the binding
/// prevents scalar and fat variables from being split across unrelated maps.
#[derive(Clone, Copy)]
pub(crate) enum VarValue {
    Scalar(VarHandle),
    Fat { ptr: VarHandle, len: VarHandle },
}

/// Context provided during code generation.
///
/// This type is exposed so downstream implementations of [`Staged`] can name
/// the codegen method's argument. Its backend state is intentionally private;
/// downstream expressions should lower by composing existing staged nodes.
///
/// ```compile_fail
/// use rust_lms::prelude::CompilationContext;
///
/// fn cannot_mutate_the_backend(ctx: &mut CompilationContext<'_>) {
///     let _ = &mut ctx.backend;
/// }
/// ```
pub struct CompilationContext<'c> {
    /// The active IR backend (Cranelift or LLVM/MLIR).
    /// `CompilationContext` derefs to this, so `ctx.<op>()` routes to the backend.
    pub(crate) backend: &'c mut dyn Backend,
    /// Mapping from staged variable IDs to shape-aware backend variable bindings.
    pub(crate) variables: &'c mut HashMap<usize, VarValue>,
    /// Cached unit value (iconst.i8 0) - avoids creating duplicate dead values
    pub(crate) unit_value: Option<ValueId>,
    /// Stack of enclosing loops' exit blocks. The innermost loop's exit is on
    /// top; `break_loop` jumps to it. Pushed/popped by the loop codegen.
    pub(crate) loop_exit_stack: Vec<BlockHandle>,
}

/// The IR-emission backend: the single interface a code generator implements.
///
/// Phase 0d of docs/llvm.md. Cranelift is the only impl today (`CraneliftBackend`);
/// an LLVM/MLIR impl slots in behind the same trait. `CompilationContext` owns the
/// codegen bookkeeping (var/slice maps, loop-exit stack) and derefs to a `dyn Backend`
/// for the primitive ops. Object-safe (all methods take concrete handles).
///
/// `pub` + `#[doc(hidden)]` only because `CompilationContext` (a public type) derefs
/// to `dyn Backend`; it is an internal, unstable contract, not a public API.
///
/// **The trait is now fully backend-neutral: no Cranelift type appears in any method
/// signature.** Phase 0f neutralized the value/control-flow surface (constants, arithmetic,
/// comparison via [`IntCmp`]/[`FloatCmp`], casts, memory, pointers, blocks, variables, stack
/// slots); the calls & signatures cluster followed once the MLIR backend gave a second
/// call/symbol model to design against — functions and signatures are named by the opaque
/// [`FuncRefId`]/[`SigRefId`] handles and the neutral [`SigSpec`], and `declare_func`/
/// `declare_extern_func` resolve *our* ids (the backend owns the id→native-function map).
/// What remains Cranelift-specific is the `compile()` driver and the not-yet-abstracted
/// `Module`/`Executable` lifecycle — not this trait.
#[doc(hidden)]
pub trait Backend {
    // constants
    fn iconst(&mut self, ty: ScalarType, imm: i64) -> ValueId;
    fn f64const(&mut self, v: f64) -> ValueId;
    fn f32const(&mut self, v: f32) -> ValueId;
    // integer arithmetic
    fn iadd(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn isub(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn imul(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn sdiv(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn udiv(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn srem(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn urem(&mut self, a: ValueId, b: ValueId) -> ValueId;
    // float arithmetic
    fn fadd(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn fsub(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn fmul(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn fdiv(&mut self, a: ValueId, b: ValueId) -> ValueId;
    // bitwise / shift
    fn band(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn bor(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn bxor(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn ishl(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn sshr(&mut self, a: ValueId, b: ValueId) -> ValueId;
    fn ushr(&mut self, a: ValueId, b: ValueId) -> ValueId;
    // compare / select
    fn icmp(&mut self, cc: IntCmp, a: ValueId, b: ValueId) -> ValueId;
    fn icmp_imm(&mut self, cc: IntCmp, a: ValueId, imm: i64) -> ValueId;
    fn fcmp(&mut self, cc: FloatCmp, a: ValueId, b: ValueId) -> ValueId;
    fn select(&mut self, cond: ValueId, a: ValueId, b: ValueId) -> ValueId;
    // casts
    fn sextend(&mut self, to: ScalarType, v: ValueId) -> ValueId;
    fn uextend(&mut self, to: ScalarType, v: ValueId) -> ValueId;
    fn ireduce(&mut self, to: ScalarType, v: ValueId) -> ValueId;
    fn fcvt_from_sint(&mut self, to: ScalarType, v: ValueId) -> ValueId;
    fn fcvt_from_uint(&mut self, to: ScalarType, v: ValueId) -> ValueId;
    fn bitcast(&mut self, to: ScalarType, v: ValueId) -> ValueId;
    // memory
    fn load(&mut self, ty: ScalarType, ptr: ValueId, offset: i32) -> ValueId;
    fn store(&mut self, val: ValueId, ptr: ValueId, offset: i32);
    fn stack_addr(&mut self, slot: StackSlotId, offset: i32) -> ValueId;
    /// Allocate an explicit stack slot of `size` bytes with alignment `1 << align_shift`.
    fn alloc_stack_slot(&mut self, size: u32, align_shift: u8) -> StackSlotId;
    fn copy_nonoverlapping(&mut self, dst: ValueId, src: ValueId, size: usize, align: usize);
    // pointers (semantic; §8b)
    fn ptr_offset_bytes(&mut self, ptr: ValueId, offset: ValueId) -> ValueId;
    fn ptr_offset_const(&mut self, ptr: ValueId, bytes: i64) -> ValueId;
    fn addr_to_ptr(&mut self, addr: ValueId) -> ValueId;
    // blocks & control flow
    fn create_block(&mut self) -> BlockHandle;
    fn append_block_param(&mut self, block: BlockHandle, ty: ScalarType) -> ValueId;
    fn block_param(&mut self, block: BlockHandle, idx: usize) -> ValueId;
    fn switch_to_block(&mut self, block: BlockHandle);
    fn seal_block(&mut self, block: BlockHandle);
    fn jump(&mut self, target: BlockHandle, args: &[ValueId]);
    fn brif(
        &mut self,
        cond: ValueId,
        then_block: BlockHandle,
        then_args: &[ValueId],
        else_block: BlockHandle,
        else_args: &[ValueId],
    );
    // variables
    fn declare_var(&mut self, ty: ScalarType) -> VarHandle;
    fn def_var(&mut self, var: VarHandle, val: ValueId);
    fn use_var(&mut self, var: VarHandle) -> ValueId;
    // calls & signatures
    fn call(&mut self, func: FuncRefId, args: &[ValueId]) -> Option<ValueId>;
    fn call_indirect(
        &mut self,
        sig: SigRefId,
        callee: ValueId,
        args: &[ValueId],
    ) -> Option<ValueId>;
    fn func_addr(&mut self, func: FuncRefId) -> ValueId;
    fn import_signature(&mut self, sig: &SigSpec) -> SigRefId;
    /// Get a callable reference to an internal (JIT-defined) function by our function id.
    fn declare_func(&mut self, func_id: usize) -> FuncRefId;
    /// Get a callable reference to a registered extern function by our extern id.
    fn declare_extern_func(&mut self, extern_id: usize) -> FuncRefId;
}

// `CompilationContext` derefs to its backend so `ctx.<op>()` routes there with no
// per-op delegators. Its own inherent methods (below) and fields take precedence.
impl<'c> Deref for CompilationContext<'c> {
    type Target = dyn Backend + 'c;
    fn deref(&self) -> &Self::Target {
        &*self.backend
    }
}
impl<'c> DerefMut for CompilationContext<'c> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.backend
    }
}

impl<'c> CompilationContext<'c> {
    /// Get or create the cached unit value (iconst.i8 0).
    ///
    /// This avoids creating duplicate dead values when sequencing side-effecting
    /// operations like `Assign` and `InitVar`.
    pub(crate) fn get_unit_value(&mut self) -> ValueId {
        if let Some(val) = self.unit_value {
            val
        } else {
            let val = self.iconst(ScalarType::I8, 0);
            self.unit_value = Some(val);
            val
        }
    }

    /// Resolve the data pointer (`*T`) of a slice operand.
    ///
    /// A slice's Staged value is a [`Value::Fat`] — the `(data ptr, len)` pair as two SSA
    /// leaves. These helpers ([`Self::slice_data_ptr`] / [`Self::slice_len`] /
    /// [`Self::slice_parts`]) are the single place slice ops go through; each just picks a
    /// half of the fat value, so there is no side-channel and no per-op memory access.
    pub(crate) fn slice_data_ptr(&mut self, slice: &impl Staged) -> ValueId {
        slice.codegen(self).parts().0
    }

    /// Resolve the length (`usize`) of a slice operand — the second half of its fat value.
    pub(crate) fn slice_len(&mut self, slice: &impl Staged) -> ValueId {
        slice.codegen(self).parts().1
    }

    /// Resolve both parts of a slice, evaluating the slice expression only once.
    pub(crate) fn slice_parts(&mut self, slice: &impl Staged) -> (ValueId, ValueId) {
        slice.codegen(self).parts()
    }

    /// Add the block parameters required to carry one staged value through a merge block.
    pub(crate) fn append_value_block_params<T: StagedType>(&mut self, block: BlockHandle) {
        if T::is_fat_pointer() {
            self.append_block_param(block, ScalarType::Ptr);
            self.append_block_param(block, ScalarType::I64);
        } else {
            self.append_block_param(block, T::scalar_type());
        }
    }

    /// Jump to `target`, passing every SSA leaf in `value` as a block argument.
    pub(crate) fn jump_value(&mut self, target: BlockHandle, value: Value) {
        match value {
            Value::Scalar(leaf) => self.jump(target, &[leaf]),
            Value::Fat { ptr, len } => self.jump(target, &[ptr, len]),
        }
    }

    /// Reconstruct a staged value from the parameters of a merge block.
    pub(crate) fn block_value<T: StagedType>(&mut self, block: BlockHandle) -> Value {
        if T::is_fat_pointer() {
            Value::fat(self.block_param(block, 0), self.block_param(block, 1))
        } else {
            Value::scalar(self.block_param(block, 0))
        }
    }

    /// Resolve a variable to its shape-aware neutral [`Value`]. The stored shape is checked
    /// against `T`, so a missing `StagedType::is_fat_pointer` implementation fails at the
    /// variable boundary instead of reaching an unrelated scalar operation.
    pub(crate) fn resolve_var<T: StagedType>(&mut self, id: usize) -> Value {
        let binding = *self
            .variables
            .get(&id)
            .unwrap_or_else(|| panic!("staged variable {id} is not defined"));

        match binding {
            VarValue::Scalar(var) if !T::is_fat_pointer() => Value::scalar(self.use_var(var)),
            VarValue::Fat { ptr, len } if T::is_fat_pointer() => {
                let ptr = self.use_var(ptr);
                let len = self.use_var(len);
                Value::fat(ptr, len)
            }
            VarValue::Scalar(_) => panic!(
                "staged variable {id} has scalar storage, but {} declares a fat value",
                std::any::type_name::<T>()
            ),
            VarValue::Fat { .. } => panic!(
                "staged variable {id} has fat storage, but {} declares a scalar value",
                std::any::type_name::<T>()
            ),
        }
    }

    /// Bind `value` to variable `id`, retaining its neutral shape. `reuse` reuses an
    /// existing binding for loop-carried assignments. The value shape and `T` must agree.
    pub(crate) fn assign_var<T: StagedType>(&mut self, id: usize, value: Value, reuse: bool) {
        let existing = reuse.then(|| self.variables.get(&id).copied()).flatten();

        match value {
            Value::Scalar(leaf) if !T::is_fat_pointer() => {
                let var = match existing {
                    Some(VarValue::Scalar(var)) => var,
                    Some(VarValue::Fat { .. }) => {
                        panic!("cannot assign a scalar value to fat staged variable {id}")
                    }
                    None => self.declare_var(T::scalar_type()),
                };
                self.def_var(var, leaf);
                self.variables.insert(id, VarValue::Scalar(var));
            }
            Value::Fat { ptr, len } if T::is_fat_pointer() => {
                let (ptr_var, len_var) = match existing {
                    Some(VarValue::Fat { ptr, len }) => (ptr, len),
                    Some(VarValue::Scalar(_)) => {
                        panic!("cannot assign a fat value to scalar staged variable {id}")
                    }
                    None => (
                        self.declare_var(ScalarType::Ptr),
                        self.declare_var(ScalarType::I64),
                    ),
                };
                self.def_var(ptr_var, ptr);
                self.def_var(len_var, len);
                self.variables.insert(
                    id,
                    VarValue::Fat {
                        ptr: ptr_var,
                        len: len_var,
                    },
                );
            }
            Value::Scalar(_) => panic!(
                "{} declares a fat value but codegen produced a scalar",
                std::any::type_name::<T>()
            ),
            Value::Fat { .. } => panic!(
                "{} declares a scalar value but codegen produced a fat value",
                std::any::type_name::<T>()
            ),
        }
    }

    /// Load a fat (slice) value from an in-memory `{ptr, len}` descriptor at `base`
    /// (`ptr` at offset 0 as `Ptr`, `len` at offset 8 as `I64`). Used when reinterpreting a
    /// pointer to such a descriptor (an FFI array field, a param's incoming pair) as a slice —
    /// the memory→fat boundary, mirror of [`Self::materialize_value`].
    pub(crate) fn load_fat(&mut self, base: ValueId) -> Value {
        let ptr = self.load(ScalarType::Ptr, base, 0);
        let len = self.load(ScalarType::I64, base, 8);
        Value::fat(ptr, len)
    }

    /// Load a staged value from its canonical in-memory representation.
    pub(crate) fn load_value<T: StagedType>(&mut self, base: ValueId) -> Value {
        if T::is_fat_pointer() {
            self.load_fat(base)
        } else if T::is_copy_struct() {
            Value::scalar(base)
        } else if T::size_of() == 0 {
            Value::scalar(self.get_unit_value())
        } else {
            Value::scalar(self.load(T::scalar_type(), base, 0))
        }
    }

    /// Store a staged value in its canonical in-memory representation.
    pub(crate) fn store_value<T: StagedType>(&mut self, base: ValueId, value: Value) {
        if T::is_fat_pointer() {
            let (ptr, len) = value.parts();
            self.store(ptr, base, 0);
            self.store(len, base, 8);
        } else if T::is_copy_struct() {
            self.copy_nonoverlapping(base, value.leaf(), T::size_of(), T::align_of());
        } else if T::size_of() != 0 {
            self.store(value.leaf(), base, 0);
        }
    }

    /// Materialize a [`Value`] to a single ABI leaf (a storage pointer for aggregates).
    /// Scalars pass through; a [`Value::Fat`] slice is written to a fresh 16-byte
    /// `{ptr, len}` stack slot and its pointer returned — the storage-pointer ABI form used
    /// when a slice crosses a call/extern boundary. Other canonical memory locations use
    /// [`Self::store_value`] directly.
    pub(crate) fn materialize_value(&mut self, value: Value) -> ValueId {
        match value {
            Value::Scalar(leaf) => leaf,
            Value::Fat { ptr, len } => {
                let slot = self.alloc_stack_slot(16, 3);
                let slot_ptr = self.stack_addr(slot, 0);
                self.store(ptr, slot_ptr, 0);
                self.store(len, slot_ptr, 8);
                slot_ptr
            }
        }
    }
}

// =============================================================================
// Core Trait: Staged
// =============================================================================

/// Anything that represents a staged computation.
///
/// Types implementing this trait can generate Cranelift IR code that produces
/// a value of type `Self::Out` at runtime.
///
/// # Safety
///
/// [`Self::codegen`] must return a value whose IR type and runtime encoding
/// exactly match `Self::Out`. Any emitted memory access, call, or control flow
/// must uphold the contracts of the staged operands it consumes. An implementation
/// whose `Out` is not [`CopyType`]
/// must not offer a safe `Copy` or `Clone` implementation that duplicates the
/// staged value's ownership capability.
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// struct UntrustedExpression;
///
/// impl Staged for UntrustedExpression {
///     type Out = i64;
///
///     fn codegen(
///         &self,
///         _ctx: &mut CompilationContext<'_>,
///     ) -> ValueId {
///         unimplemented!()
///     }
/// }
/// ```
pub unsafe trait Staged {
    /// The output type this staged computation produces
    type Out: StagedType;

    /// Generate Cranelift IR code for this computation
    fn codegen(&self, ctx: &mut CompilationContext) -> Value;
}

// =============================================================================
// VarRef<T> - Typed staged variable handle
// =============================================================================

/// A typed handle to a staged variable.
///
/// The handle is `Copy` only when the staged value has copy semantics. This is
/// significant for staged mutable references: duplicating their variable ID
/// would duplicate the exclusive capability represented by `&mut T`.
///
/// # Example
/// ```ignore
/// let x: VarRef<i64> = compiler.var();
/// let expr = add(x, x);  // x used twice - no problem, it's Copy!
/// ```
///
/// Mutable staged references are unique capabilities and cannot be copied:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn duplicate(reference: Var<SRefMut<'_, i64>>) {
///     let first = reference;
///     let second = reference;
///     let _ = (first, second);
/// }
/// ```
pub struct Var<T: StagedType> {
    pub(crate) id: usize,
    _phantom: std::marker::PhantomData<T>,
}

/// An owned, single-use occurrence of a staged variable.
///
/// This is produced internally after an API has borrowed a non-`Copy` `Var`.
/// Keeping it distinct from `Var` prevents callers from duplicating an
/// exclusive staged handle while still allowing the deferred AST node to own
/// the variable ID it will lower later.
#[doc(hidden)]
pub struct VarUse<T: StagedType> {
    id: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: CopyType> Clone for Var<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T: CopyType> Copy for Var<T> {}

impl<T: StagedType> Var<T> {
    /// Create a new variable reference with the given ID
    pub(crate) fn new(id: usize) -> Self {
        Var {
            id,
            _phantom: std::marker::PhantomData,
        }
    }

    pub(crate) fn use_once(&self) -> VarUse<T> {
        VarUse {
            id: self.id,
            _phantom: std::marker::PhantomData,
        }
    }
}

unsafe impl<T: StagedType> Staged for Var<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        ctx.resolve_var::<T>(self.id)
    }
}

unsafe impl<T: StagedType> Staged for VarUse<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        ctx.resolve_var::<T>(self.id)
    }
}

impl<T: StagedType> std::fmt::Debug for Var<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VarRef({})", self.id)
    }
}

// =============================================================================
// Const<T> - Typed constants
// =============================================================================

/// A compile-time constant that will be embedded in generated code.
///
/// `Const<T>` is Copy when `T::RuntimeValue` is Copy.
///
/// # Example
/// ```ignore
/// let five = Const::<i64>::new(5);
/// let ten = Const::<i64>::new(10);
/// ```
#[derive(Clone)]
pub struct Const<T: ConstantType> {
    value: T::RuntimeValue,
}

impl<T: ConstantType> Const<T> {
    /// Create a new constant value
    pub fn new(value: T::RuntimeValue) -> Self {
        Const { value }
    }
}

// Conditionally implement Copy when T and T::RuntimeValue are Copy
impl<T: ConstantType + Copy> Copy for Const<T> where T::RuntimeValue: Copy {}

unsafe impl<T: ConstantType> Staged for Const<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        T::codegen_constant(&self.value, ctx)
    }
}

// =============================================================================
// From implementations for ergonomic constant creation
// =============================================================================

impl From<i64> for Const<i64> {
    fn from(value: i64) -> Self {
        Const::new(value)
    }
}

impl From<u64> for Const<u64> {
    fn from(value: u64) -> Self {
        Const::new(value)
    }
}

impl From<i8> for Const<i8> {
    fn from(value: i8) -> Self {
        Const::new(value)
    }
}

impl From<u8> for Const<u8> {
    fn from(value: u8) -> Self {
        Const::new(value)
    }
}

impl From<i16> for Const<i16> {
    fn from(value: i16) -> Self {
        Const::new(value)
    }
}

impl From<u16> for Const<u16> {
    fn from(value: u16) -> Self {
        Const::new(value)
    }
}

impl From<i32> for Const<i32> {
    fn from(value: i32) -> Self {
        Const::new(value)
    }
}

impl From<u32> for Const<u32> {
    fn from(value: u32) -> Self {
        Const::new(value)
    }
}

impl From<f32> for Const<f32> {
    fn from(value: f32) -> Self {
        Const::new(value)
    }
}

impl From<f64> for Const<f64> {
    fn from(value: f64) -> Self {
        Const::new(value)
    }
}

impl From<bool> for Const<bool> {
    fn from(value: bool) -> Self {
        Const::new(value)
    }
}

impl From<()> for Const<()> {
    fn from(value: ()) -> Self {
        Const::new(value)
    }
}

// =============================================================================
// Boxing support: Enable dynamic dispatch when needed
// =============================================================================

/// Extension trait to enable boxing any Staged value for dynamic dispatch.
pub trait BoxableStaged: Staged {
    /// Box this staged value for dynamic dispatch
    fn boxed(&self) -> Box<dyn Staged<Out = Self::Out>>
    where
        Self: Clone + 'static,
        Self::Out: 'static,
    {
        Box::new(self.clone())
    }
}

// Blanket implementation: all Staged types can be boxed
impl<T: Staged> BoxableStaged for T {}

// =============================================================================
// IntoStaged trait for ergonomic constant creation
// =============================================================================

/// Trait for values that can be converted into staged expressions.
///
/// This trait enables ergonomic APIs like `assign(var, 42i64)` instead of
/// `assign(var, Const::<i64>::new(42))`.
pub trait IntoStaged<T: StagedType> {
    /// The staged type this converts to
    type Staged: Staged<Out = T>;

    /// Convert into a staged expression
    fn into_staged(self) -> Self::Staged;
}

// Implement IntoStaged for primitives
impl IntoStaged<i64> for i64 {
    type Staged = Const<i64>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<u64> for u64 {
    type Staged = Const<u64>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<i8> for i8 {
    type Staged = Const<i8>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<u8> for u8 {
    type Staged = Const<u8>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<i16> for i16 {
    type Staged = Const<i16>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<u16> for u16 {
    type Staged = Const<u16>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<i32> for i32 {
    type Staged = Const<i32>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<u32> for u32 {
    type Staged = Const<u32>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<f32> for f32 {
    type Staged = Const<f32>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<f64> for f64 {
    type Staged = Const<f64>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<bool> for bool {
    type Staged = Const<bool>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

impl IntoStaged<()> for () {
    type Staged = Const<()>;
    fn into_staged(self) -> Self::Staged {
        Const::new(self)
    }
}

// Blanket impl for anything that's already Staged
impl<T, S> IntoStaged<T> for S
where
    T: StagedType,
    S: Staged<Out = T>,
{
    type Staged = S;
    fn into_staged(self) -> Self::Staged {
        self
    }
}

// =============================================================================
// Assign<V, EXPR> - Variable assignment (side effect, returns unit)
// =============================================================================

/// Assignment expression: assigns a value to a variable.
///
/// This is a side-effecting operation that returns `()`.
/// Use with tuples to chain multiple assignments or continue with other expressions.
///
/// # Example
/// ```ignore
/// let x = compiler.var::<i64>();
/// let expr = (assign(x, 5i64), x);  // assigns 5 to x, returns x
/// ```
#[derive(Clone)]
pub struct Assign<V, EXPR> {
    var: V,
    expr: EXPR,
}

unsafe impl<T, EXPR> Staged for Assign<Var<T>, EXPR>
where
    T: StagedType,
    EXPR: Staged<Out = T>,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate the value and bind it to the variable (fat-aware: a slice reassignment
        // updates its `(ptr, len)` register pair). `reuse` keeps a stable variable across
        // repeated assignments (e.g. in a loop).
        let value = self.expr.codegen(ctx);
        ctx.assign_var::<T>(self.var.id, value, true);

        // Return cached unit value
        Value::scalar(ctx.get_unit_value())
    }
}

/// Create an assignment expression
///
/// Accepts any value that implements `IntoStaged<T>`.
/// This allows ergonomic usage like `assign(var, 42i64)` instead of
/// `assign(var, Const::<i64>::new(42))`.
pub fn assign<T, E>(var: Var<T>, expr: E) -> Assign<Var<T>, E::Staged>
where
    T: StagedType,
    E: IntoStaged<T>,
{
    Assign {
        var,
        expr: expr.into_staged(),
    }
}

/// Create a unit constant
pub fn unit() -> Const<()> {
    Const::new(())
}

// =============================================================================
// InitVar<T, EXPR> - Variable initialization wrapper
// =============================================================================

/// A variable with its initialization expression.
///
/// This type combines a variable reference with its initialization, providing
/// an ergonomic API that doesn't require manual tuple unpacking.
///
/// When used in a tuple for sequencing, it performs the initialization.
/// When used in operations (add, assign, etc.), it derefs to the underlying Var.
///
/// # Example
/// ```ignore
/// let i = compiler.let_var(0u64);  // Returns InitVar<u64, Const<u64>>
/// let expr = (i, add(*i, 5i64));   // i initializes, *i gives Var<u64>
/// ```
pub struct LetVar<T: StagedType, EXPR> {
    var: Var<T>,
    init: EXPR,
}

impl<T: StagedType, EXPR> LetVar<T, EXPR> {
    /// Create a new initialized variable wrapper
    pub(crate) fn new(var: Var<T>, init: EXPR) -> Self {
        LetVar { var, init }
    }

    /// Get the underlying variable reference
    pub fn var(&self) -> Var<T>
    where
        T: CopyType,
    {
        self.var
    }
}

impl<T: CopyType, EXPR: Clone> Clone for LetVar<T, EXPR> {
    fn clone(&self) -> Self {
        LetVar {
            var: self.var,
            init: self.init.clone(),
        }
    }
}

// InitVar is Copy when EXPR is Copy (like Const<T>)
impl<T: CopyType, EXPR: Copy> Copy for LetVar<T, EXPR> {}

// Deref to allow transparent access to the underlying Var
impl<T: StagedType, EXPR> std::ops::Deref for LetVar<T, EXPR> {
    type Target = Var<T>;

    fn deref(&self) -> &Self::Target {
        &self.var
    }
}

// When InitVar is staged, it performs the initialization
unsafe impl<T, EXPR> Staged for LetVar<T, EXPR>
where
    T: StagedType,
    EXPR: Staged<Out = T>,
{
    type Out = ();

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Generate code for the initialization value
        let value = self.init.codegen(ctx);

        ctx.assign_var::<T>(self.var.id, value, true);

        // Return cached unit value
        Value::scalar(ctx.get_unit_value())
    }
}

// Allow implicit conversion from InitVar to Var for convenience
impl<T: StagedType, EXPR> From<LetVar<T, EXPR>> for Var<T> {
    fn from(init_var: LetVar<T, EXPR>) -> Var<T> {
        init_var.var
    }
}
