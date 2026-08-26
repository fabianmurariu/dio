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

/// An opaque, typed handle to a scalar value produced during codegen.
///
/// The active backend interprets `index`: Cranelift stores its `Value` entity index and
/// MLIR stores an index into its own value arena. `ty` remains backend-neutral and is the
/// source of truth for validating the neutral operation stream. In particular, it preserves
/// `Bool` versus `I8` and `Ptr` versus `I64`, distinctions Cranelift's native types erase.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ValueId {
    index: u32,
    ty: ScalarType,
}

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
        expect_type(ptr, ScalarType::Ptr, "fat value data pointer");
        expect_type(len, ScalarType::I64, "fat value length");
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
            Value::Fat { ptr, len } => {
                expect_type(ptr, ScalarType::Ptr, "fat value data pointer");
                expect_type(len, ScalarType::I64, "fat value length");
                (ptr, len)
            }
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

/// Opaque, typed handle to a mutable variable during codegen (see [`ValueId`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct VarHandle {
    index: u32,
    ty: ScalarType,
}

/// Opaque handle to a stack allocation during codegen (see [`ValueId`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct StackSlotId(u32);

/// Opaque handle to a function reference usable by `call`/`func_addr` in the function
/// currently being built (see [`ValueId`]). Cranelift interprets it as a `FuncRef` index;
/// an MLIR backend as an index into its own symbol table.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FuncRefId {
    index: u32,
    ret: Option<ScalarType>,
}

/// Opaque handle to a signature imported for `call_indirect` (see [`ValueId`]).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct SigRefId {
    index: u32,
    ret: Option<ScalarType>,
}

/// A backend-neutral function signature: parameter types and an optional single result.
/// Backends lower it to their own signature (Cranelift `Signature`, MLIR function type).
#[derive(Clone, Debug)]
pub struct SigSpec {
    pub params: Vec<ScalarType>,
    pub ret: Option<ScalarType>,
}

// Backend/driver-only conversions between the opaque handles and Cranelift entities.
// These stay here (not in the `cranelift` module) because they touch the handles' private
// `.0` field; that privacy is exactly what stops an AST module from fabricating a handle.
impl ValueId {
    pub fn scalar_type(self) -> ScalarType {
        self.ty
    }

    pub(crate) fn with_type(self, ty: ScalarType) -> Self {
        Self {
            index: self.index,
            ty,
        }
    }

    pub(crate) fn from_cranelift(v: CraneliftValue, ty: ScalarType) -> Self {
        Self {
            index: v.as_u32(),
            ty,
        }
    }
    pub(crate) fn cranelift(self) -> CraneliftValue {
        CraneliftValue::from_u32(self.index)
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
    pub(crate) fn from_cranelift(v: Variable, ty: ScalarType) -> Self {
        Self {
            index: v.as_u32(),
            ty,
        }
    }
    pub(crate) fn cranelift(self) -> Variable {
        Variable::from_u32(self.index)
    }
    pub(crate) fn scalar_type(self) -> ScalarType {
        self.ty
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
    pub(crate) fn from_cranelift(f: FuncRef, ret: Option<ScalarType>) -> Self {
        Self {
            index: f.as_u32(),
            ret,
        }
    }
    pub(crate) fn cranelift(self) -> FuncRef {
        FuncRef::from_u32(self.index)
    }
    pub(crate) fn return_type(self) -> Option<ScalarType> {
        self.ret
    }
}
impl SigRefId {
    pub(crate) fn from_cranelift(s: SigRef, ret: Option<ScalarType>) -> Self {
        Self {
            index: s.as_u32(),
            ret,
        }
    }
    pub(crate) fn cranelift(self) -> SigRef {
        SigRef::from_u32(self.index)
    }
    pub(crate) fn return_type(self) -> Option<ScalarType> {
        self.ret
    }
}

// The LLVM/MLIR backend interprets each handle's index through its own arenas
// (docs/llvm.md §9). `ValueId` and `VarHandle` keep their neutral types alongside
// that backend-specific index; call handles also retain their optional result type.
#[cfg(feature = "llvm")]
impl ValueId {
    pub(crate) fn from_u32(index: u32, ty: ScalarType) -> Self {
        Self { index, ty }
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.index
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
    pub(crate) fn from_u32(index: u32, ty: ScalarType) -> Self {
        Self { index, ty }
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.index
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
    pub(crate) fn from_u32(index: u32, ret: Option<ScalarType>) -> Self {
        Self { index, ret }
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.index
    }
}
#[cfg(feature = "llvm")]
impl SigRefId {
    pub(crate) fn from_u32(index: u32, ret: Option<ScalarType>) -> Self {
        Self { index, ret }
    }
    pub(crate) fn as_u32(self) -> u32 {
        self.index
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
    /// The active IR backend (Cranelift or LLVM/MLIR). Checked inherent operations route
    /// through this field; `Deref` remains for type-free lifecycle operations.
    pub(crate) backend: &'c mut dyn Backend,
    /// Mapping from staged variable IDs to shape-aware backend variable bindings.
    pub(crate) variables: &'c mut HashMap<usize, VarValue>,
    /// Cached unit value (iconst.i8 0) - avoids creating duplicate dead values
    pub(crate) unit_value: Option<ValueId>,
    /// Neutral parameter types for blocks created through this context.
    pub(crate) block_params: HashMap<BlockHandle, Vec<ScalarType>>,
    /// Stack of enclosing loops' exit blocks. The innermost loop's exit is on
    /// top; `break_loop` jumps to it. Pushed/popped by the loop codegen.
    pub(crate) loop_exit_stack: Vec<BlockHandle>,
}

/// The IR-emission backend: the single interface a code generator implements.
///
/// Cranelift and LLVM/MLIR both implement this scalar lowering contract. The checked
/// inherent methods on `CompilationContext` validate typed leaves before delegating here;
/// the context also owns value-shape, block-signature, variable, and loop bookkeeping.
/// Object-safe (all methods take concrete handles).
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
    fn null_ptr(&mut self) -> ValueId;
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
    fn block_param(&mut self, block: BlockHandle, idx: usize, ty: ScalarType) -> ValueId;
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

fn expect_type(value: ValueId, expected: ScalarType, operation: &str) -> ValueId {
    assert_eq!(
        value.scalar_type(),
        expected,
        "{operation}: expected {expected:?}, found {:?}",
        value.scalar_type()
    );
    value
}

fn same_type(a: ValueId, b: ValueId, operation: &str) -> ScalarType {
    assert_eq!(
        a.scalar_type(),
        b.scalar_type(),
        "{operation}: operands must have the same type, found {:?} and {:?}",
        a.scalar_type(),
        b.scalar_type()
    );
    a.scalar_type()
}

fn integer_binary(a: ValueId, b: ValueId, operation: &str) -> ScalarType {
    let ty = same_type(a, b, operation);
    assert!(
        ty.is_integer(),
        "{operation}: expected integer operands, found {ty:?}"
    );
    ty
}

fn float_binary(a: ValueId, b: ValueId, operation: &str) -> ScalarType {
    let ty = same_type(a, b, operation);
    assert!(
        ty.is_float(),
        "{operation}: expected floating-point operands, found {ty:?}"
    );
    ty
}

fn bitwise_binary(a: ValueId, b: ValueId, operation: &str) -> ScalarType {
    let ty = same_type(a, b, operation);
    assert!(
        ty.is_integer() || ty == ScalarType::Bool,
        "{operation}: expected integer or boolean operands, found {ty:?}"
    );
    ty
}

pub(crate) fn expect_arguments(operation: &str, args: &[ValueId], params: &[ScalarType]) {
    assert_eq!(
        args.len(),
        params.len(),
        "{operation}: expected {} arguments, found {}",
        params.len(),
        args.len()
    );
    for (index, (&arg, &expected)) in args.iter().zip(params).enumerate() {
        expect_type(arg, expected, &format!("{operation} argument {index}"));
    }
}

macro_rules! checked_binary_op {
    ($name:ident, $check:ident) => {
        #[doc(hidden)]
        pub fn $name(&mut self, a: ValueId, b: ValueId) -> ValueId {
            let ty = $check(a, b, stringify!($name));
            let result = self.backend.$name(a, b);
            expect_type(result, ty, stringify!($name))
        }
    };
}

impl<'c> CompilationContext<'c> {
    // These inherent methods are the checked neutral operation boundary. They shadow the
    // backend methods exposed through `Deref`, validate neutral types once, then delegate.
    #[doc(hidden)]
    pub fn iconst(&mut self, ty: ScalarType, imm: i64) -> ValueId {
        assert!(
            ty.is_integer() || ty == ScalarType::Bool,
            "iconst: expected an integer or Bool result type, found {ty:?}"
        );
        if ty == ScalarType::Bool {
            assert!(matches!(imm, 0 | 1), "iconst: Bool constant must be 0 or 1");
        }
        let result = self.backend.iconst(ty, imm);
        expect_type(result, ty, "iconst")
    }

    #[doc(hidden)]
    pub fn null_ptr(&mut self) -> ValueId {
        let result = self.backend.null_ptr();
        expect_type(result, ScalarType::Ptr, "null_ptr")
    }

    #[doc(hidden)]
    pub fn f64const(&mut self, value: f64) -> ValueId {
        let result = self.backend.f64const(value);
        expect_type(result, ScalarType::F64, "f64const")
    }

    #[doc(hidden)]
    pub fn f32const(&mut self, value: f32) -> ValueId {
        let result = self.backend.f32const(value);
        expect_type(result, ScalarType::F32, "f32const")
    }

    checked_binary_op!(iadd, integer_binary);
    checked_binary_op!(isub, integer_binary);
    checked_binary_op!(imul, integer_binary);
    checked_binary_op!(sdiv, integer_binary);
    checked_binary_op!(udiv, integer_binary);
    checked_binary_op!(srem, integer_binary);
    checked_binary_op!(urem, integer_binary);

    checked_binary_op!(fadd, float_binary);
    checked_binary_op!(fsub, float_binary);
    checked_binary_op!(fmul, float_binary);
    checked_binary_op!(fdiv, float_binary);

    checked_binary_op!(band, bitwise_binary);
    checked_binary_op!(bor, bitwise_binary);
    checked_binary_op!(bxor, bitwise_binary);
    checked_binary_op!(ishl, bitwise_binary);
    checked_binary_op!(sshr, bitwise_binary);
    checked_binary_op!(ushr, bitwise_binary);

    #[doc(hidden)]
    pub fn icmp(&mut self, cc: IntCmp, a: ValueId, b: ValueId) -> ValueId {
        let ty = same_type(a, b, "icmp");
        assert!(
            ty.is_integer() || matches!(ty, ScalarType::Bool | ScalarType::Ptr),
            "icmp: expected integer, boolean, or pointer operands, found {ty:?}"
        );
        if ty == ScalarType::Ptr {
            assert!(
                matches!(cc, IntCmp::Eq | IntCmp::Ne),
                "icmp: pointers only support equality comparisons"
            );
        }
        let result = self.backend.icmp(cc, a, b);
        expect_type(result, ScalarType::Bool, "icmp")
    }

    #[doc(hidden)]
    pub fn icmp_imm(&mut self, cc: IntCmp, value: ValueId, imm: i64) -> ValueId {
        let ty = value.scalar_type();
        assert!(
            ty.is_integer() || matches!(ty, ScalarType::Bool | ScalarType::Ptr),
            "icmp_imm: expected an integer, boolean, or pointer operand, found {ty:?}"
        );
        if ty == ScalarType::Ptr {
            assert!(
                imm == 0 && matches!(cc, IntCmp::Eq | IntCmp::Ne),
                "icmp_imm: pointers may only be compared with null for equality"
            );
        }
        let result = self.backend.icmp_imm(cc, value, imm);
        expect_type(result, ScalarType::Bool, "icmp_imm")
    }

    #[doc(hidden)]
    pub fn fcmp(&mut self, cc: FloatCmp, a: ValueId, b: ValueId) -> ValueId {
        float_binary(a, b, "fcmp");
        let result = self.backend.fcmp(cc, a, b);
        expect_type(result, ScalarType::Bool, "fcmp")
    }

    #[doc(hidden)]
    pub fn select(&mut self, cond: ValueId, a: ValueId, b: ValueId) -> ValueId {
        expect_type(cond, ScalarType::Bool, "select condition");
        let ty = same_type(a, b, "select");
        let result = self.backend.select(cond, a, b);
        expect_type(result, ty, "select")
    }

    fn check_integer_cast(from: ScalarType, to: ScalarType, operation: &str) {
        assert!(
            from.is_integer() && to.is_integer(),
            "{operation}: expected integer types, found {from:?} -> {to:?}"
        );
    }

    #[doc(hidden)]
    pub fn sextend(&mut self, to: ScalarType, value: ValueId) -> ValueId {
        let from = value.scalar_type();
        Self::check_integer_cast(from, to, "sextend");
        assert!(
            from.bit_width() < to.bit_width(),
            "sextend: target must be wider"
        );
        let result = self.backend.sextend(to, value);
        expect_type(result, to, "sextend")
    }

    #[doc(hidden)]
    pub fn uextend(&mut self, to: ScalarType, value: ValueId) -> ValueId {
        let from = value.scalar_type();
        Self::check_integer_cast(from, to, "uextend");
        assert!(
            from.bit_width() < to.bit_width(),
            "uextend: target must be wider"
        );
        let result = self.backend.uextend(to, value);
        expect_type(result, to, "uextend")
    }

    #[doc(hidden)]
    pub fn ireduce(&mut self, to: ScalarType, value: ValueId) -> ValueId {
        let from = value.scalar_type();
        Self::check_integer_cast(from, to, "ireduce");
        assert!(
            from.bit_width() > to.bit_width(),
            "ireduce: target must be narrower"
        );
        let result = self.backend.ireduce(to, value);
        expect_type(result, to, "ireduce")
    }

    #[doc(hidden)]
    pub fn fcvt_from_sint(&mut self, to: ScalarType, value: ValueId) -> ValueId {
        assert!(
            value.scalar_type().is_integer() && to.is_float(),
            "fcvt_from_sint: expected integer -> float, found {:?} -> {to:?}",
            value.scalar_type()
        );
        let result = self.backend.fcvt_from_sint(to, value);
        expect_type(result, to, "fcvt_from_sint")
    }

    #[doc(hidden)]
    pub fn fcvt_from_uint(&mut self, to: ScalarType, value: ValueId) -> ValueId {
        assert!(
            value.scalar_type().is_integer() && to.is_float(),
            "fcvt_from_uint: expected integer -> float, found {:?} -> {to:?}",
            value.scalar_type()
        );
        let result = self.backend.fcvt_from_uint(to, value);
        expect_type(result, to, "fcvt_from_uint")
    }

    #[doc(hidden)]
    pub fn bitcast(&mut self, to: ScalarType, value: ValueId) -> ValueId {
        let from = value.scalar_type();
        assert!(
            (from.is_integer() || from.is_float()) && (to.is_integer() || to.is_float()),
            "bitcast: expected numeric scalar types, found {from:?} -> {to:?}"
        );
        assert_eq!(
            from.bit_width(),
            to.bit_width(),
            "bitcast: source and target must have equal width"
        );
        let result = self.backend.bitcast(to, value);
        expect_type(result, to, "bitcast")
    }

    #[doc(hidden)]
    pub fn load(&mut self, ty: ScalarType, ptr: ValueId, offset: i32) -> ValueId {
        expect_type(ptr, ScalarType::Ptr, "load pointer");
        let result = self.backend.load(ty, ptr, offset);
        expect_type(result, ty, "load")
    }

    #[doc(hidden)]
    pub fn store(&mut self, value: ValueId, ptr: ValueId, offset: i32) {
        expect_type(ptr, ScalarType::Ptr, "store pointer");
        self.backend.store(value, ptr, offset);
    }

    #[doc(hidden)]
    pub fn stack_addr(&mut self, slot: StackSlotId, offset: i32) -> ValueId {
        let result = self.backend.stack_addr(slot, offset);
        expect_type(result, ScalarType::Ptr, "stack_addr")
    }

    #[doc(hidden)]
    pub fn copy_nonoverlapping(&mut self, dst: ValueId, src: ValueId, size: usize, align: usize) {
        expect_type(dst, ScalarType::Ptr, "copy_nonoverlapping destination");
        expect_type(src, ScalarType::Ptr, "copy_nonoverlapping source");
        self.backend.copy_nonoverlapping(dst, src, size, align);
    }

    #[doc(hidden)]
    pub fn ptr_offset_bytes(&mut self, ptr: ValueId, offset: ValueId) -> ValueId {
        expect_type(ptr, ScalarType::Ptr, "ptr_offset_bytes pointer");
        expect_type(offset, ScalarType::I64, "ptr_offset_bytes offset");
        let result = self.backend.ptr_offset_bytes(ptr, offset);
        expect_type(result, ScalarType::Ptr, "ptr_offset_bytes")
    }

    #[doc(hidden)]
    pub fn ptr_offset_const(&mut self, ptr: ValueId, bytes: i64) -> ValueId {
        expect_type(ptr, ScalarType::Ptr, "ptr_offset_const pointer");
        let result = self.backend.ptr_offset_const(ptr, bytes);
        expect_type(result, ScalarType::Ptr, "ptr_offset_const")
    }

    #[doc(hidden)]
    pub fn addr_to_ptr(&mut self, addr: ValueId) -> ValueId {
        expect_type(addr, ScalarType::I64, "addr_to_ptr address");
        let result = self.backend.addr_to_ptr(addr);
        expect_type(result, ScalarType::Ptr, "addr_to_ptr")
    }

    #[doc(hidden)]
    pub fn append_block_param(&mut self, block: BlockHandle, ty: ScalarType) -> ValueId {
        let result = self.backend.append_block_param(block, ty);
        self.block_params
            .get_mut(&block)
            .unwrap_or_else(|| panic!("append_block_param: unknown block {block:?}"))
            .push(ty);
        expect_type(result, ty, "append_block_param")
    }

    #[doc(hidden)]
    pub fn block_param(&mut self, block: BlockHandle, index: usize, ty: ScalarType) -> ValueId {
        let actual = *self
            .block_params
            .get(&block)
            .unwrap_or_else(|| panic!("block_param: unknown block {block:?}"))
            .get(index)
            .unwrap_or_else(|| panic!("block_param: block {block:?} has no parameter {index}"));
        assert_eq!(
            actual, ty,
            "block_param: expected parameter {index} of {block:?} to be {ty:?}, found {actual:?}"
        );
        let result = self.backend.block_param(block, index, ty);
        expect_type(result, ty, "block_param")
    }

    #[doc(hidden)]
    pub fn create_block(&mut self) -> BlockHandle {
        let block = self.backend.create_block();
        assert!(
            self.block_params.insert(block, Vec::new()).is_none(),
            "create_block: backend reused block handle {block:?}"
        );
        block
    }

    fn check_block_args(&self, operation: &str, block: BlockHandle, args: &[ValueId]) {
        let params = self
            .block_params
            .get(&block)
            .unwrap_or_else(|| panic!("{operation}: unknown target block {block:?}"));
        assert_eq!(
            args.len(),
            params.len(),
            "{operation}: block {block:?} expects {} arguments, found {}",
            params.len(),
            args.len()
        );
        for (index, (&arg, &expected)) in args.iter().zip(params).enumerate() {
            expect_type(arg, expected, &format!("{operation} argument {index}"));
        }
    }

    #[doc(hidden)]
    pub fn jump(&mut self, target: BlockHandle, args: &[ValueId]) {
        self.check_block_args("jump", target, args);
        self.backend.jump(target, args);
    }

    #[doc(hidden)]
    pub fn brif(
        &mut self,
        cond: ValueId,
        then_block: BlockHandle,
        then_args: &[ValueId],
        else_block: BlockHandle,
        else_args: &[ValueId],
    ) {
        let ty = cond.scalar_type();
        assert!(
            ty == ScalarType::Bool || ty.is_integer() || ty == ScalarType::Ptr,
            "brif: expected a boolean, integer, or pointer condition, found {ty:?}"
        );
        self.check_block_args("brif then", then_block, then_args);
        self.check_block_args("brif else", else_block, else_args);
        self.backend
            .brif(cond, then_block, then_args, else_block, else_args);
    }

    #[doc(hidden)]
    pub fn declare_var(&mut self, ty: ScalarType) -> VarHandle {
        let var = self.backend.declare_var(ty);
        assert_eq!(
            var.scalar_type(),
            ty,
            "declare_var: backend returned a variable with the wrong type"
        );
        var
    }

    #[doc(hidden)]
    pub fn def_var(&mut self, var: VarHandle, value: ValueId) {
        expect_type(value, var.scalar_type(), "def_var");
        self.backend.def_var(var, value);
    }

    #[doc(hidden)]
    pub fn use_var(&mut self, var: VarHandle) -> ValueId {
        let result = self.backend.use_var(var);
        expect_type(result, var.scalar_type(), "use_var")
    }

    #[doc(hidden)]
    pub fn call(&mut self, func: FuncRefId, args: &[ValueId]) -> Option<ValueId> {
        let result = self.backend.call(func, args);
        assert_eq!(
            result.map(ValueId::scalar_type),
            func.return_type(),
            "call: backend result does not match the declared function result"
        );
        result
    }

    #[doc(hidden)]
    pub fn call_indirect(
        &mut self,
        sig: SigRefId,
        callee: ValueId,
        args: &[ValueId],
    ) -> Option<ValueId> {
        expect_type(callee, ScalarType::Ptr, "call_indirect callee");
        let result = self.backend.call_indirect(sig, callee, args);
        assert_eq!(
            result.map(ValueId::scalar_type),
            sig.return_type(),
            "call_indirect: backend result does not match the declared signature result"
        );
        result
    }

    #[doc(hidden)]
    pub fn func_addr(&mut self, func: FuncRefId) -> ValueId {
        let result = self.backend.func_addr(func);
        expect_type(result, ScalarType::Ptr, "func_addr")
    }

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
            Value::fat(
                self.block_param(block, 0, ScalarType::Ptr),
                self.block_param(block, 1, ScalarType::I64),
            )
        } else {
            Value::scalar(self.block_param(block, 0, T::scalar_type()))
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
                expect_type(leaf, T::scalar_type(), "staged variable assignment");
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
        expect_type(base, ScalarType::Ptr, "load_fat base");
        let ptr = self.load(ScalarType::Ptr, base, 0);
        let len = self.load(ScalarType::I64, base, 8);
        Value::fat(ptr, len)
    }

    /// Load a staged value from its canonical in-memory representation.
    pub(crate) fn load_value<T: StagedType>(&mut self, base: ValueId) -> Value {
        expect_type(base, ScalarType::Ptr, "load_value base");
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
        expect_type(base, ScalarType::Ptr, "store_value base");
        if T::is_fat_pointer() {
            let (ptr, len) = value.parts();
            self.store(ptr, base, 0);
            self.store(len, base, 8);
        } else if T::is_copy_struct() {
            let value = expect_type(value.leaf(), ScalarType::Ptr, "copy value");
            self.copy_nonoverlapping(base, value, T::size_of(), T::align_of());
        } else if T::size_of() != 0 {
            let value = expect_type(value.leaf(), T::scalar_type(), "stored value");
            self.store(value, base, 0);
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
