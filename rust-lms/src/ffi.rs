//! Foreign Function Interface (FFI) support for calling external Rust functions from JIT code.
//!
//! This module provides:
//! - `FatSlice<T>` / `FatSliceMut<T>`: FFI-safe slice representations
//! - `ExternFn` trait: Metadata for external functions
//! - `ExternRef<S>`: Type-safe handle to external functions
//!
//! # Usage
//!
//! ```ignore
//! use rust_lms_derive::extern_fn;
//!
//! #[extern_fn]
//! #[no_mangle]
//! pub extern "C" fn my_add(x: i64, y: i64) -> i64 {
//!     x + y
//! }
//!
//! // In staged code:
//! let my_add = compiler.extern_fn::<MyAddExtern>();
//! let result = call2(my_add, x, y);
//! ```

use std::marker::PhantomData;
use std::slice;

use crate::refer::{SMutPtr, SPtr};
use crate::staged::{CompilationContext, IntoStaged, Staged};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value};

// =============================================================================
// FatSlice<T> - FFI-safe immutable slice
// =============================================================================

/// FFI-safe representation of an immutable slice `&[T]`.
///
/// This struct has a well-defined C ABI layout and can be passed by value
/// across FFI boundaries (for slices ≤16 bytes, i.e., ptr + len = 16 bytes).
///
/// # Example
///
/// ```ignore
/// #[extern_fn]
/// pub extern "C" fn sum_slice(data: FatSlice<i64>) -> i64 {
///     data.as_slice().iter().sum()
/// }
/// ```
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct FatSlice<T> {
    /// Pointer to the first element
    pub ptr: *const T,
    /// Number of elements
    pub len: usize,
}

impl<T> FatSlice<T> {
    /// Create a new FatSlice from a pointer and length.
    ///
    /// # Safety
    /// The pointer must be valid for `len` elements of type `T`.
    pub const unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        FatSlice { ptr, len }
    }

    /// Create a FatSlice from a Rust slice.
    pub fn from_slice(slice: &[T]) -> Self {
        FatSlice {
            ptr: slice.as_ptr(),
            len: slice.len(),
        }
    }

    /// Convert back to a Rust slice.
    ///
    /// # Safety
    /// The pointer must still be valid and the data must not have been modified
    /// in a way that violates Rust's aliasing rules.
    pub unsafe fn as_slice(&self) -> &[T] {
        slice::from_raw_parts(self.ptr, self.len)
    }

    /// Returns true if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> From<&[T]> for FatSlice<T> {
    fn from(slice: &[T]) -> Self {
        FatSlice::from_slice(slice)
    }
}

// =============================================================================
// FatSliceMut<T> - FFI-safe mutable slice
// =============================================================================

/// FFI-safe representation of a mutable slice `&mut [T]`.
///
/// This struct has a well-defined C ABI layout and can be passed by value
/// across FFI boundaries.
#[repr(C)]
#[derive(Debug)]
pub struct FatSliceMut<T> {
    /// Pointer to the first element
    pub ptr: *mut T,
    /// Number of elements
    pub len: usize,
}

impl<T> FatSliceMut<T> {
    /// Create a new FatSliceMut from a pointer and length.
    ///
    /// # Safety
    /// The pointer must be valid for `len` elements of type `T`.
    pub const unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        FatSliceMut { ptr, len }
    }

    /// Create a FatSliceMut from a Rust mutable slice.
    pub fn from_slice(slice: &mut [T]) -> Self {
        FatSliceMut {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
        }
    }

    /// Convert back to a Rust mutable slice.
    ///
    /// # Safety
    /// The pointer must still be valid and the data must not have been modified
    /// in a way that violates Rust's aliasing rules.
    pub unsafe fn as_slice_mut(&mut self) -> &mut [T] {
        slice::from_raw_parts_mut(self.ptr, self.len)
    }

    /// Convert to an immutable slice.
    ///
    /// # Safety
    /// Same requirements as `as_slice_mut`.
    pub unsafe fn as_slice(&self) -> &[T] {
        slice::from_raw_parts(self.ptr, self.len)
    }

    /// Returns true if the slice is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<T> From<&mut [T]> for FatSliceMut<T> {
    fn from(slice: &mut [T]) -> Self {
        FatSliceMut::from_slice(slice)
    }
}

/// ABI-facing name for [`FatSlice`].
///
/// `FfiSlice<T>` is the concrete runtime `(ptr, len)` value passed across
/// function boundaries. `SRef<Slice<T>>` is the staged view over the same
/// representation inside generated code.
pub type FfiSlice<T> = FatSlice<T>;

/// ABI-facing name for [`FatSliceMut`].
pub type FfiSliceMut<T> = FatSliceMut<T>;

// =============================================================================
// StagedType implementations for FatSlice
// =============================================================================

/// Staged type marker for FatSlice<T>
#[derive(Clone, Copy, Debug)]
pub struct FatSliceType<T> {
    _phantom: PhantomData<T>,
}

impl<T: StagedType> StagedType for FatSliceType<T> {
    type RuntimeValue = FatSlice<T::RuntimeValue>;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        // FatSlice is 16 bytes (ptr + len), represented as pointer to stack slot
        types::I64
    }

    fn size_of() -> usize {
        std::mem::size_of::<FatSlice<()>>()
    }

    fn align_of() -> usize {
        std::mem::align_of::<FatSlice<()>>()
    }

    fn is_copy_struct() -> bool {
        true
    }

    fn num_abi_values() -> usize {
        2 // ptr + len
    }

    fn abi_types() -> Vec<cranelift_codegen::ir::Type> {
        vec![types::I64, types::I64]
    }
}

/// Staged type marker for FatSliceMut<T>
#[derive(Clone, Copy, Debug)]
pub struct FatSliceMutType<T> {
    _phantom: PhantomData<T>,
}

/// Staged type marker for [`FfiSlice`].
pub type FfiSliceType<T> = FatSliceType<T>;

/// Staged type marker for [`FfiSliceMut`].
pub type FfiSliceMutType<T> = FatSliceMutType<T>;

impl<T: StagedType> StagedType for FatSliceMutType<T> {
    type RuntimeValue = FatSliceMut<T::RuntimeValue>;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn size_of() -> usize {
        std::mem::size_of::<FatSliceMut<()>>()
    }

    fn align_of() -> usize {
        std::mem::align_of::<FatSliceMut<()>>()
    }

    fn is_copy_struct() -> bool {
        true
    }

    fn num_abi_values() -> usize {
        2 // ptr + len
    }

    fn abi_types() -> Vec<cranelift_codegen::ir::Type> {
        vec![types::I64, types::I64]
    }
}

// =============================================================================
// SliceFromRawParts: build a staged FatSlice from a raw (ptr, len)
// =============================================================================

/// Build a staged `FatSlice<T>` (`&[T]` at the ABI) from a typed pointer (`SPtr<T>`)
/// and an element length — e.g. a baked host buffer handed to an extern `&[u8]`
/// param. Materializes the `(ptr, len)` pair on a stack slot, like sub-slicing.
pub struct SliceFromRawParts<P, L, T> {
    ptr: P,
    len: L,
    _elem: PhantomData<T>,
}

impl<P: Clone, L: Clone, T> Clone for SliceFromRawParts<P, L, T> {
    fn clone(&self) -> Self {
        Self {
            ptr: self.ptr.clone(),
            len: self.len.clone(),
            _elem: PhantomData,
        }
    }
}

impl<P: Copy, L: Copy, T> Copy for SliceFromRawParts<P, L, T> {}

impl<P, L, T> Staged for SliceFromRawParts<P, L, T>
where
    P: Staged<Out = SPtr<T>>,
    L: Staged<Out = u64>,
    T: StagedType + 'static,
{
    type Out = FatSliceType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.ptr.codegen(ctx);
        let len = self.len.codegen(ctx);
        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            16,
            3,
        ));
        let slot_ptr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), ptr, slot_ptr, 0);
        ctx.builder
            .ins()
            .store(MemFlags::trusted(), len, slot_ptr, 8);
        slot_ptr
    }
}

/// Build a staged `FatSlice<T>` from a typed pointer (`SPtr<T>`) and element length.
pub fn slice_from_raw_parts<T, P, L>(ptr: P, len: L) -> SliceFromRawParts<P, L::Staged, T>
where
    T: StagedType + 'static,
    P: Staged<Out = SPtr<T>>,
    L: IntoStaged<u64>,
{
    SliceFromRawParts {
        ptr,
        len: len.into_staged(),
        _elem: PhantomData,
    }
}

// =============================================================================
// SliceRefFromRawParts: rebuild a borrowed slice `&[T]` from (ptr, len)
// =============================================================================

/// The **reference** flavor of [`slice_from_raw_parts`]: yields a
/// `SRef<Slice<T>>` (`&[T]`) instead of a `FatSlice<T>`, so it feeds the slice
/// *read* ops (`primitive`/`get_ref_unchecked`). Both share one materialization —
/// a `(ptr, len)` header on a stack slot — and that layout *is* how a
/// memory-resolved `SRef<Slice<T>>` is read (`ptr`@0, `len`@8), so this is a
/// no-extra-code reinterpret of the same header. Use it to rebuild a borrowed
/// batch from a raw descriptor pointer plus a known column count (e.g. a value
/// returned by a scan-stream extern).
pub struct SliceRefFromRawParts<P, L, T> {
    inner: SliceFromRawParts<P, L, T>,
}

impl<P: Clone, L: Clone, T> Clone for SliceRefFromRawParts<P, L, T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

impl<P: Copy, L: Copy, T> Copy for SliceRefFromRawParts<P, L, T> {}

impl<P, L, T> Staged for SliceRefFromRawParts<P, L, T>
where
    P: Staged<Out = SPtr<T>>,
    L: Staged<Out = u64>,
    T: StagedType + 'static,
{
    type Out = crate::refer::SRef<'static, crate::slice::Slice<T>>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Same `(ptr, len)` stack header as `FatSlice`; only the static type differs.
        self.inner.codegen(ctx)
    }
}

/// Build a staged `&[T]` (`SRef<Slice<T>>`) from a typed pointer and element length.
pub fn slice_ref_from_raw_parts<T, P, L>(ptr: P, len: L) -> SliceRefFromRawParts<P, L::Staged, T>
where
    T: StagedType + 'static,
    P: Staged<Out = SPtr<T>>,
    L: IntoStaged<u64>,
{
    SliceRefFromRawParts {
        inner: slice_from_raw_parts(ptr, len),
    }
}

// =============================================================================
// StackBytes: bake host bytes into a kernel-frame stack slot
// =============================================================================

/// A byte literal materialized into the kernel's own stack frame: the bytes are
/// emitted as `iconst` stores at codegen and live for the whole kernel call.
///
/// Use this — never a baked host pointer — when literal bytes must reach stage-1
/// code (e.g. an extern `&[u8]` param). A host address captured at codegen dangles
/// once the codegen-time owner (a cloned `Expr`, say) drops, long before the JIT
/// kernel runs. Returns a typed `SPtr<u8>` to the slot; pair it with the length
/// via [`slice_from_raw_parts`] to form a `&[u8]`.
#[derive(Clone)]
pub struct StackBytes {
    bytes: Vec<u8>,
}

impl Staged for StackBytes {
    type Out = SPtr<u8>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let n = self.bytes.len();
        // Round the slot up to a whole number of 8-byte words (min one word, so a
        // zero-length literal still has a valid, non-empty slot to address).
        let slot_len = ((n + 7) & !7).max(8);
        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            slot_len as u32,
            3,
        ));
        let addr = ctx.builder.ins().stack_addr(types::I64, slot, 0);
        let mut off = 0;
        while off < slot_len {
            let mut word = [0u8; 8];
            for (j, w) in word.iter_mut().enumerate() {
                if off + j < n {
                    *w = self.bytes[off + j];
                }
            }
            let v = ctx
                .builder
                .ins()
                .iconst(types::I64, u64::from_le_bytes(word) as i64);
            ctx.builder
                .ins()
                .store(MemFlags::trusted(), v, addr, off as i32);
            off += 8;
        }
        addr
    }
}

/// Bake `bytes` into the kernel's stack frame; returns a typed `SPtr<u8>` to the
/// slot. See [`StackBytes`].
pub fn stack_bytes(bytes: &[u8]) -> StackBytes {
    StackBytes {
        bytes: bytes.to_vec(),
    }
}

// =============================================================================
// StackAlloc: reserve a mutable kernel-frame stack slot for runtime scratch
// =============================================================================

/// A fresh, **uninitialised**, 8-byte-aligned stack slot of `size` bytes, as a
/// `SMutPtr<u8>` the kernel writes into — the runtime counterpart of [`StackBytes`]
/// (which stores *constant* bytes). Scratch space for building a value at runtime:
/// packing a composite GROUP BY key, a temporary buffer for an extern, etc.
///
/// The slot lives in the function frame and is allocated **once**, so evaluating this
/// inside a loop reuses the same slot every iteration (bind it once and reuse the
/// pointer). Its contents are undefined until the kernel writes them.
pub struct StackAlloc {
    size: usize,
}

impl Staged for StackAlloc {
    type Out = SMutPtr<u8>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Round up to a whole number of 8-byte words (min one word).
        let slot_len = ((self.size + 7) & !7).max(8);
        let slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            slot_len as u32,
            3, // align_shift = 3 → 8-byte aligned
        ));
        ctx.builder.ins().stack_addr(types::I64, slot, 0)
    }
}

/// Reserve `size` bytes of uninitialised, 8-byte-aligned stack scratch. See [`StackAlloc`].
pub fn stack_alloc(size: usize) -> StackAlloc {
    StackAlloc { size }
}

// =============================================================================
// ExternFn trait - Metadata for external functions
// =============================================================================

/// Trait implemented by types generated by `#[extern_fn]` macro.
///
/// This trait provides all metadata needed to register and call an external
/// function from JIT code.
///
/// # Safety
///
/// Implementations must ensure that:
/// - `PARAM_TYPES` and `RETURN_TYPE` accurately reflect the function signature
/// - `FN_PTR` points to a valid `extern "C"` function with the declared signature
pub unsafe trait ExternFn {
    /// The staged return type. Lets `call_externN` derive its `Out` from the
    /// function's own signature, so callers never restate it.
    type Ret: StagedType;

    /// Function name (for debugging and symbol resolution)
    const NAME: &'static str;

    /// Number of parameters
    const NUM_PARAMS: usize;

    /// Function pointer as raw bytes
    const FN_PTR: *const u8;

    /// Get the Cranelift ABI parameter types.
    /// Returns a Vec of (num_abi_values, types) for each logical parameter.
    fn param_abi_types() -> Vec<Vec<cranelift_codegen::ir::Type>>;

    /// Get the Cranelift ABI return types.
    fn return_abi_types() -> Vec<cranelift_codegen::ir::Type>;

    /// Returns true if the return type is a struct that should be passed by pointer
    fn return_by_pointer() -> bool {
        false
    }

    /// For each parameter, returns true if it should be passed by pointer (>16 bytes)
    fn param_by_pointer() -> Vec<bool> {
        vec![false; Self::NUM_PARAMS]
    }
}

// =============================================================================
// ExternRef<S> - Handle to an external function
// =============================================================================

/// A reference to an external function that can be called from staged code.
///
/// This is analogous to `FunRef` but for external functions. It stores the
/// Cranelift FuncRef after the function has been imported into the module.
pub struct ExternRef<S: ExternFn> {
    /// Index into the extern function table (assigned during compilation)
    pub(crate) extern_id: usize,
    _phantom: PhantomData<S>,
}

// A handle is just an index, so it is always Copy regardless of `S` (a `S: Copy`
// bound from `#[derive]` would leak into every holder, and the generated extern
// marker structs are not `Copy`).
impl<S: ExternFn> Clone for ExternRef<S> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<S: ExternFn> Copy for ExternRef<S> {}

impl<S: ExternFn> ExternRef<S> {
    pub(crate) fn new(extern_id: usize) -> Self {
        ExternRef {
            extern_id,
            _phantom: PhantomData,
        }
    }
}

// =============================================================================
// CallExtern - Expression for calling external functions
// =============================================================================

/// Call an external function with 0 arguments. `Out` is the function's own
/// return type (`S::Ret`), so callers never restate it.
pub struct CallExtern0<S: ExternFn> {
    func: ExternRef<S>,
}

impl<S: ExternFn> Staged for CallExtern0<S> {
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        let call = ctx.builder.ins().call(func_ref, &[]);
        finish_extern_call::<S::Ret>(ctx, call)
    }
}

/// Create a call to an external function with 0 arguments
pub fn call_extern0<S: ExternFn>(func: ExternRef<S>) -> CallExtern0<S> {
    CallExtern0 { func }
}

/// Call an external function with 1 argument. `Out` is `S::Ret`.
pub struct CallExtern1<S: ExternFn, A> {
    func: ExternRef<S>,
    arg: A,
}

impl<S, A, AType> Staged for CallExtern1<S, A>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    AType: StagedType,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        let arg_val = self.arg.codegen(ctx);

        let mut args = Vec::new();
        push_extern_arg::<AType>(ctx, &mut args, arg_val);

        let call = ctx.builder.ins().call(func_ref, &args);
        finish_extern_call::<S::Ret>(ctx, call)
    }
}

/// Create a call to an external function with 1 argument
pub fn call_extern1<S, A, AType>(func: ExternRef<S>, arg: A) -> CallExtern1<S, A>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    AType: StagedType,
{
    CallExtern1 { func, arg }
}

/// Call an external function with 2 arguments. `Out` is `S::Ret`.
pub struct CallExtern2<S: ExternFn, A, B> {
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
}

impl<S, A, B, AType, BType> Staged for CallExtern2<S, A, B>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    AType: StagedType,
    BType: StagedType,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        let arg0_val = self.arg0.codegen(ctx);
        let arg1_val = self.arg1.codegen(ctx);

        let mut args = Vec::new();
        push_extern_arg::<AType>(ctx, &mut args, arg0_val);
        push_extern_arg::<BType>(ctx, &mut args, arg1_val);

        let call = ctx.builder.ins().call(func_ref, &args);
        finish_extern_call::<S::Ret>(ctx, call)
    }
}

/// Create a call to an external function with 2 arguments
pub fn call_extern2<S, A, B, AType, BType>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
) -> CallExtern2<S, A, B>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    AType: StagedType,
    BType: StagedType,
{
    CallExtern2 { func, arg0, arg1 }
}

/// Append `arg`'s ABI value(s) to `args` (flattening a copy-struct passed by
/// pointer into its register-sized parts), mirroring the per-arg handling in
/// `CallExtern1`/`CallExtern2`.
fn push_extern_arg<AType: StagedType>(
    ctx: &mut CompilationContext,
    args: &mut Vec<Value>,
    arg_val: Value,
) {
    if AType::is_copy_struct() && AType::num_abi_values() > 1 {
        for i in 0..AType::num_abi_values() {
            let offset = (i * 8) as i32;
            let val = ctx.builder.ins().load(
                types::I64,
                cranelift_codegen::ir::MemFlags::trusted(),
                arg_val,
                offset,
            );
            args.push(val);
        }
    } else {
        args.push(arg_val);
    }
}

/// Materialize an extern call's result: a multi-value copy-struct return is
/// spilled to a stack slot (whose address is returned); any other return is the
/// single result value. Shared by every `CallExternN::codegen`.
fn finish_extern_call<Ret: StagedType>(
    ctx: &mut CompilationContext,
    call: cranelift_codegen::ir::Inst,
) -> Value {
    if Ret::is_copy_struct() && Ret::num_abi_values() > 1 {
        let align_shift = Ret::align_of().trailing_zeros() as u8;
        let stack_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            Ret::size_of() as u32,
            align_shift,
        ));
        let slot_ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);
        let results = ctx.builder.inst_results(call).to_vec();
        for (i, val) in results.iter().enumerate() {
            ctx.builder
                .ins()
                .store(MemFlags::trusted(), *val, slot_ptr, (i * 8) as i32);
        }
        slot_ptr
    } else {
        ctx.builder.inst_results(call)[0]
    }
}

/// Call an external function with 3 arguments. `Out` is `S::Ret`.
pub struct CallExtern3<S: ExternFn, A, B, C> {
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
}

impl<S, A, B, C, AType, BType, CType> Staged for CallExtern3<S, A, B, C>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    C: Staged<Out = CType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        let arg0_val = self.arg0.codegen(ctx);
        let arg1_val = self.arg1.codegen(ctx);
        let arg2_val = self.arg2.codegen(ctx);

        let mut args = Vec::new();
        push_extern_arg::<AType>(ctx, &mut args, arg0_val);
        push_extern_arg::<BType>(ctx, &mut args, arg1_val);
        push_extern_arg::<CType>(ctx, &mut args, arg2_val);

        let call = ctx.builder.ins().call(func_ref, &args);
        finish_extern_call::<S::Ret>(ctx, call)
    }
}

/// Create a call to an external function with 3 arguments.
pub fn call_extern3<S, A, B, C, AType, BType, CType>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
) -> CallExtern3<S, A, B, C>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    C: Staged<Out = CType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
{
    CallExtern3 {
        func,
        arg0,
        arg1,
        arg2,
    }
}

/// Call an external function with 4 arguments. `Out` is `S::Ret`.
pub struct CallExtern4<S: ExternFn, A, B, C, D> {
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
    arg3: D,
}

impl<S, A, B, C, D, AType, BType, CType, DType> Staged for CallExtern4<S, A, B, C, D>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    C: Staged<Out = CType>,
    D: Staged<Out = DType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
    DType: StagedType,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        let arg0_val = self.arg0.codegen(ctx);
        let arg1_val = self.arg1.codegen(ctx);
        let arg2_val = self.arg2.codegen(ctx);
        let arg3_val = self.arg3.codegen(ctx);

        let mut args = Vec::new();
        push_extern_arg::<AType>(ctx, &mut args, arg0_val);
        push_extern_arg::<BType>(ctx, &mut args, arg1_val);
        push_extern_arg::<CType>(ctx, &mut args, arg2_val);
        push_extern_arg::<DType>(ctx, &mut args, arg3_val);

        let call = ctx.builder.ins().call(func_ref, &args);
        finish_extern_call::<S::Ret>(ctx, call)
    }
}

/// Create a call to an external function with 4 arguments.
#[allow(clippy::too_many_arguments)]
pub fn call_extern4<S, A, B, C, D, AType, BType, CType, DType>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
    arg3: D,
) -> CallExtern4<S, A, B, C, D>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    C: Staged<Out = CType>,
    D: Staged<Out = DType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
    DType: StagedType,
{
    CallExtern4 {
        func,
        arg0,
        arg1,
        arg2,
        arg3,
    }
}

// Additional CallExternN types can be generated via macro if needed

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fat_slice_from_slice() {
        let data = [1i64, 2, 3, 4, 5];
        let fat = FatSlice::from_slice(&data);

        assert_eq!(fat.len, 5);
        assert_eq!(fat.ptr, data.as_ptr());
        assert!(!fat.is_empty());

        unsafe {
            assert_eq!(fat.as_slice(), &data);
        }
    }

    #[test]
    fn test_fat_slice_empty() {
        let data: [i64; 0] = [];
        let fat = FatSlice::from_slice(&data);

        assert_eq!(fat.len, 0);
        assert!(fat.is_empty());
    }

    #[test]
    fn test_fat_slice_mut() {
        let mut data = [1i64, 2, 3];
        let mut fat = FatSliceMut::from_slice(&mut data);

        unsafe {
            let slice = fat.as_slice_mut();
            slice[0] = 10;
            slice[1] = 20;
        }

        assert_eq!(data, [10, 20, 3]);
    }

    #[test]
    fn test_fat_slice_layout() {
        // Verify FatSlice is exactly 16 bytes (2 x 8 bytes)
        assert_eq!(std::mem::size_of::<FatSlice<i64>>(), 16);
        assert_eq!(std::mem::align_of::<FatSlice<i64>>(), 8);

        assert_eq!(std::mem::size_of::<FatSliceMut<i64>>(), 16);
        assert_eq!(std::mem::align_of::<FatSliceMut<i64>>(), 8);
    }
}
