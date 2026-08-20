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

use crate::refer::{SMutPtr, SPtr, SRef, SRefMut};
use crate::slice::Slice;
use crate::staged::{CompilationContext, IntoStaged, Staged, Var, VarUse};
use crate::types::{CopyType, RuntimeParam, RuntimeResult, StagedType};
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
#[derive(Debug)]
pub struct FatSlice<T> {
    /// Pointer to the first element
    pub ptr: *const T,
    /// Number of elements
    pub len: usize,
}

impl<T> Clone for FatSlice<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for FatSlice<T> {}

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
#[derive(Debug)]
pub struct FatSliceType<T> {
    _phantom: PhantomData<T>,
}

impl<T> Clone for FatSliceType<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for FatSliceType<T> {}

unsafe impl<T: StagedType> StagedType for FatSliceType<T> {
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
}

unsafe impl<T: StagedType> CopyType for FatSliceType<T> {}

unsafe impl<T: StagedType> RuntimeParam for FatSliceType<T> {
    type Arg<'call> = FatSlice<T::RuntimeValue>;
}

unsafe impl<T: StagedType> RuntimeResult for FatSliceType<T> {
    type Output<'call> = FatSlice<T::RuntimeValue>;
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

unsafe impl<T: StagedType> StagedType for FatSliceMutType<T> {
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
}

unsafe impl<T: StagedType> RuntimeParam for FatSliceMutType<T> {
    type Arg<'call> = FatSliceMut<T::RuntimeValue>;
}

unsafe impl<T: StagedType> RuntimeResult for FatSliceMutType<T> {
    type Output<'call> = FatSliceMut<T::RuntimeValue>;
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

unsafe impl<P, L, T> Staged for SliceFromRawParts<P, L, T>
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
///
/// # Safety
///
/// The pointer must be aligned and valid for `len` initialized elements for the
/// full duration of every generated-code use. The memory must not be mutated
/// while an extern call holds the resulting shared slice.
pub unsafe fn slice_from_raw_parts<T, P, L>(ptr: P, len: L) -> SliceFromRawParts<P, L::Staged, T>
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

unsafe impl Staged for StackBytes {
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

unsafe impl Staged for StackAlloc {
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

mod extern_args_sealed {
    pub trait Sealed {}
}

/// A sealed tuple of staged parameter types for an [`ExternFn`].
///
/// Implementations are provided for `()` and tuples up to arity eight. Keeping
/// this sealed ensures that the type-level signature is always derived from
/// the staged parameter types.
pub trait ExternArgs: extern_args_sealed::Sealed {
    /// Number of logical parameters in the tuple.
    const LEN: usize;
}

impl extern_args_sealed::Sealed for () {}

impl ExternArgs for () {
    const LEN: usize = 0;
}

macro_rules! impl_extern_args {
    ($len:literal; $($T:ident),+) => {
        impl<$($T: StagedType),+> extern_args_sealed::Sealed for ($($T,)+) {}

        impl<$($T: StagedType),+> ExternArgs for ($($T,)+) {
            const LEN: usize = $len;
        }
    };
}

impl_extern_args!(1; A);
impl_extern_args!(2; A, B);
impl_extern_args!(3; A, B, C);
impl_extern_args!(4; A, B, C, D);
impl_extern_args!(5; A, B, C, D, E);
impl_extern_args!(6; A, B, C, D, E, F);
impl_extern_args!(7; A, B, C, D, E, F, G);
impl_extern_args!(8; A, B, C, D, E, F, G, H);

/// Trait implemented by types generated by `#[extern_fn]` macro.
///
/// This trait provides all metadata needed to register and call an external
/// function from JIT code.
///
/// # Safety
///
/// Implementations must ensure that:
/// - `Args` and `Ret` exactly match the function's ABI, including preserving
///   Rust shared and mutable references as `SRef` and `SRefMut` markers.
/// - `FN_PTR` points to a valid `extern "C"` function with the declared signature
pub unsafe trait ExternFn {
    /// The complete staged parameter signature.
    type Args: ExternArgs;

    /// The staged return type. Lets `call_externN` derive its `Out` from the
    /// function's own signature, so callers never restate it.
    type Ret: StagedType;

    /// Function name (for debugging and symbol resolution)
    const NAME: &'static str;

    /// Number of parameters
    const NUM_PARAMS: usize = <Self::Args as ExternArgs>::LEN;

    /// Function pointer as raw bytes
    const FN_PTR: *const u8;
}

/// An [`ExternFn`] whose target can be invoked by safe Rust when every value in
/// its typed signature is valid.
///
/// Unsafe `extern "C"` functions intentionally do not implement this trait and
/// must be staged through a `call_externN_unchecked` constructor.
///
/// # Safety
///
/// `FN_PTR` must point to a safe `extern "C"` function with the exact `Args` and
/// `Ret` declared by [`ExternFn`].
pub unsafe trait SafeExternFn: ExternFn {}

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

/// A no-code representation cast used after an extern-argument conversion has
/// established that two staged markers have the same ABI.
#[doc(hidden)]
pub struct ExternArgCast<A, T> {
    arg: A,
    _expected: PhantomData<T>,
}

unsafe impl<A, T> Staged for ExternArgCast<A, T>
where
    A: Staged,
    T: StagedType,
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.arg.codegen(ctx)
    }

    fn var_id(&self) -> Option<usize> {
        self.arg.var_id()
    }
}

/// Convert a value into the exact staged marker required by a safe extern
/// parameter.
///
/// Ordinary staged expressions pass through unchanged. Borrowing a mutable
/// reference variable creates a single-use occurrence for this call, so the
/// original unique handle can be reborrowed by a later, sequenced call.
pub trait IntoExternArg<T: StagedType> {
    type Staged: Staged<Out = T>;

    fn into_extern_arg(self) -> Self::Staged;
}

impl<A, T> IntoExternArg<T> for A
where
    A: Staged<Out = T>,
    T: StagedType,
{
    type Staged = A;

    fn into_extern_arg(self) -> Self::Staged {
        self
    }
}

impl<'stage, T> IntoExternArg<SRefMut<'stage, T>> for &mut Var<SRefMut<'stage, T>>
where
    T: StagedType + 'stage,
{
    type Staged = VarUse<SRefMut<'stage, T>>;

    fn into_extern_arg(self) -> Self::Staged {
        self.use_once()
    }
}

impl<'stage, T> IntoExternArg<SRef<'stage, T>> for &mut Var<SRefMut<'stage, T>>
where
    T: StagedType + 'stage,
{
    type Staged = ExternArgCast<VarUse<SRefMut<'stage, T>>, SRef<'stage, T>>;

    fn into_extern_arg(self) -> Self::Staged {
        ExternArgCast {
            arg: self.use_once(),
            _expected: PhantomData,
        }
    }
}

impl<'stage, T> IntoExternArg<SRefMut<'stage, Slice<T>>> for &mut Var<SRefMut<'stage, Slice<T>>>
where
    T: StagedType + 'stage,
{
    type Staged = VarUse<SRefMut<'stage, Slice<T>>>;

    fn into_extern_arg(self) -> Self::Staged {
        self.use_once()
    }
}

impl<'stage, T> IntoExternArg<SRef<'stage, Slice<T>>> for &mut Var<SRefMut<'stage, Slice<T>>>
where
    T: StagedType + 'stage,
{
    type Staged = ExternArgCast<VarUse<SRefMut<'stage, Slice<T>>>, SRef<'stage, Slice<T>>>;

    fn into_extern_arg(self) -> Self::Staged {
        ExternArgCast {
            arg: self.use_once(),
            _expected: PhantomData,
        }
    }
}

/// ABI compatibility accepted only by `call_externN_unchecked`.
///
/// # Safety
///
/// `Self` and `Expected` must lower to identical argument ABI values. This
/// witnesses representation only; the unchecked call's caller must still
/// establish pointer validity, lifetime, and aliasing for any reference target.
#[doc(hidden)]
pub unsafe trait UncheckedExternArg<Expected: StagedType>: StagedType {}

unsafe impl<T: StagedType> UncheckedExternArg<T> for T {}

unsafe impl<'stage, T: StagedType> UncheckedExternArg<SRef<'stage, T>> for SPtr<T> {}
unsafe impl<'stage, T: StagedType> UncheckedExternArg<SRef<'stage, T>> for SMutPtr<T> {}
unsafe impl<'stage, T: StagedType> UncheckedExternArg<SRefMut<'stage, T>> for SMutPtr<T> {}
unsafe impl<'stage, T: StagedType> UncheckedExternArg<SRef<'stage, T>> for SRefMut<'stage, T> {}

unsafe impl<'stage, T: StagedType> UncheckedExternArg<SRef<'stage, Slice<T>>> for FatSliceType<T> {}
unsafe impl<'stage, T: StagedType> UncheckedExternArg<SRef<'stage, Slice<T>>>
    for FatSliceMutType<T>
{
}
unsafe impl<'stage, T: StagedType> UncheckedExternArg<SRefMut<'stage, Slice<T>>>
    for FatSliceMutType<T>
{
}
unsafe impl<'stage, T: StagedType> UncheckedExternArg<SRef<'stage, Slice<T>>>
    for SRefMut<'stage, Slice<T>>
{
}

/// Call an external function with 0 arguments. `Out` is the function's own
/// return type (`S::Ret`), so callers never restate it.
pub struct CallExtern0<S: ExternFn> {
    func: ExternRef<S>,
}

unsafe impl<S> Staged for CallExtern0<S>
where
    S: ExternFn<Args = ()>,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        emit_extern_call::<S::Ret>(ctx, func_ref, Vec::new())
    }
}

/// Create a call to an external function with 0 arguments
pub fn call_extern0<S: SafeExternFn<Args = ()>>(func: ExternRef<S>) -> CallExtern0<S> {
    CallExtern0 { func }
}

/// Stage a call to an unsafe zero-argument external function.
///
/// # Safety
///
/// The caller must uphold the target function's safety contract when the
/// generated call executes.
pub unsafe fn call_extern0_unchecked<S: ExternFn<Args = ()>>(func: ExternRef<S>) -> CallExtern0<S> {
    CallExtern0 { func }
}

/// Call an external function with 1 argument. `Out` is `S::Ret`.
pub struct CallExtern1<S: ExternFn, A> {
    func: ExternRef<S>,
    arg: A,
}

unsafe impl<S, A, AType> Staged for CallExtern1<S, A>
where
    S: ExternFn<Args = (AType,)>,
    A: Staged,
    A::Out: UncheckedExternArg<AType>,
    AType: StagedType,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);

        let mut args = Vec::new();
        push_extern_arg::<_, AType>(ctx, &mut args, &self.arg);

        emit_extern_call::<S::Ret>(ctx, func_ref, args)
    }
}

/// Create a call to a safe external function with one exactly typed argument.
///
/// Functions with a different arity are rejected at compile time:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// #[extern_fn]
/// extern "C" fn two_args(_: i64, _: i64) {}
///
/// fn main() {
///     let mut compiler = Compiler::new();
///     let function = compiler.extern_fn::<TwoArgsExtern>();
///     let _ = call_extern1(function, Const::<i64>::new(1));
/// }
/// ```
///
/// Same-shaped ABI values with different Rust types are also rejected:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// #[extern_fn]
/// extern "C" fn takes_u64(_: u64) {}
///
/// fn main() {
///     let mut compiler = Compiler::new();
///     let function = compiler.extern_fn::<TakesU64Extern>();
///     let _ = call_extern1(function, Const::<i64>::new(1));
/// }
/// ```
///
/// Unsafe callbacks require the explicit unchecked constructor:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// #[extern_fn]
/// unsafe extern "C" fn read_pointer(_: *const i64) -> i64 { 0 }
///
/// fn main() {
///     let mut compiler = Compiler::new();
///     let function = compiler.extern_fn::<ReadPointerExtern>();
///     let pointer = const_ptr::<i64>(std::ptr::null());
///     let _ = call_extern1(function, pointer);
/// }
/// ```
///
/// A raw pointer cannot satisfy a safe reference parameter:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// #[extern_fn]
/// extern "C" fn read_borrow(value: &i64) -> i64 { *value }
///
/// fn main() {
///     let mut compiler = Compiler::new();
///     let function = compiler.extern_fn::<ReadBorrowExtern>();
///     let pointer = const_ptr::<Opaque<i64>>(std::ptr::null());
///     let _ = call_extern1(function, pointer);
/// }
/// ```
///
/// Rust slice references retain their reference metadata, but are excluded
/// from the safe extern path because their C ABI is not stable:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// #[allow(improper_ctypes_definitions)]
/// #[extern_fn]
/// extern "C" fn slice_len(value: &[i64]) -> usize { value.len() }
///
/// fn require_safe<S: SafeExternFn>() {}
///
/// fn main() {
///     require_safe::<SliceLenExtern>();
/// }
/// ```
pub fn call_extern1<S, A, AType>(func: ExternRef<S>, arg: A) -> CallExtern1<S, A::Staged>
where
    S: SafeExternFn<Args = (AType,)>,
    A: IntoExternArg<AType>,
    AType: StagedType,
{
    CallExtern1 {
        func,
        arg: arg.into_extern_arg(),
    }
}

/// Stage a call to an unsafe one-argument external function.
///
/// # Safety
///
/// The caller must uphold the target function's safety contract when the
/// generated call executes. When the staged argument marker differs from the
/// declared extern marker, the caller must additionally establish the pointer
/// validity, lifetime, and aliasing required by the declared marker.
pub unsafe fn call_extern1_unchecked<S, A, AType>(func: ExternRef<S>, arg: A) -> CallExtern1<S, A>
where
    S: ExternFn<Args = (AType,)>,
    A: Staged,
    A::Out: UncheckedExternArg<AType>,
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

unsafe impl<S, A, B, AType, BType> Staged for CallExtern2<S, A, B>
where
    S: ExternFn<Args = (AType, BType)>,
    A: Staged,
    B: Staged,
    A::Out: UncheckedExternArg<AType>,
    B::Out: UncheckedExternArg<BType>,
    AType: StagedType,
    BType: StagedType,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);

        let mut args = Vec::new();
        push_extern_arg::<_, AType>(ctx, &mut args, &self.arg0);
        push_extern_arg::<_, BType>(ctx, &mut args, &self.arg1);

        emit_extern_call::<S::Ret>(ctx, func_ref, args)
    }
}

/// Create a call to an external function with 2 arguments.
///
/// A staged mutable reference is reborrowed for the call. Rust therefore
/// rejects using the same unique handle for two mutable parameters:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// #[extern_fn]
/// extern "C" fn two_mut(_: &mut i64, _: &mut i64) {}
///
/// let mut compiler = Compiler::new();
/// let function = compiler.extern_fn::<TwoMutExtern>();
/// let _kernel = compiler.fun1(
///     "aliased",
///     |_ctx, mut value: Var<SRefMut<Opaque<i64>>>| {
///         call_extern2(function, &mut value, &mut value)
///     },
/// );
/// ```
pub fn call_extern2<S, A, B, AType, BType>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
) -> CallExtern2<S, A::Staged, B::Staged>
where
    S: SafeExternFn<Args = (AType, BType)>,
    A: IntoExternArg<AType>,
    B: IntoExternArg<BType>,
    AType: StagedType,
    BType: StagedType,
{
    CallExtern2 {
        func,
        arg0: arg0.into_extern_arg(),
        arg1: arg1.into_extern_arg(),
    }
}

/// Stage a call to an unsafe two-argument external function.
///
/// # Safety
///
/// The caller must uphold the target function's safety contract when the
/// generated call executes. When a staged argument marker differs from its
/// declared extern marker, the caller must additionally establish the pointer
/// validity, lifetime, and aliasing required by the declared marker.
pub unsafe fn call_extern2_unchecked<S, A, B, AType, BType>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
) -> CallExtern2<S, A, B>
where
    S: ExternFn<Args = (AType, BType)>,
    A: Staged,
    B: Staged,
    A::Out: UncheckedExternArg<AType>,
    B::Out: UncheckedExternArg<BType>,
    AType: StagedType,
    BType: StagedType,
{
    CallExtern2 { func, arg0, arg1 }
}

/// Append a pointer to `arg`'s canonical ABI storage to `args`.
fn push_extern_arg<A, AType>(ctx: &mut CompilationContext, args: &mut Vec<Value>, arg: &A)
where
    A: Staged,
    A::Out: UncheckedExternArg<AType>,
    AType: StagedType,
{
    let arg_value = arg.codegen(ctx);
    push_extern_value::<AType>(ctx, args, arg_value);
}

pub(crate) fn push_extern_value<T: StagedType>(
    ctx: &mut CompilationContext,
    args: &mut Vec<Value>,
    value: Value,
) {
    if T::is_copy_struct() {
        args.push(value);
    } else {
        let stack_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            (T::size_of() as u32).max(1),
            T::align_of().trailing_zeros() as u8,
        ));
        let slot_ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);
        if T::size_of() != 0 {
            ctx.builder
                .ins()
                .store(MemFlags::trusted(), value, slot_ptr, 0);
        }
        args.push(slot_ptr);
    }
}

/// Call a Rust-generated thunk through the canonical storage-pointer ABI.
pub(crate) fn emit_extern_call<Ret: StagedType>(
    ctx: &mut CompilationContext,
    func_ref: cranelift_codegen::ir::FuncRef,
    mut args: Vec<Value>,
) -> Value {
    let stack_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
        StackSlotKind::ExplicitSlot,
        (Ret::size_of() as u32).max(1),
        Ret::align_of().trailing_zeros() as u8,
    ));
    let output_ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);
    args.push(output_ptr);
    ctx.builder.ins().call(func_ref, &args);

    if Ret::is_copy_struct() {
        output_ptr
    } else if Ret::size_of() == 0 {
        ctx.get_unit_value()
    } else {
        ctx.builder
            .ins()
            .load(Ret::cranelift_type(), MemFlags::trusted(), output_ptr, 0)
    }
}

/// Call an external function with 3 arguments. `Out` is `S::Ret`.
pub struct CallExtern3<S: ExternFn, A, B, C> {
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
}

unsafe impl<S, A, B, C, AType, BType, CType> Staged for CallExtern3<S, A, B, C>
where
    S: ExternFn<Args = (AType, BType, CType)>,
    A: Staged,
    B: Staged,
    C: Staged,
    A::Out: UncheckedExternArg<AType>,
    B::Out: UncheckedExternArg<BType>,
    C::Out: UncheckedExternArg<CType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);

        let mut args = Vec::new();
        push_extern_arg::<_, AType>(ctx, &mut args, &self.arg0);
        push_extern_arg::<_, BType>(ctx, &mut args, &self.arg1);
        push_extern_arg::<_, CType>(ctx, &mut args, &self.arg2);

        emit_extern_call::<S::Ret>(ctx, func_ref, args)
    }
}

/// Create a call to an external function with 3 arguments.
pub fn call_extern3<S, A, B, C, AType, BType, CType>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
) -> CallExtern3<S, A::Staged, B::Staged, C::Staged>
where
    S: SafeExternFn<Args = (AType, BType, CType)>,
    A: IntoExternArg<AType>,
    B: IntoExternArg<BType>,
    C: IntoExternArg<CType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
{
    CallExtern3 {
        func,
        arg0: arg0.into_extern_arg(),
        arg1: arg1.into_extern_arg(),
        arg2: arg2.into_extern_arg(),
    }
}

/// Stage a call to an unsafe three-argument external function.
///
/// # Safety
///
/// The caller must uphold the target function's safety contract when the
/// generated call executes. When a staged argument marker differs from its
/// declared extern marker, the caller must additionally establish the pointer
/// validity, lifetime, and aliasing required by the declared marker.
pub unsafe fn call_extern3_unchecked<S, A, B, C, AType, BType, CType>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
) -> CallExtern3<S, A, B, C>
where
    S: ExternFn<Args = (AType, BType, CType)>,
    A: Staged,
    B: Staged,
    C: Staged,
    A::Out: UncheckedExternArg<AType>,
    B::Out: UncheckedExternArg<BType>,
    C::Out: UncheckedExternArg<CType>,
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

unsafe impl<S, A, B, C, D, AType, BType, CType, DType> Staged for CallExtern4<S, A, B, C, D>
where
    S: ExternFn<Args = (AType, BType, CType, DType)>,
    A: Staged,
    B: Staged,
    C: Staged,
    D: Staged,
    A::Out: UncheckedExternArg<AType>,
    B::Out: UncheckedExternArg<BType>,
    C::Out: UncheckedExternArg<CType>,
    D::Out: UncheckedExternArg<DType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
    DType: StagedType,
{
    type Out = S::Ret;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);

        let mut args = Vec::new();
        push_extern_arg::<_, AType>(ctx, &mut args, &self.arg0);
        push_extern_arg::<_, BType>(ctx, &mut args, &self.arg1);
        push_extern_arg::<_, CType>(ctx, &mut args, &self.arg2);
        push_extern_arg::<_, DType>(ctx, &mut args, &self.arg3);

        emit_extern_call::<S::Ret>(ctx, func_ref, args)
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
) -> CallExtern4<S, A::Staged, B::Staged, C::Staged, D::Staged>
where
    S: SafeExternFn<Args = (AType, BType, CType, DType)>,
    A: IntoExternArg<AType>,
    B: IntoExternArg<BType>,
    C: IntoExternArg<CType>,
    D: IntoExternArg<DType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
    DType: StagedType,
{
    CallExtern4 {
        func,
        arg0: arg0.into_extern_arg(),
        arg1: arg1.into_extern_arg(),
        arg2: arg2.into_extern_arg(),
        arg3: arg3.into_extern_arg(),
    }
}

/// Stage a call to an unsafe four-argument external function.
///
/// # Safety
///
/// The caller must uphold the target function's safety contract when the
/// generated call executes. When a staged argument marker differs from its
/// declared extern marker, the caller must additionally establish the pointer
/// validity, lifetime, and aliasing required by the declared marker.
#[allow(clippy::too_many_arguments)]
pub unsafe fn call_extern4_unchecked<S, A, B, C, D, AType, BType, CType, DType>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
    arg3: D,
) -> CallExtern4<S, A, B, C, D>
where
    S: ExternFn<Args = (AType, BType, CType, DType)>,
    A: Staged,
    B: Staged,
    C: Staged,
    D: Staged,
    A::Out: UncheckedExternArg<AType>,
    B::Out: UncheckedExternArg<BType>,
    C::Out: UncheckedExternArg<CType>,
    D::Out: UncheckedExternArg<DType>,
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
