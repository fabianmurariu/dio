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

use crate::staged::{CompilationContext, Staged};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, InstBuilder, Value};

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

/// Call an external function with 0 arguments
pub struct CallExtern0<S: ExternFn, Out: StagedType> {
    func: ExternRef<S>,
    _phantom: PhantomData<Out>,
}

impl<S: ExternFn, Out: StagedType> Staged for CallExtern0<S, Out> {
    type Out = Out;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        let call = ctx.builder.ins().call(func_ref, &[]);
        ctx.builder.inst_results(call)[0]
    }
}

/// Create a call to an external function with 0 arguments
pub fn call_extern0<S, Out>(func: ExternRef<S>) -> CallExtern0<S, Out>
where
    S: ExternFn,
    Out: StagedType,
{
    CallExtern0 {
        func,
        _phantom: PhantomData,
    }
}

/// Call an external function with 1 argument
pub struct CallExtern1<S: ExternFn, A, Out: StagedType> {
    func: ExternRef<S>,
    arg: A,
    _phantom: PhantomData<Out>,
}

impl<S, A, AType, Out> Staged for CallExtern1<S, A, Out>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    AType: StagedType,
    Out: StagedType,
{
    type Out = Out;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        let arg_val = self.arg.codegen(ctx);

        // Handle struct arguments that need multiple ABI values
        let args = if AType::is_copy_struct() && AType::num_abi_values() > 1 {
            // Load multiple i64 values from the struct pointer
            let mut vals = Vec::with_capacity(AType::num_abi_values());
            for i in 0..AType::num_abi_values() {
                let offset = (i * 8) as i32;
                let val = ctx.builder.ins().load(
                    types::I64,
                    cranelift_codegen::ir::MemFlags::trusted(),
                    arg_val,
                    offset,
                );
                vals.push(val);
            }
            vals
        } else {
            vec![arg_val]
        };

        let call = ctx.builder.ins().call(func_ref, &args);

        // Handle struct return values
        if Out::is_copy_struct() && Out::num_abi_values() > 1 {
            // Create stack slot and store return values
            let align_shift = Out::align_of().trailing_zeros() as u8;
            let stack_slot =
                ctx.builder
                    .create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        Out::size_of() as u32,
                        align_shift,
                    ));
            let slot_ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);

            let results = ctx.builder.inst_results(call).to_vec();
            for (i, val) in results.iter().enumerate() {
                let offset = (i * 8) as i32;
                ctx.builder.ins().store(
                    cranelift_codegen::ir::MemFlags::trusted(),
                    *val,
                    slot_ptr,
                    offset,
                );
            }
            slot_ptr
        } else {
            ctx.builder.inst_results(call)[0]
        }
    }
}

/// Create a call to an external function with 1 argument
pub fn call_extern1<S, A, AType, Out>(func: ExternRef<S>, arg: A) -> CallExtern1<S, A, Out>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    AType: StagedType,
    Out: StagedType,
{
    CallExtern1 {
        func,
        arg,
        _phantom: PhantomData,
    }
}

/// Call an external function with 2 arguments
pub struct CallExtern2<S: ExternFn, A, B, Out: StagedType> {
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    _phantom: PhantomData<Out>,
}

impl<S, A, B, AType, BType, Out> Staged for CallExtern2<S, A, B, Out>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    AType: StagedType,
    BType: StagedType,
    Out: StagedType,
{
    type Out = Out;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let func_ref = ctx.get_extern_func_ref(self.func.extern_id);
        let arg0_val = self.arg0.codegen(ctx);
        let arg1_val = self.arg1.codegen(ctx);

        // Collect all ABI values
        let mut args = Vec::new();

        // Handle first argument
        if AType::is_copy_struct() && AType::num_abi_values() > 1 {
            for i in 0..AType::num_abi_values() {
                let offset = (i * 8) as i32;
                let val = ctx.builder.ins().load(
                    types::I64,
                    cranelift_codegen::ir::MemFlags::trusted(),
                    arg0_val,
                    offset,
                );
                args.push(val);
            }
        } else {
            args.push(arg0_val);
        }

        // Handle second argument
        if BType::is_copy_struct() && BType::num_abi_values() > 1 {
            for i in 0..BType::num_abi_values() {
                let offset = (i * 8) as i32;
                let val = ctx.builder.ins().load(
                    types::I64,
                    cranelift_codegen::ir::MemFlags::trusted(),
                    arg1_val,
                    offset,
                );
                args.push(val);
            }
        } else {
            args.push(arg1_val);
        }

        let call = ctx.builder.ins().call(func_ref, &args);

        // Handle struct return values
        if Out::is_copy_struct() && Out::num_abi_values() > 1 {
            let align_shift = Out::align_of().trailing_zeros() as u8;
            let stack_slot =
                ctx.builder
                    .create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        Out::size_of() as u32,
                        align_shift,
                    ));
            let slot_ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);

            let results = ctx.builder.inst_results(call).to_vec();
            for (i, val) in results.iter().enumerate() {
                let offset = (i * 8) as i32;
                ctx.builder.ins().store(
                    cranelift_codegen::ir::MemFlags::trusted(),
                    *val,
                    slot_ptr,
                    offset,
                );
            }
            slot_ptr
        } else {
            ctx.builder.inst_results(call)[0]
        }
    }
}

/// Create a call to an external function with 2 arguments
pub fn call_extern2<S, A, B, AType, BType, Out>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
) -> CallExtern2<S, A, B, Out>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    AType: StagedType,
    BType: StagedType,
    Out: StagedType,
{
    CallExtern2 {
        func,
        arg0,
        arg1,
        _phantom: PhantomData,
    }
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

/// Call an external function with 3 arguments.
pub struct CallExtern3<S: ExternFn, A, B, C, Out: StagedType> {
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
    _phantom: PhantomData<Out>,
}

impl<S, A, B, C, AType, BType, CType, Out> Staged for CallExtern3<S, A, B, C, Out>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    C: Staged<Out = CType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
    Out: StagedType,
{
    type Out = Out;

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

        if Out::is_copy_struct() && Out::num_abi_values() > 1 {
            let align_shift = Out::align_of().trailing_zeros() as u8;
            let stack_slot =
                ctx.builder
                    .create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                        cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                        Out::size_of() as u32,
                        align_shift,
                    ));
            let slot_ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);
            let results = ctx.builder.inst_results(call).to_vec();
            for (i, val) in results.iter().enumerate() {
                let offset = (i * 8) as i32;
                ctx.builder.ins().store(
                    cranelift_codegen::ir::MemFlags::trusted(),
                    *val,
                    slot_ptr,
                    offset,
                );
            }
            slot_ptr
        } else {
            ctx.builder.inst_results(call)[0]
        }
    }
}

/// Create a call to an external function with 3 arguments.
#[allow(clippy::too_many_arguments)]
pub fn call_extern3<S, A, B, C, AType, BType, CType, Out>(
    func: ExternRef<S>,
    arg0: A,
    arg1: B,
    arg2: C,
) -> CallExtern3<S, A, B, C, Out>
where
    S: ExternFn,
    A: Staged<Out = AType>,
    B: Staged<Out = BType>,
    C: Staged<Out = CType>,
    AType: StagedType,
    BType: StagedType,
    CType: StagedType,
    Out: StagedType,
{
    CallExtern3 {
        func,
        arg0,
        arg1,
        arg2,
        _phantom: PhantomData,
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
