//! Option types for staged computations with FFI-safe representation.
//!
//! This module provides:
//! - `COption<T>`: FFI-safe Option with explicit discriminant (`#[repr(C, u64)]`)
//! - `COptionType<T>`: StagedType for `COption<T>`
//! - `OptRefType<T>`: Niche-optimized `Option<&T>` (null = None)
//! - `OptMutRefType<T>`: Niche-optimized `Option<&mut T>` (null = None)
//!
//! # FFI-Safe Options
//!
//! `COption<T>` uses `#[repr(C, u64)]` for predictable layout:
//! - Discriminant at offset 0 (u64): 0 = None, 1 = Some
//! - Value at the first properly aligned offset after the discriminant
//!
//! This allows safe interop with Rust code across FFI boundaries.
//!
//! # Niche-Optimized Reference Options
//!
//! For pointer types, we use niche optimization:
//! - `OptRefType<T>` / `OptMutRefType<T>` are single pointer values
//! - null (0) = None
//! - non-null = Some(pointer)

use crate::func::Ctx;
use crate::refer::{SRef, SRefMut};
use crate::staged::{CompilationContext, IntoStaged, Staged, Value, Var};
use crate::types::{IntCmp, RuntimeParam, RuntimeResult, ScalarType, StagedType};
use std::marker::PhantomData;

// =============================================================================
// COption: FFI-safe Option type
// =============================================================================

/// FFI-safe Option with explicit u64 discriminant for cross-language compatibility.
///
/// Uses `#[repr(C, u64)]` enum:
/// - Discriminant: u64 at offset 0 (0 = None, 1 = Some)
/// - Value: T at the first properly aligned offset after the discriminant
///
/// This has identical memory layout to `struct { tag: u64, value: T }`.
#[repr(C, u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum COption<T> {
    None = 0,
    Some(T),
}

impl<T: Copy> COption<T> {
    /// Returns `true` if the option is a `Some` value.
    pub fn is_some(&self) -> bool {
        matches!(self, COption::Some(_))
    }

    /// Returns `true` if the option is a `None` value.
    pub fn is_none(&self) -> bool {
        matches!(self, COption::None)
    }

    /// Returns the contained value if Some, or the provided default.
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            COption::Some(v) => v,
            COption::None => default,
        }
    }

    /// Returns the contained value if Some.
    /// # Safety
    /// Caller must ensure this is a Some variant.
    pub unsafe fn unwrap_unchecked(self) -> T {
        unsafe {
            match self {
                COption::Some(v) => v,
                COption::None => std::hint::unreachable_unchecked(),
            }
        }
    }
}

impl<T: Copy> From<Option<T>> for COption<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => COption::Some(v),
            None => COption::None,
        }
    }
}

impl<T: Copy> From<COption<T>> for Option<T> {
    fn from(opt: COption<T>) -> Self {
        match opt {
            COption::Some(v) => Some(v),
            COption::None => None,
        }
    }
}

// =============================================================================
// COptionType: StagedType for COption<T>
// =============================================================================

/// Staged type marker for `COption<T>`.
///
/// Represented as a pointer to a stack slot containing:
/// - Offset 0: discriminant (u64)
/// - Offset 8: value (T)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct COptionType<T: StagedType> {
    _phantom: PhantomData<T>,
}

unsafe impl<T: StagedType> StagedType for COptionType<T> {
    type RuntimeValue = COption<T::RuntimeValue>;

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }

    fn is_copy_struct() -> bool {
        true
    }

    fn size_of() -> usize {
        std::mem::size_of::<COption<T::RuntimeValue>>()
    }

    fn align_of() -> usize {
        std::mem::align_of::<COption<T::RuntimeValue>>()
    }
}

impl<T: StagedType> COptionType<T> {
    /// Byte offset of the payload inside the `#[repr(C, u64)] COption<T>` layout:
    /// the 8-byte discriminant rounded up to `T`'s alignment.
    ///
    /// This is the single source of truth for the payload offset. Every codegen
    /// site that loads/stores through a `COption` — including the opaque-iterator
    /// loop in `func.rs`, which cannot see `COption`'s Rust layout directly — must
    /// call this rather than re-deriving `align_up(8, align)` inline, per the
    /// project's "slice/pointer layout lives in exactly one place" invariant.
    pub(crate) fn payload_offset() -> usize {
        let alignment = T::align_of();
        debug_assert!(alignment.is_power_of_two());
        8usize.div_ceil(alignment) * alignment
    }
}

unsafe impl<T: StagedType> RuntimeParam for COptionType<T> {
    type Arg<'call> = COption<T::RuntimeValue>;
}

unsafe impl<T: StagedType> RuntimeResult for COptionType<T> {
    type Output<'call> = COption<T::RuntimeValue>;
}

// =============================================================================
// OptRefType / OptMutRefType: Niche-optimized reference options
// =============================================================================

/// Staged type for `Option<&T>` using niche optimization.
///
/// Single i64 value: null = None, non-null = Some(&T)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptRefType<T: StagedType> {
    _phantom: PhantomData<*const T>,
}

unsafe impl<T: StagedType> StagedType for OptRefType<T> {
    /// Null is `None`; the safe, lifetime-bounded view is
    /// `RuntimeParam::Arg<'call>` / `RuntimeResult::Output<'call>`.
    type RuntimeValue = *const T::RuntimeValue;

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }
}

unsafe impl<T> RuntimeParam for OptRefType<T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = Option<&'call T::RuntimeValue>;
}

unsafe impl<T> RuntimeResult for OptRefType<T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Output<'call> = Option<&'call T::RuntimeValue>;
}

/// Staged type for `Option<&mut T>` using niche optimization.
///
/// Single i64 value: null = None, non-null = Some(&mut T)
///
/// A staged optional mutable reference remains a unique capability:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// fn duplicate(value: Var<OptMutRefType<i64>>) {
///     let first = value;
///     let second = value;
///     let _ = (first, second);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptMutRefType<T: StagedType> {
    _phantom: PhantomData<*mut T>,
}

unsafe impl<T: StagedType> StagedType for OptMutRefType<T> {
    type RuntimeValue = *mut T::RuntimeValue;

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }
}

unsafe impl<T> RuntimeParam for OptMutRefType<T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = Option<&'call mut T::RuntimeValue>;
}

unsafe impl<T> RuntimeResult for OptMutRefType<T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Output<'call> = Option<&'call mut T::RuntimeValue>;
}

// =============================================================================
// Creating COption values
// =============================================================================

/// Expression to create `COption::Some(value)`.
#[derive(Clone)]
pub struct CSome<T: StagedType, E> {
    value: E,
    _phantom: PhantomData<T>,
}

unsafe impl<T: StagedType, E: Staged<Out = T>> Staged for CSome<T, E> {
    type Out = COptionType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Get the inner value
        let value = self.value.codegen(ctx);

        // Allocate stack slot for COption<T>
        let size = COptionType::<T>::size_of() as u32;
        let alignment = COptionType::<T>::align_of();
        let stack_slot = ctx.alloc_stack_slot(size, alignment.trailing_zeros() as u8);

        let ptr = ctx.stack_addr(stack_slot, 0);

        // Store discriminant = 1 (Some)
        let one = ctx.iconst(ScalarType::I64, 1);
        ctx.store(one, ptr, 0);

        let payload_offset = COptionType::<T>::payload_offset() as i64;
        let payload_ptr = ctx.ptr_offset_const(ptr, payload_offset);

        ctx.store_value::<T>(payload_ptr, value);

        Value::scalar(ptr)
    }
}

/// Create a `COption::Some(value)` expression.
pub fn c_some<T: StagedType, E: IntoStaged<T>>(value: E) -> CSome<T, E::Staged> {
    CSome {
        value: value.into_staged(),
        _phantom: PhantomData,
    }
}

/// Expression to create `COption::None`.
#[derive(Clone, Copy)]
pub struct CNone<T: StagedType> {
    _phantom: PhantomData<T>,
}

unsafe impl<T: StagedType> Staged for CNone<T> {
    type Out = COptionType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let size = COptionType::<T>::size_of() as u32;
        let alignment = COptionType::<T>::align_of();
        let stack_slot = ctx.alloc_stack_slot(size, alignment.trailing_zeros() as u8);

        let ptr = ctx.stack_addr(stack_slot, 0);

        // Store discriminant = 0 (None)
        let zero = ctx.iconst(ScalarType::I64, 0);
        ctx.store(zero, ptr, 0);

        Value::scalar(ptr)
    }
}

/// Create a `COption::None` expression.
pub fn c_none<T: StagedType>() -> CNone<T> {
    CNone {
        _phantom: PhantomData,
    }
}

// =============================================================================
// Creating OptRef/OptMutRef values (niche-optimized)
// =============================================================================

/// Expression to create `Some(&value)` for niche-optimized reference option.
#[derive(Clone)]
pub struct OptRefSome<T: StagedType, E> {
    reference: E,
    _phantom: PhantomData<*const T>,
}

unsafe impl<T: StagedType, E: Staged<Out = SRef<T>>> Staged for OptRefSome<T, E> {
    type Out = OptRefType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // The reference is the pointer - just pass it through
        self.reference.codegen(ctx)
    }
}

/// Create an `Option<&T>::Some(ref)` expression.
pub fn opt_ref_some<T: StagedType, E: Staged<Out = SRef<T>>>(reference: E) -> OptRefSome<T, E> {
    OptRefSome {
        reference,
        _phantom: PhantomData,
    }
}

/// Expression to create `None` for niche-optimized reference option.
#[derive(Clone, Copy)]
pub struct OptRefNone<T: StagedType> {
    _phantom: PhantomData<*const T>,
}

unsafe impl<T: StagedType> Staged for OptRefNone<T> {
    type Out = OptRefType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        Value::scalar(ctx.null_ptr())
    }
}

/// Create an `Option<&T>::None` expression.
pub fn opt_ref_none<T: StagedType>() -> OptRefNone<T> {
    OptRefNone {
        _phantom: PhantomData,
    }
}

/// Expression to create `Some(&mut value)` for niche-optimized mutable reference option.
pub struct OptMutRefSome<T: StagedType, E> {
    reference: E,
    _phantom: PhantomData<*mut T>,
}

unsafe impl<T: StagedType, E: Staged<Out = SRefMut<T>>> Staged for OptMutRefSome<T, E> {
    type Out = OptMutRefType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.reference.codegen(ctx)
    }
}

/// Create an `Option<&mut T>::Some(ref)` expression.
pub fn opt_mut_ref_some<T: StagedType, E: Staged<Out = SRefMut<T>>>(
    reference: E,
) -> OptMutRefSome<T, E> {
    OptMutRefSome {
        reference,
        _phantom: PhantomData,
    }
}

/// Expression to create `None` for niche-optimized mutable reference option.
#[derive(Clone, Copy)]
pub struct OptMutRefNone<T: StagedType> {
    _phantom: PhantomData<*mut T>,
}

unsafe impl<T: StagedType> Staged for OptMutRefNone<T> {
    type Out = OptMutRefType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        Value::scalar(ctx.null_ptr())
    }
}

/// Create an `Option<&mut T>::None` expression.
pub fn opt_mut_ref_none<T: StagedType>() -> OptMutRefNone<T> {
    OptMutRefNone {
        _phantom: PhantomData,
    }
}

// =============================================================================
// Querying: is_some, is_none
// =============================================================================

/// Expression to check if a `COption` is `Some`.
#[derive(Clone)]
pub struct IsSome<E> {
    opt: E,
}

unsafe impl<T: StagedType, E: Staged<Out = COptionType<T>>> Staged for IsSome<E> {
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let opt_ptr = self.opt.codegen(ctx);
        // Load discriminant from offset 0
        let discriminant = ctx.load(ScalarType::I64, opt_ptr.leaf(), 0);
        // discriminant != 0
        Value::scalar(ctx.icmp_imm(IntCmp::Ne, discriminant, 0))
    }
}

/// Check if a `COption` is `Some`.
pub fn is_some<T: StagedType, E: Staged<Out = COptionType<T>>>(opt: E) -> IsSome<E> {
    IsSome { opt }
}

/// Expression to check if a `COption` is `None`.
#[derive(Clone)]
pub struct IsNone<E> {
    opt: E,
}

unsafe impl<T: StagedType, E: Staged<Out = COptionType<T>>> Staged for IsNone<E> {
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let opt_ptr = self.opt.codegen(ctx);
        let discriminant = ctx.load(ScalarType::I64, opt_ptr.leaf(), 0);
        // discriminant == 0
        Value::scalar(ctx.icmp_imm(IntCmp::Eq, discriminant, 0))
    }
}

/// Check if a `COption` is `None`.
pub fn is_none<T: StagedType, E: Staged<Out = COptionType<T>>>(opt: E) -> IsNone<E> {
    IsNone { opt }
}

// =============================================================================
// Querying for OptRef/OptMutRef (niche-optimized)
// =============================================================================

/// Expression to check if a niche-optimized reference option is `Some`.
#[derive(Clone)]
pub struct IsRefSome<E> {
    opt: E,
}

unsafe impl<T: StagedType, E: Staged<Out = OptRefType<T>>> Staged for IsRefSome<E> {
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);
        // ptr != null
        Value::scalar(ctx.icmp_imm(IntCmp::Ne, ptr.leaf(), 0))
    }
}

/// Check if an `Option<&T>` is `Some`.
pub fn is_ref_some<T: StagedType, E: Staged<Out = OptRefType<T>>>(opt: E) -> IsRefSome<E> {
    IsRefSome { opt }
}

/// Expression to check if a niche-optimized reference option is `None`.
#[derive(Clone)]
pub struct IsRefNone<E> {
    opt: E,
}

unsafe impl<T: StagedType, E: Staged<Out = OptRefType<T>>> Staged for IsRefNone<E> {
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);
        // ptr == null
        Value::scalar(ctx.icmp_imm(IntCmp::Eq, ptr.leaf(), 0))
    }
}

/// Check if an `Option<&T>` is `None`.
pub fn is_ref_none<T: StagedType, E: Staged<Out = OptRefType<T>>>(opt: E) -> IsRefNone<E> {
    IsRefNone { opt }
}

// Similar for OptMutRef
/// Check if an `Option<&mut T>` is `Some`.
#[derive(Clone)]
pub struct IsMutRefSome<E> {
    opt: E,
}

unsafe impl<T: StagedType, E: Staged<Out = OptMutRefType<T>>> Staged for IsMutRefSome<E> {
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);
        Value::scalar(ctx.icmp_imm(IntCmp::Ne, ptr.leaf(), 0))
    }
}

pub fn is_mut_ref_some<T: StagedType, E: Staged<Out = OptMutRefType<T>>>(
    opt: E,
) -> IsMutRefSome<E> {
    IsMutRefSome { opt }
}

/// Check if an `Option<&mut T>` is `None`.
#[derive(Clone)]
pub struct IsMutRefNone<E> {
    opt: E,
}

unsafe impl<T: StagedType, E: Staged<Out = OptMutRefType<T>>> Staged for IsMutRefNone<E> {
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);
        Value::scalar(ctx.icmp_imm(IntCmp::Eq, ptr.leaf(), 0))
    }
}

pub fn is_mut_ref_none<T: StagedType, E: Staged<Out = OptMutRefType<T>>>(
    opt: E,
) -> IsMutRefNone<E> {
    IsMutRefNone { opt }
}

// =============================================================================
// UnwrapOr: Extract value with default
// =============================================================================

/// Expression to unwrap a `COption` with a default value.
#[derive(Clone)]
pub struct UnwrapOr<E, D, T: StagedType> {
    opt: E,
    default: D,
    _phantom: PhantomData<T>,
}

unsafe impl<T: StagedType, E: Staged<Out = COptionType<T>>, D: Staged<Out = T>> Staged
    for UnwrapOr<E, D, T>
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let opt_ptr = self.opt.codegen(ctx);

        // Load discriminant
        let discriminant = ctx.load(ScalarType::I64, opt_ptr.leaf(), 0);

        // Create blocks for if-then-else
        let some_block = ctx.create_block();
        let none_block = ctx.create_block();
        let merge_block = ctx.create_block();

        ctx.append_value_block_params::<T>(merge_block);

        // Branch: if discriminant != 0, go to some_block, else none_block
        ctx.brif(discriminant, some_block, &[], none_block, &[]);

        let payload_offset = COptionType::<T>::payload_offset() as i64;

        // Some block: load the aligned payload.
        ctx.switch_to_block(some_block);
        ctx.seal_block(some_block);
        let payload_ptr = ctx.ptr_offset_const(opt_ptr.leaf(), payload_offset);
        let some_val = ctx.load_value::<T>(payload_ptr);
        ctx.jump_value(merge_block, some_val);

        // None block: use default
        ctx.switch_to_block(none_block);
        ctx.seal_block(none_block);
        let default_val = self.default.codegen(ctx);
        ctx.jump_value(merge_block, default_val);

        // Merge block
        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);

        ctx.block_value::<T>(merge_block)
    }
}

/// Unwrap a `COption` with a default value.
pub fn unwrap_or<T: StagedType, E: Staged<Out = COptionType<T>>, D: IntoStaged<T>>(
    opt: E,
    default: D,
) -> UnwrapOr<E, D::Staged, T> {
    UnwrapOr {
        opt,
        default: default.into_staged(),
        _phantom: PhantomData,
    }
}

// =============================================================================
// MatchOpt: Pattern matching with variable binding
// =============================================================================

/// Expression for pattern matching on `COption` with variable binding.
///
/// Similar to Rust's `match opt { Some(x) => ..., None => ... }`.
pub struct MatchOpt<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
{
    opt: OPT,
    some_body: SomeBody,
    none_body: NoneBody,
    /// Variable ID for the bound value in some_body
    bound_var_id: usize,
    _phantom: PhantomData<(T, OUT)>,
}

unsafe impl<T, OUT, OPT, SomeBody, NoneBody> Staged for MatchOpt<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = COptionType<T>>,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let opt_ptr = self.opt.codegen(ctx);

        // Load discriminant
        let discriminant = ctx.load(ScalarType::I64, opt_ptr.leaf(), 0);

        // Create blocks
        let some_block = ctx.create_block();
        let none_block = ctx.create_block();
        let merge_block = ctx.create_block();

        ctx.append_value_block_params::<OUT>(merge_block);

        // Branch based on discriminant
        ctx.brif(discriminant, some_block, &[], none_block, &[]);

        // Some block: bind value and execute some_body
        ctx.switch_to_block(some_block);
        ctx.seal_block(some_block);

        let payload_offset = COptionType::<T>::payload_offset() as i64;

        let payload_ptr = ctx.ptr_offset_const(opt_ptr.leaf(), payload_offset);
        let bound_val = ctx.load_value::<T>(payload_ptr);
        ctx.assign_var::<T>(self.bound_var_id, bound_val, false);

        let some_result = self.some_body.codegen(ctx);
        ctx.jump_value(merge_block, some_result);

        // None block: execute none_body
        ctx.switch_to_block(none_block);
        ctx.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.jump_value(merge_block, none_result);

        // Merge block
        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);

        ctx.block_value::<OUT>(merge_block)
    }
}

/// Pattern match on a `COption`, binding the value in the Some branch.
///
/// The `some_fn` closure receives a `Ctx` context and a `Var<T>` bound
/// to the unwrapped value, similar to how `fun1` works.
///
/// # Example
/// ```ignore
/// let result = match_opt(
///     &mut compiler,
///     some_option_expr,
///     |ctx, val| add(val, 1i64),  // Some(x) => x + 1
///     Const::<i64>::new(0),   // None => 0
/// );
/// ```
pub fn match_opt<T, OUT, OPT, SomeFn, SomeBody, NoneBody>(
    ctx: &mut Ctx,
    opt: OPT,
    some_fn: SomeFn,
    none_body: NoneBody,
) -> MatchOpt<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = COptionType<T>>,
    SomeFn: FnOnce(&mut Ctx, Var<T>) -> SomeBody,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    // Allocate variable for bound value
    let bound_var: Var<T> = unsafe { ctx.var_unchecked() };
    let bound_var_id = bound_var.id;

    // Build the some_body by calling the closure
    let some_body = some_fn(ctx, bound_var);

    MatchOpt {
        opt,
        some_body,
        none_body,
        bound_var_id,
        _phantom: PhantomData,
    }
}

// =============================================================================
// MatchOptRef: Pattern matching for niche-optimized reference options
// =============================================================================

/// Pattern match on an `Option<&T>` (niche-optimized).
pub struct MatchOptRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
{
    opt: OPT,
    some_body: SomeBody,
    none_body: NoneBody,
    bound_var_id: usize,
    _phantom: PhantomData<(T, OUT)>,
}

unsafe impl<T, OUT, OPT, SomeBody, NoneBody> Staged for MatchOptRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = OptRefType<T>>,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);

        let some_block = ctx.create_block();
        let none_block = ctx.create_block();
        let merge_block = ctx.create_block();

        ctx.append_value_block_params::<OUT>(merge_block);

        // Branch: if ptr != null, it's Some
        ctx.brif(ptr.leaf(), some_block, &[], none_block, &[]);

        // Some block: ptr IS the reference
        ctx.switch_to_block(some_block);
        ctx.seal_block(some_block);

        ctx.assign_var::<SRef<T>>(self.bound_var_id, ptr, false);

        let some_result = self.some_body.codegen(ctx);
        ctx.jump_value(merge_block, some_result);

        // None block
        ctx.switch_to_block(none_block);
        ctx.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.jump_value(merge_block, none_result);

        // Merge
        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);

        ctx.block_value::<OUT>(merge_block)
    }
}

/// Pattern match on an `Option<&T>`.
pub fn match_opt_ref<T, OUT, OPT, SomeFn, SomeBody, NoneBody>(
    ctx: &mut Ctx,
    opt: OPT,
    some_fn: SomeFn,
    none_body: NoneBody,
) -> MatchOptRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = OptRefType<T>>,
    SomeFn: FnOnce(&mut Ctx, Var<SRef<T>>) -> SomeBody,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    let bound_var: Var<SRef<T>> = unsafe { ctx.var_unchecked() };
    let bound_var_id = bound_var.id;
    let some_body = some_fn(ctx, bound_var);

    MatchOptRef {
        opt,
        some_body,
        none_body,
        bound_var_id,
        _phantom: PhantomData,
    }
}

/// Pattern match on an `Option<&mut T>`.
pub struct MatchOptMutRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
{
    opt: OPT,
    some_body: SomeBody,
    none_body: NoneBody,
    bound_var_id: usize,
    _phantom: PhantomData<(T, OUT)>,
}

unsafe impl<T, OUT, OPT, SomeBody, NoneBody> Staged
    for MatchOptMutRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = OptMutRefType<T>>,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);

        let some_block = ctx.create_block();
        let none_block = ctx.create_block();
        let merge_block = ctx.create_block();

        ctx.append_value_block_params::<OUT>(merge_block);

        ctx.brif(ptr.leaf(), some_block, &[], none_block, &[]);

        ctx.switch_to_block(some_block);
        ctx.seal_block(some_block);

        ctx.assign_var::<SRefMut<T>>(self.bound_var_id, ptr, false);

        let some_result = self.some_body.codegen(ctx);
        ctx.jump_value(merge_block, some_result);

        ctx.switch_to_block(none_block);
        ctx.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.jump_value(merge_block, none_result);

        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);

        ctx.block_value::<OUT>(merge_block)
    }
}

/// Pattern match on an `Option<&mut T>`.
pub fn match_opt_mut_ref<T, OUT, OPT, SomeFn, SomeBody, NoneBody>(
    ctx: &mut Ctx,
    opt: OPT,
    some_fn: SomeFn,
    none_body: NoneBody,
) -> MatchOptMutRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = OptMutRefType<T>>,
    SomeFn: FnOnce(&mut Ctx, Var<SRefMut<T>>) -> SomeBody,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    let bound_var: Var<SRefMut<T>> = unsafe { ctx.var_unchecked() };
    let bound_var_id = bound_var.id;
    let some_body = some_fn(ctx, bound_var);

    MatchOptMutRef {
        opt,
        some_body,
        none_body,
        bound_var_id,
        _phantom: PhantomData,
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[repr(C, align(16))]
    #[derive(Clone, Copy)]
    struct AlignedPayload([u8; 16]);

    unsafe impl StagedType for AlignedPayload {
        type RuntimeValue = Self;

        fn scalar_type() -> ScalarType {
            ScalarType::Ptr
        }

        fn size_of() -> usize {
            std::mem::size_of::<Self>()
        }

        fn align_of() -> usize {
            std::mem::align_of::<Self>()
        }

        fn is_copy_struct() -> bool {
            true
        }
    }
    #[test]
    fn c_option_respects_overaligned_payload_layout() {
        assert_eq!(COptionType::<AlignedPayload>::payload_offset(), 16);
        assert_eq!(COptionType::<AlignedPayload>::size_of(), 32);
        assert_eq!(COptionType::<AlignedPayload>::align_of(), 16);
        assert_eq!(
            COptionType::<AlignedPayload>::size_of(),
            std::mem::size_of::<COption<AlignedPayload>>()
        );
        assert_eq!(
            COptionType::<AlignedPayload>::align_of(),
            std::mem::align_of::<COption<AlignedPayload>>()
        );

        let option = COption::Some(AlignedPayload([0x5a; 16]));
        let bytes = &option as *const COption<AlignedPayload> as *const u8;
        // SAFETY: the payload offset and its first byte are within `option`.
        assert_eq!(unsafe { *bytes.add(16) }, 0x5a);
    }
}
