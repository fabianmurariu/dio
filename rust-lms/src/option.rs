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
//! - `OptRefType<T>` / `OptMutRefType<T>` are single i64 values
//! - null (0) = None
//! - non-null = Some(pointer)

use crate::func::VarBuilder;
use crate::refer::{SRef, SRefMut};
use crate::staged::{CompilationContext, IntoStaged, Staged, ValueId, Var};
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
        match self {
            COption::Some(v) => v,
            COption::None => std::hint::unreachable_unchecked(),
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
pub struct OptRefType<'a, T: StagedType> {
    _phantom: PhantomData<&'a T>,
}

unsafe impl<'a, T: StagedType> StagedType for OptRefType<'a, T> {
    type RuntimeValue = Option<&'a T::RuntimeValue>;

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }
}

unsafe impl<'stage, T> RuntimeParam for OptRefType<'stage, T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = Option<&'call T::RuntimeValue>;
}

unsafe impl<'stage, T> RuntimeResult for OptRefType<'stage, T>
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
/// fn duplicate(value: Var<OptMutRefType<'static, i64>>) {
///     let first = value;
///     let second = value;
///     let _ = (first, second);
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptMutRefType<'a, T: StagedType> {
    _phantom: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: StagedType> StagedType for OptMutRefType<'a, T> {
    type RuntimeValue = Option<&'a mut T::RuntimeValue>;

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }
}

unsafe impl<'stage, T> RuntimeParam for OptMutRefType<'stage, T>
where
    T: StagedType,
    T::RuntimeValue: 'static,
{
    type Arg<'call> = Option<&'call mut T::RuntimeValue>;
}

unsafe impl<'stage, T> RuntimeResult for OptMutRefType<'stage, T>
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

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
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

        // Store the payload at its actual aligned offset.
        if T::is_copy_struct() {
            // Aggregate staged values are addresses of their storage.
            ctx.copy_nonoverlapping(payload_ptr, value, T::size_of(), T::align_of());
        } else {
            ctx.store(value, payload_ptr, 0);
        }

        ptr
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

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let size = COptionType::<T>::size_of() as u32;
        let alignment = COptionType::<T>::align_of();
        let stack_slot = ctx.alloc_stack_slot(size, alignment.trailing_zeros() as u8);

        let ptr = ctx.stack_addr(stack_slot, 0);

        // Store discriminant = 0 (None)
        let zero = ctx.iconst(ScalarType::I64, 0);
        ctx.store(zero, ptr, 0);

        ptr
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
pub struct OptRefSome<'a, T: StagedType, E> {
    reference: E,
    _phantom: PhantomData<&'a T>,
}

unsafe impl<'a, T: StagedType, E: Staged<Out = SRef<'a, T>>> Staged for OptRefSome<'a, T, E> {
    type Out = OptRefType<'a, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        // The reference is the pointer - just pass it through
        self.reference.codegen(ctx)
    }
}

/// Create an `Option<&T>::Some(ref)` expression.
pub fn opt_ref_some<'a, T: StagedType, E: Staged<Out = SRef<'a, T>>>(
    reference: E,
) -> OptRefSome<'a, T, E> {
    OptRefSome {
        reference,
        _phantom: PhantomData,
    }
}

/// Expression to create `None` for niche-optimized reference option.
#[derive(Clone, Copy)]
pub struct OptRefNone<'a, T: StagedType> {
    _phantom: PhantomData<&'a T>,
}

unsafe impl<'a, T: StagedType> Staged for OptRefNone<'a, T> {
    type Out = OptRefType<'a, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        // None is represented as null pointer
        ctx.iconst(ScalarType::I64, 0)
    }
}

/// Create an `Option<&T>::None` expression.
pub fn opt_ref_none<'a, T: StagedType>() -> OptRefNone<'a, T> {
    OptRefNone {
        _phantom: PhantomData,
    }
}

/// Expression to create `Some(&mut value)` for niche-optimized mutable reference option.
pub struct OptMutRefSome<'a, T: StagedType, E> {
    reference: E,
    _phantom: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: StagedType, E: Staged<Out = SRefMut<'a, T>>> Staged for OptMutRefSome<'a, T, E> {
    type Out = OptMutRefType<'a, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        self.reference.codegen(ctx)
    }
}

/// Create an `Option<&mut T>::Some(ref)` expression.
pub fn opt_mut_ref_some<'a, T: StagedType, E: Staged<Out = SRefMut<'a, T>>>(
    reference: E,
) -> OptMutRefSome<'a, T, E> {
    OptMutRefSome {
        reference,
        _phantom: PhantomData,
    }
}

/// Expression to create `None` for niche-optimized mutable reference option.
#[derive(Clone, Copy)]
pub struct OptMutRefNone<'a, T: StagedType> {
    _phantom: PhantomData<&'a mut T>,
}

unsafe impl<'a, T: StagedType> Staged for OptMutRefNone<'a, T> {
    type Out = OptMutRefType<'a, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        ctx.iconst(ScalarType::I64, 0)
    }
}

/// Create an `Option<&mut T>::None` expression.
pub fn opt_mut_ref_none<'a, T: StagedType>() -> OptMutRefNone<'a, T> {
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

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let opt_ptr = self.opt.codegen(ctx);
        // Load discriminant from offset 0
        let discriminant = ctx.load(ScalarType::I64, opt_ptr, 0);
        // discriminant != 0
        ctx.icmp_imm(IntCmp::Ne, discriminant, 0)
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

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let opt_ptr = self.opt.codegen(ctx);
        let discriminant = ctx.load(ScalarType::I64, opt_ptr, 0);
        // discriminant == 0
        ctx.icmp_imm(IntCmp::Eq, discriminant, 0)
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

unsafe impl<'a, T: StagedType + 'a, E: Staged<Out = OptRefType<'a, T>>> Staged for IsRefSome<E> {
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.opt.codegen(ctx);
        // ptr != null
        ctx.icmp_imm(IntCmp::Ne, ptr, 0)
    }
}

/// Check if an `Option<&T>` is `Some`.
pub fn is_ref_some<'a, T: StagedType + 'a, E: Staged<Out = OptRefType<'a, T>>>(
    opt: E,
) -> IsRefSome<E> {
    IsRefSome { opt }
}

/// Expression to check if a niche-optimized reference option is `None`.
#[derive(Clone)]
pub struct IsRefNone<E> {
    opt: E,
}

unsafe impl<'a, T: StagedType + 'a, E: Staged<Out = OptRefType<'a, T>>> Staged for IsRefNone<E> {
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.opt.codegen(ctx);
        // ptr == null
        ctx.icmp_imm(IntCmp::Eq, ptr, 0)
    }
}

/// Check if an `Option<&T>` is `None`.
pub fn is_ref_none<'a, T: StagedType + 'a, E: Staged<Out = OptRefType<'a, T>>>(
    opt: E,
) -> IsRefNone<E> {
    IsRefNone { opt }
}

// Similar for OptMutRef
/// Check if an `Option<&mut T>` is `Some`.
#[derive(Clone)]
pub struct IsMutRefSome<E> {
    opt: E,
}

unsafe impl<'a, T: StagedType + 'a, E: Staged<Out = OptMutRefType<'a, T>>> Staged
    for IsMutRefSome<E>
{
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.opt.codegen(ctx);
        ctx.icmp_imm(IntCmp::Ne, ptr, 0)
    }
}

pub fn is_mut_ref_some<'a, T: StagedType + 'a, E: Staged<Out = OptMutRefType<'a, T>>>(
    opt: E,
) -> IsMutRefSome<E> {
    IsMutRefSome { opt }
}

/// Check if an `Option<&mut T>` is `None`.
#[derive(Clone)]
pub struct IsMutRefNone<E> {
    opt: E,
}

unsafe impl<'a, T: StagedType + 'a, E: Staged<Out = OptMutRefType<'a, T>>> Staged
    for IsMutRefNone<E>
{
    type Out = bool;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.opt.codegen(ctx);
        ctx.icmp_imm(IntCmp::Eq, ptr, 0)
    }
}

pub fn is_mut_ref_none<'a, T: StagedType + 'a, E: Staged<Out = OptMutRefType<'a, T>>>(
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

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let opt_ptr = self.opt.codegen(ctx);

        // Load discriminant
        let discriminant = ctx.load(ScalarType::I64, opt_ptr, 0);

        // Create blocks for if-then-else
        let some_block = ctx.create_block();
        let none_block = ctx.create_block();
        let merge_block = ctx.create_block();

        // Add block parameter for the result
        let result_type = T::scalar_type();
        ctx.append_block_param(merge_block, result_type);

        // Branch: if discriminant != 0, go to some_block, else none_block
        ctx.brif(discriminant, some_block, &[], none_block, &[]);

        let payload_offset = COptionType::<T>::payload_offset() as i64;

        // Some block: load the aligned payload.
        ctx.switch_to_block(some_block);
        ctx.seal_block(some_block);
        let some_val = if T::is_copy_struct() {
            ctx.ptr_offset_const(opt_ptr, payload_offset)
        } else {
            let payload_ptr = ctx.ptr_offset_const(opt_ptr, payload_offset);
            ctx.load(T::scalar_type(), payload_ptr, 0)
        };
        ctx.jump(merge_block, &[some_val]);

        // None block: use default
        ctx.switch_to_block(none_block);
        ctx.seal_block(none_block);
        let default_val = self.default.codegen(ctx);
        ctx.jump(merge_block, &[default_val]);

        // Merge block
        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);

        ctx.block_param(merge_block, 0)
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

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let opt_ptr = self.opt.codegen(ctx);

        // Load discriminant
        let discriminant = ctx.load(ScalarType::I64, opt_ptr, 0);

        // Create blocks
        let some_block = ctx.create_block();
        let none_block = ctx.create_block();
        let merge_block = ctx.create_block();

        let result_type = OUT::scalar_type();
        ctx.append_block_param(merge_block, result_type);

        // Branch based on discriminant
        ctx.brif(discriminant, some_block, &[], none_block, &[]);

        // Some block: bind value and execute some_body
        ctx.switch_to_block(some_block);
        ctx.seal_block(some_block);

        let payload_offset = COptionType::<T>::payload_offset() as i64;

        // Load the value and bind it to the variable.
        let bound_val = if T::is_copy_struct() {
            ctx.ptr_offset_const(opt_ptr, payload_offset)
        } else {
            let payload_ptr = ctx.ptr_offset_const(opt_ptr, payload_offset);
            ctx.load(T::scalar_type(), payload_ptr, 0)
        };

        // Declare and define the bound variable
        let bound_var = ctx.declare_var(T::scalar_type());
        ctx.def_var(bound_var, bound_val);
        ctx.var_map.insert(self.bound_var_id, bound_var);

        let some_result = self.some_body.codegen(ctx);
        ctx.jump(merge_block, &[some_result]);

        // None block: execute none_body
        ctx.switch_to_block(none_block);
        ctx.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.jump(merge_block, &[none_result]);

        // Merge block
        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);

        ctx.block_param(merge_block, 0)
    }
}

/// Pattern match on a `COption`, binding the value in the Some branch.
///
/// The `some_fn` closure receives a `VarBuilder` context and a `Var<T>` bound
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
    var_builder: &mut VarBuilder,
    opt: OPT,
    some_fn: SomeFn,
    none_body: NoneBody,
) -> MatchOpt<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = COptionType<T>>,
    SomeFn: FnOnce(&mut VarBuilder, Var<T>) -> SomeBody,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    // Allocate variable for bound value
    let bound_var: Var<T> = unsafe { var_builder.var_unchecked() };
    let bound_var_id = bound_var.id;

    // Build the some_body by calling the closure
    let some_body = some_fn(var_builder, bound_var);

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

unsafe impl<'a, T, OUT, OPT, SomeBody, NoneBody> Staged
    for MatchOptRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType + 'a,
    OUT: StagedType,
    OPT: Staged<Out = OptRefType<'a, T>>,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.opt.codegen(ctx);

        let some_block = ctx.create_block();
        let none_block = ctx.create_block();
        let merge_block = ctx.create_block();

        let result_type = OUT::scalar_type();
        ctx.append_block_param(merge_block, result_type);

        // Branch: if ptr != null, it's Some
        ctx.brif(ptr, some_block, &[], none_block, &[]);

        // Some block: ptr IS the reference
        ctx.switch_to_block(some_block);
        ctx.seal_block(some_block);

        // Bind the pointer as SRef<T>
        let bound_var = ctx.declare_var(ScalarType::I64);
        ctx.def_var(bound_var, ptr);
        ctx.var_map.insert(self.bound_var_id, bound_var);

        let some_result = self.some_body.codegen(ctx);
        ctx.jump(merge_block, &[some_result]);

        // None block
        ctx.switch_to_block(none_block);
        ctx.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.jump(merge_block, &[none_result]);

        // Merge
        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);

        ctx.block_param(merge_block, 0)
    }
}

/// Pattern match on an `Option<&T>`.
pub fn match_opt_ref<'a, T, OUT, OPT, SomeFn, SomeBody, NoneBody>(
    var_builder: &mut VarBuilder,
    opt: OPT,
    some_fn: SomeFn,
    none_body: NoneBody,
) -> MatchOptRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType + 'a,
    OUT: StagedType,
    OPT: Staged<Out = OptRefType<'a, T>>,
    SomeFn: FnOnce(&mut VarBuilder, Var<SRef<'a, T>>) -> SomeBody,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    let bound_var: Var<SRef<'a, T>> = unsafe { var_builder.var_unchecked() };
    let bound_var_id = bound_var.id;
    let some_body = some_fn(var_builder, bound_var);

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

unsafe impl<'a, T, OUT, OPT, SomeBody, NoneBody> Staged
    for MatchOptMutRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType + 'a,
    OUT: StagedType,
    OPT: Staged<Out = OptMutRefType<'a, T>>,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> ValueId {
        let ptr = self.opt.codegen(ctx);

        let some_block = ctx.create_block();
        let none_block = ctx.create_block();
        let merge_block = ctx.create_block();

        let result_type = OUT::scalar_type();
        ctx.append_block_param(merge_block, result_type);

        ctx.brif(ptr, some_block, &[], none_block, &[]);

        ctx.switch_to_block(some_block);
        ctx.seal_block(some_block);

        let bound_var = ctx.declare_var(ScalarType::I64);
        ctx.def_var(bound_var, ptr);
        ctx.var_map.insert(self.bound_var_id, bound_var);

        let some_result = self.some_body.codegen(ctx);
        ctx.jump(merge_block, &[some_result]);

        ctx.switch_to_block(none_block);
        ctx.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.jump(merge_block, &[none_result]);

        ctx.switch_to_block(merge_block);
        ctx.seal_block(merge_block);

        ctx.block_param(merge_block, 0)
    }
}

/// Pattern match on an `Option<&mut T>`.
pub fn match_opt_mut_ref<'a, T, OUT, OPT, SomeFn, SomeBody, NoneBody>(
    var_builder: &mut VarBuilder,
    opt: OPT,
    some_fn: SomeFn,
    none_body: NoneBody,
) -> MatchOptMutRef<T, OUT, OPT, SomeBody, NoneBody>
where
    T: StagedType + 'a,
    OUT: StagedType,
    OPT: Staged<Out = OptMutRefType<'a, T>>,
    SomeFn: FnOnce(&mut VarBuilder, Var<SRefMut<'a, T>>) -> SomeBody,
    SomeBody: Staged<Out = OUT>,
    NoneBody: Staged<Out = OUT>,
{
    let bound_var: Var<SRefMut<'a, T>> = unsafe { var_builder.var_unchecked() };
    let bound_var_id = bound_var.id;
    let some_body = some_fn(var_builder, bound_var);

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
    use crate::func::Compiler;
    use crate::num::add;
    use crate::prelude::*;

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
    fn test_c_option_layout() {
        // Verify layout matches our assumptions
        assert_eq!(std::mem::size_of::<COption<i64>>(), 16);
        assert_eq!(std::mem::align_of::<COption<i64>>(), 8);

        // Verify discriminant values
        let none: COption<i64> = COption::None;
        let some: COption<i64> = COption::Some(42);

        // Check that discriminant is at offset 0
        let none_ptr = &none as *const COption<i64> as *const u64;
        let some_ptr = &some as *const COption<i64> as *const u64;

        unsafe {
            assert_eq!(*none_ptr, 0); // None discriminant
            assert_eq!(*some_ptr, 1); // Some discriminant
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

    #[test]
    fn test_c_some_i64() {
        let compiler = Compiler::new();

        // Create COption::Some(42)
        let expr = c_some::<i64, _>(42i64);
        let wrapped = unwrap_or(expr, 0i64);

        let compiled = compiler.compile(wrapped).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_c_none_i64() {
        let compiler = Compiler::new();

        // Create COption::None, unwrap_or should return default
        let expr = c_none::<i64>();
        let wrapped = unwrap_or(expr, 99i64);

        let compiled = compiler.compile(wrapped).expect("compilation failed");
        assert_eq!(compiled.run(), 99);
    }

    #[test]
    fn test_is_some() {
        let compiler = Compiler::new();

        let some_expr = c_some::<i64, _>(42i64);
        let check = is_some(some_expr);

        let compiled = compiler.compile(check).expect("compilation failed");
        assert_eq!(compiled.run(), true);
    }

    #[test]
    fn test_is_none() {
        let compiler = Compiler::new();

        let none_expr = c_none::<i64>();
        let check = is_none(none_expr);

        let compiled = compiler.compile(check).expect("compilation failed");
        assert_eq!(compiled.run(), true);
    }

    #[test]
    fn test_match_opt_some() {
        let mut compiler = Compiler::new();

        // match Some(10) { Some(x) => x + 5, None => 0 }
        let func = compiler.fun1("test", |ctx, _dummy: Var<i64>| {
            let opt = c_some::<i64, _>(10i64);
            match_opt(ctx, opt, |_ctx, val| add(val, 5i64), Const::<i64>::new(0))
        });

        let compiled = compiler
            .compile(call1(func, 0i64))
            .expect("compilation failed");
        assert_eq!(compiled.run(), 15);
    }

    #[test]
    fn test_match_opt_none() {
        let mut compiler = Compiler::new();

        // match None { Some(x) => x + 5, None => 99 }
        let func = compiler.fun1("test", |ctx, _dummy: Var<i64>| {
            let opt = c_none::<i64>();
            match_opt(ctx, opt, |_ctx, val| add(val, 5i64), Const::<i64>::new(99))
        });

        let compiled = compiler
            .compile(call1(func, 0i64))
            .expect("compilation failed");
        assert_eq!(compiled.run(), 99);
    }

    #[test]
    fn test_coption_from_option() {
        let some: COption<i64> = Some(42).into();
        assert_eq!(some, COption::Some(42));

        let none: COption<i64> = None.into();
        assert_eq!(none, COption::None);
    }

    #[test]
    fn test_option_from_coption() {
        let some: Option<i64> = COption::Some(42).into();
        assert_eq!(some, Some(42));

        let none: Option<i64> = COption::<i64>::None.into();
        assert_eq!(none, None);
    }

    // =========================================================================
    // Function Pointer Tests: COption<i64>
    // =========================================================================

    #[test]
    fn test_fn_taking_coption_i64() {
        let mut compiler = Compiler::new();

        // fn unwrap_or_default(opt: COption<i64>) -> i64
        let unwrap_fn = compiler.fun1("unwrap_or_default", |_ctx, opt: Var<COptionType<i64>>| {
            unwrap_or(opt, -1i64)
        });

        let compiled = compiler.compile(unwrap_fn).expect("compilation failed");
        let f = compiled.as_fn();

        // Test with Some
        assert_eq!(f.call(COption::Some(42)), 42);
        assert_eq!(f.call(COption::Some(0)), 0);
        assert_eq!(f.call(COption::Some(-100)), -100);

        // Test with None
        assert_eq!(f.call(COption::None), -1);
    }

    #[test]
    fn test_fn_returning_coption_i64() {
        let mut compiler = Compiler::new();

        // fn maybe_double(x: i64) -> COption<i64>
        // Returns Some(x * 2) if x > 0, else None
        let maybe_double = compiler.fun1("maybe_double", |ctx, x: Var<i64>| {
            let doubled = c_some::<i64, _>(mul(x, 2i64));
            let none = c_none::<i64>();
            // if x > 0 then Some(x*2) else None
            match_opt(
                ctx,
                if_then_else(lt(0i64, x), doubled, none),
                |_ctx, val| c_some::<i64, _>(val),
                c_none::<i64>(),
            )
        });

        let compiled = compiler.compile(maybe_double).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(5), COption::Some(10));
        assert_eq!(f.call(1), COption::Some(2));
        assert_eq!(f.call(0), COption::None);
        assert_eq!(f.call(-5), COption::None);
    }

    #[test]
    fn test_fn_coption_i64_roundtrip() {
        let mut compiler = Compiler::new();

        // fn add_one_if_some(opt: COption<i64>) -> COption<i64>
        let add_one = compiler.fun1("add_one_if_some", |ctx, opt: Var<COptionType<i64>>| {
            match_opt(
                ctx,
                opt,
                |_ctx, val| c_some::<i64, _>(add(val, 1i64)),
                c_none::<i64>(),
            )
        });

        let compiled = compiler.compile(add_one).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(COption::Some(10)), COption::Some(11));
        assert_eq!(f.call(COption::Some(-1)), COption::Some(0));
        assert_eq!(f.call(COption::None), COption::None);
    }

    // =========================================================================
    // Function Pointer Tests: COption<f64>
    // =========================================================================

    #[test]
    fn test_c_option_f64_layout() {
        // Verify layout for f64 variant
        assert_eq!(std::mem::size_of::<COption<f64>>(), 16);
        assert_eq!(std::mem::align_of::<COption<f64>>(), 8);

        let some: COption<f64> = COption::Some(3.15);
        let ptr = &some as *const COption<f64> as *const u8;
        unsafe {
            // Discriminant at offset 0
            let disc = *(ptr as *const u64);
            assert_eq!(disc, 1, "discriminant should be 1 for Some");
            // Value at offset 8
            let val = *((ptr.add(8)) as *const f64);
            assert_eq!(val, 3.15, "value should be 3.15");
        }
    }

    #[test]
    fn test_fn_taking_coption_f64() {
        let mut compiler = Compiler::new();

        // fn unwrap_or_zero(opt: COption<f64>) -> f64
        let unwrap_fn = compiler.fun1("unwrap_or_zero", |_ctx, opt: Var<COptionType<f64>>| {
            unwrap_or(opt, 0.0f64)
        });

        let compiled = compiler.compile(unwrap_fn).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(COption::Some(3.15)), 3.15);
        assert_eq!(f.call(COption::Some(-2.5)), -2.5);
        assert_eq!(f.call(COption::None), 0.0);
    }

    #[test]
    fn test_fn_returning_coption_f64() {
        let mut compiler = Compiler::new();

        // fn wrap_f64(x: f64) -> COption<f64>
        // Always returns Some(x)
        let wrap = compiler.fun1("wrap_f64", |_ctx, x: Var<f64>| c_some::<f64, _>(x));

        let compiled = compiler.compile(wrap).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(3.15), COption::Some(3.15));
        assert_eq!(f.call(0.0), COption::Some(0.0));
        assert_eq!(f.call(-1.5), COption::Some(-1.5));
    }

    // =========================================================================
    // Function Pointer Tests: OptRefType (Option<&T>)
    // =========================================================================

    #[test]
    fn test_fn_taking_opt_ref_i64() {
        let mut compiler = Compiler::new();

        // fn deref_or_default(opt: Option<&i64>) -> i64
        let deref_fn = compiler.fun1("deref_or_default", |ctx, opt: Var<OptRefType<i64>>| {
            use crate::refer::load_ref;
            // if some, load the value; else return -1
            match_opt_ref(ctx, opt, |_ctx, ptr| load_ref(ptr), Const::<i64>::new(-1))
        });

        let compiled = compiler.compile(deref_fn).expect("compilation failed");
        let f = compiled.as_fn();

        let val = 42i64;
        assert_eq!(f.call(Some(&val)), 42);

        let val2 = -100i64;
        assert_eq!(f.call(Some(&val2)), -100);

        assert_eq!(f.call(None), -1);
    }

    #[test]
    fn test_fn_returning_opt_ref_i64() {
        let mut compiler = Compiler::new();

        // fn make_ref(ptr: &i64) -> Option<&i64>
        // Just wraps the reference in Some
        let make_ref = compiler.fun1("make_ref", |_ctx, ptr: Var<SRef<i64>>| {
            opt_ref_some::<i64, _>(ptr)
        });

        let compiled = compiler.compile(make_ref).expect("compilation failed");
        let f = compiled.as_fn();

        let val = 99i64;
        let result = f.call(&val);
        assert_eq!(result, Some(&99i64));
    }

    #[test]
    fn test_fn_opt_ref_conditional() {
        let mut compiler = Compiler::new();

        // fn ref_if_positive(ptr: &i64) -> Option<&i64>
        // Returns Some(ptr) if *ptr > 0, else None
        let ref_if_pos = compiler.fun1("ref_if_positive", |_ctx, ptr: Var<SRef<i64>>| {
            use crate::refer::load_ref;
            let val = load_ref(ptr);
            if_then_else(
                lt(0i64, val), // val > 0
                opt_ref_some::<i64, _>(ptr),
                opt_ref_none::<i64>(),
            )
        });

        let compiled = compiler.compile(ref_if_pos).expect("compilation failed");
        let f = compiled.as_fn();

        let pos = 42i64;
        assert_eq!(f.call(&pos), Some(&42i64));

        let zero = 0i64;
        assert_eq!(f.call(&zero), None);

        let neg = -10i64;
        assert_eq!(f.call(&neg), None);
    }

    // =========================================================================
    // Function Pointer Tests: OptMutRefType (Option<&mut T>)
    // =========================================================================

    #[test]
    fn test_fn_taking_opt_mut_ref_i64() {
        let mut compiler = Compiler::new();

        // fn read_and_double(opt: Option<&mut i64>) -> i64
        // If Some, reads the value and returns it doubled (without mutating); else returns -1
        let read_fn = compiler.fun1("read_and_double", |ctx, opt: Var<OptMutRefType<i64>>| {
            use crate::refer::load_ref_mut;
            match_opt_mut_ref(
                ctx,
                opt,
                |_ctx, mut ptr| mul(load_ref_mut(&mut ptr), 2i64),
                Const::<i64>::new(-1),
            )
        });

        let compiled = compiler.compile(read_fn).expect("compilation failed");
        let f = compiled.as_fn();

        // Test with Some - just reading, not mutating
        let mut val = 21i64;
        assert_eq!(f.call(Some(&mut val)), 42);

        let mut val2 = 5i64;
        assert_eq!(f.call(Some(&mut val2)), 10);

        // None case
        assert_eq!(f.call(None), -1);
    }

    #[test]
    fn test_fn_mutating_opt_mut_ref_i64() {
        let mut compiler = Compiler::new();

        // fn increment_in_place(opt: Option<&mut i64>) -> i64
        // If Some, increments the value in place and returns new value; else returns -1
        let incr_fn = compiler.fun1("increment_in_place", |ctx, opt: Var<OptMutRefType<i64>>| {
            use crate::refer::{load_ref_mut, store_ref};
            // Use a local variable to hold the new value
            let result = ctx.let_var(0i64);
            (
                result,
                match_opt_mut_ref(
                    ctx,
                    opt,
                    |_ctx, mut ptr| {
                        // Load current value, add 1, store back, and assign to result
                        let incremented = add(load_ref_mut(&mut ptr), 1i64);
                        (
                            store_ref(&mut ptr, incremented),
                            assign(*result, load_ref_mut(&mut ptr)),
                        )
                    },
                    assign(*result, -1i64),
                ),
                *result,
            )
        });

        let compiled = compiler.compile(incr_fn).expect("compilation failed");
        let f = compiled.as_fn();

        // Test mutation
        let mut val = 41i64;
        let returned = f.call(Some(&mut val));
        assert_eq!(returned, 42);
        assert_eq!(val, 42);

        // None case
        assert_eq!(f.call(None), -1);
    }

    #[test]
    fn test_fn_returning_opt_mut_ref_i64() {
        let mut compiler = Compiler::new();

        // fn make_mut_ref(ptr: &mut i64) -> Option<&mut i64>
        let make_ref = compiler.fun1("make_mut_ref", |_ctx, ptr: Var<SRefMut<i64>>| {
            opt_mut_ref_some::<i64, _>(ptr)
        });

        let compiled = compiler.compile(make_ref).expect("compilation failed");
        let f = compiled.as_fn();

        let mut val = 99i64;
        let result = f.call(&mut val);
        assert!(result.is_some());
        if let Some(r) = result {
            assert_eq!(*r, 99);
            *r = 100;
        }
        assert_eq!(val, 100);
    }

    // =========================================================================
    // Function Pointer Tests: COption<f64> advanced
    // =========================================================================

    #[test]
    fn test_fn_coption_f64_roundtrip() {
        let mut compiler = Compiler::new();

        // fn square_if_some(opt: COption<f64>) -> COption<f64>
        let square_fn = compiler.fun1("square_if_some", |ctx, opt: Var<COptionType<f64>>| {
            use crate::num::mul;
            match_opt(
                ctx,
                opt,
                |_ctx, val| c_some::<f64, _>(mul(val, val)),
                c_none::<f64>(),
            )
        });

        let compiled = compiler.compile(square_fn).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(COption::Some(3.0)), COption::Some(9.0));
        assert_eq!(f.call(COption::Some(-2.0)), COption::Some(4.0));
        assert_eq!(f.call(COption::None), COption::None);
    }

    // =========================================================================
    // Function Pointer Tests: Multi-argument functions with options
    // =========================================================================

    #[test]
    fn test_fn2_with_coption() {
        let mut compiler = Compiler::new();

        // fn add_options(a: COption<i64>, b: COption<i64>) -> COption<i64>
        // Returns Some(a + b) if both are Some, else None
        let add_opts = compiler.fun2(
            "add_options",
            |ctx, a: Var<COptionType<i64>>, b: Var<COptionType<i64>>| {
                match_opt(
                    ctx,
                    a,
                    |ctx, a_val| {
                        match_opt(
                            ctx,
                            b,
                            |_ctx, b_val| c_some::<i64, _>(add(a_val, b_val)),
                            c_none::<i64>(),
                        )
                    },
                    c_none::<i64>(),
                )
            },
        );

        let compiled = compiler.compile(add_opts).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(
            f.call(COption::Some(10), COption::Some(20)),
            COption::Some(30)
        );
        assert_eq!(f.call(COption::Some(5), COption::None), COption::None);
        assert_eq!(f.call(COption::None, COption::Some(5)), COption::None);
        assert_eq!(f.call(COption::None, COption::None), COption::None);
    }

    #[test]
    fn test_fn2_mixed_option_and_primitive() {
        let mut compiler = Compiler::new();

        // fn unwrap_or_add(opt: COption<i64>, default: i64) -> i64
        let unwrap_add = compiler.fun2(
            "unwrap_or_add",
            |_ctx, opt: Var<COptionType<i64>>, default: Var<i64>| unwrap_or(opt, default),
        );

        let compiled = compiler.compile(unwrap_add).expect("compilation failed");
        let f = compiled.as_fn();

        assert_eq!(f.call(COption::Some(42), 0), 42);
        assert_eq!(f.call(COption::None, 99), 99);
        assert_eq!(f.call(COption::Some(10), 99), 10);
    }
}
