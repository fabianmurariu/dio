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
//! `COption<T>` uses `#[repr(C, u64)]` for predictable ABI:
//! - Discriminant at offset 0 (u64): 0 = None, 1 = Some
//! - Value at offset 8: T
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
use crate::staged::{CompilationContext, IntoStaged, Staged, Var};
use crate::types::{BoolType, StagedType};
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value};
use std::marker::PhantomData;

// =============================================================================
// COption: FFI-safe Option type
// =============================================================================

/// FFI-safe Option with explicit discriminant for cross-language compatibility.
///
/// Uses `#[repr(C, u64)]` for predictable layout:
/// - Offset 0: discriminant (u64) - 0 = None, 1 = Some
/// - Offset 8: value (T)
#[repr(C, u64)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum COption<T> {
    None = 0,
    Some(T),
}

impl<T> From<Option<T>> for COption<T> {
    fn from(opt: Option<T>) -> Self {
        match opt {
            Some(v) => COption::Some(v),
            None => COption::None,
        }
    }
}

impl<T> From<COption<T>> for Option<T> {
    fn from(opt: COption<T>) -> Self {
        match opt {
            COption::Some(v) => Some(v),
            COption::None => None,
        }
    }
}

impl<T> COption<T> {
    /// Returns `true` if the option is a `Some` value.
    pub fn is_some(&self) -> bool {
        matches!(self, COption::Some(_))
    }

    /// Returns `true` if the option is a `None` value.
    pub fn is_none(&self) -> bool {
        matches!(self, COption::None)
    }

    /// Returns the contained `Some` value, or a provided default.
    pub fn unwrap_or(self, default: T) -> T {
        match self {
            COption::Some(v) => v,
            COption::None => default,
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

impl<T: StagedType> StagedType for COptionType<T> {
    type RuntimeValue<'a> = COption<T::RuntimeValue<'a>>;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Pointer to stack slot
    }

    fn is_copy_struct() -> bool {
        true
    }

    fn size_of() -> usize {
        // u64 discriminant + T (with alignment)
        8 + T::size_of()
    }

    fn align_of() -> usize {
        8 // u64 alignment
    }

    fn num_abi_values() -> usize {
        // discriminant + T's abi values
        1 + T::num_abi_values()
    }

    fn abi_types() -> Vec<cranelift_codegen::ir::Type> {
        let mut result = vec![types::I64]; // discriminant
        result.extend(T::abi_types());
        result
    }
}

// =============================================================================
// OptRefType / OptMutRefType: Niche-optimized reference options
// =============================================================================

/// Staged type for `Option<&T>` using niche optimization.
///
/// Single i64 value: null = None, non-null = Some(&T)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptRefType<T: StagedType> {
    _phantom: PhantomData<T>,
}

impl<T: StagedType> StagedType for OptRefType<T> {
    type RuntimeValue<'a> = Option<&'a T::RuntimeValue<'a>>;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Single pointer, null = None
    }
}

/// Staged type for `Option<&mut T>` using niche optimization.
///
/// Single i64 value: null = None, non-null = Some(&mut T)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct OptMutRefType<T: StagedType> {
    _phantom: PhantomData<T>,
}

impl<T: StagedType> StagedType for OptMutRefType<T> {
    type RuntimeValue<'a> = Option<&'a mut T::RuntimeValue<'a>>;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64 // Single pointer, null = None
    }
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

impl<T: StagedType, E: Staged<Out = T>> Staged for CSome<T, E> {
    type Out = COptionType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Get the inner value
        let value = self.value.codegen(ctx);

        // Allocate stack slot for COption<T>
        let size = (8 + T::size_of()) as u32;
        let stack_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            size,
            3, // 8-byte alignment (2^3 = 8)
        ));

        let ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);

        // Store discriminant = 1 (Some)
        let one = ctx.builder.ins().iconst(types::I64, 1);
        ctx.builder.ins().store(MemFlags::trusted(), one, ptr, 0);

        // Store value at offset 8
        if T::is_copy_struct() {
            // For structs, value is a pointer - copy the struct data
            let struct_size = T::size_of();
            for i in 0..(struct_size / 8) {
                let offset = (i * 8) as i32;
                let val = ctx
                    .builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), value, offset);
                ctx.builder
                    .ins()
                    .store(MemFlags::trusted(), val, ptr, 8 + offset);
            }
        } else {
            // For primitives, store directly
            ctx.builder.ins().store(MemFlags::trusted(), value, ptr, 8);
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

impl<T: StagedType> Staged for CNone<T> {
    type Out = COptionType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let size = (8 + T::size_of()) as u32;
        let stack_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            size,
            3,
        ));

        let ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);

        // Store discriminant = 0 (None)
        let zero = ctx.builder.ins().iconst(types::I64, 0);
        ctx.builder.ins().store(MemFlags::trusted(), zero, ptr, 0);

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
pub struct OptRefSome<T: StagedType, E> {
    reference: E,
    _phantom: PhantomData<T>,
}

impl<T: StagedType, E: Staged<Out = SRef<T>>> Staged for OptRefSome<T, E> {
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
    _phantom: PhantomData<T>,
}

impl<T: StagedType> Staged for OptRefNone<T> {
    type Out = OptRefType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // None is represented as null pointer
        ctx.builder.ins().iconst(types::I64, 0)
    }
}

/// Create an `Option<&T>::None` expression.
pub fn opt_ref_none<T: StagedType>() -> OptRefNone<T> {
    OptRefNone {
        _phantom: PhantomData,
    }
}

/// Expression to create `Some(&mut value)` for niche-optimized mutable reference option.
#[derive(Clone)]
pub struct OptMutRefSome<T: StagedType, E> {
    reference: E,
    _phantom: PhantomData<T>,
}

impl<T: StagedType, E: Staged<Out = SRefMut<T>>> Staged for OptMutRefSome<T, E> {
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
    _phantom: PhantomData<T>,
}

impl<T: StagedType> Staged for OptMutRefNone<T> {
    type Out = OptMutRefType<T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        ctx.builder.ins().iconst(types::I64, 0)
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

impl<T: StagedType, E: Staged<Out = COptionType<T>>> Staged for IsSome<E> {
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let opt_ptr = self.opt.codegen(ctx);
        // Load discriminant from offset 0
        let discriminant = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), opt_ptr, 0);
        // discriminant != 0
        ctx.builder.ins().icmp_imm(
            cranelift_codegen::ir::condcodes::IntCC::NotEqual,
            discriminant,
            0,
        )
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

impl<T: StagedType, E: Staged<Out = COptionType<T>>> Staged for IsNone<E> {
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let opt_ptr = self.opt.codegen(ctx);
        let discriminant = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), opt_ptr, 0);
        // discriminant == 0
        ctx.builder.ins().icmp_imm(
            cranelift_codegen::ir::condcodes::IntCC::Equal,
            discriminant,
            0,
        )
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

impl<T: StagedType, E: Staged<Out = OptRefType<T>>> Staged for IsRefSome<E> {
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);
        // ptr != null
        ctx.builder.ins().icmp_imm(
            cranelift_codegen::ir::condcodes::IntCC::NotEqual,
            ptr,
            0,
        )
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

impl<T: StagedType, E: Staged<Out = OptRefType<T>>> Staged for IsRefNone<E> {
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);
        // ptr == null
        ctx.builder
            .ins()
            .icmp_imm(cranelift_codegen::ir::condcodes::IntCC::Equal, ptr, 0)
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

impl<T: StagedType, E: Staged<Out = OptMutRefType<T>>> Staged for IsMutRefSome<E> {
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);
        ctx.builder.ins().icmp_imm(
            cranelift_codegen::ir::condcodes::IntCC::NotEqual,
            ptr,
            0,
        )
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

impl<T: StagedType, E: Staged<Out = OptMutRefType<T>>> Staged for IsMutRefNone<E> {
    type Out = BoolType;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let ptr = self.opt.codegen(ctx);
        ctx.builder
            .ins()
            .icmp_imm(cranelift_codegen::ir::condcodes::IntCC::Equal, ptr, 0)
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

impl<T: StagedType, E: Staged<Out = COptionType<T>>, D: Staged<Out = T>> Staged
    for UnwrapOr<E, D, T>
{
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        use cranelift_codegen::ir::BlockArg;

        let opt_ptr = self.opt.codegen(ctx);

        // Load discriminant
        let discriminant = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), opt_ptr, 0);

        // Create blocks for if-then-else
        let some_block = ctx.builder.create_block();
        let none_block = ctx.builder.create_block();
        let merge_block = ctx.builder.create_block();

        // Add block parameter for the result
        let result_type = T::cranelift_type();
        ctx.builder.append_block_param(merge_block, result_type);

        // Branch: if discriminant != 0, go to some_block, else none_block
        ctx.builder.ins().brif(
            discriminant,
            some_block,
            &[],
            none_block,
            &[],
        );

        // Some block: load value from offset 8
        ctx.builder.switch_to_block(some_block);
        ctx.builder.seal_block(some_block);
        let some_val = if T::is_copy_struct() {
            // For structs, return pointer to the value (offset 8 from opt_ptr)
            ctx.builder.ins().iadd_imm(opt_ptr, 8)
        } else {
            ctx.builder
                .ins()
                .load(T::cranelift_type(), MemFlags::trusted(), opt_ptr, 8)
        };
        ctx.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(some_val)]);

        // None block: use default
        ctx.builder.switch_to_block(none_block);
        ctx.builder.seal_block(none_block);
        let default_val = self.default.codegen(ctx);
        ctx.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(default_val)]);

        // Merge block
        ctx.builder.switch_to_block(merge_block);
        ctx.builder.seal_block(merge_block);

        ctx.builder.block_params(merge_block)[0]
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
pub struct MatchOpt<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
{
    opt: OPT,
    some_body: SOME_BODY,
    none_body: NONE_BODY,
    /// Variable ID for the bound value in some_body
    bound_var_id: usize,
    _phantom: PhantomData<(T, OUT)>,
}

impl<T, OUT, OPT, SOME_BODY, NONE_BODY> Staged for MatchOpt<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = COptionType<T>>,
    SOME_BODY: Staged<Out = OUT>,
    NONE_BODY: Staged<Out = OUT>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        use cranelift_codegen::ir::BlockArg;

        let opt_ptr = self.opt.codegen(ctx);

        // Load discriminant
        let discriminant = ctx
            .builder
            .ins()
            .load(types::I64, MemFlags::trusted(), opt_ptr, 0);

        // Create blocks
        let some_block = ctx.builder.create_block();
        let none_block = ctx.builder.create_block();
        let merge_block = ctx.builder.create_block();

        let result_type = OUT::cranelift_type();
        ctx.builder.append_block_param(merge_block, result_type);

        // Branch based on discriminant
        ctx.builder
            .ins()
            .brif(discriminant, some_block, &[], none_block, &[]);

        // Some block: bind value and execute some_body
        ctx.builder.switch_to_block(some_block);
        ctx.builder.seal_block(some_block);

        // Load the value and bind it to the variable
        let bound_val = if T::is_copy_struct() {
            // For structs, the variable holds a pointer to the value
            ctx.builder.ins().iadd_imm(opt_ptr, 8)
        } else {
            ctx.builder
                .ins()
                .load(T::cranelift_type(), MemFlags::trusted(), opt_ptr, 8)
        };

        // Declare and define the bound variable
        let bound_var = ctx.builder.declare_var(T::cranelift_type());
        ctx.builder.def_var(bound_var, bound_val);
        ctx.var_map.insert(self.bound_var_id, bound_var);

        let some_result = self.some_body.codegen(ctx);
        ctx.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(some_result)]);

        // None block: execute none_body
        ctx.builder.switch_to_block(none_block);
        ctx.builder.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(none_result)]);

        // Merge block
        ctx.builder.switch_to_block(merge_block);
        ctx.builder.seal_block(merge_block);

        ctx.builder.block_params(merge_block)[0]
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
///     Const::<I64Type>::new(0),   // None => 0
/// );
/// ```
pub fn match_opt<T, OUT, OPT, SOME_FN, SOME_BODY, NONE_BODY>(
    var_builder: &mut VarBuilder,
    opt: OPT,
    some_fn: SOME_FN,
    none_body: NONE_BODY,
) -> MatchOpt<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = COptionType<T>>,
    SOME_FN: FnOnce(&mut VarBuilder, Var<T>) -> SOME_BODY,
    SOME_BODY: Staged<Out = OUT>,
    NONE_BODY: Staged<Out = OUT>,
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
pub struct MatchOptRef<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
{
    opt: OPT,
    some_body: SOME_BODY,
    none_body: NONE_BODY,
    bound_var_id: usize,
    _phantom: PhantomData<(T, OUT)>,
}

impl<T, OUT, OPT, SOME_BODY, NONE_BODY> Staged for MatchOptRef<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = OptRefType<T>>,
    SOME_BODY: Staged<Out = OUT>,
    NONE_BODY: Staged<Out = OUT>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        use cranelift_codegen::ir::BlockArg;

        let ptr = self.opt.codegen(ctx);

        let some_block = ctx.builder.create_block();
        let none_block = ctx.builder.create_block();
        let merge_block = ctx.builder.create_block();

        let result_type = OUT::cranelift_type();
        ctx.builder.append_block_param(merge_block, result_type);

        // Branch: if ptr != null, it's Some
        ctx.builder
            .ins()
            .brif(ptr, some_block, &[], none_block, &[]);

        // Some block: ptr IS the reference
        ctx.builder.switch_to_block(some_block);
        ctx.builder.seal_block(some_block);

        // Bind the pointer as SRef<T>
        let bound_var = ctx.builder.declare_var(types::I64);
        ctx.builder.def_var(bound_var, ptr);
        ctx.var_map.insert(self.bound_var_id, bound_var);

        let some_result = self.some_body.codegen(ctx);
        ctx.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(some_result)]);

        // None block
        ctx.builder.switch_to_block(none_block);
        ctx.builder.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(none_result)]);

        // Merge
        ctx.builder.switch_to_block(merge_block);
        ctx.builder.seal_block(merge_block);

        ctx.builder.block_params(merge_block)[0]
    }
}

/// Pattern match on an `Option<&T>`.
pub fn match_opt_ref<T, OUT, OPT, SOME_FN, SOME_BODY, NONE_BODY>(
    var_builder: &mut VarBuilder,
    opt: OPT,
    some_fn: SOME_FN,
    none_body: NONE_BODY,
) -> MatchOptRef<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = OptRefType<T>>,
    SOME_FN: FnOnce(&mut VarBuilder, Var<SRef<T>>) -> SOME_BODY,
    SOME_BODY: Staged<Out = OUT>,
    NONE_BODY: Staged<Out = OUT>,
{
    let bound_var: Var<SRef<T>> = unsafe { var_builder.var_unchecked() };
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
pub struct MatchOptMutRef<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
{
    opt: OPT,
    some_body: SOME_BODY,
    none_body: NONE_BODY,
    bound_var_id: usize,
    _phantom: PhantomData<(T, OUT)>,
}

impl<T, OUT, OPT, SOME_BODY, NONE_BODY> Staged
    for MatchOptMutRef<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = OptMutRefType<T>>,
    SOME_BODY: Staged<Out = OUT>,
    NONE_BODY: Staged<Out = OUT>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        use cranelift_codegen::ir::BlockArg;

        let ptr = self.opt.codegen(ctx);

        let some_block = ctx.builder.create_block();
        let none_block = ctx.builder.create_block();
        let merge_block = ctx.builder.create_block();

        let result_type = OUT::cranelift_type();
        ctx.builder.append_block_param(merge_block, result_type);

        ctx.builder
            .ins()
            .brif(ptr, some_block, &[], none_block, &[]);

        ctx.builder.switch_to_block(some_block);
        ctx.builder.seal_block(some_block);

        let bound_var = ctx.builder.declare_var(types::I64);
        ctx.builder.def_var(bound_var, ptr);
        ctx.var_map.insert(self.bound_var_id, bound_var);

        let some_result = self.some_body.codegen(ctx);
        ctx.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(some_result)]);

        ctx.builder.switch_to_block(none_block);
        ctx.builder.seal_block(none_block);
        let none_result = self.none_body.codegen(ctx);
        ctx.builder
            .ins()
            .jump(merge_block, &[BlockArg::Value(none_result)]);

        ctx.builder.switch_to_block(merge_block);
        ctx.builder.seal_block(merge_block);

        ctx.builder.block_params(merge_block)[0]
    }
}

/// Pattern match on an `Option<&mut T>`.
pub fn match_opt_mut_ref<T, OUT, OPT, SOME_FN, SOME_BODY, NONE_BODY>(
    var_builder: &mut VarBuilder,
    opt: OPT,
    some_fn: SOME_FN,
    none_body: NONE_BODY,
) -> MatchOptMutRef<T, OUT, OPT, SOME_BODY, NONE_BODY>
where
    T: StagedType,
    OUT: StagedType,
    OPT: Staged<Out = OptMutRefType<T>>,
    SOME_FN: FnOnce(&mut VarBuilder, Var<SRefMut<T>>) -> SOME_BODY,
    SOME_BODY: Staged<Out = OUT>,
    NONE_BODY: Staged<Out = OUT>,
{
    let bound_var: Var<SRefMut<T>> = unsafe { var_builder.var_unchecked() };
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
    fn test_c_some_i64() {
        let compiler = Compiler::new();

        // Create COption::Some(42)
        let expr = c_some::<I64Type, _>(42i64);
        let wrapped = unwrap_or(expr, 0i64);

        let compiled = compiler.compile(wrapped).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_c_none_i64() {
        let compiler = Compiler::new();

        // Create COption::None, unwrap_or should return default
        let expr = c_none::<I64Type>();
        let wrapped = unwrap_or(expr, 99i64);

        let compiled = compiler.compile(wrapped).expect("compilation failed");
        assert_eq!(compiled.run(), 99);
    }

    #[test]
    fn test_is_some() {
        let compiler = Compiler::new();

        let some_expr = c_some::<I64Type, _>(42i64);
        let check = is_some(some_expr);

        let compiled = compiler.compile(check).expect("compilation failed");
        assert_eq!(compiled.run(), true);
    }

    #[test]
    fn test_is_none() {
        let compiler = Compiler::new();

        let none_expr = c_none::<I64Type>();
        let check = is_none(none_expr);

        let compiled = compiler.compile(check).expect("compilation failed");
        assert_eq!(compiled.run(), true);
    }

    #[test]
    fn test_match_opt_some() {
        let mut compiler = Compiler::new();

        // match Some(10) { Some(x) => x + 5, None => 0 }
        let func = compiler.fun1("test", |ctx, _dummy: Var<I64Type>| {
            let opt = c_some::<I64Type, _>(10i64);
            match_opt(ctx, opt, |_ctx, val| add(val, 5i64), Const::<I64Type>::new(0))
        });

        let compiled = compiler.compile(call1(func, 0i64)).expect("compilation failed");
        assert_eq!(compiled.run(), 15);
    }

    #[test]
    fn test_match_opt_none() {
        let mut compiler = Compiler::new();

        // match None { Some(x) => x + 5, None => 99 }
        let func = compiler.fun1("test", |ctx, _dummy: Var<I64Type>| {
            let opt = c_none::<I64Type>();
            match_opt(ctx, opt, |_ctx, val| add(val, 5i64), Const::<I64Type>::new(99))
        });

        let compiled = compiler.compile(call1(func, 0i64)).expect("compilation failed");
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
}
