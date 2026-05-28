//! Multi-parameter function support via macro generation.
//!
//! This module provides:
//! - `TypeInfo`: Runtime type information for parameters/returns
//! - `codegen_call`: Single codegen implementation for all function calls
//! - Macro-generated `FunTypeN`, `FunRefN`, `CallN` for N = 0..8

use crate::func::StructInfo;
use crate::staged::{CompilationContext, IntoStaged, Staged};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value};
use cranelift_module::Module;
use std::marker::PhantomData;

// =============================================================================
// TypeInfo: Runtime type information for a parameter or return value
// =============================================================================

/// Runtime type information for a single logical parameter or return value.
///
/// This captures everything needed for ABI handling without static type info.
#[derive(Clone, Debug)]
pub struct TypeInfo {
    /// ABI types for this value (1 for primitives, N for structs ≤16 bytes, 1 pointer for >16 bytes)
    pub abi_types: Vec<cranelift_codegen::ir::Type>,
    /// Struct info if this is a struct, None for primitives
    pub struct_info: Option<StructInfo>,
    /// True if this struct should be passed by pointer (for structs >16 bytes on ARM64)
    pub pass_by_pointer: bool,
    /// True if this is a fat pointer (slice reference) that can be stored in separate registers
    pub is_fat_pointer: bool,
}

impl TypeInfo {
    /// Create TypeInfo from a StagedType
    pub fn from_staged_type<T: StagedType>() -> Self {
        let pass_by_pointer = T::should_pass_by_pointer();

        TypeInfo {
            // For pass-by-pointer structs, we only need one i64 (the pointer)
            abi_types: if pass_by_pointer {
                vec![types::I64]
            } else {
                T::abi_types()
            },
            struct_info: if T::is_copy_struct() {
                Some(StructInfo {
                    size: T::size_of() as u32,
                    alignment: T::align_of() as u32,
                    num_abi_values: if pass_by_pointer { 1 } else { T::num_abi_values() },
                })
            } else {
                None
            },
            pass_by_pointer,
            is_fat_pointer: T::is_fat_pointer(),
        }
    }

    /// Check if this is a struct type
    pub fn is_struct(&self) -> bool {
        self.struct_info.is_some()
    }

    /// Get total number of ABI slots needed
    pub fn num_abi_slots(&self) -> usize {
        self.abi_types.len()
    }
}

// =============================================================================
// Codegen helpers: Single implementation for all function calls
// =============================================================================

/// Generate code for a function call.
///
/// This is THE single codegen implementation for all function calls.
/// Takes:
/// - func_id: The function's ID in the Compiler
/// - param_infos: Type info for each parameter
/// - return_info: Type info for the return value
/// - arg_values: One Cranelift Value per logical parameter (pointer for structs)
pub fn codegen_call(
    ctx: &mut CompilationContext,
    func_id: usize,
    param_infos: &[TypeInfo],
    return_info: &TypeInfo,
    arg_values: &[Value],
) -> Value {
    assert_eq!(
        arg_values.len(),
        param_infos.len(),
        "Argument count mismatch: expected {}, got {}",
        param_infos.len(),
        arg_values.len()
    );

    // Look up the function ID in our map
    let cranelift_func_id = ctx
        .func_map
        .get(&func_id)
        .expect(&format!("Function {} not found in func_map", func_id));

    // Declare the function for calling
    let func_ref = ctx
        .module
        .declare_func_in_func(*cranelift_func_id, ctx.builder.func);

    // Prepare call arguments: expand small structs to multiple i64 values,
    // pass large structs by pointer
    let mut call_args: Vec<Value> = Vec::new();

    for (arg_value, param_info) in arg_values.iter().zip(param_infos.iter()) {
        if let Some(ref struct_info) = param_info.struct_info {
            if param_info.pass_by_pointer {
                // LARGE STRUCT (>16 bytes): Pass pointer directly
                // The caller already has the struct in memory, just pass the pointer
                call_args.push(*arg_value);
            } else {
                // SMALL STRUCT (≤16 bytes): Load multiple i64 values from the pointer
                for i in 0..struct_info.num_abi_values {
                    let offset = (i * 8) as i32;
                    let val =
                        ctx.builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), *arg_value, offset);
                    call_args.push(val);
                }
            }
        } else {
            // PRIMITIVE ARGUMENT: Single value
            call_args.push(*arg_value);
        }
    }

    // Generate the call
    let call = ctx.builder.ins().call(func_ref, &call_args);

    // Handle return value
    if let Some(ref struct_info) = return_info.struct_info {
        if return_info.pass_by_pointer {
            // LARGE STRUCT RETURN (>16 bytes): Function returns a pointer
            // Just return the pointer directly
            ctx.builder.inst_results(call)[0]
        } else {
            // SMALL STRUCT RETURN (≤16 bytes): Collect multiple return values
            let results: Vec<Value> = ctx.builder.inst_results(call).to_vec();

            // Store multiple return values to a stack slot
            let align_shift = struct_info.alignment.trailing_zeros() as u8;

            let stack_slot = ctx.builder.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                struct_info.size,
                align_shift,
            ));

            let slot_ptr = ctx.builder.ins().stack_addr(types::I64, stack_slot, 0);

            // Store each return value to the stack slot
            for (i, &result) in results.iter().enumerate() {
                let offset = (i * 8) as i32;
                ctx.builder
                    .ins()
                    .store(MemFlags::trusted(), result, slot_ptr, offset);
            }

            // Return the pointer to the stack slot
            slot_ptr
        }
    } else {
        // PRIMITIVE RETURN: Single value
        ctx.builder.inst_results(call)[0]
    }
}

/// Generate code to get a function's address (for returning function pointers)
pub fn codegen_func_addr(ctx: &mut CompilationContext, func_id: usize) -> Value {
    let cranelift_func_id = ctx
        .func_map
        .get(&func_id)
        .expect(&format!("Function {} not found in func_map", func_id));

    let func_ref = ctx
        .module
        .declare_func_in_func(*cranelift_func_id, ctx.builder.func);
    ctx.builder.ins().func_addr(types::I64, func_ref)
}

// =============================================================================
// Macro for generating FunTypeN, FunRefN, CallN, callN
// =============================================================================

/// Macro to generate function types for a specific arity.
///
/// Generates:
/// - `FunTypeN<T0, T1, ..., OUT>` - StagedType for function pointers
/// - `FunRefN<T0, T1, ..., OUT>` - Type-safe function reference (Copy, stores just ID)
/// - `CallN<T0, T1, ..., OUT, ARG0, ARG1, ...>` - Call expression
/// - `callN(func, arg0, arg1, ...)` - Helper to create Call expression
macro_rules! impl_fun_n {
    // Base case: zero parameters
    (0, $FunType:ident, $FunRef:ident, $Call:ident, $call_fn:ident) => {
        /// Function type marker for 0-ary functions
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $FunType<OUT> {
            _phantom: PhantomData<OUT>,
        }

        impl<OUT: StagedType> StagedType for $FunType<OUT> {
            type RuntimeValue = extern "C" fn() -> OUT::RuntimeValue;

            fn cranelift_type() -> cranelift_codegen::ir::Type {
                types::I64
            }
        }

        /// Type-safe function reference for 0-ary functions
        #[derive(Clone, Copy)]
        pub struct $FunRef<OUT: StagedType> {
            pub(crate) id: usize,
            _phantom: PhantomData<OUT>,
        }

        impl<OUT: StagedType> $FunRef<OUT> {
            pub fn new(id: usize) -> Self {
                $FunRef {
                    id,
                    _phantom: PhantomData,
                }
            }

            pub fn id(&self) -> usize {
                self.id
            }
        }

        impl<OUT: StagedType> std::fmt::Debug for $FunRef<OUT> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($FunRef), self.id)
            }
        }

        impl<OUT: StagedType> Staged for $FunRef<OUT> {
            type Out = $FunType<OUT>;

            fn codegen(&self, ctx: &mut CompilationContext) -> Value {
                codegen_func_addr(ctx, self.id)
            }
        }

        /// Call expression for 0-ary functions
        #[derive(Clone, Copy)]
        pub struct $Call<OUT: StagedType> {
            func: $FunRef<OUT>,
        }

        impl<OUT: StagedType> Staged for $Call<OUT> {
            type Out = OUT;

            fn codegen(&self, ctx: &mut CompilationContext) -> Value {
                let return_info = TypeInfo::from_staged_type::<OUT>();
                codegen_call(ctx, self.func.id, &[], &return_info, &[])
            }
        }

        /// Create a call expression for a 0-ary function
        pub fn $call_fn<OUT: StagedType>(func: $FunRef<OUT>) -> $Call<OUT> {
            $Call { func }
        }
    };

    // N parameters (N >= 1)
    ($n:tt, $FunType:ident, $FunRef:ident, $Call:ident, $call_fn:ident,
     [$($T:ident),+], [$($arg:ident : $Arg:ident),+], [$($idx:tt),+]) => {

        /// Function type marker
        #[derive(Clone, Copy, Debug, PartialEq, Eq)]
        pub struct $FunType<$($T,)+ OUT> {
            _phantom: PhantomData<($($T,)+ OUT)>,
        }

        impl<$($T: StagedType,)+ OUT: StagedType> StagedType for $FunType<$($T,)+ OUT> {
            type RuntimeValue = extern "C" fn($($T::RuntimeValue,)+) -> OUT::RuntimeValue;

            fn cranelift_type() -> cranelift_codegen::ir::Type {
                types::I64
            }
        }

        /// Type-safe function reference (Copy, stores just ID)
        #[derive(Clone, Copy)]
        pub struct $FunRef<$($T: StagedType,)+ OUT: StagedType> {
            pub(crate) id: usize,
            _phantom: PhantomData<($($T,)+ OUT)>,
        }

        impl<$($T: StagedType,)+ OUT: StagedType> $FunRef<$($T,)+ OUT> {
            pub fn new(id: usize) -> Self {
                $FunRef {
                    id,
                    _phantom: PhantomData,
                }
            }

            pub fn id(&self) -> usize {
                self.id
            }
        }

        impl<$($T: StagedType,)+ OUT: StagedType> std::fmt::Debug for $FunRef<$($T,)+ OUT> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(f, "{}({})", stringify!($FunRef), self.id)
            }
        }

        impl<$($T: StagedType,)+ OUT: StagedType> Staged for $FunRef<$($T,)+ OUT> {
            type Out = $FunType<$($T,)+ OUT>;

            fn codegen(&self, ctx: &mut CompilationContext) -> Value {
                codegen_func_addr(ctx, self.id)
            }
        }

        /// Call expression
        #[derive(Clone)]
        pub struct $Call<$($T: StagedType,)+ OUT: StagedType, $($Arg),+> {
            func: $FunRef<$($T,)+ OUT>,
            $($arg: $Arg,)+
        }

        impl<$($T: StagedType,)+ OUT: StagedType, $($Arg),+> Staged for $Call<$($T,)+ OUT, $($Arg),+>
        where
            $($Arg: Staged<Out = $T>,)+
        {
            type Out = OUT;

            fn codegen(&self, ctx: &mut CompilationContext) -> Value {
                // Build type info from type parameters
                let param_infos = [$(TypeInfo::from_staged_type::<$T>()),+];
                let return_info = TypeInfo::from_staged_type::<OUT>();

                // Generate arg values
                let args = [$(self.$arg.codegen(ctx)),+];

                codegen_call(ctx, self.func.id, &param_infos, &return_info, &args)
            }
        }

        /// Create a call expression
        pub fn $call_fn<$($T,)+ OUT, $($Arg),+>(
            func: $FunRef<$($T,)+ OUT>,
            $($arg: $Arg,)+
        ) -> $Call<$($T,)+ OUT, $(<$Arg as IntoStaged<$T>>::Staged),+>
        where
            $($T: StagedType,)+
            OUT: StagedType,
            $($Arg: IntoStaged<$T>,)+
        {
            $Call {
                func,
                $($arg: $arg.into_staged(),)+
            }
        }
    };
}

// Generate types for arities 0-8
impl_fun_n!(0, FunType0, FunRef0, Call0, call0);
impl_fun_n!(1, FunType1, FunRef1, Call1, call1, [A], [arg0: Arg0], [0]);
impl_fun_n!(2, FunType2, FunRef2, Call2, call2, [A, B], [arg0: Arg0, arg1: Arg1], [0, 1]);
impl_fun_n!(3, FunType3, FunRef3, Call3, call3, [A, B, C], [arg0: Arg0, arg1: Arg1, arg2: Arg2], [0, 1, 2]);
impl_fun_n!(4, FunType4, FunRef4, Call4, call4, [A, B, C, D], [arg0: Arg0, arg1: Arg1, arg2: Arg2, arg3: Arg3], [0, 1, 2, 3]);
impl_fun_n!(5, FunType5, FunRef5, Call5, call5, [A, B, C, D, E], [arg0: Arg0, arg1: Arg1, arg2: Arg2, arg3: Arg3, arg4: Arg4], [0, 1, 2, 3, 4]);
impl_fun_n!(6, FunType6, FunRef6, Call6, call6, [A, B, C, D, E, F], [arg0: Arg0, arg1: Arg1, arg2: Arg2, arg3: Arg3, arg4: Arg4, arg5: Arg5], [0, 1, 2, 3, 4, 5]);
impl_fun_n!(7, FunType7, FunRef7, Call7, call7, [A, B, C, D, E, F, G], [arg0: Arg0, arg1: Arg1, arg2: Arg2, arg3: Arg3, arg4: Arg4, arg5: Arg5, arg6: Arg6], [0, 1, 2, 3, 4, 5, 6]);
impl_fun_n!(8, FunType8, FunRef8, Call8, call8, [A, B, C, D, E, F, G, H], [arg0: Arg0, arg1: Arg1, arg2: Arg2, arg3: Arg3, arg4: Arg4, arg5: Arg5, arg6: Arg6, arg7: Arg7], [0, 1, 2, 3, 4, 5, 6, 7]);

