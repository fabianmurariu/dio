//! First-class function support for staged computations.
//!
//! This module provides:
//! - `Compiler`: The central coordinator that owns function and variable definitions
//! - `FunRefN<T0, ..., OUT>`: Type-safe handles to function definitions (N = 0..8)
//! - `CallN`: Function call expressions
//! - `Compiled<T>`: The result of compilation, owns the JIT module
//!
//! # Multi-Parameter Functions
//!
//! Functions with 0-8 parameters are supported via `fun0`, `fun1`, ..., `fun8`.
//! Each returns a type-safe `FunRefN` that encodes the parameter and return types.
//!
//! # Struct Pass-by-Value
//!
//! For `#[repr(C)] Copy` structs, this module implements pass-by-value semantics:
//! - At the Rust ABI level: `fn(Point) -> Point` (value semantics)
//! - At the Cranelift level: multiple i64 parameters, stored to stack slot
//! - Internally: pointer to stack slot for field access

use crate::staged::{CompilationContext, Staged, Var};
use crate::types::StagedType;
use cranelift_codegen::ir::{
    types, AbiParam, InstBuilder, MemFlags, StackSlotData, StackSlotKind, Value,
};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use std::collections::HashMap;
use std::marker::PhantomData;

pub use crate::func_impl::*;

// =============================================================================
// StructInfo: Information about struct types for ABI handling
// =============================================================================

/// Information about a struct type needed for ABI handling.
///
/// This captures the struct's layout so we can:
/// 1. Pass it as multiple i64 values at the ABI boundary
/// 2. Create a properly-sized stack slot
/// 3. Store/load the values to/from the stack slot
#[derive(Clone, Debug)]
pub struct StructInfo {
    /// Size in bytes
    pub size: u32,
    /// Alignment in bytes
    pub alignment: u32,
    /// Number of i64 values needed at ABI boundary
    pub num_abi_values: usize,
}


// =============================================================================
// Internal: FunDef - Stored function definition
// =============================================================================

/// Internal storage for a function definition (type-erased body)
struct FunDef {
    name: String,
    /// The body expression, type-erased but we know its signature
    body: Box<dyn FnOnce(&mut CompilationContext) -> Value>,
    /// Type info for each parameter (supports 0..N parameters)
    param_infos: Vec<TypeInfo>,
    /// Return type info
    return_info: TypeInfo,
    /// Variable IDs for each parameter (one per logical parameter)
    param_var_ids: Vec<usize>,
}

// =============================================================================
// VarBuilder: Context for creating local variables within functions
// =============================================================================

/// A builder context for creating variables within function bodies.
///
/// This is passed to closures in `fun1` and `fun1_rec` to allow local
/// variable creation without exposing the entire Compiler.
pub struct VarBuilder<'a> {
    next_var_id: &'a mut usize,
}

impl<'a> VarBuilder<'a> {
    /// Create an uninitialized variable reference.
    ///
    /// # Safety
    /// You MUST assign to this variable before reading from it, otherwise codegen will panic.
    /// Prefer using `let_var()` which ensures initialization.
    pub unsafe fn var_unchecked<T: StagedType>(&mut self) -> Var<T> {
        let id = *self.next_var_id;
        *self.next_var_id += 1;
        Var::new(id)
    }

    /// Create a variable with an initial value.
    ///
    /// Returns an `InitVar` that can be used directly without tuple unpacking.
    /// Accepts any value that can be converted into a staged expression.
    pub fn let_var<T, E>(&mut self, init: E) -> crate::staged::InitVar<T, E::Staged>
    where
        T: StagedType,
        E: crate::staged::IntoStaged<T>,
    {
        let var = unsafe { self.var_unchecked() };
        crate::staged::InitVar::new(var, init.into_staged())
    }
}

// =============================================================================
// Compiler: Owns everything, coordinates compilation
// =============================================================================

/// The central coordinator for staged computations.
///
/// `Compiler` owns all function definitions and variable IDs. It provides
/// methods to create functions and variables, and to compile expressions
/// to native code.
pub struct Compiler<'a> {
    /// Function definitions indexed by ID
    functions: Vec<Option<FunDef>>,
    /// Next variable ID to assign
    next_var_id: usize,
    _marker: PhantomData<&'a ()>,
}

impl<'a> Default for Compiler<'a> {
    fn default() -> Self {
        Self::new()
    }
}

impl<'a> Compiler<'a> {
    /// Create a new compiler
    pub fn new() -> Self {
        Compiler {
            functions: Vec::new(),
            next_var_id: 0,
            _marker: PhantomData,
        }
    }

    /// Create an uninitialized variable reference.
    ///
    /// # Safety
    /// You MUST assign to this variable before reading from it, otherwise codegen will panic.
    /// Prefer using `let_var()` which ensures initialization.
    ///
    /// This is useful for variables that will be captured by function closures and
    /// initialized within the closure body.
    ///
    /// # Example
    /// ```ignore
    /// let i = compiler.var_unchecked::<I64Type>();
    /// let func = compiler.fun1("example", |n| {
    ///     seq(assign(i, Const::new(0)), i) // Must assign before use
    /// });
    /// ```
    pub unsafe fn var_unchecked<T: StagedType>(&mut self) -> Var<T> {
        let id = self.next_var_id;
        self.next_var_id += 1;
        Var::new(id)
    }

    /// Create a variable with an initial value.
    ///
    /// Returns an `InitVar` that can be used directly without tuple unpacking.
    /// The initialization happens automatically when the variable is used in a tuple.
    ///
    /// Accepts any value that can be converted into a staged expression.
    /// This allows ergonomic usage like `let_var(42i64)` instead of
    /// `let_var(Const::<I64Type>::new(42))`.
    ///
    /// # Example
    /// ```ignore
    /// let x = compiler.let_var(42i64);
    /// let expr = (x, add(x, 8i64)); // x initializes, then is used in add
    /// ```
    pub fn let_var<T, E>(&mut self, init: E) -> crate::staged::InitVar<T, E::Staged>
    where
        T: StagedType,
        E: crate::staged::IntoStaged<T>,
    {
        let var = unsafe { self.var_unchecked() };
        crate::staged::InitVar::new(var, init.into_staged())
    }

    /// Define a unary function.
    ///
    /// The body function is called immediately to build the expression tree.
    /// No Cranelift calls happen until `compile()` is called.
    ///
    /// The body function receives a `VarBuilder` context that allows creating
    /// local variables within the function.
    ///
    /// # Struct Pass-by-Value
    ///
    /// If `A` is a `#[repr(C)] Copy` struct, the function will accept the struct
    /// by value at the Rust level (`fn(Point)`), but internally store it in a
    /// stack slot for field access.
    pub fn fun1<A, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef<A, OUT>
    where
        A: StagedType,
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder, Var<A>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        // Create the parameter variable
        let param_id = self.next_var_id;
        self.next_var_id += 1;
        let param_var = Var::<A>::new(param_id);

        // Create a VarBuilder for local variable creation
        let mut var_builder = VarBuilder {
            next_var_id: &mut self.next_var_id,
        };

        // Call body_fn immediately to build the expression tree
        let body_expr = body_fn(&mut var_builder, param_var);

        // Store the function definition
        let func_id = self.functions.len();
        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| body_expr.codegen(ctx)),
            param_infos: vec![TypeInfo::from_staged_type::<A>()],
            return_info: TypeInfo::from_staged_type::<OUT>(),
            param_var_ids: vec![param_id],
        };

        self.functions.push(Some(func_def));

        FunRef::new(func_id)
    }

    /// Define a recursive unary function.
    ///
    /// Similar to `fun1`, but the body function receives a reference to itself,
    /// allowing for recursive calls. The function reference is passed as the first
    /// argument to the body closure, followed by the VarBuilder and parameter.
    ///
    /// # Example
    /// ```ignore
    /// let factorial = compiler.fun1_rec("factorial", |f, ctx, x: Var<I64Type>| {
    ///     // Can create local variables
    ///     let temp = ctx.let_var(0i64);
    ///     // Recursive call: f(x - 1)
    ///     call1(f, sub(x, Const::<I64Type>::new(1)))
    /// });
    /// ```
    pub fn fun1_rec<A, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef<A, OUT>
    where
        A: StagedType,
        OUT: StagedType,
        F: FnOnce(FunRef<A, OUT>, &mut VarBuilder, Var<A>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        // Create the parameter variable
        let param_id = self.next_var_id;
        self.next_var_id += 1;
        let param_var = Var::<A>::new(param_id);

        // Pre-allocate the function ID and create FunRef
        let func_id = self.functions.len();
        let func_ref = FunRef::new(func_id);

        // Push a placeholder first (None) to reserve the slot
        self.functions.push(None);

        // Create a VarBuilder for local variable creation
        let mut var_builder = VarBuilder {
            next_var_id: &mut self.next_var_id,
        };

        // Now call body_fn with the function reference, VarBuilder, and parameter
        let body_expr = body_fn(func_ref, &mut var_builder, param_var);

        // Create the actual function definition
        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| body_expr.codegen(ctx)),
            param_infos: vec![TypeInfo::from_staged_type::<A>()],
            return_info: TypeInfo::from_staged_type::<OUT>(),
            param_var_ids: vec![param_id],
        };

        // Replace the placeholder with the actual definition
        self.functions[func_id] = Some(func_def);

        FunRef::new(func_id)
    }

    /// Define a zero-argument function.
    pub fn fun0<OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef0<OUT>
    where
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        let mut var_builder = VarBuilder {
            next_var_id: &mut self.next_var_id,
        };
        let body_expr = body_fn(&mut var_builder);

        let func_id = self.functions.len();
        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| body_expr.codegen(ctx)),
            param_infos: vec![],
            return_info: TypeInfo::from_staged_type::<OUT>(),
            param_var_ids: vec![],
        };

        self.functions.push(Some(func_def));
        FunRef0::new(func_id)
    }

    /// Define a recursive zero-argument function.
    pub fn fun0_rec<OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef0<OUT>
    where
        OUT: StagedType,
        F: FnOnce(FunRef0<OUT>, &mut VarBuilder) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        let func_id = self.functions.len();
        let func_ref = FunRef0::new(func_id);
        self.functions.push(None);

        let mut var_builder = VarBuilder {
            next_var_id: &mut self.next_var_id,
        };
        let body_expr = body_fn(func_ref, &mut var_builder);

        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| body_expr.codegen(ctx)),
            param_infos: vec![],
            return_info: TypeInfo::from_staged_type::<OUT>(),
            param_var_ids: vec![],
        };

        self.functions[func_id] = Some(func_def);
        FunRef0::new(func_id)
    }

    /// Define a binary function.
    pub fn fun2<A, B, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef2<A, B, OUT>
    where
        A: StagedType,
        B: StagedType,
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder, Var<A>, Var<B>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        let param0_id = self.next_var_id;
        self.next_var_id += 1;
        let param0_var = Var::<A>::new(param0_id);

        let param1_id = self.next_var_id;
        self.next_var_id += 1;
        let param1_var = Var::<B>::new(param1_id);

        let mut var_builder = VarBuilder {
            next_var_id: &mut self.next_var_id,
        };
        let body_expr = body_fn(&mut var_builder, param0_var, param1_var);

        let func_id = self.functions.len();
        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| body_expr.codegen(ctx)),
            param_infos: vec![
                TypeInfo::from_staged_type::<A>(),
                TypeInfo::from_staged_type::<B>(),
            ],
            return_info: TypeInfo::from_staged_type::<OUT>(),
            param_var_ids: vec![param0_id, param1_id],
        };

        self.functions.push(Some(func_def));
        FunRef2::new(func_id)
    }

    /// Define a recursive binary function.
    pub fn fun2_rec<A, B, OUT, F, BODY>(
        &mut self,
        name: &str,
        body_fn: F,
    ) -> FunRef2<A, B, OUT>
    where
        A: StagedType,
        B: StagedType,
        OUT: StagedType,
        F: FnOnce(FunRef2<A, B, OUT>, &mut VarBuilder, Var<A>, Var<B>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        let param0_id = self.next_var_id;
        self.next_var_id += 1;
        let param0_var = Var::<A>::new(param0_id);

        let param1_id = self.next_var_id;
        self.next_var_id += 1;
        let param1_var = Var::<B>::new(param1_id);

        let func_id = self.functions.len();
        let func_ref = FunRef2::new(func_id);
        self.functions.push(None);

        let mut var_builder = VarBuilder {
            next_var_id: &mut self.next_var_id,
        };
        let body_expr = body_fn(func_ref, &mut var_builder, param0_var, param1_var);

        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| body_expr.codegen(ctx)),
            param_infos: vec![
                TypeInfo::from_staged_type::<A>(),
                TypeInfo::from_staged_type::<B>(),
            ],
            return_info: TypeInfo::from_staged_type::<OUT>(),
            param_var_ids: vec![param0_id, param1_id],
        };

        self.functions[func_id] = Some(func_def);
        FunRef2::new(func_id)
    }

    /// Define a ternary function.
    pub fn fun3<A, B, C, OUT, F, BODY>(
        &mut self,
        name: &str,
        body_fn: F,
    ) -> FunRef3<A, B, C, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        OUT: StagedType,
        F: FnOnce(&mut VarBuilder, Var<A>, Var<B>, Var<C>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        let param0_id = self.next_var_id;
        self.next_var_id += 1;
        let param0_var = Var::<A>::new(param0_id);

        let param1_id = self.next_var_id;
        self.next_var_id += 1;
        let param1_var = Var::<B>::new(param1_id);

        let param2_id = self.next_var_id;
        self.next_var_id += 1;
        let param2_var = Var::<C>::new(param2_id);

        let mut var_builder = VarBuilder {
            next_var_id: &mut self.next_var_id,
        };
        let body_expr = body_fn(&mut var_builder, param0_var, param1_var, param2_var);

        let func_id = self.functions.len();
        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| body_expr.codegen(ctx)),
            param_infos: vec![
                TypeInfo::from_staged_type::<A>(),
                TypeInfo::from_staged_type::<B>(),
                TypeInfo::from_staged_type::<C>(),
            ],
            return_info: TypeInfo::from_staged_type::<OUT>(),
            param_var_ids: vec![param0_id, param1_id, param2_id],
        };

        self.functions.push(Some(func_def));
        FunRef3::new(func_id)
    }

    /// Define a recursive ternary function.
    pub fn fun3_rec<A, B, C, OUT, F, BODY>(
        &mut self,
        name: &str,
        body_fn: F,
    ) -> FunRef3<A, B, C, OUT>
    where
        A: StagedType,
        B: StagedType,
        C: StagedType,
        OUT: StagedType,
        F: FnOnce(FunRef3<A, B, C, OUT>, &mut VarBuilder, Var<A>, Var<B>, Var<C>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        let param0_id = self.next_var_id;
        self.next_var_id += 1;
        let param0_var = Var::<A>::new(param0_id);

        let param1_id = self.next_var_id;
        self.next_var_id += 1;
        let param1_var = Var::<B>::new(param1_id);

        let param2_id = self.next_var_id;
        self.next_var_id += 1;
        let param2_var = Var::<C>::new(param2_id);

        let func_id = self.functions.len();
        let func_ref = FunRef3::new(func_id);
        self.functions.push(None);

        let mut var_builder = VarBuilder {
            next_var_id: &mut self.next_var_id,
        };
        let body_expr = body_fn(
            func_ref,
            &mut var_builder,
            param0_var,
            param1_var,
            param2_var,
        );

        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| body_expr.codegen(ctx)),
            param_infos: vec![
                TypeInfo::from_staged_type::<A>(),
                TypeInfo::from_staged_type::<B>(),
                TypeInfo::from_staged_type::<C>(),
            ],
            return_info: TypeInfo::from_staged_type::<OUT>(),
            param_var_ids: vec![param0_id, param1_id, param2_id],
        };

        self.functions[func_id] = Some(func_def);
        FunRef3::new(func_id)
    }

    /// Compile an expression to native code.
    ///
    /// This compiles all referenced functions and the main expression,
    /// returning a `Compiled<T>` that owns the JIT module and can extract
    /// the computed value.
    ///
    /// # Struct Handling
    ///
    /// For functions with struct parameters or returns:
    /// - Parameters: Multiple i64 values are received, stored to a stack slot,
    ///   and the variable holds the stack slot pointer
    /// - Returns: The result pointer is used to load multiple i64 values for return
    pub fn compile<S: Staged + 'static>(
        self,
        expr: S,
    ) -> Result<Compiled<'a, S::Out>, CompileError> {
        // Create the JIT module
        let builder = JITBuilder::new(default_libcall_names())
            .map_err(|e| CompileError::JitError(e.to_string()))?;
        let mut module = JITModule::new(builder);

        // Get the default calling convention for this platform
        // This ensures proper ABI compatibility, especially important for external function calls
        // Using SystemV on Unix-like systems, WindowsFastcall on Windows
        let call_conv = if cfg!(target_os = "windows") {
            cranelift_codegen::isa::CallConv::WindowsFastcall
        } else {
            cranelift_codegen::isa::CallConv::SystemV
        };

        // First pass: declare all functions
        let mut func_map: HashMap<usize, FuncId> = HashMap::new();

        for (id, func_opt) in self.functions.iter().enumerate() {
            if let Some(func_def) = func_opt {
                let mut sig = module.make_signature();
                sig.call_conv = call_conv;

                // Add all ABI param types (multiple per logical param for structs)
                for param_info in &func_def.param_infos {
                    for abi_type in &param_info.abi_types {
                        sig.params.push(AbiParam::new(*abi_type));
                    }
                }

                // Add all ABI return types (multiple for structs)
                for abi_type in &func_def.return_info.abi_types {
                    sig.returns.push(AbiParam::new(*abi_type));
                }

                let func_id = module
                    .declare_function(&func_def.name, Linkage::Local, &sig)
                    .map_err(|e| CompileError::ModuleError(e.to_string()))?;

                func_map.insert(id, func_id);
            }
        }

        // Declare the main function
        let mut main_sig = module.make_signature();
        main_sig.call_conv = call_conv;

        // For main function, use the expression's return type ABI types
        for abi_type in S::Out::abi_types() {
            main_sig.returns.push(AbiParam::new(abi_type));
        }

        let main_func_id = module
            .declare_function("__main__", Linkage::Local, &main_sig)
            .map_err(|e| CompileError::ModuleError(e.to_string()))?;

        // Second pass: define all functions
        // We need to consume self.functions since FunDef contains FnOnce
        let mut functions = self.functions;

        for (id, func_opt) in functions.iter_mut().enumerate() {
            if let Some(func_def) = func_opt.take() {
                let func_id = func_map[&id];

                let mut sig = module.make_signature();
                sig.call_conv = call_conv;

                for param_info in &func_def.param_infos {
                    for abi_type in &param_info.abi_types {
                        sig.params.push(AbiParam::new(*abi_type));
                    }
                }
                for abi_type in &func_def.return_info.abi_types {
                    sig.returns.push(AbiParam::new(*abi_type));
                }

                let mut func_ctx = module.make_context();
                func_ctx.func.signature = sig;

                {
                    let mut builder_context = FunctionBuilderContext::new();
                    let mut builder =
                        FunctionBuilder::new(&mut func_ctx.func, &mut builder_context);
                    let entry_block = builder.create_block();
                    builder.append_block_params_for_function_params(entry_block);
                    builder.switch_to_block(entry_block);
                    builder.seal_block(entry_block);

                    // Create var_map for this function
                    let mut var_map: HashMap<usize, Variable> = HashMap::new();

                    // Handle all parameters
                    let block_params = builder.block_params(entry_block).to_vec();
                    let mut abi_idx = 0;

                    for (param_idx, param_info) in func_def.param_infos.iter().enumerate() {
                        let var_id = func_def.param_var_ids[param_idx];

                        if let Some(ref struct_info) = param_info.struct_info {
                            // STRUCT PARAMETER: Create stack slot, store values, use pointer
                            let align_shift = struct_info.alignment.trailing_zeros() as u8;
                            let stack_slot = builder.create_sized_stack_slot(StackSlotData::new(
                                StackSlotKind::ExplicitSlot,
                                struct_info.size,
                                align_shift,
                            ));

                            let slot_ptr = builder.ins().stack_addr(types::I64, stack_slot, 0);

                            // Store each i64 value to the stack slot
                            for i in 0..struct_info.num_abi_values {
                                let offset = (i * 8) as i32;
                                builder.ins().store(
                                    MemFlags::trusted(),
                                    block_params[abi_idx],
                                    slot_ptr,
                                    offset,
                                );
                                abi_idx += 1;
                            }

                            // Declare variable to hold the stack pointer
                            let param_var = builder.declare_var(types::I64);
                            builder.def_var(param_var, slot_ptr);
                            var_map.insert(var_id, param_var);
                        } else {
                            // PRIMITIVE PARAMETER: Direct binding
                            let param_value = block_params[abi_idx];
                            let param_type = param_info.abi_types[0];
                            let param_var = builder.declare_var(param_type);
                            builder.def_var(param_var, param_value);
                            var_map.insert(var_id, param_var);
                            abi_idx += 1;
                        }
                    }

                    // Generate the body code
                    let result = {
                        let mut ctx = CompilationContext {
                            builder: &mut builder,
                            module: &mut module,
                            var_map: &mut var_map,
                            func_map: &func_map,
                        };
                        (func_def.body)(&mut ctx)
                    };

                    // Handle return - either primitive or struct
                    if let Some(ref struct_info) = func_def.return_info.struct_info {
                        // STRUCT RETURN: Load multiple values from the result pointer
                        let mut return_values = Vec::with_capacity(struct_info.num_abi_values);
                        for i in 0..struct_info.num_abi_values {
                            let offset = (i * 8) as i32;
                            let val =
                                builder
                                    .ins()
                                    .load(types::I64, MemFlags::trusted(), result, offset);
                            return_values.push(val);
                        }
                        builder.ins().return_(&return_values);
                    } else {
                        // PRIMITIVE RETURN: Direct return
                        builder.ins().return_(&[result]);
                    }

                    builder.finalize();
                }

                // Debug output for Cranelift IR
                if std::env::var("RUST_LMS_DEBUG_IR").is_ok() {
                    eprintln!("=== Function: {} ===", func_def.name);
                    eprintln!("{}", func_ctx.func);
                    eprintln!();
                }

                module
                    .define_function(func_id, &mut func_ctx)
                    .map_err(|e| CompileError::ModuleError(e.to_string()))?;
                module.clear_context(&mut func_ctx);
            }
        }

        // Define the main function
        {
            let mut func_ctx = module.make_context();
            func_ctx.func.signature = main_sig;

            {
                let mut builder_context = FunctionBuilderContext::new();
                let mut builder = FunctionBuilder::new(&mut func_ctx.func, &mut builder_context);
                let entry_block = builder.create_block();
                builder.switch_to_block(entry_block);
                builder.seal_block(entry_block);

                let mut var_map: HashMap<usize, Variable> = HashMap::new();

                let result = {
                    let mut ctx = CompilationContext {
                        builder: &mut builder,
                        module: &mut module,
                        var_map: &mut var_map,
                        func_map: &func_map,
                    };
                    expr.codegen(&mut ctx)
                };

                // Handle return for main function
                if S::Out::is_copy_struct() {
                    // STRUCT RETURN: Load multiple values from the result pointer
                    let num_values = S::Out::num_abi_values();
                    let mut return_values = Vec::with_capacity(num_values);
                    for i in 0..num_values {
                        let offset = (i * 8) as i32;
                        let val =
                            builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), result, offset);
                        return_values.push(val);
                    }
                    builder.ins().return_(&return_values);
                } else {
                    // PRIMITIVE RETURN
                    builder.ins().return_(&[result]);
                }

                builder.finalize();
            }

            // Debug output for main function IR
            if std::env::var("RUST_LMS_DEBUG_IR").is_ok() {
                eprintln!("=== Function: __main__ ===");
                eprintln!("{}", func_ctx.func);
                eprintln!();
            }

            module
                .define_function(main_func_id, &mut func_ctx)
                .map_err(|e| CompileError::ModuleError(e.to_string()))?;
            module.clear_context(&mut func_ctx);
        }

        // Finalize the module
        module
            .finalize_definitions()
            .map_err(|e| CompileError::ModuleError(e.to_string()))?;

        // Get the main function pointer
        let main_ptr = module.get_finalized_function(main_func_id);

        Ok(Compiled {
            module,
            main_ptr,
            _phantom: PhantomData,
        })
    }
}

// =============================================================================
// CompileError
// =============================================================================

/// Errors that can occur during compilation
#[derive(Debug)]
pub enum CompileError {
    JitError(String),
    ModuleError(String),
}

impl std::fmt::Display for CompileError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CompileError::JitError(msg) => write!(f, "JIT error: {}", msg),
            CompileError::ModuleError(msg) => write!(f, "Module error: {}", msg),
        }
    }
}

impl std::error::Error for CompileError {}

// =============================================================================
// Compiled<T>: The result of compilation
// =============================================================================

/// A compiled expression that owns its JIT module.
///
/// Use `.run()` to execute the compiled code and get the result.
pub struct Compiled<'a, T: StagedType> {
    #[allow(dead_code)]
    module: JITModule,
    main_ptr: *const u8,
    _phantom: PhantomData<&'a T>,
}

impl<'a, T: StagedType> Compiled<'a, T> {
    /// Execute the compiled code and return the result.
    ///
    /// # Safety
    /// This is safe as long as the compilation was done correctly.
    pub fn run(&self) -> T::RuntimeValue<'a>
    where
        T::RuntimeValue<'a>: Copy,
    {
        let func: fn() -> T::RuntimeValue<'a> = unsafe { std::mem::transmute(self.main_ptr) };
        func()
    }
}

// Special implementation for function types to get the function pointer
impl<'a, A: StagedType, OUT: StagedType> Compiled<'a, FunType1<A, OUT>> {
    /// Get the compiled function as a callable function pointer.
    ///
    /// Returns an `extern "C"` function pointer to match the System V calling
    /// convention used by Cranelift. This is important for structs passed by value,
    /// as Rust's default calling convention may differ from the C ABI.
    pub fn as_fn(&self) -> extern "C" fn(A::RuntimeValue<'a>) -> OUT::RuntimeValue<'a> {
        // The main function returns a function pointer (as i64)
        // We need to call the main function to get that pointer
        let get_ptr: fn() -> i64 = unsafe { std::mem::transmute(self.main_ptr) };
        let fn_ptr = get_ptr();
        unsafe { std::mem::transmute(fn_ptr) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;

    #[test]
    fn test_simple_constant() {
        let compiler = Compiler::new();
        let five = Const::<I64Type>::new(5);

        let compiled = compiler.compile(five).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 5);
    }

    #[test]
    fn test_simple_addition() {
        let compiler = Compiler::new();
        let expr = add::<I64Type, _, _>(3i64, 4i64);

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 7);
    }

    #[test]
    fn test_nested_arithmetic() {
        let compiler = Compiler::new();
        // (3 + 4) * 2 = 14
        let expr = mul(
            add(Const::<I64Type>::new(3), Const::<I64Type>::new(4)),
            Const::<I64Type>::new(2),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 14);
    }

    #[test]
    fn test_fun1_and_call() {
        let mut compiler = Compiler::new();

        // Define: square(x) = x * x
        let square = compiler.fun1("square", |_ctx, x: Var<I64Type>| mul(x, x));

        // Call: square(5) = 25
        let expr = call1(square, Const::<I64Type>::new(5));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 25);
    }

    #[test]
    fn test_var_before_fun1() {
        let mut compiler = Compiler::new();

        // Create a variable first (this would break the old code that used func_id as param_id)
        let _unused = unsafe { compiler.var_unchecked::<I64Type>() };

        // Define: double(x) = x + x
        let double = compiler.fun1("double", |_ctx, x: Var<I64Type>| add(x, x));

        // Call: double(7) = 14
        let expr = call1(double, Const::<I64Type>::new(7));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 14);
    }

    #[test]
    fn test_return_function_pointer() {
        let mut compiler = Compiler::new();

        // Define: cube(x) = x * x * x
        let cube = compiler.fun1("cube", |_ctx, x: Var<I64Type>| mul(mul(x, x), x));

        // Compile the function reference itself (not a call)
        let compiled = compiler.compile(cube).expect("compilation failed");

        // Extract the function pointer
        let cube_fn = compiled.as_fn();

        // Test the function with various inputs
        assert_eq!(cube_fn(2), 8);
        assert_eq!(cube_fn(3), 27);
        assert_eq!(cube_fn(5), 125);
        assert_eq!(cube_fn(-2), -8);
    }

    #[test]
    fn test_recursive_function_compiles() {
        let mut compiler = Compiler::new();

        // Define a recursive function: rec(x) = x + rec(x - 1)
        // Note: This will infinite loop if called, but we're just testing
        // that it compiles and the function can reference itself
        let _rec = compiler.fun1_rec("recursive", |f, _ctx, x: Var<I64Type>| {
            // Body references itself: call f recursively
            add(x, call1(f, sub(x, Const::<I64Type>::new(1))))
        });

        // Just test that compilation succeeds
        // We don't call it since it would infinite loop without conditionals
        let expr = Const::<I64Type>::new(42);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    // =========================================================================
    // Control Flow Tests
    // =========================================================================

    #[test]
    fn test_if_then_else_true_branch() {
        let compiler = Compiler::new();

        // if true then 10 else 20 => 10
        let expr = if_then_else(
            Const::<BoolType>::new(true),
            Const::<I64Type>::new(10),
            Const::<I64Type>::new(20),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 10);
    }

    #[test]
    fn test_if_then_else_false_branch() {
        let compiler = Compiler::new();

        // if false then 10 else 20 => 20
        let expr = if_then_else(
            Const::<BoolType>::new(false),
            Const::<I64Type>::new(10),
            Const::<I64Type>::new(20),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 20);
    }

    #[test]
    fn test_if_then_else_with_comparison() {
        let mut compiler = Compiler::new();

        // Define: clamp_max(x) = if x < 10 then x else 10
        let clamp_max = compiler.fun1("clamp_max", |_ctx, x: Var<I64Type>| {
            if_then_else(
                lt(x, Const::<I64Type>::new(10)),
                x,
                Const::<I64Type>::new(10),
            )
        });

        let expr = call1(clamp_max, Const::<I64Type>::new(5));
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 5); // 5 < 10, so return 5
    }

    #[test]
    fn test_if_then_else_clamps_at_max() {
        let mut compiler = Compiler::new();

        // Define: clamp_max(x) = if x < 10 then x else 10
        let clamp_max = compiler.fun1("clamp_max", |_ctx, x: Var<I64Type>| {
            if_then_else(
                lt(x, Const::<I64Type>::new(10)),
                x,
                Const::<I64Type>::new(10),
            )
        });

        let expr = call1(clamp_max, Const::<I64Type>::new(15));
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 10); // 15 >= 10, so return 10
    }

    #[test]
    fn test_nested_if_then_else() {
        let mut compiler = Compiler::new();

        // clamp(x) = if x < 0 then 0 else (if x > 10 then 10 else x)
        let clamp = compiler.fun1("clamp", |_ctx, x: Var<I64Type>| {
            if_then_else(
                lt(x, Const::<I64Type>::new(0)),
                Const::<I64Type>::new(0),
                if_then_else(
                    lt(Const::<I64Type>::new(10), x), // x > 10
                    Const::<I64Type>::new(10),
                    x,
                ),
            )
        });

        // Test values
        let compiled_neg = compiler
            .compile(call1(clamp, Const::<I64Type>::new(-5)))
            .expect("compilation failed");
        assert_eq!(compiled_neg.run(), 0); // Clamped at min
    }

    #[test]
    fn test_seq_basic() {
        let compiler = Compiler::new();

        // (5, 10) => 10 (first value ignored, second returned)
        let expr = (Const::<I64Type>::new(5), Const::<I64Type>::new(10));

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 10);
    }

    #[test]
    fn test_let_var() {
        let mut compiler = Compiler::new();

        // Test new ergonomic let_var API
        let x = compiler.let_var(42i64);
        let y = compiler.let_var(8i64);

        // x and y are InitVar, use *x to get Var<I64Type>
        let expr = (x, y, add::<I64Type, _, _>(*x, *y));

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 50); // 42 + 8 = 50
    }

    #[test]
    fn test_ergonomic_assign() {
        let mut compiler = Compiler::new();

        let x = unsafe { compiler.var_unchecked::<I64Type>() };
        let y = unsafe { compiler.var_unchecked::<I64Type>() };

        // Test ergonomic assign with primitive values (no Const::new needed)
        let expr = (
            assign(x, 10i64), // Instead of assign(x, Const::<I64Type>::new(10))
            assign(y, 32i64), // Instead of assign(y, Const::<I64Type>::new(32))
            add(x, y),
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_factorial() {
        let mut compiler = Compiler::new();

        // factorial(n) = if n <= 1 then 1 else n * factorial(n - 1)
        let factorial = compiler.fun1_rec("factorial", |f, _ctx, n: Var<I64Type>| {
            if_then_else(
                lt(n, 2), // n < 2 means n <= 1
                Const::new(1),
                mul(n, call1(f, sub(n, Const::<I64Type>::new(1)))),
            )
        });

        // Test factorial(5) = 120
        let expr = call1(factorial, Const::<I64Type>::new(5));
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 120);
    }

    #[test]
    fn test_factorial_various_inputs() {
        let mut compiler = Compiler::new();

        // factorial(n) = if n <= 1 then 1 else n * factorial(n - 1)
        let factorial = compiler.fun1_rec("factorial", |f, _ctx, n: Var<I64Type>| {
            if_then_else(
                lt(n, Const::<I64Type>::new(2)),
                Const::<I64Type>::new(1),
                mul(n, call1(f, sub(n, Const::<I64Type>::new(1)))),
            )
        });

        // Compile and get function pointer
        let compiled = compiler.compile(factorial).expect("compilation failed");
        let factorial_fn = compiled.as_fn();

        // Test various inputs
        assert_eq!(factorial_fn(0), 1); // 0! = 1
        assert_eq!(factorial_fn(1), 1); // 1! = 1
        assert_eq!(factorial_fn(2), 2); // 2! = 2
        assert_eq!(factorial_fn(3), 6); // 3! = 6
        assert_eq!(factorial_fn(4), 24); // 4! = 24
        assert_eq!(factorial_fn(5), 120); // 5! = 120
        assert_eq!(factorial_fn(6), 720); // 6! = 720
        assert_eq!(factorial_fn(10), 3628800); // 10! = 3628800
    }

    #[test]
    fn test_fibonacci() {
        let mut compiler = Compiler::new();

        // fib(n) = if n < 2 then n else fib(n-1) + fib(n-2)
        let fib = compiler.fun1_rec("fib", |f, _ctx, n: Var<I64Type>| {
            if_then_else(
                lt(n, Const::<I64Type>::new(2)),
                n, // fib(0) = 0, fib(1) = 1
                add(
                    call1(f, sub(n, Const::<I64Type>::new(1))),
                    call1(f, sub(n, Const::<I64Type>::new(2))),
                ),
            )
        });

        // Compile and get function pointer
        let compiled = compiler.compile(fib).expect("compilation failed");
        let fib_fn = compiled.as_fn();

        // Test Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
        assert_eq!(fib_fn(0), 0);
        assert_eq!(fib_fn(1), 1);
        assert_eq!(fib_fn(2), 1);
        assert_eq!(fib_fn(3), 2);
        assert_eq!(fib_fn(4), 3);
        assert_eq!(fib_fn(5), 5);
        assert_eq!(fib_fn(6), 8);
        assert_eq!(fib_fn(7), 13);
        assert_eq!(fib_fn(10), 55);
    }

    // =========================================================================
    // While Loop Tests
    // =========================================================================

    #[test]
    fn test_while_loop_zero_iterations() {
        let mut compiler = Compiler::new();

        // Create local variable
        let result = compiler.let_var(0i64);

        // while(false) { result = 999 } ; result = 42
        // Loop body never executes, result should be 42
        let expr = (
            result,
            while_loop(false, assign(*result, 999)),
            assign(*result, 42),
            *result,
        );

        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_while_loop_factorial() {
        let mut compiler = Compiler::new();

        // Create local variables BEFORE fun1
        let i = compiler.let_var(1i64);
        let result = compiler.let_var(1i64);

        // Iterative factorial using while loop
        let factorial_iter = compiler.fun1("factorial_iter", |_ctx, n: Var<I64Type>| {
            // i = 1; result = 1;
            // while (i <= n) { result = result * i; i = i + 1; }
            // return result;
            (
                (i, result),
                while_loop(
                    lt(*i, add(n, 1)), // i <= n
                    (assign(*result, mul(*result, *i)), assign(*i, add(*i, 1))),
                ),
                *result,
            )
        });

        let compiled = compiler
            .compile(factorial_iter)
            .expect("compilation failed");
        let factorial_fn = compiled.as_fn();

        assert_eq!(factorial_fn(0), 1); // 0! = 1
        assert_eq!(factorial_fn(1), 1); // 1! = 1
        assert_eq!(factorial_fn(2), 2); // 2! = 2
        assert_eq!(factorial_fn(3), 6); // 3! = 6
        assert_eq!(factorial_fn(5), 120); // 5! = 120
        assert_eq!(factorial_fn(10), 3628800); // 10! = 3628800
    }

    #[test]
    fn test_while_loop_fibonacci_iterative() {
        let mut compiler = Compiler::new();

        // Create local variables BEFORE fun1
        let i = compiler.let_var(2i64);
        let a = compiler.let_var(0i64); // fib(i-2)
        let b = compiler.let_var(1i64); // fib(i-1)
        let temp = compiler.let_var(0i64);

        // Iterative Fibonacci using while loop
        // Much faster than recursive version!
        let fib_iter = compiler.fun1("fib_iter", |_ctx, n: Var<I64Type>| {
            // if n < 2 return n
            // else: a = 0; b = 1; i = 2;
            //       while (i <= n) { temp = a + b; a = b; b = temp; i = i + 1; }
            //       return b;
            (
                (i, a, b, temp),
                if_then_else(
                    lt(n, 2),
                    n,
                    (
                        while_loop(
                            lt(*i, add(n, 1)), // i <= n
                            (
                                assign(*temp, add(*a, *b)),
                                assign(*a, *b),
                                assign(*b, *temp),
                                assign(*i, add(*i, 1)),
                            ),
                        ),
                        *b,
                    ),
                ),
            )
        });

        let compiled = compiler.compile(fib_iter).expect("compilation failed");
        let fib_fn = compiled.as_fn();

        // Test Fibonacci sequence
        assert_eq!(fib_fn(0), 0);
        assert_eq!(fib_fn(1), 1);
        assert_eq!(fib_fn(2), 1);
        assert_eq!(fib_fn(3), 2);
        assert_eq!(fib_fn(4), 3);
        assert_eq!(fib_fn(5), 5);
        assert_eq!(fib_fn(10), 55);
        assert_eq!(fib_fn(20), 6765);
        assert_eq!(fib_fn(30), 832040);
    }

    #[test]
    fn test_local_variables_in_fun1() {
        use crate::refer::SRef;
        use crate::slice::Slice;

        let mut compiler = Compiler::new();

        // Function that sums elements > 5 using local variables
        // fn sum_gt_5(arr: &[i64]) -> i64
        let sum_gt_5 = compiler.fun1("sum_gt_5", |ctx, arr: Var<SRef<Slice<I64Type>>>| {
            // Create local variables inside the function using ctx
            let i = ctx.let_var(0u64);
            let sum = ctx.let_var(0i64);
            let v = ctx.let_var(0i64);

            (
                (i, sum, v),
                while_loop(
                    lt(*i, arr.len()),
                    (
                        // v = arr.get_unchecked(i)
                        assign(*v, arr.get_unchecked(*i)),
                        // sum = if v > 5 then sum + v else sum
                        assign(
                            *sum,
                            if_then_else(
                                lt(5, *v), // v > 5
                                add(*sum, *v),
                                *sum,
                            ),
                        ),
                        assign(*i, add(*i, 1u64)),
                    ),
                ),
                *sum,
            )
        });

        let compiled = compiler.compile(sum_gt_5).expect("compilation failed");
        let f = compiled.as_fn();

        // Test with array [0, 3, 5, 7, 2, 8, 1, 9, 4, 6]
        // Elements > 5: 7, 8, 9, 6 => sum = 30
        let data: [i64; 10] = [0, 3, 5, 7, 2, 8, 1, 9, 4, 6];
        let slice: &[i64] = &data;

        let result = f(slice);
        assert_eq!(result, 30); // 7 + 8 + 9 + 6 = 30
    }

    // =========================================================================
    // Multi-Parameter Function Tests
    // =========================================================================

    #[test]
    fn test_fun0_constant() {
        use crate::func_impl::call0;

        let mut compiler = Compiler::new();

        // Define: get_answer() = 42
        let get_answer = compiler.fun0("get_answer", |_ctx| Const::<I64Type>::new(42));

        let expr = call0(get_answer);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_fun2_add() {
        use crate::func_impl::call2;

        let mut compiler = Compiler::new();

        // Define: add_two(a, b) = a + b
        let add_two = compiler.fun2("add_two", |_ctx, a: Var<I64Type>, b: Var<I64Type>| {
            add(a, b)
        });

        let expr = call2(add_two, 10i64, 32i64);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 42);
    }

    #[test]
    fn test_fun2_multiply_and_add() {
        use crate::func_impl::call2;

        let mut compiler = Compiler::new();

        // Define: mul_add(a, b) = a * b + a
        let mul_add = compiler.fun2("mul_add", |_ctx, a: Var<I64Type>, b: Var<I64Type>| {
            add(mul(a, b), a)
        });

        // 3 * 5 + 3 = 18
        let expr = call2(mul_add, 3i64, 5i64);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 18);
    }

    #[test]
    fn test_fun3_clamp() {
        use crate::func_impl::call3;

        let mut compiler = Compiler::new();

        // Define: clamp(x, min, max) = if x < min then min else (if x > max then max else x)
        let clamp_fn = compiler.fun3(
            "clamp",
            |_ctx, x: Var<I64Type>, min: Var<I64Type>, max: Var<I64Type>| {
                if_then_else(
                    lt(x, min),
                    min,
                    if_then_else(lt(max, x), max, x), // x > max == max < x
                )
            },
        );

        // Test: clamp(5, 0, 10) = 5
        let expr1 = call3(clamp_fn, 5i64, 0i64, 10i64);
        let compiled1 = compiler.compile(expr1).expect("compilation failed");
        assert_eq!(compiled1.run(), 5);
    }

    #[test]
    fn test_fun3_clamp_at_min() {
        use crate::func_impl::call3;

        let mut compiler = Compiler::new();

        let clamp_fn = compiler.fun3(
            "clamp",
            |_ctx, x: Var<I64Type>, min: Var<I64Type>, max: Var<I64Type>| {
                if_then_else(lt(x, min), min, if_then_else(lt(max, x), max, x))
            },
        );

        // Test: clamp(-5, 0, 10) = 0 (clamped at min)
        let expr = call3(clamp_fn, Const::<I64Type>::new(-5), 0i64, 10i64);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 0);
    }

    #[test]
    fn test_fun2_rec_gcd() {
        use crate::func_impl::call2;

        let mut compiler = Compiler::new();

        // Define: gcd(a, b) = if b == 0 then a else gcd(b, a % b)
        // Note: We'll use a different implementation since we don't have modulo
        // gcd(a, b) = if b == 0 then a else gcd(b, a - b * (a / b))
        let gcd = compiler.fun2_rec(
            "gcd",
            |f, _ctx, a: Var<I64Type>, b: Var<I64Type>| {
                if_then_else(
                    eq(b, 0i64),
                    a,
                    call2(f, b, sub(a, mul(b, div(a, b)))), // a % b = a - b * (a / b)
                )
            },
        );

        // gcd(48, 18) = 6
        let expr = call2(gcd, 48i64, 18i64);
        let compiled = compiler.compile(expr).expect("compilation failed");
        assert_eq!(compiled.run(), 6);
    }

    // #[test]
    // fn test_return_fun2_pointer() {
    //     use crate::func_impl::call2;
    //
    //     let mut compiler = Compiler::new();
    //
    //     // Define: add(a, b) = a + b
    //     let add_fn = compiler.fun2("add", |_ctx, a: Var<I64Type>, b: Var<I64Type>| {
    //         add(a, b)
    //     });
    //
    //     // Compile the function reference itself (not a call)
    //     let compiled = compiler.compile(add_fn).expect("compilation failed");
    //
    //     // Extract the function pointer
    //     let add_ptr = compiled.as_fn();
    //
    //     // Test the function with various inputs
    //     assert_eq!(add_ptr(10, 32), 42);
    //     assert_eq!(add_ptr(-5, 5), 0);
    //     assert_eq!(add_ptr(100, 200), 300);
    // }
}
