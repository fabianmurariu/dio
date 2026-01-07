//! Code generation and JIT compilation infrastructure.
//!
//! This module contains the `Compiler` and `CompilerBuilder` types that handle
//! the process of compiling staged expressions into executable machine code
//! using Cranelift.

use cranelift_codegen::ir::{
    types, AbiParam, FuncRef, Function, InstBuilder, MemFlags, Signature,
};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::collections::HashMap; // Still needed for external_funcs in compile_nary
use std::sync::Arc;

use crate::expr::Expr;
use crate::num::StagedI64;
use crate::runtime::CompiledNary;
use crate::{DataType, StagedBuilder, StagingError};

/// A JIT compiler that can compile staged functions to machine code
///
/// The Compiler owns a `JITModule` that gets transferred to `CompiledNary` via Arc.
/// Each `Compiler` can only compile one function, after which it is consumed.
/// This ensures that compiled code remains valid as long as the `CompiledNary`
/// holding a reference to the module exists.
pub struct Compiler {
    pub(crate) module: JITModule,
    pub(crate) external_functions: crate::ffi::ExternalFunctionRegistry,
}

/// Builder for creating a Compiler with pre-registered external function symbols
///
/// This wraps Cranelift's `JITBuilder` and provides a convenient builder pattern
/// for configuring the compiler before creation.
pub struct CompilerBuilder {
    jit_builder: JITBuilder,
}

impl CompilerBuilder {
    /// Create a new compiler builder with the native ISA
    pub fn new() -> Result<Self, StagingError> {
        let isa = cranelift_native::builder()
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to create ISA: {}", e),
            })?
            .finish(settings::Flags::new(settings::builder()))
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to finish ISA: {}", e),
            })?;

        Ok(Self {
            jit_builder: JITBuilder::with_isa(isa, default_libcall_names()),
        })
    }

    /// Register an external function symbol with the JIT
    ///
    /// This registers the function pointer with Cranelift's JIT module.
    /// You'll still need to call `register_external_signature()` after building
    /// to register the type signature for type checking during compilation.
    ///
    /// # Example
    /// ```ignore
    /// extern "C" fn my_add(x: i64, y: i64) -> i64 { x + y }
    ///
    /// let compiler = CompilerBuilder::new()?
    ///     .with_symbol("my_add", my_add as *const u8)
    ///     .build()?;
    /// ```
    pub fn with_symbol(mut self, name: &str, fn_ptr: *const u8) -> Self {
        self.jit_builder.symbol(name, fn_ptr);
        self
    }

    /// Build the compiler with all registered symbols
    pub fn build(self) -> Result<Compiler, StagingError> {
        let module = JITModule::new(self.jit_builder);

        Ok(Compiler {
            module,
            external_functions: crate::ffi::ExternalFunctionRegistry::new(),
        })
    }
}

impl Compiler {
    /// Create a new compiler instance with no external functions
    ///
    /// For compilers that need external functions, use `CompilerBuilder::new()?`
    /// and register symbols before calling `.build()`.
    pub fn new() -> Result<Self, StagingError> {
        CompilerBuilder::new()?.build()
    }

    /// Register an external function signature for type checking
    ///
    /// This is separate from symbol registration (which happens via CompilerBuilder).
    /// The signature is used for type checking before Cranelift compilation.
    ///
    /// # Example
    /// ```ignore
    /// extern "C" fn my_add(x: i64, y: i64) -> i64 { x + y }
    ///
    /// let mut compiler = CompilerBuilder::new()
    ///     .with_symbol("my_add", my_add as *const u8)
    ///     .build()?;
    ///
    /// compiler.register_external_signature(
    ///     "my_add",
    ///     vec![DataType::I64, DataType::I64],
    ///     DataType::I64,
    /// );
    /// ```
    pub fn register_external_signature(
        &mut self,
        name: &str,
        params: Vec<DataType>,
        return_type: DataType,
    ) {
        // We pass a null pointer since the actual pointer was registered with JITBuilder
        // This is just for type checking
        self.external_functions.register(
            name.to_string(),
            params,
            return_type,
            std::ptr::null(),
        );
    }

    /// Compile a generic n-ary function with typed parameters and return value
    ///
    /// This is the primary compilation method that supports:
    /// - Multiple parameters of different types (scalars and arrays)
    /// - Generic return types
    /// - External function calls
    ///
    /// # Parameters
    /// - `param_types`: The types of the function parameters
    /// - `return_type`: The return type of the function
    /// - `body`: A closure that builds the function body using staged values
    ///
    /// # Returns
    /// A `CompiledNary` function that can be called with `ScalarValue` arguments
    ///
    /// # Example
    ///
    /// ```
    /// use tutorial::{Compiler, DataType, StagedU64, Expr};
    ///
    /// let mut compiler = Compiler::new().unwrap();
    /// let compiled = compiler.compile_nary(
    ///     vec![DataType::U64, DataType::U64],
    ///     DataType::U64,
    ///     |_, vars| {
    ///         let x = StagedU64::variable(vars[0]);
    ///         let y = StagedU64::variable(vars[1]);
    ///         Expr::U64(x + y)
    ///     }
    /// ).unwrap();
    /// ```
    pub fn compile_nary<E: Into<Expr>>(
        mut self,
        param_types: Vec<DataType>,
        return_type: DataType,
        body: impl FnOnce(&mut StagedBuilder, &[Variable]) -> E,
    ) -> Result<CompiledNary, StagingError> {
        // Count total variables needed (scalars use 1 var, arrays use 2: ptr + len)
        let mut total_vars = 0;
        let mut total_slots = 0;
        for param_type in &param_types {
            match param_type {
                DataType::Array { .. } => {
                    total_vars += 2; // ptr and len
                    total_slots += 2; // ptr and len in parameter array
                }
                _ => {
                    total_vars += 1;
                    total_slots += 1;
                }
            }
        }

        // Create function signature: *const u64 -> <return_type>
        // We pass a pointer to an array of u64 values. Each scalar takes 1 slot,
        // each array takes 2 slots (pointer, length).
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64)); // pointer (as i64) to u64 array
        sig.returns
            .push(AbiParam::new(return_type.to_cranelift_type())); // return value

        // Create the function
        let mut func = Function::new();
        func.signature = sig;

        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create entry block with parameter (pointer to array)
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let params_ptr = builder.block_params(entry_block)[0];

        // Load each parameter from the u64 array and assign to variables
        // Scalars: single value
        // Arrays: two values (pointer, length)
        let mut param_vars = Vec::new();
        let mut slot_offset = 0;
        let mut var_id = 0u32;

        for param_type in &param_types {
            match param_type {
                DataType::Array { .. } => {
                    // Arrays take 2 slots: pointer and length

                    // Load pointer - Cranelift 0.127+: declare_var returns the Variable
                    // Variable IDs are now assigned by Cranelift, not by us
                    let ptr_var = builder.declare_var(types::I64);
                    var_id += 1;
                    let ptr_offset = builder.ins().iconst(types::I64, (slot_offset * 8) as i64);
                    let ptr_addr = builder.ins().iadd(params_ptr, ptr_offset);
                    let ptr_val = builder.ins().load(types::I64, MemFlags::trusted(), ptr_addr, 0);
                    builder.def_var(ptr_var, ptr_val);
                    slot_offset += 1;

                    // Load length - Cranelift 0.127+: declare_var returns the Variable
                    // Variable IDs are now assigned by Cranelift, not by us
                    let len_var = builder.declare_var(types::I64);
                    var_id += 1;
                    let len_offset = builder.ins().iconst(types::I64, (slot_offset * 8) as i64);
                    let len_addr = builder.ins().iadd(params_ptr, len_offset);
                    let len_val = builder.ins().load(types::I64, MemFlags::trusted(), len_addr, 0);
                    builder.def_var(len_var, len_val);
                    slot_offset += 1;

                    // For backward compatibility, add both to param_vars
                    // (The user will need to know arrays use 2 consecutive vars)
                    param_vars.push(ptr_var);
                    param_vars.push(len_var);
                }
                _ => {
                    // Scalars take 1 slot - Cranelift 0.127+: declare_var returns the Variable
                    // Variable IDs are now assigned by Cranelift, not by us
                    let cranelift_type = param_type.to_cranelift_type();
                    let var = builder.declare_var(cranelift_type);
                    var_id += 1;

                    let byte_offset = slot_offset * 8;
                    let offset = builder.ins().iconst(types::I64, byte_offset as i64);
                    let param_addr = builder.ins().iadd(params_ptr, offset);
                    let param_val = builder
                        .ins()
                        .load(cranelift_type, MemFlags::trusted(), param_addr, 0);
                    builder.def_var(var, param_val);
                    param_vars.push(var);
                    slot_offset += 1;
                }
            }
        }

        // Declare and import all registered external functions using the Module API
        // This works with symbols pre-registered via JITBuilder::symbol()
        let mut external_funcs: HashMap<String, FuncRef> = HashMap::new();
        for (func_name, (func_sig, _func_ptr)) in self.external_functions.iter() {
            // Create Cranelift signature for this external function
            let mut sig = Signature::new(CallConv::SystemV);
            for param_type in &func_sig.params {
                sig.params.push(AbiParam::new(param_type.to_cranelift_type()));
            }
            sig.returns
                .push(AbiParam::new(func_sig.return_type.to_cranelift_type()));

            // Declare the function as an import in the module
            // This links to the symbol we registered with JITBuilder::symbol()
            let func_id = self
                .module
                .declare_function(func_name, Linkage::Import, &sig)
                .map_err(|e| StagingError::CompilationFailed {
                    reason: format!("Failed to declare external function '{}': {}", func_name, e),
                })?;

            // Import the declared function into the current function being built
            let func_ref = self.module.declare_func_in_func(func_id, builder.func);
            external_funcs.insert(func_name.clone(), func_ref);
        }

        // Generate the function body using StagedBuilder
        let mut staged_builder = StagedBuilder::new();
        let result_expr = body(&mut staged_builder, &param_vars);
        let result_expr = result_expr.into();

        // Debug output: expression tree
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            eprintln!("\n========== EXPRESSION TREE ==========");
            eprintln!("{}", result_expr);
            eprintln!("=====================================\n");
        }

        // Verify return type matches
        if result_expr.data_type() != &return_type {
            return Err(StagingError::TypeMismatch {
                expected: format!("{:?}", return_type),
                actual: format!("{:?}", result_expr.data_type()),
            });
        }

        let result_val = result_expr.codegen_with_externals(&mut builder, Some(&external_funcs));

        // Return the result
        builder.ins().return_(&[result_val]);

        // Finalize
        builder.seal_all_blocks();
        builder.finalize();

        // Debug output if requested
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            eprintln!("\n========== CRANELIFT IR ==========");
            eprintln!("{}", func);
            eprintln!("==================================\n");
        }

        // Compile to machine code
        let mut ctx = Context::new();
        ctx.func = func;

        let func_id = self
            .module
            .declare_function("staged_func_nary", Linkage::Export, &ctx.func.signature)
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to declare function: {}", e),
            })?;

        self.module
            .define_function(func_id, &mut ctx)
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to define function: {}", e),
            })?;

        self.module.clear_context(&mut ctx);
        self.module
            .finalize_definitions()
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to finalize: {}", e),
            })?;

        let code_ptr = self.module.get_finalized_function(func_id);

        // Wrap the module in Arc and transfer ownership to CompiledNary
        // This ensures the compiled code remains valid as long as CompiledNary exists
        Ok(CompiledNary::new(
            Arc::new(self.module),
            code_ptr,
            param_types,
            return_type,
        ))
    }

    /// Compile a staged function that takes multiple i64 parameters (as a slice) and returns i64
    ///
    /// This is a convenience wrapper around `compile_nary` for the common case of
    /// i64-only functions. It now uses the same `StagedBuilder` API as `compile_nary`.
    ///
    /// # Example
    ///
    /// To compile `f(x, y, z) = (x + y) * z`:
    ///
    /// ```
    /// use tutorial::{Compiler, StagedI64, Expr};
    ///
    /// let mut compiler = Compiler::new().unwrap();
    /// let mut compiled = compiler.compile_nary_i64(3, |_, vars| {
    ///     let x = StagedI64::variable(vars[0]);
    ///     let y = StagedI64::variable(vars[1]);
    ///     let z = StagedI64::variable(vars[2]);
    ///     let sum = x + y;
    ///     Expr::I64(sum * z)
    /// }).unwrap();
    ///
    /// assert_eq!(compiled.call_i64(&[2, 3, 4]), 20); // (2 + 3) * 4 = 20
    /// ```
    pub fn compile_nary_i64<E: Into<Expr>>(
        self,
        num_params: usize,
        body: impl FnOnce(&mut StagedBuilder, &[Variable]) -> E,
    ) -> Result<CompiledNary, StagingError> {
        // Create parameter types: all i64
        let param_types = vec![DataType::I64; num_params];

        // Delegate to compile_nary - this now uses the same API
        self.compile_nary(param_types, DataType::I64, body)
    }
}
