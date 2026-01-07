//! First-class function support for staged computations.
//!
//! This module provides:
//! - `Compiler`: The central coordinator that owns function and variable definitions
//! - `FunRef<A, OUT>`: A Copy-able handle to a function definition
//! - `Call1<F, ARG>`: Function call expression
//! - `Compiled<T>`: The result of compilation, owns the JIT module

use crate::staged::{CompilationContext, Staged, VarRef};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, FuncId, Linkage, Module};
use std::collections::HashMap;
use std::marker::PhantomData;

// =============================================================================
// FunType1: Type marker for unary functions
// =============================================================================

/// Type marker for unary functions: A -> OUT
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FunType1<A, OUT> {
    _phantom: PhantomData<(A, OUT)>,
}

impl<A: StagedType, OUT: StagedType> StagedType for FunType1<A, OUT> {
    type RuntimeValue = fn(A::RuntimeValue) -> OUT::RuntimeValue;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        // Function pointers are represented as i64 (pointer-sized)
        types::I64
    }
}

// =============================================================================
// FunRef: Copy-able handle to a function definition
// =============================================================================

/// A lightweight reference to a staged function.
///
/// `FunRef<A, OUT>` is just an index into the Compiler's function tracking.
/// It's always Copy, enabling easy reuse in expressions.
#[derive(Clone, Copy)]
pub struct FunRef<A: StagedType, OUT: StagedType> {
    pub(crate) id: usize,
    _phantom: PhantomData<(A, OUT)>,
}

impl<A: StagedType, OUT: StagedType> FunRef<A, OUT> {
    fn new(id: usize) -> Self {
        FunRef {
            id,
            _phantom: PhantomData,
        }
    }
}

impl<A: StagedType, OUT: StagedType> std::fmt::Debug for FunRef<A, OUT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "FunRef({})", self.id)
    }
}

// =============================================================================
// Call1: Function call expression
// =============================================================================

/// A function call expression: applies a function to an argument.
///
/// This implements `Staged<Out = OUT>` where OUT is the function's return type.
#[derive(Clone)]
pub struct Call1<F, ARG> {
    func: F,
    arg: ARG,
}

impl<A, OUT, ARG> Staged for Call1<FunRef<A, OUT>, ARG>
where
    A: StagedType,
    OUT: StagedType,
    ARG: Staged<Out = A>,
{
    type Out = OUT;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Look up the function ID in our map
        let func_id = ctx
            .func_map
            .get(&self.func.id)
            .expect(&format!("Function {} not found in func_map", self.func.id));

        // Declare the function for calling
        let func_ref = ctx
            .module
            .declare_func_in_func(*func_id, ctx.builder.func);

        // Generate code for the argument
        let arg_value = self.arg.codegen(ctx);

        // Generate the call
        let call = ctx.builder.ins().call(func_ref, &[arg_value]);
        ctx.builder.inst_results(call)[0]
    }
}

/// Create a function call expression
pub fn call1<A, OUT, ARG>(func: FunRef<A, OUT>, arg: ARG) -> Call1<FunRef<A, OUT>, ARG>
where
    A: StagedType,
    OUT: StagedType,
    ARG: Staged<Out = A>,
{
    Call1 { func, arg }
}

// =============================================================================
// Internal: FunDef - Stored function definition
// =============================================================================

/// Internal storage for a function definition (type-erased body)
struct FunDef {
    name: String,
    /// The body expression, type-erased but we know its signature
    body: Box<dyn FnOnce(&mut CompilationContext) -> Value>,
    /// Parameter type for Cranelift
    param_type: cranelift_codegen::ir::Type,
    /// Return type for Cranelift
    return_type: cranelift_codegen::ir::Type,
    /// The variable ID used for the parameter (for var_map lookup)
    param_var_id: usize,
}

// =============================================================================
// Compiler: Owns everything, coordinates compilation
// =============================================================================

/// The central coordinator for staged computations.
///
/// `Compiler` owns all function definitions and variable IDs. It provides
/// methods to create functions and variables, and to compile expressions
/// to native code.
pub struct Compiler {
    /// Function definitions indexed by ID
    functions: Vec<Option<FunDef>>,
    /// Next variable ID to assign
    next_var_id: usize,
}

impl Default for Compiler {
    fn default() -> Self {
        Self::new()
    }
}

impl Compiler {
    /// Create a new compiler
    pub fn new() -> Self {
        Compiler {
            functions: Vec::new(),
            next_var_id: 0,
        }
    }

    /// Create a new variable reference of the given type.
    ///
    /// The actual Cranelift Variable will be created during compilation.
    pub fn var<T: StagedType>(&mut self) -> VarRef<T> {
        let id = self.next_var_id;
        self.next_var_id += 1;
        VarRef::new(id)
    }

    /// Define a unary function.
    ///
    /// The body function is called immediately to build the expression tree.
    /// No Cranelift calls happen until `compile()` is called.
    pub fn fun1<A, OUT, F, BODY>(&mut self, name: &str, body_fn: F) -> FunRef<A, OUT>
    where
        A: StagedType,
        OUT: StagedType,
        F: FnOnce(VarRef<A>) -> BODY,
        BODY: Staged<Out = OUT> + 'static,
    {
        // Create the parameter variable (ID 0 within this function's scope)
        // We use the compiler's var counter to ensure unique IDs across functions
        let param_id = self.next_var_id;
        self.next_var_id += 1;
        let param_var = VarRef::<A>::new(param_id);

        // Call body_fn immediately to build the expression tree
        let body_expr = body_fn(param_var);

        // Store the function definition
        let func_id = self.functions.len();
        let func_def = FunDef {
            name: name.to_string(),
            body: Box::new(move |ctx: &mut CompilationContext| {
                // During codegen, the parameter variable should already be defined
                body_expr.codegen(ctx)
            }),
            param_type: A::cranelift_type(),
            return_type: OUT::cranelift_type(),
            param_var_id: param_id,
        };

        self.functions.push(Some(func_def));

        FunRef::new(func_id)
    }

    /// Compile an expression to native code.
    ///
    /// This compiles all referenced functions and the main expression,
    /// returning a `Compiled<T>` that owns the JIT module and can extract
    /// the computed value.
    pub fn compile<S: Staged + 'static>(self, expr: S) -> Result<Compiled<S::Out>, CompileError> {
        // Create the JIT module
        let builder = JITBuilder::new(default_libcall_names())
            .map_err(|e| CompileError::JitError(e.to_string()))?;
        let mut module = JITModule::new(builder);

        // First pass: declare all functions
        let mut func_map: HashMap<usize, FuncId> = HashMap::new();

        for (id, func_opt) in self.functions.iter().enumerate() {
            if let Some(func_def) = func_opt {
                let mut sig = module.make_signature();
                sig.params.push(AbiParam::new(func_def.param_type));
                sig.returns.push(AbiParam::new(func_def.return_type));

                let func_id = module
                    .declare_function(&func_def.name, Linkage::Local, &sig)
                    .map_err(|e| CompileError::ModuleError(e.to_string()))?;

                func_map.insert(id, func_id);
            }
        }

        // Declare the main function
        let mut main_sig = module.make_signature();
        main_sig.returns.push(AbiParam::new(S::Out::cranelift_type()));

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
                sig.params.push(AbiParam::new(func_def.param_type));
                sig.returns.push(AbiParam::new(func_def.return_type));

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

                    // Get the parameter value
                    let param_value = builder.block_params(entry_block)[0];

                    // Declare the parameter variable and bind it
                    let param_var = builder.declare_var(func_def.param_type);
                    builder.def_var(param_var, param_value);

                    // Create var_map with the parameter using the stored param_var_id
                    let mut var_map: HashMap<usize, Variable> = HashMap::new();
                    var_map.insert(func_def.param_var_id, param_var);

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

                    builder.ins().return_(&[result]);
                    builder.finalize();
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

                builder.ins().return_(&[result]);
                builder.finalize();
            }

            module
                .define_function(main_func_id, &mut func_ctx)
                .map_err(|e| CompileError::ModuleError(e.to_string()))?;
            module.clear_context(&mut func_ctx);
        }

        // Finalize the module
        module.finalize_definitions().map_err(|e| CompileError::ModuleError(e.to_string()))?;

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
pub struct Compiled<T: StagedType> {
    #[allow(dead_code)]
    module: JITModule,
    main_ptr: *const u8,
    _phantom: PhantomData<T>,
}

impl<T: StagedType> Compiled<T> {
    /// Execute the compiled code and return the result.
    ///
    /// # Safety
    /// This is safe as long as the compilation was done correctly.
    pub fn run(&self) -> T::RuntimeValue
    where
        T::RuntimeValue: Copy,
    {
        let func: fn() -> T::RuntimeValue = unsafe { std::mem::transmute(self.main_ptr) };
        func()
    }
}

// Special implementation for function types to get the function pointer
impl<A: StagedType, OUT: StagedType> Compiled<FunType1<A, OUT>> {
    /// Get the compiled function as a callable function pointer.
    pub fn as_fn(&self) -> fn(A::RuntimeValue) -> OUT::RuntimeValue {
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
        let expr = add(Const::<I64Type>::new(3), Const::<I64Type>::new(4));

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
        let square = compiler.fun1("square", |x: VarRef<I64Type>| mul(x, x));

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
        let _unused = compiler.var::<I64Type>();

        // Define: double(x) = x + x
        let double = compiler.fun1("double", |x: VarRef<I64Type>| add(x, x));

        // Call: double(7) = 14
        let expr = call1(double, Const::<I64Type>::new(7));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();

        assert_eq!(result, 14);
    }
}
