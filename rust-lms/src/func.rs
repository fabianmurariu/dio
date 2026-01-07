//! First-class function support for staged computations.
//!
//! This module provides `Fun1`, `Fun2`, etc. for creating staged functions
//! that compile to native function pointers.

use crate::staged::{CompilationContext, Staged, Var};
use crate::types::StagedType;
use cranelift_codegen::ir::{types, AbiParam, InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_module::{Linkage, Module};
use std::marker::PhantomData;
use std::ops::Deref;
use std::rc::Rc;

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

// Note: FunType1 does NOT implement ConstantType - functions cannot be constants!

// =============================================================================
// Fun1: Unary function definition
// =============================================================================

/// A staged unary function: A -> OUT
///
/// This represents a function that will be compiled to native code and can be
/// called from Rust.
pub struct Fun1<A, BODY> {
    name: String,
    body: BODY,
    _phantom: PhantomData<A>,
}

/// Constructor for unary functions
///
/// # Example
/// ```ignore
/// use rust_lms::prelude::*;
/// use rust_lms::func::fun1;
///
/// let square = fun1("square", |x: Var<I64Type>| {
///     mul(x, x)
/// });
/// ```
pub fn fun1<A, OUT, F, BODY>(name: &str, body_fn: F) -> Fun1<A, BODY>
where
    A: StagedType,
    OUT: StagedType,
    F: FnOnce(Var<A>) -> BODY,
    BODY: Staged<Out = OUT>,
{
    // Create the parameter variable (Variable ID 0 will be the function parameter)
    let param_var = Var::new(Variable::from_u32(0));

    // Call body_fn immediately to build the expression tree
    let body_expr = body_fn(param_var);

    Fun1 {
        name: name.to_string(),
        body: body_expr,
        _phantom: PhantomData,
    }
}

impl<A, BODY> Staged for Fun1<A, BODY>
where
    A: StagedType,
    BODY: Staged,
    BODY::Out: StagedType,
{
    type Out = FunType1<A, BODY::Out>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // Create function signature: A -> OUT
        let mut sig = ctx.module.make_signature();
        sig.params.push(AbiParam::new(A::cranelift_type()));
        sig.returns.push(AbiParam::new(BODY::Out::cranelift_type()));

        // Declare the function in the module
        let func_id = ctx
            .module
            .declare_function(&self.name, Linkage::Export, &sig)
            .expect("Failed to declare function");

        // Create a new function context and builder for this function
        let mut func_ctx = ctx.module.make_context();
        func_ctx.func.signature = sig.clone();

        {
            let mut builder_context = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut func_ctx.func, &mut builder_context);
            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            // Get the parameter value
            let param_value = builder.block_params(entry_block)[0];

            // Declare Variable 0 (which the body expects) and bind it to the parameter
            let param_var = builder.declare_var(A::cranelift_type());
            assert_eq!(param_var.as_u32(), 0, "Parameter variable must be ID 0");
            builder.def_var(param_var, param_value);

            // Generate the body code in a nested scope to drop the CompilationContext
            let result = {
                let mut body_ctx = CompilationContext {
                    builder: &mut builder,
                    module: ctx.module,
                };
                self.body.codegen(&mut body_ctx)
            };

            // Return the result
            builder.ins().return_(&[result]);

            // Create entry block with parameter
            builder.finalize();
        }

        // Define the function in the module
        ctx.module
            .define_function(func_id, &mut func_ctx)
            .expect("Failed to define function");

        ctx.module.clear_context(&mut func_ctx);

        // Return the function pointer as an i64
        // Note: We need to finalize the module before we can get the function pointer
        // In a real implementation, this would be handled at a higher level
        // For now, we return a placeholder (the function ID cast to i64)
        ctx.builder
            .ins()
            .iconst(types::I64, func_id.as_u32() as i64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prelude::*;
    use cranelift_jit::{JITBuilder, JITModule};
    use cranelift_module::{default_libcall_names, Module};

    #[test]
    fn test_fun1_square() {
        // Create a function: square(x) = x * x
        let _square_fn = fun1("square", |x: Var<I64Type>| mul(x, x));

        // Test that the function definition compiles successfully
        // (Full integration testing would require more infrastructure)
    }

    #[test]
    fn test_fun1_increment() {
        // Create a function: inc(x) = x + 1
        let _inc_fn = fun1("increment", |x: Var<I64Type>| add(x, Const::new(1)));

        // Test that the function definition compiles successfully
        // (Full integration testing would require more infrastructure)
    }
}
