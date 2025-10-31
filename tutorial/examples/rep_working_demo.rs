//! Working demonstration of Rep<T> compilation and execution
//!
//! This example shows that Rep<T> can actually compile and execute functions,
//! not just build ASTs.

use cranelift_codegen::ir::{types, AbiParam, Function, InstBuilder, Signature, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::ops::{Add, Mul, Sub};

// =============================================================================
// Rep<T> Infrastructure (copied from rep_example.rs)
// =============================================================================

#[derive(Clone)]
pub enum Rep<T: Staged> {
    Constant(T::RuntimeValue),
    Variable(Variable),
    BinOp(Box<Rep<T>>, BinOpKind, Box<Rep<T>>),
}

#[derive(Clone, Debug)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
}

pub trait Staged: 'static + Clone {
    type RuntimeValue: Clone;
    fn cranelift_type() -> cranelift_codegen::ir::Type;
    fn codegen_constant(value: &Self::RuntimeValue, builder: &mut FunctionBuilder) -> Value;
}

pub trait SupportsBinOp: Staged {
    fn codegen_binop(
        kind: BinOpKind,
        left: Value,
        right: Value,
        builder: &mut FunctionBuilder,
    ) -> Value;
}

#[derive(Clone)]
pub struct I64Type;

impl Staged for I64Type {
    type RuntimeValue = i64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn codegen_constant(value: &i64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value)
    }
}

impl SupportsBinOp for I64Type {
    fn codegen_binop(
        kind: BinOpKind,
        left: Value,
        right: Value,
        builder: &mut FunctionBuilder,
    ) -> Value {
        match kind {
            BinOpKind::Add => builder.ins().iadd(left, right),
            BinOpKind::Sub => builder.ins().isub(left, right),
            BinOpKind::Mul => builder.ins().imul(left, right),
        }
    }
}

impl<T: SupportsBinOp> Add for Rep<T> {
    type Output = Rep<T>;

    fn add(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Add, Box::new(rhs))
    }
}

impl<T: SupportsBinOp> Sub for Rep<T> {
    type Output = Rep<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Sub, Box::new(rhs))
    }
}

impl<T: SupportsBinOp> Mul for Rep<T> {
    type Output = Rep<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        Rep::BinOp(Box::new(self), BinOpKind::Mul, Box::new(rhs))
    }
}

impl<T: Staged> Rep<T> {
    pub fn constant(value: T::RuntimeValue) -> Self {
        Rep::Constant(value)
    }

    pub fn variable(var: Variable) -> Self {
        Rep::Variable(var)
    }
}

impl<T: SupportsBinOp> Rep<T> {
    pub fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            Rep::Constant(val) => T::codegen_constant(val, builder),
            Rep::Variable(var) => builder.use_var(*var),
            Rep::BinOp(left, op, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                T::codegen_binop(op.clone(), left_val, right_val, builder)
            }
        }
    }
}

pub type RepI64 = Rep<I64Type>;

// =============================================================================
// Mini Compiler
// =============================================================================

pub fn compile_and_run() -> Result<(), String> {
    // Create JIT module
    let isa = cranelift_native::builder()
        .map_err(|e| format!("Failed to create ISA: {}", e))?
        .finish(settings::Flags::new(settings::builder()))
        .map_err(|e| format!("Failed to finish ISA: {}", e))?;

    let builder = JITBuilder::with_isa(isa, default_libcall_names());
    let mut module = JITModule::new(builder);

    // Create function signature: (i64) -> i64
    let mut sig = Signature::new(CallConv::SystemV);
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));

    // Create function
    let mut func = Function::new();
    func.signature = sig;

    let mut func_ctx = FunctionBuilderContext::new();
    let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

    // Create entry block
    let entry_block = builder.create_block();
    builder.append_block_params_for_function_params(entry_block);
    builder.switch_to_block(entry_block);

    // Get parameter
    let x_val = builder.block_params(entry_block)[0];
    let x_var = Variable::from_u32(0);
    builder.declare_var(x_var, types::I64);
    builder.def_var(x_var, x_val);

    // Build staged computation: f(x) = (x + 5) * 2
    println!("Building staged expression: f(x) = (x + 5) * 2");
    let x = RepI64::variable(x_var);
    let five = RepI64::constant(5);
    let two = RepI64::constant(2);
    let result_expr = (x + five) * two;

    // Generate code
    println!("Generating Cranelift IR...");
    let result_val = result_expr.codegen(&mut builder);

    // Return
    builder.ins().return_(&[result_val]);
    builder.seal_all_blocks();
    builder.finalize();

    // Compile
    println!("Compiling to machine code...");
    let mut ctx = Context::new();
    ctx.func = func;

    let func_id = module
        .declare_function("test_func", Linkage::Export, &ctx.func.signature)
        .map_err(|e| format!("Failed to declare: {}", e))?;

    module
        .define_function(func_id, &mut ctx)
        .map_err(|e| format!("Failed to define: {}", e))?;

    module.clear_context(&mut ctx);
    module
        .finalize_definitions()
        .map_err(|e| format!("Failed to finalize: {}", e))?;

    let code_ptr = module.get_finalized_function(func_id);

    // Execute!
    println!("\nExecuting compiled function:");
    unsafe {
        let func: extern "C" fn(i64) -> i64 = std::mem::transmute(code_ptr);

        let test_cases = vec![0, 1, 5, 10, -3];
        for input in test_cases {
            let output = func(input);
            let expected = (input + 5) * 2;
            println!("  f({}) = {} (expected: {})", input, output, expected);
            assert_eq!(output, expected, "Mismatch for input {}", input);
        }
    }

    println!("\n✅ All tests passed! Rep<T> works!");

    Ok(())
}

fn main() {
    println!("=== Rep<T> Working Demonstration ===\n");

    match compile_and_run() {
        Ok(()) => println!("\n🎉 Success!"),
        Err(e) => eprintln!("❌ Error: {}", e),
    }
}