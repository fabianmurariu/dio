//! # Tutorial: Building a Partial Evaluation Compiler with Rep<T> (LMS-style)
//!
//! This is an alternative version of the tutorial that uses a generic Rep<T> type
//! similar to Scala LMS's Rep[T]. This demonstrates a more advanced abstraction
//! that unifies all staged types under a single generic type.

use cranelift_codegen::ir::{types, AbiParam, Function, InstBuilder, MemFlags, Signature, Value};
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings;
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::ops::{Add, Mul, Sub};
use thiserror::Error;

// =============================================================================
// CORE ABSTRACTION: Rep<T> - Generic Staged Computation
// =============================================================================

/// Represents a staged computation that will produce a value of type T at runtime
///
/// This is the Rust equivalent of Scala LMS's Rep[T]. Instead of having separate
/// types like StagedI64, StagedU64, etc., we have a single generic type Rep<T>
/// where T is a phantom type that indicates what kind of value we're staging.
#[derive(Clone)]
pub enum Rep<T: Staged> {
    /// A constant value known at compile time
    Constant(T::RuntimeValue),
    /// A variable (function parameter) known only at runtime
    Variable(Variable),
    /// A binary operation on two staged values
    BinOp(Box<Rep<T>>, BinOpKind, Box<Rep<T>>),
}

#[derive(Clone, Debug)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
}

// =============================================================================
// TRAIT SYSTEM: What types can be staged?
// =============================================================================

/// Core trait: Types that can participate in staged computation
///
/// This trait defines what it means for a type to be stageable.
/// Any type T that implements Staged can be used with Rep<T>.
pub trait Staged: 'static + Clone {
    /// The actual runtime type (e.g., i64 for I64Type)
    type RuntimeValue: Clone;

    /// Get the Cranelift type representation
    fn cranelift_type() -> cranelift_codegen::ir::Type;

    /// Generate code for a constant value
    fn codegen_constant(value: &Self::RuntimeValue, builder: &mut FunctionBuilder) -> Value;
}

/// Extended trait: Types that support binary operations
///
/// Not all types support all operations (e.g., can't add booleans).
/// This trait allows us to conditionally implement Add, Sub, Mul
/// only for types that support these operations.
pub trait SupportsBinOp: Staged {
    /// Generate code for a binary operation
    fn codegen_binop(
        kind: BinOpKind,
        left: Value,
        right: Value,
        builder: &mut FunctionBuilder,
    ) -> Value;
}

// =============================================================================
// PHANTOM TYPES: Marker types for different staged types
// =============================================================================

/// Marker type for i64 values
///
/// This is a zero-sized type that exists only at compile time.
/// It's used as a type parameter to Rep<T> to indicate we're
/// working with i64 values.
#[derive(Clone)]
pub struct I64Type;

#[derive(Clone)]
pub struct U64Type;

#[derive(Clone)]
pub struct BoolType;

// =============================================================================
// IMPLEMENT STAGED FOR CONCRETE TYPES
// =============================================================================

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
            BinOpKind::Div => builder.ins().sdiv(left, right),
        }
    }
}

impl Staged for U64Type {
    type RuntimeValue = u64;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I64
    }

    fn codegen_constant(value: &u64, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *value as i64)
    }
}

impl SupportsBinOp for U64Type {
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
            BinOpKind::Div => builder.ins().udiv(left, right),
        }
    }
}

impl Staged for BoolType {
    type RuntimeValue = bool;

    fn cranelift_type() -> cranelift_codegen::ir::Type {
        types::I8
    }

    fn codegen_constant(value: &bool, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I8, if *value { 1 } else { 0 })
    }
}

// =============================================================================
// OPERATOR OVERLOADING: Make Rep<T> work with +, -, *, etc.
// =============================================================================

// Key insight: We only implement Add for Rep<T> where T: SupportsBinOp
// This means you can't accidentally add booleans (which don't implement SupportsBinOp)
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

// =============================================================================
// HELPER METHODS FOR Rep<T>
// =============================================================================

impl<T: Staged> Rep<T> {
    /// Create a constant staged value
    pub fn constant(value: T::RuntimeValue) -> Self {
        Rep::Constant(value)
    }

    /// Create a variable staged value (represents a function parameter)
    pub fn variable(var: Variable) -> Self {
        Rep::Variable(var)
    }
}

impl<T: SupportsBinOp> Rep<T> {
    /// Generate Cranelift IR code for this staged computation
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

// =============================================================================
// TYPE ALIASES FOR CONVENIENCE
// =============================================================================

/// Staged i64 value - use this instead of writing Rep<I64Type>
pub type RepI64 = Rep<I64Type>;

/// Staged u64 value
pub type RepU64 = Rep<U64Type>;

/// Staged boolean value
pub type RepBool = Rep<BoolType>;

// =============================================================================
// ERROR HANDLING
// =============================================================================

#[derive(Error, Debug)]
pub enum StagingError {
    #[error("Type mismatch: expected {expected}, got {actual}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Compilation failed: {reason}")]
    CompilationFailed { reason: String },

    #[error("Execution failed: {reason}")]
    ExecutionFailed { reason: String },
}

// =============================================================================
// COMPILER: Generic over staged types!
// =============================================================================

pub struct Compiler {
    module: JITModule,
}

impl Compiler {
    pub fn new() -> Result<Self, StagingError> {
        let isa = cranelift_native::builder()
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to create ISA: {}", e),
            })?
            .finish(settings::Flags::new(settings::builder()))
            .map_err(|e| StagingError::CompilationFailed {
                reason: format!("Failed to finish ISA: {}", e),
            })?;

        let builder = JITBuilder::with_isa(isa, default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self { module })
    }

    /// Compile a staged function taking N parameters of type T, returning type T
    ///
    /// This is the GENERIC version - it works for any type T!
    /// Notice how we don't need separate compile_nary_i64, compile_nary_u64, etc.
    /// We have one compile_nary<T> that works for all types!
    pub fn compile_nary<T>(
        &mut self,
        num_params: usize,
        body: impl FnOnce(&mut FunctionBuilder, &[Variable]) -> Rep<T>,
    ) -> Result<CompiledNary<T>, StagingError>
    where
        T: SupportsBinOp,
    {
        // Create function signature: *const T -> T
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64)); // pointer to params array
        sig.returns.push(AbiParam::new(T::cranelift_type())); // return value

        // Create the function
        let mut func = Function::new();
        func.signature = sig;

        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create entry block
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let params_ptr = builder.block_params(entry_block)[0];

        // Load each parameter from the array
        let mut param_vars = Vec::new();
        for i in 0..num_params {
            let var = Variable::from_u32(i as u32);
            builder.declare_var(var, T::cranelift_type());

            // Compute element size based on type
            let element_size = match T::cranelift_type() {
                t if t == types::I64 => 8,
                t if t == types::I32 => 4,
                t if t == types::I8 => 1,
                _ => 8, // default to 8 bytes
            };

            let offset = builder.ins().iconst(types::I64, (i * element_size) as i64);
            let param_addr = builder.ins().iadd(params_ptr, offset);
            let param_val = builder
                .ins()
                .load(T::cranelift_type(), MemFlags::trusted(), param_addr, 0);

            builder.def_var(var, param_val);
            param_vars.push(var);
        }

        // Generate the function body
        let result_expr = body(&mut builder, &param_vars);
        let result_val = result_expr.codegen(&mut builder);

        // Return the result
        builder.ins().return_(&[result_val]);

        // Finalize
        builder.seal_all_blocks();
        builder.finalize();

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
        self.module.finalize_definitions().map_err(|e| {
            StagingError::CompilationFailed {
                reason: format!("Failed to finalize: {}", e),
            }
        })?;

        let code_ptr = self.module.get_finalized_function(func_id);

        Ok(CompiledNary {
            code_ptr,
            num_params,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Convenience method for i64 - delegates to the generic version
    pub fn compile_nary_i64(
        &mut self,
        num_params: usize,
        body: impl FnOnce(&mut FunctionBuilder, &[Variable]) -> RepI64,
    ) -> Result<CompiledNaryI64, StagingError> {
        self.compile_nary::<I64Type>(num_params, body)
            .map(|compiled| CompiledNaryI64 {
                code_ptr: compiled.code_ptr,
                num_params: compiled.num_params,
            })
    }
}

// =============================================================================
// COMPILED FUNCTIONS
// =============================================================================

/// A compiled function that takes N parameters of type T and returns T
pub struct CompiledNary<T: Staged> {
    code_ptr: *const u8,
    num_params: usize,
    _phantom: std::marker::PhantomData<T>,
}

/// Specialized version for i64 (for compatibility with existing tests)
pub struct CompiledNaryI64 {
    code_ptr: *const u8,
    num_params: usize,
}

impl CompiledNaryI64 {
    pub fn call(&self, args: &[i64]) -> i64 {
        assert!(
            args.len() >= self.num_params,
            "Expected at least {} arguments, got {}",
            self.num_params,
            args.len()
        );

        unsafe {
            let func: extern "C" fn(*const i64) -> i64 = std::mem::transmute(self.code_ptr);
            func(args.as_ptr())
        }
    }

    pub fn num_params(&self) -> usize {
        self.num_params
    }
}

// =============================================================================
// TESTS - Same tests as before, but now using operator overloading!
// =============================================================================

#[cfg(test)]]
mod tests {
    use super::*;

    #[test]
    fn test_lesson1_constant_addition() {
        // Notice how we can now use the + operator directly!
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary_i64(1, |_builder, vars| {
                let x = RepI64::variable(vars[0]);
                let five = RepI64::constant(5);
                x + five // Natural syntax!
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 15);
        assert_eq!(compiled.call(&[0]), 5);
        assert_eq!(compiled.call(&[-3]), 2);
    }

    #[test]
    fn test_lesson1_double_addition() {
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary_i64(1, |_builder, vars| {
                let x = RepI64::variable(vars[0]);
                let x2 = RepI64::variable(vars[0]);
                x + x2
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 20);
        assert_eq!(compiled.call(&[7]), 14);
    }

    #[test]
    fn test_lesson1_nested_addition() {
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary_i64(1, |_builder, vars| {
                let x = RepI64::variable(vars[0]);
                let one = RepI64::constant(1);
                let two = RepI64::constant(2);
                (x + one) + two // Parentheses work naturally!
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 13);
        assert_eq!(compiled.call(&[0]), 3);
    }

    #[test]
    fn test_lesson1b_ternary_expression() {
        // Shows off the power of operator overloading!
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary_i64(3, |_builder, vars| {
                let x = RepI64::variable(vars[0]);
                let y = RepI64::variable(vars[1]);
                let z = RepI64::variable(vars[2]);
                (x + y) * z // Looks just like regular Rust!
            })
            .unwrap();

        assert_eq!(compiled.call(&[2, 3, 4]), 20);
        assert_eq!(compiled.call(&[10, 5, 2]), 30);
        assert_eq!(compiled.call(&[1, 1, 100]), 200);
    }

    #[test]
    fn test_lesson2_simple_subtraction() {
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary_i64(1, |_builder, vars| {
                let x = RepI64::variable(vars[0]);
                let three = RepI64::constant(3);
                x - three
            })
            .unwrap();

        assert_eq!(compiled.call(&[10]), 7);
        assert_eq!(compiled.call(&[5]), 2);
        assert_eq!(compiled.call(&[0]), -3);
    }

    #[test]
    fn test_lesson3_complex_expression() {
        // This is beautiful with operator overloading!
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary_i64(1, |_builder, vars| {
                let x1 = RepI64::variable(vars[0]);
                let x2 = RepI64::variable(vars[0]);
                let five = RepI64::constant(5);
                let two = RepI64::constant(2);
                (x1 + five) * (x2 - two)
            })
            .unwrap();

        assert_eq!(compiled.call(&[3]), 8);
        assert_eq!(compiled.call(&[4]), 18);
    }
}