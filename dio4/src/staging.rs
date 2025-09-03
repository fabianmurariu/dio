//! # Staging Framework for Dio v4
//!
//! This module implements the core staging types that enable compile-time code generation
//! using Futamura projections. The staging system allows building Cranelift IR at compile
//! time rather than runtime, eliminating interpretation overhead.

use cranelift_codegen::ir::{types, InstBuilder, Type, UserFuncName, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use std::marker::PhantomData;
use thiserror::Error;

/// Errors that can occur during staging
#[derive(Error, Debug, Clone)]
pub enum StagingError {
    #[error("Type mismatch in staging operation: expected {{expected:?}}, got {{actual:?}}")]
    TypeMismatch { expected: String, actual: String },

    #[error("Unsupported staging operation: {{operation}} with operand types [{{operand_types}}]")]
    UnsupportedOperation {
        operation: String,
        operand_types: String,
    },

    #[error("Code generation failed: {{reason}}")]
    CodeGenerationFailed { reason: String },

    #[error("Memory layout error: required alignment {{required}}, got {{actual}}")]
    MemoryLayoutError { required: usize, actual: usize },
}

/// Core trait for staged values that can generate Cranelift code
pub trait Staged {
    /// The runtime type this staged value represents
    type RuntimeType;

    /// Generate Cranelift IR code that produces this value at runtime
    fn codegen(&self, builder: &mut FunctionBuilder) -> Value;

    /// Get the Cranelift type for this staged value
    fn cranelift_type() -> Type;
}

/// Trait for primitive types that can be used in staging
pub trait StagedPrimitive: Staged + Clone {
    /// Convert a runtime value to a constant staged value
    fn constant(value: Self::RuntimeType) -> StagedConstant<Self::RuntimeType>;
}

/// A staged constant value known at compile time
#[derive(Debug, Clone)]
pub struct StagedConstant<T> {
    value: T,
    _phantom: PhantomData<T>,
}

impl<T> StagedConstant<T> {
    pub fn new(value: T) -> Self {
        Self {
            value,
            _phantom: PhantomData,
        }
    }

    pub fn value(&self) -> &T {
        &self.value
    }
}

/// Staged 64-bit unsigned integer
#[derive(Debug, Clone)]
pub enum StagedU64 {
    Constant(u64),
    Variable(StagedVariable),
    Add(Box<StagedU64>, Box<StagedU64>),
    Mul(Box<StagedU64>, Box<StagedU64>),
    DirectValue(Value), // Hold a Cranelift Value directly
}

impl StagedU64 {
    pub fn constant(value: u64) -> Self {
        StagedU64::Constant(value)
    }

    pub fn variable(var: StagedVariable) -> Self {
        StagedU64::Variable(var)
    }

    pub fn add(left: StagedU64, right: StagedU64) -> Self {
        StagedU64::Add(Box::new(left), Box::new(right))
    }

    pub fn mul(left: StagedU64, right: StagedU64) -> Self {
        StagedU64::Mul(Box::new(left), Box::new(right))
    }

    /// Generate equality comparison
    pub fn eq(self, other: StagedU64) -> StagedBool {
        StagedBool::U64Equal(Box::new(self), Box::new(other))
    }

    /// Generate greater-than comparison  
    pub fn gt(self, other: StagedU64) -> StagedBool {
        StagedBool::U64GreaterThan(Box::new(self), Box::new(other))
    }

    /// Generate less-than comparison
    pub fn lt(self, other: StagedU64) -> StagedBool {
        StagedBool::U64LessThan(Box::new(self), Box::new(other))
    }
}

impl Staged for StagedU64 {
    type RuntimeType = u64;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            StagedU64::Constant(value) => builder.ins().iconst(types::I64, *value as i64),
            StagedU64::Variable(var) => var.codegen(builder),
            StagedU64::Add(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().iadd(left_val, right_val)
            }
            StagedU64::Mul(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().imul(left_val, right_val)
            }
            StagedU64::DirectValue(value) => *value,
        }
    }

    fn cranelift_type() -> Type {
        types::I64
    }
}

impl StagedPrimitive for StagedU64 {
    fn constant(value: Self::RuntimeType) -> StagedConstant<Self::RuntimeType> {
        StagedConstant::new(value)
    }
}

/// Staged 64-bit signed integer
#[derive(Debug, Clone)]
pub enum StagedI64 {
    Constant(i64),
    Variable(StagedVariable),
    Add(Box<StagedI64>, Box<StagedI64>),
    Sub(Box<StagedI64>, Box<StagedI64>),
    Mul(Box<StagedI64>, Box<StagedI64>),
}

impl StagedI64 {
    pub fn constant(value: i64) -> Self {
        StagedI64::Constant(value)
    }

    pub fn variable(var: StagedVariable) -> Self {
        StagedI64::Variable(var)
    }

    pub fn add(left: StagedI64, right: StagedI64) -> Self {
        StagedI64::Add(Box::new(left), Box::new(right))
    }

    pub fn sub(left: StagedI64, right: StagedI64) -> Self {
        StagedI64::Sub(Box::new(left), Box::new(right))
    }

    pub fn mul(left: StagedI64, right: StagedI64) -> Self {
        StagedI64::Mul(Box::new(left), Box::new(right))
    }
}

impl Staged for StagedI64 {
    type RuntimeType = i64;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        match self {
            StagedI64::Constant(value) => builder.ins().iconst(types::I64, *value),
            StagedI64::Variable(var) => var.codegen(builder),
            StagedI64::Add(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().iadd(left_val, right_val)
            }
            StagedI64::Sub(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().isub(left_val, right_val)
            }
            StagedI64::Mul(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().imul(left_val, right_val)
            }
        }
    }

    fn cranelift_type() -> Type {
        types::I64
    }
}

impl StagedPrimitive for StagedI64 {
    fn constant(value: Self::RuntimeType) -> StagedConstant<Self::RuntimeType> {
        StagedConstant::new(value)
    }
}

impl Staged for u64 {
    type RuntimeType = u64;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *self as i64)
    }

    fn cranelift_type() -> Type {
        types::I64
    }
}

impl StagedPrimitive for u64 {
    fn constant(value: Self::RuntimeType) -> StagedConstant<Self::RuntimeType> {
        StagedConstant::new(value)
    }
}

impl Staged for i64 {
    type RuntimeType = i64;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        builder.ins().iconst(types::I64, *self)
    }

    fn cranelift_type() -> Type {
        types::I64
    }
}

impl StagedPrimitive for i64 {
    fn constant(value: Self::RuntimeType) -> StagedConstant<Self::RuntimeType> {
        StagedConstant::new(value)
    }
}

/// Staged boolean value
#[derive(Debug, Clone)]
pub enum StagedBool {
    Constant(bool),
    Variable(StagedVariable),
    LessThan(Box<StagedI64>, Box<StagedI64>),
    Equal(Box<StagedI64>, Box<StagedI64>),
    And(Box<StagedBool>, Box<StagedBool>),
    Or(Box<StagedBool>, Box<StagedBool>),
    Not(Box<StagedBool>),
    // U64 comparisons
    U64Equal(Box<StagedU64>, Box<StagedU64>),
    U64GreaterThan(Box<StagedU64>, Box<StagedU64>),
    U64LessThan(Box<StagedU64>, Box<StagedU64>),
}

impl StagedBool {
    pub fn constant(value: bool) -> Self {
        StagedBool::Constant(value)
    }

    pub fn variable(var: StagedVariable) -> Self {
        StagedBool::Variable(var)
    }

    pub fn less_than(left: StagedI64, right: StagedI64) -> Self {
        StagedBool::LessThan(Box::new(left), Box::new(right))
    }

    pub fn equal(left: StagedI64, right: StagedI64) -> Self {
        StagedBool::Equal(Box::new(left), Box::new(right))
    }

    pub fn and(left: StagedBool, right: StagedBool) -> Self {
        StagedBool::And(Box::new(left), Box::new(right))
    }

    pub fn or(left: StagedBool, right: StagedBool) -> Self {
        StagedBool::Or(Box::new(left), Box::new(right))
    }

    pub fn not(operand: StagedBool) -> Self {
        StagedBool::Not(Box::new(operand))
    }
}

impl Staged for StagedBool {
    type RuntimeType = bool;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        use cranelift_codegen::ir::condcodes::IntCC;

        match self {
            StagedBool::Constant(value) => {
                builder.ins().iconst(types::I8, if *value { 1 } else { 0 })
            }
            StagedBool::Variable(var) => var.codegen(builder),
            StagedBool::LessThan(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, left_val, right_val)
            }
            StagedBool::Equal(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::Equal, left_val, right_val)
            }
            StagedBool::And(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().band(left_val, right_val)
            }
            StagedBool::Or(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().bor(left_val, right_val)
            }
            StagedBool::Not(operand) => {
                let operand_val = operand.codegen(builder);
                builder.ins().bnot(operand_val)
            }
            StagedBool::U64Equal(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::Equal, left_val, right_val)
            }
            StagedBool::U64GreaterThan(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::UnsignedGreaterThan, left_val, right_val)
            }
            StagedBool::U64LessThan(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::UnsignedLessThan, left_val, right_val)
            }
        }
    }

    fn cranelift_type() -> Type {
        types::I8
    }
}

/// Represents a staged variable that will be bound to a Cranelift variable
#[derive(Debug, Clone)]
pub struct StagedVariable {
    pub id: u32,
    pub cranelift_type: Type,
}

impl StagedVariable {
    pub fn new(id: u32, cranelift_type: Type) -> Self {
        Self { id, cranelift_type }
    }
}

impl Staged for StagedVariable {
    type RuntimeType = (); // Variables don't have a specific runtime type

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        let var = Variable::from_u32(self.id);
        builder.use_var(var)
    }

    fn cranelift_type() -> Type {
        // This will be overridden by the specific variable's type
        types::I64
    }
}

/// Staged array reference with compile-time element type information
#[derive(Debug, Clone)]
pub struct StagedArray<T> {
    base_ptr: StagedPointer,
    length: StagedU64,
    _element_type: PhantomData<T>,
}

impl<T: Clone> StagedArray<T> {
    pub fn new(base_ptr: StagedPointer, length: StagedU64) -> Self {
        Self {
            base_ptr,
            length,
            _element_type: PhantomData,
        }
    }

    /// Get an element at the specified index
    pub fn get(&self, index: StagedU64) -> StagedElement<T> {
        StagedElement {
            array: self.clone(),
            index,
            _element_type: PhantomData,
        }
    }

    /// Get the length of this array
    pub fn len(&self) -> &StagedU64 {
        &self.length
    }
}

/// Staged pointer to memory
#[derive(Debug, Clone)]
pub struct StagedPointer {
    value: StagedU64,
}

impl StagedPointer {
    pub fn new(value: StagedU64) -> Self {
        Self { value }
    }

    pub fn from_constant_address(addr: u64) -> Self {
        Self {
            value: StagedU64::constant(addr),
        }
    }
}

impl Staged for StagedPointer {
    type RuntimeType = *const u8;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        self.value.codegen(builder)
    }

    fn cranelift_type() -> Type {
        types::I64 // Pointers are 64-bit on our target platforms
    }
}

/// Staged element access from an array
#[derive(Debug, Clone)]
pub struct StagedElement<T> {
    array: StagedArray<T>,
    index: StagedU64,
    _element_type: PhantomData<T>,
}

impl<T> StagedElement<T>
where
    T: StagedPrimitive + Clone,
{
    /// Generate code to load this element from memory
    pub fn load(&self, builder: &mut FunctionBuilder) -> Value {
        let base_ptr = self.array.base_ptr.codegen(builder);
        let index = self.index.codegen(builder);

        // Calculate element size based on type T
        let element_size = std::mem::size_of::<T::RuntimeType>() as i64;
        let offset = builder.ins().imul_imm(index, element_size);
        let addr = builder.ins().iadd(base_ptr, offset);

        // Load the value with the appropriate type
        use cranelift_codegen::ir::MemFlags;
        builder
            .ins()
            .load(T::cranelift_type(), MemFlags::trusted(), addr, 0)
    }

    /// Generate code to store a value to this element
    pub fn store(&self, value: Value, builder: &mut FunctionBuilder) {
        let base_ptr = self.array.base_ptr.codegen(builder);
        let index = self.index.codegen(builder);

        // Calculate element size based on type T
        let element_size = std::mem::size_of::<T::RuntimeType>() as i64;
        let offset = builder.ins().imul_imm(index, element_size);
        let addr = builder.ins().iadd(base_ptr, offset);

        // Store the value
        use cranelift_codegen::ir::MemFlags;
        builder.ins().store(MemFlags::trusted(), value, addr, 0);
    }
}

// Implement Staged for element access to u64 arrays
impl Staged for StagedElement<u64> {
    type RuntimeType = u64;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        self.load(builder)
    }

    fn cranelift_type() -> Type {
        types::I64
    }
}

// Implement Staged for element access to i64 arrays
impl Staged for StagedElement<i64> {
    type RuntimeType = i64;

    fn codegen(&self, builder: &mut FunctionBuilder) -> Value {
        self.load(builder)
    }

    fn cranelift_type() -> Type {
        types::I64
    }
}

/// Helper functions for creating staged operations
pub mod ops {
    use super::*;

    /// Create a staged addition of two i64 values
    pub fn staged_add_i64(left: StagedI64, right: StagedI64) -> StagedI64 {
        StagedI64::add(left, right)
    }

    /// Create a staged subtraction of two i64 values
    pub fn staged_sub_i64(left: StagedI64, right: StagedI64) -> StagedI64 {
        StagedI64::sub(left, right)
    }

    /// Create a staged multiplication of two i64 values
    pub fn staged_mul_i64(left: StagedI64, right: StagedI64) -> StagedI64 {
        StagedI64::mul(left, right)
    }

    /// Create a staged addition of two u64 values
    pub fn staged_add_u64(left: StagedU64, right: StagedU64) -> StagedU64 {
        StagedU64::add(left, right)
    }

    /// Create a staged multiplication of two u64 values
    pub fn staged_mul_u64(left: StagedU64, right: StagedU64) -> StagedU64 {
        StagedU64::mul(left, right)
    }

    /// Create a staged less-than comparison
    pub fn staged_less_than(left: StagedI64, right: StagedI64) -> StagedBool {
        StagedBool::less_than(left, right)
    }

    /// Create a staged equality comparison
    pub fn staged_equal(left: StagedI64, right: StagedI64) -> StagedBool {
        StagedBool::equal(left, right)
    }

    /// Create a staged logical AND
    pub fn staged_and(left: StagedBool, right: StagedBool) -> StagedBool {
        StagedBool::and(left, right)
    }

    /// Create a staged logical OR
    pub fn staged_or(left: StagedBool, right: StagedBool) -> StagedBool {
        StagedBool::or(left, right)
    }
}

/// Staged control flow operations for loops and conditionals
pub mod control_flow {
    use super::*;
    use cranelift_codegen::ir::{Block, InstBuilder};
    
    /// Represents a staged for-loop that generates Cranelift IR
    pub struct StagedForLoop {
        start: StagedU64,
        end: StagedU64,
    }
    
    impl StagedForLoop {
        /// Create a new staged for loop from start to end (exclusive)
        pub fn new(start: StagedU64, end: StagedU64) -> Self {
            Self { start, end }
        }
        
        /// Generate a for loop with a body function that receives the loop index
        pub fn generate_loop<F>(
            &self,
            builder: &mut FunctionBuilder,
            mut body: F,
        ) -> Result<(), StagingError>
        where
            F: FnMut(&mut FunctionBuilder, StagedU64) -> Result<(), StagingError>,
        {
            // Create blocks for loop structure
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();
            
            // Add loop variable parameter to header block
            builder.append_block_param(loop_header, types::I64);
            
            // Generate start value and jump to loop header
            let start_val = self.start.codegen(builder);
            builder.ins().jump(loop_header, &[start_val]);
            
            // Generate loop header (condition check)
            builder.switch_to_block(loop_header);
            let loop_var = builder.block_params(loop_header)[0];
            let end_val = self.end.codegen(builder);
            let condition = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::UnsignedLessThan, loop_var, end_val);
            builder.ins().brif(condition, loop_body, &[], loop_exit, &[]);
            
            // Generate loop body
            builder.switch_to_block(loop_body);
            // Use the loop variable directly as a StagedU64::DirectValue
            let staged_index = StagedU64::DirectValue(loop_var);
            body(builder, staged_index)?;
            
            // Increment and continue loop
            let one = builder.ins().iconst(types::I64, 1);
            let next_val = builder.ins().iadd(loop_var, one);
            builder.ins().jump(loop_header, &[next_val]);
            
            // Continue after loop
            builder.switch_to_block(loop_exit);
            Ok(())
        }
    }
    
    /// Staged conditional (if-then-else) operation
    pub struct StagedConditional {
        condition: StagedBool,
    }
    
    impl StagedConditional {
        pub fn new(condition: StagedBool) -> Self {
            Self { condition }
        }
        
        /// Generate if-then-else with optional else branch
        pub fn generate_if_else<ThenF, ElseF>(
            &self,
            builder: &mut FunctionBuilder,
            mut then_branch: ThenF,
            mut else_branch: Option<ElseF>,
        ) -> Result<(), StagingError>
        where
            ThenF: FnMut(&mut FunctionBuilder) -> Result<(), StagingError>,
            ElseF: FnMut(&mut FunctionBuilder) -> Result<(), StagingError>,
        {
            let then_block = builder.create_block();
            let else_block = builder.create_block();
            let merge_block = builder.create_block();
            
            let condition_val = self.condition.codegen(builder);
            builder.ins().brif(condition_val, then_block, &[], else_block, &[]);
            
            // Generate then branch
            builder.switch_to_block(then_block);
            then_branch(builder)?;
            builder.ins().jump(merge_block, &[]);
            
            // Generate else branch  
            builder.switch_to_block(else_block);
            if let Some(ref mut else_fn) = else_branch {
                else_fn(builder)?;
            }
            builder.ins().jump(merge_block, &[]);
            
            // Continue after conditional
            builder.switch_to_block(merge_block);
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::{Function, Signature, AbiParam};
    use cranelift_codegen::isa::CallConv;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    fn create_test_context() -> (Function, FunctionBuilderContext) {
        let mut sig = cranelift_codegen::ir::Signature::new(CallConv::SystemV);
        sig.returns
            .push(cranelift_codegen::ir::AbiParam::new(types::I64));

        let name = UserFuncName::testcase("test");
        let func = Function::with_name_signature(name, sig);

        let builder_ctx = FunctionBuilderContext::new();
        (func, builder_ctx)
    }

    #[test]
    fn test_staged_constant_i64() {
        let (mut func, mut builder_ctx) = create_test_context();
        let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);

        let entry = builder.create_block();
        builder.switch_to_block(entry);

        let staged_val = StagedI64::constant(42);
        let cranelift_val = staged_val.codegen(&mut builder);

        builder.ins().return_(&[cranelift_val]);
        builder.seal_block(entry);
        builder.finalize();

        // Verify the function was created successfully
        assert!(func.dfg.num_blocks() > 0);
    }

    #[test]
    fn test_staged_addition_generates_correct_ir() {
        let (mut func, mut builder_ctx) = create_test_context();
        let mut builder = FunctionBuilder::new(&mut func, &mut builder_ctx);

        let entry = builder.create_block();
        builder.switch_to_block(entry);

        let left = StagedI64::constant(10);
        let right = StagedI64::constant(20);
        let sum = StagedI64::add(left, right);
        let result = sum.codegen(&mut builder);

        builder.ins().return_(&[result]);
        builder.seal_block(entry);
        builder.finalize();

        let ir = func.display().to_string();
        println!("Generated IR:\n{}", ir);

        // Check that the generated IR contains the iadd instruction
        assert!(
            ir.contains("iadd"),
            "Generated IR should contain an iadd instruction"
        );

        // Check for the constant values
        assert!(
            ir.contains("iconst.i64 10"),
            "Generated IR should contain the constant 10"
        );
        assert!(
            ir.contains("iconst.i64 20"),
            "Generated IR should contain the constant 20"
        );
    }

    #[test]
    fn test_staged_loop_generation() {
        use super::control_flow::*;
        
        let mut sig = Signature::new(CallConv::SystemV);
        sig.returns.push(AbiParam::new(types::I64));
        
        let mut func = Function::new();
        func.signature = sig;
        
        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);
        
        let entry = builder.create_block();
        builder.switch_to_block(entry);
        
        // Create a simple loop from 0 to 3
        let start = StagedU64::Constant(0);
        let end = StagedU64::Constant(3);
        let loop_gen = StagedForLoop::new(start, end);
        
        let mut sum_var = Variable::from_u32(0);
        builder.declare_var(sum_var, types::I64);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(sum_var, zero);
        
        // Generate loop that accumulates index values
        loop_gen.generate_loop(&mut builder, |builder, index| {
            let current_sum = builder.use_var(sum_var);
            let index_val = index.codegen(builder);
            let new_sum = builder.ins().iadd(current_sum, index_val);
            builder.def_var(sum_var, new_sum);
            Ok(())
        }).unwrap();
        
        let final_sum = builder.use_var(sum_var);
        builder.ins().return_(&[final_sum]);
        
        // Seal all blocks
        builder.seal_all_blocks();
        builder.finalize();
        
        let ir = func.display().to_string();
        println!("Loop IR:\n{}", ir);
        
        // Check for loop constructs
        assert!(ir.contains("brif"), "Generated IR should contain conditional branches");
        assert!(ir.contains("jump"), "Generated IR should contain jumps");
        assert!(ir.contains("iadd"), "Generated IR should contain addition");
    }
}
