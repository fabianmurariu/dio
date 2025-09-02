//! # Staging Framework for Dio v4
//!
//! This module implements the core staging types that enable compile-time code generation
//! using Futamura projections. The staging system allows building Cranelift IR at compile
//! time rather than runtime, eliminating interpretation overhead.

use cranelift_codegen::ir::{types, Type, Value};
use cranelift_frontend::FunctionBuilder;
use std::marker::PhantomData;
use thiserror::Error;

/// Errors that can occur during staging
#[derive(Error, Debug, Clone)]
pub enum StagingError {
    #[error("Type mismatch in staging operation: expected {expected:?}, got {actual:?}")]
    TypeMismatch { expected: String, actual: String },
    
    #[error("Unsupported staging operation: {operation} with operand types [{operand_types}]")]
    UnsupportedOperation {
        operation: String,
        operand_types: String,
    },
    
    #[error("Code generation failed: {reason}")]
    CodeGenerationFailed { reason: String },
    
    #[error("Memory layout error: required alignment {required}, got {actual}")]
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
pub trait StagedPrimitive: Staged {
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
            StagedBool::Constant(value) => builder.ins().iconst(types::I8, if *value { 1 } else { 0 }),
            StagedBool::Variable(var) => var.codegen(builder),
            StagedBool::LessThan(left, right) => {
                let left_val = left.codegen(builder);
                let right_val = right.codegen(builder);
                builder.ins().icmp(IntCC::SignedLessThan, left_val, right_val)
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
        use cranelift_frontend::Variable;
        let var = Variable::new(self.id as usize);
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

impl<T> StagedArray<T> {
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
    T: StagedPrimitive,
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
        builder.ins().load(T::cranelift_type(), MemFlags::trusted(), addr, 0)
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

#[cfg(test)]
mod tests {
    use super::*;
    use cranelift_codegen::ir::Function;
    use cranelift_codegen::isa::{self, CallConv};
    use cranelift_codegen::settings;
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    
    fn create_test_builder() -> (Function, FunctionBuilder<'static>) {
        let mut sig = cranelift_codegen::ir::Signature::new(CallConv::SystemV);
        sig.returns.push(cranelift_codegen::ir::AbiParam::new(types::I64));
        
        let mut func = Function::with_name_signature(
            cranelift_codegen::ir::ExternalName::testcase("test"),
            sig,
        );
        
        let mut builder_ctx = FunctionBuilderContext::new();
        let builder = FunctionBuilder::new(&mut func, &mut builder_ctx);
        
        // This is a workaround for the lifetime issue - in real usage, 
        // we'd need to manage lifetimes properly
        unsafe { std::mem::transmute(builder) }
    }
    
    #[test]
    fn test_staged_constant_i64() {
        let (mut func, mut builder) = create_test_builder();
        let entry = builder.create_block();
        builder.switch_to_block(entry);
        
        let staged_val = StagedI64::constant(42);
        let cranelift_val = staged_val.codegen(&mut builder);
        
        builder.ins().return_(&[cranelift_val]);
        builder.finalize();
        
        // Verify the function was created successfully
        assert!(!func.dfg.blocks().is_empty());
    }
    
    #[test]
    fn test_staged_addition() {
        let (mut func, mut builder) = create_test_builder();
        let entry = builder.create_block();
        builder.switch_to_block(entry);
        
        let left = StagedI64::constant(10);
        let right = StagedI64::constant(20);
        let sum = StagedI64::add(left, right);
        let result = sum.codegen(&mut builder);
        
        builder.ins().return_(&[result]);
        builder.finalize();
        
        // Verify the function was created successfully
        assert!(!func.dfg.blocks().is_empty());
    }
    
    #[test]
    fn test_staged_bool_comparison() {
        let (mut func, mut builder) = create_test_builder();
        let entry = builder.create_block();
        builder.switch_to_block(entry);
        
        let left = StagedI64::constant(5);
        let right = StagedI64::constant(10);
        let comparison = StagedBool::less_than(left, right);
        let result = comparison.codegen(&mut builder);
        
        // Convert bool result back to i64 for return
        let extended = builder.ins().uextend(types::I64, result);
        builder.ins().return_(&[extended]);
        builder.finalize();
        
        // Verify the function was created successfully
        assert!(!func.dfg.blocks().is_empty());
    }
}