//! Simple Filter MVP demonstrating Futamura projection
//!
//! This module implements a basic staged filter that takes an array and a predicate,
//! and generates specialized Cranelift code for filtering. This demonstrates the
//! Futamura projection principle where an interpreter (filter logic) is specialized
//! for a specific program (the predicate) to produce compiled code.

use crate::staging::{
    control_flow::{StagedConditional, StagedForLoop},
    StagedArray, StagedBool, StagedU64, Staged, StagingError,
};
use cranelift_codegen::ir::{types, AbiParam, Function, Signature, InstBuilder};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

/// A staged predicate that can be evaluated on array elements
#[derive(Debug, Clone)]
pub enum StagedPredicate {
    /// Element equals a constant value
    EqualTo(u64),
    /// Element is greater than a constant value  
    GreaterThan(u64),
    /// Element is less than a constant value
    LessThan(u64),
}

impl StagedPredicate {
    /// Generate a boolean condition for the given array element
    pub fn evaluate(&self, element: StagedU64) -> StagedBool {
        match self {
            StagedPredicate::EqualTo(value) => {
                let constant = StagedU64::Constant(*value);
                element.eq(constant)
            }
            StagedPredicate::GreaterThan(value) => {
                let constant = StagedU64::Constant(*value);
                element.gt(constant)
            }
            StagedPredicate::LessThan(value) => {
                let constant = StagedU64::Constant(*value);
                element.lt(constant)
            }
        }
    }
}

/// The staged filter itself - this represents the "interpreter" being specialized
pub struct StagedFilter {
    predicate: StagedPredicate,
}

impl StagedFilter {
    pub fn new(predicate: StagedPredicate) -> Self {
        Self { predicate }
    }

    /// Generate specialized filter code using Futamura projection
    /// This takes the "program" (predicate) and generates compiled code
    /// Input: fn(input_ptr: *const u64, input_len: u64, output_ptr: *mut u64) -> u64  
    /// Returns: number of elements written to output
    pub fn generate_filter_function(&self) -> Result<Function, StagingError> {
        // Create function signature: fn(input_ptr, input_len, output_ptr) -> count
        let mut sig = Signature::new(CallConv::SystemV);
        sig.params.push(AbiParam::new(types::I64)); // input_ptr
        sig.params.push(AbiParam::new(types::I64)); // input_len
        sig.params.push(AbiParam::new(types::I64)); // output_ptr
        sig.returns.push(AbiParam::new(types::I64)); // count

        let mut func = Function::new();
        func.signature = sig;

        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        // Get function parameters
        let params = builder.block_params(entry_block);
        let input_ptr = params[0];
        let input_len = params[1];
        let output_ptr = params[2];

        // Create variables for our filter loop
        let count_var = Variable::from_u32(0);
        builder.declare_var(count_var, types::I64);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(count_var, zero);

        // Create staged loop from 0 to input_len
        let start = StagedU64::Constant(0);
        let end = StagedU64::Variable(crate::staging::StagedVariable::new(1, types::I64));
        
        // We need to create a variable for the length
        let len_var = Variable::from_u32(1);
        builder.declare_var(len_var, types::I64);
        builder.def_var(len_var, input_len);

        let filter_loop = StagedForLoop::new(start, end);

        // Generate the filter loop
        filter_loop.generate_loop(&mut builder, |builder, index| {
            // Load element at current index
            let index_val = index.codegen(builder);
            let element_size = builder.ins().iconst(types::I64, 8); // 8 bytes for u64
            let offset = builder.ins().imul(index_val, element_size);
            let element_ptr = builder.ins().iadd(input_ptr, offset);
            let element_val = builder.ins().load(types::I64, cranelift_codegen::ir::MemFlags::new(), element_ptr, 0);
            
            let staged_element = StagedU64::Variable(crate::staging::StagedVariable::new(2, types::I64));
            let element_var = Variable::from_u32(2);
            builder.declare_var(element_var, types::I64);
            builder.def_var(element_var, element_val);

            // Evaluate predicate
            let condition = self.predicate.evaluate(staged_element);
            let staged_cond = StagedConditional::new(condition);

            // If condition is true, write to output array
            staged_cond.generate_if_else(builder, 
                |builder| {
                    // Get current count and calculate output offset
                    let current_count = builder.use_var(count_var);
                    let output_offset = builder.ins().imul(current_count, element_size);
                    let output_element_ptr = builder.ins().iadd(output_ptr, output_offset);
                    
                    // Store the element
                    let element_to_store = builder.use_var(element_var);
                    builder.ins().store(cranelift_codegen::ir::MemFlags::new(), element_to_store, output_element_ptr, 0);
                    
                    // Increment count
                    let one = builder.ins().iconst(types::I64, 1);
                    let new_count = builder.ins().iadd(current_count, one);
                    builder.def_var(count_var, new_count);
                    
                    Ok(())
                },
                None::<fn(&mut FunctionBuilder) -> Result<(), StagingError>>, // No else branch needed
            )?;

            Ok(())
        })?;

        // Return the final count
        let final_count = builder.use_var(count_var);
        builder.ins().return_(&[final_count]);

        builder.seal_all_blocks();
        builder.finalize();

        Ok(func)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_filter_greater_than_5() {
        // Create a filter for elements > 5
        let predicate = StagedPredicate::GreaterThan(5);
        let filter = StagedFilter::new(predicate);
        
        // Generate the specialized filter function
        let func = filter.generate_filter_function().unwrap();
        
        // Check the generated IR
        let ir = func.display().to_string();
        println!("Filter IR (> 5):\n{}", ir);
        
        // Verify key elements of the generated code
        assert!(ir.contains("load"), "Should contain load instructions");
        assert!(ir.contains("store"), "Should contain store instructions"); 
        assert!(ir.contains("icmp"), "Should contain comparison instructions");
        assert!(ir.contains("brif"), "Should contain conditional branches");
        assert!(ir.contains("iconst.i64 5"), "Should contain the constant 5");
    }

    #[test]
    fn test_filter_equal_to_10() {
        // Create a filter for elements == 10
        let predicate = StagedPredicate::EqualTo(10);
        let filter = StagedFilter::new(predicate);
        
        // Generate the specialized filter function
        let func = filter.generate_filter_function().unwrap();
        
        // Check the generated IR
        let ir = func.display().to_string();
        println!("Filter IR (== 10):\n{}", ir);
        
        // Verify the constant appears in the IR
        assert!(ir.contains("iconst.i64 10"), "Should contain the constant 10");
        assert!(ir.contains("icmp eq"), "Should contain equality comparison");
    }
}