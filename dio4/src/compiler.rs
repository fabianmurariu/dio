//! Cranelift compilation and execution module for dio4
//!
//! This module provides the compilation and JIT execution capabilities for
//! staged functions, completing the Futamura projection by taking generated
//! Cranelift IR and producing executable machine code.

use crate::staging::StagingError;
use cranelift_codegen::ir::Function;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_codegen::{self as codegen, Context};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::collections::HashMap;
use thiserror::Error;

/// Errors that can occur during compilation
#[derive(Error, Debug)]
pub enum CompilerError {
    #[error("ISA creation failed: {reason}")]
    IsaCreationFailed { reason: String },

    #[error("Function compilation failed: {reason}")]
    CompilationFailed { reason: String },

    #[error("Function finalization failed: {reason}")]
    FinalizationFailed { reason: String },

    #[error("Staging error: {0}")]
    StagingError(#[from] StagingError),
}

/// JIT compiler for staged functions
pub struct StagedCompiler {
    context: Context,
    module: JITModule,
}

impl StagedCompiler {
    /// Create a new staged compiler
    pub fn new() -> Result<Self, CompilerError> {
        let isa = cranelift_native::builder()
            .map_err(|e| CompilerError::IsaCreationFailed {
                reason: e.to_string(),
            })?
            .finish(settings::Flags::new(settings::builder()))
            .map_err(|e| CompilerError::IsaCreationFailed {
                reason: e.to_string(),
            })?;

        let builder = JITBuilder::with_isa(isa, default_libcall_names());
        let module = JITModule::new(builder);

        Ok(Self {
            context: module.make_context(),
            module,
        })
    }

    /// Compile a Cranelift function to machine code and return a callable function pointer
    pub fn compile_function(&mut self, func: Function) -> Result<CompiledFunction, CompilerError> {
        self.context.func = func;

        // Declare the function first
        let func_id = self
            .module
            .declare_function("staged_func", Linkage::Export, &self.context.func.signature)
            .map_err(|e| CompilerError::CompilationFailed {
                reason: e.to_string(),
            })?;

        // Compile the function
        self.module
            .define_function(func_id, &mut self.context)
            .map_err(|e| CompilerError::CompilationFailed {
                reason: e.to_string(),
            })?;

        // Clear the function for reuse
        self.context.func.clear();

        // Finalize the compilation
        self.module
            .finalize_definitions()
            .map_err(|e| CompilerError::FinalizationFailed {
                reason: e.to_string(),
            })?;

        // Get the function pointer
        let func_ptr = self.module.get_finalized_function(func_id);

        Ok(CompiledFunction { func_ptr })
    }
}

/// A compiled function that can be executed
pub struct CompiledFunction {
    func_ptr: *const u8,
}

impl CompiledFunction {
    /// Execute the compiled filter function
    /// 
    /// # Safety
    /// 
    /// This function is unsafe because it calls compiled machine code. The caller
    /// must ensure that:
    /// - The input and output pointers are valid
    /// - The input_len matches the actual size of the input array
    /// - The output array has sufficient capacity for filtered results
    pub unsafe fn execute_filter(
        &self,
        input_ptr: *const u64,
        input_len: u64,
        output_ptr: *mut u64,
    ) -> u64 {
        let func: extern "C" fn(*const u64, u64, *mut u64) -> u64 =
            std::mem::transmute(self.func_ptr);
        func(input_ptr, input_len, output_ptr)
    }
}

/// High-level execution context for staged operations
pub struct ExecutionContext {
    compiler: StagedCompiler,
    compiled_functions: HashMap<String, CompiledFunction>,
}

impl ExecutionContext {
    /// Create a new execution context
    pub fn new() -> Result<Self, CompilerError> {
        Ok(Self {
            compiler: StagedCompiler::new()?,
            compiled_functions: HashMap::new(),
        })
    }

    /// Compile and cache a function for later execution
    pub fn compile_function(
        &mut self,
        name: String,
        func: Function,
    ) -> Result<&CompiledFunction, CompilerError> {
        let compiled = self.compiler.compile_function(func)?;
        self.compiled_functions.insert(name.clone(), compiled);
        Ok(self.compiled_functions.get(&name).unwrap())
    }

    /// Execute a compiled filter function with safe array handling
    pub fn execute_filter(
        &self,
        function_name: &str,
        input: &[u64],
        output: &mut Vec<u64>,
    ) -> Result<usize, CompilerError> {
        let compiled_func = self.compiled_functions.get(function_name).ok_or(
            CompilerError::CompilationFailed {
                reason: format!("Function {} not found", function_name),
            },
        )?;

        // Ensure output has sufficient capacity
        output.clear();
        output.reserve(input.len());

        let result_count = unsafe {
            compiled_func.execute_filter(
                input.as_ptr(),
                input.len() as u64,
                output.as_mut_ptr(),
            )
        };

        // Set the actual length of the output vector
        unsafe {
            output.set_len(result_count as usize);
        }

        Ok(result_count as usize)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::filter::{StagedFilter, StagedPredicate};

    #[test]
    fn test_end_to_end_filter_compilation() {
        // Create a filter for elements > 5
        let predicate = StagedPredicate::GreaterThan(5);
        let filter = StagedFilter::new(predicate);

        // Generate the function
        let func = filter.generate_filter_function().unwrap();

        // Compile it
        let mut context = ExecutionContext::new().unwrap();
        context.compile_function("filter_gt_5".to_string(), func).unwrap();

        // Test it with data
        let input = vec![1, 3, 6, 8, 2, 10, 4];
        let mut output = Vec::new();

        let count = context
            .execute_filter("filter_gt_5", &input, &mut output)
            .unwrap();

        println!("Input: {:?}", input);
        println!("Output: {:?}", output);
        println!("Count: {}", count);

        // Verify the results
        let expected = vec![6, 8, 10];
        assert_eq!(output, expected);
        assert_eq!(count, 3);
    }

    #[test]
    fn test_end_to_end_equality_filter() {
        // Create a filter for elements == 42
        let predicate = StagedPredicate::EqualTo(42);
        let filter = StagedFilter::new(predicate);

        // Generate and compile
        let func = filter.generate_filter_function().unwrap();
        let mut context = ExecutionContext::new().unwrap();
        context.compile_function("filter_eq_42".to_string(), func).unwrap();

        // Test with data containing 42
        let input = vec![1, 42, 3, 42, 5, 42];
        let mut output = Vec::new();

        let count = context
            .execute_filter("filter_eq_42", &input, &mut output)
            .unwrap();

        println!("Input: {:?}", input);
        println!("Output: {:?}", output);
        println!("Count: {}", count);

        // Verify results
        let expected = vec![42, 42, 42];
        assert_eq!(output, expected);
        assert_eq!(count, 3);
    }
}