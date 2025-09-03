//! End-to-end execution engine integrating pipeline compilation with JIT execution
//!
//! This module provides the complete vertical slice from pipeline description
//! to executable machine code, demonstrating the full Futamura projection.

use crate::pipeline::{CompiledPipeline, PipelineBuilder};
use crate::operators::ComparisonOp;
use crate::staging::StagingError;
use cranelift_codegen::ir::Function;
use cranelift_codegen::settings;
use cranelift_codegen::Context;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use thiserror::Error;

/// Errors that can occur during execution
#[derive(Error, Debug)]
pub enum ExecutionError {
    #[error("Pipeline compilation failed: {0}")]
    CompilationFailed(#[from] StagingError),
    
    #[error("JIT compilation failed: {reason}")]
    JitCompilationFailed { reason: String },
    
    #[error("Function execution failed: {reason}")]
    ExecutionFailed { reason: String },
}

/// Complete execution engine that compiles and runs query pipelines
pub struct QueryEngine {
    jit_module: JITModule,
    context: Context,
}

impl QueryEngine {
    /// Create a new query engine
    pub fn new() -> Result<Self, ExecutionError> {
        let isa = cranelift_native::builder()
            .map_err(|e| ExecutionError::JitCompilationFailed {
                reason: format!("ISA creation failed: {}", e),
            })?
            .finish(settings::Flags::new(settings::builder()))
            .map_err(|e| ExecutionError::JitCompilationFailed {
                reason: format!("ISA finalization failed: {}", e),
            })?;

        let builder = JITBuilder::with_isa(isa, default_libcall_names());
        let jit_module = JITModule::new(builder);

        Ok(Self {
            context: jit_module.make_context(),
            jit_module,
        })
    }

    /// Compile a pipeline into executable machine code
    pub fn compile_pipeline(&mut self, pipeline: CompiledPipeline) -> Result<ExecutableQuery, ExecutionError> {
        // Set up the Cranelift function
        self.context.func = pipeline.function;

        // Declare and define the function in the JIT module
        let func_id = self
            .jit_module
            .declare_function("query_func", Linkage::Export, &self.context.func.signature)
            .map_err(|e| ExecutionError::JitCompilationFailed {
                reason: format!("Function declaration failed: {}", e),
            })?;

        self.jit_module
            .define_function(func_id, &mut self.context)
            .map_err(|e| ExecutionError::JitCompilationFailed {
                reason: format!("Function definition failed: {}", e),
            })?;

        // Clear for reuse
        self.context.func.clear();

        // Finalize the JIT compilation
        self.jit_module
            .finalize_definitions()
            .map_err(|e| ExecutionError::JitCompilationFailed {
                reason: format!("JIT finalization failed: {}", e),
            })?;

        // Get the compiled function pointer
        let func_ptr = self.jit_module.get_finalized_function(func_id);

        Ok(ExecutableQuery {
            func_ptr,
            input_schema: pipeline.input_schema,
            output_column: pipeline.output_column,
        })
    }
}

/// An executable query that can be run on actual data
pub struct ExecutableQuery {
    func_ptr: *const u8,
    input_schema: crate::operators::Schema,
    output_column: usize,
}

impl ExecutableQuery {
    /// Execute the query on input columns, returning filtered results
    ///
    /// # Safety
    /// This function is unsafe because it calls compiled machine code.
    pub unsafe fn execute(&self, input_columns: &[&[u64]], output_buffer: &mut Vec<u64>) -> Result<usize, ExecutionError> {
        if input_columns.is_empty() {
            return Ok(0);
        }

        // Verify all columns have the same length
        let length = input_columns[0].len();
        for (i, col) in input_columns.iter().enumerate() {
            if col.len() != length {
                return Err(ExecutionError::ExecutionFailed {
                    reason: format!("Column {} length {} doesn't match expected length {}", i, col.len(), length),
                });
            }
        }

        // Prepare output buffer
        output_buffer.clear();
        output_buffer.reserve(length);

        // Build function arguments: (col0_ptr, col1_ptr, ..., length, output_ptr)
        let mut args = Vec::new();
        for col in input_columns {
            args.push(col.as_ptr() as i64);
        }
        args.push(length as i64);
        args.push(output_buffer.as_mut_ptr() as i64);

        // Call the compiled function based on the number of input columns
        let result_count = match input_columns.len() {
            1 => {
                let func: extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(self.func_ptr);
                func(args[0], args[1], args[2])
            }
            2 => {
                let func: extern "C" fn(i64, i64, i64, i64) -> i64 = std::mem::transmute(self.func_ptr);
                func(args[0], args[1], args[2], args[3])
            }
            3 => {
                let func: extern "C" fn(i64, i64, i64, i64, i64) -> i64 = std::mem::transmute(self.func_ptr);
                func(args[0], args[1], args[2], args[3], args[4])
            }
            4 => {
                let func: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 = std::mem::transmute(self.func_ptr);
                func(args[0], args[1], args[2], args[3], args[4], args[5])
            }
            _ => {
                return Err(ExecutionError::ExecutionFailed {
                    reason: format!("Unsupported number of input columns: {}", input_columns.len()),
                });
            }
        };

        // Set the actual length of the output buffer
        output_buffer.set_len(result_count as usize);

        Ok(result_count as usize)
    }

    /// Get a description of what this query does
    pub fn description(&self) -> String {
        format!(
            "Compiled query: outputs column '{}' based on filter conditions",
            self.input_schema.columns[self.output_column].name
        )
    }
}

/// High-level API for the complete vertical slice: Pipeline -> Compiled Function -> Execution
pub struct QueryProcessor {
    engine: QueryEngine,
}

impl QueryProcessor {
    pub fn new() -> Result<Self, ExecutionError> {
        Ok(Self {
            engine: QueryEngine::new()?,
        })
    }

    /// Execute a complete scan + filter pipeline
    /// 
    /// Example: Scan(Filter(col2 > 12)) -> compiled function -> execution
    pub fn scan_filter(
        &mut self,
        column_names: Vec<String>,
        filter_column: usize,
        comparison: ComparisonOp,
        threshold: u64,
        output_column: usize,
        input_data: &[&[u64]],
    ) -> Result<Vec<u64>, ExecutionError> {
        // Step 1: Build the pipeline description
        let pipeline = PipelineBuilder::scan(column_names)
            .filter(filter_column, comparison, threshold)
            .compile(output_column)?;

        println!("Generated pipeline: {}", pipeline.signature_description());
        
        // Step 2: Compile to executable machine code
        let executable = self.engine.compile_pipeline(pipeline)?;
        
        println!("Compiled executable: {}", executable.description());

        // Step 3: Execute on actual data
        let mut results = Vec::new();
        let count = unsafe { executable.execute(input_data, &mut results)? };
        
        println!("Execution completed: {} results", count);
        
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_vertical_slice() {
        let mut processor = QueryProcessor::new().unwrap();

        // Test data: 3 columns of sample data
        let col0_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let col1_data = vec![10, 15, 8, 25, 30, 12, 18, 22, 5, 35];
        let col2_data = vec![5, 10, 15, 20, 25, 30, 8, 12, 18, 40];

        let input_columns = vec![&col0_data[..], &col1_data[..], &col2_data[..] ];

        // Execute: Scan(Filter(col2 > 12)) -> output col2 values
        let results = processor.scan_filter(
            vec!["id".to_string(), "age".to_string(), "score".to_string()],
            2,                                    // filter on col2 (score)
            ComparisonOp::GreaterThan,           // >
            12,                                  // threshold
            2,                                   // output col2 (score)
            &input_columns,
        ).unwrap();

        println!("Input data:");
        println!("  col0 (id): {:?}", col0_data);
        println!("  col1 (age): {:?}", col1_data);
        println!("  col2 (score): {:?}", col2_data);
        println!("Filter: score > 12");
        println!("Results: {:?}", results);

        // Expected: values from col2 where col2 > 12
        // col2: [5, 10, 15, 20, 25, 30, 8, 12, 18, 40]
        //         -   -  ✓   ✓   ✓   ✓  -   -   ✓   ✓
        let expected = vec![15, 20, 25, 30, 18, 40];
        assert_eq!(results, expected);
    }

    #[test]
    fn test_equality_filter() {
        let mut processor = QueryProcessor::new().unwrap();

        let data = vec![1, 5, 3, 5, 2, 5, 4];
        let input_columns = vec![&data[..]];

        // Filter for values equal to 5
        let results = processor.scan_filter(
            vec!["values".to_string()],
            0,                                    // filter on col0
            ComparisonOp::Equal,                 // ==
            5,                                   // threshold
            0,                                   // output col0
            &input_columns,
        ).unwrap();

        println!("Input: {:?}", data);
        println!("Filter: value == 5");  
        println!("Results: {:?}", results);

        let expected = vec![5, 5, 5]; // Three 5's in the input
        assert_eq!(results, expected);
    }

    #[test]
    fn test_no_matches() {
        let mut processor = QueryProcessor::new().unwrap();

        let data = vec![1, 2, 3, 4, 5];
        let input_columns = vec![&data[..]];

        // Filter for values greater than 10 (no matches)
        let results = processor.scan_filter(
            vec!["values".to_string()],
            0,
            ComparisonOp::GreaterThan,
            10,
            0,
            &input_columns,
        ).unwrap();

        println!("Input: {:?}", data);
        println!("Filter: value > 10");
        println!("Results: {:?}", results);

        assert_eq!(results, Vec::<u64>::new()); // No matches
    }

    #[test]
    fn test_all_matches() {
        let mut processor = QueryProcessor::new().unwrap();

        let data = vec![10, 20, 30, 40, 50];
        let input_columns = vec![&data[..]];

        // Filter for values greater than 5 (all match)
        let results = processor.scan_filter(
            vec!["values".to_string()],
            0,
            ComparisonOp::GreaterThan,
            5,
            0,
            &input_columns,
        ).unwrap();

        println!("Input: {:?}", data);
        println!("Filter: value > 5");
        println!("Results: {:?}", results);

        assert_eq!(results, data); // All values match
    }
}