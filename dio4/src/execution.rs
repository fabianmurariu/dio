//! End-to-end execution engine integrating pipeline compilation with JIT execution
//!
//! This module provides the complete vertical slice from pipeline description
//! to executable machine code, demonstrating the full Futamura projection.

use crate::pipeline::{CompiledPipeline, PipelineBuilder};
use crate::operators::ComparisonOp;
use crate::staging::StagingError;
use arrow::array::ArrayRef;
use cranelift_codegen::ir::Function;
use cranelift_codegen::settings;
use cranelift_codegen::Context;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::env;
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
        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Starting JIT compilation");
            println!("[DEBUG] Input schema: {:?}", pipeline.input_schema);
            println!("[DEBUG] Output column: {}", pipeline.output_column);
        }
        
        // Set up the Cranelift function
        self.context.func = pipeline.function;

        // Declare and define the function in the JIT module
        let func_id = self
            .jit_module
            .declare_function("query_func", Linkage::Export, &self.context.func.signature)
            .map_err(|e| ExecutionError::JitCompilationFailed {
                reason: format!("Function declaration failed: {}", e),
            })?;

        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Function declared with ID: {:?}", func_id);
        }

        self.jit_module
            .define_function(func_id, &mut self.context)
            .map_err(|e| ExecutionError::JitCompilationFailed {
                reason: format!("Function definition failed: {}", e),
            })?;

        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Function defined successfully");
        }

        // Clear for reuse
        self.context.func.clear();

        // Finalize the JIT compilation
        self.jit_module
            .finalize_definitions()
            .map_err(|e| ExecutionError::JitCompilationFailed {
                reason: format!("JIT finalization failed: {}", e),
            })?;

        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] JIT compilation finalized");
        }

        // Get the compiled function pointer
        let func_ptr = self.jit_module.get_finalized_function(func_id);

        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Got function pointer: {:p}", func_ptr);
        }

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
    /// Execute the query on input columns, returning filtered results as ArrayRef
    ///
    /// # Safety
    /// This function is unsafe because it calls compiled machine code.
    pub unsafe fn execute(&self, input_arrays: &[ArrayRef]) -> Result<ArrayRef, ExecutionError> {
        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Starting query execution");
            println!("[DEBUG] Input arrays count: {}", input_arrays.len());
            for (i, array) in input_arrays.iter().enumerate() {
                println!("[DEBUG] Array {}: length={}, type={:?}", i, array.len(), array.data_type());
            }
        }
        
        if input_arrays.is_empty() {
            return Err(ExecutionError::ExecutionFailed {
                reason: "Must provide at least one input array".to_string(),
            });
        }

        // Verify all arrays have the same length
        let length = input_arrays[0].len();
        for (i, array) in input_arrays.iter().enumerate() {
            if array.len() != length {
                return Err(ExecutionError::ExecutionFailed {
                    reason: format!("Array {} length {} doesn't match expected length {}", i, array.len(), length),
                });
            }
        }

        // Extract raw data pointers from Arrow arrays
        let mut raw_ptrs = Vec::new();
        for (i, array) in input_arrays.iter().enumerate() {
            if let Some(u64_array) = array.as_any().downcast_ref::<arrow::array::UInt64Array>() {
                raw_ptrs.push(u64_array.values().as_ptr() as *const u8);
            } else if let Some(bool_array) = array.as_any().downcast_ref::<arrow::array::BooleanArray>() {
                // For boolean arrays, we need the bitmap data
                // Arrow stores boolean data as bitmaps (Vec<u8>)
                let bitmap_buffer = bool_array.values();
                // Convert BooleanBuffer to Buffer to get raw pointer
                let buffer: &arrow::buffer::Buffer = bitmap_buffer.inner();
                raw_ptrs.push(buffer.as_ptr());
            } else {
                return Err(ExecutionError::ExecutionFailed {
                    reason: format!("Unsupported array type for column {}: {:?}", i, array.data_type()),
                });
            }
        }

        // Prepare output buffer
        let mut output_buffer = vec![0u64; length];

        // Call the compiled function using the NaryOpFn signature
        // fn(array_of_arrays_ptr, input_count, output_ptr, length) -> result_count
        type NaryOpFn = extern "C" fn(*const *const u8, u32, *mut u8, u64) -> i64;
        let func: NaryOpFn = std::mem::transmute(self.func_ptr);
        
        let result_count = func(
            raw_ptrs.as_ptr(),                    // array of input arrays
            input_arrays.len() as u32,            // number of input arrays
            output_buffer.as_mut_ptr() as *mut u8, // output array pointer
            length as u64,                        // input array length
        );

        // Truncate output buffer to actual result count
        output_buffer.truncate(result_count as usize);

        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Execution completed");
            println!("[DEBUG] Result count: {}", result_count);
            println!("[DEBUG] Result values: {:?}", output_buffer);
        }

        // Convert back to ArrayRef
        use arrow::array::UInt64Array;
        let result_array = UInt64Array::from(output_buffer);
        Ok(std::sync::Arc::new(result_array))
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
        input_arrays: &[ArrayRef],
    ) -> Result<ArrayRef, ExecutionError> {
        // Step 1: Build the pipeline description
        let pipeline = PipelineBuilder::scan(column_names)
            .filter(filter_column, comparison, threshold)
            .compile(output_column)?;

        println!("Generated pipeline: {}", pipeline.signature_description());
        
        // Step 2: Compile to executable machine code
        let executable = self.engine.compile_pipeline(pipeline)?;
        
        println!("Compiled executable: {}", executable.description());

        // Step 3: Execute on actual data
        let result_array = unsafe { executable.execute(input_arrays)? };
        
        println!("Execution completed: {} results", result_array.len());
        
        Ok(result_array)
    }

    /// Execute a complete scan + project + filter pipeline
    /// 
    /// Example: Scan(Project(col1, col2), Filter(col1 > 12)) -> compiled function -> execution
    pub fn scan_project_filter(
        &mut self,
        column_names: Vec<String>,
        project_columns: Vec<usize>,
        filter_column: usize,
        comparison: ComparisonOp,
        threshold: u64,
        output_column: usize,
        input_arrays: &[ArrayRef],
    ) -> Result<ArrayRef, ExecutionError> {
        // Step 1: Build the pipeline description
        let pipeline = PipelineBuilder::scan(column_names)
            .project(project_columns)
            .filter(filter_column, comparison, threshold)
            .compile(output_column)?;

        println!("Generated pipeline: {}", pipeline.signature_description());
        
        // Step 2: Compile to executable machine code
        let executable = self.engine.compile_pipeline(pipeline)?;
        
        println!("Compiled executable: {}", executable.description());

        // Step 3: Execute on actual data
        let result_array = unsafe { executable.execute(input_arrays)? };
        
        println!("Execution completed: {} results", result_array.len());
        
        Ok(result_array)
    }
    
    /// Prepare a pipeline for execution (for testing with custom pipelines)
    pub fn prepare_query(&mut self, pipeline: CompiledPipeline) -> Result<ExecutableQuery, ExecutionError> {
        self.engine.compile_pipeline(pipeline)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::UInt64Array;
    use crate::operators::{ComparisonOp, DataType, StagedPredicate};
    use crate::pipeline::PipelineBuilder;

    #[test]
    fn test_complete_vertical_slice() {
        let mut processor = QueryProcessor::new().unwrap();

        // Test data: 3 columns of sample data
        let col0_data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let col1_data = vec![10, 15, 8, 25, 30, 12, 18, 22, 5, 35];
        let col2_data = vec![5, 10, 15, 20, 25, 30, 8, 12, 18, 40];

        // Convert to Arrow arrays
        use arrow::array::UInt64Array;
        let col0_array = std::sync::Arc::new(UInt64Array::from(col0_data.clone()));
        let col1_array = std::sync::Arc::new(UInt64Array::from(col1_data.clone()));
        let col2_array = std::sync::Arc::new(UInt64Array::from(col2_data.clone()));
        let input_arrays = vec![col0_array as ArrayRef, col1_array as ArrayRef, col2_array as ArrayRef];

        // Execute: Scan(Filter(col2 > 12)) -> output col2 values
        let results = processor.scan_filter(
            vec!["id".to_string(), "age".to_string(), "score".to_string()],
            2,                                    // filter on col2 (score)
            ComparisonOp::GreaterThan,           // >
            12,                                  // threshold
            2,                                   // output col2 (score)
            &input_arrays,
        ).unwrap();

        println!("Input data:");
        println!("  col0 (id): {:?}", col0_data);
        println!("  col1 (age): {:?}", col1_data);
        println!("  col2 (score): {:?}", col2_data);
        println!("Filter: score > 12");
        
        // Convert result array back to Vec<u64> for comparison
        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        println!("Results: {:?}", result_values);

        // Expected: values from col2 where col2 > 12
        // col2: [5, 10, 15, 20, 25, 30, 8, 12, 18, 40]
        //         -   -  ✓   ✓   ✓   ✓  -   -   ✓   ✓
        let expected = vec![15, 20, 25, 30, 18, 40];
        assert_eq!(result_values, expected);
    }

    #[test]
    fn test_equality_filter() {
        let mut processor = QueryProcessor::new().unwrap();

        let data = vec![1, 5, 3, 5, 2, 5, 4];
        let data_array = std::sync::Arc::new(UInt64Array::from(data.clone()));
        let input_arrays = vec![data_array as ArrayRef];

        // Filter for values equal to 5
        let results = processor.scan_filter(
            vec!["values".to_string()],
            0,                                    // filter on col0
            ComparisonOp::Equal,                 // ==
            5,                                   // threshold
            0,                                   // output col0
            &input_arrays,
        ).unwrap();

        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        println!("Input: {:?}", data);
        println!("Filter: value == 5");  
        println!("Results: {:?}", result_values);

        let expected = vec![5, 5, 5]; // Three 5's in the input
        assert_eq!(result_values, expected);
    }

    #[test]
    fn test_no_matches() {
        let mut processor = QueryProcessor::new().unwrap();

        let data = vec![1, 2, 3, 4, 5];
        let data_array = std::sync::Arc::new(UInt64Array::from(data.clone()));
        let input_arrays = vec![data_array as ArrayRef];

        // Filter for values greater than 10 (no matches)
        let results = processor.scan_filter(
            vec!["values".to_string()],
            0,
            ComparisonOp::GreaterThan,
            10,
            0,
            &input_arrays,
        ).unwrap();

        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        println!("Input: {:?}", data);
        println!("Filter: value > 10");
        println!("Results: {:?}", result_values);

        assert_eq!(result_values, Vec::<u64>::new()); // No matches
    }

    #[test]
    fn test_all_matches() {
        let mut processor = QueryProcessor::new().unwrap();

        let data = vec![10, 20, 30, 40, 50];
        let data_array = std::sync::Arc::new(UInt64Array::from(data.clone()));
        let input_arrays = vec![data_array as ArrayRef];

        // Filter for values greater than 5 (all match)
        let results = processor.scan_filter(
            vec!["values".to_string()],
            0,
            ComparisonOp::GreaterThan,
            5,
            0,
            &input_arrays,
        ).unwrap();

        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        println!("Input: {:?}", data);
        println!("Filter: value > 5");
        println!("Results: {:?}", result_values);

        assert_eq!(result_values, data); // All values match
    }

    #[test]
    fn test_scan_project_filter_workflow() {
        let mut processor = QueryProcessor::new().unwrap();

        // Test data: 4 columns
        let col0_data = vec![1, 2, 3, 4, 5, 6];         // id
        let col1_data = vec![100, 200, 150, 300, 250, 180]; // score
        let col2_data = vec![10, 20, 15, 30, 25, 18];   // age  
        let col3_data = vec![5, 10, 7, 15, 12, 9];      // bonus

        // Convert to Arrow arrays
        let col0_array = std::sync::Arc::new(UInt64Array::from(col0_data.clone()));
        let col1_array = std::sync::Arc::new(UInt64Array::from(col1_data.clone()));
        let col2_array = std::sync::Arc::new(UInt64Array::from(col2_data.clone()));
        let col3_array = std::sync::Arc::new(UInt64Array::from(col3_data.clone()));
        let input_arrays = vec![col0_array as ArrayRef, col1_array as ArrayRef, 
                               col2_array as ArrayRef, col3_array as ArrayRef];

        // Execute: Scan(Project(col1, col2), Filter(col0 > 200)) -> output col1 values
        // Note: After projection, col1 becomes index 0, col2 becomes index 1
        // So we filter on index 0 (original col1/score) > 200
        // And output index 0 (original col1/score values that pass filter)
        let results = processor.scan_project_filter(
            vec!["id".to_string(), "score".to_string(), "age".to_string(), "bonus".to_string()],
            vec![1, 2],                              // Project: select col1 (score), col2 (age)
            0,                                       // Filter on projected col0 (original score) > 200
            ComparisonOp::GreaterThan,
            200,
            0,                                       // Output projected col0 (original score values)
            &input_arrays,
        ).unwrap();

        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();

        println!("Input data:");
        println!("  col0 (id): {:?}", col0_data);
        println!("  col1 (score): {:?}", col1_data);
        println!("  col2 (age): {:?}", col2_data);
        println!("  col3 (bonus): {:?}", col3_data);
        println!("Project: select score, age");
        println!("Filter: score > 200");
        println!("Output: score values");
        println!("Results: {:?}", result_values);

        // Expected: score values where score > 200
        // col1 (score): [100, 200, 150, 300, 250, 180]
        //                 -    -    -   ✓    ✓    -
        let expected = vec![300, 250];
        assert_eq!(result_values, expected);
    }
    
    #[test]
    fn test_boolean_array_filtering() {
        use arrow::array::BooleanArray;
        
        let mut processor = QueryProcessor::new().unwrap();
        
        // Test data: boolean column with values [true, false, true, false, true, false]
        let bool_data = vec![true, false, true, false, true, false];
        let bool_array = std::sync::Arc::new(BooleanArray::from(bool_data.clone()));
        let input_arrays = vec![bool_array as ArrayRef];
        
        // Build pipeline: Scan(Bool) -> Filter(bool_col == true) -> Output(bool_col)
        let pipeline = PipelineBuilder::scan_with_types(vec![("is_active".to_string(), DataType::Bool)])
            .filter_with_predicate(StagedPredicate::bool_equal(0, true))
            .compile(0)
            .unwrap();
            
        // Execute the query
        let query = processor.prepare_query(pipeline).unwrap();
        let results = unsafe { query.execute(&input_arrays) }.unwrap();
        
        // Convert results back to u64 (booleans are extended to u64)
        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        
        println!("Boolean input: {:?}", bool_data);
        println!("Filter: bool_col == true");
        println!("Results (as u64): {:?}", result_values);
        
        // Expected: true values converted to 1 (u64)
        // bool_data: [true, false, true, false, true, false]
        //              ✓     -     ✓     -     ✓     -
        let expected = vec![1, 1, 1];  // three true values, each extended to u64(1)
        assert_eq!(result_values, expected);
    }
    
    #[test] 
    fn test_boolean_array_filter_false() {
        use arrow::array::BooleanArray;
        
        let mut processor = QueryProcessor::new().unwrap();
        
        // Test filtering for false values
        let bool_data = vec![true, false, true, false, true, false];
        let bool_array = std::sync::Arc::new(BooleanArray::from(bool_data.clone()));
        let input_arrays = vec![bool_array as ArrayRef];
        
        // Build pipeline: Filter(bool_col == false)
        let pipeline = PipelineBuilder::scan_with_types(vec![("is_disabled".to_string(), DataType::Bool)])
            .filter_with_predicate(StagedPredicate::bool_equal(0, false))
            .compile(0)
            .unwrap();
            
        let query = processor.prepare_query(pipeline).unwrap();
        let results = unsafe { query.execute(&input_arrays) }.unwrap();
        
        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        
        println!("Boolean input: {:?}", bool_data);
        println!("Filter: bool_col == false");
        println!("Results (as u64): {:?}", result_values);
        
        // Expected: false values converted to 0 (u64)
        // bool_data: [true, false, true, false, true, false]
        //              -     ✓     -     ✓     -     ✓
        let expected = vec![0, 0, 0];  // three false values, each extended to u64(0)
        assert_eq!(result_values, expected);
    }
    
    #[test]
    fn test_boolean_operation_and() {
        use arrow::array::BooleanArray;
        
        let mut processor = QueryProcessor::new().unwrap();
        
        // Test boolean AND operation between two columns
        let col0_data = vec![true, true, false, false];
        let col1_data = vec![true, false, true, false];
        
        let col0_array = std::sync::Arc::new(BooleanArray::from(col0_data.clone()));
        let col1_array = std::sync::Arc::new(BooleanArray::from(col1_data.clone()));
        let input_arrays = vec![col0_array as ArrayRef, col1_array as ArrayRef];
        
        // Build pipeline: Filter(col0 AND col1) -> Output(col0)
        let pipeline = PipelineBuilder::scan_with_types(vec![
            ("flag_a".to_string(), DataType::Bool),
            ("flag_b".to_string(), DataType::Bool)
        ])
            .filter_with_predicate(StagedPredicate::bool_and(0, 1))
            .compile(0)  // Output col0 values where (col0 AND col1) is true
            .unwrap();
            
        let query = processor.prepare_query(pipeline).unwrap();
        let results = unsafe { query.execute(&input_arrays) }.unwrap();
        
        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        
        println!("col0: {:?}", col0_data);
        println!("col1: {:?}", col1_data);
        println!("Filter: col0 AND col1");
        println!("Results: {:?}", result_values);
        
        // Expected: col0 values where (col0 AND col1) is true
        // col0: [true, true, false, false]
        // col1: [true, false, true, false]
        // AND:  [true, false, false, false]
        //         ✓     -      -      -
        let expected = vec![1]; // Only first row passes: true AND true = true -> output true (1)
        assert_eq!(result_values, expected);
    }
    
    #[test]
    fn test_boolean_operation_or() {
        use arrow::array::BooleanArray;
        
        let mut processor = QueryProcessor::new().unwrap();
        
        // Test boolean OR operation
        let col0_data = vec![true, true, false, false];
        let col1_data = vec![true, false, true, false];
        
        let col0_array = std::sync::Arc::new(BooleanArray::from(col0_data.clone()));
        let col1_array = std::sync::Arc::new(BooleanArray::from(col1_data.clone()));
        let input_arrays = vec![col0_array as ArrayRef, col1_array as ArrayRef];
        
        // Build pipeline: Filter(col0 OR col1) -> Output(col1)
        let pipeline = PipelineBuilder::scan_with_types(vec![
            ("flag_a".to_string(), DataType::Bool),
            ("flag_b".to_string(), DataType::Bool)
        ])
            .filter_with_predicate(StagedPredicate::bool_or(0, 1))
            .compile(1)  // Output col1 values where (col0 OR col1) is true
            .unwrap();
            
        let query = processor.prepare_query(pipeline).unwrap();
        let results = unsafe { query.execute(&input_arrays) }.unwrap();
        
        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        
        println!("col0: {:?}", col0_data);
        println!("col1: {:?}", col1_data);
        println!("Filter: col0 OR col1");
        println!("Results: {:?}", result_values);
        
        // Expected: col1 values where (col0 OR col1) is true
        // col0: [true, true, false, false]
        // col1: [true, false, true, false]
        // OR:   [true, true, true, false]
        //         ✓     ✓     ✓     -
        let expected = vec![1, 0, 1]; // col1 values: true, false, true -> 1, 0, 1
        assert_eq!(result_values, expected);
    }
    
    #[test]
    fn test_mixed_types_u64_and_boolean() {
        use arrow::array::BooleanArray;
        
        let mut processor = QueryProcessor::new().unwrap();
        
        // Test with mixed types: u64 and boolean columns
        let id_data = vec![1, 2, 3, 4, 5, 6];
        let active_data = vec![true, false, true, true, false, true];
        
        let id_array = std::sync::Arc::new(UInt64Array::from(id_data.clone()));
        let active_array = std::sync::Arc::new(BooleanArray::from(active_data.clone()));
        let input_arrays = vec![id_array as ArrayRef, active_array as ArrayRef];
        
        // Complex filter: (id > 3) AND (is_active == true)
        // This combines numeric and boolean predicates
        let numeric_filter = StagedPredicate::greater_than(0, 3);  // id > 3
        let boolean_filter = StagedPredicate::bool_equal(1, true); // is_active == true
        let combined_filter = StagedPredicate::And(Box::new(numeric_filter), Box::new(boolean_filter));
        
        let pipeline = PipelineBuilder::scan_with_types(vec![
            ("id".to_string(), DataType::U64),
            ("is_active".to_string(), DataType::Bool)
        ])
            .filter_with_predicate(combined_filter)
            .compile(0)  // Output id values
            .unwrap();
            
        let query = processor.prepare_query(pipeline).unwrap();
        let results = unsafe { query.execute(&input_arrays) }.unwrap();
        
        let result_u64_array = results.as_any().downcast_ref::<UInt64Array>().unwrap();
        let result_values: Vec<u64> = result_u64_array.values().to_vec();
        
        println!("id: {:?}", id_data);
        println!("is_active: {:?}", active_data);
        println!("Filter: (id > 3) AND (is_active == true)");
        println!("Results: {:?}", result_values);
        
        // Expected: id values where (id > 3) AND (is_active == true)
        // id:        [1, 2, 3, 4, 5, 6]
        // is_active: [T, F, T, T, F, T]
        // id > 3:    [F, F, F, T, T, T]
        // AND:       [F, F, F, T, F, T]
        //             -  -  -  ✓  -  ✓
        let expected = vec![4, 6]; // Only rows 4 and 6 satisfy both conditions
        assert_eq!(result_values, expected);
    }
}