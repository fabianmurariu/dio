//! Query pipeline construction and compilation
//!
//! This module implements the high-level API for building and compiling query pipelines
//! using the callback-based operator model. It provides a fluent API for constructing
//! pipelines and generates specialized Cranelift functions via Futamura projection.

use crate::operators::{
    BooleanOp, ComparisonOp, Consumer, DataType, Operator, OutputConsumer, 
    ProjectionExpr, ProjectionOperator, ScanOperator, Schema, 
    SelectionOperator, StagedPredicate, ColumnInfo
};
use crate::staging::{StagedVariable, StagingError};
use cranelift_codegen::ir::{types, AbiParam, Function, Signature, InstBuilder};
use cranelift_codegen::isa::CallConv;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext, Variable};

/// High-level pipeline builder implementing fluent API
pub struct PipelineBuilder {
    operators: Vec<PipelineOperator>,
    schema: Schema,
}

/// Internal representation of operators in the pipeline
#[derive(Debug)]
pub enum PipelineOperator {
    Scan {
        schema: Schema,
    },
    Project {
        expressions: Vec<ProjectionExpr>,
    },
    Filter {
        predicate: StagedPredicate,
    },
    // Future operators: Join, Aggregate, Sort, etc.
}

impl PipelineOperator {
    /// Get the input schema for this operator (only applies to Scan)
    fn schema(&self) -> &Schema {
        match self {
            PipelineOperator::Scan { schema } => schema,
            _ => panic!("schema() only applies to Scan operator"),
        }
    }
}

impl PipelineBuilder {
    /// Create a new pipeline starting with a scan (defaults all columns to U64)
    pub fn scan(column_names: Vec<String>) -> Self {
        let schema = Schema {
            columns: column_names.into_iter().map(|name| ColumnInfo {
                name,
                data_type: DataType::U64, // Simplified for now
            }).collect(),
        };

        Self {
            operators: vec![PipelineOperator::Scan {
                schema: schema.clone(),
            }],
            schema,
        }
    }
    
    /// Create a new pipeline starting with a scan with specified column types
    pub fn scan_with_types(columns: Vec<(String, DataType)>) -> Self {
        let schema = Schema {
            columns: columns.into_iter().map(|(name, data_type)| ColumnInfo {
                name,
                data_type,
            }).collect(),
        };

        Self {
            operators: vec![PipelineOperator::Scan {
                schema: schema.clone(),
            }],
            schema,
        }
    }

    /// Add a projection operation to the pipeline
    pub fn project(mut self, column_indices: Vec<usize>) -> Self {
        // Convert column indices to projection expressions
        let expressions: Vec<ProjectionExpr> = column_indices.into_iter()
            .map(ProjectionExpr::Column)
            .collect();
        
        // Update pipeline schema to reflect the projection
        let projected_columns: Vec<ColumnInfo> = expressions.iter()
            .map(|expr| {
                match expr {
                    ProjectionExpr::Column(idx) => self.schema.columns[*idx].clone(),
                }
            })
            .collect();
        
        self.schema = Schema { columns: projected_columns };
        self.operators.push(PipelineOperator::Project { expressions });
        self
    }

    /// Add a filter operation to the pipeline
    pub fn filter(mut self, column_index: usize, op: ComparisonOp, value: u64) -> Self {
        let predicate = StagedPredicate::ColumnComparison {
            column_index,
            op,
            value,
        };

        self.operators.push(PipelineOperator::Filter { predicate });
        self
    }

    /// Add a filter with a custom predicate
    pub fn filter_with_predicate(mut self, predicate: StagedPredicate) -> Self {
        self.operators.push(PipelineOperator::Filter { predicate });
        self
    }

    /// Compile the entire pipeline into a specialized Cranelift function
    pub fn compile(self, output_column: usize) -> Result<CompiledPipeline, StagingError> {
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Starting pipeline compilation");
            println!("[DEBUG] Pipeline schema: {:?}", self.schema);
            println!("[DEBUG] Pipeline operators: {:?}", self.operators);
            println!("[DEBUG] Output column: {}", output_column);
        }
        
        let mut pipeline_compiler = PipelineCompiler::new(self.schema.clone());
        let result = pipeline_compiler.compile_pipeline(self.operators, output_column);
        
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            match &result {
                Ok(pipeline) => {
                    println!("[DEBUG] Pipeline compilation successful");
                    println!("[DEBUG] Generated function signature: {}", pipeline.signature_description());
                    println!("[DEBUG] Cranelift IR:\n{}", pipeline.function.display());
                }
                Err(e) => {
                    println!("[DEBUG] Pipeline compilation failed: {:?}", e);
                }
            }
        }
        
        result
    }
}

/// Compiled pipeline that can be executed
pub struct CompiledPipeline {
    pub function: Function,
    pub input_schema: Schema,
    pub output_column: usize,
}

impl CompiledPipeline {
    /// Get the function signature description
    pub fn signature_description(&self) -> String {
        let input_cols: Vec<String> = self.input_schema.columns
            .iter()
            .map(|col| format!("{}: {}", col.name, format!("{:?}", col.data_type)))
            .collect();
        
        let output_col = &self.input_schema.columns[self.output_column];
        
        format!(
            "fn(arrays_ptr: *const *const u8, input_count: u32, output_ptr: *mut u8, length: u64) -> count: u64\nInput arrays: [{}]\nOutputs: {} values",
            input_cols.join(", "),
            output_col.name
        )
    }
}

/// Internal compiler that builds the callback chain and generates Cranelift code
struct PipelineCompiler {
    schema: Schema,
}

impl PipelineCompiler {
    fn new(schema: Schema) -> Self {
        Self { schema }
    }

    fn compile_pipeline(
        &mut self, 
        operators: Vec<PipelineOperator>, 
        output_column: usize
    ) -> Result<CompiledPipeline, StagingError> {
        // Create function signature using NaryOpFn format
        // fn(array_of_arrays_ptr, input_count, output_ptr, length) -> count
        let scan_schema = match operators.first() {
            Some(PipelineOperator::Scan { schema }) => schema,
            _ => return Err(StagingError::CodeGenerationFailed {
                reason: "Pipeline must start with a Scan operator".to_string(),
            }),
        };
        
        let mut sig = Signature::new(CallConv::SystemV);
        
        // NaryOpFn signature: (*const *const u8, u32, *mut u8, u64) -> i64
        sig.params.push(AbiParam::new(types::I64)); // array of input arrays pointer
        sig.params.push(AbiParam::new(types::I32)); // number of input arrays 
        sig.params.push(AbiParam::new(types::I64)); // output array pointer
        sig.params.push(AbiParam::new(types::I64)); // input array length
        sig.returns.push(AbiParam::new(types::I64)); // result count

        // Create Cranelift function
        let mut func = Function::new();
        func.signature = sig;

        let mut func_ctx = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut func, &mut func_ctx);

        // Create entry block with parameters
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        // Initialize output count variable
        let count_var = Variable::from_u32(0);
        builder.declare_var(count_var, types::I64);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.def_var(count_var, zero);

        // Build the callback chain from the operators
        let consumer_chain = self.build_consumer_chain(&operators, output_column, count_var)?;

        // Create the scan operator (always the first/root operator)
        let scan_op = ScanOperator::new(scan_schema.clone());

        // Execute the pipeline - scan produces to the consumer chain
        let mut ctx = crate::operators::ExecutionContext::new();
        scan_op.produce(consumer_chain.as_ref(), &mut builder, &mut ctx)?;

        // Return the final count
        let final_count = builder.use_var(count_var);
        builder.ins().return_(&[final_count]);

        // Finalize the function
        builder.seal_all_blocks();
        builder.finalize();

        Ok(CompiledPipeline {
            function: func,
            input_schema: scan_schema.clone(),
            output_column,
        })
    }

    /// Build the consumer callback chain from right to left (reverse order)
    fn build_consumer_chain(
        &self,
        operators: &[PipelineOperator],
        output_column: usize,
        count_var: Variable,
    ) -> Result<Box<dyn Consumer>, StagingError> {
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Building consumer chain from {} operators", operators.len());
        }
        
        // Start with the output consumer (rightmost in chain)
        let mut consumer: Box<dyn Consumer> = Box::new(OutputConsumer::new(
            0, // output param index (simplified)
            output_column,
            self.schema.clone(),
            count_var,
        ));

        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Created OutputConsumer for column {} with schema: {:?}", output_column, self.schema);
        }

        // Build chain backwards (right to left), excluding the scan
        // First, compute all schema transformations in forward order
        let mut schemas = vec![];
        let mut current_schema = operators.first().unwrap().schema().clone(); // Start with scan schema
        schemas.push(current_schema.clone());
        
        for op in operators.iter().skip(1) { // Skip scan, process in forward order
            current_schema = match op {
                PipelineOperator::Scan { schema } => schema.clone(),
                PipelineOperator::Project { expressions } => {
                    // Compute projected schema
                    let projected_columns: Vec<ColumnInfo> = expressions.iter()
                        .map(|expr| match expr {
                            ProjectionExpr::Column(idx) => current_schema.columns[*idx].clone(),
                        })
                        .collect();
                    Schema { columns: projected_columns }
                }
                PipelineOperator::Filter { .. } => current_schema.clone(), // Filter doesn't change schema
            };
            schemas.push(current_schema.clone());
        }
        
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Computed schemas: {:?}", schemas);
        }
        
        // Now build consumers in reverse order using the computed schemas
        for (idx, op) in operators.iter().skip(1).rev().enumerate() { // Skip scan and reverse
            let schema_idx = operators.len() - 1 - idx; // Map reverse index to forward schema index
            let input_schema = schemas[schema_idx - 1].clone(); // Input schema is from previous stage
            
            if std::env::var("DIO_DEBUG_JIT").is_ok() {
                println!("[DEBUG] Processing operator: {:?}", op);
                println!("[DEBUG] Using input schema[{}]: {:?}", schema_idx - 1, input_schema);
            }
            
            consumer = match op {
                PipelineOperator::Scan { .. } => {
                    // This should never happen since we skip(1)
                    consumer
                }
                PipelineOperator::Project { expressions } => {
                    if std::env::var("DIO_DEBUG_JIT").is_ok() {
                        println!("[DEBUG] Creating ProjectionOperator with expressions: {:?}", expressions);
                        println!("[DEBUG] Input schema: {:?}", input_schema);
                    }
                    
                    let proj_op = ProjectionOperator::new(
                        expressions.clone(),
                        consumer,
                        input_schema,
                    )?;
                    Box::new(proj_op)
                }
                PipelineOperator::Filter { predicate } => {
                    if std::env::var("DIO_DEBUG_JIT").is_ok() {
                        println!("[DEBUG] Creating SelectionOperator with predicate: {:?}", predicate);
                        println!("[DEBUG] Input schema: {:?}", input_schema);
                    }
                    
                    Box::new(SelectionOperator::new(
                        predicate.clone(),
                        consumer,
                        input_schema,
                    ))
                }
            };
        }

        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Consumer chain built successfully");
        }

        Ok(consumer)
    }
}

/// Helper functions for common pipeline patterns
impl StagedPredicate {
    /// Create a simple column > value predicate
    pub fn greater_than(column_index: usize, value: u64) -> Self {
        StagedPredicate::ColumnComparison {
            column_index,
            op: ComparisonOp::GreaterThan,
            value,
        }
    }

    /// Create a simple column == value predicate
    pub fn equal_to(column_index: usize, value: u64) -> Self {
        StagedPredicate::ColumnComparison {
            column_index,
            op: ComparisonOp::Equal,
            value,
        }
    }

    /// Create a simple column < value predicate
    pub fn less_than(column_index: usize, value: u64) -> Self {
        StagedPredicate::ColumnComparison {
            column_index,
            op: ComparisonOp::LessThan,
            value,
        }
    }
    
    /// Create a boolean column == boolean value predicate
    pub fn bool_equal(column_index: usize, value: bool) -> Self {
        StagedPredicate::BooleanComparison {
            column_index,
            value,
        }
    }
    
    /// Create a boolean operation between two columns (column_a AND column_b)
    pub fn bool_and(left_column: usize, right_column: usize) -> Self {
        StagedPredicate::BooleanOperation {
            left_column,
            op: BooleanOp::And,
            right_column,
        }
    }
    
    /// Create a boolean operation between two columns (column_a OR column_b)
    pub fn bool_or(left_column: usize, right_column: usize) -> Self {
        StagedPredicate::BooleanOperation {
            left_column,
            op: BooleanOp::Or,
            right_column,
        }
    }
    
    /// Create a boolean equality between two columns (column_a == column_b)
    pub fn bool_equal_columns(left_column: usize, right_column: usize) -> Self {
        StagedPredicate::BooleanOperation {
            left_column,
            op: BooleanOp::Equal,
            right_column,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pipeline_builder_api() {
        // Test the fluent API for building pipelines
        let pipeline = PipelineBuilder::scan(vec!["col0".to_string(), "col1".to_string(), "col2".to_string()])
            .filter(2, ComparisonOp::GreaterThan, 12) // col2 > 12
            .compile(2) // output col2 values
            .unwrap();

        println!("Pipeline signature: {}", pipeline.signature_description());
        
        // Verify function signature - NaryOpFn format
        let sig = &pipeline.function.signature;
        assert_eq!(sig.params.len(), 4); // arrays_ptr + input_count + output_ptr + length
        assert_eq!(sig.returns.len(), 1); // count

        // Verify IR contains the filter constant
        let ir = pipeline.function.display().to_string();
        assert!(ir.contains("iconst.i64 12"), "Should contain the filter constant 12");
        assert!(ir.contains("icmp ugt"), "Should contain unsigned greater than comparison");
    }

    #[test]
    fn test_complex_predicate() {
        // Test with a custom predicate
        let predicate = StagedPredicate::greater_than(1, 100);
        
        let pipeline = PipelineBuilder::scan(vec!["name".to_string(), "age".to_string(), "salary".to_string()])
            .filter_with_predicate(predicate)
            .compile(1) // output age values where age > 100
            .unwrap();

        let ir = pipeline.function.display().to_string();
        println!("Complex predicate IR:\n{}", ir);
        
        assert!(ir.contains("iconst.i64 100"), "Should contain the filter constant 100");
    }

    #[test]
    fn test_scan_only_pipeline() {
        // Test a pipeline with just a scan (no filters)
        let pipeline = PipelineBuilder::scan(vec!["data".to_string()])
            .compile(0) // output all data values
            .unwrap();

        let ir = pipeline.function.display().to_string();
        println!("Scan-only pipeline IR:\n{}", ir);
        
        // Should still generate a loop structure
        assert!(ir.contains("jump"), "Should contain loop structure");
        // Should contain loop bounds checking (icmp ult) but no filter comparisons (icmp ugt)
        assert!(ir.contains("icmp ult"), "Should contain loop bounds checking");
        assert!(!ir.contains("icmp ugt"), "Should not contain filter comparison instructions");
    }

    #[test]
    fn test_scan_project_pipeline() {
        // Test a pipeline: Scan -> Project(select col1, col0) -> Output col0 from projected schema
        let pipeline = PipelineBuilder::scan(vec!["col0".to_string(), "col1".to_string(), "col2".to_string()])
            .project(vec![1, 0]) // Select col1, col0 (reorder and subset)
            .compile(1) // Output col0 from the projected schema (which is now at index 1)
            .unwrap();

        println!("Scan->Project pipeline signature: {}", pipeline.signature_description());
        
        // Verify function signature - NaryOpFn format
        let sig = &pipeline.function.signature;
        assert_eq!(sig.params.len(), 4); // arrays_ptr + input_count + output_ptr + length
        assert_eq!(sig.returns.len(), 1); // count

        let ir = pipeline.function.display().to_string();
        println!("Scan->Project pipeline IR:\n{}", ir);
        
        // Should contain loop structure but no filter comparisons
        assert!(ir.contains("jump"), "Should contain loop structure");
        assert!(ir.contains("icmp ult"), "Should contain loop bounds checking");
        assert!(!ir.contains("icmp ugt"), "Should not contain filter comparison instructions");
    }
}