//! Query pipeline construction and compilation
//!
//! This module implements the high-level API for building and compiling query pipelines
//! using the callback-based operator model. It provides a fluent API for constructing
//! pipelines and generates specialized Cranelift functions via Futamura projection.

use crate::operators::{
    ComparisonOp, Consumer, DataType, Operator, OutputConsumer, 
    ScanOperator, Schema, SelectionOperator, StagedPredicate, ColumnInfo
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
    Filter {
        predicate: StagedPredicate,
    },
    // Future operators: Join, Aggregate, Sort, etc.
}

impl PipelineBuilder {
    /// Create a new pipeline starting with a scan
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
        let mut pipeline_compiler = PipelineCompiler::new(self.schema.clone());
        pipeline_compiler.compile_pipeline(self.operators, output_column)
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
            "fn({}, length: u64, output_ptr: *mut u64) -> count: u64\nOutputs: {} values",
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
        // Create function signature
        // fn(col0_ptr, col1_ptr, ..., length, output_ptr) -> count
        let mut sig = Signature::new(CallConv::SystemV);
        
        // Add input column parameters
        for _ in &self.schema.columns {
            sig.params.push(AbiParam::new(types::I64)); // column pointer
        }
        sig.params.push(AbiParam::new(types::I64)); // length
        sig.params.push(AbiParam::new(types::I64)); // output pointer
        sig.returns.push(AbiParam::new(types::I64)); // count

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
        let scan_op = ScanOperator::new(self.schema.clone());

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
            input_schema: self.schema.clone(),
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
        // Start with the output consumer (rightmost in chain)
        let mut consumer: Box<dyn Consumer> = Box::new(OutputConsumer::new(
            0, // output param index (simplified)
            output_column,
            self.schema.clone(),
            count_var,
        ));

        // Build chain backwards (right to left), excluding the scan
        for op in operators.iter().skip(1) { // Skip the scan (first operator)
            consumer = match op {
                PipelineOperator::Scan { .. } => {
                    // This should never happen since we skip(1)
                    consumer
                }
                PipelineOperator::Filter { predicate } => {
                    Box::new(SelectionOperator::new(
                        predicate.clone(),
                        consumer,
                        self.schema.clone(),
                    ))
                }
            };
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
        
        // Verify function signature
        let sig = &pipeline.function.signature;
        assert_eq!(sig.params.len(), 5); // 3 columns + length + output_ptr
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
}