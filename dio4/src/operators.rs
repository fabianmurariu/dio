//! Callback-based operators implementing the push-based execution model
//!
//! This module implements the core operator framework based on the callback-based
//! compilation approach from "How to Architect a Query Compiler, Revisited".
//! Each operator follows the producer/consumer pattern with staged code generation.

use crate::staging::{
    control_flow::{StagedConditional, StagedForLoop},
    StagedBool, StagedU64, StagedVariable, Staged, StagingError,
};
use cranelift_codegen::ir::{types, InstBuilder, Value};
use cranelift_frontend::{FunctionBuilder, Variable};
use std::marker::PhantomData;

/// Core operator trait implementing the callback-based push model
pub trait Operator {
    /// Start data production, calling consume() on the consumer for each record
    fn produce(&self, consumer: &dyn Consumer, builder: &mut FunctionBuilder, ctx: &mut ExecutionContext) -> Result<(), StagingError>;
    
    /// Get the output schema produced by this operator
    fn output_schema(&self) -> &Schema;
}

/// Consumer trait for processing records in the callback chain
pub trait Consumer {
    /// Process a single record and potentially pass to downstream consumers
    fn consume(&self, record: &Record, builder: &mut FunctionBuilder, ctx: &mut ExecutionContext) -> Result<(), StagingError>;
    
    /// Get the schema this consumer expects
    fn input_schema(&self) -> &Schema;
}

/// Execution context that tracks variable allocation and state
pub struct ExecutionContext {
    pub variable_counter: u32,
    pub current_record_vars: Vec<Variable>,
}

impl ExecutionContext {
    pub fn new() -> Self {
        Self {
            variable_counter: 1000, // Start high to avoid conflicts
            current_record_vars: Vec::new(),
        }
    }

    /// Allocate a fresh variable for staging
    pub fn fresh_variable(&mut self, builder: &mut FunctionBuilder) -> Variable {
        let var = Variable::from_u32(self.variable_counter);
        self.variable_counter += 1;
        builder.declare_var(var, types::I64);
        var
    }
}

/// Schema describing the structure of records
#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub columns: Vec<ColumnInfo>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ColumnInfo {
    pub name: String,
    pub data_type: DataType,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    U64,
    I64,
    F64,
    Bool,
}

/// A record represents a single row with staged column values
pub struct Record {
    pub columns: Vec<StagedU64>, // Simplified to U64 for now
    pub schema: Schema,
}

impl Record {
    pub fn get_column(&self, index: usize) -> Option<&StagedU64> {
        self.columns.get(index)
    }
}

/// Scan operator - produces records from input arrays (data source)
pub struct ScanOperator {
    pub schema: Schema,
    pub input_params: Vec<usize>, // Parameter indices for input arrays
}

impl ScanOperator {
    pub fn new(schema: Schema) -> Self {
        Self {
            input_params: (0..schema.columns.len()).collect(),
            schema,
        }
    }
}

impl Operator for ScanOperator {
    fn produce(&self, consumer: &dyn Consumer, builder: &mut FunctionBuilder, ctx: &mut ExecutionContext) -> Result<(), StagingError> {
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] ScanOperator::produce - schema: {:?}", self.schema);
        }
        
        // Get function parameters (array pointers and length)
        let entry_block = builder.current_block().unwrap();
        let params = builder.block_params(entry_block);
        
        if std::env::var("DIO_DEBUG_JIT").is_ok() {
            println!("[DEBUG] Function parameters count: {}", params.len());
            println!("[DEBUG] Expected: {} input columns + length + output", self.schema.columns.len());
        }
        
        // Function signature: (array0_ptr, array1_ptr, ..., length, output_ptr) -> count
        let length_param = params[params.len() - 2]; // Second to last param is length
        let array_ptrs: Vec<Value> = params[..params.len() - 2].to_vec();

        // Create length variable for the loop
        let length_var = ctx.fresh_variable(builder);
        builder.def_var(length_var, length_param);
        let staged_length = StagedU64::Variable(StagedVariable::new(length_var.as_u32(), types::I64));

        // Generate loop from 0 to length
        let start = StagedU64::Constant(0);
        let scan_loop = StagedForLoop::new(start, staged_length);

        scan_loop.generate_loop(builder, |loop_builder, row_index| {
            // Load all columns for this row
            let mut record_columns = Vec::new();
            for (col_idx, &array_ptr) in array_ptrs.iter().enumerate() {
                if std::env::var("DIO_DEBUG_JIT").is_ok() {
                    println!("[DEBUG] Loading column {} from array_ptr", col_idx);
                }
                // Calculate element address: array_ptr + (row_index * 8)
                let row_idx_val = row_index.codegen(loop_builder);
                let element_size = loop_builder.ins().iconst(types::I64, 8); // 8 bytes for u64
                let offset = loop_builder.ins().imul(row_idx_val, element_size);
                let element_ptr = loop_builder.ins().iadd(array_ptr, offset);
                
                // Load the element value
                let element_val = loop_builder.ins().load(
                    types::I64, 
                    cranelift_codegen::ir::MemFlags::new(), 
                    element_ptr, 
                    0
                );

                // Create a variable to hold this column value
                let col_var = ctx.fresh_variable(loop_builder);
                loop_builder.def_var(col_var, element_val);
                let staged_column = StagedU64::Variable(StagedVariable::new(col_var.as_u32(), types::I64));
                record_columns.push(staged_column);
                ctx.current_record_vars.push(col_var);
            }

            // Create record and pass to consumer
            let record = Record {
                columns: record_columns,
                schema: self.schema.clone(),
            };

            consumer.consume(&record, loop_builder, ctx)?;
            
            Ok(())
        })
    }

    fn output_schema(&self) -> &Schema {
        &self.schema
    }
}

/// Selection (Filter) operator - filters records based on a predicate
pub struct SelectionOperator {
    pub predicate: StagedPredicate,
    pub downstream: Box<dyn Consumer>,
    pub schema: Schema,
}

impl SelectionOperator {
    pub fn new(predicate: StagedPredicate, downstream: Box<dyn Consumer>, schema: Schema) -> Self {
        Self { predicate, downstream, schema }
    }
}

impl Consumer for SelectionOperator {
    fn consume(&self, record: &Record, builder: &mut FunctionBuilder, ctx: &mut ExecutionContext) -> Result<(), StagingError> {
        // Evaluate predicate on the record
        let condition = self.predicate.evaluate(record)?;
        let staged_cond = StagedConditional::new(condition);

        // Generate conditional: if (predicate) then pass to downstream
        staged_cond.generate_if_else(
            builder,
            |inner_builder| {
                self.downstream.consume(record, inner_builder, ctx)
            },
            None::<fn(&mut FunctionBuilder) -> Result<(), StagingError>>,
        )
    }

    fn input_schema(&self) -> &Schema {
        &self.schema
    }
}

/// Simple expression for column selection in projections
#[derive(Debug, Clone)]
pub enum ProjectionExpr {
    /// Select a column by index
    Column(usize),
    // Future: computed expressions, literals, etc.
}

/// Predicate for filtering - can be extended for more complex expressions
#[derive(Debug, Clone)]
pub enum StagedPredicate {
    /// Column comparison: column_index op constant_value
    ColumnComparison {
        column_index: usize,
        op: ComparisonOp,
        value: u64,
    },
    /// Logical AND of two predicates
    And(Box<StagedPredicate>, Box<StagedPredicate>),
    /// Logical OR of two predicates
    Or(Box<StagedPredicate>, Box<StagedPredicate>),
    /// Logical NOT of a predicate
    Not(Box<StagedPredicate>),
}

#[derive(Debug, Clone)]
pub enum ComparisonOp {
    Equal,
    NotEqual,
    LessThan,
    LessThanOrEqual,
    GreaterThan,
    GreaterThanOrEqual,
}

impl StagedPredicate {
    pub fn evaluate(&self, record: &Record) -> Result<StagedBool, StagingError> {
        match self {
            StagedPredicate::ColumnComparison { column_index, op, value } => {
                let column_val = record.get_column(*column_index)
                    .ok_or_else(|| StagingError::CodeGenerationFailed {
                        reason: format!("Column index {} out of bounds", column_index),
                    })?;
                
                let constant = StagedU64::Constant(*value);
                let result = match op {
                    ComparisonOp::Equal => column_val.clone().eq(constant),
                    ComparisonOp::NotEqual => {
                        // NOT(column == constant)
                        todo!("Implement NOT operation")
                    }
                    ComparisonOp::LessThan => column_val.clone().lt(constant),
                    ComparisonOp::LessThanOrEqual => {
                        // column < constant OR column == constant
                        todo!("Implement OR operation")
                    }
                    ComparisonOp::GreaterThan => column_val.clone().gt(constant),
                    ComparisonOp::GreaterThanOrEqual => {
                        // column > constant OR column == constant
                        todo!("Implement OR operation")
                    }
                };
                Ok(result)
            }
            StagedPredicate::And(left, right) => {
                let left_val = left.evaluate(record)?;
                let right_val = right.evaluate(record)?;
                // TODO: Implement staged AND
                todo!("Implement staged AND")
            }
            StagedPredicate::Or(left, right) => {
                let left_val = left.evaluate(record)?;
                let right_val = right.evaluate(record)?;
                // TODO: Implement staged OR
                todo!("Implement staged OR")
            }
            StagedPredicate::Not(inner) => {
                let inner_val = inner.evaluate(record)?;
                // TODO: Implement staged NOT
                todo!("Implement staged NOT")
            }
        }
    }
}

/// Output consumer - writes filtered results to output array
pub struct OutputConsumer {
    pub output_param_index: usize, // Parameter index for output array
    pub column_index: usize,       // Which column to output
    pub schema: Schema,
    pub count_var: Variable,       // Variable tracking output count
}

impl OutputConsumer {
    pub fn new(output_param_index: usize, column_index: usize, schema: Schema, count_var: Variable) -> Self {
        Self {
            output_param_index,
            column_index,
            schema,
            count_var,
        }
    }
}

impl Consumer for OutputConsumer {
    fn consume(&self, record: &Record, builder: &mut FunctionBuilder, ctx: &mut ExecutionContext) -> Result<(), StagingError> {
        // Get output array pointer from function parameters
        // We need to get the entry block, not the current block
        let func = &builder.func;
        let entry_block = func.layout.entry_block().expect("Function should have entry block");
        let params = func.dfg.block_params(entry_block);
        let output_ptr = params[params.len() - 1]; // Last param is output ptr

        // Get the column value to write
        let column_val = record.get_column(self.column_index)
            .ok_or_else(|| StagingError::CodeGenerationFailed {
                reason: format!("Column index {} out of bounds", self.column_index),
            })?;

        // Get current output count
        let current_count = builder.use_var(self.count_var);
        
        // Calculate output address: output_ptr + (count * 8)
        let element_size = builder.ins().iconst(types::I64, 8);
        let output_offset = builder.ins().imul(current_count, element_size);
        let output_element_ptr = builder.ins().iadd(output_ptr, output_offset);

        // Store the column value
        let value_to_store = column_val.codegen(builder);
        builder.ins().store(
            cranelift_codegen::ir::MemFlags::new(),
            value_to_store,
            output_element_ptr,
            0
        );

        // Increment count
        let one = builder.ins().iconst(types::I64, 1);
        let new_count = builder.ins().iadd(current_count, one);
        builder.def_var(self.count_var, new_count);

        Ok(())
    }

    fn input_schema(&self) -> &Schema {
        &self.schema
    }
}

/// Projection operator - selects and reorders columns based on expressions
pub struct ProjectionOperator {
    pub expressions: Vec<ProjectionExpr>,
    pub downstream: Box<dyn Consumer>,
    pub input_schema: Schema,
    pub output_schema: Schema,
}

impl ProjectionOperator {
    pub fn new(
        expressions: Vec<ProjectionExpr>, 
        downstream: Box<dyn Consumer>, 
        input_schema: Schema
    ) -> Result<Self, StagingError> {
        // Build output schema from projection expressions
        let mut output_columns = Vec::new();
        for expr in &expressions {
            match expr {
                ProjectionExpr::Column(col_idx) => {
                    if *col_idx >= input_schema.columns.len() {
                        return Err(StagingError::CodeGenerationFailed {
                            reason: format!("Column index {} out of bounds", col_idx),
                        });
                    }
                    output_columns.push(input_schema.columns[*col_idx].clone());
                }
            }
        }
        
        let output_schema = Schema { columns: output_columns };
        
        Ok(Self { 
            expressions, 
            downstream, 
            input_schema,
            output_schema,
        })
    }
}

impl Consumer for ProjectionOperator {
    fn consume(&self, record: &Record, builder: &mut FunctionBuilder, ctx: &mut ExecutionContext) -> Result<(), StagingError> {
        // Project the record to create a new record with selected columns
        let mut projected_columns = Vec::new();
        
        for expr in &self.expressions {
            match expr {
                ProjectionExpr::Column(col_idx) => {
                    let column_val = record.get_column(*col_idx)
                        .ok_or_else(|| StagingError::CodeGenerationFailed {
                            reason: format!("Column index {} out of bounds", col_idx),
                        })?;
                    projected_columns.push(column_val.clone());
                }
            }
        }
        
        // Create new record with projected columns and updated schema
        let projected_record = Record {
            columns: projected_columns,
            schema: self.output_schema.clone(),
        };
        
        // Pass projected record to downstream consumer
        self.downstream.consume(&projected_record, builder, ctx)
    }

    fn input_schema(&self) -> &Schema {
        &self.input_schema
    }
}