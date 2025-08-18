# Dio: JIT Compiled Columnar Expression Evaluator

## Overview

Dio is a high-performance expression evaluation engine that compiles arithmetic expressions into optimized machine code using Cranelift JIT compilation. It operates on Arrow arrays to provide efficient columnar data processing.

## Architecture

### 1. Expression DSL

#### Basic Structure
```rust
#[derive(Debug, Clone)]
pub enum Expr {
    // Literals
    Column(String),           // Column reference: "a", "b", "x"
    Literal(Value),          // Constants: 42, 3.14
    
    // Arithmetic operations
    Add(Vec<Expr>),          // (+ a b c)
    Sub(Box<Expr>, Box<Expr>), // (- x y)
    Mul(Vec<Expr>),          // (* a b c)
    Div(Box<Expr>, Box<Expr>), // (/ x y)
    
    // Aggregations
    Sum(Box<Expr>),          // (sum (+ a b))
    Count(Box<Expr>),        // (count a)
    
    // Variable binding (future)
    Let(String, Box<Expr>),  // (let a (+ x y))
}

#[derive(Debug, Clone)]
pub enum Value {
    Int64(i64),
    Float64(f64),
}
```

#### Expression Types
- **Elementwise operations**: Operations that produce arrays of same length as inputs
  - `(+ a b c)` → `output[i] = a[i] + b[i] + c[i]`
  - `(- x y)` → `output[i] = x[i] - y[i]`
- **Reduction operations**: Operations that reduce arrays to scalars
  - `(sum a)` → single scalar value
  - `(count a)` → single scalar count

### 2. Compilation Pipeline

The compilation pipeline transforms expressions through multiple phases:
**AST → SSA IR → Optimized SSA IR → Cranelift IR → Machine Code**

#### Phase 1: Expression Analysis
```rust
pub struct ExprAnalysis {
    pub input_columns: Vec<String>,
    pub output_type: OutputType,
    pub requires_loop: bool,
    pub max_depth: usize,
}

pub enum OutputType {
    Array(ArrowDataType),    // Elementwise operations
    Scalar(ArrowDataType),   // Reductions
}
```

#### Phase 2: AST to SSA IR Conversion
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SsaValue(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct BlockId(pub u32);

#[derive(Debug, Clone)]
pub enum SsaInstruction {
    // Input/Output operations
    LoadArray { dest: SsaValue, column: String, data_type: DataType },
    LoadScalar { dest: SsaValue, value: Value },
    StoreArray { array: SsaValue, dest_column: String },
    StoreScalar { value: SsaValue, dest_name: String },
    
    // Arithmetic operations
    Add { dest: SsaValue, operands: Vec<SsaValue> },
    Sub { dest: SsaValue, lhs: SsaValue, rhs: SsaValue },
    Mul { dest: SsaValue, operands: Vec<SsaValue> },
    Div { dest: SsaValue, lhs: SsaValue, rhs: SsaValue },
    
    // Array access and indexing
    ArrayAccess { dest: SsaValue, array: SsaValue, index: SsaValue },
    ArrayLength { dest: SsaValue, array: SsaValue },
    
    // Reduction operations
    InitAccumulator { dest: SsaValue, initial_value: Value, data_type: DataType },
    Accumulate { dest: SsaValue, accumulator: SsaValue, value: SsaValue, op: AccumulateOp },
    
    // Control flow
    Loop { index_var: SsaValue, start: SsaValue, end: SsaValue, body: BlockId },
    Branch { condition: SsaValue, true_block: BlockId, false_block: BlockId },
    Jump { target: BlockId },
    Return { value: Option<SsaValue> },
}

#[derive(Debug, Clone)]
pub enum AccumulateOp {
    Sum,
    Count,
    Min,
    Max,
}

#[derive(Debug, Clone)]
pub enum DataType {
    Int64,
    Float64,
    Array(Box<DataType>),
}

#[derive(Debug, Clone)]
pub struct SsaBlock {
    pub id: BlockId,
    pub instructions: Vec<SsaInstruction>,
    pub terminator: Option<SsaInstruction>, // Jump, Branch, or Return
}

#[derive(Debug, Clone)]
pub struct SsaProgram {
    pub blocks: HashMap<BlockId, SsaBlock>,
    pub entry_block: BlockId,
    pub value_types: HashMap<SsaValue, DataType>,
    pub next_value_id: u32,
    pub next_block_id: u32,
}
```

#### Phase 3: SSA IR Optimization
```rust
pub struct SsaOptimizer;

impl SsaOptimizer {
    pub fn optimize(program: &mut SsaProgram) -> Result<(), OptimizeError> {
        Self::dead_code_elimination(program)?;
        Self::common_subexpression_elimination(program)?;
        Self::constant_propagation(program)?;
        Self::loop_invariant_code_motion(program)?;
        Ok(())
    }
    
    fn dead_code_elimination(program: &mut SsaProgram) -> Result<(), OptimizeError> {
        // Remove unused SSA values and instructions
    }
    
    fn common_subexpression_elimination(program: &mut SsaProgram) -> Result<(), OptimizeError> {
        // Identify and eliminate redundant computations
    }
    
    fn constant_propagation(program: &mut SsaProgram) -> Result<(), OptimizeError> {
        // Propagate known constant values
    }
    
    fn loop_invariant_code_motion(program: &mut SsaProgram) -> Result<(), OptimizeError> {
        // Move loop-invariant computations outside loops
    }
}
```

#### Phase 4: Cranelift Code Generation
```rust
pub struct SsaToCraneliftConverter {
    builder_context: cranelift::frontend::FunctionBuilderContext,
    ctx: cranelift::codegen::Context,
    module: cranelift_jit::JITModule,
    value_map: HashMap<SsaValue, cranelift::prelude::Value>,
    block_map: HashMap<BlockId, cranelift::prelude::Block>,
}

impl SsaToCraneliftConverter {
    pub fn convert(ssa_program: &SsaProgram) -> Result<CompiledFunction, CompileError> {
        // Convert SSA IR to Cranelift IR
        // Generate optimized machine code
    }
}
```

### 3. AST to SSA IR Conversion\n\n#### SSA Builder\n```rust\npub struct SsaBuilder {\n    program: SsaProgram,\n    current_block: BlockId,\n    variable_map: HashMap<String, SsaValue>, // Column name -> SSA value mapping\n}\n\nimpl SsaBuilder {\n    pub fn new() -> Self {\n        let entry_block = BlockId(0);\n        let mut program = SsaProgram {\n            blocks: HashMap::new(),\n            entry_block,\n            value_types: HashMap::new(),\n            next_value_id: 0,\n            next_block_id: 1,\n        };\n        \n        program.blocks.insert(entry_block, SsaBlock {\n            id: entry_block,\n            instructions: Vec::new(),\n            terminator: None,\n        });\n        \n        Self {\n            program,\n            current_block: entry_block,\n            variable_map: HashMap::new(),\n        }\n    }\n    \n    pub fn convert_expr(&mut self, expr: &Expr, analysis: &ExprAnalysis) -> Result<SsaProgram, ConversionError> {\n        match analysis.output_type {\n            OutputType::Array(_) => self.convert_elementwise_expr(expr),\n            OutputType::Scalar(_) => self.convert_reduction_expr(expr),\n        }\n    }\n    \n    fn convert_elementwise_expr(&mut self, expr: &Expr) -> Result<SsaProgram, ConversionError> {\n        // Generate SSA for elementwise operations\n        // Pattern: for i in 0..array_length { output[i] = f(input1[i], input2[i], ...) }\n        \n        let array_length = self.new_value(DataType::Int64);\n        let loop_index = self.new_value(DataType::Int64);\n        let zero = self.new_value(DataType::Int64);\n        \n        // Load array length from first input array\n        let first_input = self.get_first_array_input(expr)?;\n        self.emit_instruction(SsaInstruction::LoadScalar { \n            dest: zero, \n            value: Value::Int64(0) \n        });\n        self.emit_instruction(SsaInstruction::ArrayLength { \n            dest: array_length, \n            array: first_input \n        });\n        \n        // Create loop body block\n        let loop_body = self.new_block();\n        let loop_exit = self.new_block();\n        \n        // Generate loop\n        self.emit_instruction(SsaInstruction::Loop {\n            index_var: loop_index,\n            start: zero,\n            end: array_length,\n            body: loop_body,\n        });\n        \n        // Switch to loop body\n        self.current_block = loop_body;\n        \n        // Convert expression for single element\n        let result_value = self.convert_expr_recursive(expr)?;\n        \n        // Store result\n        let output_array = self.get_output_array();\n        self.emit_instruction(SsaInstruction::ArrayAccess {\n            dest: self.new_value(DataType::Float64),\n            array: output_array,\n            index: loop_index,\n        });\n        \n        // Jump to loop exit\n        self.emit_instruction(SsaInstruction::Jump { target: loop_exit });\n        \n        Ok(self.program.clone())\n    }\n    \n    fn convert_reduction_expr(&mut self, expr: &Expr) -> Result<SsaProgram, ConversionError> {\n        // Generate SSA for reduction operations\n        // Pattern: acc = init; for i in 0..array_length { acc = reduce_op(acc, input[i]) }\n        \n        match expr {\n            Expr::Sum(inner_expr) => {\n                let accumulator = self.new_value(DataType::Float64);\n                self.emit_instruction(SsaInstruction::InitAccumulator {\n                    dest: accumulator,\n                    initial_value: Value::Float64(0.0),\n                    data_type: DataType::Float64,\n                });\n                \n                // Generate loop over array elements\n                let array_length = self.new_value(DataType::Int64);\n                let loop_index = self.new_value(DataType::Int64);\n                let zero = self.new_value(DataType::Int64);\n                \n                let first_input = self.get_first_array_input(inner_expr)?;\n                self.emit_instruction(SsaInstruction::LoadScalar { \n                    dest: zero, \n                    value: Value::Int64(0) \n                });\n                self.emit_instruction(SsaInstruction::ArrayLength { \n                    dest: array_length, \n                    array: first_input \n                });\n                \n                let loop_body = self.new_block();\n                let loop_exit = self.new_block();\n                \n                self.emit_instruction(SsaInstruction::Loop {\n                    index_var: loop_index,\n                    start: zero,\n                    end: array_length,\n                    body: loop_body,\n                });\n                \n                self.current_block = loop_body;\n                \n                // Evaluate inner expression for current index\n                let element_value = self.convert_expr_recursive(inner_expr)?;\n                \n                // Accumulate result\n                let new_accumulator = self.new_value(DataType::Float64);\n                self.emit_instruction(SsaInstruction::Accumulate {\n                    dest: new_accumulator,\n                    accumulator,\n                    value: element_value,\n                    op: AccumulateOp::Sum,\n                });\n                \n                self.emit_instruction(SsaInstruction::Jump { target: loop_exit });\n                \n                // Return accumulator value\n                self.current_block = loop_exit;\n                self.emit_instruction(SsaInstruction::Return { \n                    value: Some(new_accumulator) \n                });\n                \n                Ok(self.program.clone())\n            }\n            _ => Err(ConversionError::UnsupportedReduction),\n        }\n    }\n    \n    fn convert_expr_recursive(&mut self, expr: &Expr) -> Result<SsaValue, ConversionError> {\n        match expr {\n            Expr::Column(name) => {\n                if let Some(&value) = self.variable_map.get(name) {\n                    Ok(value)\n                } else {\n                    let value = self.new_value(DataType::Array(Box::new(DataType::Float64)));\n                    self.emit_instruction(SsaInstruction::LoadArray {\n                        dest: value,\n                        column: name.clone(),\n                        data_type: DataType::Array(Box::new(DataType::Float64)),\n                    });\n                    self.variable_map.insert(name.clone(), value);\n                    Ok(value)\n                }\n            }\n            \n            Expr::Literal(val) => {\n                let value = self.new_value(match val {\n                    Value::Int64(_) => DataType::Int64,\n                    Value::Float64(_) => DataType::Float64,\n                });\n                self.emit_instruction(SsaInstruction::LoadScalar {\n                    dest: value,\n                    value: val.clone(),\n                });\n                Ok(value)\n            }\n            \n            Expr::Add(operands) => {\n                let operand_values: Result<Vec<_>, _> = operands\n                    .iter()\n                    .map(|op| self.convert_expr_recursive(op))\n                    .collect();\n                let operand_values = operand_values?;\n                \n                let result = self.new_value(DataType::Float64);\n                self.emit_instruction(SsaInstruction::Add {\n                    dest: result,\n                    operands: operand_values,\n                });\n                Ok(result)\n            }\n            \n            Expr::Sub(lhs, rhs) => {\n                let lhs_val = self.convert_expr_recursive(lhs)?;\n                let rhs_val = self.convert_expr_recursive(rhs)?;\n                \n                let result = self.new_value(DataType::Float64);\n                self.emit_instruction(SsaInstruction::Sub {\n                    dest: result,\n                    lhs: lhs_val,\n                    rhs: rhs_val,\n                });\n                Ok(result)\n            }\n            \n            Expr::Mul(operands) => {\n                let operand_values: Result<Vec<_>, _> = operands\n                    .iter()\n                    .map(|op| self.convert_expr_recursive(op))\n                    .collect();\n                let operand_values = operand_values?;\n                \n                let result = self.new_value(DataType::Float64);\n                self.emit_instruction(SsaInstruction::Mul {\n                    dest: result,\n                    operands: operand_values,\n                });\n                Ok(result)\n            }\n            \n            Expr::Div(lhs, rhs) => {\n                let lhs_val = self.convert_expr_recursive(lhs)?;\n                let rhs_val = self.convert_expr_recursive(rhs)?;\n                \n                let result = self.new_value(DataType::Float64);\n                self.emit_instruction(SsaInstruction::Div {\n                    dest: result,\n                    lhs: lhs_val,\n                    rhs: rhs_val,\n                });\n                Ok(result)\n            }\n            \n            _ => Err(ConversionError::UnsupportedExpression),\n        }\n    }\n    \n    fn new_value(&mut self, data_type: DataType) -> SsaValue {\n        let value = SsaValue(self.program.next_value_id);\n        self.program.next_value_id += 1;\n        self.program.value_types.insert(value, data_type);\n        value\n    }\n    \n    fn new_block(&mut self) -> BlockId {\n        let block_id = BlockId(self.program.next_block_id);\n        self.program.next_block_id += 1;\n        self.program.blocks.insert(block_id, SsaBlock {\n            id: block_id,\n            instructions: Vec::new(),\n            terminator: None,\n        });\n        block_id\n    }\n    \n    fn emit_instruction(&mut self, instruction: SsaInstruction) {\n        if let Some(block) = self.program.blocks.get_mut(&self.current_block) {\n            block.instructions.push(instruction);\n        }\n    }\n    \n    fn get_first_array_input(&self, expr: &Expr) -> Result<SsaValue, ConversionError> {\n        // Helper to find first array input for determining loop bounds\n        match expr {\n            Expr::Column(name) => {\n                self.variable_map.get(name)\n                    .copied()\n                    .ok_or(ConversionError::UnknownColumn(name.clone()))\n            }\n            Expr::Add(operands) | Expr::Mul(operands) => {\n                for operand in operands {\n                    if let Ok(value) = self.get_first_array_input(operand) {\n                        return Ok(value);\n                    }\n                }\n                Err(ConversionError::NoArrayInput)\n            }\n            Expr::Sub(lhs, _) | Expr::Div(lhs, _) => self.get_first_array_input(lhs),\n            _ => Err(ConversionError::NoArrayInput),\n        }\n    }\n    \n    fn get_output_array(&mut self) -> SsaValue {\n        self.new_value(DataType::Array(Box::new(DataType::Float64)))\n    }\n}\n\n#[derive(Debug, thiserror::Error)]\npub enum ConversionError {\n    #[error(\"Unsupported expression type\")]\n    UnsupportedExpression,\n    #[error(\"Unsupported reduction operation\")]\n    UnsupportedReduction,\n    #[error(\"Unknown column: {0}\")]\n    UnknownColumn(String),\n    #[error(\"No array input found for determining loop bounds\")]\n    NoArrayInput,\n}\n```\n\n### 4. Array Pointer Management

#### Memory Layout Strategy
```rust
pub struct ArrayDescriptor {
    pub data_ptr: *const u8,
    pub length: usize,
    pub data_type: ArrowDataType,
    pub null_bitmap: Option<*const u8>,  // Future: null handling
}

pub struct ExecutionContext {
    pub input_arrays: HashMap<String, ArrayDescriptor>,
    pub output_buffer: *mut u8,
}
```

#### Zero-Copy Pointer Passing
- Use Cranelift's pointer types to pass array descriptors
- Generate functions with signature: `fn(input_ptrs: *const *const u8, lengths: *const usize, output: *mut u8)`
- Array data stays in original Arrow format, no copying required

### 4. Implementation Phases

#### Phase 1: Basic Arithmetic (MVP)
- [x] Expression DSL for basic operations (+, -, *, /)
- [x] SSA IR generation for elementwise operations
- [x] SSA IR optimization passes
- [x] Cranelift JIT compilation from SSA IR
- [x] Arrow array integration
- [x] Simple test cases

**Complete Compilation Example: `(+ a b c)`**

*AST:*
```rust
Add(vec![
    Column("a"),
    Column("b"), 
    Column("c")
])
```

*SSA IR:*
```rust
// Block 0 (entry):
%0 = LoadArray { column: "a", data_type: Array(Float64) }     // Load array 'a'
%1 = LoadArray { column: "b", data_type: Array(Float64) }     // Load array 'b'  
%2 = LoadArray { column: "c", data_type: Array(Float64) }     // Load array 'c'
%3 = LoadScalar { value: Int64(0) }                           // Loop start
%4 = ArrayLength { array: %0 }                               // Get array length
%5 = Loop { index_var: %6, start: %3, end: %4, body: Block1 } // Main loop

// Block 1 (loop body):
%7 = ArrayAccess { array: %0, index: %6 }                    // a[i]
%8 = ArrayAccess { array: %1, index: %6 }                    // b[i]
%9 = ArrayAccess { array: %2, index: %6 }                    // c[i]
%10 = Add { operands: [%7, %8, %9] }                         // a[i] + b[i] + c[i]
%11 = StoreArray { array: %output, index: %6, value: %10 }   // output[i] = result
Jump { target: Block0 }                                       // Continue loop
```

*Generated Cranelift IR (simplified):*
```cranelift
function u0:0(i64, i64, i64, i64, i64) -> i64 {
block0(v0: i64, v1: i64, v2: i64, v3: i64, v4: i64):  ; %0=a_ptr, %1=b_ptr, %2=c_ptr, %3=out_ptr, %4=length
    v5 = iconst.i64 0
    jump block1(v5)

block1(v6: i64):                                         ; Loop header
    v7 = icmp ult v6, v4                                ; i < length
    brif v7, block2, block3

block2:                                                  ; Loop body
    v8 = imul v6, 8                                     ; Calculate offset
    v9 = iadd v0, v8                                    ; a_ptr + offset
    v10 = load.f64 v9                                   ; Load a[i]
    v11 = iadd v1, v8                                   ; b_ptr + offset  
    v12 = load.f64 v11                                  ; Load b[i]
    v13 = iadd v2, v8                                   ; c_ptr + offset
    v14 = load.f64 v13                                  ; Load c[i]
    v15 = fadd v10, v12                                 ; a[i] + b[i]
    v16 = fadd v15, v14                                 ; + c[i]
    v17 = iadd v3, v8                                   ; output_ptr + offset
    store v16, v17                                      ; Store result
    v18 = iadd v6, 1                                    ; i++
    jump block1(v18)

block3:                                                  ; Loop exit
    return v3
}
```

#### Phase 2: Reductions
- [x] Sum, Count aggregation operations
- [x] Scalar output handling
- [x] Accumulator-based code generation
- [x] SSA IR support for reduction patterns

**Complete Reduction Example: `(sum (+ a b))`**

*AST:*
```rust
Sum(Box::new(Add(vec![
    Column("a"),
    Column("b")
])))
```

*SSA IR:*
```rust
// Block 0 (entry):
%0 = LoadArray { column: "a", data_type: Array(Float64) }     // Load array 'a'
%1 = LoadArray { column: "b", data_type: Array(Float64) }     // Load array 'b'
%2 = InitAccumulator { initial_value: Float64(0.0), data_type: Float64 } // acc = 0.0
%3 = LoadScalar { value: Int64(0) }                           // Loop start
%4 = ArrayLength { array: %0 }                               // Get array length
%5 = Loop { index_var: %6, start: %3, end: %4, body: Block1 } // Main loop

// Block 1 (loop body):
%7 = ArrayAccess { array: %0, index: %6 }                    // a[i]
%8 = ArrayAccess { array: %1, index: %6 }                    // b[i]
%9 = Add { operands: [%7, %8] }                              // a[i] + b[i]
%10 = Accumulate { accumulator: %2, value: %9, op: Sum }     // acc += (a[i] + b[i])
Jump { target: Block0 }                                       // Continue loop

// Block 2 (exit):
Return { value: %10 }                                         // Return accumulator
```

*Generated Cranelift IR (simplified):*
```cranelift
function u0:0(i64, i64, i64) -> f64 {
block0(v0: i64, v1: i64, v2: i64):                         ; %0=a_ptr, %1=b_ptr, %2=length
    v3 = f64const 0.0                                      ; Initialize accumulator
    v4 = iconst.i64 0                                      ; Initialize loop index
    jump block1(v4, v3)

block1(v5: i64, v6: f64):                                  ; Loop header
    v7 = icmp ult v5, v2                                   ; i < length
    brif v7, block2, block3

block2:                                                     ; Loop body
    v8 = imul v5, 8                                        ; Calculate offset
    v9 = iadd v0, v8                                       ; a_ptr + offset
    v10 = load.f64 v9                                      ; Load a[i]
    v11 = iadd v1, v8                                      ; b_ptr + offset
    v12 = load.f64 v11                                     ; Load b[i]
    v13 = fadd v10, v12                                    ; a[i] + b[i]
    v14 = fadd v6, v13                                     ; acc += (a[i] + b[i])
    v15 = iadd v5, 1                                       ; i++
    jump block1(v15, v14)

block3:                                                     ; Loop exit
    return v6                                              ; Return final accumulator
}
```

#### Benefits of SSA IR Architecture

**Optimization Opportunities:**
- **Common Subexpression Elimination**: `(+ a b)` computed once, reused multiple times
- **Dead Code Elimination**: Unused intermediate values are automatically removed
- **Constant Propagation**: Known values are propagated through the computation graph
- **Loop Invariant Code Motion**: Move expensive computations outside loops

**Analysis and Debugging:**
- **Clear Intermediate Representation**: Easy to inspect and debug generated code
- **Dependency Analysis**: Understand data flow and dependencies between operations
- **Performance Profiling**: Identify bottlenecks in the SSA IR before compilation

**Extensibility:**
- **Easy to Add Operations**: New operations just need SSA instruction definitions
- **Multiple Backends**: SSA IR can target different backends (LLVM, WASM, etc.)
- **Custom Optimizations**: Add domain-specific optimizations at SSA level

#### Phase 3: Advanced Features
- [ ] Null value handling with Arrow's null bitmaps
- [ ] More data types (integers, strings, dates)  
- [ ] Complex expressions with Let bindings
- [ ] SIMD optimization hints for Cranelift
- [ ] Advanced SSA optimizations (loop unrolling, vectorization)
- [ ] Multiple backend support (LLVM, WebAssembly)

### 5. Core Components

#### 5.1 Expression Parser
```rust
pub struct ExpressionParser;

impl ExpressionParser {
    pub fn parse(input: &str) -> Result<Expr, ParseError> {
        // Simple recursive descent parser
        // "(+ a b c)" -> Add(vec![Column("a"), Column("b"), Column("c")])
    }
}
```

#### 5.2 JIT Engine
```rust
pub struct JitEngine {
    ssa_converter: SsaToCraneliftConverter,
    compiled_functions: HashMap<ExpressionHash, CompiledFunction>,
    optimizer: SsaOptimizer,
}

impl JitEngine {
    pub fn compile(&mut self, expr: &Expr) -> Result<CompiledFunction, CompileError> {
        // Phase 1: Analyze expression
        let analysis = analyze_expression(expr)?;
        
        // Phase 2: Convert AST to SSA IR
        let mut ssa_builder = SsaBuilder::new();
        let mut ssa_program = ssa_builder.convert_expr(expr, &analysis)?;
        
        // Phase 3: Optimize SSA IR
        self.optimizer.optimize(&mut ssa_program)?;
        
        // Phase 4: Generate Cranelift IR and compile
        let compiled_function = self.ssa_converter.convert(&ssa_program)?;
        
        // Cache compiled function
        let expr_hash = self.hash_expression(expr);
        self.compiled_functions.insert(expr_hash, compiled_function.clone());
        
        Ok(compiled_function)
    }
    
    pub fn execute(&self, func: CompiledFunction, arrays: &ExecutionContext) -> Result<ArrayOutput, RuntimeError> {
        // Call compiled function with array pointers
        unsafe { func.call(arrays) }
    }
    
    fn hash_expression(&self, expr: &Expr) -> ExpressionHash {
        // Generate hash of expression for caching
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        expr.hash(&mut hasher);
        ExpressionHash(hasher.finish())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpressionHash(u64);
```

#### 5.3 SSA IR to Cranelift Conversion
```rust
impl SsaToCraneliftConverter {
    pub fn new() -> Self {
        let isa = cranelift_native::builder().unwrap().finish(cranelift::prelude::settings::Flags::new(cranelift::prelude::settings::builder())).unwrap();
        let module = cranelift_jit::JITModule::new(cranelift_jit::JITBuilder::with_isa(isa, cranelift_module::default_libcall_names()));
        
        Self {
            builder_context: cranelift::frontend::FunctionBuilderContext::new(),
            ctx: cranelift::codegen::Context::new(),
            module,
            value_map: HashMap::new(),
            block_map: HashMap::new(),
        }
    }
    
    pub fn convert(&mut self, ssa_program: &SsaProgram) -> Result<CompiledFunction, CompileError> {
        // Create function signature based on SSA program analysis
        let signature = self.create_function_signature(ssa_program)?;
        
        // Declare function in module
        let func_id = self.module.declare_function("compiled_expr", cranelift_module::Linkage::Local, &signature)?;
        
        // Create function context
        self.ctx.func = cranelift::prelude::Function::with_name_signature(cranelift::prelude::UserFuncName::user(0, 0), signature);
        
        {
            let mut builder = cranelift::frontend::FunctionBuilder::new(&mut self.ctx.func, &mut self.builder_context);
            
            // Create blocks for all SSA blocks
            for (&block_id, _) in &ssa_program.blocks {
                let cranelift_block = builder.create_block();\n                self.block_map.insert(block_id, cranelift_block);
            }
            
            // Convert entry block first
            let entry_block = self.block_map[&ssa_program.entry_block];
            builder.switch_to_block(entry_block);
            
            // Get function parameters (array pointers, lengths, etc.)
            let params = builder.block_params(entry_block).to_vec();
            self.setup_function_parameters(&params, ssa_program)?;
            
            // Convert all blocks
            for (&block_id, ssa_block) in &ssa_program.blocks {
                self.convert_block(&mut builder, ssa_block, ssa_program)?;
            }
            
            builder.seal_all_blocks();
            builder.finalize();
        }
        
        // Compile function
        let compiled_func = self.module.define_function(func_id, &mut self.ctx)?;
        let code_ptr = self.module.get_finalized_function(func_id);
        
        Ok(CompiledFunction::new(code_ptr, signature))
    }
    
    fn convert_block(&mut self, builder: &mut cranelift::frontend::FunctionBuilder, ssa_block: &SsaBlock, program: &SsaProgram) -> Result<(), CompileError> {
        let cranelift_block = self.block_map[&ssa_block.id];
        builder.switch_to_block(cranelift_block);
        
        // Convert each instruction
        for instruction in &ssa_block.instructions {
            self.convert_instruction(builder, instruction, program)?;
        }
        
        // Handle terminator instruction
        if let Some(terminator) = &ssa_block.terminator {
            self.convert_instruction(builder, terminator, program)?;
        }
        
        Ok(())
    }
    
    fn convert_instruction(&mut self, builder: &mut cranelift::frontend::FunctionBuilder, instruction: &SsaInstruction, program: &SsaProgram) -> Result<(), CompileError> {
        match instruction {
            SsaInstruction::LoadScalar { dest, value } => {
                let cranelift_value = match value {
                    Value::Int64(val) => builder.ins().iconst(cranelift::prelude::types::I64, *val),
                    Value::Float64(val) => builder.ins().f64const(*val),
                };
                self.value_map.insert(*dest, cranelift_value);
            }
            
            SsaInstruction::Add { dest, operands } => {
                if operands.len() == 2 {
                    let lhs = self.value_map[&operands[0]];
                    let rhs = self.value_map[&operands[1]];
                    let result = builder.ins().fadd(lhs, rhs);
                    self.value_map.insert(*dest, result);
                } else {
                    // Handle n-ary addition by chaining binary adds
                    let mut result = self.value_map[&operands[0]];
                    for &operand in &operands[1..] {
                        let rhs = self.value_map[&operand];
                        result = builder.ins().fadd(result, rhs);
                    }
                    self.value_map.insert(*dest, result);
                }
            }
            
            SsaInstruction::Sub { dest, lhs, rhs } => {
                let lhs_val = self.value_map[lhs];
                let rhs_val = self.value_map[rhs];
                let result = builder.ins().fsub(lhs_val, rhs_val);
                self.value_map.insert(*dest, result);
            }
            
            SsaInstruction::Mul { dest, operands } => {
                if operands.len() == 2 {
                    let lhs = self.value_map[&operands[0]];
                    let rhs = self.value_map[&operands[1]];
                    let result = builder.ins().fmul(lhs, rhs);
                    self.value_map.insert(*dest, result);
                } else {
                    // Handle n-ary multiplication
                    let mut result = self.value_map[&operands[0]];
                    for &operand in &operands[1..] {
                        let rhs = self.value_map[&operand];
                        result = builder.ins().fmul(result, rhs);
                    }
                    self.value_map.insert(*dest, result);
                }
            }
            
            SsaInstruction::Div { dest, lhs, rhs } => {
                let lhs_val = self.value_map[lhs];
                let rhs_val = self.value_map[rhs];
                let result = builder.ins().fdiv(lhs_val, rhs_val);
                self.value_map.insert(*dest, result);
            }
            
            SsaInstruction::ArrayAccess { dest, array, index } => {
                // Generate pointer arithmetic for array access
                let array_ptr = self.value_map[array];
                let index_val = self.value_map[index];
                
                // Calculate offset: index * sizeof(element)
                let elem_size = builder.ins().iconst(cranelift::prelude::types::I64, 8); // 8 bytes for f64
                let offset = builder.ins().imul(index_val, elem_size);
                
                // Add offset to base pointer
                let element_ptr = builder.ins().iadd(array_ptr, offset);
                
                // Load value from calculated address
                let loaded_value = builder.ins().load(cranelift::prelude::types::F64, cranelift::prelude::MemFlags::new(), element_ptr, 0);
                self.value_map.insert(*dest, loaded_value);
            }
            
            SsaInstruction::Loop { index_var, start, end, body } => {
                // Create loop header, body, and exit blocks
                let loop_header = builder.create_block();
                let loop_body_block = self.block_map[body];
                let loop_exit = builder.create_block();
                
                // Initialize loop variable
                let start_val = self.value_map[start];
                let end_val = self.value_map[end];
                
                // Jump to loop header
                builder.ins().jump(loop_header, &[start_val]);
                
                // Loop header: check condition and branch
                builder.switch_to_block(loop_header);
                builder.append_block_param(loop_header, cranelift::prelude::types::I64);
                let current_index = builder.block_params(loop_header)[0];
                self.value_map.insert(*index_var, current_index);
                
                let condition = builder.ins().icmp(cranelift::prelude::IntCC::UnsignedLessThan, current_index, end_val);
                builder.ins().brif(condition, loop_body_block, &[], loop_exit, &[]);
                
                // Loop body is handled separately as it's already converted
                // We just need to add the increment and back-edge
                builder.switch_to_block(loop_body_block);
                let one = builder.ins().iconst(cranelift::prelude::types::I64, 1);
                let next_index = builder.ins().iadd(current_index, one);
                builder.ins().jump(loop_header, &[next_index]);
                
                // Continue with loop exit
                builder.switch_to_block(loop_exit);
            }
            
            SsaInstruction::Return { value } => {
                if let Some(val) = value {
                    let return_val = self.value_map[val];
                    builder.ins().return_(&[return_val]);
                } else {
                    builder.ins().return_(&[]);
                }
            }
            
            _ => {
                // Handle other instructions as needed
                return Err(CompileError::UnsupportedInstruction);
            }
        }
        
        Ok(())
    }
    
    fn create_function_signature(&self, ssa_program: &SsaProgram) -> Result<cranelift::prelude::Signature, CompileError> {
        let mut sig = self.module.make_signature();
        
        // Add parameters for input arrays (pointers + lengths)
        // Add return type based on program analysis
        // This is simplified - real implementation would analyze SSA program
        
        sig.params.push(cranelift::prelude::AbiParam::new(cranelift::prelude::types::I64)); // array pointer
        sig.params.push(cranelift::prelude::AbiParam::new(cranelift::prelude::types::I64)); // array length
        sig.returns.push(cranelift::prelude::AbiParam::new(cranelift::prelude::types::F64)); // result
        
        Ok(sig)
    }
    
    fn setup_function_parameters(&mut self, params: &[cranelift::prelude::Value], ssa_program: &SsaProgram) -> Result<(), CompileError> {
        // Map SSA values for input parameters
        // This would be more sophisticated in real implementation
        Ok(())
    }
}

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error(\"Unsupported SSA instruction\")]
    UnsupportedInstruction,
    #[error(\"Cranelift error: {0}\")]
    CraneliftError(String),
}

#[derive(Debug, Clone)]
pub struct CompiledFunction {
    code_ptr: *const u8,
    signature: cranelift::prelude::Signature,
}

impl CompiledFunction {
    pub fn new(code_ptr: *const u8, signature: cranelift::prelude::Signature) -> Self {
        Self { code_ptr, signature }
    }
    
    pub unsafe fn call(&self, args: &ExecutionContext) -> Result<ArrayOutput, RuntimeError> {
        // Cast function pointer and call with appropriate arguments
        // This is highly unsafe and simplified for illustration
        todo!(\"Implement safe function calling\")
    }
}
```

### 6. Performance Optimizations

#### SSA IR Optimization Passes

**Dead Code Elimination**
```rust
impl SsaOptimizer {
    fn dead_code_elimination(&mut self, program: &mut SsaProgram) -> Result<(), OptimizeError> {
        let mut live_values = HashSet::new();
        let mut worklist = Vec::new();
        
        // Mark all values used in return statements as live
        for block in program.blocks.values() {
            if let Some(SsaInstruction::Return { value: Some(val) }) = &block.terminator {
                if !live_values.contains(val) {
                    live_values.insert(*val);
                    worklist.push(*val);
                }
            }
        }
        
        // Propagate liveness backwards
        while let Some(value) = worklist.pop() {
            for block in program.blocks.values() {
                for instruction in &block.instructions {
                    if instruction.defines_value(value) {
                        // Mark all operands as live
                        for operand in instruction.operands() {
                            if !live_values.contains(&operand) {
                                live_values.insert(operand);
                                worklist.push(operand);
                            }
                        }
                    }
                }
            }
        }
        
        // Remove dead instructions
        for block in program.blocks.values_mut() {
            block.instructions.retain(|inst| {
                if let Some(def) = inst.defined_value() {
                    live_values.contains(&def)
                } else {
                    true // Keep side-effecting instructions
                }
            });
        }
        
        Ok(())
    }
}
```

**Common Subexpression Elimination**
```rust
impl SsaOptimizer {
    fn common_subexpression_elimination(&mut self, program: &mut SsaProgram) -> Result<(), OptimizeError> {
        let mut expression_map: HashMap<InstructionKey, SsaValue> = HashMap::new();
        let mut value_replacements: HashMap<SsaValue, SsaValue> = HashMap::new();
        
        for block in program.blocks.values_mut() {
            for instruction in &mut block.instructions {
                if let Some(key) = instruction.to_key() {
                    if let Some(&existing_value) = expression_map.get(&key) {
                        // Found duplicate expression
                        if let Some(defined_value) = instruction.defined_value() {
                            value_replacements.insert(defined_value, existing_value);
                        }
                    } else {
                        // New expression
                        if let Some(defined_value) = instruction.defined_value() {
                            expression_map.insert(key, defined_value);
                        }
                    }
                }
                
                // Apply replacements to operands
                instruction.replace_operands(&value_replacements);
            }
        }
        
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum InstructionKey {
    Add(Vec<SsaValue>),
    Sub(SsaValue, SsaValue),
    Mul(Vec<SsaValue>),
    Div(SsaValue, SsaValue),
    LoadScalar(Value),
}
```

**Loop Invariant Code Motion**
```rust
impl SsaOptimizer {
    fn loop_invariant_code_motion(&mut self, program: &mut SsaProgram) -> Result<(), OptimizeError> {
        // Identify loop structures
        let loop_info = self.analyze_loops(program)?;
        
        for loop_header in loop_info.loop_headers {
            let loop_blocks = self.find_loop_blocks(program, loop_header)?;
            let mut invariant_instructions = Vec::new();
            
            // Find loop-invariant instructions
            for &block_id in &loop_blocks {
                let block = &program.blocks[&block_id];
                for (idx, instruction) in block.instructions.iter().enumerate() {
                    if self.is_loop_invariant(instruction, &loop_blocks, program) {
                        invariant_instructions.push((block_id, idx));
                    }
                }
            }
            
            // Move invariant instructions to loop preheader
            let preheader = self.get_or_create_preheader(program, loop_header)?;
            for (block_id, instruction_idx) in invariant_instructions.into_iter().rev() {
                let instruction = program.blocks.get_mut(&block_id).unwrap()
                    .instructions.remove(instruction_idx);
                program.blocks.get_mut(&preheader).unwrap()
                    .instructions.push(instruction);
            }
        }
        
        Ok(())
    }
    
    fn is_loop_invariant(&self, instruction: &SsaInstruction, loop_blocks: &HashSet<BlockId>, program: &SsaProgram) -> bool {
        // An instruction is loop-invariant if all its operands are either:
        // 1. Constants
        // 2. Defined outside the loop
        // 3. Loop-invariant themselves
        
        for operand in instruction.operands() {
            if !self.is_value_loop_invariant(operand, loop_blocks, program) {
                return false;
            }
        }
        true
    }
}
```

#### Compilation Caching
- **Expression hashing**: Hash SSA IR programs to avoid recompilation of equivalent expressions
- **Multi-level caching**: Cache at AST, SSA IR, and compiled function levels
- **LRU eviction**: Intelligent cache eviction based on usage patterns
- **Persistent cache**: Store compiled functions across sessions with versioning

#### Memory Efficiency
- **Zero-copy execution**: Direct pointer manipulation, no array copying during execution
- **SSA value reuse**: Reuse SSA values when safe to reduce register pressure
- **Stack allocation**: Use stack for small intermediates and loop variables
- **Buffer reuse**: Reuse output buffers across multiple expression evaluations

#### SSA IR Level Optimizations
- **Constant propagation**: Propagate known constants through SSA values
- **Strength reduction**: Replace expensive operations (e.g., `x * 2` → `x + x`)
- **Loop unrolling**: Unroll small loops for better instruction-level parallelism
- **Vectorization preparation**: Arrange instructions for SIMD generation

#### SIMD and Vectorization
- **Auto-vectorization**: Cranelift can generate SIMD instructions for simple loops
- **Explicit vectorization**: SSA IR can be extended with vector operations
- **Array operation fusion**: Combine multiple operations into single vectorized loops
- **Target-specific optimizations**: Use AVX, NEON, or other ISA extensions

### 7. Testing Strategy

#### Unit Tests
```rust
#[cfg(test)]
mod tests {
    #[test]
    fn test_simple_addition() {
        let expr = parse("(+ a b)").unwrap();
        let arrays = create_test_arrays(&[
            ("a", vec![1.0, 2.0, 3.0]),
            ("b", vec![4.0, 5.0, 6.0]),
        ]);
        let result = execute_expression(expr, arrays);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }
    
    #[test]
    fn test_sum_reduction() {
        let expr = parse("(sum a)").unwrap();
        let arrays = create_test_arrays(&[
            ("a", vec![1.0, 2.0, 3.0, 4.0]),
        ]);
        let result = execute_expression(expr, arrays);
        assert_eq!(result, 10.0);
    }
}
```

#### Integration Tests
- Complex nested expressions
- Large array performance benchmarks
- Memory leak detection
- Comparison with naive interpreted evaluation

### 8. Dependencies

```toml
[dependencies]
cranelift = "0.105"
cranelift-jit = "0.105"
cranelift-module = "0.105"
cranelift-frontend = "0.105"
arrow = "50.0"
thiserror = "1.0"

[dev-dependencies]
criterion = "0.5"  # For benchmarking
```

### 9. Future Enhancements

#### Advanced Features
- **String operations**: Concatenation, substring, regex
- **Date/time arithmetic**: Temporal operations
- **Conditional expressions**: If-then-else logic
- **User-defined functions**: Plugin system for custom operations

#### Optimization Opportunities
- **Fused operations**: Combine multiple ops into single loop
- **Lazy evaluation**: Only compute required columns
- **Parallel execution**: Multi-threading for large arrays
- **GPU acceleration**: Future CUDA/OpenCL backend

### 10. Implementation Timeline

1. **Week 1-2**: Basic DSL and expression parsing
2. **Week 3-4**: Cranelift integration and simple arithmetic
3. **Week 5-6**: Arrow array integration and pointer management
4. **Week 7-8**: Reduction operations and aggregations
5. **Week 9-10**: Performance optimization and caching
6. **Week 11-12**: Comprehensive testing and documentation

This plan provides a solid foundation for building a high-performance columnar expression evaluator that can compete with established systems like Apache Arrow Compute and DataFusion's physical expressions.