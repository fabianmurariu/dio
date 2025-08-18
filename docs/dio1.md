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

#### Phase 1: Expression Analysis
```rust
pub struct ExprAnalysis {
    pub input_columns: Vec<String>,
    pub output_type: OutputType,
    pub requires_loop: bool,
}

pub enum OutputType {
    Array(ArrowDataType),    // Elementwise operations
    Scalar(ArrowDataType),   // Reductions
}
```

#### Phase 2: Code Generation Strategy
```rust
pub enum CompilationStrategy {
    ElementwiseLoop {
        input_arrays: Vec<String>,
        output_array: String,
        loop_body: Vec<Instruction>,
    },
    ReductionLoop {
        input_arrays: Vec<String>,
        accumulator: String,
        loop_body: Vec<Instruction>,
        initial_value: Value,
    },
}
```

#### Phase 3: Cranelift JIT Compilation
```rust
pub struct JitCompiler {
    builder_context: cranelift::frontend::FunctionBuilderContext,
    ctx: cranelift::codegen::Context,
    module: cranelift_jit::JITModule,
}
```

### 3. Array Pointer Management

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
- [x] Cranelift JIT compilation for elementwise operations
- [x] Arrow array integration
- [x] Simple test cases

**Function Generation Example:**
```rust
// For expression: (+ a b c)
// Generated signature: fn(*const f64, *const f64, *const f64, *mut f64, usize)
pub fn generate_add_three_arrays() -> CompiledFunction {
    // Cranelift IR generation:
    // for i in 0..length {
    //     output[i] = input_a[i] + input_b[i] + input_c[i]
    // }
}
```

#### Phase 2: Reductions
- [x] Sum, Count aggregation operations
- [x] Scalar output handling
- [x] Accumulator-based code generation

**Reduction Example:**
```rust
// For expression: (sum (+ a b))
// Generated signature: fn(*const f64, *const f64, usize) -> f64
pub fn generate_sum_add() -> CompiledFunction {
    // Cranelift IR generation:
    // let mut acc = 0.0;
    // for i in 0..length {
    //     acc += input_a[i] + input_b[i]
    // }
    // return acc;
}
```

#### Phase 3: Advanced Features
- [ ] Null value handling with Arrow's null bitmaps
- [ ] More data types (integers, strings, dates)
- [ ] Complex expressions with Let bindings
- [ ] SIMD optimization hints for Cranelift

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
    compiler: JitCompiler,
    compiled_functions: HashMap<ExpressionHash, CompiledFunction>,
}

impl JitEngine {
    pub fn compile(&mut self, expr: &Expr) -> Result<CompiledFunction, CompileError> {
        let analysis = analyze_expression(expr)?;
        let strategy = plan_compilation(analysis)?;
        self.generate_cranelift_ir(strategy)
    }
    
    pub fn execute(&self, func: CompiledFunction, arrays: &ExecutionContext) -> Result<ArrayOutput, RuntimeError> {
        // Call compiled function with array pointers
        unsafe { func.call(arrays) }
    }
}
```

#### 5.3 Cranelift Integration
```rust
impl JitEngine {
    fn generate_cranelift_ir(&mut self, strategy: CompilationStrategy) -> Result<CompiledFunction, CompileError> {
        let mut builder = FunctionBuilder::new(&mut self.func, &mut self.builder_context);
        
        match strategy {
            CompilationStrategy::ElementwiseLoop { .. } => {
                // Generate loop with array access patterns
                self.generate_elementwise_loop(&mut builder, strategy)
            }
            CompilationStrategy::ReductionLoop { .. } => {
                // Generate accumulator-based reduction loop
                self.generate_reduction_loop(&mut builder, strategy)
            }
        }
    }
}
```

### 6. Performance Optimizations

#### Compilation Caching
- Hash expressions to avoid recompilation
- LRU cache for compiled functions
- Persistent cache across sessions

#### Memory Efficiency
- Direct pointer manipulation, no array copying
- Stack allocation for small intermediates
- Reuse output buffers when possible

#### SIMD Potential
- Cranelift can generate vectorized code
- Future: explicit SIMD operations for arithmetic
- Auto-vectorization for simple loops

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