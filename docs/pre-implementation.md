# Pre-Implementation Analysis: Rust Crates for Dio Expression Evaluator

This document analyzes the most suitable Rust crates for implementing the Dio JIT compiled columnar expression evaluator, based on the architecture outlined in `dio1.md`.

## Executive Summary

**Recommended Core Stack:**
- **Parsing**: `nom` for flexible expression parsing
- **AST**: Custom enum (simple, no need for egg complexity)
- **SSA IR**: Custom implementation with `petgraph` for CFG analysis
- **JIT Compilation**: `cranelift` ecosystem
- **Error Handling**: `ariadne` for beautiful diagnostics
- **Arrow Integration**: `arrow` crate with custom array descriptors
- **Testing**: `criterion` for benchmarking, `proptest` for property testing

## 1. Core Architecture & JIT Compilation

### Cranelift Ecosystem ⭐⭐⭐⭐⭐ (RECOMMENDED)

```toml
cranelift = "0.105"
cranelift-jit = "0.105" 
cranelift-module = "0.105"
cranelift-frontend = "0.105"
cranelift-codegen = "0.105"
```

**Why Cranelift over alternatives:**
- **Built-in SSA support**: Perfect match for our SSA IR design
- **Fast compilation**: 10x faster than LLVM for JIT scenarios  
- **Excellent documentation**: Well-maintained with clear examples
- **Memory safety**: Rust-native, safer than LLVM bindings
- **Active development**: Actively used by Wasmtime, well-supported

**Alternatives considered:**
- `inkwell` (LLVM bindings): Slower compilation, more complex setup
- Direct assembly generation: Too low-level, error-prone

### Control Flow Graph Analysis

**petgraph** ⭐⭐⭐⭐ (RECOMMENDED)
```toml
petgraph = "0.6"
```

**Usage in Dio:**
- CFG construction for SSA IR blocks
- Dominance tree computation for SSA optimization
- Loop detection for invariant code motion
- Graph algorithms for optimization passes

```rust
use petgraph::Graph;
use petgraph::algo::dominators;

pub struct ControlFlowGraph {
    graph: Graph<BlockId, ()>,
    dominance_tree: Option<dominators::Dominators<petgraph::graph::NodeIndex>>,
}
```

## 2. AST Representation

### Custom Enum vs Egg E-graphs

**Recommendation: Custom Enum** ⭐⭐⭐⭐⭐

Our expression AST is simple enough that a custom enum is the best approach:

```rust
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Expr {
    Column(String),
    Literal(Value),
    Add(Vec<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Vec<Expr>),
    Div(Box<Expr>, Box<Expr>),
    Sum(Box<Expr>),
    Count(Box<Expr>),
    Let(String, Box<Expr>),
}
```

**Why not egg?** ⭐⭐ (NOT RECOMMENDED for our use case)
```toml
egg = "0.9"  # Don't use for Dio
```

**Reasons against egg:**
- **Overkill complexity**: Egg is designed for complex term rewriting and equality saturation
- **Our AST is simple**: Basic arithmetic expressions don't need e-graph sophistication  
- **Single-pass compilation**: We convert AST→SSA once, no need for complex transformations
- **Performance overhead**: E-graphs add memory and computation overhead for simple cases
- **Learning curve**: Adds complexity without significant benefit

**When egg would be useful:**
- Complex algebraic simplifications (e.g., `x + 0 = x`, `x * 1 = x`)
- Mathematical identity discovery
- Complex optimization with many equivalent forms

**For Dio**: Our optimizations happen at the SSA IR level, making egg unnecessary.

## 3. Parsing & Frontend

### Expression Parsing: nom vs alternatives

**nom** ⭐⭐⭐⭐⭐ (RECOMMENDED)
```toml
nom = "7.1"
```

**Perfect for Dio because:**
- **Flexible combinators**: Ideal for Lisp-like syntax `(+ a b c)`
- **Great error recovery**: Can provide meaningful parse errors
- **Zero-copy parsing**: Efficient for our use case
- **Incremental parsing**: Easy to extend with new operations

**Example nom parser:**
```rust
use nom::{
    branch::alt,
    bytes::complete::tag,
    character::complete::{alpha1, char, multispace0, multispace1},
    combinator::map,
    multi::separated_list0,
    sequence::{delimited, preceded},
    IResult,
};

fn parse_expr(input: &str) -> IResult<&str, Expr> {
    alt((
        parse_list_expr,
        parse_column,
        parse_literal,
    ))(input)
}

fn parse_list_expr(input: &str) -> IResult<&str, Expr> {
    delimited(
        char('('),
        preceded(
            multispace0,
            alt((
                parse_add,
                parse_sub,
                parse_sum,
            ))
        ),
        preceded(multispace0, char(')')),
    )(input)
}
```

**Alternatives considered:**

**lalrpop** ⭐⭐⭐ (GOOD but not needed)
```toml
lalrpop = "0.20"
```
- Great for complex grammars
- Overkill for simple Lisp-like syntax
- Build-time code generation adds complexity

**pest** ⭐⭐⭐ (GOOD but less flexible)  
```toml
pest = "2.7"
```
- Clean PEG grammar syntax
- Less flexible than nom for our dynamic needs
- External grammar files add complexity

## 4. SSA IR Representation

### Custom SSA IR Implementation ⭐⭐⭐⭐⭐ (RECOMMENDED)

**Recommendation**: Build custom SSA IR as designed in `dio1.md`

```rust
pub struct SsaProgram {
    pub blocks: HashMap<BlockId, SsaBlock>,
    pub entry_block: BlockId,
    pub value_types: HashMap<SsaValue, DataType>,
    // ... rest of design from dio1.md
}
```

**Why custom over existing solutions:**
- **Domain-specific**: Tailored for columnar operations and Arrow arrays
- **Simple and efficient**: No unnecessary features or overhead
- **Full control**: Can optimize for our specific use cases
- **Integration**: Designed specifically for Cranelift backend

**Supporting crate for SSA construction:**
**petgraph** for dominance analysis and CFG algorithms (already recommended above)

## 5. Error Handling & Diagnostics

### ariadne ⭐⭐⭐⭐⭐ (RECOMMENDED)

```toml
ariadne = "0.4"
```

**Perfect for Dio:**
- **Beautiful output**: Unicode-rich diagnostic messages
- **Source highlighting**: Points to exact error locations
- **Multi-stage errors**: Can show AST, SSA IR, and Cranelift errors
- **Customizable themes**: Professional appearance

**Example integration:**
```rust
use ariadne::{Color, Fmt, Label, Report, ReportKind, Source};

pub fn report_parse_error(source: &str, error: ParseError) {
    Report::build(ReportKind::Error, (), error.span.start)
        .with_code(3)
        .with_message("Parse error in expression")
        .with_label(
            Label::new(error.span.clone())
                .with_message(format!("Expected {}, found {}", 
                    error.expected.fg(Color::Green), 
                    error.found.fg(Color::Red)))
                .with_color(Color::Red)
        )
        .finish()
        .print(Source::from(source))
        .unwrap();
}
```

**Alternative considered:**

**codespan-reporting** ⭐⭐⭐⭐ (EXCELLENT alternative)
```toml
codespan-reporting = "0.11"
```
- More mature and stable
- Used by many Rust projects
- Slightly more verbose API
- Both are excellent choices

## 6. Arrow Integration & Array Processing

### arrow crate ecosystem ⭐⭐⭐⭐⭐ (REQUIRED)

```toml
arrow = "50.0"
arrow-array = "50.0"
arrow-buffer = "50.0"
arrow-data = "50.0"
arrow-schema = "50.0"
```

**Integration strategy:**
- Use Arrow's native array formats
- Custom `ArrayDescriptor` wrapper for JIT integration
- Zero-copy pointer extraction for Cranelift functions

```rust
use arrow::array::{Float64Array, Int64Array};
use arrow::buffer::Buffer;

pub struct ArrayDescriptor {
    pub data_ptr: *const u8,
    pub length: usize,
    pub data_type: ArrowDataType,
    pub null_bitmap: Option<*const u8>,
}

impl From<&Float64Array> for ArrayDescriptor {
    fn from(array: &Float64Array) -> Self {
        Self {
            data_ptr: array.values().as_ptr() as *const u8,
            length: array.len(),
            data_type: ArrowDataType::Float64,
            null_bitmap: array.nulls().map(|nulls| nulls.buffer().as_ptr()),
        }
    }
}
```

## 7. Testing & Quality Assurance

### Benchmarking: criterion ⭐⭐⭐⭐⭐ (RECOMMENDED)

```toml
[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
```

**Usage for Dio:**
- JIT compilation time benchmarks
- Expression evaluation performance
- Memory usage analysis
- Comparison with interpreted evaluation

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn benchmark_compilation(c: &mut Criterion) {
    let expr = parse("(+ a b c)").unwrap();
    let arrays = create_test_arrays(1000000);
    
    c.bench_function("compile_add_three", |b| {
        b.iter(|| {
            let mut engine = JitEngine::new();
            black_box(engine.compile(&expr))
        })
    });
}
```

### Property Testing: proptest ⭐⭐⭐⭐ (RECOMMENDED)

```toml
proptest = "1.4"
```

**Usage for Dio:**
- Generate random valid expressions
- Test compilation pipeline with diverse inputs
- Verify optimization correctness

```rust
use proptest::prelude::*;

prop_compose! {
    fn arb_expr(depth: u32)(op in arb_binop(), 
                           left in arb_expr_recursive(depth/2),
                           right in arb_expr_recursive(depth/2)) -> Expr {
        match op {
            BinOp::Add => Expr::Add(vec![left, right]),
            BinOp::Sub => Expr::Sub(Box::new(left), Box::new(right)),
            // ... other operations
        }
    }
}
```

## 8. Additional Utility Crates

### Error Handling: thiserror ⭐⭐⭐⭐⭐ (RECOMMENDED)

```toml
thiserror = "1.0"
```

**Clean error definitions:**
```rust
#[derive(Debug, thiserror::Error)]
pub enum DioError {
    #[error("Parse error: {0}")]
    Parse(#[from] ParseError),
    
    #[error("SSA conversion error: {0}")]
    SsaConversion(#[from] ConversionError),
    
    #[error("Compilation error: {0}")]
    Compilation(#[from] CompileError),
    
    #[error("Runtime error: {0}")]
    Runtime(String),
}
```

### Hashing: ahash ⭐⭐⭐⭐ (RECOMMENDED for caching)

```toml
ahash = "0.8"
```

**For expression caching:**
```rust
use ahash::AHashMap;

pub struct JitEngine {
    compiled_functions: AHashMap<ExpressionHash, CompiledFunction>,
    // ...
}
```

### Serialization: serde ⭐⭐⭐⭐ (OPTIONAL)

```toml
serde = { version = "1.0", features = ["derive"] }
```

**For persistent caching:**
```rust
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CachedFunction {
    pub expression_hash: u64,
    pub compiled_code: Vec<u8>,
    pub signature: FunctionSignature,
}
```

## 9. Complete Cargo.toml Recommendation

```toml
[package]
name = "dio"
version = "0.1.0"
edition = "2021"

[dependencies]
# Core JIT compilation
cranelift = "0.105"
cranelift-jit = "0.105"
cranelift-module = "0.105"
cranelift-frontend = "0.105"
cranelift-codegen = "0.105"

# Arrow integration
arrow = "50.0"
arrow-array = "50.0"

# Parsing
nom = "7.1"

# Error handling & diagnostics  
thiserror = "1.0"
ariadne = "0.4"

# Graph algorithms for SSA
petgraph = "0.6"

# Performance utilities
ahash = "0.8"

# Optional features
serde = { version = "1.0", features = ["derive"], optional = true }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1.4"

[features]
default = []
serialization = ["serde"]

[[bench]]
name = "expression_evaluation"
harness = false
```

## 10. Implementation Roadmap

### Phase 1: Foundation (Weeks 1-2)
1. Set up basic project structure with recommended crates
2. Implement AST with custom enum
3. Build nom-based parser for basic expressions
4. Set up ariadne error reporting

### Phase 2: SSA IR (Weeks 3-4)  
1. Implement custom SSA IR data structures
2. Build AST to SSA conversion with petgraph CFG
3. Add basic SSA optimizations (dead code elimination)
4. Comprehensive testing with proptest

### Phase 3: JIT Integration (Weeks 5-6)
1. Implement Cranelift backend for SSA IR
2. Arrow array integration with zero-copy pointers
3. Build execution engine with caching
4. Performance benchmarking with criterion

### Phase 4: Optimization & Polish (Weeks 7-8)
1. Advanced SSA optimizations
2. Error handling polish with ariadne
3. Documentation and examples
4. Production readiness

## 11. Dependency Analysis

**Compilation Time Impact:**
- **Fast**: nom, thiserror, ahash, ariadne (~5-10s incremental)
- **Medium**: arrow, petgraph (~10-15s incremental)  
- **Slow**: cranelift ecosystem (~30-45s incremental, ~2-3min clean)

**Runtime Performance:**
- **Zero overhead**: Custom AST, SSA IR, ahash
- **Minimal overhead**: arrow integration, petgraph
- **High performance**: cranelift JIT compilation

**Maintenance & Ecosystem:**
- All recommended crates are well-maintained with active communities
- Cranelift ecosystem is backed by Bytecode Alliance
- Arrow is part of Apache Arrow project with excellent stability

This stack provides an optimal balance of performance, maintainability, and development velocity for the Dio expression evaluator.