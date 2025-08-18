# Dio Expression Evaluator

A high-performance JIT compiled columnar expression evaluator for arithmetic operations on Apache Arrow arrays.

## Phase 1: Lisp Parser to AST ✅

This phase implements a robust parser for Lisp-like arithmetic expressions, converting them into an Abstract Syntax Tree (AST).

### Features Implemented

- **Complete Lisp Expression Parser** using nom combinators
- **Rich AST** with support for:
  - Column references (`a`, `column_name`)
  - Numeric literals (integers and floats with scientific notation)
  - Arithmetic operations (`+`, `-`, `*`, `/`)
  - Reduction operations (`sum`, `count`)
  - Variable bindings (`let`)
- **Comprehensive Error Handling** with detailed error messages
- **Property-based Testing** with proptest for robustness
- **Zero-copy Parsing** for optimal performance

### Expression Syntax

#### Basic Operations
```lisp
a                          ; Column reference
42                         ; Integer literal
3.14                       ; Float literal  
1.5e-3                     ; Scientific notation

(+ a b c)                  ; N-ary addition
(- x y)                    ; Binary subtraction
(* price 1.1)              ; N-ary multiplication
(/ total count)            ; Binary division
```

#### Reductions
```lisp
(sum sales)                ; Sum reduction
(count records)            ; Count reduction
(sum (+ price tax))        ; Sum of expression
```

#### Nested Expressions
```lisp
(+ a (* b 2))                              ; Simple nesting
(/ (sum revenue) (count customers))        ; Average pattern
(sum (+ price (* quantity discount)))      ; Complex expression
```

#### Variable Bindings (Future)
```lisp
(let tmp (+ a b))          ; Variable binding
```

### Usage

```rust
use dio::parse_expr;

fn main() {
    // Parse a simple expression
    let expr = parse_expr("(+ a (* b 2))").unwrap();
    println!("Parsed: {}", expr);
    
    // Check expression properties
    if expr.is_elementwise() {
        println!("This is an elementwise operation");
    }
    
    let columns = expr.get_column_references();
    println!("Columns used: {:?}", columns);
    
    println!("Complexity: {}", expr.complexity());
}
```

### Error Handling

The parser provides detailed error messages with source location information:

```rust
match parse_expr("(+ a b") {
    Ok(expr) => println!("Success: {}", expr),
    Err(e) => println!("Error: {}", e),
}
```

Example error output:
```
Parse error: Expected 'character', found ' '
```

### Testing

Run the comprehensive test suite:

```bash
cargo test
```

Run the interactive demo:
```bash
cargo run --example basic_parser
```

### Performance

- **Zero-copy parsing**: Input strings are parsed without allocation
- **Efficient AST**: Compact representation with minimal overhead
- **Fast combinators**: nom-based parser is highly optimized
- **Property testing**: Ensures correctness across diverse inputs

### Architecture

```
Input String → nom Parser → AST (Expr enum)
```

The parser uses nom combinators for:
- **Whitespace handling**: Flexible whitespace between tokens
- **Error recovery**: Meaningful error messages with cut combinators
- **Recursive parsing**: Proper handling of nested expressions
- **Type-safe parsing**: Strong typing throughout the pipeline

### AST Structure

```rust
pub enum Expr {
    Column(String),                    // Column reference
    Literal(Value),                    // Numeric literal
    Add(Vec<Expr>),                    // N-ary addition
    Sub(Box<Expr>, Box<Expr>),         // Binary subtraction
    Mul(Vec<Expr>),                    // N-ary multiplication
    Div(Box<Expr>, Box<Expr>),         // Binary division
    Sum(Box<Expr>),                    // Sum reduction
    Count(Box<Expr>),                  // Count reduction
    Let(String, Box<Expr>),            // Variable binding
}

pub enum Value {
    Int64(i64),                        // Integer values
    Float64(OrderedFloat64),           // Float values (with Eq/Hash)
}
```

### What's Next

This parser implementation is the foundation for the complete Dio expression evaluator. Future phases will add:

1. **SSA IR Generation**: Convert AST to Static Single Assignment form
2. **Optimization Passes**: Dead code elimination, common subexpression elimination
3. **Cranelift Integration**: JIT compilation to machine code
4. **Arrow Integration**: Zero-copy execution on columnar data
5. **Advanced Features**: Null handling, more data types, SIMD optimization

See `docs/dio1.md` for the complete architecture plan.

## Dependencies

- **nom**: Parser combinators for flexible, efficient parsing
- **thiserror**: Ergonomic error handling
- **ariadne**: Beautiful diagnostic error reporting
- **proptest**: Property-based testing for robustness

## License

This project is part of the Dio expression evaluator research implementation.