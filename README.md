# Dio v3: JIT Compiled Columnar Expression Evaluator

A high-performance JIT compiled columnar expression evaluator that transforms arithmetic expressions written in a Lisp-like DSL into optimized machine code using a multi-stage compilation pipeline.

## 📚 **NEW: Complete Documentation Available**

**📖 [Read the full Dio v3 documentation →](docs/dio3.md)**

**⚡ [Understanding Generic Execution →](docs/generic_execution.md)**

The new documentation covers:
- Complete architecture overview of the new ByteCode pipeline
- Detailed explanation of all compilation stages
- Performance characteristics and optimization opportunities  
- Getting started guide for new contributors
- Comprehensive API reference

## 🆕 New ByteCode Pipeline (v3)

Dio v3 introduces a revolutionary **4-stage compilation pipeline**:

```
Lisp Expression → ByteCode IR → SSA v2 IR → Cranelift IR → Native Machine Code
```

### Key Features

- **🔍 Debuggable Pipeline**: Full visibility into every transformation stage
- **🚀 Zero-Copy Execution**: Direct Arrow array pointer passing
- **⚡ Unified Code Generation**: Both elementwise and reductions use consistent patterns
- **🎯 Direct Type Extraction**: Lambda expressions provide explicit return types
- **🔄 Expression Caching**: Compiled functions cached by expression hash

### Quick Start

```rust
use dio::{parse_expr, execute_generic_bytecode};
use dio::array_support::create_u64_array_from_vec;

fn main() {
    // Parse a Lambda expression with explicit types
    let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
    
    // Create input arrays
    let a = create_u64_array_from_vec(vec![1, 2, 3, 4, 5]).unwrap();
    let b = create_u64_array_from_vec(vec![10, 20, 30, 40, 50]).unwrap();
    
    // Execute using the new ByteCode pipeline
    let result = execute_generic_bytecode(&expr, &[a, b]).unwrap();
    println!("Result: {:?}", result); // [11, 22, 33, 44, 55]
}
```

### Lambda Expression Syntax

Dio v3 uses **typed Lambda expressions** for explicit type information:

```lisp
; Elementwise addition
(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))

; Elementwise operations with multiple operands
(lambda ([U64Array a] [U64Array b] [U64Array c] U64Array) (+ a b c))

; Reduction operations (unified as length-1 vectors)
(lambda ([U64Array a] U64) (sum a))
(lambda ([U64Array a] [U64Array b] U64) (sum (+ a b)))

; Mixed operations
(lambda ([U64Array x] [U64Array y] U64Array) (+ (* x 2) y))
```

**Supported Types:**
- `U64Array`, `I64Array` - Array types for unsigned/signed 64-bit integers
- `U64`, `I64` - Scalar types (for reduction results)

## 🔧 Development

### Debug Pipeline Transformations

Enable debug tracing to see transformations at each stage:

```bash
DIO_DEBUG_PIPELINE=1 cargo test test_generic_execute_binary_u64 -- --nocapture
```

Example output:
```
=== PIPELINE DEBUG ===
--- Input AST ---
(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))

--- ByteCode (C-like) ---
function(u64[] a, u64[] b, u64 length) -> u64[] {
  for (i = 0; i < length; i += 1) {
    output[i] = (a[i] + b[i]);
  }
  return;
}

--- SSA v2 ---
Entry block: BlockId(0)
Block BlockId(0):
  Parameters: SsaValue(0): U64, ...

--- Cranelift IR (SSA v2) ---
function u0:0(i64, i64, i64, i64) apple_aarch64 {
  ...
}
```

### Build Commands

```bash
# Build the project
cargo build

# Run all tests
cargo test

# Run specific test with debug output  
DIO_DEBUG_PIPELINE=1 cargo test test_name -- --nocapture

# Check compilation without building
cargo check
```

## 🏗️ Architecture

The v3 pipeline provides significant improvements:

### Before v3 (Deprecated):
```
Lisp → SSA v1 → Cranelift → Machine Code
```
- Complex type inference
- Separate code paths for elementwise/reductions
- Limited debuggability

### v3 Pipeline:
```
Lisp → ByteCode → SSA v2 → Cranelift → Machine Code
```
- Direct Lambda return type usage  
- Unified vector approach (reductions as length-1 arrays)
- Full pipeline visibility with debug tracing
- Consistent code generation patterns

## 📖 Documentation

- **[docs/dio3.md](docs/dio3.md)** - Complete project documentation
- **[docs/generic_execution.md](docs/generic_execution.md)** - Generic execution system details

## 🧪 Current Status

### ✅ Working Features
- **Elementwise Operations**: `+`, `-`, `*`, `/` with proper array indexing
- **Debug Pipeline**: Complete visibility into all transformation stages
- **Type System**: Robust typed Lambda expressions  
- **Zero-Copy Execution**: Direct Arrow array pointer passing
- **Unified Approach**: Both elementwise and reductions use consistent patterns

### 🚧 Known Issues
- **Reduction SSA Generation**: Variable scoping across loop boundaries needs phi nodes
- **Limited Type Support**: Currently only U64/I64 integers

### 📋 Future Enhancements
- **Additional Types**: F64 floating point support
- **More Operations**: Comparison operators, conditional expressions
- **Advanced Optimizations**: Loop unrolling, vectorization hints

## 📄 License

This project is part of the Dio expression evaluator research implementation.