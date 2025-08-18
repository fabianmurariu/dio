# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dio is a JIT compiled columnar expression evaluator that compiles arithmetic expressions into optimized machine code using Cranelift. It operates on Apache Arrow arrays for efficient columnar data processing.

## Build and Development Commands

### Basic Commands
- `cargo build` - Build the project
- `cargo check` - Check compilation without producing binaries
- `cargo test` - Run all tests
- `cargo test <test_name>` - Run specific test by name

### Development Workflow
- `cargo fmt` - Format code (when rustfmt is configured)
- `cargo clippy` - Run linter (when clippy is available)

## Architecture Overview

The project is designed around a three-phase compilation pipeline:

### Core Components

1. **Expression DSL** (`Expr` enum): Represents arithmetic expressions and reductions
   - Elementwise operations: `Add`, `Sub`, `Mul`, `Div` that operate on arrays element-by-element
   - Reduction operations: `Sum`, `Count` that reduce arrays to scalars
   - Column references and literals

2. **JIT Compilation Pipeline**:
   - **Phase 1**: Expression analysis to determine input columns and output types
   - **Phase 2**: Compilation strategy selection (elementwise vs reduction loops)  
   - **Phase 3**: Cranelift IR generation and JIT compilation

3. **Array Pointer Management**: Zero-copy integration with Arrow arrays
   - `ArrayDescriptor`: Wraps Arrow array pointers with metadata
   - `ExecutionContext`: Manages input arrays and output buffers
   - Direct pointer passing to compiled functions without copying

### Key Design Principles

- **Zero-copy execution**: Array data remains in Arrow format, passed as pointers to compiled functions
- **Expression caching**: Compiled functions are cached by expression hash to avoid recompilation
- **Two execution modes**: 
  - Elementwise: `output[i] = f(input1[i], input2[i], ...)` 
  - Reduction: `accumulator = reduce(input_arrays)`

## Implementation Status

This is an early-stage project. The current codebase contains:
- Basic Cargo.toml setup
- Empty lib.rs stub
- Comprehensive design document in `docs/dio1.md`

Planned dependencies (from design doc):
- `cranelift*` crates for JIT compilation
- `arrow` for columnar data structures  
- `thiserror` for error handling

## Testing Strategy

When implementing, follow the testing patterns outlined in the design document:
- Unit tests for individual expression evaluation
- Integration tests for complex nested expressions  
- Performance benchmarks using `criterion`
- Memory leak detection for unsafe pointer operations