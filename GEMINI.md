# Project: Dio Expression Evaluator

## Project Overview

This project, "Dio," is a high-performance, JIT-compiled columnar expression evaluator written in Rust. It is designed to work with Apache Arrow arrays. The core functionality involves parsing Lisp-like arithmetic expressions, converting them into an Abstract Syntax Tree (AST), then to a Static Single Assignment (SSA) intermediate representation, and finally JIT-compiling it to native machine code using the Cranelift backend.

The primary goal of this project is to provide a fast and efficient way to perform calculations on large datasets stored in a columnar format.

**Key Technologies:**

*   **Rust:** The project is entirely written in Rust, leveraging its performance and safety features.
*   **Apache Arrow:** The evaluator is designed to operate on Apache Arrow arrays, a standard for in-memory columnar data.
*   **`nom`:** The parser is built using the `nom` parser combinator library, which allows for efficient, zero-copy parsing of the expression syntax.
*   **Cranelift:** The JIT compilation is handled by the Cranelift code generation backend, which compiles the intermediate representation into machine code at runtime.
*   **`proptest`:** The project uses property-based testing to ensure the robustness and correctness of the parser.

**Architecture:**

The compilation pipeline follows these steps:

1.  **Parsing:** The input Lisp-like expression string is parsed by a `nom`-based parser into an Abstract Syntax Tree (AST).
2.  **AST Representation:** The `Expr` enum in `src/ast.rs` represents the structure of the parsed expression.
3.  **SSA Conversion:** The AST is converted into a Static Single Assignment (SSA) form, which is a lower-level intermediate representation suitable for optimization and code generation.
4.  **JIT Compilation:** The SSA representation is then passed to the Cranelift backend, which JIT-compiles it into native machine code.

## Building and Running

**Build the project:**

```bash
cargo build
```

**Run the test suite:**

```bash
cargo test
```

**Run the basic parser example:**

```bash
cargo run --example basic_parser
```

## Development Conventions

*   **Code Style:** The code follows standard Rust conventions and is formatted with `rustfmt`.
*   **Error Handling:** The project uses the `thiserror` library for ergonomic error handling, with a custom `DioError` enum.
*   **Testing:** The project has a comprehensive test suite, including unit tests for each module and property-based tests for the parser.
*   **Modularity:** The codebase is well-structured, with clear separation of concerns between the parser, AST, SSA, and backend modules.
