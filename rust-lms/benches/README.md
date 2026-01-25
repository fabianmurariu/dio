# JIT Comparison Benchmarks

This benchmark compares **rust-lms** against native Rust and other JIT/interpreter systems.

## Task

Filter + aggregation pipeline:
```rust
for value in data:
    if value > threshold:
        count += 1
        sum += value
        min = min(min, value)
        max = max(max, value)
avg = sum / count
```

## Running

```bash
# Basic benchmark (rust-lms vs native Rust)
cargo bench --bench jit_comparison_simple

# Full benchmark with optional systems
cargo bench --bench jit_comparison                           # native + rust-lms only
cargo bench --bench jit_comparison --features bench-rhai     # + Rhai (interpreted)
cargo bench --bench jit_comparison --features bench-wasm     # + WASM (via wasmtime)
cargo bench --bench jit_comparison --features bench-luajit   # + LuaJIT (requires system install)
cargo bench --bench jit_comparison --features "bench-rhai,bench-wasm"  # multiple features

# View HTML report
open target/criterion/report/index.html
```

## Systems Compared

- **Native Rust** - hand-written loop (baseline)
- **rust-lms** - JIT-compiled via Cranelift
- **WASM** (optional) - WebAssembly via wasmtime (also uses Cranelift)
- **Rhai** (optional) - Interpreted scripting language
- **LuaJIT** (optional) - Lua with JIT (requires system LuaJIT installation)

Data sizes: 1K, 10K, 100K elements

## Results

| Data Size | Native Rust | rust-lms Cold | rust-lms Warm | Warm vs Native |
|-----------|-------------|---------------|---------------|----------------|
| 1K        | 782 ns      | 92.6 µs       | 1.25 µs       | 1.6x slower    |
| 10K       | 9.36 µs     | 135.6 µs      | 15.2 µs       | 1.6x slower    |
| 100K      | 234 µs      | 538.5 µs      | 347 µs        | 1.5x slower    |

Key observations:
- **Cold start** includes ~90µs JIT compilation overhead
- **Warm execution** is ~1.5-1.6x slower than native Rust
- This is reasonable for Cranelift JIT vs LLVM-optimized native code
- The relative gap narrows as data size increases

## API Notes

For mutable reference parameters (`&mut T`), use:
- `store_ref(ptr, value)` to write to `SRefMut<T>`
- `load_ref_mut(ptr)` to read from `SRefMut<T>`
- Create loop variables inside functions with `ctx.let_var(init)`, not outside

For raw pointer parameters (needed for criterion benchmark lifetime issues):
- Use `SPtr<T>` for `*const T` and `SMutPtr<T>` for `*mut T`
- Use `store(ptr, value)` and `load_mut(ptr)` for raw pointers
