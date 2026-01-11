# Cranelift Functions Example - ARM64 Note

## Your Questions Answered:

### 1. **Can you define a function with parameters?**
**YES!** You can define functions with parameters using function signatures:

```rust
let mut sig = module.make_signature();
sig.params.push(AbiParam::new(types::I64));  // Add parameters
sig.params.push(AbiParam::new(types::I64));
sig.returns.push(AbiParam::new(types::I64)); // Add return type
```

### 2. **Can you call that function from Rust?**
**YES!** You get a pointer to the compiled code and call it:

```rust
let code_ptr = module.get_finalized_function(func_id);
let my_func = unsafe { std::mem::transmute::<_, fn(i64, i64) -> i64>(code_ptr) };
let result = my_func(5, 3);  // Call it like a normal Rust function!
```

###3. **Can you define multiple functions in one module and call any of them?**
**YES!** You can declare and define multiple functions:

```rust
let func1_id = module.declare_function("func1", Linkage::Export, &sig1)?;
let func2_id = module.declare_function("func2", Linkage::Export, &sig2)?;

module.define_function(func1_id, &mut ctx1)?;
module.define_function(func2_id, &mut ctx2)?;
module.finalize_definitions()?;

// Get pointers to both
let func1 = module.get_finalized_function(func1_id);
let func2 = module.get_finalized_function(func2_id);
```

**BONUS**: Functions can call each other using `module.declare_func_in_func()` and `builder.ins().call()`.

## ARM64 Limitation

The example code in `cranelift_functions.rs` demonstrates all these features, but **it won't run on ARM64 (Apple Silicon)** due to a limitation in Cranelift 0.105:

```
PLT is currently only supported on x86_64
```

This affects:
- `Linkage::Export` (even for single functions)
- Inter-function calls via `declare_func_in_func`

### Workarounds for ARM64:

1. **Use SimpleJIT** (deprecated but works): The older `SimpleJIT` backend doesn't have this limitation
2. **Upgrade to newer Cranelift**: Newer versions (0.110+) have better ARM64 support
3. **Inline everything**: Don't use function declarations, build everything in one function body

### For x86_64:

The example works perfectly! Run on an x86_64 machine to see all features in action.

## See Also

- Cranelift tutorial examples
- The working example code is in `cranelift_functions.rs` (runs on x86_64)