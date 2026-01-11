//! Demonstration of Cranelift function capabilities:
//! 1. Defining functions with parameters
//! 2. Calling compiled functions from Rust
//! 3. Multiple functions in one module
//! 4. Functions calling each other

use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{Linkage, Module};

fn main() {
    println!("=== Cranelift Function Capabilities ===\n");

    // Example 1: Single function with parameters
    example1_single_function_with_params();

    println!();

    // Example 2: Multiple functions in one module
    example2_multiple_functions();

    println!();

    // Example 3: Functions calling each other
    // Note: Inter-function calls require PLT support, which is not available on ARM64
    // #[cfg(target_arch = "x86_64")]
    example3_function_calls();

    #[cfg(not(target_arch = "x86_64"))]
    println!("Example 3: Functions calling each other (skipped on ARM64 - PLT not supported)");
}

/// Example 1: Define and call a single function with parameters
fn example1_single_function_with_params() {
    println!("Example 1: Single function with parameters");

    let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
    let mut module = JITModule::new(builder);

    // Define function signature: (i64, i64) -> i64
    let mut sig = module.make_signature();
    sig.params.push(AbiParam::new(types::I64));
    sig.params.push(AbiParam::new(types::I64));
    sig.returns.push(AbiParam::new(types::I64));

    // Declare the function
    let func_id = module
        .declare_function("add_multiply", Linkage::Export, &sig)
        .unwrap();

    // Create function builder context
    let mut ctx = module.make_context();
    ctx.func.signature = sig;

    // Build the function body: (a + b) * 2
    {
        let mut builder_context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        // Get parameters
        let a = builder.block_params(entry_block)[0];
        let b = builder.block_params(entry_block)[1];

        // Compute (a + b) * 2
        let sum = builder.ins().iadd(a, b);
        let two = builder.ins().iconst(types::I64, 2);
        let result = builder.ins().imul(sum, two);

        builder.ins().return_(&[result]);
        builder.finalize();
    }

    // Compile the function
    module.define_function(func_id, &mut ctx).unwrap();
    module.clear_context(&mut ctx);
    module.finalize_definitions().unwrap();

    // Get a pointer to the compiled function
    let code_ptr = module.get_finalized_function(func_id);

    // Cast to Rust function pointer and call it!
    let add_multiply = unsafe { std::mem::transmute::<_, fn(i64, i64) -> i64>(code_ptr) };

    let result = add_multiply(5, 3);
    println!("  add_multiply(5, 3) = (5 + 3) * 2 = {}", result);
    assert_eq!(result, 16);
    println!("  ✓ Successfully called compiled function from Rust!");
}

/// Example 2: Multiple independent functions in one module
fn example2_multiple_functions() {
    println!("Example 2: Multiple functions in one module");

    let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
    let mut module = JITModule::new(builder);

    // Function 1: square(x) = x * x
    let square_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = module
            .declare_function("square", Linkage::Export, &sig)
            .unwrap();

        let mut ctx = module.make_context();
        ctx.func.signature = sig;

        {
            let mut builder_context = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let x = builder.block_params(entry_block)[0];
            let result = builder.ins().imul(x, x);
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        module.define_function(func_id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);

        func_id
    };

    // Function 2: double(x) = x * 2
    let double_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));

        let func_id = module
            .declare_function("double", Linkage::Export, &sig)
            .unwrap();

        let mut ctx = module.make_context();
        ctx.func.signature = sig;

        {
            let mut builder_context = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let x = builder.block_params(entry_block)[0];
            let two = builder.ins().iconst(types::I64, 2);
            let result = builder.ins().imul(x, two);
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        module.define_function(func_id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);

        func_id
    };

    // Function 3: is_positive(x) = x > 0
    let is_positive_id = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I8)); // Boolean as i8

        let func_id = module
            .declare_function("is_positive", Linkage::Export, &sig)
            .unwrap();

        let mut ctx = module.make_context();
        ctx.func.signature = sig;

        {
            let mut builder_context = FunctionBuilderContext::new();
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

            let entry_block = builder.create_block();
            builder.append_block_params_for_function_params(entry_block);
            builder.switch_to_block(entry_block);
            builder.seal_block(entry_block);

            let x = builder.block_params(entry_block)[0];
            let zero = builder.ins().iconst(types::I64, 0);
            let result = builder.ins().icmp(IntCC::SignedGreaterThan, x, zero);
            builder.ins().return_(&[result]);
            builder.finalize();
        }

        module.define_function(func_id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);

        func_id
    };

    module.finalize_definitions().unwrap();

    // Get pointers to all three functions
    let square = unsafe {
        std::mem::transmute::<_, fn(i64) -> i64>(module.get_finalized_function(square_id))
    };
    let double = unsafe {
        std::mem::transmute::<_, fn(i64) -> i64>(module.get_finalized_function(double_id))
    };
    let is_positive = unsafe {
        std::mem::transmute::<_, fn(i64) -> bool>(module.get_finalized_function(is_positive_id))
    };

    // Call all three functions!
    println!("  square(7) = {}", square(7));
    println!("  double(7) = {}", double(7));
    println!("  is_positive(7) = {}", is_positive(7));
    println!("  is_positive(-3) = {}", is_positive(-3));

    assert_eq!(square(7), 49);
    assert_eq!(double(7), 14);
    assert_eq!(is_positive(7), true);
    assert_eq!(is_positive(-3), false);

    println!("  ✓ All three functions work independently!");
}

/// Example 3: Functions calling each other
fn example3_function_calls() {
    println!("Example 3: Functions calling each other");

    let builder = JITBuilder::new(cranelift_module::default_libcall_names()).unwrap();
    let mut module = JITModule::new(builder);

    // Helper function: add(a, b) = a + b
    let add_sig = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        sig
    };

    let add_id = module
        .declare_function("add", Linkage::Export, &add_sig)
        .unwrap();

    // Define add function
    {
        let mut ctx = module.make_context();
        ctx.func.signature = add_sig.clone();

        let mut builder_context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let a = builder.block_params(entry_block)[0];
        let b = builder.block_params(entry_block)[1];
        let result = builder.ins().iadd(a, b);
        builder.ins().return_(&[result]);
        builder.finalize();

        module.define_function(add_id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);
    }

    // Main function: sum_of_squares(a, b) = add(a*a, b*b)
    let sum_of_squares_sig = {
        let mut sig = module.make_signature();
        sig.params.push(AbiParam::new(types::I64));
        sig.params.push(AbiParam::new(types::I64));
        sig.returns.push(AbiParam::new(types::I64));
        sig
    };

    let sum_of_squares_id = module
        .declare_function("sum_of_squares", Linkage::Export, &sum_of_squares_sig)
        .unwrap();

    // Define sum_of_squares function (calls add)
    {
        let mut ctx = module.make_context();
        ctx.func.signature = sum_of_squares_sig;

        let mut builder_context = FunctionBuilderContext::new();
        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut builder_context);

        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        builder.seal_block(entry_block);

        let a = builder.block_params(entry_block)[0];
        let b = builder.block_params(entry_block)[1];

        // Compute a*a and b*b
        let a_squared = builder.ins().imul(a, a);
        let b_squared = builder.ins().imul(b, b);

        // Call add(a_squared, b_squared)
        let add_func_ref = module.declare_func_in_func(add_id, &mut builder.func);
        let call_inst = builder.ins().call(add_func_ref, &[a_squared, b_squared]);
        let result = builder.inst_results(call_inst)[0];

        builder.ins().return_(&[result]);
        builder.finalize();

        module.define_function(sum_of_squares_id, &mut ctx).unwrap();
        module.clear_context(&mut ctx);
    }

    module.finalize_definitions().unwrap();

    // Get pointer and call the main function
    let sum_of_squares = unsafe {
        std::mem::transmute::<_, fn(i64, i64) -> i64>(
            module.get_finalized_function(sum_of_squares_id),
        )
    };

    let result = sum_of_squares(3, 4);
    println!("  sum_of_squares(3, 4) = 3² + 4² = {}", result);
    assert_eq!(result, 25);

    println!("  ✓ Functions can call each other within the same module!");
}
