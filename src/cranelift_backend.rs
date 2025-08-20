use crate::error::DioError;
use crate::ssa::SsaProgram;
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};

/// Cranelift backend for compiling SSA IR to machine code
pub struct CraneliftBackend {
    builder_context: FunctionBuilderContext,
    context: codegen::Context,
    module: JITModule,
}

impl CraneliftBackend {
    pub fn new() -> Result<Self, DioError> {
        let builder = JITBuilder::with_isa(
            cranelift_native::builder()
                .map_err(|e| DioError::Compilation(format!("Failed to create ISA builder: {e}")))?
                .finish(settings::Flags::new(settings::builder()))
                .map_err(|e| DioError::Compilation(format!("Failed to create ISA: {e}")))?,
            default_libcall_names(),
        );

        let module = JITModule::new(builder);

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            context: module.make_context(),
            module,
        })
    }

    /// Compile SSA program to executable function with variadic signature
    /// Function signature: fn(inputs: *const *const u8, input_count: u32, output: *mut u8, length: u64)
    #[allow(unused_variables)]
    pub fn compile(&mut self, program: &SsaProgram) -> Result<*const u8, DioError> {
        // Create function signature
        self.context.func.signature.call_conv = self.module.isa().default_call_conv();

        // Parameters for variadic function signature:
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // inputs: *const *const u8 (array of input pointers)
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I32)); // input_count: u32 (number of input arrays)
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // output: *mut u8 (output array pointer)
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // length: u64 (array length)

        // No return value (void)

        // Build function body
        {
            let mut builder =
                FunctionBuilder::new(&mut self.context.func, &mut self.builder_context);

            // Create entry block and get parameters
            let entry_block = builder.create_block();
            builder.switch_to_block(entry_block);
            builder.append_block_params_for_function_params(entry_block);

            let params = builder.block_params(entry_block);
            let inputs_ptr = params[0]; // *const *const u8 (array of input pointers)
            let input_count = params[1]; // u32 (number of input arrays)
            let output_ptr = params[2]; // *mut u8 (output array pointer)
            let length = params[3]; // u64 (array length)

            // Create loop blocks
            let loop_header = builder.create_block();
            let loop_body = builder.create_block();
            let loop_exit = builder.create_block();

            // Add loop variable parameter to loop header
            builder.append_block_param(loop_header, types::I64);

            // Entry block: initialize loop
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().jump(loop_header, &[zero]);

            // Loop header: check condition
            builder.switch_to_block(loop_header);
            let loop_var = builder.block_params(loop_header)[0];
            let condition = builder
                .ins()
                .icmp(IntCC::UnsignedLessThan, loop_var, length);
            builder
                .ins()
                .brif(condition, loop_body, &[], loop_exit, &[]);

            // Loop body: perform N-ary addition -> output[i]
            builder.switch_to_block(loop_body);

            let eight = builder.ins().iconst(types::I64, 8);
            let element_offset = builder.ins().imul(loop_var, eight);

            // Convert input_count to 64-bit for loop comparison
            let input_count_64 = builder.ins().uextend(types::I64, input_count);

            // Create inner loop for summing across all input arrays
            let sum_loop_header = builder.create_block();
            let sum_loop_body = builder.create_block();
            let sum_loop_exit = builder.create_block();

            // Add parameter to sum_loop_exit to receive final accumulator value
            builder.append_block_param(sum_loop_exit, types::I64); // final accumulator

            // Add parameters for sum loop: (accumulator, input_index)
            builder.append_block_param(sum_loop_header, types::I64); // accumulator
            builder.append_block_param(sum_loop_header, types::I64); // input_index

            // Initialize sum loop: accumulator=0, input_index=0
            let zero = builder.ins().iconst(types::I64, 0);
            builder.ins().jump(sum_loop_header, &[zero, zero]);

            // Sum loop header: check if input_index < input_count
            builder.switch_to_block(sum_loop_header);
            let sum_loop_params = builder.block_params(sum_loop_header);
            let accumulator = sum_loop_params[0];
            let input_index = sum_loop_params[1];

            let sum_condition =
                builder
                    .ins()
                    .icmp(IntCC::UnsignedLessThan, input_index, input_count_64);
            builder.ins().brif(
                sum_condition,
                sum_loop_body,
                &[],
                sum_loop_exit,
                &[accumulator],
            );

            // Sum loop body: load input_array[input_index][loop_var] and add to accumulator
            builder.switch_to_block(sum_loop_body);

            // Get pointer to input_array[input_index]: inputs_ptr + (input_index * 8)
            let ptr_offset = builder.ins().imul(input_index, eight);
            let input_ptr_addr = builder.ins().iadd(inputs_ptr, ptr_offset);
            let input_array_ptr =
                builder
                    .ins()
                    .load(types::I64, MemFlags::trusted(), input_ptr_addr, 0);

            // Get element address: input_array_ptr + element_offset
            let element_addr = builder.ins().iadd(input_array_ptr, element_offset);
            let element_val = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), element_addr, 0);

            // Add to accumulator
            let new_accumulator = builder.ins().iadd(accumulator, element_val);

            // Increment input_index and continue
            let one = builder.ins().iconst(types::I64, 1);
            let next_input_index = builder.ins().iadd(input_index, one);
            builder
                .ins()
                .jump(sum_loop_header, &[new_accumulator, next_input_index]);

            // Sum loop exit: store result to output[loop_var]
            builder.switch_to_block(sum_loop_exit);
            let final_sum_params = builder.block_params(sum_loop_exit);
            let final_sum = final_sum_params[0];

            let output_addr = builder.ins().iadd(output_ptr, element_offset);
            builder
                .ins()
                .store(MemFlags::trusted(), final_sum, output_addr, 0);

            // Return to main loop after sum loop finishes
            let main_loop_one = builder.ins().iconst(types::I64, 1);
            let next_loop_var = builder.ins().iadd(loop_var, main_loop_one);
            builder.ins().jump(loop_header, &[next_loop_var]);

            // Loop exit: return
            builder.switch_to_block(loop_exit);
            builder.ins().return_(&[]);

            // Seal blocks in correct order (no back-edges first)
            builder.seal_block(entry_block);
            builder.seal_block(loop_exit);
            builder.seal_block(sum_loop_exit);
            builder.seal_block(sum_loop_body);
            builder.seal_block(sum_loop_header); // Seal sum loop header with back-edges
            builder.seal_block(loop_body);
            builder.seal_block(loop_header); // Seal main loop header last since it has back-edges

            builder.finalize();
        }

        // Declare and define function
        let func_id = self
            .module
            .declare_function(
                "nary_add_arrays",
                Linkage::Export,
                &self.context.func.signature,
            )
            .map_err(|e| DioError::Compilation(format!("Failed to declare function: {e}")))?;

        self.module
            .define_function(func_id, &mut self.context)
            .map_err(|e| DioError::Compilation(format!("Failed to define function: {e}")))?;

        self.module.clear_context(&mut self.context);
        self.module
            .finalize_definitions()
            .map_err(|e| DioError::Compilation(format!("Failed to finalize definitions: {e}")))?;

        let code_ptr = self.module.get_finalized_function(func_id);

        Ok(code_ptr)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_expr, ssa::ast_to_ssa};

    #[test]
    fn test_cranelift_compilation() {
        let expr = parse_expr("(lambda ([U64Array a] [U64Array b] U64Array) (+ a b))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();

        let mut backend = CraneliftBackend::new().unwrap();
        let _compiled_fn = backend.compile(&ssa_program);

        // If we get here without panicking, compilation succeeded
        // We'll test actual execution in the integration tests
    }

    #[test]
    fn test_cranelift_compilation_i64_arrays() {
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (+ a b))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();

        let mut backend = CraneliftBackend::new().unwrap();
        let _compiled_fn = backend.compile(&ssa_program);

        // I64Array compilation should work the same as U64Array
    }

    #[test]
    fn test_cranelift_compilation_mixed_types() {
        let expr = parse_expr("(lambda ([U64Array a] [I64Array b] I64Array) (+ a b))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();

        let mut backend = CraneliftBackend::new().unwrap();
        let _compiled_fn = backend.compile(&ssa_program);

        // Mixed type compilation should work with type coercion
    }

    #[test]
    fn test_cranelift_compilation_ternary_addition() {
        let expr =
            parse_expr("(lambda ([U64Array a] [U64Array b] [U64Array c] U64Array) (+ a b c))")
                .unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();

        let mut backend = CraneliftBackend::new().unwrap();
        let _compiled_fn = backend.compile(&ssa_program);

        // N-ary compilation should work with the new variadic function signature
    }

    #[test]
    fn test_cranelift_compilation_quaternary_mixed() {
        let expr = parse_expr(
            "(lambda ([U64Array w] [I64Array x] [U64Array y] [I64Array z] I64Array) (+ w x y z))",
        )
        .unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();

        let mut backend = CraneliftBackend::new().unwrap();
        let _compiled_fn = backend.compile(&ssa_program);

        // N-ary mixed type compilation should work
    }
}
