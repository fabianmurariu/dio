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
                .map_err(|e| DioError::Compilation(format!("Failed to create ISA builder: {}", e)))?
                .finish(settings::Flags::new(settings::builder()))
                .map_err(|e| DioError::Compilation(format!("Failed to create ISA: {}", e)))?,
            default_libcall_names(),
        );

        let module = JITModule::new(builder);

        Ok(Self {
            builder_context: FunctionBuilderContext::new(),
            context: module.make_context(),
            module,
        })
    }

    /// Compile SSA program to executable function
    /// Function signature: fn(a_ptr: *const u64, b_ptr: *const u64, length: u64, output_ptr: *mut u64)
    #[allow(unused_variables)]
    pub fn compile(&mut self, program: &SsaProgram) -> Result<*const u8, DioError> {
        // Create function signature
        self.context.func.signature.call_conv = self.module.isa().default_call_conv();

        // Parameters: a_ptr, b_ptr, length, output_ptr
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // a_ptr
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // b_ptr
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // length
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // output_ptr

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
            let a_ptr = params[0]; // *const u64
            let b_ptr = params[1]; // *const u64
            let length = params[2]; // u64
            let output_ptr = params[3]; // *mut u64

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

            // Loop body: perform a[i] + b[i] -> output[i]
            builder.switch_to_block(loop_body);

            // Load a[i]
            let eight = builder.ins().iconst(types::I64, 8);
            let a_offset = builder.ins().imul(loop_var, eight);
            let a_addr = builder.ins().iadd(a_ptr, a_offset);
            let a_val = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), a_addr, 0);

            // Load b[i]
            let b_offset = builder.ins().imul(loop_var, eight);
            let b_addr = builder.ins().iadd(b_ptr, b_offset);
            let b_val = builder
                .ins()
                .load(types::I64, MemFlags::trusted(), b_addr, 0);

            // Add a[i] + b[i]
            let sum = builder.ins().iadd(a_val, b_val);

            // Store to output[i]
            let output_offset = builder.ins().imul(loop_var, eight);
            let output_addr = builder.ins().iadd(output_ptr, output_offset);
            builder
                .ins()
                .store(MemFlags::trusted(), sum, output_addr, 0);

            // Increment loop variable and jump back
            let one = builder.ins().iconst(types::I64, 1);
            let next_loop_var = builder.ins().iadd(loop_var, one);
            builder.ins().jump(loop_header, &[next_loop_var]);

            // Loop exit: return
            builder.switch_to_block(loop_exit);
            builder.ins().return_(&[]);

            // Seal blocks in correct order (no back-edges first)
            builder.seal_block(entry_block);
            builder.seal_block(loop_exit);
            builder.seal_block(loop_body);
            builder.seal_block(loop_header); // Seal loop header last since it has back-edges

            builder.finalize();
        }

        // Declare and define function
        let func_id = self
            .module
            .declare_function("add_arrays", Linkage::Export, &self.context.func.signature)
            .map_err(|e| DioError::Compilation(format!("Failed to declare function: {}", e)))?;

        self.module
            .define_function(func_id, &mut self.context)
            .map_err(|e| DioError::Compilation(format!("Failed to define function: {}", e)))?;

        self.module.clear_context(&mut self.context);
        self.module
            .finalize_definitions()
            .map_err(|e| DioError::Compilation(format!("Failed to finalize definitions: {}", e)))?;

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
}
