use crate::error::DioError;
use crate::ssa::{SsaInstruction, SsaProgram};
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::env;

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
    /// Generates optimized single-loop IR: for i in 0..length: out[i] = input[0][i] op input[1][i] op ...
    pub fn compile(&mut self, program: &SsaProgram) -> Result<*const u8, DioError> {
        // Extract information from SSA before creating builder
        let operation = Self::extract_operation_from_ssa(program)?;
        let input_array_count = Self::count_input_arrays(program)?;
        
        self.setup_function_signature();
        
        let mut builder = FunctionBuilder::new(&mut self.context.func, &mut self.builder_context);
        
        // Create entry block inline
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        
        // Generate single loop IR inline
        let inputs_ptr = builder.block_params(entry_block)[0];
        let output_ptr = builder.block_params(entry_block)[2];
        let length = builder.block_params(entry_block)[3];
        
        // Create loop structure
        let loop_header = builder.create_block();
        builder.append_block_param(loop_header, types::I64); // loop index i
        let loop_body = builder.create_block();
        let exit_block = builder.create_block();
        
        // Initialize loop: jump to header with i = 0
        let zero = builder.ins().iconst(types::I64, 0);
        builder.ins().jump(loop_header, &[zero]);
        
        // Loop header: check condition i < length
        builder.switch_to_block(loop_header);
        let i = builder.block_params(loop_header)[0];
        let condition = builder.ins().icmp(IntCC::UnsignedLessThan, i, length);
        builder.ins().brif(condition, loop_body, &[], exit_block, &[]);
        
        // Loop body: compute out[i] = input[0][i] op input[1][i] op ...
        builder.switch_to_block(loop_body);
        let eight = builder.ins().iconst(types::I64, 8);
        let element_offset = builder.ins().imul(i, eight);
        
        // Load first array element as accumulator
        let first_array_ptr = builder.ins().load(types::I64, MemFlags::trusted(), inputs_ptr, 0);
        let first_element_addr = builder.ins().iadd(first_array_ptr, element_offset);
        let mut accumulator = builder.ins().load(types::I64, MemFlags::trusted(), first_element_addr, 0);
        
        // Apply operation with remaining arrays: acc = acc op input[j][i]
        for j in 1..input_array_count {
            let j_offset = builder.ins().iconst(types::I64, (j as i64) * 8);
            let array_ptr_addr = builder.ins().iadd(inputs_ptr, j_offset);
            let array_ptr = builder.ins().load(types::I64, MemFlags::trusted(), array_ptr_addr, 0);
            let element_addr = builder.ins().iadd(array_ptr, element_offset);
            let element = builder.ins().load(types::I64, MemFlags::trusted(), element_addr, 0);
            
            accumulator = match operation {
                "add" => builder.ins().iadd(accumulator, element),
                "sub" => builder.ins().isub(accumulator, element),
                _ => return Err(DioError::Compilation(format!("Unsupported operation: {}", operation))),
            };
        }
        
        // Store result: out[i] = accumulator
        let output_addr = builder.ins().iadd(output_ptr, element_offset);
        builder.ins().store(MemFlags::trusted(), accumulator, output_addr, 0);
        
        // Increment loop counter and continue
        let next_i = builder.ins().iadd_imm(i, 1);
        builder.ins().jump(loop_header, &[next_i]);
        
        // Exit block
        builder.switch_to_block(exit_block);
        builder.ins().return_(&[]);
        
        builder.seal_all_blocks();
        builder.finalize();
        
        self.debug_print_ir();
        
        self.finalize_and_get_function_ptr()
    }
    
    /// Set up the function signature for the variadic JIT function
    fn setup_function_signature(&mut self) {
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // inputs ptr
        self.context.func.signature.params.push(AbiParam::new(types::I32)); // input_count  
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // output ptr
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // length
    }
    
    /// Extract the arithmetic operation from SSA program
    fn extract_operation_from_ssa(program: &SsaProgram) -> Result<&'static str, DioError> {
        program
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .find_map(|inst| match inst {
                SsaInstruction::Add { .. } => Some("add"),
                SsaInstruction::Sub { .. } => Some("sub"),
                _ => None,
            })
            .ok_or_else(|| DioError::Compilation("No arithmetic operation found in SSA".to_string()))
    }
    
    /// Count the number of input arrays from LoadArrayParam instructions
    fn count_input_arrays(program: &SsaProgram) -> Result<u32, DioError> {
        let count = program
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .filter_map(|inst| match inst {
                SsaInstruction::LoadArrayParam { param_index, .. } => Some(*param_index),
                _ => None,
            })
            .max()
            .map(|max_idx| max_idx + 1)
            .unwrap_or(0);
        
        if count == 0 {
            return Err(DioError::Compilation("No input arrays found in SSA".to_string()));
        }
        
        Ok(count)
    }
    
    /// Print Cranelift IR for debugging if DIO_DEBUG_JIT is set
    fn debug_print_ir(&self) {
        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("--- Cranelift IR (Single Loop) ---");
            println!("{}", self.context.func.display());
            println!("----------------------------------");
        }
    }
    
    /// Finalize the function and return the executable pointer
    fn finalize_and_get_function_ptr(&mut self) -> Result<*const u8, DioError> {
        let func_id = self.module.declare_function("dio_jitted_func", Linkage::Export, &self.context.func.signature)
            .map_err(|e| DioError::Compilation(format!("Failed to declare function: {e}")))?;
        self.module.define_function(func_id, &mut self.context)
            .map_err(|e| DioError::Compilation(format!("Failed to define function: {e}")))?;
        self.module.clear_context(&mut self.context);
        self.module.finalize_definitions().map_err(|e| DioError::Compilation(format!("Failed to finalize definitions: {e}")))?;
        
        Ok(self.module.get_finalized_function(func_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{parse_expr, ssa::ast_to_ssa};

    #[test]
    fn test_cranelift_compilation_add() {
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (+ a b))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        let mut backend = CraneliftBackend::new().unwrap();
        backend.compile(&ssa_program).unwrap();
    }

    #[test]
    fn test_cranelift_compilation_sub() {
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] I64Array) (- a b))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        let mut backend = CraneliftBackend::new().unwrap();
        backend.compile(&ssa_program).unwrap();
    }

    #[test]
    fn test_cranelift_compilation_nary_add() {
        let expr = parse_expr("(lambda ([I64Array a] [I64Array b] [I64Array c] I64Array) (+ a b c))").unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        let mut backend = CraneliftBackend::new().unwrap();
        backend.compile(&ssa_program).unwrap();
    }
}