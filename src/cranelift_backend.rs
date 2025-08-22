use crate::error::DioError;
use crate::ssa::{SsaInstruction, SsaProgram, SsaValue};
use cranelift::prelude::*;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{default_libcall_names, Linkage, Module};
use std::{collections::HashMap, env};

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

    
    /// Set up the function signature for the variadic JIT function
    fn setup_function_signature(&mut self) {
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // inputs ptr
        self.context.func.signature.params.push(AbiParam::new(types::I32)); // input_count  
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // output ptr
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // length
    }
    
    
    /// Compile SSA program by interpreting SSA instructions directly
    /// Maps SSA instructions to Cranelift IR with single-loop optimization
    pub fn compile(&mut self, program: &SsaProgram) -> Result<*const u8, DioError> {
        self.setup_function_signature();
        
        let mut builder = FunctionBuilder::new(&mut self.context.func, &mut self.builder_context);
        
        // Create entry block  
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);
        
        // Track SSA value mappings to Cranelift values
        let mut ssa_to_cranelift: HashMap<SsaValue, Value> = HashMap::new();
        
        // Map function parameters
        let inputs_ptr = builder.block_params(entry_block)[0];
        let _input_count = builder.block_params(entry_block)[1]; 
        let output_ptr = builder.block_params(entry_block)[2];
        let length = builder.block_params(entry_block)[3];
        
        // Since SSA doesn't have explicit loops, we need to infer the loop structure
        // The pattern is: process each instruction for each array element
        
        // Create the main loop structure
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
        
        // Loop body: interpret SSA instructions
        builder.switch_to_block(loop_body);
        let eight = builder.ins().iconst(types::I64, 8);
        let element_offset = builder.ins().imul(i, eight);
        
        // Process the entry block instructions
        if let Some(entry_ssa_block) = program.blocks.get(&program.entry_block) {
            for instruction in &entry_ssa_block.instructions {
                match instruction {
                    SsaInstruction::LoadArrayParam { dest, param_index, data_type: _ } => {
                        // Load array pointer from inputs[param_index]
                        let param_offset = builder.ins().iconst(types::I64, (*param_index as i64) * 8);
                        let array_ptr_addr = builder.ins().iadd(inputs_ptr, param_offset);
                        let array_ptr = builder.ins().load(types::I64, MemFlags::trusted(), array_ptr_addr, 0);
                        
                        // Store the array pointer for this SSA value
                        ssa_to_cranelift.insert(*dest, array_ptr);
                    },
                    SsaInstruction::ArrayAccess { dest, array, index: _ } => {
                        // Get the array pointer and compute element address
                        if let Some(&array_ptr) = ssa_to_cranelift.get(array) {
                            let element_addr = builder.ins().iadd(array_ptr, element_offset);
                            let element = builder.ins().load(types::I64, MemFlags::trusted(), element_addr, 0);
                            ssa_to_cranelift.insert(*dest, element);
                        } else {
                            return Err(DioError::Compilation(format!("Array {:?} not found in SSA mapping", array)));
                        }
                    },
                    SsaInstruction::Add { dest, lhs, rhs } => {
                        if let (Some(&lhs_val), Some(&rhs_val)) = (ssa_to_cranelift.get(lhs), ssa_to_cranelift.get(rhs)) {
                            let result = builder.ins().iadd(lhs_val, rhs_val);
                            ssa_to_cranelift.insert(*dest, result);
                        } else {
                            return Err(DioError::Compilation(format!("Add operands {:?}, {:?} not found", lhs, rhs)));
                        }
                    },
                    SsaInstruction::Sub { dest, lhs, rhs } => {
                        if let (Some(&lhs_val), Some(&rhs_val)) = (ssa_to_cranelift.get(lhs), ssa_to_cranelift.get(rhs)) {
                            let result = builder.ins().isub(lhs_val, rhs_val);
                            ssa_to_cranelift.insert(*dest, result);
                        } else {
                            return Err(DioError::Compilation(format!("Sub operands {:?}, {:?} not found", lhs, rhs)));
                        }
                    },
                    SsaInstruction::StoreArrayElement { array: _, index: _, value } => {
                        // Store the final result to output[i]
                        if let Some(&result_val) = ssa_to_cranelift.get(value) {
                            let output_addr = builder.ins().iadd(output_ptr, element_offset);
                            builder.ins().store(MemFlags::trusted(), result_val, output_addr, 0);
                        } else {
                            return Err(DioError::Compilation(format!("Store value {:?} not found", value)));
                        }
                    },
                    SsaInstruction::Return { .. } => {
                        // Handled outside the loop
                    },
                    _ => {
                        return Err(DioError::Compilation(format!("Unsupported SSA instruction: {:?}", instruction)));
                    }
                }
            }
        }
        
        // Increment loop counter and continue
        let next_i = builder.ins().iadd_imm(i, 1);
        builder.ins().jump(loop_header, &[next_i]);
        
        // Exit block
        builder.switch_to_block(exit_block);
        builder.ins().return_(&[]);
        
        builder.seal_all_blocks();
        builder.finalize();
        
        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("--- Cranelift IR (SSA-driven) ---");
            println!("{}", self.context.func.display());
            println!("---------------------------------");
        }
        
        self.finalize_and_get_function_ptr()
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