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
    pub fn compile(&mut self, program: &SsaProgram) -> Result<*const u8, DioError> {
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // inputs ptr
        self.context.func.signature.params.push(AbiParam::new(types::I32)); // input_count
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // output ptr
        self.context.func.signature.params.push(AbiParam::new(types::I64)); // length

        let mut builder = FunctionBuilder::new(&mut self.context.func, &mut self.builder_context);
        let entry_block = builder.create_block();
        builder.append_block_params_for_function_params(entry_block);
        builder.switch_to_block(entry_block);

        let operation = program
            .blocks
            .values()
            .flat_map(|b| &b.instructions)
            .find_map(|inst| match inst {
                SsaInstruction::Add { .. } => Some("add"),
                SsaInstruction::Sub { .. } => Some("sub"),
                _ => None,
            })
            .ok_or_else(|| DioError::Compilation("No arithmetic operation found in SSA".to_string()))?;

        let length = builder.block_params(entry_block)[3];
        let outer_loop_header = builder.create_block();
        builder.append_block_param(outer_loop_header, types::I64);
        let zero = builder.ins().iconst(types::I64, 0);
        builder.ins().jump(outer_loop_header, &[zero]);

        builder.switch_to_block(outer_loop_header);
        let i = builder.block_params(outer_loop_header)[0];
        let condition = builder.ins().icmp(IntCC::UnsignedLessThan, i, length);
        let body_block = builder.create_block();
        let exit_block = builder.create_block();
        builder.ins().brif(condition, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        let inputs_ptr = builder.block_params(entry_block)[0];
        let input_count = builder.block_params(entry_block)[1];
        let output_ptr = builder.block_params(entry_block)[2];
        let eight = builder.ins().iconst(types::I64, 8);
        let element_offset = builder.ins().imul(i, eight);

        let first_array_ptr = builder.ins().load(types::I64, MemFlags::trusted(), inputs_ptr, 0);
        let first_element_addr = builder.ins().iadd(first_array_ptr, element_offset);
        let accumulator = builder.ins().load(types::I64, MemFlags::trusted(), first_element_addr, 0);

        let inner_loop_header = builder.create_block();
        builder.append_block_param(inner_loop_header, types::I64); // accumulator
        builder.append_block_param(inner_loop_header, types::I64); // inner loop var j
        let one = builder.ins().iconst(types::I64, 1);
        builder.ins().jump(inner_loop_header, &[accumulator, one]);
        
        builder.switch_to_block(inner_loop_header);
        let acc = builder.block_params(inner_loop_header)[0];
        let j = builder.block_params(inner_loop_header)[1];
        let j_32 = builder.ins().ireduce(types::I32, j);
        let inner_cond = builder.ins().icmp(IntCC::UnsignedLessThan, j_32, input_count);
        let inner_body = builder.create_block();
        let inner_exit = builder.create_block();
        builder.append_block_param(inner_exit, types::I64);
        builder.ins().brif(inner_cond, inner_body, &[], inner_exit, &[acc]);

        builder.switch_to_block(inner_body);
        let ptr_offset = builder.ins().imul(j, eight);
        let array_ptr_addr = builder.ins().iadd(inputs_ptr, ptr_offset);
        let array_ptr = builder.ins().load(types::I64, MemFlags::trusted(), array_ptr_addr, 0);
        let element_addr = builder.ins().iadd(array_ptr, element_offset);
        let element = builder.ins().load(types::I64, MemFlags::trusted(), element_addr, 0);

        let new_acc = match operation {
            "add" => builder.ins().iadd(acc, element),
            "sub" => builder.ins().isub(acc, element),
            _ => unreachable!(),
        };

        let next_j = builder.ins().iadd_imm(j, 1);
        builder.ins().jump(inner_loop_header, &[new_acc, next_j]);
        
        builder.switch_to_block(inner_exit);
        let final_acc = builder.block_params(inner_exit)[0];
        let output_addr = builder.ins().iadd(output_ptr, element_offset);
        builder.ins().store(MemFlags::trusted(), final_acc, output_addr, 0);

        let next_i = builder.ins().iadd_imm(i, 1);
        builder.ins().jump(outer_loop_header, &[next_i]);

        builder.switch_to_block(exit_block);
        builder.ins().return_(&[]);
        
        builder.seal_all_blocks();
        builder.finalize();

        if env::var("DIO_DEBUG_JIT").is_ok() {
            println!("--- Cranelift IR ---");
            println!("{}", self.context.func.display());
            println!("--------------------");
        }

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