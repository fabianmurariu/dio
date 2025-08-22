use crate::error::DioError;
use crate::ssa::{BlockId, SsaInstruction, SsaProgram, SsaValue};
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
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // inputs ptr
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I32)); // input_count
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // output ptr
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // length
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

        // Create a mapping from SSA blocks to Cranelift blocks
        let mut ssa_to_cranelift_blocks: HashMap<BlockId, Block> = HashMap::new();

        // Create all Cranelift blocks first, but use the entry block we already have
        ssa_to_cranelift_blocks.insert(program.entry_block, entry_block);
        for block_id in program.blocks.keys() {
            if *block_id != program.entry_block {
                let cranelift_block = builder.create_block();
                ssa_to_cranelift_blocks.insert(*block_id, cranelift_block);
            }
        }

        // Pre-load all array pointers in the entry block (hoist invariant loads)
        if let Some(entry_ssa_block) = program.blocks.get(&program.entry_block) {
            for instruction in &entry_ssa_block.instructions {
                if let SsaInstruction::LoadArrayParam {
                    dest,
                    param_index,
                    data_type: _,
                } = instruction
                {
                    // Load array pointer from inputs[param_index] once, outside the loop
                    let param_offset = builder.ins().iconst(types::I64, (*param_index as i64) * 8);
                    let array_ptr_addr = builder.ins().iadd(inputs_ptr, param_offset);
                    let array_ptr =
                        builder
                            .ins()
                            .load(types::I64, MemFlags::trusted(), array_ptr_addr, 0);

                    // Store the array pointer for this SSA value
                    ssa_to_cranelift.insert(*dest, array_ptr);
                }
            }
        }

        // For each SSA block that needs loop parameters, add them
        for (block_id, ssa_block) in &program.blocks {
            let cranelift_block = ssa_to_cranelift_blocks[block_id];

            // Check if this block needs loop index parameter (has LessThan instruction)
            let needs_loop_param = ssa_block
                .instructions
                .iter()
                .any(|inst| matches!(inst, SsaInstruction::LessThan { .. }));

            if needs_loop_param {
                builder.append_block_param(cranelift_block, types::I64); // loop index
            }
        }

        // Process blocks directly without helper methods to avoid borrowing issues
        // Start with entry block (we're already in it)

        // Process entry block first
        let entry_ssa_block = &program.blocks[&program.entry_block];
        for instruction in &entry_ssa_block.instructions {
            match instruction {
                SsaInstruction::LoadArrayParam { .. } => {
                    // Already handled outside the loop - skip
                }
                SsaInstruction::LoadLengthParam {
                    dest,
                    param_index: _,
                } => {
                    // Map to function parameter
                    ssa_to_cranelift.insert(*dest, length);
                }
                SsaInstruction::LoadConstant { dest, value } => {
                    let constant_val = builder.ins().iconst(types::I64, *value);
                    ssa_to_cranelift.insert(*dest, constant_val);
                }
                SsaInstruction::ArrayAccess { dest, array, index } => {
                    // Get the array pointer and index, compute element address
                    if let (Some(&array_ptr), Some(&index_val)) =
                        (ssa_to_cranelift.get(array), ssa_to_cranelift.get(index))
                    {
                        let eight = builder.ins().iconst(types::I64, 8);
                        let element_offset = builder.ins().imul(index_val, eight);
                        let element_addr = builder.ins().iadd(array_ptr, element_offset);
                        let element =
                            builder
                                .ins()
                                .load(types::I64, MemFlags::trusted(), element_addr, 0);
                        ssa_to_cranelift.insert(*dest, element);
                    } else {
                        return Err(DioError::Compilation(format!(
                            "Array {:?} or index {:?} not found in SSA mapping",
                            array, index
                        )));
                    }
                }
                SsaInstruction::Jump { target } => {
                    let target_block = ssa_to_cranelift_blocks[target];
                    // Jump to loop header with initial value (0)
                    if let Some(&zero_val) = ssa_to_cranelift.get(&SsaValue(3)) {
                        // zero constant
                        builder.ins().jump(target_block, &[zero_val]);
                    } else {
                        let zero = builder.ins().iconst(types::I64, 0);
                        builder.ins().jump(target_block, &[zero]);
                    }
                }
                SsaInstruction::Return { .. } => {
                    builder.ins().return_(&[]);
                }
                _ => {
                    return Err(DioError::Compilation(format!(
                        "Unsupported SSA instruction in entry block: {:?}",
                        instruction
                    )));
                }
            }
        }

        // Process other blocks
        for (block_id, ssa_block) in &program.blocks {
            if *block_id == program.entry_block {
                continue; // Already processed
            }

            let cranelift_block = ssa_to_cranelift_blocks[block_id];
            builder.switch_to_block(cranelift_block);

            // Handle block parameters (e.g., loop index)
            let block_params = builder.block_params(cranelift_block);
            if !block_params.is_empty() && block_params.len() == 1 {
                // This is likely a loop header with index parameter
                // Map the loop index SSA value to the block parameter
                let loop_index_ssa = ssa_block.instructions.iter().find_map(|inst| match inst {
                    SsaInstruction::LessThan { lhs, .. } => Some(*lhs),
                    _ => None,
                });

                if let Some(loop_index) = loop_index_ssa {
                    ssa_to_cranelift.insert(loop_index, block_params[0]);
                }
            }

            // Process all instructions in this block
            for instruction in &ssa_block.instructions {
                match instruction {
                    SsaInstruction::LoadArrayParam { .. } => {
                        // Already handled outside the loop - skip
                    }
                    SsaInstruction::LoadLengthParam {
                        dest,
                        param_index: _,
                    } => {
                        // Map to function parameter
                        ssa_to_cranelift.insert(*dest, length);
                    }
                    SsaInstruction::LoadConstant { dest, value } => {
                        let constant_val = builder.ins().iconst(types::I64, *value);
                        ssa_to_cranelift.insert(*dest, constant_val);
                    }
                    SsaInstruction::ArrayAccess { dest, array, index } => {
                        // Get the array pointer and index, compute element address
                        if let (Some(&array_ptr), Some(&index_val)) =
                            (ssa_to_cranelift.get(array), ssa_to_cranelift.get(index))
                        {
                            let eight = builder.ins().iconst(types::I64, 8);
                            let element_offset = builder.ins().imul(index_val, eight);
                            let element_addr = builder.ins().iadd(array_ptr, element_offset);
                            let element = builder.ins().load(
                                types::I64,
                                MemFlags::trusted(),
                                element_addr,
                                0,
                            );
                            ssa_to_cranelift.insert(*dest, element);
                        } else {
                            return Err(DioError::Compilation(format!(
                                "Array {:?} or index {:?} not found in SSA mapping",
                                array, index
                            )));
                        }
                    }
                    SsaInstruction::Add { dest, lhs, rhs } => {
                        if let (Some(&lhs_val), Some(&rhs_val)) =
                            (ssa_to_cranelift.get(lhs), ssa_to_cranelift.get(rhs))
                        {
                            let result = builder.ins().iadd(lhs_val, rhs_val);
                            ssa_to_cranelift.insert(*dest, result);
                        } else {
                            return Err(DioError::Compilation(format!(
                                "Add operands {:?}, {:?} not found",
                                lhs, rhs
                            )));
                        }
                    }
                    SsaInstruction::Sub { dest, lhs, rhs } => {
                        if let (Some(&lhs_val), Some(&rhs_val)) =
                            (ssa_to_cranelift.get(lhs), ssa_to_cranelift.get(rhs))
                        {
                            let result = builder.ins().isub(lhs_val, rhs_val);
                            ssa_to_cranelift.insert(*dest, result);
                        } else {
                            return Err(DioError::Compilation(format!(
                                "Sub operands {:?}, {:?} not found",
                                lhs, rhs
                            )));
                        }
                    }
                    SsaInstruction::LessThan { dest, lhs, rhs } => {
                        if let (Some(&lhs_val), Some(&rhs_val)) =
                            (ssa_to_cranelift.get(lhs), ssa_to_cranelift.get(rhs))
                        {
                            let result =
                                builder
                                    .ins()
                                    .icmp(IntCC::UnsignedLessThan, lhs_val, rhs_val);
                            ssa_to_cranelift.insert(*dest, result);
                        } else {
                            return Err(DioError::Compilation(format!(
                                "LessThan operands {:?}, {:?} not found",
                                lhs, rhs
                            )));
                        }
                    }
                    SsaInstruction::StoreArrayElement {
                        array: _,
                        index,
                        value,
                    } => {
                        // Store the result to output[index]
                        if let (Some(&result_val), Some(&index_val)) =
                            (ssa_to_cranelift.get(value), ssa_to_cranelift.get(index))
                        {
                            let eight = builder.ins().iconst(types::I64, 8);
                            let element_offset = builder.ins().imul(index_val, eight);
                            let output_addr = builder.ins().iadd(output_ptr, element_offset);
                            builder
                                .ins()
                                .store(MemFlags::trusted(), result_val, output_addr, 0);
                        } else {
                            return Err(DioError::Compilation(format!(
                                "Store value {:?} or index {:?} not found",
                                value, index
                            )));
                        }
                    }
                    SsaInstruction::Jump { target } => {
                        let target_block = ssa_to_cranelift_blocks[target];
                        // This is likely the back-edge jump with incremented index
                        // Look for the incremented value in the current block
                        let incremented_value =
                            ssa_block.instructions.iter().find_map(|inst| match inst {
                                SsaInstruction::Add { dest, .. } => {
                                    // This might be the i + 1 instruction
                                    if let Some(&val) = ssa_to_cranelift.get(dest) {
                                        Some(val)
                                    } else {
                                        None
                                    }
                                }
                                _ => None,
                            });

                        if let Some(inc_val) = incremented_value {
                            builder.ins().jump(target_block, &[inc_val]);
                        } else {
                            builder.ins().jump(target_block, &[]);
                        }
                    }
                    SsaInstruction::Branch {
                        condition,
                        true_target,
                        false_target,
                    } => {
                        if let Some(&condition_val) = ssa_to_cranelift.get(condition) {
                            let true_block = ssa_to_cranelift_blocks[true_target];
                            let false_block = ssa_to_cranelift_blocks[false_target];
                            builder
                                .ins()
                                .brif(condition_val, true_block, &[], false_block, &[]);
                        } else {
                            return Err(DioError::Compilation(format!(
                                "Branch condition {:?} not found",
                                condition
                            )));
                        }
                    }
                    SsaInstruction::Return { .. } => {
                        builder.ins().return_(&[]);
                    }
                }
            }
        }

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
        let func_id = self
            .module
            .declare_function(
                "dio_jitted_func",
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
        let expr =
            parse_expr("(lambda ([I64Array a] [I64Array b] [I64Array c] I64Array) (+ a b c))")
                .unwrap();
        let ssa_program = ast_to_ssa(&expr).unwrap();
        let mut backend = CraneliftBackend::new().unwrap();
        backend.compile(&ssa_program).unwrap();
    }
}
