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
    fn setup_function_signature(&mut self, is_reduction: bool) {
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
        
        if is_reduction {
            // For reductions, we return a scalar value instead of writing to output buffer
            self.context
                .func
                .signature
                .returns
                .push(AbiParam::new(types::I64)); // scalar result
        } else {
            // For elementwise, we write to output buffer
            self.context
                .func
                .signature
                .params
                .push(AbiParam::new(types::I64)); // output ptr
        }
        
        self.context
            .func
            .signature
            .params
            .push(AbiParam::new(types::I64)); // length
    }

    /// Compile SSA program by interpreting SSA instructions directly
    /// Maps SSA instructions to Cranelift IR with single-loop optimization
    pub fn compile(&mut self, program: &SsaProgram) -> Result<*const u8, DioError> {
        // Detect if this is a reduction by checking for InitAccumulator instruction
        let is_reduction = program.blocks.values().any(|block| {
            block.instructions.iter().any(|inst| {
                matches!(inst, SsaInstruction::InitAccumulator { .. })
            })
        });
        
        self.setup_function_signature(is_reduction);

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
        let (output_ptr, length) = if is_reduction {
            // For reductions: inputs_ptr, input_count, length (no output_ptr)
            (None, builder.block_params(entry_block)[2])
        } else {
            // For elementwise: inputs_ptr, input_count, output_ptr, length
            (Some(builder.block_params(entry_block)[2]), builder.block_params(entry_block)[3])
        };

        // Create a mapping from SSA blocks to Cranelift blocks
        let mut ssa_to_cranelift_blocks: HashMap<BlockId, Block> = HashMap::new();

        // Create all Cranelift blocks first, but use the entry block we already have
        ssa_to_cranelift_blocks.insert(program.entry_block, entry_block);
        
        // Sort block IDs for deterministic iteration order
        let mut block_ids: Vec<_> = program.blocks.keys().cloned().collect();
        block_ids.sort_by_key(|id| id.0);
        
        for block_id in block_ids {
            if block_id != program.entry_block {
                let cranelift_block = builder.create_block();
                ssa_to_cranelift_blocks.insert(block_id, cranelift_block);
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

        // For each SSA block that needs loop parameters, add them (using sorted order)
        let mut block_ids: Vec<_> = program.blocks.keys().cloned().collect();
        block_ids.sort_by_key(|id| id.0);
        
        for block_id in &block_ids {
            let ssa_block = &program.blocks[block_id];
            let cranelift_block = ssa_to_cranelift_blocks[block_id];

            // Check if this block needs loop parameters
            let has_less_than = ssa_block
                .instructions
                .iter()
                .any(|inst| matches!(inst, SsaInstruction::LessThan { .. }));
            let has_accumulator = is_reduction && ssa_block
                .instructions
                .iter()
                .any(|inst| matches!(inst, SsaInstruction::LessThan { .. }));

            if has_less_than {
                builder.append_block_param(cranelift_block, types::I64); // loop index
                if has_accumulator {
                    builder.append_block_param(cranelift_block, types::I64); // accumulator
                }
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
                SsaInstruction::InitAccumulator { dest, initial_value, .. } => {
                    let constant_val = builder.ins().iconst(types::I64, *initial_value);
                    ssa_to_cranelift.insert(*dest, constant_val);
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
                    // Find the zero constant SSA value in the entry block
                    let zero_ssa_value = entry_ssa_block.instructions.iter().find_map(|inst| {
                        if let SsaInstruction::LoadConstant { dest, value } = inst {
                            if *value == 0 {
                                Some(*dest)
                            } else {
                                None
                            }
                        } else {
                            None
                        }
                    });
                    
                    let zero_val = if let Some(zero_ssa) = zero_ssa_value {
                        if let Some(&zero_val) = ssa_to_cranelift.get(&zero_ssa) {
                            zero_val
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        }
                    } else {
                        builder.ins().iconst(types::I64, 0)
                    };

                    if is_reduction {
                        // For reductions, pass both loop index (0) and initial accumulator (0)
                        // Find the initial accumulator value from InitAccumulator instruction
                        let init_acc_ssa = entry_ssa_block.instructions.iter().find_map(|inst| {
                            if let SsaInstruction::InitAccumulator { dest, .. } = inst {
                                Some(*dest)
                            } else {
                                None
                            }
                        });
                        
                        let init_acc = if let Some(acc_ssa) = init_acc_ssa {
                            if let Some(&acc_val) = ssa_to_cranelift.get(&acc_ssa) {
                                acc_val
                            } else {
                                builder.ins().iconst(types::I64, 0)
                            }
                        } else {
                            builder.ins().iconst(types::I64, 0)
                        };
                        builder.ins().jump(target_block, &[zero_val, init_acc]);
                    } else {
                        // For elementwise, just pass loop index (0)
                        builder.ins().jump(target_block, &[zero_val]);
                    }
                }
                SsaInstruction::UpdateAccumulator { .. } => {
                    return Err(DioError::Compilation(
                        "UpdateAccumulator should not be used in new reduction implementation".to_string()
                    ));
                }
                SsaInstruction::Return { value } => {
                    if let Some(ret_val) = value {
                        if let Some(&cranelift_val) = ssa_to_cranelift.get(ret_val) {
                            builder.ins().return_(&[cranelift_val]);
                        } else {
                            return Err(DioError::Compilation(format!(
                                "Return value {:?} not found in SSA mapping",
                                ret_val
                            )));
                        }
                    } else {
                        builder.ins().return_(&[]);
                    }
                }
                _ => {
                    return Err(DioError::Compilation(format!(
                        "Unsupported SSA instruction in entry block: {:?}",
                        instruction
                    )));
                }
            }
        }

        // Process other blocks (using sorted order for deterministic behavior)
        let mut block_ids: Vec<_> = program.blocks.keys().cloned().collect();
        block_ids.sort_by_key(|id| id.0);
        
        for block_id in block_ids {
            if block_id == program.entry_block {
                continue; // Already processed
            }

            let ssa_block = &program.blocks[&block_id];
            let cranelift_block = ssa_to_cranelift_blocks[&block_id];
            builder.switch_to_block(cranelift_block);

            // Handle block parameters (e.g., loop index and accumulator)
            let block_params = builder.block_params(cranelift_block);
            if !block_params.is_empty() {
                // Find the loop index SSA value (used in LessThan)
                let loop_index_ssa = ssa_block.instructions.iter().find_map(|inst| match inst {
                    SsaInstruction::LessThan { lhs, .. } => Some(*lhs),
                    _ => None,
                });

                if let Some(loop_index) = loop_index_ssa {
                    ssa_to_cranelift.insert(loop_index, block_params[0]);
                }

                // If this is a reduction with 2 parameters, find the accumulator SSA value
                if block_params.len() == 2 && is_reduction {
                    if std::env::var("DIO_DEBUG_JIT").is_ok() {
                        println!("DEBUG: Found block with 2 params for reduction, looking for accumulator...");
                    }
                    // Look for the accumulator update Add instruction in the loop body
                    // It's the Add instruction where one operand comes from an inner expression
                    // Find the accumulator update Add instruction
                    // This is the Add where rhs comes from inner expr and lhs is NOT used as lhs in another Add
                    let accumulator_ssa = program.blocks.values()
                        .flat_map(|block| &block.instructions)
                        .find_map(|inst| match inst {
                            SsaInstruction::Add { dest: _, lhs, rhs } => {
                                // Check if rhs comes from inner expression (Add/Sub) or direct ArrayAccess
                                let rhs_from_inner_expr = program.blocks.values()
                                    .flat_map(|b| &b.instructions)
                                    .any(|i| match i {
                                        SsaInstruction::Add { dest, .. } => dest == rhs,
                                        SsaInstruction::Sub { dest, .. } => dest == rhs,
                                        _ => false,
                                    });
                                
                                let rhs_from_array_access = program.blocks.values()
                                    .flat_map(|b| &b.instructions)
                                    .any(|i| match i {
                                        SsaInstruction::ArrayAccess { dest, .. } => dest == rhs,
                                        _ => false,
                                    });
                                
                                // Make sure this is not the loop increment (lhs should not be loop index)
                                let lhs_is_loop_index = program.blocks.values()
                                    .flat_map(|b| &b.instructions)
                                    .any(|i| match i {
                                        SsaInstruction::LessThan { lhs: loop_idx, .. } => loop_idx == lhs,
                                        _ => false,
                                    });
                                
                                // Check if lhs is defined anywhere (accumulator should not be defined, it's a block param)
                                let lhs_is_defined = program.blocks.values()
                                    .flat_map(|b| &b.instructions)
                                    .any(|i| match i {
                                        SsaInstruction::LoadArrayParam { dest, .. } => dest == lhs,
                                        SsaInstruction::LoadLengthParam { dest, .. } => dest == lhs,
                                        SsaInstruction::LoadConstant { dest, .. } => dest == lhs,
                                        SsaInstruction::ArrayAccess { dest, .. } => dest == lhs,
                                        SsaInstruction::Add { dest, .. } => dest == lhs,
                                        SsaInstruction::Sub { dest, .. } => dest == lhs,
                                        SsaInstruction::InitAccumulator { dest, .. } => dest == lhs,
                                        _ => false,
                                    });
                                
                                // The accumulator update has undefined lhs (block param) and defined rhs (inner expr result)
                                if (rhs_from_inner_expr || rhs_from_array_access) && !lhs_is_loop_index && !lhs_is_defined {
                                    if std::env::var("DIO_DEBUG_JIT").is_ok() {
                                        println!("DEBUG: Found accumulator update: lhs={:?}, rhs={:?}", lhs, rhs);
                                    }
                                    Some(*lhs)
                                } else {
                                    None
                                }
                            }
                            _ => None,
                        });

                    if let Some(accumulator) = accumulator_ssa {
                        if std::env::var("DIO_DEBUG_JIT").is_ok() {
                            println!("DEBUG: Mapping accumulator {:?} to block_params[1]", accumulator);
                        }
                        ssa_to_cranelift.insert(accumulator, block_params[1]);
                    } else if std::env::var("DIO_DEBUG_JIT").is_ok() {
                        println!("DEBUG: Could not find accumulator SSA value");
                    }
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
                    SsaInstruction::InitAccumulator { dest, initial_value, .. } => {
                        let constant_val = builder.ins().iconst(types::I64, *initial_value);
                        ssa_to_cranelift.insert(*dest, constant_val);
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
                        // Store the result to output[index] (only for elementwise operations)
                        if let Some(out_ptr) = output_ptr {
                            if let (Some(&result_val), Some(&index_val)) =
                                (ssa_to_cranelift.get(value), ssa_to_cranelift.get(index))
                            {
                                let eight = builder.ins().iconst(types::I64, 8);
                                let element_offset = builder.ins().imul(index_val, eight);
                                let output_addr = builder.ins().iadd(out_ptr, element_offset);
                                builder
                                    .ins()
                                    .store(MemFlags::trusted(), result_val, output_addr, 0);
                            } else {
                                return Err(DioError::Compilation(format!(
                                    "Store value {:?} or index {:?} not found",
                                    value, index
                                )));
                            }
                        } else {
                            return Err(DioError::Compilation(
                                "StoreArrayElement not supported in reduction operations".to_string()
                            ));
                        }
                    }
                    SsaInstruction::Jump { target } => {
                        let target_block = ssa_to_cranelift_blocks[target];
                        
                        if is_reduction {
                            // For reductions, we need both incremented index and updated accumulator
                            // Find the Add instructions (there should be 2: accumulator update and index increment)
                            let mut add_values = Vec::new();
                            for inst in &ssa_block.instructions {
                                if let SsaInstruction::Add { dest, .. } = inst {
                                    if let Some(&val) = ssa_to_cranelift.get(dest) {
                                        add_values.push(val);
                                    }
                                }
                            }
                            
                            if add_values.len() >= 3 {
                                // For complex reduction with multiple Add instructions
                                // The last Add is always index increment, second-to-last is accumulator update
                                let incremented_index = add_values[add_values.len() - 1];   // last: index increment
                                let updated_accumulator = add_values[add_values.len() - 2]; // second-to-last: accumulator update
                                builder.ins().jump(target_block, &[incremented_index, updated_accumulator]);
                            } else if add_values.len() >= 2 {
                                // For simple reduction: [accumulator_update, index_increment]
                                let updated_accumulator = add_values[0]; // accumulator update
                                let incremented_index = add_values[1];   // index increment
                                builder.ins().jump(target_block, &[incremented_index, updated_accumulator]);
                            } else {
                                builder.ins().jump(target_block, &[]);
                            }
                        } else {
                            // For elementwise, find the loop increment
                            let incremented_value = ssa_block.instructions.iter().rev().find_map(|inst| match inst {
                                SsaInstruction::Add { dest, .. } => {
                                    // The last Add instruction should be the loop increment
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
                    SsaInstruction::UpdateAccumulator { .. } => {
                        return Err(DioError::Compilation(
                            "UpdateAccumulator should not be used in new reduction implementation".to_string()
                        ));
                    }
                    SsaInstruction::Return { value } => {
                        if let Some(ret_val) = value {
                            if let Some(&cranelift_val) = ssa_to_cranelift.get(ret_val) {
                                builder.ins().return_(&[cranelift_val]);
                            } else {
                                return Err(DioError::Compilation(format!(
                                    "Return value {:?} not found in SSA mapping",
                                    ret_val
                                )));
                            }
                        } else {
                            builder.ins().return_(&[]);
                        }
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
