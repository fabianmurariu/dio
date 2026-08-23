//! Cranelift backend (docs/llvm.md). The default code generator — it JITs the neutral
//! [`Backend`](crate::staged::Backend) op-stream to native machine code via Cranelift, and
//! is the companion to the [`llvm`](crate::llvm) module.
//!
//! This module holds [`CraneliftBackend`] (the `impl Backend`) and the memcpy helper the
//! `compile()` driver in `func.rs` also uses. The opaque-handle ↔ Cranelift-entity
//! conversions stay in `staged` (they touch the handles private field); this module only
//! calls them. The module/JIT *lifecycle* (building a `JITModule`, finalizing, `lookup`)
//! still lives in the `compile()` driver in `func.rs` (docs/llvm.md: the `Module`/
//! `Executable` boundary).

use std::collections::HashMap;

use cranelift_codegen::ir::{
    types, AbiParam, BlockArg, FuncRef, InstBuilder, MemFlags, Signature, StackSlotData,
    StackSlotKind, Value,
};
use cranelift_codegen::isa::TargetFrontendConfig;
use cranelift_frontend::FunctionBuilder;
use cranelift_jit::JITModule;
use cranelift_module::{FuncId, Module};

use crate::staged::{
    Backend, BlockHandle, FuncRefId, SigRefId, SigSpec, StackSlotId, ValueId, VarHandle,
};
use crate::types::{FloatCmp, IntCmp, ScalarType};

/// Emit an exact copy between non-overlapping, equally aligned runtime slots.
pub(crate) fn emit_copy_nonoverlapping(
    builder: &mut FunctionBuilder<'_>,
    config: TargetFrontendConfig,
    destination: Value,
    source: Value,
    size: usize,
    alignment: usize,
) {
    let alignment = u8::try_from(alignment).expect("runtime alignment exceeds u8");
    builder.emit_small_memory_copy(
        config,
        destination,
        source,
        size as u64,
        alignment,
        alignment,
        true,
        MemFlags::trusted(),
    );
}

/// The Cranelift implementation of [`Backend`]: owns the per-function
/// `FunctionBuilder` plus the JIT module.
pub(crate) struct CraneliftBackend<'a, 'b> {
    pub(crate) builder: &'b mut FunctionBuilder<'a>,
    pub(crate) module: &'b mut JITModule,
    /// Module-level maps (our id → Cranelift `FuncId`), shared across all functions.
    pub(crate) func_ids: &'b HashMap<usize, FuncId>,
    pub(crate) extern_func_ids: &'b HashMap<usize, FuncId>,
    /// Per-function caches of imported `FuncRef`s (a fresh backend is built per function).
    pub(crate) func_ref_cache: HashMap<usize, FuncRef>,
    pub(crate) extern_ref_cache: HashMap<usize, FuncRef>,
}

/// Encode a `Vec<BlockArg>` from opaque `ValueId`s for a branch/jump.
fn block_args(args: &[ValueId]) -> Vec<BlockArg> {
    args.iter()
        .map(|&v| BlockArg::Value(v.cranelift()))
        .collect()
}

impl<'a, 'b> Backend for CraneliftBackend<'a, 'b> {
    // ---- constants ----
    fn iconst(&mut self, ty: ScalarType, imm: i64) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().iconst(ty.to_cranelift(), imm))
    }
    fn f64const(&mut self, v: f64) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().f64const(v))
    }
    fn f32const(&mut self, v: f32) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().f32const(v))
    }
    // ---- integer arithmetic ----
    fn iadd(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().iadd(a.cranelift(), b.cranelift()))
    }
    fn isub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().isub(a.cranelift(), b.cranelift()))
    }
    fn imul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().imul(a.cranelift(), b.cranelift()))
    }
    fn sdiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().sdiv(a.cranelift(), b.cranelift()))
    }
    fn udiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().udiv(a.cranelift(), b.cranelift()))
    }
    fn srem(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().srem(a.cranelift(), b.cranelift()))
    }
    fn urem(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().urem(a.cranelift(), b.cranelift()))
    }
    // ---- float arithmetic ----
    fn fadd(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().fadd(a.cranelift(), b.cranelift()))
    }
    fn fsub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().fsub(a.cranelift(), b.cranelift()))
    }
    fn fmul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().fmul(a.cranelift(), b.cranelift()))
    }
    fn fdiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().fdiv(a.cranelift(), b.cranelift()))
    }
    // ---- bitwise / shift ----
    fn band(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().band(a.cranelift(), b.cranelift()))
    }
    fn bor(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().bor(a.cranelift(), b.cranelift()))
    }
    fn bxor(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().bxor(a.cranelift(), b.cranelift()))
    }
    fn ishl(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().ishl(a.cranelift(), b.cranelift()))
    }
    fn sshr(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().sshr(a.cranelift(), b.cranelift()))
    }
    fn ushr(&mut self, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().ushr(a.cranelift(), b.cranelift()))
    }
    // ---- compare / select ----
    fn icmp(&mut self, cc: IntCmp, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().icmp(
            cc.to_cranelift(),
            a.cranelift(),
            b.cranelift(),
        ))
    }
    fn icmp_imm(&mut self, cc: IntCmp, a: ValueId, imm: i64) -> ValueId {
        ValueId::from_cranelift(
            self.builder
                .ins()
                .icmp_imm(cc.to_cranelift(), a.cranelift(), imm),
        )
    }
    fn fcmp(&mut self, cc: FloatCmp, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().fcmp(
            cc.to_cranelift(),
            a.cranelift(),
            b.cranelift(),
        ))
    }
    fn select(&mut self, cond: ValueId, a: ValueId, b: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().select(
            cond.cranelift(),
            a.cranelift(),
            b.cranelift(),
        ))
    }
    // ---- casts ----
    fn sextend(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().sextend(to.to_cranelift(), v.cranelift()))
    }
    fn uextend(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().uextend(to.to_cranelift(), v.cranelift()))
    }
    fn ireduce(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().ireduce(to.to_cranelift(), v.cranelift()))
    }
    fn fcvt_from_sint(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        ValueId::from_cranelift(
            self.builder
                .ins()
                .fcvt_from_sint(to.to_cranelift(), v.cranelift()),
        )
    }
    fn fcvt_from_uint(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        ValueId::from_cranelift(
            self.builder
                .ins()
                .fcvt_from_uint(to.to_cranelift(), v.cranelift()),
        )
    }
    fn bitcast(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let to_ty = to.to_cranelift();
        // Neutral types that differ (e.g. `Ptr` vs `I64`) can still lower to the
        // same Cranelift type; a same-type `bitcast` is invalid IR, so no-op it.
        if self.builder.func.dfg.value_type(v.cranelift()) == to_ty {
            return v;
        }
        ValueId::from_cranelift(
            self.builder
                .ins()
                .bitcast(to_ty, MemFlags::new(), v.cranelift()),
        )
    }
    // ---- memory ----
    fn load(&mut self, ty: ScalarType, ptr: ValueId, offset: i32) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().load(
            ty.to_cranelift(),
            MemFlags::trusted(),
            ptr.cranelift(),
            offset,
        ))
    }
    fn store(&mut self, val: ValueId, ptr: ValueId, offset: i32) {
        self.builder.ins().store(
            MemFlags::trusted(),
            val.cranelift(),
            ptr.cranelift(),
            offset,
        );
    }
    fn stack_addr(&mut self, slot: StackSlotId, offset: i32) -> ValueId {
        ValueId::from_cranelift(
            self.builder
                .ins()
                .stack_addr(types::I64, slot.cranelift(), offset),
        )
    }
    fn alloc_stack_slot(&mut self, size: u32, align_shift: u8) -> StackSlotId {
        let slot = self.builder.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            size,
            align_shift,
        ));
        StackSlotId::from_cranelift(slot)
    }
    fn copy_nonoverlapping(&mut self, dst: ValueId, src: ValueId, size: usize, align: usize) {
        let config = self.module.isa().frontend_config();
        emit_copy_nonoverlapping(
            self.builder,
            config,
            dst.cranelift(),
            src.cranelift(),
            size,
            align,
        );
    }
    // ---- pointers (semantic; §8b) ----
    fn ptr_offset_bytes(&mut self, ptr: ValueId, offset: ValueId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().iadd(ptr.cranelift(), offset.cranelift()))
    }
    fn ptr_offset_const(&mut self, ptr: ValueId, bytes: i64) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().iadd_imm(ptr.cranelift(), bytes))
    }
    fn addr_to_ptr(&mut self, addr: ValueId) -> ValueId {
        addr
    }
    // ---- blocks & control flow ----
    fn create_block(&mut self) -> BlockHandle {
        BlockHandle::from_cranelift(self.builder.create_block())
    }
    fn append_block_param(&mut self, block: BlockHandle, ty: ScalarType) -> ValueId {
        ValueId::from_cranelift(
            self.builder
                .append_block_param(block.cranelift(), ty.to_cranelift()),
        )
    }
    fn block_param(&mut self, block: BlockHandle, idx: usize) -> ValueId {
        ValueId::from_cranelift(self.builder.block_params(block.cranelift())[idx])
    }
    fn switch_to_block(&mut self, block: BlockHandle) {
        self.builder.switch_to_block(block.cranelift());
    }
    fn seal_block(&mut self, block: BlockHandle) {
        self.builder.seal_block(block.cranelift());
    }
    fn jump(&mut self, target: BlockHandle, args: &[ValueId]) {
        let ba = block_args(args);
        self.builder.ins().jump(target.cranelift(), &ba);
    }
    fn brif(
        &mut self,
        cond: ValueId,
        then_block: BlockHandle,
        then_args: &[ValueId],
        else_block: BlockHandle,
        else_args: &[ValueId],
    ) {
        let ta = block_args(then_args);
        let ea = block_args(else_args);
        self.builder.ins().brif(
            cond.cranelift(),
            then_block.cranelift(),
            &ta,
            else_block.cranelift(),
            &ea,
        );
    }
    // ---- variables ----
    fn declare_var(&mut self, ty: ScalarType) -> VarHandle {
        VarHandle::from_cranelift(self.builder.declare_var(ty.to_cranelift()))
    }
    fn def_var(&mut self, var: VarHandle, val: ValueId) {
        self.builder.def_var(var.cranelift(), val.cranelift());
    }
    fn use_var(&mut self, var: VarHandle) -> ValueId {
        ValueId::from_cranelift(self.builder.use_var(var.cranelift()))
    }
    // ---- calls & signatures ----
    fn call(&mut self, func: FuncRefId, args: &[ValueId]) -> Option<ValueId> {
        let cargs: Vec<Value> = args.iter().map(|&v| v.cranelift()).collect();
        let inst = self.builder.ins().call(func.cranelift(), &cargs);
        self.builder
            .inst_results(inst)
            .first()
            .copied()
            .map(ValueId::from_cranelift)
    }
    fn call_indirect(
        &mut self,
        sig: SigRefId,
        callee: ValueId,
        args: &[ValueId],
    ) -> Option<ValueId> {
        let cargs: Vec<Value> = args.iter().map(|&v| v.cranelift()).collect();
        let inst = self
            .builder
            .ins()
            .call_indirect(sig.cranelift(), callee.cranelift(), &cargs);
        self.builder
            .inst_results(inst)
            .first()
            .copied()
            .map(ValueId::from_cranelift)
    }
    fn func_addr(&mut self, func: FuncRefId) -> ValueId {
        ValueId::from_cranelift(self.builder.ins().func_addr(types::I64, func.cranelift()))
    }
    fn import_signature(&mut self, sig: &SigSpec) -> SigRefId {
        let mut signature = Signature::new(self.module.isa().default_call_conv());
        for param in &sig.params {
            signature.params.push(AbiParam::new(param.to_cranelift()));
        }
        if let Some(ret) = sig.ret {
            signature.returns.push(AbiParam::new(ret.to_cranelift()));
        }
        SigRefId::from_cranelift(self.builder.import_signature(signature))
    }
    fn declare_func(&mut self, func_id: usize) -> FuncRefId {
        if let Some(&func_ref) = self.func_ref_cache.get(&func_id) {
            return FuncRefId::from_cranelift(func_ref);
        }
        let cranelift_id = *self
            .func_ids
            .get(&func_id)
            .unwrap_or_else(|| panic!("Function {func_id} not found in func map"));
        let func_ref = self
            .module
            .declare_func_in_func(cranelift_id, self.builder.func);
        self.func_ref_cache.insert(func_id, func_ref);
        FuncRefId::from_cranelift(func_ref)
    }
    fn declare_extern_func(&mut self, extern_id: usize) -> FuncRefId {
        if let Some(&func_ref) = self.extern_ref_cache.get(&extern_id) {
            return FuncRefId::from_cranelift(func_ref);
        }
        let cranelift_id = *self
            .extern_func_ids
            .get(&extern_id)
            .unwrap_or_else(|| panic!("Extern function {extern_id} not found"));
        let func_ref = self
            .module
            .declare_func_in_func(cranelift_id, self.builder.func);
        self.extern_ref_cache.insert(extern_id, func_ref);
        FuncRefId::from_cranelift(func_ref)
    }
}
