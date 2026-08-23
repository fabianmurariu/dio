//! LLVM/MLIR backend (docs/llvm.md, Phase 1+).
//!
//! **Feature-gated behind `--features llvm`.** This module is the second code
//! generator alongside Cranelift. It needs a system LLVM/MLIR 22 install reachable
//! through melior / mlir-sys (set `MLIR_SYS_220_PREFIX`), so it is compiled *only*
//! when the `llvm` feature is on — the default pure-Rust Cranelift build never pulls
//! melior in.
//!
//! ## Where this is (Phases 1–2)
//!
//! Phase -1 (the `mlir-spike/` crate) proved the JIT/ABI/lowering pipeline against
//! *textual* MLIR. Phases 1–2 prove the same pipeline built **programmatically** through
//! melior's op builders — the alpha-API risk the spike deliberately skipped:
//!
//! - [`make_context`] — the known-good context/dialect/translation setup (from the spike).
//! - [`jit_return_i64_const`] — the nullary-constant milestone: a `() -> i64` `func.func`
//!   returning `arith.constant`, verified, lowered, JIT-run via the **native**
//!   `ExecutionEngine::lookup` pointer.
//! - [`MlirBackend`] — the MLIR analogue of `CraneliftBackend`, addressed by opaque
//!   [`crate::staged::ValueId`]/[`crate::staged::BlockHandle`]/[`crate::staged::VarHandle`]s:
//!   - *values (Phase 1):* constants, integer/float arithmetic, bitwise, shifts,
//!     `IntCmp`/`FloatCmp` compares, branchless `select`, sign/zero casts, `alloca`/`load`/`store`;
//!   - *variables & control flow (Phase 2, §5):* `declare_var`/`def_var`/`use_var` as
//!     entry-block `alloca` + `store`/`load`, `create_block`/`append_block_param`/
//!     `switch_to_block`/`jump`/`brif` over `cf`, and a no-op `seal_block`.
//!   Still an inherent API (not yet the shared [`crate::staged::Backend`] trait — that
//!   waits on the calls/signatures neutralization, Phase 3/4).
//! - [`jit_run_i64_unary`] — drives `MlirBackend` end to end: build a `(i64) -> i64` body
//!   (straight-line *or* with loops/branches), lower, JIT, run.
//!
//! ## The value arena (docs/llvm.md §9 — confirmed)
//!
//! The opaque [`crate::staged::ValueId`] maps to an MLIR value the same stateless way it
//! maps to a Cranelift `Value`: melior's `Value<'c, 'a>` is `#[repr(transparent)]` over a
//! lifetime-free `mlir_sys::MlirValue`, reachable via the public `ValueLike::to_raw` and
//! reconstructable via `Value::from_raw`. So [`MlirBackend`]'s `Vec<MlirValue>` indexed by
//! `ValueId` is the MLIR analogue of Cranelift's entity arena — an ordinary safe `Vec`, no
//! self-referential-struct problem: every op appends to the entry block and immediately
//! stashes the result's raw value, so the block borrow never outlives one method call.

use melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};
use melior::dialect::llvm::{self, AllocaOptions, LoadStoreOptions};
use melior::dialect::{cf, func, DialectRegistry};
use melior::ir::attribute::{IntegerAttribute, StringAttribute, TypeAttribute};
use melior::ir::block::BlockLike;
use melior::ir::operation::{Operation, OperationLike};
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{Block, Location, Module, Region, RegionLike, Type, Value, ValueLike};
use melior::pass::{self, PassManager};
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use melior::{Context, ExecutionEngine};
use mlir_sys::MlirValue;

use crate::staged::{BlockHandle, ValueId, VarHandle};
use crate::types::{FloatCmp, IntCmp, ScalarType};

/// The optimization level passed to `ExecutionEngine`. Must be ≥ 2 so LLVM's own
/// `mem2reg` promotes our entry-block allocas to SSA (the Phase -1 spike established
/// that mlir-sys 220 exposes no MLIR-level mem2reg constructor — see docs/llvm.md §5).
const JIT_OPT_LEVEL: usize = 2;

/// A context with all dialects and LLVM translations registered — the known-good
/// setup proven by the Phase -1 spike.
pub(crate) fn make_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);
    context
}

/// Build a nullary `() -> i64` kernel that returns `value`, JIT it, and run it.
///
/// The Phase 1 end-to-end proof, built entirely through melior's op builders:
/// `arith.constant` + `func.return` inside a `func.func`, verified, lowered to the LLVM
/// dialect, and executed via the **native** function pointer from
/// `ExecutionEngine::lookup` (the Phase-3 ABI, not `invoke_packed`).
pub fn jit_return_i64_const(value: i64) -> i64 {
    let context = make_context();
    let location = Location::unknown(&context);
    let mut module = Module::new(location);

    let i64_ty: Type = IntegerType::new(&context, 64).into();
    let fn_ty = FunctionType::new(&context, &[], &[i64_ty]);

    let function = func::func(
        &context,
        StringAttribute::new(&context, "kernel"),
        TypeAttribute::new(fn_ty.into()),
        {
            let block = Block::new(&[]);
            let constant = block
                .append_operation(arith::constant(
                    &context,
                    IntegerAttribute::new(i64_ty, value).into(),
                    location,
                ))
                .result(0)
                .expect("arith.constant has one result")
                .into();
            block.append_operation(func::r#return(&[constant], location));

            let region = Region::new();
            region.append_block(block);
            region
        },
        &[],
        location,
    );
    module.body().append_operation(function);

    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification before lowering"
    );

    let pass_manager = PassManager::new(&context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager
        .run(&mut module)
        .expect("lower to the LLVM dialect");
    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification after lowering"
    );

    let engine = ExecutionEngine::new(&module, JIT_OPT_LEVEL, &[], false, false);
    let pointer = engine.lookup("kernel");
    assert!(
        !pointer.is_null(),
        "ExecutionEngine::lookup(\"kernel\") is null"
    );

    // SAFETY: `kernel` was emitted with the nullary `() -> i64` signature and the
    // engine (which owns the executable memory) outlives this call.
    let kernel: extern "C" fn() -> i64 = unsafe { std::mem::transmute(pointer) };
    kernel()
}

/// Map a backend-neutral [`ScalarType`] to its MLIR type (docs/llvm.md §8).
///
/// `Bool` is the logical `i1` MLIR uses at comparisons/branches (storage `i1`↔`i8`
/// conversion, §8c, arrives with bool memory ops); `Ptr` is an opaque `llvm.ptr`.
fn scalar_to_mlir(context: &Context, ty: ScalarType) -> Type<'_> {
    match ty {
        ScalarType::Bool => IntegerType::new(context, 1).into(),
        ScalarType::I8 => IntegerType::new(context, 8).into(),
        ScalarType::I16 => IntegerType::new(context, 16).into(),
        ScalarType::I32 => IntegerType::new(context, 32).into(),
        ScalarType::I64 => IntegerType::new(context, 64).into(),
        ScalarType::F32 => Type::float32(context),
        ScalarType::F64 => Type::float64(context),
        ScalarType::Ptr => llvm::r#type::pointer(context, 0),
    }
}

fn int_predicate(cc: IntCmp) -> CmpiPredicate {
    match cc {
        IntCmp::Eq => CmpiPredicate::Eq,
        IntCmp::Ne => CmpiPredicate::Ne,
        IntCmp::Slt => CmpiPredicate::Slt,
        IntCmp::Sgt => CmpiPredicate::Sgt,
        IntCmp::Ult => CmpiPredicate::Ult,
        IntCmp::Ugt => CmpiPredicate::Ugt,
    }
}

fn float_predicate(cc: FloatCmp) -> CmpfPredicate {
    match cc {
        // Ordered predicates (neither operand is NaN) — the Cranelift `fcmp` default.
        FloatCmp::Eq => CmpfPredicate::Oeq,
        FloatCmp::Lt => CmpfPredicate::Olt,
        FloatCmp::Gt => CmpfPredicate::Ogt,
    }
}

/// The MLIR code generator: builds one function body across `cf` blocks, mapping opaque
/// [`ValueId`]/[`BlockHandle`]/[`VarHandle`]s to MLIR entities through `Vec` arenas
/// (docs/llvm.md §5, §9). The MLIR analogue of `CraneliftBackend`.
///
/// **Block model (§5).** There is a dedicated `entry` block holding the function
/// parameters and *all* variable `llvm.alloca`s — the mandatory "entry-block alloca"
/// placement that lets LLVM's `mem2reg` promote them to SSA. It ends with an
/// unconditional branch to body block 0 (the "start" block), appended at
/// [`into_module`](Self::into_module) so allocas declared lazily mid-codegen still land
/// before the entry terminator. All user ops go into the body blocks (`blocks`), indexed
/// by [`BlockHandle`]; `current` is the cursor `switch_to_block` moves.
///
/// **Variables (§5).** `declare_var` → entry `llvm.alloca`; `def_var` → `llvm.store`;
/// `use_var` → `llvm.load`; `seal_block` is a no-op (MLIR block args are explicit).
///
/// Still an **inherent** API, not yet the shared [`crate::staged::Backend`] trait — that
/// waits on neutralizing the trait's calls/signatures cluster (Phase 3/4). Each op stashes
/// its result as a lifetime-free raw `MlirValue`, so the block borrow ends immediately and
/// the arenas are plain safe `Vec`s.
pub struct MlirBackend<'c> {
    context: &'c Context,
    location: Location<'c>,
    param_types: Vec<Type<'c>>,
    /// Function parameters (as block args) + all variable allocas; branches to `blocks[0]`.
    entry: Block<'c>,
    /// Body blocks; `blocks[0]` is the "start" block. Indexed by [`BlockHandle`].
    blocks: Vec<Block<'c>>,
    /// Index into `blocks` of the block ops currently append to.
    current: usize,
    /// `ValueId` → MLIR value arena.
    values: Vec<MlirValue>,
    /// `VarHandle` → (alloca `llvm.ptr`, element type).
    vars: Vec<(MlirValue, ScalarType)>,
}

impl<'c> MlirBackend<'c> {
    /// Begin a function body whose entry block takes `param_types`. The parameters are
    /// interned as the first `ValueId`s, reachable via [`MlirBackend::param`]. Codegen
    /// starts in body block 0.
    pub fn new(context: &'c Context, param_types: Vec<Type<'c>>) -> Self {
        let location = Location::unknown(context);
        let block_args: Vec<_> = param_types.iter().map(|t| (*t, location)).collect();
        let entry = Block::new(&block_args);
        let mut values = Vec::with_capacity(param_types.len());
        for index in 0..param_types.len() {
            let argument = entry.argument(index).expect("entry block argument exists");
            values.push(argument.to_raw());
        }
        Self {
            context,
            location,
            param_types,
            entry,
            blocks: vec![Block::new(&[])],
            current: 0,
            values,
            vars: Vec::new(),
        }
    }

    /// The [`ValueId`] of entry-block parameter `index`.
    pub fn param(&self, index: usize) -> ValueId {
        ValueId::from_u32(index as u32)
    }

    fn intern(&mut self, raw: MlirValue) -> ValueId {
        let id = ValueId::from_u32(self.values.len() as u32);
        self.values.push(raw);
        id
    }

    fn get(&self, id: ValueId) -> Value<'c, '_> {
        // SAFETY: every raw came from a `Value` produced into a block owned by `self`
        // (alive for the backend's whole lifetime), and ids are only minted by `intern`/`new`.
        unsafe { Value::from_raw(self.values[id.as_u32() as usize]) }
    }

    fn emit_value(&mut self, operation: Operation<'c>) -> ValueId {
        let raw = self.blocks[self.current]
            .append_operation(operation)
            .result(0)
            .expect("operation produces one result")
            .to_raw();
        self.intern(raw)
    }

    fn emit(&mut self, operation: Operation<'c>) {
        self.blocks[self.current].append_operation(operation);
    }

    // ---- constants ----
    pub fn iconst(&mut self, ty: ScalarType, value: i64) -> ValueId {
        let ty = scalar_to_mlir(self.context, ty);
        let attribute = IntegerAttribute::new(ty, value).into();
        self.emit_value(arith::constant(self.context, attribute, self.location))
    }

    // ---- integer arithmetic ----
    pub fn iadd(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::addi(a, b, self.location))
    }
    pub fn isub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::subi(a, b, self.location))
    }
    pub fn imul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::muli(a, b, self.location))
    }
    pub fn sdiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::divsi(a, b, self.location))
    }
    pub fn udiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::divui(a, b, self.location))
    }
    pub fn srem(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::remsi(a, b, self.location))
    }
    pub fn urem(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::remui(a, b, self.location))
    }

    // ---- bitwise / shift ----
    pub fn band(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::andi(a, b, self.location))
    }
    pub fn bor(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::ori(a, b, self.location))
    }
    pub fn bxor(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::xori(a, b, self.location))
    }
    pub fn ishl(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::shli(a, b, self.location))
    }
    pub fn sshr(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::shrsi(a, b, self.location))
    }
    pub fn ushr(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::shrui(a, b, self.location))
    }

    // ---- float arithmetic ----
    pub fn fadd(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::addf(a, b, self.location))
    }
    pub fn fsub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::subf(a, b, self.location))
    }
    pub fn fmul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::mulf(a, b, self.location))
    }
    pub fn fdiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::divf(a, b, self.location))
    }

    // ---- compare / select ----
    pub fn icmp(&mut self, cc: IntCmp, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::cmpi(
            self.context,
            int_predicate(cc),
            a,
            b,
            self.location,
        ))
    }
    pub fn fcmp(&mut self, cc: FloatCmp, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::cmpf(
            self.context,
            float_predicate(cc),
            a,
            b,
            self.location,
        ))
    }
    pub fn select(&mut self, cond: ValueId, a: ValueId, b: ValueId) -> ValueId {
        let (cond, a, b) = (self.get(cond), self.get(a), self.get(b));
        self.emit_value(arith::select(cond, a, b, self.location))
    }

    // ---- casts ----
    pub fn sextend(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::extsi(v, ty, self.location))
    }
    pub fn uextend(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::extui(v, ty, self.location))
    }
    pub fn ireduce(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::trunci(v, ty, self.location))
    }
    pub fn fcvt_from_sint(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::sitofp(v, ty, self.location))
    }
    pub fn fcvt_from_uint(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::uitofp(v, ty, self.location))
    }

    // ---- memory ----
    /// Allocate one `elem`-typed slot in the current block and return its `llvm.ptr`.
    pub fn alloca(&mut self, elem: ScalarType) -> ValueId {
        let elem_ty = scalar_to_mlir(self.context, elem);
        let size = self.iconst(ScalarType::I64, 1);
        let size = self.get(size);
        let ptr_ty = llvm::r#type::pointer(self.context, 0);
        let options = AllocaOptions::new().elem_type(Some(TypeAttribute::new(elem_ty)));
        self.emit_value(llvm::alloca(
            self.context,
            size,
            ptr_ty,
            self.location,
            options,
        ))
    }
    pub fn store(&mut self, value: ValueId, ptr: ValueId) {
        let (value, ptr) = (self.get(value), self.get(ptr));
        self.emit(llvm::store(
            self.context,
            value,
            ptr,
            self.location,
            LoadStoreOptions::new(),
        ));
    }
    pub fn load(&mut self, ty: ScalarType, ptr: ValueId) -> ValueId {
        let mlir_ty = scalar_to_mlir(self.context, ty);
        let ptr = self.get(ptr);
        self.emit_value(llvm::load(
            self.context,
            ptr,
            mlir_ty,
            self.location,
            LoadStoreOptions::new(),
        ))
    }

    // ---- variables (§5: entry-block alloca + load/store) ----
    /// Declare a mutable variable: an `llvm.alloca` of `ty` placed in the **entry block**
    /// (never at the current position), so `mem2reg` can promote it to SSA.
    pub fn declare_var(&mut self, ty: ScalarType) -> VarHandle {
        let elem_ty = scalar_to_mlir(self.context, ty);
        let i64_ty = scalar_to_mlir(self.context, ScalarType::I64);
        // Size and alloca both go in the entry block (which has no terminator until
        // `into_module`), so lazily-declared variables still precede the entry branch.
        let size_raw = self
            .entry
            .append_operation(arith::constant(
                self.context,
                IntegerAttribute::new(i64_ty, 1).into(),
                self.location,
            ))
            .result(0)
            .expect("constant result")
            .to_raw();
        // SAFETY: `size_raw` is a value in `self.entry`, alive for the backend's lifetime.
        let size = unsafe { Value::from_raw(size_raw) };
        let ptr_ty = llvm::r#type::pointer(self.context, 0);
        let options = AllocaOptions::new().elem_type(Some(TypeAttribute::new(elem_ty)));
        let ptr_raw = self
            .entry
            .append_operation(llvm::alloca(
                self.context,
                size,
                ptr_ty,
                self.location,
                options,
            ))
            .result(0)
            .expect("alloca result")
            .to_raw();
        let handle = VarHandle::from_u32(self.vars.len() as u32);
        self.vars.push((ptr_raw, ty));
        handle
    }

    /// Store `value` into variable `var` (`llvm.store` in the current block).
    pub fn def_var(&mut self, var: VarHandle, value: ValueId) {
        let (ptr_raw, _) = self.vars[var.as_u32() as usize];
        // SAFETY: `ptr_raw` is the alloca value in `self.entry`, alive for the lifetime.
        let ptr = unsafe { Value::from_raw(ptr_raw) };
        let value = self.get(value);
        self.blocks[self.current].append_operation(llvm::store(
            self.context,
            value,
            ptr,
            self.location,
            LoadStoreOptions::new(),
        ));
    }

    /// Load variable `var` (`llvm.load` in the current block).
    pub fn use_var(&mut self, var: VarHandle) -> ValueId {
        let (ptr_raw, ty) = self.vars[var.as_u32() as usize];
        // SAFETY: as in `def_var`.
        let ptr = unsafe { Value::from_raw(ptr_raw) };
        let mlir_ty = scalar_to_mlir(self.context, ty);
        let raw = self.blocks[self.current]
            .append_operation(llvm::load(
                self.context,
                ptr,
                mlir_ty,
                self.location,
                LoadStoreOptions::new(),
            ))
            .result(0)
            .expect("load result")
            .to_raw();
        self.intern(raw)
    }

    // ---- blocks & control flow ----
    /// Create a fresh (empty, argument-less) body block and return its handle.
    pub fn create_block(&mut self) -> BlockHandle {
        let handle = BlockHandle::from_u32(self.blocks.len() as u32);
        self.blocks.push(Block::new(&[]));
        handle
    }

    /// Append a block-argument (phi) of `ty` to `block`; returns its value.
    pub fn append_block_param(&mut self, block: BlockHandle, ty: ScalarType) -> ValueId {
        let mlir_ty = scalar_to_mlir(self.context, ty);
        let raw = self.blocks[block.as_u32() as usize]
            .add_argument(mlir_ty, self.location)
            .to_raw();
        self.intern(raw)
    }

    /// The `index`-th argument (phi) of `block`.
    pub fn block_param(&mut self, block: BlockHandle, index: usize) -> ValueId {
        let raw = self.blocks[block.as_u32() as usize]
            .argument(index)
            .expect("block argument exists")
            .to_raw();
        self.intern(raw)
    }

    /// Point subsequent ops at `block`.
    pub fn switch_to_block(&mut self, block: BlockHandle) {
        self.current = block.as_u32() as usize;
    }

    /// Seal a block. No-op on MLIR (block arguments are explicit — see §5).
    pub fn seal_block(&mut self, _block: BlockHandle) {}

    /// Unconditional branch to `target`, passing `args` as its block arguments.
    pub fn jump(&mut self, target: BlockHandle, args: &[ValueId]) {
        let operands: Vec<Value> = args.iter().map(|id| self.get(*id)).collect();
        let successor = &self.blocks[target.as_u32() as usize];
        let operation = cf::br(successor, &operands, self.location);
        self.blocks[self.current].append_operation(operation);
    }

    /// Conditional branch on `cond` (an `i1`): to `then_block`/`else_block`, each with its
    /// own block-argument operands.
    pub fn brif(
        &mut self,
        cond: ValueId,
        then_block: BlockHandle,
        then_args: &[ValueId],
        else_block: BlockHandle,
        else_args: &[ValueId],
    ) {
        let cond = self.get(cond);
        let then_ops: Vec<Value> = then_args.iter().map(|id| self.get(*id)).collect();
        let else_ops: Vec<Value> = else_args.iter().map(|id| self.get(*id)).collect();
        let then_succ = &self.blocks[then_block.as_u32() as usize];
        let else_succ = &self.blocks[else_block.as_u32() as usize];
        let operation = cf::cond_br(
            self.context,
            cond,
            then_succ,
            else_succ,
            &then_ops,
            &else_ops,
            self.location,
        );
        self.blocks[self.current].append_operation(operation);
    }

    // ---- return ----
    pub fn ret(&mut self, value: Option<ValueId>) {
        let operands: Vec<Value> = value.map(|v| self.get(v)).into_iter().collect();
        self.emit(func::r#return(&operands, self.location));
    }

    /// Consume the backend and wrap its blocks in a `func.func @name` inside a fresh
    /// module. `result_types` are the function's return types (must match the `ret`).
    ///
    /// The entry block is terminated here — after all variable allocas have been declared
    /// into it — with an unconditional branch to body block 0, then the entry and body
    /// blocks are appended to the function region in order.
    pub fn into_module(self, name: &str, result_types: &[ScalarType]) -> Module<'c> {
        let MlirBackend {
            context,
            location,
            param_types,
            entry,
            blocks,
            ..
        } = self;

        let results: Vec<Type> = result_types
            .iter()
            .map(|t| scalar_to_mlir(context, *t))
            .collect();
        let function_type = FunctionType::new(context, &param_types, &results);

        // Terminate the entry block (alloca-only) with a branch to the start block.
        entry.append_operation(cf::br(&blocks[0], &[], location));

        let region = Region::new();
        region.append_block(entry);
        for block in blocks {
            region.append_block(block);
        }

        let function = func::func(
            context,
            StringAttribute::new(context, name),
            TypeAttribute::new(function_type.into()),
            region,
            &[],
            location,
        );

        let module = Module::new(location);
        module.body().append_operation(function);
        module
    }
}

/// Lower, verify, JIT, and look up `module`'s function `name` as a native pointer.
///
/// Shared tail of the JIT drivers: verify → `create_to_llvm` → verify → `ExecutionEngine`
/// → `lookup`. Returns the raw pointer; the caller transmutes to the concrete ABI.
fn jit_lookup(context: &Context, mut module: Module, name: &str) -> (ExecutionEngine, *mut ()) {
    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification before lowering"
    );
    let pass_manager = PassManager::new(context);
    pass_manager.add_pass(pass::conversion::create_to_llvm());
    pass_manager
        .run(&mut module)
        .expect("lower to the LLVM dialect");
    assert!(
        module.as_operation().verify(),
        "MLIR module failed verification after lowering"
    );

    let engine = ExecutionEngine::new(&module, JIT_OPT_LEVEL, &[], false, false);
    let pointer = engine.lookup(name);
    assert!(
        !pointer.is_null(),
        "ExecutionEngine::lookup found no symbol"
    );
    (engine, pointer)
}

/// Build a `(i64) -> i64` kernel whose body is `build`, JIT it, and run it on `arg`.
///
/// The Phase 1 value-op proof: `build` emits the body through [`MlirBackend`]'s neutral
/// value ops (arithmetic, compares, casts, memory), driven by opaque [`ValueId`]s.
pub fn jit_run_i64_unary(
    build: impl for<'c> FnOnce(&mut MlirBackend<'c>, ValueId) -> ValueId,
    arg: i64,
) -> i64 {
    let context = make_context();
    let i64_ty = scalar_to_mlir(&context, ScalarType::I64);

    let mut backend = MlirBackend::new(&context, vec![i64_ty]);
    let x = backend.param(0);
    let result = build(&mut backend, x);
    backend.ret(Some(result));
    let module = backend.into_module("kernel", &[ScalarType::I64]);

    let (engine, pointer) = jit_lookup(&context, module, "kernel");
    // SAFETY: emitted with the `(i64) -> i64` signature; `engine` owns the executable
    // memory and outlives the call below.
    let kernel: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(pointer) };
    let output = kernel(arg);
    drop(engine);
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nullary_i64_constant_jits_and_runs() {
        assert_eq!(jit_return_i64_const(42), 42);
        assert_eq!(jit_return_i64_const(-7), -7);
        assert_eq!(jit_return_i64_const(0), 0);
        assert_eq!(jit_return_i64_const(i64::MAX), i64::MAX);
    }

    #[test]
    fn arithmetic_on_param_and_constants() {
        // f(x) = x * 3 + 7
        let f = |b: &mut MlirBackend, x: ValueId| {
            let three = b.iconst(ScalarType::I64, 3);
            let seven = b.iconst(ScalarType::I64, 7);
            let scaled = b.imul(x, three);
            b.iadd(scaled, seven)
        };
        for x in [-4i64, 0, 5, 1000] {
            assert_eq!(jit_run_i64_unary(f, x), x * 3 + 7);
        }
    }

    #[test]
    fn compare_and_branchless_select() {
        // f(x) = if x < 0 { -x } else { x }  — abs via icmp + select, no branches.
        let f = |b: &mut MlirBackend, x: ValueId| {
            let zero = b.iconst(ScalarType::I64, 0);
            let neg = b.isub(zero, x);
            let is_neg = b.icmp(IntCmp::Slt, x, zero);
            b.select(is_neg, neg, x)
        };
        for x in [-9i64, -1, 0, 3, 42] {
            assert_eq!(jit_run_i64_unary(f, x), x.abs());
        }
    }

    #[test]
    fn memory_roundtrip_through_alloca() {
        // f(x) = { p = alloca i64; store x + 1, p; load p }
        let f = |b: &mut MlirBackend, x: ValueId| {
            let one = b.iconst(ScalarType::I64, 1);
            let incremented = b.iadd(x, one);
            let slot = b.alloca(ScalarType::I64);
            b.store(incremented, slot);
            b.load(ScalarType::I64, slot)
        };
        for x in [-1i64, 0, 41, 99] {
            assert_eq!(jit_run_i64_unary(f, x), x + 1);
        }
    }

    #[test]
    fn narrowing_then_signed_widening_cast() {
        // f(x) = sextend_i64(ireduce_i32(x)) — truncates to 32 bits, sign-extends back.
        let f = |b: &mut MlirBackend, x: ValueId| {
            let narrow = b.ireduce(ScalarType::I32, x);
            b.sextend(ScalarType::I64, narrow)
        };
        for x in [0i64, 5, -5, i64::from(i32::MAX), i64::from(i32::MIN)] {
            assert_eq!(jit_run_i64_unary(f, x), i64::from(x as i32));
        }
        // A value with nonzero high bits truncates to its low 32 bits (sign-extended).
        let x = 0x1_2345_6789i64;
        assert_eq!(jit_run_i64_unary(f, x), i64::from(x as i32));
    }

    #[test]
    fn mutable_loop_with_alloca_vars() {
        // sum_to(n) = 0 + 1 + ... + (n-1), via two entry-block alloca variables mutated
        // across a `cf` back-edge (promoted to SSA by mem2reg). Exercises declare/def/use_var,
        // create_block, switch_to_block, jump, and brif.
        let sum_to = |b: &mut MlirBackend, n: ValueId| {
            let acc = b.declare_var(ScalarType::I64);
            let iv = b.declare_var(ScalarType::I64);
            let zero = b.iconst(ScalarType::I64, 0);
            b.def_var(acc, zero);
            b.def_var(iv, zero);

            let header = b.create_block();
            let body = b.create_block();
            let exit = b.create_block();
            b.jump(header, &[]);

            b.switch_to_block(header);
            let i = b.use_var(iv);
            let cond = b.icmp(IntCmp::Slt, i, n);
            b.brif(cond, body, &[], exit, &[]);

            b.switch_to_block(body);
            let a = b.use_var(acc);
            let i2 = b.use_var(iv);
            let a2 = b.iadd(a, i2);
            b.def_var(acc, a2);
            let one = b.iconst(ScalarType::I64, 1);
            let i3 = b.iadd(i2, one);
            b.def_var(iv, i3);
            b.jump(header, &[]);

            b.switch_to_block(exit);
            b.use_var(acc)
        };
        for n in [0i64, 1, 5, 10, 100] {
            let expected = (0..n).sum::<i64>();
            assert_eq!(jit_run_i64_unary(sum_to, n), expected);
        }
    }

    #[test]
    fn if_then_else_with_block_argument_phi() {
        // f(x) = if x < 10 { x * 2 } else { x + 100 }, merged through a real `cf` block
        // argument (the phi) rather than a variable — the analogue of Cranelift's merge
        // block param.
        let f = |b: &mut MlirBackend, x: ValueId| {
            let ten = b.iconst(ScalarType::I64, 10);
            let cond = b.icmp(IntCmp::Slt, x, ten);

            let then_block = b.create_block();
            let else_block = b.create_block();
            let merge = b.create_block();
            let phi = b.append_block_param(merge, ScalarType::I64);
            b.brif(cond, then_block, &[], else_block, &[]);

            b.switch_to_block(then_block);
            let two = b.iconst(ScalarType::I64, 2);
            let doubled = b.imul(x, two);
            b.jump(merge, &[doubled]);

            b.switch_to_block(else_block);
            let hundred = b.iconst(ScalarType::I64, 100);
            let bumped = b.iadd(x, hundred);
            b.jump(merge, &[bumped]);

            b.switch_to_block(merge);
            phi
        };
        for x in [-5i64, 0, 9, 10, 11, 50] {
            let expected = if x < 10 { x * 2 } else { x + 100 };
            assert_eq!(jit_run_i64_unary(f, x), expected);
        }
    }
}
