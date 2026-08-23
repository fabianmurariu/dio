//! The MLIR code generator (`MlirBackend`) and its `ScalarType`→MLIR type / comparison
//! mappings. See the [`super`] module docs for where this sits in Phases 1–3.

use melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};
use melior::dialect::llvm::{self, AllocaOptions, LoadStoreOptions};
use melior::dialect::{cf, func};
use melior::ir::attribute::{
    FlatSymbolRefAttribute, IntegerAttribute, StringAttribute, TypeAttribute,
};
use melior::ir::block::BlockLike;
use melior::ir::operation::Operation;
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{Block, Identifier, Location, Module, Region, RegionLike, Type, Value, ValueLike};
use melior::Context;
use mlir_sys::MlirValue;

use crate::staged::{BlockHandle, ValueId, VarHandle};
use crate::types::{FloatCmp, IntCmp, ScalarType};

/// Map a backend-neutral [`ScalarType`] to its MLIR type (docs/llvm.md §8).
///
/// `Bool` is the logical `i1` MLIR uses at comparisons/branches (storage `i1`↔`i8`
/// conversion, §8c, arrives with bool memory ops); `Ptr` is an opaque `llvm.ptr`.
pub(super) fn scalar_to_mlir(context: &Context, ty: ScalarType) -> Type<'_> {
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
    /// External functions referenced by `call`, emitted as module-level `func.func
    /// private` declarations at [`into_module`](Self::into_module).
    externs: Vec<ExternDecl>,
}

/// An external function referenced by [`MlirBackend::call`]: its symbol name, parameter
/// types, and optional result type. Emitted as a `func.func private` declaration and bound
/// to a host address by the driver's `ExecutionEngine::register_symbol`.
struct ExternDecl {
    name: String,
    params: Vec<ScalarType>,
    ret: Option<ScalarType>,
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
            externs: Vec::new(),
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

    // ---- calls & externs ----
    /// Record an external function `name(params) -> ret`, callable via [`call`](Self::call).
    /// Emitted as a module-level `func.func private` declaration at [`into_module`], and
    /// bound to a host address by the driver's `ExecutionEngine::register_symbol`.
    pub fn declare_extern(&mut self, name: &str, params: &[ScalarType], ret: Option<ScalarType>) {
        self.externs.push(ExternDecl {
            name: name.to_string(),
            params: params.to_vec(),
            ret,
        });
    }

    /// Emit a direct call to function `name` (an extern or another defined function) with
    /// `args`, in the current block. Returns the result `ValueId` when `ret` is `Some`.
    pub fn call(
        &mut self,
        name: &str,
        args: &[ValueId],
        ret: Option<ScalarType>,
    ) -> Option<ValueId> {
        let operands: Vec<Value> = args.iter().map(|id| self.get(*id)).collect();
        let result_types: Vec<Type> = ret
            .iter()
            .map(|t| scalar_to_mlir(self.context, *t))
            .collect();
        let callee = FlatSymbolRefAttribute::new(self.context, name);
        let operation = func::call(
            self.context,
            callee,
            &operands,
            &result_types,
            self.location,
        );
        // Confine the block borrow to this scope: extract the raw result (if any) before
        // `intern` takes `&mut self`. A void call still appends its operation here.
        let raw = {
            let call_ref = self.blocks[self.current].append_operation(operation);
            ret.map(|_| {
                call_ref
                    .result(0)
                    .expect("call produces one result")
                    .to_raw()
            })
        };
        raw.map(|raw| self.intern(raw))
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
            externs,
            ..
        } = self;

        let module = Module::new(location);

        // Module-level `func.func private` declarations for every referenced extern.
        for ExternDecl { name, params, ret } in &externs {
            let params: Vec<Type> = params.iter().map(|t| scalar_to_mlir(context, *t)).collect();
            let results: Vec<Type> = ret.iter().map(|t| scalar_to_mlir(context, *t)).collect();
            let signature = FunctionType::new(context, &params, &results);
            let declaration = func::func(
                context,
                StringAttribute::new(context, name),
                TypeAttribute::new(signature.into()),
                Region::new(), // empty region => external declaration
                &[(
                    Identifier::new(context, "sym_visibility"),
                    StringAttribute::new(context, "private").into(),
                )],
                location,
            );
            module.body().append_operation(declaration);
        }

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
        module.body().append_operation(function);
        module
    }
}
