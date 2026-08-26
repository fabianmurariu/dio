//! The MLIR code generator (`MlirBackend`) — the MLIR implementation of the neutral
//! [`Backend`](crate::staged::Backend) trait, plus the `ScalarType`→MLIR type and
//! comparison-predicate mappings. See the [`super`] module docs for where this sits.
//!
//! `MlirBackend` implements the full `Backend` op surface. Construction/finalization
//! (`new`, `param`, `ret`, `into_module`) and the extern/internal-function *setup*
//! (`declare_extern`/`declare_internal_func`, which record a symbol for the trait's
//! id-based `declare_extern_func`/`declare_func` to resolve) stay inherent — they are the
//! MLIR analogue of the `compile()` driver in `func.rs`, not `Backend` ops.

use melior::dialect::arith::{self, CmpfPredicate, CmpiPredicate};
use melior::dialect::llvm::{self, AllocaOptions, LoadStoreOptions};
use melior::dialect::{cf, func};
use melior::ir::attribute::{
    DenseI32ArrayAttribute, FlatSymbolRefAttribute, IntegerAttribute, StringAttribute,
    TypeAttribute,
};
use melior::ir::block::BlockLike;
use melior::ir::operation::{Operation, OperationBuilder};
use melior::ir::r#type::{FunctionType, IntegerType};
use melior::ir::{Block, Identifier, Location, Module, Region, RegionLike, Type, Value, ValueLike};
use melior::Context;
use mlir_sys::MlirValue;

use crate::staged::{
    expect_arguments, Backend, BlockHandle, FuncRefId, SigRefId, SigSpec, StackSlotId, ValueId,
    VarHandle,
};
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

/// A function referenced by symbol name: its parameters and optional result. Used for
/// externs, internal functions, and the resolved `FuncRefId` table.
pub(super) struct FuncDecl {
    name: String,
    params: Vec<ScalarType>,
    ret: Option<ScalarType>,
}

/// The MLIR code generator: builds one function body across `cf` blocks, mapping opaque
/// [`ValueId`]/[`BlockHandle`]/[`VarHandle`]/[`StackSlotId`]/[`FuncRefId`]/[`SigRefId`]s to
/// MLIR entities through `Vec` arenas (docs/llvm.md §5, §9). The MLIR analogue of
/// `CraneliftBackend`.
///
/// **Block model (§5).** A dedicated `entry` block holds the function parameters and *all*
/// variable/stack-slot `llvm.alloca`s — the "entry-block alloca" placement that lets
/// `mem2reg` promote the promotable ones to SSA. It ends with an unconditional branch to
/// body block 0, appended at [`into_module`](Self::into_module) so lazily-declared allocas
/// still precede the entry terminator. User ops go into the body blocks (`blocks`); `current`
/// is the cursor `switch_to_block` moves.
///
/// Each op stashes its result as a lifetime-free raw `MlirValue`, so the block borrow ends
/// immediately and the arenas are plain safe `Vec`s.
pub struct MlirBackend<'c> {
    context: &'c Context,
    location: Location<'c>,
    param_types: Vec<ScalarType>,
    /// Function parameters (as block args) + all allocas; branches to `blocks[0]`.
    entry: Block<'c>,
    /// Body blocks; `blocks[0]` is the "start" block. Indexed by [`BlockHandle`].
    blocks: Vec<Block<'c>>,
    /// Index into `blocks` of the block ops currently append to.
    current: usize,
    /// `ValueId` → MLIR value arena.
    values: Vec<MlirValue>,
    /// `VarHandle` → alloca pointer. The element type lives in the handle.
    vars: Vec<ValueId>,
    /// `StackSlotId` → alloca ptr `ValueId` (an `llvm.ptr` to a byte buffer).
    slots: Vec<ValueId>,
    /// `SigRefId` → (param types, optional result) for `call_indirect`.
    sigs: Vec<(Vec<ScalarType>, Option<ScalarType>)>,
    /// `FuncRefId` → resolved callable (symbol name + signature).
    func_refs: Vec<FuncDecl>,
    /// Externs by extern id (see [`declare_extern`](Self::declare_extern)); emitted as
    /// module-level `func.func private` declarations at [`into_module`](Self::into_module).
    externs: Vec<FuncDecl>,
    /// Internal (JIT-defined) functions by function id (see
    /// [`declare_internal_func`](Self::declare_internal_func)).
    internal_funcs: Vec<FuncDecl>,
}

impl<'c> MlirBackend<'c> {
    /// Begin a function body whose entry block takes `param_types`. The parameters are
    /// interned as the first `ValueId`s, reachable via [`MlirBackend::param`]. Codegen
    /// starts in body block 0.
    pub fn new(context: &'c Context, param_types: Vec<ScalarType>) -> Self {
        let location = Location::unknown(context);
        let block_args: Vec<_> = param_types
            .iter()
            .map(|ty| (scalar_to_mlir(context, *ty), location))
            .collect();
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
            slots: Vec::new(),
            sigs: Vec::new(),
            func_refs: Vec::new(),
            externs: Vec::new(),
            internal_funcs: Vec::new(),
        }
    }

    /// The [`ValueId`] of entry-block parameter `index`.
    pub fn param(&self, index: usize) -> ValueId {
        ValueId::from_u32(index as u32, self.param_types[index])
    }

    /// Record an external function `name(params) -> ret` and return its extern id (for the
    /// trait's [`declare_extern_func`](Backend::declare_extern_func)). Emitted as a
    /// module-level `func.func private` declaration at [`into_module`](Self::into_module) and
    /// bound to a host address by the driver's `ExecutionEngine::register_symbol`.
    pub fn declare_extern(
        &mut self,
        name: &str,
        params: &[ScalarType],
        ret: Option<ScalarType>,
    ) -> usize {
        let id = self.externs.len();
        self.externs.push(FuncDecl {
            name: name.to_string(),
            params: params.to_vec(),
            ret,
        });
        id
    }

    /// Record an internal (JIT-defined) function `name(params) -> ret` and return its
    /// function id (for the trait's [`declare_func`](Backend::declare_func)). The function's
    /// *definition* is assembled by the (future) multi-function MLIR driver.
    pub fn declare_internal_func(
        &mut self,
        name: &str,
        params: &[ScalarType],
        ret: Option<ScalarType>,
    ) -> usize {
        let id = self.internal_funcs.len();
        self.internal_funcs.push(FuncDecl {
            name: name.to_string(),
            params: params.to_vec(),
            ret,
        });
        id
    }

    /// Emit a `func.return` of `value` (or void) in the current block.
    pub fn ret(&mut self, value: Option<ValueId>) {
        let operands: Vec<Value> = value.map(|v| self.get(v)).into_iter().collect();
        self.emit(func::r#return(&operands, self.location));
    }

    /// Consume the backend and wrap its blocks in a `func.func @name` inside a fresh module.
    /// `result_types` are the function's return types (must match the `ret`).
    ///
    /// The entry block is terminated here — after all allocas have been declared into it —
    /// with an unconditional branch to body block 0; then entry and body blocks are appended
    /// to the function region in order. Referenced externs become `func.func private`
    /// declarations.
    pub fn into_module(self, name: &str, result_types: &[ScalarType]) -> Module<'c> {
        let context = self.context;
        let location = self.location;
        let (function, externs) = self.into_function_op(name, result_types);

        let module = Module::new(location);
        emit_extern_declarations(context, &module, &externs);
        module.body().append_operation(function);
        module
    }

    /// Finalize this backend into a single `func.func` operation (terminating the entry
    /// block's branch to the body). Returns the op plus the externs it references (which the
    /// module assembler emits as `func.func private` declarations). This is the multi-function
    /// building block: several ops share one [`Module`].
    pub(super) fn into_function_op(
        self,
        name: &str,
        result_types: &[ScalarType],
    ) -> (Operation<'c>, Vec<FuncDecl>) {
        let MlirBackend {
            context,
            location,
            param_types,
            entry,
            blocks,
            externs,
            ..
        } = self;

        let results: Vec<Type> = result_types
            .iter()
            .map(|t| scalar_to_mlir(context, *t))
            .collect();
        let params: Vec<Type> = param_types
            .iter()
            .map(|ty| scalar_to_mlir(context, *ty))
            .collect();
        let function_type = FunctionType::new(context, &params, &results);

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
        (function, externs)
    }

    // ---- internal helpers ----
    fn intern(&mut self, raw: MlirValue, ty: ScalarType) -> ValueId {
        let id = ValueId::from_u32(self.values.len() as u32, ty);
        self.values.push(raw);
        id
    }

    fn get(&self, id: ValueId) -> Value<'c, '_> {
        // SAFETY: every raw came from a `Value` produced into a block owned by `self`
        // (alive for the backend's whole lifetime); ids are only minted by `intern`/`new`.
        let value = unsafe { Value::from_raw(self.values[id.as_u32() as usize]) };
        assert_eq!(
            value.r#type(),
            scalar_to_mlir(self.context, id.scalar_type()),
            "ValueId {:?} disagrees with its MLIR value type",
            id
        );
        value
    }

    /// Append a single-result op to the current block and intern its result.
    fn emit_value(&mut self, operation: Operation<'c>, ty: ScalarType) -> ValueId {
        let raw = self.blocks[self.current]
            .append_operation(operation)
            .result(0)
            .expect("operation produces one result")
            .to_raw();
        self.intern(raw, ty)
    }

    /// Append a result-less op to the current block.
    fn emit(&mut self, operation: Operation<'c>) {
        self.blocks[self.current].append_operation(operation);
    }

    /// Allocate `count × elem_ty` in the **entry block** (so `mem2reg` can see it), aligned
    /// to `align` bytes, and return the resulting `llvm.ptr` as a `ValueId`.
    fn alloca_entry(&mut self, elem_ty: Type<'c>, count: i64, align: i64) -> ValueId {
        let i64_ty = scalar_to_mlir(self.context, ScalarType::I64);
        let count_raw = self
            .entry
            .append_operation(arith::constant(
                self.context,
                IntegerAttribute::new(i64_ty, count).into(),
                self.location,
            ))
            .result(0)
            .expect("constant result")
            .to_raw();
        // SAFETY: `count_raw` is a value in `self.entry`, alive for the backend's lifetime.
        let count = unsafe { Value::from_raw(count_raw) };
        let ptr_ty = llvm::r#type::pointer(self.context, 0);
        let options = AllocaOptions::new()
            .elem_type(Some(TypeAttribute::new(elem_ty)))
            .align(Some(IntegerAttribute::new(i64_ty, align)));
        let ptr_raw = self
            .entry
            .append_operation(llvm::alloca(
                self.context,
                count,
                ptr_ty,
                self.location,
                options,
            ))
            .result(0)
            .expect("alloca result")
            .to_raw();
        self.intern(ptr_raw, ScalarType::Ptr)
    }

    /// `ptr + bytes` (a byte-indexed `llvm.getelementptr` over `i8`); no-op when `bytes == 0`.
    fn offset_ptr_const(&mut self, ptr: ValueId, bytes: i32) -> ValueId {
        if bytes == 0 {
            return ptr;
        }
        let base = self.get(ptr);
        let i8_ty = scalar_to_mlir(self.context, ScalarType::I8);
        let ptr_ty = llvm::r#type::pointer(self.context, 0);
        let indices = DenseI32ArrayAttribute::new(self.context, &[bytes]);
        self.emit_value(
            llvm::get_element_ptr(self.context, base, indices, i8_ty, ptr_ty, self.location),
            ScalarType::Ptr,
        )
    }

    /// Reduce a branch condition to `i1`. Cranelift's `brif` treats any nonzero integer as
    /// true, so the neutral AST is allowed to branch on a wider integer (e.g. a `u64`
    /// `COption` discriminant); MLIR's `cf.cond_br` strictly requires `i1`. Already-`i1`
    /// conditions pass through untouched; anything else becomes `cond != 0`.
    fn coerce_to_bool(&mut self, cond: ValueId) -> ValueId {
        if cond.scalar_type() == ScalarType::Bool {
            return cond;
        }
        let cond = self.ptr_to_int(cond);
        assert!(
            cond.scalar_type().is_integer(),
            "branch condition must be Bool, an integer, or Ptr"
        );
        let cond_ty = scalar_to_mlir(self.context, cond.scalar_type());
        let zero = self.emit_value(
            arith::constant(
                self.context,
                IntegerAttribute::new(cond_ty, 0).into(),
                self.location,
            ),
            cond.scalar_type(),
        );
        let (a, b) = (self.get(cond), self.get(zero));
        self.emit_value(
            arith::cmpi(self.context, CmpiPredicate::Ne, a, b, self.location),
            ScalarType::Bool,
        )
    }

    /// Reinterpret an `llvm.ptr` as its `i64` address. Cranelift compares/measures pointers
    /// as integers directly; MLIR's `arith` ops reject `!llvm.ptr`, so pointer operands are
    /// `llvm.ptrtoint`-ed to `i64` first (e.g. an `icmp`/`ptr_is_null` on a pointer, or
    /// pointer difference). A non-pointer value passes through.
    fn ptr_to_int(&mut self, v: ValueId) -> ValueId {
        if v.scalar_type() != ScalarType::Ptr {
            return v;
        }
        let val = self.get(v);
        let i64_ty = scalar_to_mlir(self.context, ScalarType::I64);
        let op = OperationBuilder::new("llvm.ptrtoint", self.location)
            .add_operands(&[val])
            .add_results(&[i64_ty])
            .build()
            .expect("valid llvm.ptrtoint");
        self.emit_value(op, ScalarType::I64)
    }
}

impl<'c> Backend for MlirBackend<'c> {
    // ---- constants ----
    fn iconst(&mut self, ty: ScalarType, imm: i64) -> ValueId {
        let mlir_ty = scalar_to_mlir(self.context, ty);
        let attribute = IntegerAttribute::new(mlir_ty, imm).into();
        self.emit_value(arith::constant(self.context, attribute, self.location), ty)
    }
    fn null_ptr(&mut self) -> ValueId {
        let ty = scalar_to_mlir(self.context, ScalarType::Ptr);
        self.emit_value(llvm::zero(ty, self.location), ScalarType::Ptr)
    }
    fn f64const(&mut self, v: f64) -> ValueId {
        let ty = scalar_to_mlir(self.context, ScalarType::F64);
        let attribute = melior::ir::attribute::FloatAttribute::new(self.context, ty, v).into();
        self.emit_value(
            arith::constant(self.context, attribute, self.location),
            ScalarType::F64,
        )
    }
    fn f32const(&mut self, v: f32) -> ValueId {
        let ty = scalar_to_mlir(self.context, ScalarType::F32);
        let attribute =
            melior::ir::attribute::FloatAttribute::new(self.context, ty, v as f64).into();
        self.emit_value(
            arith::constant(self.context, attribute, self.location),
            ScalarType::F32,
        )
    }

    // ---- integer arithmetic ----
    fn iadd(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::addi(a, b, self.location), ty)
    }
    fn isub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::subi(a, b, self.location), ty)
    }
    fn imul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::muli(a, b, self.location), ty)
    }
    fn sdiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::divsi(a, b, self.location), ty)
    }
    fn udiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::divui(a, b, self.location), ty)
    }
    fn srem(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::remsi(a, b, self.location), ty)
    }
    fn urem(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::remui(a, b, self.location), ty)
    }

    // ---- float arithmetic ----
    fn fadd(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::addf(a, b, self.location), ty)
    }
    fn fsub(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::subf(a, b, self.location), ty)
    }
    fn fmul(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::mulf(a, b, self.location), ty)
    }
    fn fdiv(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::divf(a, b, self.location), ty)
    }

    // ---- bitwise / shift ----
    fn band(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::andi(a, b, self.location), ty)
    }
    fn bor(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::ori(a, b, self.location), ty)
    }
    fn bxor(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::xori(a, b, self.location), ty)
    }
    fn ishl(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::shli(a, b, self.location), ty)
    }
    fn sshr(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::shrsi(a, b, self.location), ty)
    }
    fn ushr(&mut self, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(arith::shrui(a, b, self.location), ty)
    }

    // ---- compare / select ----
    fn icmp(&mut self, cc: IntCmp, a: ValueId, b: ValueId) -> ValueId {
        // Pointer operands (e.g. comparing two `SPtr`s) become `i64` addresses first.
        let a = self.ptr_to_int(a);
        let b = self.ptr_to_int(b);
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(
            arith::cmpi(self.context, int_predicate(cc), a, b, self.location),
            ScalarType::Bool,
        )
    }
    fn icmp_imm(&mut self, cc: IntCmp, a: ValueId, imm: i64) -> ValueId {
        // A pointer operand (e.g. `ptr_is_null`, which compares against 0) becomes its `i64`
        // address, so the materialized constant and the compare are plain integer ops.
        let a = self.ptr_to_int(a);
        let ty = scalar_to_mlir(self.context, a.scalar_type());
        let imm_raw = self.blocks[self.current]
            .append_operation(arith::constant(
                self.context,
                IntegerAttribute::new(ty, imm).into(),
                self.location,
            ))
            .result(0)
            .expect("constant result")
            .to_raw();
        let imm_id = self.intern(imm_raw, a.scalar_type());
        self.icmp(cc, a, imm_id)
    }
    fn fcmp(&mut self, cc: FloatCmp, a: ValueId, b: ValueId) -> ValueId {
        let (a, b) = (self.get(a), self.get(b));
        self.emit_value(
            arith::cmpf(self.context, float_predicate(cc), a, b, self.location),
            ScalarType::Bool,
        )
    }
    fn select(&mut self, cond: ValueId, a: ValueId, b: ValueId) -> ValueId {
        let ty = a.scalar_type();
        let (cond, a, b) = (self.get(cond), self.get(a), self.get(b));
        self.emit_value(arith::select(cond, a, b, self.location), ty)
    }

    // ---- casts ----
    fn sextend(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::extsi(v, ty, self.location), to)
    }
    fn uextend(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::extui(v, ty, self.location), to)
    }
    fn ireduce(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::trunci(v, ty, self.location), to)
    }
    fn fcvt_from_sint(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::sitofp(v, ty, self.location), to)
    }
    fn fcvt_from_uint(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::uitofp(v, ty, self.location), to)
    }
    fn bitcast(&mut self, to: ScalarType, v: ValueId) -> ValueId {
        // Same-width reinterpret between non-pointer scalars (e.g. f64↔i64), the Cranelift
        // `bitcast` contract.
        let ty = scalar_to_mlir(self.context, to);
        let v = self.get(v);
        self.emit_value(arith::bitcast(v, ty, self.location), to)
    }

    // ---- memory ----
    fn load(&mut self, ty: ScalarType, ptr: ValueId, offset: i32) -> ValueId {
        let ptr = self.offset_ptr_const(ptr, offset);
        let mlir_ty = scalar_to_mlir(self.context, ty);
        let ptr = self.get(ptr);
        self.emit_value(
            llvm::load(
                self.context,
                ptr,
                mlir_ty,
                self.location,
                LoadStoreOptions::new(),
            ),
            ty,
        )
    }
    fn store(&mut self, val: ValueId, ptr: ValueId, offset: i32) {
        let ptr = self.offset_ptr_const(ptr, offset);
        let (val, ptr) = (self.get(val), self.get(ptr));
        self.emit(llvm::store(
            self.context,
            val,
            ptr,
            self.location,
            LoadStoreOptions::new(),
        ));
    }
    fn stack_addr(&mut self, slot: StackSlotId, offset: i32) -> ValueId {
        let base = self.slots[slot.as_u32() as usize];
        self.offset_ptr_const(base, offset)
    }
    fn alloc_stack_slot(&mut self, size: u32, align_shift: u8) -> StackSlotId {
        let i8_ty = scalar_to_mlir(self.context, ScalarType::I8);
        let ptr = self.alloca_entry(i8_ty, size.max(1) as i64, 1i64 << align_shift);
        let slot = StackSlotId::from_u32(self.slots.len() as u32);
        self.slots.push(ptr);
        slot
    }
    fn copy_nonoverlapping(&mut self, dst: ValueId, src: ValueId, size: usize, _align: usize) {
        // Unrolled byte copy — `size` is a compile-time constant and the aggregates copied
        // here (slices, options, small structs) are small.
        let i8_ty = scalar_to_mlir(self.context, ScalarType::I8);
        for offset in 0..size as i32 {
            let src_ptr = self.offset_ptr_const(src, offset);
            let src_val = self.get(src_ptr);
            let byte_raw = self.blocks[self.current]
                .append_operation(llvm::load(
                    self.context,
                    src_val,
                    i8_ty,
                    self.location,
                    LoadStoreOptions::new(),
                ))
                .result(0)
                .expect("load result")
                .to_raw();
            let byte = self.intern(byte_raw, ScalarType::I8);
            let dst_ptr = self.offset_ptr_const(dst, offset);
            let (byte_val, dst_val) = (self.get(byte), self.get(dst_ptr));
            self.emit(llvm::store(
                self.context,
                byte_val,
                dst_val,
                self.location,
                LoadStoreOptions::new(),
            ));
        }
    }

    // ---- pointers (semantic; §8b) ----
    fn ptr_offset_bytes(&mut self, ptr: ValueId, offset: ValueId) -> ValueId {
        let base = self.get(ptr);
        let index = self.get(offset);
        let i8_ty = scalar_to_mlir(self.context, ScalarType::I8);
        let ptr_ty = llvm::r#type::pointer(self.context, 0);
        self.emit_value(
            llvm::get_element_ptr_dynamic(
                self.context,
                base,
                &[index],
                i8_ty,
                ptr_ty,
                self.location,
            ),
            ScalarType::Ptr,
        )
    }
    fn ptr_offset_const(&mut self, ptr: ValueId, bytes: i64) -> ValueId {
        self.offset_ptr_const(ptr, bytes as i32)
    }
    fn addr_to_ptr(&mut self, addr: ValueId) -> ValueId {
        // `inttoptr` — no melior builder, so build the op directly.
        let addr = self.get(addr);
        let ptr_ty = llvm::r#type::pointer(self.context, 0);
        let operation =
            melior::ir::operation::OperationBuilder::new("llvm.inttoptr", self.location)
                .add_operands(&[addr])
                .add_results(&[ptr_ty])
                .build()
                .expect("valid llvm.inttoptr");
        self.emit_value(operation, ScalarType::Ptr)
    }

    // ---- blocks & control flow ----
    fn create_block(&mut self) -> BlockHandle {
        let handle = BlockHandle::from_u32(self.blocks.len() as u32);
        self.blocks.push(Block::new(&[]));
        handle
    }
    fn append_block_param(&mut self, block: BlockHandle, ty: ScalarType) -> ValueId {
        let mlir_ty = scalar_to_mlir(self.context, ty);
        let raw = self.blocks[block.as_u32() as usize]
            .add_argument(mlir_ty, self.location)
            .to_raw();
        self.intern(raw, ty)
    }
    fn block_param(&mut self, block: BlockHandle, idx: usize, ty: ScalarType) -> ValueId {
        let raw = self.blocks[block.as_u32() as usize]
            .argument(idx)
            .expect("block argument exists")
            .to_raw();
        self.intern(raw, ty)
    }
    fn switch_to_block(&mut self, block: BlockHandle) {
        self.current = block.as_u32() as usize;
    }
    /// No-op on MLIR (block arguments are explicit — see §5).
    fn seal_block(&mut self, _block: BlockHandle) {}
    fn jump(&mut self, target: BlockHandle, args: &[ValueId]) {
        let operands: Vec<Value> = args.iter().map(|id| self.get(*id)).collect();
        let successor = &self.blocks[target.as_u32() as usize];
        let operation = cf::br(successor, &operands, self.location);
        self.blocks[self.current].append_operation(operation);
    }
    fn brif(
        &mut self,
        cond: ValueId,
        then_block: BlockHandle,
        then_args: &[ValueId],
        else_block: BlockHandle,
        else_args: &[ValueId],
    ) {
        let cond = self.coerce_to_bool(cond);
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

    // ---- variables (§5: entry-block alloca + load/store) ----
    fn declare_var(&mut self, ty: ScalarType) -> VarHandle {
        let elem_ty = scalar_to_mlir(self.context, ty);
        let align = ty.size_bytes() as i64;
        let ptr = self.alloca_entry(elem_ty, 1, align);
        let handle = VarHandle::from_u32(self.vars.len() as u32, ty);
        self.vars.push(ptr);
        handle
    }
    fn def_var(&mut self, var: VarHandle, val: ValueId) {
        let ptr = self.vars[var.as_u32() as usize];
        let (val, ptr) = (self.get(val), self.get(ptr));
        self.emit(llvm::store(
            self.context,
            val,
            ptr,
            self.location,
            LoadStoreOptions::new(),
        ));
    }
    fn use_var(&mut self, var: VarHandle) -> ValueId {
        let ptr = self.vars[var.as_u32() as usize];
        let ty = var.scalar_type();
        let mlir_ty = scalar_to_mlir(self.context, ty);
        let ptr = self.get(ptr);
        self.emit_value(
            llvm::load(
                self.context,
                ptr,
                mlir_ty,
                self.location,
                LoadStoreOptions::new(),
            ),
            ty,
        )
    }

    // ---- calls & signatures ----
    fn call(&mut self, func: FuncRefId, args: &[ValueId]) -> Option<ValueId> {
        let decl = &self.func_refs[func.as_u32() as usize];
        let name = decl.name.clone();
        let params = decl.params.clone();
        let ret = decl.ret;
        expect_arguments("call", args, &params);
        let operands: Vec<Value> = args.iter().map(|id| self.get(*id)).collect();
        let result_types: Vec<Type> = ret
            .iter()
            .map(|t| scalar_to_mlir(self.context, *t))
            .collect();
        let callee = FlatSymbolRefAttribute::new(self.context, &name);
        let operation = func::call(
            self.context,
            callee,
            &operands,
            &result_types,
            self.location,
        );
        let raw = {
            let call_ref = self.blocks[self.current].append_operation(operation);
            ret.map(|_| {
                call_ref
                    .result(0)
                    .expect("call produces one result")
                    .to_raw()
            })
        };
        raw.map(|raw| self.intern(raw, func.return_type().unwrap()))
    }
    fn call_indirect(
        &mut self,
        sig: SigRefId,
        callee: ValueId,
        args: &[ValueId],
    ) -> Option<ValueId> {
        let (params, ret) = self.sigs[sig.as_u32() as usize].clone();
        expect_arguments("call_indirect", args, &params);
        // The callee is a typed pointer (including opaque-iterator vtables). LLVM calls
        // through an `!llvm.ptr`. `func.call_indirect` demands a `!func.func` callee, so
        // use an indirect `llvm.call` instead: the pointer is the first `callee_operands` value
        // (no `callee` symbol, no `var_callee_type` since these are non-variadic).
        assert_eq!(
            callee.scalar_type(),
            ScalarType::Ptr,
            "call_indirect callee must be Ptr"
        );
        let mut operands: Vec<Value> = Vec::with_capacity(1 + args.len());
        operands.push(self.get(callee));
        operands.extend(args.iter().map(|id| self.get(*id)));
        let result_types: Vec<Type> = ret
            .iter()
            .map(|t| scalar_to_mlir(self.context, *t))
            .collect();
        // `llvm.call` has `AttrSizedOperandSegments`: [callee_operands, op_bundle_operands].
        let seg = DenseI32ArrayAttribute::new(self.context, &[operands.len() as i32, 0]);
        let bundle_sizes = DenseI32ArrayAttribute::new(self.context, &[]);
        let operation = OperationBuilder::new("llvm.call", self.location)
            .add_operands(&operands)
            .add_results(&result_types)
            .add_attributes(&[
                (
                    Identifier::new(self.context, "operandSegmentSizes"),
                    seg.into(),
                ),
                (
                    Identifier::new(self.context, "op_bundle_sizes"),
                    bundle_sizes.into(),
                ),
            ])
            .build()
            .expect("valid llvm.call");
        let raw = {
            let call_ref = self.blocks[self.current].append_operation(operation);
            ret.map(|_| {
                call_ref
                    .result(0)
                    .expect("indirect call produces one result")
                    .to_raw()
            })
        };
        raw.map(|raw| self.intern(raw, sig.return_type().unwrap()))
    }
    fn func_addr(&mut self, func: FuncRefId) -> ValueId {
        let decl = &self.func_refs[func.as_u32() as usize];
        let name = decl.name.clone();
        let params: Vec<Type> = decl
            .params
            .iter()
            .map(|t| scalar_to_mlir(self.context, *t))
            .collect();
        let results: Vec<Type> = decl
            .ret
            .iter()
            .map(|t| scalar_to_mlir(self.context, *t))
            .collect();
        let fn_ty = FunctionType::new(self.context, &params, &results);
        let callee = FlatSymbolRefAttribute::new(self.context, &name);
        let function_raw = self.blocks[self.current]
            .append_operation(func::constant(self.context, callee, fn_ty, self.location))
            .result(0)
            .expect("function constant produces one result")
            .to_raw();

        // `func.constant` yields a *function-typed* value; the storage-pointer ABI needs the
        // function's *address* as an `llvm.ptr`. Bridge with an unrealized conversion cast —
        // `create_to_llvm` lowers `func.constant` to `llvm.mlir.addressof` (a real `ptr`) and
        // reconciles the (now ptr→ptr) cast away.
        // SAFETY: the value belongs to the current block, which is owned by `self`.
        let function_value = unsafe { Value::from_raw(function_raw) };
        let ptr_ty = llvm::r#type::pointer(self.context, 0);
        let cast = melior::ir::operation::OperationBuilder::new(
            "builtin.unrealized_conversion_cast",
            self.location,
        )
        .add_operands(&[function_value])
        .add_results(&[ptr_ty])
        .build()
        .expect("valid unrealized_conversion_cast");
        self.emit_value(cast, ScalarType::Ptr)
    }
    fn import_signature(&mut self, sig: &SigSpec) -> SigRefId {
        let id = SigRefId::from_u32(self.sigs.len() as u32, sig.ret);
        self.sigs.push((sig.params.clone(), sig.ret));
        id
    }
    fn declare_func(&mut self, func_id: usize) -> FuncRefId {
        let decl = &self.internal_funcs[func_id];
        let resolved = FuncDecl {
            name: decl.name.clone(),
            params: decl.params.clone(),
            ret: decl.ret,
        };
        let id = FuncRefId::from_u32(self.func_refs.len() as u32, decl.ret);
        self.func_refs.push(resolved);
        id
    }
    fn declare_extern_func(&mut self, extern_id: usize) -> FuncRefId {
        let decl = &self.externs[extern_id];
        let resolved = FuncDecl {
            name: decl.name.clone(),
            params: decl.params.clone(),
            ret: decl.ret,
        };
        let id = FuncRefId::from_u32(self.func_refs.len() as u32, decl.ret);
        self.func_refs.push(resolved);
        id
    }
}

/// Emit a module-level `func.func private` declaration for each referenced extern
/// (deduplicated by name), so `func.call`s to them verify and can be bound by symbol.
fn emit_extern_declarations(context: &Context, module: &Module, externs: &[FuncDecl]) {
    let location = Location::unknown(context);
    let mut seen = std::collections::HashSet::new();
    for FuncDecl { name, params, ret } in externs {
        if !seen.insert(name.as_str()) {
            continue;
        }
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
}

/// Assemble several `func.func` operations (built by [`MlirBackend::into_function_op`]) into
/// one [`Module`], emitting the union of their extern declarations. This is how a compilation
/// with helper functions + `__main__` becomes a single JIT-able module.
pub(super) fn assemble_module<'c>(
    context: &'c Context,
    functions: Vec<Operation<'c>>,
    externs: &[FuncDecl],
) -> Module<'c> {
    let location = Location::unknown(context);
    let module = Module::new(location);
    emit_extern_declarations(context, &module, externs);
    for function in functions {
        module.body().append_operation(function);
    }
    module
}
