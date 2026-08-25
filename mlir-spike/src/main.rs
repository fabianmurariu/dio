//! Phase -1 MLIR de-risking spike (see docs/llvm.md §10).
//!
//! Proves the five load-bearing mechanics of an LLVM/MLIR backend work end to end via
//! melior + system LLVM/MLIR 22, BEFORE committing to the wide `dyn Backend` refactor:
//!
//!   1. JIT a function and call it through `ExecutionEngine::lookup` (the NATIVE
//!      function pointer — not `invoke_packed`), preserving the Phase-3 ABI.
//!   2. A mutable loop built from entry-block `llvm.alloca` + load/store + `cf` blocks,
//!      promoted to SSA by LLVM's mem2reg at JIT opt level.
//!   3. A load through a real `llvm.ptr`.
//!   4. A call into a registered Rust `extern "C"` symbol through a pointer argument.
//!   5. Module verification before and after lowering.
//!
//! The IR is written as textual MLIR and `Module::parse`d: the spike's job is to prove
//! the JIT/ABI/lowering pipeline (the real risk), not to exercise melior's alpha
//! op-builder helpers (that is Phase 1's job).
//!
//! Build/run (Homebrew llvm@22 shown):
//!   MLIR_SYS_220_PREFIX=/opt/homebrew/opt/llvm LLVM_SYS_220_PREFIX=/opt/homebrew/opt/llvm \
//!   DYLD_LIBRARY_PATH=/opt/homebrew/opt/llvm/lib cargo run -p mlir-spike

use melior::dialect::DialectRegistry;
use melior::ir::operation::OperationLike;
use melior::ir::Module;
use melior::pass::{self, PassManager};
use melior::utility::{register_all_dialects, register_all_llvm_translations};
use melior::{Context, ExecutionEngine};

/// A context with all dialects + LLVM translations registered (the known-good setup).
fn make_context() -> Context {
    let registry = DialectRegistry::new();
    register_all_dialects(&registry);
    let context = Context::new();
    context.append_dialect_registry(&registry);
    context.load_all_available_dialects();
    register_all_llvm_translations(&context);
    context
}

/// Parse `src`, verify, lower to the LLVM dialect, verify again, and JIT it.
/// Returns the ExecutionEngine (opt level 2 → LLVM runs mem2reg etc. on our allocas).
fn compile(context: &Context, src: &str) -> ExecutionEngine {
    let mut module = Module::parse(context, src).expect("parse MLIR");
    assert!(
        module.as_operation().verify(),
        "module failed verification BEFORE lowering"
    );

    let pm = PassManager::new(context);
    pm.add_pass(pass::conversion::create_to_llvm());
    pm.run(&mut module).expect("lower to LLVM dialect");

    assert!(
        module.as_operation().verify(),
        "module failed verification AFTER lowering"
    );

    ExecutionEngine::new(&module, 2, &[], false, false)
}

fn check1_lookup_and_call_native() {
    let context = make_context();
    let engine = compile(
        &context,
        r#"
        module {
          func.func @add(%a: i64, %b: i64) -> i64 {
            %s = arith.addi %a, %b : i64
            return %s : i64
          }
        }
        "#,
    );

    let ptr = engine.lookup("add");
    assert!(!ptr.is_null(), "lookup(\"add\") returned null");
    // Call the NATIVE function pointer directly — the Phase-3 by-value/by-pointer path,
    // NOT invoke_packed's void(void**) wrapper.
    let add: extern "C" fn(i64, i64) -> i64 = unsafe { std::mem::transmute(ptr) };
    let got = add(40, 2);
    assert_eq!(got, 42, "native add(40,2)");
    println!("  [1] lookup + native call: add(40,2) = {got}  OK");
}

fn check2_mutable_loop_alloca_cf() {
    let context = make_context();
    // sum_to(n) = 0+1+...+(n-1), via a mutable accumulator + induction var in
    // ENTRY-BLOCK llvm.alloca, a cf loop, and load/store mutation.
    let engine = compile(
        &context,
        r#"
        module {
          func.func @sum_to(%n: i64) -> i64 {
            %zero = arith.constant 0 : i64
            %one  = arith.constant 1 : i64
            %sz   = arith.constant 1 : i64
            %acc = llvm.alloca %sz x i64 : (i64) -> !llvm.ptr
            %iv  = llvm.alloca %sz x i64 : (i64) -> !llvm.ptr
            llvm.store %zero, %acc : i64, !llvm.ptr
            llvm.store %zero, %iv  : i64, !llvm.ptr
            cf.br ^cond
          ^cond:
            %i = llvm.load %iv : !llvm.ptr -> i64
            %c = arith.cmpi slt, %i, %n : i64
            cf.cond_br %c, ^body, ^exit
          ^body:
            %a = llvm.load %acc : !llvm.ptr -> i64
            %i2 = llvm.load %iv : !llvm.ptr -> i64
            %a2 = arith.addi %a, %i2 : i64
            llvm.store %a2, %acc : i64, !llvm.ptr
            %i3 = arith.addi %i2, %one : i64
            llvm.store %i3, %iv : i64, !llvm.ptr
            cf.br ^cond
          ^exit:
            %r = llvm.load %acc : !llvm.ptr -> i64
            return %r : i64
          }
        }
        "#,
    );

    let ptr = engine.lookup("sum_to");
    assert!(!ptr.is_null(), "lookup(\"sum_to\") returned null");
    let sum_to: extern "C" fn(i64) -> i64 = unsafe { std::mem::transmute(ptr) };
    let got = sum_to(5);
    assert_eq!(got, 10, "sum_to(5) = 0+1+2+3+4");
    println!("  [2] entry-block alloca + cf mutable loop: sum_to(5) = {got}  OK");
}

fn check3_load_through_llvm_ptr() {
    let context = make_context();
    let engine = compile(
        &context,
        r#"
        module {
          func.func @deref(%p: !llvm.ptr) -> i64 {
            %v = llvm.load %p : !llvm.ptr -> i64
            return %v : i64
          }
        }
        "#,
    );

    let ptr = engine.lookup("deref");
    assert!(!ptr.is_null(), "lookup(\"deref\") returned null");
    let deref: extern "C" fn(*const i64) -> i64 = unsafe { std::mem::transmute(ptr) };
    let value: i64 = 42;
    let got = deref(&value as *const i64);
    assert_eq!(got, 42, "deref(&42)");
    println!("  [3] load through real llvm.ptr: deref(&42) = {got}  OK");
}

// The registered Rust extern for check 4: reads an i64 through a pointer argument
// (mirrors the Phase-3 storage-pointer ABI shape).
extern "C" fn spike_read(p: *const i64) -> i64 {
    unsafe { *p }
}

fn check4_call_registered_extern() {
    let context = make_context();
    let engine = compile(
        &context,
        r#"
        module {
          func.func private @spike_read(!llvm.ptr) -> i64
          func.func @call_read(%p: !llvm.ptr) -> i64 {
            %r = func.call @spike_read(%p) : (!llvm.ptr) -> i64
            return %r : i64
          }
        }
        "#,
    );

    // Register the host symbol BEFORE looking up / running the caller.
    unsafe {
        engine.register_symbol("spike_read", spike_read as *mut ());
    }

    let ptr = engine.lookup("call_read");
    assert!(!ptr.is_null(), "lookup(\"call_read\") returned null");
    let call_read: extern "C" fn(*const i64) -> i64 = unsafe { std::mem::transmute(ptr) };
    let value: i64 = 41;
    let got = call_read(&value as *const i64);
    assert_eq!(got, 41, "call_read(&41) via registered extern");
    println!("  [4] registered Rust extern via pointer arg: call_read(&41) = {got}  OK");
}

fn main() {
    println!("Phase -1 MLIR spike (melior + system LLVM/MLIR 22):");
    check1_lookup_and_call_native();
    check2_mutable_loop_alloca_cf();
    check3_load_through_llvm_ptr();
    check4_call_registered_extern();
    println!("[5] module verified before AND after lowering in every check above  OK");
    println!("\nAll five spike checks passed. The dyn Backend + ValueId direction is de-risked.");
    println!("Note: mlir-sys 220 exposes no explicit mem2reg pass constructor; alloca");
    println!("promotion is handled by LLVM's own mem2reg at ExecutionEngine opt level 2.");
}
