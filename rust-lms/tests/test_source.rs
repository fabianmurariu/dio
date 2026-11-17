//! Readable text of the Cranelift IR (`Compiler::with_debug_ir_source`). Cranelift only.

use rust_lms::prelude::*;
use rust_lms_derive::extern_fn;

#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn src_add(x: i64, y: i64) -> i64 {
    x + y
}

/// The source kept for `name`.
fn source<'a>(compiled: &'a Compiled<impl StagedType>, name: &str) -> &'a str {
    let (_, text) = (compiled.sources().iter())
        .find(|(n, _)| n == name)
        .unwrap_or_else(|| panic!("no source for {name}"));
    text
}

#[test]
fn loop_reads_as_blocks_and_tuple_assignments() {
    let mut compiler = Compiler::new().with_debug_ir_source(true);
    let count_to_n = compiler.fun1("count_to_n", |ctx, n: Var<i64>| {
        let i = ctx.var(0i64);
        let sum = ctx.var(0i64);
        ctx.while_loop(lt(i, n), move |ctx| {
            ctx.store(sum, sum + i);
            ctx.store(i, i + 1i64);
        });
        sum
    });
    let compiled = compiler.compile(count_to_n).unwrap();
    assert_eq!(compiled.as_fn().call(5), 1 + 2 + 3 + 4);
    // `v0` points at `n` and `v1` at the result (the storage-pointer ABI); `v5` is `i`
    // and `v8` is `sum`, carried round the loop as `b1`'s parameters. Unoptimised, so
    // every instruction keeps its id; the driver's own `return` has none.
    assert_eq!(
        source(&compiled, "count_to_n"),
        "\
fn count_to_n(v0: i64, v1: i64) {
b0:
    v2: i64 = *v0  // @1
    v3 = 0  // @2
    v4 = 0  // @3
    (v5, v8) = (v3, v4)
    goto b1  // @4
b1(v5: i64, v8: i64):
    v7 = v5 < v2  // @5
    if v7 goto b3 else b2  // @6
b3:
    v9 = v8 + v5  // @7
    v10 = 0  // @8
    v11 = 1  // @9
    v12 = v5 + v11  // @10
    (v5, v8) = (v12, v9)
    goto b1  // @11
b2:
    *v1 = v8  // @12
    return
}"
    );
}

#[test]
fn off_by_default() {
    let mut compiler = Compiler::new();
    let inc = compiler.fun1("inc", |_ctx, x: Var<i64>| x + 1i64);
    let compiled = compiler.compile(inc).unwrap();
    if std::env::var_os("RUST_LMS_DEBUG_IR_SOURCE").is_none() {
        assert!(compiled.sources().is_empty());
    }
}

#[test]
fn calls_read_by_name() {
    let mut compiler = Compiler::new().with_debug_ir_source(true);
    let add = compiler.extern_fn::<SrcAddExtern>();
    let f = compiler.fun2("add", |_ctx, x: Var<i64>, y: Var<i64>| {
        call_extern2(add, x, y)
    });
    let compiled = compiler.compile(f).unwrap();
    assert_eq!(compiled.as_fn().call(2, 3), 5);
    assert_eq!(
        source(&compiled, "add"),
        "\
fn add(v0: i64, v1: i64, v2: i64) {
b0:
    v3: i64 = *v0  // @1
    v4: i64 = *v1  // @2
    v5 = &ss0  // @3
    *v5 = v3  // @4
    v6 = &ss1  // @5
    *v6 = v4  // @6
    v7 = &ss2  // @7
    src_add(v5, v6, v7)  // @8
    v8: i64 = *v7  // @9
    *v2 = v8  // @10
    return
}"
    );
    // `__main__` hands back the address of `add`, by name.
    assert!(source(&compiled, "__main__").contains("v1 = &add  // @11\n"));
}

#[test]
fn float_compare_and_select() {
    let mut compiler = Compiler::new().with_debug_ir_source(true);
    let f = compiler.fun2("pick", |ctx, a: Var<f64>, b: Var<f64>| {
        ctx.bind(select(lt(a, b), 3i64, 7i64))
    });
    let compiled = compiler.compile(f).unwrap();
    assert_eq!(compiled.as_fn().call(1.0, 2.0), 3);
    assert_eq!(
        source(&compiled, "pick"),
        "\
fn pick(v0: i64, v1: i64, v2: i64) {
b0:
    v3: f64 = *v0  // @1
    v4: f64 = *v1  // @2
    v5 = v3 < v4  // @3
    v6 = 3  // @4
    v7 = 7  // @5
    v8 = v5 ? v6 : v7  // @6
    *v2 = v8  // @7
    return
}"
    );
}
