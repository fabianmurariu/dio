//! Filtered-sum on two slices: sum of x[i] + y[i] for indices where the
//! pairwise sum exceeds threshold z (i64).
//!
//! Variants:
//!   - native            : hand-written Rust loop
//!   - rust-lms-cold     : compile + execute (JIT compile inside b.iter)
//!   - rust-lms-slices   : warm execution, passes &[i64] directly (no copy)
//!   - wasm-call-only    : warm execution, data pre-copied into linear memory
//!   - wasm-copy+call    : copy x and y into linear memory, then call
//!
//! Sizes: 10_000, 100_000, 1_000_000.
//!
//! Run with:
//!     cargo bench --bench filtered_sum_two_slices --features bench-wasm

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Duration;

use rust_lms::num::gt;
use rust_lms::prelude::*;

const SIZES: &[usize] = &[10_000, 100_000, 1_000_000];

// ============================================================================
// Data
// ============================================================================

fn generate_data(size: usize, seed: u64) -> (Vec<i64>, Vec<i64>, i64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let x: Vec<i64> = (0..size).map(|_| rng.gen_range(-1_000i64..1_000)).collect();
    let y: Vec<i64> = (0..size).map(|_| rng.gen_range(-1_000i64..1_000)).collect();
    // z = 0 gives ~50% selectivity since (x+y) is symmetric around 0
    (x, y, 0i64)
}

// ============================================================================
// Native baseline
// ============================================================================

#[inline(never)]
fn filtered_sum_native(x: &[i64], y: &[i64], z: i64) -> i64 {
    let mut acc: i64 = 0;
    let n = x.len();
    assert_eq!(n, y.len());
    for i in 0..n {
        let v = x[i].wrapping_add(y[i]);
        if v > z {
            acc = acc.wrapping_add(v);
        }
    }
    acc
}

// ============================================================================
// rust-lms compile (uses &[i64] slice parameters directly — no copy)
// ============================================================================

fn build_filtered_sum_lms() -> impl Fn(&[i64], &[i64], i64) -> i64 {
    let mut compiler = Compiler::new();

    let f = compiler.fun3(
        "filtered_sum",
        |ctx, x: Var<SRef<Slice<i64>>>, y: Var<SRef<Slice<i64>>>, z: Var<i64>| {
            let i = ctx.let_var(0u64);
            let acc = ctx.let_var(0i64);
            let v = ctx.let_var(0i64);

            // while (i < x.len()) {
            //   v = x[i] + y[i];
            //   if (v > z) acc += v;
            //   i += 1;
            // }
            let loop_body = while_loop(
                lt(*i, x.len()),
                (
                    assign(*v, add(x.get_unchecked(*i), y.get_unchecked(*i))),
                    if_then(gt(*v, z), assign(*acc, add(*acc, *v))),
                    assign(*i, add(*i, 1u64)),
                ),
            );

            (i, acc, v, loop_body, *acc)
        },
    );

    let compiled = compiler.compile(f).expect("rust-lms compile failed");
    move |x, y, z| {
        let func = compiled.as_fn();
        func(x, y, z)
    }
}

// ============================================================================
// WASM module — i64 slices, single bulk memcpy per input
// ============================================================================

#[cfg(feature = "bench-wasm")]
use wasmtime::{Config, Engine, Instance, Memory, Module, OptLevel, Store as WasmStore};

// 1_000_000 * 8 * 2 = 16 MB = 256 pages. Use 257 for safety margin.
#[cfg(feature = "bench-wasm")]
const WASM_MODULE: &str = r#"
(module
  (memory (export "memory") 257)

  (func $filtered_sum (export "filtered_sum")
    (param $x_ptr i32) (param $y_ptr i32) (param $len i32) (param $z i64)
    (result i64)

    (local $i_byte i32)
    (local $end_byte i32)
    (local $acc i64)
    (local $v i64)

    (local.set $acc (i64.const 0))
    (local.set $i_byte (i32.const 0))
    (local.set $end_byte (i32.mul (local.get $len) (i32.const 8)))

    (block $break
      (loop $continue
        (br_if $break (i32.ge_u (local.get $i_byte) (local.get $end_byte)))

        (local.set $v
          (i64.add
            (i64.load (i32.add (local.get $x_ptr) (local.get $i_byte)))
            (i64.load (i32.add (local.get $y_ptr) (local.get $i_byte)))))

        (if (i64.gt_s (local.get $v) (local.get $z))
          (then
            (local.set $acc (i64.add (local.get $acc) (local.get $v)))))

        (local.set $i_byte (i32.add (local.get $i_byte) (i32.const 8)))
        (br $continue)
      )
    )

    (local.get $acc)
  )
)
"#;

#[cfg(feature = "bench-wasm")]
fn setup_wasm() -> (Engine, Module) {
    let mut config = Config::new();
    config.cranelift_opt_level(OptLevel::Speed);
    let engine = Engine::new(&config).expect("create wasmtime engine");
    let module = Module::new(&engine, WASM_MODULE).expect("compile WASM module");
    (engine, module)
}

#[cfg(feature = "bench-wasm")]
fn instantiate(engine: &Engine, module: &Module) -> (WasmStore<()>, Instance, Memory) {
    let mut store = WasmStore::new(engine, ());
    let instance = Instance::new(&mut store, module, &[]).expect("instantiate WASM");
    let memory = instance
        .get_memory(&mut store, "memory")
        .expect("WASM memory export missing");
    (store, instance, memory)
}

#[cfg(feature = "bench-wasm")]
fn copy_slice_to_wasm(memory: &Memory, store: &mut WasmStore<()>, offset: usize, data: &[i64]) {
    let bytes = data.len() * std::mem::size_of::<i64>();
    let mem = memory.data_mut(store);
    let src = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, bytes) };
    mem[offset..offset + bytes].copy_from_slice(src);
}

// ============================================================================
// Benchmark
// ============================================================================

fn bench_filtered_sum(c: &mut Criterion) {
    let mut group = c.benchmark_group("filtered_sum");
    group.measurement_time(Duration::from_secs(8));
    group.sample_size(30);

    // Compile rust-lms once and reuse across all sizes (the function is
    // size-agnostic).
    let lms = build_filtered_sum_lms();

    #[cfg(feature = "bench-wasm")]
    let (wasm_engine, wasm_module) = setup_wasm();

    for &size in SIZES {
        let (x, y, z) = generate_data(size, 42);
        let expected = filtered_sum_native(&x, &y, z);

        group.throughput(Throughput::Elements(size as u64));

        // ---------------- native ----------------
        group.bench_with_input(BenchmarkId::new("native", size), &size, |b, _| {
            b.iter(|| {
                black_box(filtered_sum_native(
                    black_box(&x[..]),
                    black_box(&y[..]),
                    black_box(z),
                ))
            });
        });

        // ---------------- rust-lms cold ----------------
        group.bench_with_input(BenchmarkId::new("rust-lms-cold", size), &size, |b, _| {
            b.iter(|| {
                let f = build_filtered_sum_lms();
                black_box(f(black_box(&x[..]), black_box(&y[..]), black_box(z)))
            });
        });

        // ---------------- rust-lms warm (slices, no copy) ----------------
        // Sanity check
        let got = lms(&x[..], &y[..], z);
        assert_eq!(
            got, expected,
            "rust-lms result mismatch at size {}: got {}, expected {}",
            size, got, expected
        );

        group.bench_with_input(BenchmarkId::new("rust-lms-slices", size), &size, |b, _| {
            b.iter(|| black_box(lms(black_box(&x[..]), black_box(&y[..]), black_box(z))));
        });

        // ---------------- WASM variants ----------------
        #[cfg(feature = "bench-wasm")]
        {
            let (mut store, instance, memory) = instantiate(&wasm_engine, &wasm_module);
            let func_wasm = instance
                .get_typed_func::<(i32, i32, i32, i64), i64>(&mut store, "filtered_sum")
                .expect("WASM function not found");

            let x_ptr: i32 = 0;
            let y_ptr: i32 = (size * std::mem::size_of::<i64>()) as i32;

            // Pre-copy data for the call-only variant.
            copy_slice_to_wasm(&memory, &mut store, x_ptr as usize, &x);
            copy_slice_to_wasm(&memory, &mut store, y_ptr as usize, &y);

            // Sanity check
            let wasm_got = func_wasm
                .call(&mut store, (x_ptr, y_ptr, size as i32, z))
                .expect("WASM call failed");
            assert_eq!(
                wasm_got, expected,
                "WASM result mismatch at size {}: got {}, expected {}",
                size, wasm_got, expected
            );

            // -------- wasm-call-only --------
            group.bench_with_input(BenchmarkId::new("wasm-call-only", size), &size, |b, _| {
                b.iter(|| {
                    black_box(
                        func_wasm
                            .call(
                                &mut store,
                                (
                                    black_box(x_ptr),
                                    black_box(y_ptr),
                                    black_box(size as i32),
                                    black_box(z),
                                ),
                            )
                            .unwrap(),
                    )
                });
            });

            // -------- wasm-copy+call --------
            group.bench_with_input(BenchmarkId::new("wasm-copy+call", size), &size, |b, _| {
                b.iter(|| {
                    copy_slice_to_wasm(&memory, &mut store, x_ptr as usize, &x);
                    copy_slice_to_wasm(&memory, &mut store, y_ptr as usize, &y);
                    black_box(
                        func_wasm
                            .call(
                                &mut store,
                                (
                                    black_box(x_ptr),
                                    black_box(y_ptr),
                                    black_box(size as i32),
                                    black_box(z),
                                ),
                            )
                            .unwrap(),
                    )
                });
            });
        }
    }

    group.finish();
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(8))
        .sample_size(30);
    targets = bench_filtered_sum
}

criterion_main!(benches);
