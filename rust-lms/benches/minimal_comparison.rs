use criterion::{black_box, criterion_group, criterion_main, Criterion};
use std::time::Duration;

fn generate_test_data(size: usize) -> Vec<f64> {
    (0..size).map(|x| x as f64).collect()
}

// rust-lms version
use rust_lms::prelude::*;
use rust_lms::num::{gt, select};

fn bench_rust_lms_minimal(c: &mut Criterion, data_size: usize) {
    let data = generate_test_data(data_size);
    let threshold = (data_size / 2) as f64;

    let mut compiler = Compiler::new();

    let count_fn = compiler.fun4(
        "count_gt",
        |ctx,
         data: Var<SPtr<F64Type>>,
         len: Var<I64Type>,
         thresh: Var<F64Type>,
         result: Var<SMutPtr<I64Type>>| {
            let i = ctx.let_var(0i64);
            let count = ctx.let_var(0i64);

            let loop_body = while_loop(
                lt(*i, len),
                {
                    let val = array_index(data, *i);
                    let process = if_then(
                        rust_lms::num::gt(val, thresh),
                        assign(*count, add(*count, 1i64))
                    );
                    (process, assign(*i, add(*i, 1i64)))
                }
            );

            (i, count, loop_body, store(result, *count))
        }
    );

    let compiled = compiler.compile(count_fn).expect("Failed to compile");
    let func = compiled.as_fn();

    let mut group = c.benchmark_group(format!("minimal/{}", data_size));

    group.bench_function("rust-lms", |b| {
        let mut result = 0i64;
        b.iter(|| {
            func(
                black_box(data.as_ptr()),
                black_box(data.len() as i64),
                black_box(threshold),
                black_box(&mut result as *mut i64),
            );
            black_box(result)
        });
    });

    // Also benchmark select-based branchless version
    let mut compiler2 = Compiler::new();

    let count_fn_select = compiler2.fun4(
        "count_gt_select",
        |ctx,
         data: Var<SPtr<F64Type>>,
         len: Var<I64Type>,
         thresh: Var<F64Type>,
         result: Var<SMutPtr<I64Type>>| {
            let i = ctx.let_var(0i64);
            let count = ctx.let_var(0i64);

            let loop_body = while_loop(
                lt(*i, len),
                {
                    let val = array_index(data, *i);
                    // Branchless: count += (val > threshold) ? 1 : 0
                    let increment = select(gt(val, thresh), Const::new(1i64), Const::new(0i64));
                    let update_count = assign(*count, add(*count, increment));
                    (update_count, assign(*i, add(*i, 1i64)))
                }
            );

            (i, count, loop_body, store(result, *count))
        }
    );

    let compiled2 = compiler2.compile(count_fn_select).expect("Failed to compile");
    let func2 = compiled2.as_fn();

    group.bench_function("rust-lms-select", |b| {
        let mut result = 0i64;
        b.iter(|| {
            func2(
                black_box(data.as_ptr()),
                black_box(data.len() as i64),
                black_box(threshold),
                black_box(&mut result as *mut i64),
            );
            black_box(result)
        });
    });

    group.finish();
}

// WASM version
#[cfg(feature = "bench-wasm")]
use wasmtime::{Config, Engine, Instance, Memory, Module, OptLevel, Store as WasmStore};

#[cfg(feature = "bench-wasm")]
const WASM_MODULE: &str = r#"
(module
  (memory (export "memory") 16)

  (func $count_gt (export "count_gt")
    (param $data_ptr i32) (param $len i32) (param $threshold f64) (param $result_ptr i32)
    (local $i i32)
    (local $count i64)
    (local $value f64)

    (local.set $count (i64.const 0))
    (local.set $i (i32.const 0))

    (block $break
      (loop $continue
        (br_if $break (i32.ge_u (local.get $i) (local.get $len)))

        (local.set $value
          (f64.load (i32.add (local.get $data_ptr) (i32.mul (local.get $i) (i32.const 8)))))

        (if (f64.gt (local.get $value) (local.get $threshold))
          (then
            (local.set $count (i64.add (local.get $count) (i64.const 1)))))

        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)
      )
    )

    (i64.store (local.get $result_ptr) (local.get $count))
  )
)
"#;

#[cfg(feature = "bench-wasm")]
fn bench_wasm_minimal(c: &mut Criterion, data_size: usize) {
    let data = generate_test_data(data_size);
    let threshold = (data_size / 2) as f64;

    let mut config = Config::new();
    config.cranelift_opt_level(OptLevel::Speed);
    let engine = Engine::new(&config).unwrap();
    let module = Module::new(&engine, WASM_MODULE).unwrap();
    let mut store = WasmStore::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();

    let mut group = c.benchmark_group(format!("minimal/{}", data_size));

    group.bench_function("wasm", |b| {
        b.iter(|| {
            // Copy data to WASM memory
            let data_ptr = 0i32;
            let result_ptr = (data.len() * 8) as i32;
            {
                let mem_data = memory.data_mut(&mut store);
                for (i, &value) in data.iter().enumerate() {
                    let offset = i * 8;
                    mem_data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
                }
            }

            let func = instance
                .get_typed_func::<(i32, i32, f64, i32), ()>(&mut store, "count_gt")
                .unwrap();

            func.call(
                &mut store,
                (
                    black_box(data_ptr),
                    black_box(data.len() as i32),
                    black_box(threshold),
                    black_box(result_ptr),
                ),
            ).unwrap();

            let mem_data = memory.data(&store);
            let result = i64::from_le_bytes(
                mem_data[result_ptr as usize..(result_ptr + 8) as usize].try_into().unwrap()
            );
            black_box(result)
        });
    });

    group.finish();
}

fn benchmark_minimal(c: &mut Criterion) {
    for &size in &[10_000] {
        bench_rust_lms_minimal(c, size);
        #[cfg(feature = "bench-wasm")]
        bench_wasm_minimal(c, size);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets = benchmark_minimal
}

criterion_main!(benches);
