use criterion::{black_box, criterion_group, criterion_main, Criterion};
use rand::Rng;
use std::time::Duration;

// ============================================================================
// Data Generation
// ============================================================================

fn generate_test_data(size: usize) -> (Vec<f64>, f64) {
    let mut rng = rand::thread_rng();
    let mut data: Vec<f64> = (0..size)
        .map(|_| rng.gen_range(-10000.0..10000.0))
        .collect();

    // Find median (50th percentile) as predicate value
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = data[size / 2];

    // Shuffle back to random order
    use rand::seq::SliceRandom;
    data.shuffle(&mut rng);

    (data, median)
}

// ============================================================================
// Native Rust Baseline
// ============================================================================

fn compute_stats_native(
    data: &[f64],
    v: f64,
    count: &mut u64,
    min: &mut f64,
    max: &mut f64,
    sum: &mut f64,
    avg: &mut f64,
) {
    *count = 0;
    *min = f64::INFINITY;
    *max = f64::NEG_INFINITY;
    *sum = 0.0;

    for &value in data {
        if value > v {
            *count += 1;
            *sum += value;
            if value < *min {
                *min = value;
            }
            if value > *max {
                *max = value;
            }
        }
    }

    *avg = if *count > 0 {
        *sum / (*count as f64)
    } else {
        0.0
    };
}

// ============================================================================
// rust-lms Implementation
// ============================================================================

use rust_lms::prelude::*;
use rust_lms::num::{gt, min, max, select};

fn compile_and_run_rust_lms(
    data: &[f64],
    v: f64,
    count: &mut u64,
    min: &mut f64,
    max: &mut f64,
    sum: &mut f64,
) {
    let mut compiler = Compiler::new();

    let stats_fn = compiler.fun6(
        "compute_stats",
        |ctx,
         data: Var<SRef<Slice<F64Type>>>,
         v: Var<F64Type>,
         count_ptr: Var<SRefMut<U64Type>>,
         min_ptr: Var<SRefMut<F64Type>>,
         max_ptr: Var<SRefMut<F64Type>>,
         sum_ptr: Var<SRefMut<F64Type>>,
         | {
            // Create loop variable
            let i = ctx.let_var(0u64);

            // Create LOCAL VARIABLES for accumulators - keeps values in registers!
            // This avoids load/store on every iteration
            let count = ctx.let_var(0u64);
            let min = ctx.let_var(f64::INFINITY);
            let max = ctx.let_var(f64::NEG_INFINITY);
            let sum = ctx.let_var(0.0f64);

            // Create a variable to hold the current value (avoid repeated loads)
            let val = ctx.let_var(0.0f64);

            // Main loop - all operations on local variables (registers)
            let loop_body = while_loop(lt(*i, data.len()), {
                // Load value into local variable ONCE per iteration
                let load_val = assign(*val, data.get_unchecked(*i));

                // Check if value > v
                let process = if_then(gt(*val, v), {
                    // Increment count (register operation)
                    let inc_count = assign(*count, add(*count, 1u64));

                    // Add to sum (register operation)
                    let add_sum = assign(*sum, add(*sum, *val));

                    // Update min (register operation)
                    let update_min = if_then(lt(*val, *min), assign(*min, *val));

                    // Update max (register operation)
                    let update_max = if_then(gt(*val, *max), assign(*max, *val));

                    (inc_count, add_sum, update_min, update_max)
                });

                // Increment loop counter
                (load_val, process, assign(*i, add(*i, 1u64)))
            });

            // Store final results to output pointers (only ONCE after loop)
            let store_results = (
                store_ref(count_ptr, *count),
                store_ref(min_ptr, *min),
                store_ref(max_ptr, *max),
                store_ref(sum_ptr, *sum),
            );

            (i, count, min, max, sum, val, loop_body, store_results)
        }
    );

    let compiled = compiler.compile(stats_fn).expect("Failed to compile rust-lms function");
    let func = compiled.as_fn();
    func(data, v, count, min, max, sum);
}

// ============================================================================
// LuaJIT Implementation (optional: cargo bench --features bench-luajit)
// ============================================================================

#[cfg(feature = "bench-luajit")]
use mlua::{Lua, Function};

#[cfg(feature = "bench-luajit")]
fn create_luajit() -> Lua {
    Lua::new()
}

#[cfg(feature = "bench-luajit")]
fn compile_luajit(lua: &Lua) -> mlua::Result<Function> {
    lua.load(r#"
        return function(data, v, count_ptr, min_ptr, max_ptr, sum_ptr)
            local count = 0
            local min = math.huge
            local max = -math.huge
            local sum = 0.0

            for i = 1, #data do
                local value = data[i]
                if value > v then
                    count = count + 1
                    sum = sum + value
                    if value < min then
                        min = value
                    end
                    if value > max then
                        max = value
                    end
                end
            end

            -- Return results
            return count, min, max, sum
        end
    "#).eval()
}

#[cfg(feature = "bench-luajit")]
fn compute_stats_luajit(
    lua: &Lua,
    func: &Function,
    data: &[f64],
    v: f64,
    count: &mut u64,
    min: &mut f64,
    max: &mut f64,
    sum: &mut f64,
) -> mlua::Result<()> {
    // Convert data to Lua table
    let table = lua.create_table()?;
    for (i, &value) in data.iter().enumerate() {
        table.set(i + 1, value)?;
    }

    // Call function
    let results: (u64, f64, f64, f64) = func.call((table, v, (), (), (), ()))?;

    *count = results.0;
    *min = results.1;
    *max = results.2;
    *sum = results.3;

    Ok(())
}

// ============================================================================
// WASM Implementation (optional: cargo bench --features bench-wasm)
// ============================================================================

#[cfg(feature = "bench-wasm")]
use wasmtime::{Config, Engine, Instance, Memory, Module, OptLevel, Store as WasmStore};

#[cfg(feature = "bench-wasm")]
fn create_wasm_engine() -> Engine {
    let mut config = Config::new();
    config.cranelift_opt_level(OptLevel::Speed);
    Engine::new(&config).unwrap()
}

#[cfg(feature = "bench-wasm")]
const WASM_MODULE: &str = r#"
(module
  (memory (export "memory") 16)  ;; 16 pages = 1MB, enough for 100K f64s

  (func $compute_stats (export "compute_stats")
    (param $data_ptr i32) (param $len i32) (param $v f64)
    (param $count_ptr i32) (param $min_ptr i32) (param $max_ptr i32) (param $sum_ptr i32)

    (local $i i32)
    (local $count i64)
    (local $min f64)
    (local $max f64)
    (local $sum f64)
    (local $value f64)

    ;; Initialize
    (local.set $count (i64.const 0))
    (local.set $min (f64.const inf))
    (local.set $max (f64.const -inf))
    (local.set $sum (f64.const 0))
    (local.set $i (i32.const 0))

    ;; Loop over data
    (block $break
      (loop $continue
        ;; Check loop condition: i < len
        (br_if $break (i32.ge_u (local.get $i) (local.get $len)))

        ;; Load value: data[i] (assuming f64 array, 8 bytes per element)
        (local.set $value
          (f64.load (i32.add (local.get $data_ptr) (i32.mul (local.get $i) (i32.const 8)))))

        ;; Check if value > v
        (if (f64.gt (local.get $value) (local.get $v))
          (then
            ;; count++
            (local.set $count (i64.add (local.get $count) (i64.const 1)))

            ;; sum += value
            (local.set $sum (f64.add (local.get $sum) (local.get $value)))

            ;; Update min
            (if (f64.lt (local.get $value) (local.get $min))
              (then (local.set $min (local.get $value))))

            ;; Update max
            (if (f64.gt (local.get $value) (local.get $max))
              (then (local.set $max (local.get $value))))
          )
        )

        ;; i++
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)
      )
    )

    ;; Store results
    (i64.store (local.get $count_ptr) (local.get $count))
    (f64.store (local.get $min_ptr) (local.get $min))
    (f64.store (local.get $max_ptr) (local.get $max))
    (f64.store (local.get $sum_ptr) (local.get $sum))
  )
)
"#;

#[cfg(feature = "bench-wasm")]
fn compile_wasm(engine: &Engine) -> (Module, WasmStore<()>, Instance, Memory) {
    let module = Module::new(engine, WASM_MODULE).unwrap();
    let mut store = WasmStore::new(engine, ());
    let instance = Instance::new(&mut store, &module, &[]).unwrap();
    let memory = instance.get_memory(&mut store, "memory").unwrap();

    (module, store, instance, memory)
}

#[cfg(feature = "bench-wasm")]
fn compute_stats_wasm(
    store: &mut WasmStore<()>,
    instance: &Instance,
    memory: &Memory,
    data: &[f64],
    v: f64,
    count: &mut u64,
    min: &mut f64,
    max: &mut f64,
    sum: &mut f64,
) {
    // Copy data to WASM memory
    let data_ptr = 0;
    let count_ptr = data.len() * 8;
    let min_ptr = count_ptr + 8;
    let max_ptr = min_ptr + 8;
    let sum_ptr = max_ptr + 8;

    {
        let mem_data = memory.data_mut(&mut *store);

        // Write data array
        for (i, &value) in data.iter().enumerate() {
            let offset = data_ptr + i * 8;
            mem_data[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        }
    }

    // Get function
    let func = instance
        .get_typed_func::<(i32, i32, f64, i32, i32, i32, i32), ()>(&mut *store, "compute_stats")
        .unwrap();

    // Call function
    func.call(
        &mut *store,
        (
            data_ptr as i32,
            data.len() as i32,
            v,
            count_ptr as i32,
            min_ptr as i32,
            max_ptr as i32,
            sum_ptr as i32,
        ),
    ).unwrap();

    // Read results
    let mem_data = memory.data(&*store);
    *count = u64::from_le_bytes(mem_data[count_ptr..count_ptr + 8].try_into().unwrap());
    *min = f64::from_le_bytes(mem_data[min_ptr..min_ptr + 8].try_into().unwrap());
    *max = f64::from_le_bytes(mem_data[max_ptr..max_ptr + 8].try_into().unwrap());
    *sum = f64::from_le_bytes(mem_data[sum_ptr..sum_ptr + 8].try_into().unwrap());
}

// ============================================================================
// Rhai Implementation (optional: cargo bench --features bench-rhai)
// ============================================================================

#[cfg(feature = "bench-rhai")]
use rhai::{Engine as RhaiEngine, AST, Scope};

#[cfg(feature = "bench-rhai")]
fn create_rhai() -> RhaiEngine {
    RhaiEngine::new()
}

#[cfg(feature = "bench-rhai")]
fn compile_rhai(engine: &RhaiEngine) -> AST {
    engine.compile(r#"
        let count = 0;
        let min = 1.0e308;
        let max = -1.0e308;
        let sum = 0.0;

        let len = data.len();
        let i = 0;
        while i < len {
            let value = data[i];
            if value > v {
                count += 1;
                sum += value;
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
            }
            i += 1;
        }

        #{count: count, min: min, max: max, sum: sum}
    "#).unwrap()
}

#[cfg(feature = "bench-rhai")]
fn compute_stats_rhai(
    engine: &RhaiEngine,
    ast: &AST,
    data: &[f64],
    v: f64,
    count: &mut u64,
    min: &mut f64,
    max: &mut f64,
    sum: &mut f64,
) {
    let mut scope = Scope::new();
    // Convert to Rhai Array
    let rhai_data: rhai::Array = data.iter().map(|&x| rhai::Dynamic::from_float(x)).collect();
    scope.push("data", rhai_data);
    scope.push("v", v);

    let result: rhai::Map = engine.eval_ast_with_scope(&mut scope, ast).unwrap();

    *count = result.get("count").unwrap().as_int().unwrap() as u64;
    *min = result.get("min").unwrap().as_float().unwrap();
    *max = result.get("max").unwrap().as_float().unwrap();
    *sum = result.get("sum").unwrap().as_float().unwrap();
}

// ============================================================================
// Benchmarks
// ============================================================================

fn bench_native(c: &mut Criterion, data_size: usize) {
    let (data, v) = generate_test_data(data_size);

    let mut group = c.benchmark_group(format!("native/{}", data_size));
    group.bench_function("execution", |b| {
        let mut count = 0u64;
        let mut min = 0.0;
        let mut max = 0.0;
        let mut sum = 0.0;
        let mut avg = 0.0;

        b.iter(|| {
            compute_stats_native(
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
                &mut avg,
            );
            black_box((count, min, max, sum, avg))
        });
    });
    group.finish();
}

fn bench_rust_lms(c: &mut Criterion, data_size: usize) {
    let (data, v) = generate_test_data(data_size);

    let mut group = c.benchmark_group(format!("rust-lms/{}", data_size));

    // Cold start: compile + execute
    group.bench_function("cold_start", |b| {
        b.iter(|| {
            let mut count = 0u64;
            let mut min = 0.0;
            let mut max = 0.0;
            let mut sum = 0.0;

            compile_and_run_rust_lms(
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
            );
            black_box((count, min, max, sum))
        });
    });

    // Warm execution: compile once, then measure only execution
    // Use raw pointers (SMutPtr) to avoid lifetime issues with criterion's b.iter()
    let mut compiler = Compiler::new();

    let stats_fn = compiler.fun7(
        "compute_stats_ptr",
        |ctx,
         data: Var<SPtr<F64Type>>,       // raw pointer to data
         len: Var<I64Type>,               // length of data (i64)
         v: Var<F64Type>,
         count_ptr: Var<SMutPtr<I64Type>>,  // use i64 for count too
         min_ptr: Var<SMutPtr<F64Type>>,
         max_ptr: Var<SMutPtr<F64Type>>,
         sum_ptr: Var<SMutPtr<F64Type>>,
         | {
            let i = ctx.let_var(0i64);

            // LOCAL VARIABLES for accumulators - keeps values in registers!
            let l_count = ctx.let_var(0i64);
            let l_min = ctx.let_var(f64::INFINITY);
            let l_max = ctx.let_var(f64::NEG_INFINITY);
            let l_sum = ctx.let_var(0.0f64);

            // Create a variable to hold the current value
            let val = ctx.let_var(0.0f64);

            let loop_body = while_loop(
                lt(*i, len),
                {
                    // Load value into the variable
                    let load_val = assign(*val, array_index(data, *i));

                    let process = if_then(
                        gt(*val, v),
                        {
                            // All operations on register variables
                            let inc_count = assign(*l_count, add(*l_count, 1i64));
                            let add_sum = assign(*l_sum, add(*l_sum, *val));

                            // Use select for branchless min/max updates
                            let new_min = select(lt(*val, *l_min), *val, *l_min);
                            let update_min = assign(*l_min, new_min);

                            let new_max = select(gt(*val, *l_max), *val, *l_max);
                            let update_max = assign(*l_max, new_max);

                            (inc_count, add_sum, update_min, update_max)
                        }
                    );

                    (load_val, process, assign(*i, add(*i, 1i64)))
                }
            );

            // Store final results (only once after loop)
            let store_results = (
                store(count_ptr, *l_count),
                store(min_ptr, *l_min),
                store(max_ptr, *l_max),
                store(sum_ptr, *l_sum),
            );

            (i, l_count, l_min, l_max, l_sum, val, loop_body, store_results)
        }
    );

    let compiled = compiler.compile(stats_fn).expect("Failed to compile rust-lms function");
    let func = compiled.as_fn();

    group.bench_function("warm_execution", |b| {
        let mut count = 0i64;
        let mut min = 0.0;
        let mut max = 0.0;
        let mut sum = 0.0;

        b.iter(|| {
            func(
                black_box(data.as_ptr()),
                black_box(data.len() as i64),
                black_box(v),
                black_box(&mut count as *mut i64),
                black_box(&mut min as *mut f64),
                black_box(&mut max as *mut f64),
                black_box(&mut sum as *mut f64),
            );
        });
    });

    group.finish();
}

#[cfg(feature = "bench-luajit")]
fn bench_luajit(c: &mut Criterion, data_size: usize) {
    let (data, v) = generate_test_data(data_size);

    let mut group = c.benchmark_group(format!("luajit/{}", data_size));

    // Cold start
    group.bench_function("cold_start", |b| {
        b.iter(|| {
            let lua = create_luajit();
            let func = compile_luajit(&lua).unwrap();
            let mut count = 0u64;
            let mut min = 0.0;
            let mut max = 0.0;
            let mut sum = 0.0;

            compute_stats_luajit(
                black_box(&lua),
                black_box(&func),
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
            ).unwrap();
            black_box((count, min, max, sum))
        });
    });

    // Warm
    let lua = create_luajit();
    let func = compile_luajit(&lua).unwrap();
    group.bench_function("warm_execution", |b| {
        let mut count = 0u64;
        let mut min = 0.0;
        let mut max = 0.0;
        let mut sum = 0.0;

        b.iter(|| {
            compute_stats_luajit(
                black_box(&lua),
                black_box(&func),
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
            ).unwrap();
            black_box((count, min, max, sum))
        });
    });

    group.finish();
}

#[cfg(feature = "bench-wasm")]
fn bench_wasm(c: &mut Criterion, data_size: usize) {
    let (data, v) = generate_test_data(data_size);

    let mut group = c.benchmark_group(format!("wasm/{}", data_size));

    // Cold start
    group.bench_function("cold_start", |b| {
        b.iter(|| {
            let engine = create_wasm_engine();
            let (_module, mut store, instance, memory) = compile_wasm(&engine);
            let mut count = 0u64;
            let mut min = 0.0;
            let mut max = 0.0;
            let mut sum = 0.0;

            compute_stats_wasm(
                black_box(&mut store),
                black_box(&instance),
                black_box(&memory),
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
            );
            black_box((count, min, max, sum))
        });
    });

    // Warm
    let engine = create_wasm_engine();
    let (_module, mut store, instance, memory) = compile_wasm(&engine);
    group.bench_function("warm_execution", |b| {
        let mut count = 0u64;
        let mut min = 0.0;
        let mut max = 0.0;
        let mut sum = 0.0;

        b.iter(|| {
            compute_stats_wasm(
                black_box(&mut store),
                black_box(&instance),
                black_box(&memory),
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
            );
            black_box((count, min, max, sum))
        });
    });

    group.finish();
}

#[cfg(feature = "bench-rhai")]
fn bench_rhai(c: &mut Criterion, data_size: usize) {
    let (data, v) = generate_test_data(data_size);

    let mut group = c.benchmark_group(format!("rhai/{}", data_size));

    // Cold start
    group.bench_function("cold_start", |b| {
        b.iter(|| {
            let engine = create_rhai();
            let ast = compile_rhai(&engine);
            let mut count = 0u64;
            let mut min = 0.0;
            let mut max = 0.0;
            let mut sum = 0.0;

            compute_stats_rhai(
                black_box(&engine),
                black_box(&ast),
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
            );
            black_box((count, min, max, sum))
        });
    });

    // Warm
    let engine = create_rhai();
    let ast = compile_rhai(&engine);
    group.bench_function("warm_execution", |b| {
        let mut count = 0u64;
        let mut min = 0.0;
        let mut max = 0.0;
        let mut sum = 0.0;

        b.iter(|| {
            compute_stats_rhai(
                black_box(&engine),
                black_box(&ast),
                black_box(&data),
                black_box(v),
                &mut count,
                &mut min,
                &mut max,
                &mut sum,
            );
            black_box((count, min, max, sum))
        });
    });

    group.finish();
}

fn benchmark_all_sizes(c: &mut Criterion) {
    for &size in &[1_000, 10_000, 100_000] {
        bench_native(c, size);
        bench_rust_lms(c, size);

        #[cfg(feature = "bench-luajit")]
        bench_luajit(c, size);

        #[cfg(feature = "bench-wasm")]
        bench_wasm(c, size);

        #[cfg(feature = "bench-rhai")]
        bench_rhai(c, size);
    }
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(100);
    targets = benchmark_all_sizes
}

criterion_main!(benches);
