#![cfg(feature = "llvm")]

use std::sync::{Arc, Barrier};
use std::thread;

use proptest::prelude::*;
use rust_lms::prelude::*;

fn assert_send_sync_static<T: Send + Sync + 'static>() {}

#[test]
fn compiled_owners_and_entry_points_are_send_sync_static() {
    assert_send_sync_static::<Compiled<'static, FunType1<i64, i64>>>();
    assert_send_sync_static::<CompiledFn<FunType1<i64, i64>>>();
}

proptest! {
    #![proptest_config(ProptestConfig {
        // Every case performs an LLVM JIT compilation. Keep the suite useful in debug builds
        // while varying code, inputs, concurrency, and the worker that performs final drop.
        cases: 24,
        max_shrink_iters: 256,
        failure_persistence: Some(Box::new(
            proptest::test_runner::FileFailurePersistence::Direct(
                "proptest-regressions/threaded_compiled.txt",
            ),
        )),
        ..ProptestConfig::default()
    })]

    #[test]
    fn llvm_compiled_function_runs_and_drops_across_worker_threads(
        multiplier in -32i64..=32,
        offset in -10_000i64..=10_000,
        inputs in prop::collection::vec(-1_000_000i64..=1_000_000, 8..96),
        worker_count in 2usize..=8,
        repetitions in 4usize..=24,
    ) {
        let mut compiler = Compiler::new().with_backend(JitBackend::Llvm);
        let kernel = compiler.fun1("parallel_affine", move |_ctx, x: Var<i64>| {
            add(mul(x, multiplier), offset)
        });
        let compiled = compiler.compile(kernel).expect("LLVM compilation");
        let function = compiled.as_fn();

        // Both public owners are independently movable/shareable.
        fn assert_value_send_sync<T: Send + Sync>(_: &T) {}
        assert_value_send_sync(&compiled);
        assert_value_send_sync(&function);

        let inputs = Arc::new(inputs);
        let barrier = Arc::new(Barrier::new(worker_count + 1));
        let mut workers = Vec::with_capacity(worker_count);

        for worker_index in 0..worker_count {
            let function = function.clone();
            let inputs = Arc::clone(&inputs);
            let barrier = Arc::clone(&barrier);
            workers.push(thread::spawn(move || {
                barrier.wait();
                for repetition in 0..repetitions {
                    for (input_index, &input) in inputs.iter().enumerate() {
                        // Rotate the work order to make threads overlap different native calls.
                        if (input_index + repetition) % worker_count == worker_index {
                            assert_eq!(
                                function.call(input),
                                multiplier * input + offset,
                                "worker={worker_index}, repetition={repetition}, input={input}",
                            );
                        }
                    }
                }
                // This clone is dropped on its worker. Once all workers finish, whichever one
                // finishes last destroys the MLIR ExecutionEngine on a non-creator thread.
            }));
        }

        // Ensure no owner remains on the property-test thread. Executable memory now belongs
        // exclusively to the worker-held Arc clones.
        drop(function);
        drop(compiled);
        barrier.wait();

        for worker in workers {
            worker.join().expect("compiled worker panicked");
        }
    }
}
