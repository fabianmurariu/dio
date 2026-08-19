use rust_lms::func::*;
use rust_lms::prelude::*;

#[test]
fn test_simple_constant() {
    let compiler = Compiler::new();
    let five = Const::<i64>::new(5);

    let compiled = compiler.compile(five).expect("compilation failed");
    let result = compiled.run();

    assert_eq!(result, 5);
}

#[test]
fn test_simple_addition() {
    let compiler = Compiler::new();
    let expr = add::<i64, _, _>(3i64, 4i64);

    let compiled = compiler.compile(expr).expect("compilation failed");
    let result = compiled.run();

    assert_eq!(result, 7);
}

#[test]
fn test_nested_arithmetic() {
    let compiler = Compiler::new();
    // (3 + 4) * 2 = 14
    let expr = mul(
        add(Const::<i64>::new(3), Const::<i64>::new(4)),
        Const::<i64>::new(2),
    );

    let compiled = compiler.compile(expr).expect("compilation failed");
    let result = compiled.run();

    assert_eq!(result, 14);
}

#[test]
fn test_fun1_and_call() {
    let mut compiler = Compiler::new();

    // Define: square(x) = x * x
    let square = compiler.fun1("square", |_ctx, x: Var<i64>| mul(x, x));

    // Call: square(5) = 25
    let expr = call1(square, Const::<i64>::new(5));

    let compiled = compiler.compile(expr).expect("compilation failed");
    let result = compiled.run();

    assert_eq!(result, 25);
}

#[test]
fn test_var_before_fun1() {
    let mut compiler = Compiler::new();

    // Ensure that functions defined after internal var allocation don't get wrong param IDs.
    // Allocate a var inside a fun0 first to advance the counter.
    let _prime = compiler.fun0("prime_counter", |ctx| {
        let _x = ctx.var(0i64);
        Const::<i64>::new(0)
    });

    // Define: double(x) = x + x
    let double = compiler.fun1("double", |_ctx, x: Var<i64>| add(x, x));

    // Call: double(7) = 14
    let expr = call1(double, Const::<i64>::new(7));

    let compiled = compiler.compile(expr).expect("compilation failed");
    let result = compiled.run();

    assert_eq!(result, 14);
}

#[test]
fn test_return_function_pointer() {
    let mut compiler = Compiler::new();

    // Define: cube(x) = x * x * x
    let cube = compiler.fun1("cube", |_ctx, x: Var<i64>| mul(mul(x, x), x));

    // Compile the function reference itself (not a call)
    let compiled = compiler.compile(cube).expect("compilation failed");

    // Borrow a callable entry point from the compiled module.
    let cube_fn = compiled.as_fn();

    // Test the function with various inputs
    assert_eq!(cube_fn.call(2), 8);
    assert_eq!(cube_fn.call(3), 27);
    assert_eq!(cube_fn.call(5), 125);
    assert_eq!(cube_fn.call(-2), -8);
}

#[test]
fn test_unchecked_function_pointer_while_owner_is_live() {
    let mut compiler = Compiler::new();
    let add = compiler.fun2("add", |_ctx, a: Var<i64>, b: Var<i64>| add(a, b));
    let compiled = compiler.compile(add).expect("compilation failed");

    // SAFETY: `compiled` remains live for every invocation of this detached
    // pointer, and the function embeds no host allocation addresses.
    let add_fn = unsafe { compiled.as_fn_unchecked() };

    assert_eq!(add_fn(20, 22), 42);
}

#[test]
fn test_recursive_function_compiles() {
    let mut compiler = Compiler::new();

    // Define a recursive function: rec(x) = x + rec(x - 1)
    // Note: This will infinite loop if called, but we're just testing
    // that it compiles and the function can reference itself
    let _rec = compiler.fun1_rec("recursive", |f, _ctx, x: Var<i64>| {
        // Body references itself: call f recursively
        add(x, call1(f, sub(x, Const::<i64>::new(1))))
    });

    // Just test that compilation succeeds
    // We don't call it since it would infinite loop without conditionals
    let expr = Const::<i64>::new(42);
    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 42);
}

// =========================================================================
// Control Flow Tests
// =========================================================================

#[test]
fn test_if_then_else_basic() {
    // Test true branch
    let compiler = Compiler::new();
    let expr_true = if_then_else(true, Const::<i64>::new(10), Const::<i64>::new(20));
    assert_eq!(compiler.compile(expr_true).unwrap().run(), 10);

    // Test false branch
    let compiler = Compiler::new();
    let expr_false = if_then_else(false, Const::<i64>::new(10), Const::<i64>::new(20));
    assert_eq!(compiler.compile(expr_false).unwrap().run(), 20);
}

#[test]
fn test_if_then_else_clamp() {
    let mut compiler = Compiler::new();

    // clamp(x) = if x < 0 then 0 else (if x > 10 then 10 else x)
    let clamp = compiler.fun1("clamp", |_ctx, x: Var<i64>| {
        if_then_else(
            lt(x, 0),
            Const::<i64>::new(0),
            if_then_else(lt(10, x), Const::<i64>::new(10), x),
        )
    });

    let compiled = compiler.compile(clamp).expect("compilation failed");
    let clamp_fn = compiled.as_fn();

    assert_eq!(clamp_fn.call(-5), 0); // Clamped at min
    assert_eq!(clamp_fn.call(5), 5); // In range
    assert_eq!(clamp_fn.call(15), 10); // Clamped at max
}

#[test]
fn test_seq_basic() {
    let compiler = Compiler::new();

    // (5, 10) => 10 (first value ignored, second returned)
    let expr = (Const::<i64>::new(5), Const::<i64>::new(10));

    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 10);
}

#[test]
fn test_let_var() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("let_var_test", |ctx| {
        let x = ctx.var(42i64);
        let y = ctx.var(8i64);
        add::<i64, _, _>(x, y)
    });

    let compiled = compiler.compile(call0(f)).expect("compilation failed");
    assert_eq!(compiled.run(), 50);
}

#[test]
fn test_ergonomic_assign() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("ergonomic_assign", |ctx| {
        let x = ctx.var(0i64);
        let y = ctx.var(0i64);
        ctx.store(x, 10i64);
        ctx.store(y, 32i64);
        add(x, y)
    });

    let compiled = compiler.compile(call0(f)).expect("compilation failed");
    assert_eq!(compiled.run(), 42);
}

#[test]
fn test_recursive_factorial() {
    let mut compiler = Compiler::new();

    // factorial(n) = if n <= 1 then 1 else n * factorial(n - 1)
    let factorial = compiler.fun1_rec("factorial", |f, _ctx, n: Var<i64>| {
        if_then_else(lt(n, 2), Const::new(1), mul(n, call1(f, sub(n, 1))))
    });

    let compiled = compiler.compile(factorial).expect("compilation failed");
    let factorial_fn = compiled.as_fn();

    assert_eq!(factorial_fn.call(0), 1);
    assert_eq!(factorial_fn.call(1), 1);
    assert_eq!(factorial_fn.call(5), 120);
    assert_eq!(factorial_fn.call(10), 3628800);
}

#[test]
fn test_fibonacci() {
    let mut compiler = Compiler::new();

    // fib(n) = if n < 2 then n else fib(n-1) + fib(n-2)
    let fib = compiler.fun1_rec("fib", |f, _ctx, n: Var<i64>| {
        if_then_else(
            lt(n, Const::<i64>::new(2)),
            n, // fib(0) = 0, fib(1) = 1
            add(
                call1(f, sub(n, Const::<i64>::new(1))),
                call1(f, sub(n, Const::<i64>::new(2))),
            ),
        )
    });

    // Compile and get function pointer
    let compiled = compiler.compile(fib).expect("compilation failed");
    let fib_fn = compiled.as_fn();

    // Test Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34
    assert_eq!(fib_fn.call(0), 0);
    assert_eq!(fib_fn.call(1), 1);
    assert_eq!(fib_fn.call(2), 1);
    assert_eq!(fib_fn.call(3), 2);
    assert_eq!(fib_fn.call(4), 3);
    assert_eq!(fib_fn.call(5), 5);
    assert_eq!(fib_fn.call(6), 8);
    assert_eq!(fib_fn.call(7), 13);
    assert_eq!(fib_fn.call(10), 55);
}

// =========================================================================
// While Loop Tests
// =========================================================================

#[test]
fn test_while_loop_zero_iterations() {
    let mut compiler = Compiler::new();

    let f = compiler.fun0("while_zero", |ctx| {
        let result = ctx.var(0i64);
        ctx.while_loop(false, move |ctx| {
            ctx.store(result, 999i64);
        });
        ctx.store(result, 42i64);
        result
    });

    let compiled = compiler.compile(call0(f)).expect("compilation failed");
    assert_eq!(compiled.run(), 42);
}

#[test]
fn test_while_loop_factorial() {
    let mut compiler = Compiler::new();

    let factorial_iter = compiler.fun1("factorial_iter", |ctx, n: Var<i64>| {
        let i = ctx.var(1i64);
        let result = ctx.var(1i64);
        ctx.while_loop(lt(i, add(n, 1i64)), move |ctx| {
            ctx.store(result, mul(result, i));
            ctx.store(i, add(i, 1i64));
        });
        result
    });

    let compiled = compiler
        .compile(factorial_iter)
        .expect("compilation failed");
    let factorial_fn = compiled.as_fn();

    assert_eq!(factorial_fn.call(0), 1);
    assert_eq!(factorial_fn.call(1), 1);
    assert_eq!(factorial_fn.call(2), 2);
    assert_eq!(factorial_fn.call(3), 6);
    assert_eq!(factorial_fn.call(5), 120);
    assert_eq!(factorial_fn.call(10), 3628800);
}

#[test]
fn test_while_loop_fibonacci_iterative() {
    let mut compiler = Compiler::new();

    let fib_iter = compiler.fun1("fib_iter", |ctx, n: Var<i64>| {
        let i = ctx.var(2i64);
        let a = ctx.var(0i64);
        let b = ctx.var(1i64);
        let temp = ctx.var(0i64);
        if_then_else(lt(n, 2), n, {
            ctx.while_loop(lt(i, add(n, 1i64)), move |ctx| {
                ctx.store(temp, add(a, b));
                ctx.store(a, b);
                ctx.store(b, temp);
                ctx.store(i, add(i, 1i64));
            });
            b
        })
    });

    let compiled = compiler.compile(fib_iter).expect("compilation failed");
    let fib_fn = compiled.as_fn();

    assert_eq!(fib_fn.call(0), 0);
    assert_eq!(fib_fn.call(1), 1);
    assert_eq!(fib_fn.call(2), 1);
    assert_eq!(fib_fn.call(3), 2);
    assert_eq!(fib_fn.call(4), 3);
    assert_eq!(fib_fn.call(5), 5);
    assert_eq!(fib_fn.call(10), 55);
    assert_eq!(fib_fn.call(20), 6765);
    assert_eq!(fib_fn.call(30), 832040);
}

#[test]
fn test_local_variables_in_fun1() {
    use rust_lms::refer::SRef;
    use rust_lms::slice::Slice;

    let mut compiler = Compiler::new();

    // Function that sums elements > 5 using local variables
    // fn sum_gt_5(arr: &[i64]) -> i64
    let sum_gt_5 = compiler.fun1("sum_gt_5", |ctx, arr: Var<SRef<Slice<i64>>>| {
        // Create local variables inside the function using ctx
        let i = ctx.let_var(0u64);
        let sum = ctx.let_var(0i64);
        let v = ctx.let_var(0i64);

        (
            (i, sum, v),
            while_loop(
                lt(*i, arr.len()),
                (
                    // v = arr.get_unchecked(i)
                    // SAFETY: the surrounding loop proves `i < arr.len()`.
                    assign(*v, unsafe { arr.get_unchecked(*i) }),
                    // sum = if v > 5 then sum + v else sum
                    assign(
                        *sum,
                        if_then_else(
                            lt(5, *v), // v > 5
                            add(*sum, *v),
                            *sum,
                        ),
                    ),
                    assign(*i, add(*i, 1u64)),
                ),
            ),
            *sum,
        )
    });

    let compiled = compiler.compile(sum_gt_5).expect("compilation failed");
    let f = compiled.as_fn();

    // Test with array [0, 3, 5, 7, 2, 8, 1, 9, 4, 6]
    // Elements > 5: 7, 8, 9, 6 => sum = 30
    let data: [i64; 10] = [0, 3, 5, 7, 2, 8, 1, 9, 4, 6];
    let slice: &[i64] = &data;

    let result = f.call(slice);
    assert_eq!(result, 30); // 7 + 8 + 9 + 6 = 30
}

// =========================================================================
// Multi-Parameter Function Tests
// =========================================================================

#[test]
fn test_fun0_constant() {
    let mut compiler = Compiler::new();

    // Define: get_answer() = 42
    let get_answer = compiler.fun0("get_answer", |_ctx| Const::<i64>::new(42));

    let expr = call0(get_answer);
    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 42);
}

#[test]
fn test_fun2_add() {
    let mut compiler = Compiler::new();

    let add_fn = compiler.fun2("add", |_ctx, a: Var<i64>, b: Var<i64>| add(a, b));

    let compiled = compiler.compile(add_fn).expect("compilation failed");
    let add_ptr = compiled.as_fn();

    assert_eq!(add_ptr.call(10, 32), 42);
    assert_eq!(add_ptr.call(-5, 5), 0);
}

#[test]
fn test_fun3_clamp() {
    let mut compiler = Compiler::new();

    // clamp(x, min, max) = if x < min then min else (if x > max then max else x)
    let clamp_fn = compiler.fun3(
        "clamp",
        |_ctx, x: Var<i64>, min: Var<i64>, max: Var<i64>| {
            if_then_else(lt(x, min), min, if_then_else(lt(max, x), max, x))
        },
    );

    let compiled = compiler.compile(clamp_fn).expect("compilation failed");
    let clamp = compiled.as_fn();

    assert_eq!(clamp.call(-5, 0, 10), 0); // Clamped at min
    assert_eq!(clamp.call(5, 0, 10), 5); // In range
    assert_eq!(clamp.call(15, 0, 10), 10); // Clamped at max
}

#[test]
fn test_fun2_rec_gcd() {
    let mut compiler = Compiler::new();

    // Define: gcd(a, b) = if b == 0 then a else gcd(b, a % b)
    // Note: We'll use a different implementation since we don't have modulo
    // gcd(a, b) = if b == 0 then a else gcd(b, a - b * (a / b))
    let gcd = compiler.fun2_rec("gcd", |f, _ctx, a: Var<i64>, b: Var<i64>| {
        if_then_else(
            eq(b, 0i64),
            a,
            call2(f, b, sub(a, mul(b, div(a, b)))), // a % b = a - b * (a / b)
        )
    });

    // gcd(48, 18) = 6
    let expr = call2(gcd, 48i64, 18i64);
    let compiled = compiler.compile(expr).expect("compilation failed");
    assert_eq!(compiled.run(), 6);
}
