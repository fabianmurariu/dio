//! Basic usage example of the rust-lms library.
//!
//! This example demonstrates:
//! - Creating typed variables and constants using the Compiler
//! - Building type-safe expressions with std::ops (`+`, `-`, `*`, `/`)
//! - Compile-time type checking
//! - Copy semantics for `Var<T>` and `Const<T>`
//! - Compiling and running expressions

use rust_lms::prelude::*;

fn main() {
    println!("=== rust-lms: Type-Safe Staged Computation ===\n");

    // Example 1: Basic arithmetic with constants
    println!("Example 1: Simple arithmetic");
    {
        let compiler = Compiler::new();
        let five = Const::<i64>::new(5);
        let two = Const::<i64>::new(2);

        // Operators work directly on Const<T> via std::ops impls.
        let expr = (five + Const::<i64>::new(3)) * two; // (5 + 3) * 2 = 16

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();
        println!("  (5 + 3) * 2 = {}", result);
        assert_eq!(result, 16);
        println!("  Compiled and ran successfully!\n");
    }

    // Example 2: Functions with Var
    println!("Example 2: Function definition and calling");
    {
        let mut compiler = Compiler::new();

        // Define: square(x) = x * x
        let square = compiler.fun1("square", |_ctx, x: Var<i64>| x * x);

        // Call: square(7) = 49
        let expr = call1(square, Const::<i64>::new(7));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();
        println!("  square(7) = {}", result);
        assert_eq!(result, 49);
        println!("  Function compiled and called successfully!\n");
    }

    // Example 3: Nested function calls
    println!("Example 3: Nested function calls");
    {
        let mut compiler = Compiler::new();

        let double = compiler.fun1("double", |_ctx, x: Var<i64>| x + x);
        let add_one = compiler.fun1("add_one", |_ctx, x: Var<i64>| x + 1i64);

        // Compute: double(add_one(5)) = double(6) = 12
        let expr = call1(double, call1(add_one, Const::<i64>::new(5)));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();
        println!("  double(add_one(5)) = {}", result);
        assert_eq!(result, 12);
        println!("  Nested calls work!\n");
    }

    // Example 4: Var is Copy
    println!("Example 4: Var is Copy");
    {
        let mut compiler = Compiler::new();

        // x is used multiple times in the body - no clone needed.
        let cube = compiler.fun1("cube", |_ctx, x: Var<i64>| x * x * x);

        let expr = call1(cube, Const::<i64>::new(3));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();
        println!("  cube(3) = {} (x used 3 times, no cloning)", result);
        assert_eq!(result, 27);
        println!("  Copy semantics confirmed!\n");
    }

    // Example 5: Heterogeneous operations (type transformations)
    println!("Example 5: Heterogeneous operations");
    {
        let compiler = Compiler::new();

        // Comparisons return BoolType (cannot use `<` since PartialOrd::lt
        // returns a runtime bool, not a staged BoolType).
        let comparison = lt(Const::<i64>::new(5), Const::<i64>::new(10));

        let compiled = compiler.compile(comparison).expect("compilation failed");
        let result = compiled.run();
        println!("  5 < 10 = {} (returns BoolType)", result);
        assert!(result);
        println!("  Type transformation works!\n");
    }

    // Example 6: Complex nested expressions
    println!("Example 6: Complex nested expressions");
    {
        let mut compiler = Compiler::new();

        // f(x) = (x + 3) * (10 - x)
        let f = compiler.fun1("f", |_ctx, x: Var<i64>| (x + 3i64) * (Const::<i64>::new(10) - x));

        let expr = call1(f, Const::<i64>::new(2));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();
        println!("  f(2) where f(x) = (x + 3) * (10 - x)");
        println!("  = (2 + 3) * (10 - 2) = 5 * 8 = {}", result);
        assert_eq!(result, 40);
        println!("  Complex expressions work!\n");
    }

    // Example 7: Boxing for dynamic dispatch
    println!("Example 7: Dynamic dispatch via boxing");
    {
        let c = Const::<i64>::new(42);
        let _boxed: Box<dyn Staged<Out = i64>> = c.boxed();
        println!("  Boxed Const<i64> as Box<dyn Staged<Out = i64>>");

        let expr = Const::<i64>::new(1) + Const::<i64>::new(2);
        let _boxed_expr: Box<dyn Staged<Out = i64>> = expr.boxed();
        println!("  Can box operations too");
        println!("  Can store heterogeneous expression trees in collections\n");
    }

    println!("=== Summary ===");
    println!("All examples passed!");
    println!("Type-safe: invalid operations caught at compile time");
    println!("Ergonomic: Var/Const are Copy; std::ops works on them directly");
    println!("Flexible: type transformations (e.g. comparisons return BoolType)");
    println!("Composable: any Staged value works anywhere");
    println!("Dynamic: boxing when needed");
    println!("JIT-compiled: expressions compile to native code and run");
}
