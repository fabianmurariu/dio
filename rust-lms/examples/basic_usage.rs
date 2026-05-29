//! Basic usage example of the rust-lms library.
//!
//! This example demonstrates:
//! - Creating typed variables and constants using the Compiler
//! - Building type-safe expressions
//! - Compile-time type checking
//! - Copy semantics for VarRef<T> and Const<T>
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

        let expr = mul(add(five, Const::new(3)), two); // (5 + 3) * 2 = 16

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();
        println!("  (5 + 3) * 2 = {}", result);
        assert_eq!(result, 16);
        println!("  Compiled and ran successfully!\n");
    }

    // Example 2: Functions with VarRef
    println!("Example 2: Function definition and calling");
    {
        let mut compiler = Compiler::new();

        // Define: square(x) = x * x
        let square = compiler.fun1("square", |_ctx, x: Var<i64>| mul(x, x));

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

        // Define: double(x) = x + x
        let double = compiler.fun1("double", |_ctx, x: Var<i64>| add(x, x));

        // Define: add_one(x) = x + 1
        let add_one = compiler.fun1("add_one", |_ctx, x: Var<i64>| add(x, Const::new(1)));

        // Compute: double(add_one(5)) = double(6) = 12
        let expr = call1(double, call1(add_one, Const::<i64>::new(5)));

        let compiled = compiler.compile(expr).expect("compilation failed");
        let result = compiled.run();
        println!("  double(add_one(5)) = {}", result);
        assert_eq!(result, 12);
        println!("  Nested calls work!\n");
    }

    // Example 4: VarRef is Copy
    println!("Example 4: VarRef is Copy");
    {
        let mut compiler = Compiler::new();

        // x is used multiple times in the body - no clone needed!
        let cube = compiler.fun1("cube", |_ctx, x: Var<i64>| {
            mul(mul(x, x), x) // x * x * x
        });

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

        // Comparisons change type to Bool
        let comparison = lt(Const::<i64>::new(5), Const::new(10)); // 5 < 10 : BoolType

        // Note: We can compile and get a bool result
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

        // Define: f(x) = (x + 3) * (10 - x)
        let f = compiler.fun1("f", |_ctx, x: Var<i64>| {
            mul(add(x, Const::new(3)), sub(Const::new(10), x))
        });

        // f(2) = (2 + 3) * (10 - 2) = 5 * 8 = 40
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

        let expr = add(Const::new(1), Const::new(2));
        let _boxed_expr: Box<dyn Staged<Out = i64>> = expr.boxed();
        println!("  Can box operations too");
        println!("  Can store heterogeneous expression trees in collections\n");
    }

    println!("=== Summary ===");
    println!("All examples passed!");
    println!("Type-safe: Invalid operations caught at compile time");
    println!("Ergonomic: VarRef<T>/Const<T> are Copy");
    println!("Flexible: Supports type transformations (e.g., comparison -> bool)");
    println!("Composable: Any Staged value works anywhere");
    println!("Dynamic: Boxing support when needed");
    println!("JIT Compiled: Expressions compile to native code and run!");
}
