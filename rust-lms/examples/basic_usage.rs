//! Basic usage example of the rust-lms library.
//!
//! This example demonstrates:
//! - Creating typed variables and constants
//! - Building type-safe expressions
//! - Compile-time type checking
//! - Copy semantics for Var<T> and Const<T>

use cranelift_frontend::Variable;
use rust_lms::prelude::*;

fn main() {
    println!("=== rust-lms: Type-Safe Staged Computation ===\n");

    // Example 1: Basic arithmetic with Copy semantics
    println!("Example 1: Copy semantics");
    let x = Var::<I64Type>::new(Variable::from_u32(0));
    let five = Const::<I64Type>::new(5);
    let two = Const::<I64Type>::new(2);

    // x is Copy, so we can use it multiple times without cloning!
    let expr1 = add(x, five); // (x + 5)
    let expr2 = mul(expr1, two); // (x + 5) * 2
    let expr3 = add(x, x); // x + x - no clone needed!

    println!("  Created expression: (x + 5) * 2");
    println!("  Created expression: x + x");
    println!("  ✓ x was used multiple times without explicit cloning\n");

    // Example 2: Heterogeneous operations (type transformations)
    println!("Example 2: Heterogeneous operations");
    let comparison1 = lt(x, Const::new(100)); // x < 100 : BoolType
    let comparison2 = eq(add(x, five), Const::new(10)); // (x + 5) == 10 : BoolType

    println!("  Created: x < 100 -> BoolType");
    println!("  Created: (x + 5) == 10 -> BoolType");
    println!("  ✓ Comparisons change output type from I64Type to BoolType\n");

    // Example 3: Compile-time type safety
    println!("Example 3: Compile-time type safety");
    println!("  The following would NOT compile:");
    println!("    // let bad = add(x, comparison1);");
    println!("    // ERROR: cannot add I64Type and BoolType");
    println!("  ✓ Type errors caught at compile time!\n");

    // Example 4: Multiple numeric types
    println!("Example 4: Multiple numeric types");
    let u_val = Var::<U64Type>::new(Variable::from_u32(1));
    let f_val = Var::<F64Type>::new(Variable::from_u32(2));

    let u_expr = mul(u_val, Const::<U64Type>::new(10));
    let f_expr = div(f_val, Const::<F64Type>::new(2.0));

    println!("  Created U64 expression: u_val * 10");
    println!("  Created F64 expression: f_val / 2.0");
    println!("  ✓ Different types work independently\n");

    // Example 5: Complex nested expressions
    println!("Example 5: Complex nested expressions");
    let complex = mul(
        add(x, Const::new(3)),
        sub(Const::new(10), x),
    ); // (x + 3) * (10 - x)

    println!("  Created: (x + 3) * (10 - x)");
    println!("  ✓ Arbitrary nesting works seamlessly\n");

    // Example 6: Boxing for dynamic dispatch
    println!("Example 6: Dynamic dispatch via boxing");
    let boxed: Box<dyn Staged<Out = I64Type>> = x.boxed();
    println!("  Boxed Var<I64Type> as Box<dyn Staged<Out = I64Type>>");
    println!("  ✓ Can store heterogeneous expression trees in collections\n");

    println!("=== Summary ===");
    println!("✓ Type-safe: Invalid operations caught at compile time");
    println!("✓ Ergonomic: Var<T>/Const<T> are Copy when possible");
    println!("✓ Flexible: Supports type transformations (e.g., comparison → bool)");
    println!("✓ Composable: Any Staged value works anywhere");
    println!("✓ Dynamic: Boxing support when needed");

    // Note: We're not actually generating code here, just demonstrating
    // the type system and API ergonomics. In a real use case, you'd call
    // .codegen() with a FunctionBuilder to generate actual Cranelift IR.
}