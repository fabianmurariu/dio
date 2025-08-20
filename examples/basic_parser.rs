use dio::parse_expr;

fn main() {
    println!("Dio Expression Parser Demo\n");

    let test_expressions = [
        "a",                                     // Simple column
        "42",                                    // Integer literal
        "3.14",                                  // Float literal
        "(+ a b)",                               // Simple addition
        "(+ a b c 10)",                          // N-ary addition
        "(- x y)",                               // Subtraction
        "(* price 1.1)",                         // Multiplication
        "(/ total count)",                       // Division
        "(sum sales)",                           // Sum reduction
        "(count records)",                       // Count reduction
        "(+ a (* b 2))",                         // Nested expression
        "(sum (+ price (* quantity discount)))", // Complex nested
        "(/ (sum revenue) (count customers))",   // Average pattern
        "(let tmp (+ a b))",                     // Let binding
    ];

    for expr_str in &test_expressions {
        match parse_expr(expr_str) {
            Ok(expr) => {
                println!("✅ Input:  {expr_str}");
                println!("   AST:    {expr:?}");
                println!("   Pretty: {expr}");

                // Show expression properties
                if expr.is_elementwise() {
                    println!("   Type:   Elementwise operation");
                } else if expr.is_reduction() {
                    println!("   Type:   Reduction operation");
                } else {
                    println!("   Type:   Scalar operation");
                }

                let columns = expr.get_column_references();
                if !columns.is_empty() {
                    println!("   Columns: {columns:?}");
                }

                println!("   Complexity: {}", expr.complexity());
                println!();
            }
            Err(e) => {
                println!("❌ Input: {expr_str}");
                println!("   Error: {e}");
                println!();
            }
        }
    }

    // Demonstrate error handling with problematic expressions
    println!("Error Handling Examples:\n");

    let error_expressions = [
        "(+ a b",        // Unbalanced parentheses
        "(- a)",         // Wrong argument count
        "(unknown a b)", // Unknown operation
        "3.14.15",       // Invalid number
        "()",            // Empty expression
    ];

    for expr_str in &error_expressions {
        match parse_expr(expr_str) {
            Ok(expr) => {
                println!("✅ Unexpected success for: {expr_str}");
                println!("   Result: {expr}");
            }
            Err(e) => {
                println!("❌ Input: {expr_str}");
                println!("   Error: {e}");

                // Show error span information
                println!("   Error type: Parse error");
            }
        }
        println!();
    }
}
