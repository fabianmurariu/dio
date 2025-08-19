use dio::{parse_expr, execute_add_u64};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse a typed lambda expression
    let expr_str = "(lambda ([U64Array x] [U64Array y] U64Array) (+ x y))";
    println!("Parsing: {}", expr_str);
    
    let expr = parse_expr(expr_str)?;
    println!("Parsed AST: {}", expr);
    
    // Create some test data
    let x = vec![1, 2, 3, 4, 5];
    let y = vec![10, 20, 30, 40, 50];
    
    println!("Input arrays:");
    println!("  x = {:?}", x);
    println!("  y = {:?}", y);
    
    // Compile and execute
    println!("Compiling to machine code with Cranelift JIT...");
    let result = execute_add_u64(&expr, &x, &y)?;
    
    println!("Result: {:?}", result);
    println!("Expected: {:?}", vec![11, 22, 33, 44, 55]);
    
    assert_eq!(result, vec![11, 22, 33, 44, 55]);
    println!("✅ Success! The typed lambda compiled and executed correctly.");
    
    Ok(())
}