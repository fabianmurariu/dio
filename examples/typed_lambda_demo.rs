use dio::{execute_generic_cached, parse_expr};
use dio::array_support::create_u64_array_from_vec;
use arrow::array::UInt64Array;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse a typed lambda expression
    let expr_str = "(lambda ([U64Array x] [U64Array y] U64Array) (+ x y))";
    println!("Parsing: {expr_str}");

    let expr = parse_expr(expr_str)?;
    println!("Parsed AST: {expr}");

    // Create some test data
    let x_vec = vec![1, 2, 3, 4, 5];
    let y_vec = vec![10, 20, 30, 40, 50];

    // Convert to Arrow arrays
    let x_array = create_u64_array_from_vec(x_vec.clone())?;
    let y_array = create_u64_array_from_vec(y_vec.clone())?;


    println!("Input arrays:");
    println!("  x = {x_vec:?}");
    println!("  y = {y_vec:?}");

    // Compile and execute
    println!("Compiling to machine code with Cranelift JIT...");
    let result_array = execute_generic_cached(&expr, &[x_array, y_array])?;

    // Convert result back to Vec<u64> for verification
    let result_u64 = result_array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or("Expected UInt64Array result")?;
    let result = result_u64.values().to_vec();

    println!("Result: {result:?}");
    println!("Expected: {:?}", vec![11, 22, 33, 44, 55]);

    assert_eq!(result, vec![11, 22, 33, 44, 55]);
    println!("✅ Success! The typed lambda compiled and executed correctly.");

    Ok(())
}
