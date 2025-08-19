use dio::{execute_add_i64, execute_add_mixed_u64_i64, execute_add_u64, parse_expr};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Dio Mixed Type Support Demo");
    println!("==============================\n");

    // Demo 1: Pure U64Array addition
    println!("1️⃣ U64Array + U64Array -> U64Array");
    let expr_u64 = parse_expr("(lambda ([U64Array x] [U64Array y] U64Array) (+ x y))")?;
    println!("   Expression: {}", expr_u64);

    let a_u64 = vec![1u64, 2, 3];
    let b_u64 = vec![10u64, 20, 30];
    println!("   Input: {:?} + {:?}", a_u64, b_u64);

    let result_u64 = execute_add_u64(&expr_u64, &a_u64, &b_u64)?;
    println!("   Result: {:?}\n", result_u64);

    // Demo 2: Pure I64Array addition (with negative numbers)
    println!("2️⃣ I64Array + I64Array -> I64Array");
    let expr_i64 = parse_expr("(lambda ([I64Array x] [I64Array y] I64Array) (+ x y))")?;
    println!("   Expression: {}", expr_i64);

    let a_i64 = vec![-5i64, 10, -15];
    let b_i64 = vec![3i64, -7, 25];
    println!("   Input: {:?} + {:?}", a_i64, b_i64);

    let result_i64 = execute_add_i64(&expr_i64, &a_i64, &b_i64)?;
    println!("   Result: {:?}\n", result_i64);

    // Demo 3: Mixed type addition with automatic coercion
    println!("3️⃣ U64Array + I64Array -> I64Array (type coercion)");
    let expr_mixed = parse_expr("(lambda ([U64Array x] [I64Array y] I64Array) (+ x y))")?;
    println!("   Expression: {}", expr_mixed);

    let a_mixed_u64 = vec![100u64, 200, 300];
    let b_mixed_i64 = vec![-50i64, -75, -100];
    println!("   Input: {:?} + {:?}", a_mixed_u64, b_mixed_i64);

    let result_mixed = execute_add_mixed_u64_i64(&expr_mixed, &a_mixed_u64, &b_mixed_i64)?;
    println!("   Result: {:?}", result_mixed);
    println!("   Note: Unsigned values automatically cast to signed (SQL standard)\n");

    // Demo 4: Show type safety
    println!("4️⃣ Type Safety Demonstration");
    println!("   The following would fail at compile time:");
    println!("   ❌ (lambda ([U64Array x] [I64Array y] U64Array) (+ x y))");
    println!("   ✅ (lambda ([U64Array x] [I64Array y] I64Array) (+ x y))");
    println!(
        "   
   Type coercion rules (following SQL/DuckDB standards):
   • U64Array + U64Array -> U64Array
   • I64Array + I64Array -> I64Array  
   • U64Array + I64Array -> I64Array (signed takes precedence)
   • Mixed scalar/array types are rejected"
    );

    println!("\n🎉 All operations compiled to native machine code via Cranelift JIT!");

    Ok(())
}
