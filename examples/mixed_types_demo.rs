use dio::{
    array_support::{create_i64_array_from_vec, create_u64_array_from_vec},
    execute_generic_cached, parse_expr,
};
use arrow::array::{Int64Array, UInt64Array};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔢 Dio Mixed Type Support Demo");
    println!("==============================\n");

    // Demo 1: Pure U64Array addition
    println!("1️⃣ U64Array + U64Array -> U64Array");
    let expr_u64 = parse_expr("(lambda ([U64Array x] [U64Array y] U64Array) (+ x y))")?;
    println!("   Expression: {}", expr_u64);

    let a_u64_vec = vec![1u64, 2, 3];
    let b_u64_vec = vec![10u64, 20, 30];
    let a_u64 = create_u64_array_from_vec(a_u64_vec.clone())?;
    let b_u64 = create_u64_array_from_vec(b_u64_vec.clone())?;
    println!("   Input: {:?} + {:?}", a_u64_vec, b_u64_vec);

    let result_u64_array = execute_generic_cached(&expr_u64, &[a_u64, b_u64])?;
    let result_u64 = result_u64_array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or("Expected UInt64Array")?
        .values()
        .to_vec();
    println!("   Result: {:?}\n", result_u64);

    // Demo 2: Pure I64Array addition (with negative numbers)
    println!("2️⃣ I64Array + I64Array -> I64Array");
    let expr_i64 = parse_expr("(lambda ([I64Array x] [I64Array y] I64Array) (+ x y))")?;
    println!("   Expression: {}", expr_i64);

    let a_i64_vec = vec![-5i64, 10, -15];
    let b_i64_vec = vec![3i64, -7, 25];
    let a_i64 = create_i64_array_from_vec(a_i64_vec.clone())?;
    let b_i64 = create_i64_array_from_vec(b_i64_vec.clone())?;
    println!("   Input: {:?} + {:?}", a_i64_vec, b_i64_vec);

    let result_i64_array = execute_generic_cached(&expr_i64, &[a_i64, b_i64])?;
    let result_i64 = result_i64_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or("Expected Int64Array")?
        .values()
        .to_vec();
    println!("   Result: {:?}\n", result_i64);

    // Demo 3: Mixed type addition with automatic coercion
    println!("3️⃣ U64Array + I64Array -> I64Array (type coercion)");
    let expr_mixed = parse_expr("(lambda ([U64Array x] [I64Array y] I64Array) (+ x y))")?;
    println!("   Expression: {}", expr_mixed);

    let a_mixed_u64_vec = vec![100u64, 200, 300];
    let b_mixed_i64_vec = vec![-50i64, -75, -100];
    let a_mixed_u64 = create_u64_array_from_vec(a_mixed_u64_vec.clone())?;
    let b_mixed_i64 = create_i64_array_from_vec(b_mixed_i64_vec.clone())?;
    println!("   Input: {:?} + {:?}", a_mixed_u64_vec, b_mixed_i64_vec);

    let result_mixed_array = execute_generic_cached(&expr_mixed, &[a_mixed_u64, b_mixed_i64])?;
    let result_mixed = result_mixed_array
        .as_any()
        .downcast_ref::<Int64Array>()
        .ok_or("Expected Int64Array")?
        .values()
        .to_vec();
    println!("   Result: {:?}", result_mixed);
    println!("   Note: Unsigned values automatically cast to signed (SQL standard)\n");

    // Demo 4: Show type safety
    println!("4️⃣ Type Safety Demonstration");
    println!("   The following would fail at compile time:");
    println!("   ❌ (lambda ([U64Array x] [I64Array y] U64Array) (+ x y))");
    println!("   ✅ (lambda ([U64Array x] [I64Array y] I64Array) (+ x y))");
    println!(
        "   \n   Type coercion rules (following SQL/DuckDB standards):\n   • U64Array + U64Array -> U64Array\n   • I64Array + I64Array -> I64Array  \n   • U64Array + I64Array -> I64Array (signed takes precedence)\n   • Mixed scalar/array types are rejected"
    );

    println!("\n🎉 All operations compiled to native machine code via Cranelift JIT!");

    Ok(())
}

