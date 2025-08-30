use crate::ast::{OrderedFloat64, Type};
use crate::{parse_expr, Expr, Value};
use proptest::prelude::*;

/// Test simple atomic expressions
#[test]
fn test_atoms() {
    // Column references
    assert_eq!(parse_expr("a").unwrap(), Expr::Column("a".to_string()));
    assert_eq!(
        parse_expr("column_name").unwrap(),
        Expr::Column("column_name".to_string())
    );
    assert_eq!(
        parse_expr("x123").unwrap(),
        Expr::Column("x123".to_string())
    );

    // Integer literals
    assert_eq!(parse_expr("0").unwrap(), Expr::Literal(Value::Int64(0)));
    assert_eq!(parse_expr("42").unwrap(), Expr::Literal(Value::Int64(42)));
    assert_eq!(parse_expr("-17").unwrap(), Expr::Literal(Value::Int64(-17)));
    assert_eq!(parse_expr("+99").unwrap(), Expr::Literal(Value::Int64(99)));

    // Float literals
    assert_eq!(
        parse_expr("3.141592653589793").unwrap(),
        Expr::Literal(Value::Float64(OrderedFloat64::from(std::f64::consts::PI)))
    );
    assert_eq!(
        parse_expr("-2.5").unwrap(),
        Expr::Literal(Value::Float64(OrderedFloat64::from(-2.5)))
    );
    assert_eq!(
        parse_expr("0.0").unwrap(),
        Expr::Literal(Value::Float64(OrderedFloat64::from(0.0)))
    );

    // Scientific notation
    assert_eq!(
        parse_expr("1e5").unwrap(),
        Expr::Literal(Value::Float64(OrderedFloat64::from(100000.0)))
    );
    assert_eq!(
        parse_expr("1.5e-3").unwrap(),
        Expr::Literal(Value::Float64(OrderedFloat64::from(0.0015)))
    );
    assert_eq!(
        parse_expr("2.5E+2").unwrap(),
        Expr::Literal(Value::Float64(OrderedFloat64::from(250.0)))
    );
}

/// Test basic binary operations
#[test]
fn test_basic_operations() {
    // Addition
    assert_eq!(
        parse_expr("(+ a b)").unwrap(),
        Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Column("b".to_string()),
        ])
    );

    // Subtraction
    assert_eq!(
        parse_expr("(- x 10)").unwrap(),
        Expr::Sub(
            Box::new(Expr::Column("x".to_string())),
            Box::new(Expr::Literal(Value::Int64(10))),
        )
    );

    // Multiplication
    assert_eq!(
        parse_expr("(* price 1.1)").unwrap(),
        Expr::Mul(vec![
            Expr::Column("price".to_string()),
            Expr::Literal(Value::Float64(OrderedFloat64::from(1.1))),
        ])
    );

    // Division
    assert_eq!(
        parse_expr("(/ total count)").unwrap(),
        Expr::Div(
            Box::new(Expr::Column("total".to_string())),
            Box::new(Expr::Column("count".to_string())),
        )
    );
}

/// Test n-ary operations (addition and multiplication)
#[test]
fn test_nary_operations() {
    // Multi-argument addition
    assert_eq!(
        parse_expr("(+ a b c d 10)").unwrap(),
        Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Column("b".to_string()),
            Expr::Column("c".to_string()),
            Expr::Column("d".to_string()),
            Expr::Literal(Value::Int64(10)),
        ])
    );

    // Multi-argument multiplication
    assert_eq!(
        parse_expr("(* x 2 y 3.5)").unwrap(),
        Expr::Mul(vec![
            Expr::Column("x".to_string()),
            Expr::Literal(Value::Int64(2)),
            Expr::Column("y".to_string()),
            Expr::Literal(Value::Float64(OrderedFloat64::from(3.5))),
        ])
    );
}

/// Test reduction operations
#[test]
fn test_reductions() {
    // Sum
    assert_eq!(
        parse_expr("(sum sales)").unwrap(),
        Expr::Sum(Box::new(Expr::Column("sales".to_string())))
    );

    // Count
    assert_eq!(
        parse_expr("(count records)").unwrap(),
        Expr::Count(Box::new(Expr::Column("records".to_string())))
    );

    // Sum of expression
    assert_eq!(
        parse_expr("(sum (+ price tax))").unwrap(),
        Expr::Sum(Box::new(Expr::Add(vec![
            Expr::Column("price".to_string()),
            Expr::Column("tax".to_string()),
        ])))
    );
}

/// Test let bindings (now support multiple typed bindings with reductive functions)
#[test]
fn test_let_bindings() {
    // Single typed let binding
    assert_eq!(
        parse_expr("(let [U64 s (sum a)] (+ s b))").unwrap(),
        Expr::Let {
            bindings: vec![(Type::U64, "s".to_string(), Expr::Sum(Box::new(Expr::Column("a".to_string()))))],
            body: Box::new(Expr::Add(vec![
                Expr::Column("s".to_string()),
                Expr::Column("b".to_string()),
            ]))
        }
    );
    
    // Multiple typed let bindings
    assert_eq!(
        parse_expr("(let [U64 s (sum a) U64 c (count b)] (+ s c))").unwrap(),
        Expr::Let {
            bindings: vec![
                (Type::U64, "s".to_string(), Expr::Sum(Box::new(Expr::Column("a".to_string())))),
                (Type::U64, "c".to_string(), Expr::Count(Box::new(Expr::Column("b".to_string())))),
            ],
            body: Box::new(Expr::Add(vec![
                Expr::Column("s".to_string()),
                Expr::Column("c".to_string()),
            ]))
        }
    );
    
    // Test that elementwise operations are rejected in let bindings
    assert!(parse_expr("(let [U64 tmp (+ a b)] (sum tmp))").is_err());
}

/// Test nested expressions
#[test]
fn test_nested_expressions() {
    // Simple nesting
    assert_eq!(
        parse_expr("(+ a (* b c))").unwrap(),
        Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Mul(vec![
                Expr::Column("b".to_string()),
                Expr::Column("c".to_string()),
            ]),
        ])
    );

    // Deep nesting
    assert_eq!(
        parse_expr("(* (+ a b) (- c d))").unwrap(),
        Expr::Mul(vec![
            Expr::Add(vec![
                Expr::Column("a".to_string()),
                Expr::Column("b".to_string()),
            ]),
            Expr::Sub(
                Box::new(Expr::Column("c".to_string())),
                Box::new(Expr::Column("d".to_string())),
            ),
        ])
    );

    // Very deep nesting
    assert_eq!(
        parse_expr("(sum (+ a (* b (/ c (- d 1)))))").unwrap(),
        Expr::Sum(Box::new(Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Mul(vec![
                Expr::Column("b".to_string()),
                Expr::Div(
                    Box::new(Expr::Column("c".to_string())),
                    Box::new(Expr::Sub(
                        Box::new(Expr::Column("d".to_string())),
                        Box::new(Expr::Literal(Value::Int64(1))),
                    )),
                ),
            ]),
        ])))
    );
}

/// Test complex real-world-like expressions
#[test]
fn test_complex_expressions() {
    // Financial calculation: total_with_tax = sum(price * (1 + tax_rate))
    assert_eq!(
        parse_expr("(sum (* price (+ 1 tax_rate)))").unwrap(),
        Expr::Sum(Box::new(Expr::Mul(vec![
            Expr::Column("price".to_string()),
            Expr::Add(vec![
                Expr::Literal(Value::Int64(1)),
                Expr::Column("tax_rate".to_string()),
            ]),
        ])))
    );

    // Average: sum(values) / count(values)
    assert_eq!(
        parse_expr("(/ (sum values) (count values))").unwrap(),
        Expr::Div(
            Box::new(Expr::Sum(Box::new(Expr::Column("values".to_string())))),
            Box::new(Expr::Count(Box::new(Expr::Column("values".to_string())))),
        )
    );

    // Weighted average: sum(value * weight) / sum(weight)
    assert_eq!(
        parse_expr("(/ (sum (* value weight)) (sum weight))").unwrap(),
        Expr::Div(
            Box::new(Expr::Sum(Box::new(Expr::Mul(vec![
                Expr::Column("value".to_string()),
                Expr::Column("weight".to_string()),
            ])))),
            Box::new(Expr::Sum(Box::new(Expr::Column("weight".to_string())))),
        )
    );

    // Compound expression with multiple operations
    assert_eq!(
        parse_expr("(+ (* base_salary overtime_rate) (- bonus deductions))").unwrap(),
        Expr::Add(vec![
            Expr::Mul(vec![
                Expr::Column("base_salary".to_string()),
                Expr::Column("overtime_rate".to_string()),
            ]),
            Expr::Sub(
                Box::new(Expr::Column("bonus".to_string())),
                Box::new(Expr::Column("deductions".to_string())),
            ),
        ])
    );
}

/// Test whitespace handling
#[test]
fn test_whitespace_handling() {
    let expressions = [
        "(+ a b)",         // Minimal spaces
        "( + a b )",       // Extra spaces
        "(\n+\n a\n b\n)", // Newlines
        "(\t+\ta\tb\t)",   // Tabs
        "  ( + a b )  ",   // Leading/trailing spaces
    ];

    let expected = Expr::Add(vec![
        Expr::Column("a".to_string()),
        Expr::Column("b".to_string()),
    ]);

    for expr_str in &expressions {
        assert_eq!(parse_expr(expr_str).unwrap(), expected);
    }
}

/// Test parse errors
#[test]
fn test_parse_errors() {
    // Unbalanced parentheses
    assert!(parse_expr("(+ a b").is_err());
    assert!(parse_expr("+ a b)").is_err());

    // Empty expressions
    assert!(parse_expr("()").is_err());
    assert!(parse_expr("").is_err());

    // Wrong argument counts
    assert!(parse_expr("(-)").is_err()); // Subtraction needs 2 args
    assert!(parse_expr("(- a)").is_err()); // Subtraction needs 2 args
    assert!(parse_expr("(- a b c)").is_err()); // Subtraction needs exactly 2 args

    assert!(parse_expr("(/)").is_err()); // Division needs 2 args
    assert!(parse_expr("(/ a)").is_err()); // Division needs 2 args
    assert!(parse_expr("(/ a b c)").is_err()); // Division needs exactly 2 args

    assert!(parse_expr("(sum)").is_err()); // Sum needs 1 arg
    assert!(parse_expr("(sum a b)").is_err()); // Sum needs exactly 1 arg

    assert!(parse_expr("(count)").is_err()); // Count needs 1 arg
    assert!(parse_expr("(count a b)").is_err()); // Count needs exactly 1 arg

    // Empty n-ary operations
    assert!(parse_expr("(+)").is_err()); // Addition needs at least 1 arg
    assert!(parse_expr("(*)").is_err()); // Multiplication needs at least 1 arg

    // Invalid tokens
    assert!(parse_expr("(+ a @)").is_err()); // Invalid identifier
    assert!(parse_expr("(unknown a b)").is_err()); // Unknown operation

    // Invalid numbers
    assert!(parse_expr("std::f64::consts::PI.15").is_err()); // Multiple decimal points

    // Trailing tokens
    assert!(parse_expr("(+ a b) extra").is_err());
}

/// Test AST utility methods
#[test]
fn test_ast_utilities() {
    let expr1 = Expr::Add(vec![
        Expr::Column("a".to_string()),
        Expr::Column("b".to_string()),
    ]);
    assert!(expr1.is_elementwise());
    assert!(!expr1.is_reduction());
    assert_eq!(expr1.get_column_references(), vec!["a", "b"]);

    let expr2 = Expr::Sum(Box::new(Expr::Column("x".to_string())));
    assert!(!expr2.is_elementwise());
    assert!(expr2.is_reduction());
    assert_eq!(expr2.get_column_references(), vec!["x"]);

    // Test complexity calculation
    assert_eq!(Expr::Column("a".to_string()).complexity(), 1);
    assert_eq!(expr1.complexity(), 3); // 1 for Add + 1 for each column
    assert_eq!(expr2.complexity(), 3); // 2 for Sum + 1 for column
}

/// Test Display formatting
#[test]
fn test_display_formatting() {
    assert_eq!(Expr::Column("test".to_string()).to_string(), "test");
    assert_eq!(Expr::Literal(Value::Int64(42)).to_string(), "42");
    assert_eq!(
        Expr::Literal(Value::Float64(OrderedFloat64::from(std::f64::consts::PI))).to_string(),
        "3.141592653589793"
    );

    let expr = Expr::Add(vec![
        Expr::Column("a".to_string()),
        Expr::Literal(Value::Int64(1)),
        Expr::Column("b".to_string()),
    ]);
    assert_eq!(expr.to_string(), "(+ a 1 b)");

    let nested = Expr::Sum(Box::new(Expr::Mul(vec![
        Expr::Column("x".to_string()),
        Expr::Literal(Value::Float64(OrderedFloat64::from(2.0))),
    ])));
    assert_eq!(nested.to_string(), "(sum (* x 2))");
}

// Property-based testing with proptest
prop_compose! {
    fn arb_identifier()(s in "[a-zA-Z][a-zA-Z0-9_]{0,10}") -> String { s }
}

prop_compose! {
    fn arb_value()(
        int_val in any::<i64>(),
        float_val in any::<f64>().prop_filter("reasonable_finite", |f|
            f.is_finite() && f.abs() < 1e15), // Avoid extreme values that cause display issues
        is_int in any::<bool>()
    ) -> Value {
        if is_int {
            Value::Int64(int_val)
        } else {
            Value::Float64(OrderedFloat64::from(float_val))
        }
    }
}

fn arb_expr_leaf() -> impl Strategy<Value = Expr> {
    prop_oneof![
        arb_identifier().prop_map(Expr::Column),
        arb_value().prop_map(Expr::Literal),
    ]
}

fn arb_expr_recursive(depth: u32) -> impl Strategy<Value = Expr> {
    if depth == 0 {
        arb_expr_leaf().boxed()
    } else {
        prop_oneof![
            // Leaf nodes
            arb_expr_leaf(),
            // Binary operations
            (arb_expr_recursive(depth - 1), arb_expr_recursive(depth - 1))
                .prop_map(|(l, r)| Expr::Sub(Box::new(l), Box::new(r))),
            (arb_expr_recursive(depth - 1), arb_expr_recursive(depth - 1))
                .prop_map(|(l, r)| Expr::Div(Box::new(l), Box::new(r))),
            // N-ary operations
            prop::collection::vec(arb_expr_recursive(depth - 1), 1..5).prop_map(Expr::Add),
            prop::collection::vec(arb_expr_recursive(depth - 1), 1..5).prop_map(Expr::Mul),
            // Unary operations
            arb_expr_recursive(depth - 1).prop_map(|e| Expr::Sum(Box::new(e))),
            arb_expr_recursive(depth - 1).prop_map(|e| Expr::Count(Box::new(e))),
        ]
        .boxed()
    }
}

proptest! {
    #[test]
    fn test_parse_roundtrip(expr in arb_expr_recursive(3)) {
        let formatted = expr.to_string();
        let parsed = parse_expr(&formatted);
        prop_assert!(parsed.is_ok(), "Failed to parse generated expression: {}", formatted);

        // Note: Float64(0.0) displays as "0" and parses back as Int64(0)
        // This is expected behavior - both represent the same mathematical value
        let parsed_expr = parsed.unwrap();
        if expr != parsed_expr {
            // Check if it's just a Float64(x.0) vs Int64(x) difference for whole numbers
            prop_assert!(expr_values_equivalent(&expr, &parsed_expr),
                "Expressions not equivalent:\nOriginal: {:?}\nParsed: {:?}\nFormatted: {}",
                expr, parsed_expr, formatted);
        }
    }

    #[test]
    fn test_column_references_consistency(expr in arb_expr_recursive(2)) {
        let columns = expr.get_column_references();
        // All returned column names should be unique
        let mut sorted_columns = columns.clone();
        sorted_columns.sort();
        sorted_columns.dedup();
        prop_assert_eq!(columns, sorted_columns);

        // Complexity should be at least the number of nodes
        prop_assert!(expr.complexity() >= 1);
    }
}

/// Check if two expressions are mathematically equivalent
/// (handles Float64 vs Int64 for whole numbers)
fn expr_values_equivalent(a: &Expr, b: &Expr) -> bool {
    match (a, b) {
        (Expr::Literal(v1), Expr::Literal(v2)) => values_equivalent(v1, v2),
        (Expr::Column(n1), Expr::Column(n2)) => n1 == n2,
        (Expr::Add(ops1), Expr::Add(ops2)) | (Expr::Mul(ops1), Expr::Mul(ops2)) => {
            ops1.len() == ops2.len()
                && ops1
                    .iter()
                    .zip(ops2.iter())
                    .all(|(e1, e2)| expr_values_equivalent(e1, e2))
        }
        (Expr::Sub(l1, r1), Expr::Sub(l2, r2)) | (Expr::Div(l1, r1), Expr::Div(l2, r2)) => {
            expr_values_equivalent(l1, l2) && expr_values_equivalent(r1, r2)
        }
        (Expr::Sum(e1), Expr::Sum(e2)) | (Expr::Count(e1), Expr::Count(e2)) => {
            expr_values_equivalent(e1, e2)
        }
        (Expr::Let { bindings: b1, body: e1 }, Expr::Let { bindings: b2, body: e2 }) => {
            b1.len() == b2.len() 
                && b1.iter().zip(b2.iter()).all(|((t1, n1, be1), (t2, n2, be2))| t1 == t2 && n1 == n2 && expr_values_equivalent(be1, be2))
                && expr_values_equivalent(e1, e2)
        }
        _ => false,
    }
}

/// Check if two values are mathematically equivalent
fn values_equivalent(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int64(i1), Value::Int64(i2)) => i1 == i2,
        (Value::Float64(f1), Value::Float64(f2)) => (f1.0 - f2.0).abs() < f64::EPSILON,
        (Value::Int64(i), Value::Float64(f)) | (Value::Float64(f), Value::Int64(i)) => {
            (*i as f64 - f.0).abs() < f64::EPSILON
        }
    }
}
