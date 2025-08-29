use std::fmt;

/// Type definitions for the typed Lisp
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Type {
    /// U64 scalar type
    U64,
    /// U64 array type
    U64Array,
    /// I64 scalar type
    I64,
    /// I64 array type
    I64Array,
    /// F64 scalar type (future extension)
    F64,
    /// F64 array type (future extension)
    F64Array,
}

impl Type {
    pub fn is_integer(&self) -> bool {
        matches!(self, Type::U64 | Type::I64 | Type::U64Array | Type::I64Array)
    }

    pub fn is_i64(&self) -> bool {
        matches!(self, Type::I64 | Type::I64Array)
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Type::U64Array | Type::I64Array | Type::F64Array)
    }

    pub fn is_scalar(&self) -> bool {
        !self.is_array()
    }
}

/// Typed parameter for lambda expressions
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct TypedParam {
    pub name: String,
    pub type_: Type,
}

/// Abstract Syntax Tree for Dio expressions
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Expr {
    /// Column reference: "a", "b", "x"
    Column(String),

    /// Literal constants: 42, std::f64::consts::PI
    Literal(Value),

    /// Addition: (+ a b c) - supports n-ary operations
    Add(Vec<Expr>),

    /// Subtraction: (- x y) - binary operation
    Sub(Box<Expr>, Box<Expr>),

    /// Multiplication: (* a b c) - supports n-ary operations
    Mul(Vec<Expr>),

    /// Division: (/ x y) - binary operation
    Div(Box<Expr>, Box<Expr>),

    /// Sum reduction: (sum (+ a b)) - reduces array to scalar
    Sum(Box<Expr>),

    /// Count reduction: (count a) - counts non-null values
    Count(Box<Expr>),

    /// Variable binding: (let [var_name binding_expr] body_expr)
    /// Binding expressions must be reductive functions like (sum a) or (count a)
    Let {
        var_name: String,
        binding: Box<Expr>,
        body: Box<Expr>,
    },

    /// Typed lambda expression: (lambda ([U64Array x] [U64Array y] U64Array) (+ x y))
    Lambda {
        params: Vec<TypedParam>,
        return_type: Type,
        body: Box<Expr>,
    },
}

/// Literal values that can appear in expressions
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum Value {
    Int64(i64),
    Float64(OrderedFloat64),
}

/// Wrapper for f64 to enable Eq and Hash traits
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct OrderedFloat64(pub f64);

impl Eq for OrderedFloat64 {}

impl std::hash::Hash for OrderedFloat64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl From<f64> for OrderedFloat64 {
    fn from(f: f64) -> Self {
        OrderedFloat64(f)
    }
}

impl From<OrderedFloat64> for f64 {
    fn from(f: OrderedFloat64) -> Self {
        f.0
    }
}

impl fmt::Display for Type {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Type::U64 => write!(f, "U64"),
            Type::U64Array => write!(f, "U64Array"),
            Type::I64 => write!(f, "I64"),
            Type::I64Array => write!(f, "I64Array"),
            Type::F64 => write!(f, "F64"),
            Type::F64Array => write!(f, "F64Array"),
        }
    }
}

impl fmt::Display for TypedParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{} {}]", self.type_, self.name)
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int64(i) => write!(f, "{i}"),
            Value::Float64(OrderedFloat64(fl)) => {
                // Check if it's a whole number that fits in i64 range
                if fl.fract() == 0.0
                    && fl.is_finite()
                    && *fl >= i64::MIN as f64
                    && *fl <= i64::MAX as f64
                {
                    write!(f, "{}", *fl as i64)
                } else {
                    write!(f, "{fl}")
                }
            }
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Column(name) => write!(f, "{name}"),
            Expr::Literal(value) => write!(f, "{value}"),
            Expr::Add(operands) => {
                write!(f, "(+")?;
                for operand in operands {
                    write!(f, " {operand}")?;
                }
                write!(f, ")")
            }
            Expr::Sub(lhs, rhs) => write!(f, "(- {lhs} {rhs})"),
            Expr::Mul(operands) => {
                write!(f, "(*")?;
                for operand in operands {
                    write!(f, " {operand}")?;
                }
                write!(f, ")")
            }
            Expr::Div(lhs, rhs) => write!(f, "(/ {lhs} {rhs})"),
            Expr::Sum(expr) => write!(f, "(sum {expr})"),
            Expr::Count(expr) => write!(f, "(count {expr})"),
            Expr::Let { var_name, binding, body } => {
                write!(f, "(let [{var_name} {binding}] {body})")
            }
            Expr::Lambda {
                params,
                return_type,
                body,
            } => {
                write!(f, "(lambda (")?;
                for (i, param) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, " ")?;
                    }
                    write!(f, "{param}")?;
                }
                write!(f, " {return_type}) {body})")
            }
        }
    }
}

impl Expr {
    /// Returns true if this expression produces an array output (elementwise operations)
    pub fn is_elementwise(&self) -> bool {
        match self {
            Expr::Column(_) => true,
            Expr::Literal(_) => false, // Scalars
            Expr::Add(_) | Expr::Sub(_, _) | Expr::Mul(_) | Expr::Div(_, _) => true,
            Expr::Sum(_) | Expr::Count(_) => false, // Reductions produce scalars
            Expr::Let { body, .. } => body.is_elementwise(),
            Expr::Lambda { return_type, .. } => matches!(
                return_type,
                Type::U64Array | Type::I64Array | Type::F64Array
            ),
        }
    }

    /// Returns true if this expression is a reduction operation
    pub fn is_reduction(&self) -> bool {
        match self {
            Expr::Sum(_) | Expr::Count(_) => true,
            Expr::Lambda { return_type, .. } => {
                matches!(return_type, Type::U64 | Type::I64 | Type::F64)
            }
            _ => false,
        }
    }

    /// Get all column references in this expression
    pub fn get_column_references(&self) -> Vec<&str> {
        let mut columns = Vec::new();
        self.collect_columns(&mut columns);
        columns.sort();
        columns.dedup();
        columns
    }

    fn collect_columns<'a>(&'a self, columns: &mut Vec<&'a str>) {
        match self {
            Expr::Column(name) => columns.push(name),
            Expr::Literal(_) => {}
            Expr::Add(operands) | Expr::Mul(operands) => {
                for operand in operands {
                    operand.collect_columns(columns);
                }
            }
            Expr::Sub(lhs, rhs) | Expr::Div(lhs, rhs) => {
                lhs.collect_columns(columns);
                rhs.collect_columns(columns);
            }
            Expr::Sum(expr) | Expr::Count(expr) => {
                expr.collect_columns(columns);
            }
            Expr::Let { binding, body, .. } => {
                binding.collect_columns(columns);
                body.collect_columns(columns);
            }
            Expr::Lambda { body, .. } => {
                body.collect_columns(columns);
            }
        }
    }

    /// Estimate the complexity of this expression (for optimization decisions)
    pub fn complexity(&self) -> usize {
        match self {
            Expr::Column(_) | Expr::Literal(_) => 1,
            Expr::Add(operands) | Expr::Mul(operands) => {
                1 + operands.iter().map(|e| e.complexity()).sum::<usize>()
            }
            Expr::Sub(lhs, rhs) | Expr::Div(lhs, rhs) => 1 + lhs.complexity() + rhs.complexity(),
            Expr::Sum(expr) | Expr::Count(expr) => {
                2 + expr.complexity() // Reductions are more expensive
            }
            Expr::Let { binding, body, .. } => 1 + binding.complexity() + body.complexity(),
            Expr::Lambda { body, .. } => 1 + body.complexity(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expr_display() {
        let expr = Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Literal(Value::Int64(42)),
            Expr::Column("b".to_string()),
        ]);
        assert_eq!(expr.to_string(), "(+ a 42 b)");
    }

    #[test]
    fn test_nested_expr_display() {
        let expr = Expr::Sum(Box::new(Expr::Add(vec![
            Expr::Column("x".to_string()),
            Expr::Mul(vec![
                Expr::Column("y".to_string()),
                Expr::Literal(Value::Float64(std::f64::consts::PI.into())),
            ]),
        ])));
        assert_eq!(expr.to_string(), "(sum (+ x (* y 3.141592653589793)))");
    }

    #[test]
    fn test_is_elementwise() {
        assert!(Expr::Column("a".to_string()).is_elementwise());
        assert!(Expr::Add(vec![Expr::Column("a".to_string())]).is_elementwise());
        assert!(!Expr::Sum(Box::new(Expr::Column("a".to_string()))).is_elementwise());
        assert!(!Expr::Literal(Value::Int64(42)).is_elementwise());
    }

    #[test]
    fn test_is_reduction() {
        assert!(Expr::Sum(Box::new(Expr::Column("a".to_string()))).is_reduction());
        assert!(Expr::Count(Box::new(Expr::Column("a".to_string()))).is_reduction());
        assert!(!Expr::Add(vec![Expr::Column("a".to_string())]).is_reduction());
    }

    #[test]
    fn test_get_column_references() {
        let expr = Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Sub(
                Box::new(Expr::Column("b".to_string())),
                Box::new(Expr::Column("a".to_string())), // Duplicate should be deduped
            ),
            Expr::Column("c".to_string()),
        ]);

        let mut columns = expr.get_column_references();
        columns.sort();
        assert_eq!(columns, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_complexity() {
        assert_eq!(Expr::Column("a".to_string()).complexity(), 1);
        assert_eq!(Expr::Literal(Value::Int64(42)).complexity(), 1);

        let simple_add = Expr::Add(vec![
            Expr::Column("a".to_string()),
            Expr::Column("b".to_string()),
        ]);
        assert_eq!(simple_add.complexity(), 3); // 1 for Add + 1 for each column

        let sum_expr = Expr::Sum(Box::new(Expr::Column("a".to_string())));
        assert_eq!(sum_expr.complexity(), 3); // 2 for Sum + 1 for column
    }

    #[test]
    fn test_ordered_float64() {
        let f1 = OrderedFloat64(std::f64::consts::PI);
        let f2 = OrderedFloat64(std::f64::consts::PI);
        let f3 = OrderedFloat64(2.71);

        assert_eq!(f1, f2);
        assert_ne!(f1, f3);

        // Test hash consistency
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher1 = DefaultHasher::new();
        let mut hasher2 = DefaultHasher::new();
        f1.hash(&mut hasher1);
        f2.hash(&mut hasher2);
        assert_eq!(hasher1.finish(), hasher2.finish());
    }
}