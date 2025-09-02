use crate::ast::Type;
use crate::error::DioError;

/// Casting rules following SQL/DuckDB standards
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CastKind {
    None,
    Implicit,
    Explicit,
    Impossible,
}

impl Type {
    /// Check if this type can be cast to another type and what kind of cast is needed
    pub fn cast_to(&self, target: &Type) -> CastKind {
        use Type::*;
        match (self, target) {
            (a, b) if a == b => CastKind::None,
            (U64Array, I64Array) | (I64Array, U64Array) => CastKind::Implicit,
            (U64, I64) | (I64, U64) => CastKind::Implicit,
            (F64, F64Array) | (F64Array, F64) => CastKind::Explicit,
            (U64 | I64, F64) => CastKind::Explicit,
            (F64, U64 | I64) => CastKind::Explicit,
            (U64Array | I64Array, F64Array) => CastKind::Explicit,
            (F64Array, U64Array | I64Array) => CastKind::Explicit,
            (U64 | I64 | F64, U64Array | I64Array | F64Array) => CastKind::Impossible,
            (U64Array | I64Array | F64Array, U64 | I64 | F64) => CastKind::Impossible,
            _ => CastKind::Explicit,
        }
    }

    pub fn can_implicit_cast_to(&self, target: &Type) -> bool {
        matches!(self.cast_to(target), CastKind::None | CastKind::Implicit)
    }

    pub fn can_explicit_cast_to(&self, target: &Type) -> bool {
        !matches!(self.cast_to(target), CastKind::Impossible)
    }

    pub fn element_type(&self) -> Option<Type> {
        match self {
            Type::U64Array => Some(Type::U64),
            Type::I64Array => Some(Type::I64),
            Type::F64Array => Some(Type::F64),
            _ => None,
        }
    }

    pub fn array_type(&self) -> Option<Type> {
        match self {
            Type::U64 => Some(Type::U64Array),
            Type::I64 => Some(Type::I64Array),
            Type::F64 => Some(Type::F64Array),
            _ => None,
        }
    }
}

/// Type coercion for N-ary operations following SQL rules
pub fn coerce_nary_op_types(operand_types: &[Type]) -> Result<Type, DioError> {
    if operand_types.is_empty() {
        return Err(DioError::TypeMismatch {
            expected: "at least one operand".to_string(),
            found: "no operands".to_string(),
            context: "N-ary operations require at least one operand".to_string(),
        });
    }

    let mut result_type = operand_types[0].clone();
    for operand_type in &operand_types[1..] {
        result_type = coerce_binary_op_types(&result_type, operand_type)?;
    }
    Ok(result_type)
}

/// Type coercion for binary operations following SQL rules
pub fn coerce_binary_op_types(left: &Type, right: &Type) -> Result<Type, DioError> {
    use Type::*;
    match (left, right) {
        (a, b) if a == b => Ok(a.clone()),
        (U64Array, I64Array) | (I64Array, U64Array) => Ok(I64Array),
        (U64, I64) | (I64, U64) => Ok(I64),
        (scalar, array) if scalar.is_scalar() && array.is_array() => Err(DioError::TypeMismatch {
            expected: "compatible types".to_string(),
            found: format!("{scalar} and {array}"),
            context: "Cannot mix scalar and array types in binary operations".to_string(),
        }),
        (array, scalar) if array.is_array() && scalar.is_scalar() => Err(DioError::TypeMismatch {
            expected: "compatible types".to_string(),
            found: format!("{array} and {scalar}"),
            context: "Cannot mix scalar and array types in binary operations".to_string(),
        }),
        _ => Err(DioError::TypeMismatch {
            expected: "compatible types".to_string(),
            found: format!("{left} and {right}"),
            context: "Unsupported type combination for binary operation".to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_same_type_casting() {
        assert_eq!(Type::U64Array.cast_to(&Type::U64Array), CastKind::None);
        assert_eq!(Type::I64.cast_to(&Type::I64), CastKind::None);
    }

    #[test]
    fn test_implicit_casting() {
        assert_eq!(Type::U64Array.cast_to(&Type::I64Array), CastKind::Implicit);
        assert_eq!(Type::I64Array.cast_to(&Type::U64Array), CastKind::Implicit);
        assert_eq!(Type::U64.cast_to(&Type::I64), CastKind::Implicit);
        assert_eq!(Type::I64.cast_to(&Type::U64), CastKind::Implicit);
    }

    #[test]
    fn test_impossible_casting() {
        assert_eq!(Type::U64.cast_to(&Type::U64Array), CastKind::Impossible);
        assert_eq!(Type::I64Array.cast_to(&Type::I64), CastKind::Impossible);
    }

    #[test]
    fn test_type_properties() {
        assert!(Type::U64Array.is_array());
        assert!(Type::I64.is_scalar());
        assert!(Type::U64.is_integer());
    }

    #[test]
    fn test_binary_coercion() {
        assert_eq!(coerce_binary_op_types(&Type::U64Array, &Type::U64Array).unwrap(), Type::U64Array);
        assert_eq!(coerce_binary_op_types(&Type::U64Array, &Type::I64Array).unwrap(), Type::I64Array);
        assert_eq!(coerce_binary_op_types(&Type::U64, &Type::I64).unwrap(), Type::I64);
    }

    #[test]
    fn test_binary_coercion_errors() {
        assert!(coerce_binary_op_types(&Type::U64, &Type::U64Array).is_err());
        assert!(coerce_binary_op_types(&Type::I64Array, &Type::I64).is_err());
    }

    #[test]
    fn test_nary_coercion() {
        assert_eq!(coerce_nary_op_types(&[Type::U64Array]).unwrap(), Type::U64Array);
        assert_eq!(coerce_nary_op_types(&[Type::U64Array, Type::I64Array]).unwrap(), Type::I64Array);
        assert_eq!(coerce_nary_op_types(&[Type::U64Array, Type::U64Array, Type::I64Array]).unwrap(), Type::I64Array);
        assert_eq!(coerce_nary_op_types(&[Type::I64Array, Type::I64Array, Type::I64Array]).unwrap(), Type::I64Array);
    }

    #[test]
    fn test_nary_coercion_errors() {
        assert!(coerce_nary_op_types(&[]).is_err());
        assert!(coerce_nary_op_types(&[Type::U64Array, Type::U64]).is_err());
    }
}