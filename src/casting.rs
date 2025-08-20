use crate::ast::Type;
use crate::error::DioError;

/// Casting rules following SQL/DuckDB standards
///
/// Rules based on DuckDB's casting behavior:
/// 1. Implicit casting between compatible integer types (with potential data loss warnings)
/// 2. Explicit casting for all type conversions
/// 3. Numeric widening: i32 -> i64 -> f64 is safe
/// 4. Numeric narrowing: f64 -> i64 -> i32 requires explicit cast and may truncate

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CastKind {
    /// No casting needed - types are identical
    None,
    /// Implicit casting allowed - compatible types
    Implicit,
    /// Explicit casting required - incompatible types or potential data loss
    Explicit,
    /// Impossible casting - types are fundamentally incompatible
    Impossible,
}

impl Type {
    /// Check if this type can be cast to another type and what kind of cast is needed
    pub fn cast_to(&self, target: &Type) -> CastKind {
        use Type::*;

        match (self, target) {
            // Same types - no cast needed
            (a, b) if a == b => CastKind::None,

            // Implicit casts for compatible array types
            // U64Array <-> I64Array: Compatible at runtime (both 64-bit integers)
            (U64Array, I64Array) | (I64Array, U64Array) => CastKind::Implicit,

            // Implicit casts for compatible scalar types
            (U64, I64) | (I64, U64) => CastKind::Implicit,

            // Explicit casts for narrowing/widening between different bit sizes
            // These would be future extensions when we add U32, I32, etc.

            // Float conversions (future)
            (F64, F64Array) | (F64Array, F64) => CastKind::Explicit, // Scalar <-> Array
            (U64 | I64, F64) => CastKind::Explicit,                  // Integer to float
            (F64, U64 | I64) => CastKind::Explicit, // Float to integer (truncation)
            (U64Array | I64Array, F64Array) => CastKind::Explicit, // Integer array to float array
            (F64Array, U64Array | I64Array) => CastKind::Explicit, // Float array to integer array

            // Scalar to array conversions are impossible without broadcast operations
            (U64 | I64 | F64, U64Array | I64Array | F64Array) => CastKind::Impossible,
            (U64Array | I64Array | F64Array, U64 | I64 | F64) => CastKind::Impossible,

            // Future types - for now treat as explicit
            _ => CastKind::Explicit,
        }
    }

    /// Check if this type can be implicitly cast to another type
    pub fn can_implicit_cast_to(&self, target: &Type) -> bool {
        matches!(self.cast_to(target), CastKind::None | CastKind::Implicit)
    }

    /// Check if this type can be explicitly cast to another type
    pub fn can_explicit_cast_to(&self, target: &Type) -> bool {
        !matches!(self.cast_to(target), CastKind::Impossible)
    }

    /// Get the element type for array types
    pub fn element_type(&self) -> Option<Type> {
        match self {
            Type::U64Array => Some(Type::U64),
            Type::I64Array => Some(Type::I64),
            Type::F64Array => Some(Type::F64),
            _ => None,
        }
    }

    /// Get the array type for scalar types
    pub fn array_type(&self) -> Option<Type> {
        match self {
            Type::U64 => Some(Type::U64Array),
            Type::I64 => Some(Type::I64Array),
            Type::F64 => Some(Type::F64Array),
            _ => None,
        }
    }

    /// Check if this is an array type
    pub fn is_array(&self) -> bool {
        matches!(self, Type::U64Array | Type::I64Array | Type::F64Array)
    }

    /// Check if this is a scalar type
    pub fn is_scalar(&self) -> bool {
        matches!(self, Type::U64 | Type::I64 | Type::F64)
    }

    /// Check if this is an integer type (scalar or array)
    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Type::U64 | Type::I64 | Type::U64Array | Type::I64Array
        )
    }

    /// Check if this is a float type (scalar or array)
    pub fn is_float(&self) -> bool {
        matches!(self, Type::F64 | Type::F64Array)
    }

    /// Check if this is a signed integer type
    pub fn is_signed_integer(&self) -> bool {
        matches!(self, Type::I64 | Type::I64Array)
    }

    /// Check if this is an unsigned integer type
    pub fn is_unsigned_integer(&self) -> bool {
        matches!(self, Type::U64 | Type::U64Array)
    }
}

/// Type coercion for N-ary operations following SQL rules
/// Determines the common type for operations like (+ a b c d...)
pub fn coerce_nary_op_types(operand_types: &[Type]) -> Result<Type, DioError> {
    if operand_types.is_empty() {
        return Err(DioError::TypeMismatch {
            expected: "at least one operand".to_string(),
            found: "no operands".to_string(),
            context: "N-ary operations require at least one operand".to_string(),
        });
    }

    // Start with the first type and coerce with each subsequent type
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
        // Same types
        (a, b) if a == b => Ok(a.clone()),

        // Compatible array types - choose signed over unsigned for safety
        (U64Array, I64Array) | (I64Array, U64Array) => Ok(I64Array),
        (U64, I64) | (I64, U64) => Ok(I64),

        // Mixed scalar/array is an error
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

        // Future: Add float coercion rules
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
        // Scalar to array without broadcast
        assert_eq!(Type::U64.cast_to(&Type::U64Array), CastKind::Impossible);
        assert_eq!(Type::I64Array.cast_to(&Type::I64), CastKind::Impossible);
    }

    #[test]
    fn test_type_properties() {
        assert!(Type::U64Array.is_array());
        assert!(Type::I64.is_scalar());
        assert!(Type::U64.is_integer());
        assert!(Type::I64Array.is_signed_integer());
        assert!(Type::U64.is_unsigned_integer());
    }

    #[test]
    fn test_binary_coercion() {
        // Same types
        assert_eq!(
            coerce_binary_op_types(&Type::U64Array, &Type::U64Array).unwrap(),
            Type::U64Array
        );

        // Compatible types - prefer signed
        assert_eq!(
            coerce_binary_op_types(&Type::U64Array, &Type::I64Array).unwrap(),
            Type::I64Array
        );
        assert_eq!(
            coerce_binary_op_types(&Type::U64, &Type::I64).unwrap(),
            Type::I64
        );
    }

    #[test]
    fn test_binary_coercion_errors() {
        // Mixed scalar/array should error
        assert!(coerce_binary_op_types(&Type::U64, &Type::U64Array).is_err());
        assert!(coerce_binary_op_types(&Type::I64Array, &Type::I64).is_err());
    }

    #[test]
    fn test_nary_coercion() {
        // Single type
        assert_eq!(
            coerce_nary_op_types(&[Type::U64Array]).unwrap(),
            Type::U64Array
        );

        // Two types (same as binary)
        assert_eq!(
            coerce_nary_op_types(&[Type::U64Array, Type::I64Array]).unwrap(),
            Type::I64Array
        );

        // Multiple types - should coerce to most general (signed)
        assert_eq!(
            coerce_nary_op_types(&[Type::U64Array, Type::U64Array, Type::I64Array]).unwrap(),
            Type::I64Array
        );

        // All signed should stay signed
        assert_eq!(
            coerce_nary_op_types(&[Type::I64Array, Type::I64Array, Type::I64Array]).unwrap(),
            Type::I64Array
        );
    }

    #[test]
    fn test_nary_coercion_errors() {
        // Empty operands
        assert!(coerce_nary_op_types(&[]).is_err());

        // Mixed scalar/array in N-ary
        assert!(coerce_nary_op_types(&[Type::U64Array, Type::U64]).is_err());
    }
}
