use crate::ast::Type;
use crate::error::DioError;
use arrow::array::{Array, ArrayRef, UInt64Array, Int64Array};
use arrow::datatypes::DataType as ArrowDataType;
use std::sync::Arc;

/// Metadata extracted from Arrow arrays for JIT compilation
#[derive(Debug, Clone)]
pub struct ArrayMetadata {
    /// Arrow data type
    pub data_type: ArrowDataType,
    /// Number of elements in the array
    pub length: usize,
    /// Raw data pointer for direct Cranelift access
    pub data_ptr: *const u8,
    /// Null bitmap pointer (for future null handling)
    pub null_bitmap: Option<*const u8>,
}

impl ArrayMetadata {
    /// Extract metadata from an Arrow ArrayRef for JIT compilation
    pub fn from_array_ref(array: &ArrayRef) -> Result<Self, DioError> {
        let length = array.len();
        let data_type = array.data_type().clone();
        
        // Extract raw data pointer based on array type
        let data_ptr = match array.data_type() {
            ArrowDataType::UInt64 => {
                let typed_array = array
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| DioError::Runtime("Failed to downcast to UInt64Array".to_string()))?;
                typed_array.values().as_ptr() as *const u8
            }
            ArrowDataType::Int64 => {
                let typed_array = array
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| DioError::Runtime("Failed to downcast to Int64Array".to_string()))?;
                typed_array.values().as_ptr() as *const u8
            }
            _ => {
                return Err(DioError::Runtime(format!(
                    "Unsupported array type for JIT compilation: {:?}",
                    data_type
                )));
            }
        };
        
        // Extract null bitmap if present
        let null_bitmap = array.nulls().map(|nulls| nulls.buffer().as_ptr());
        
        Ok(ArrayMetadata {
            data_type,
            length,
            data_ptr,
            null_bitmap,
        })
    }
    
    /// Get the element size in bytes for this array type
    pub fn element_size(&self) -> Result<usize, DioError> {
        match self.data_type {
            ArrowDataType::UInt64 | ArrowDataType::Int64 => Ok(8),
            _ => Err(DioError::Runtime(format!(
                "Unknown element size for type: {:?}",
                self.data_type
            ))),
        }
    }
    
    /// Check if this array has null values
    pub fn has_nulls(&self) -> bool {
        self.null_bitmap.is_some()
    }
}

/// Convert Dio's Type enum to Arrow's DataType
pub fn dio_type_to_arrow(dio_type: &Type) -> Result<ArrowDataType, DioError> {
    match dio_type {
        Type::U64 => Ok(ArrowDataType::UInt64),
        Type::U64Array => Ok(ArrowDataType::UInt64),
        Type::I64 => Ok(ArrowDataType::Int64),
        Type::I64Array => Ok(ArrowDataType::Int64),
        Type::F64 => Ok(ArrowDataType::Float64),
        Type::F64Array => Ok(ArrowDataType::Float64),
    }
}

/// Convert Arrow's DataType to Dio's Type enum (for arrays only)
pub fn arrow_type_to_dio_array(arrow_type: &ArrowDataType) -> Result<Type, DioError> {
    match arrow_type {
        ArrowDataType::UInt64 => Ok(Type::U64Array),
        ArrowDataType::Int64 => Ok(Type::I64Array),
        ArrowDataType::Float64 => Ok(Type::F64Array),
        _ => Err(DioError::Runtime(format!(
            "Unsupported Arrow type conversion: {:?}",
            arrow_type
        ))),
    }
}

/// Create an Arrow array from raw data for output
pub fn create_output_array(
    data_type: &ArrowDataType,
    length: usize,
) -> Result<ArrayRef, DioError> {
    match data_type {
        ArrowDataType::UInt64 => {
            let mut builder = UInt64Array::builder(length);
            builder.append_nulls(length);
            Ok(Arc::new(builder.finish()))
        }
        ArrowDataType::Int64 => {
            let mut builder = Int64Array::builder(length);
            builder.append_nulls(length); 
            Ok(Arc::new(builder.finish()))
        }
        _ => Err(DioError::Runtime(format!(
            "Cannot create output array for type: {:?}",
            data_type
        ))),
    }
}

/// Extract mutable data pointer from an Arrow array for output
pub unsafe fn extract_mut_data_ptr(array: &ArrayRef) -> Result<*mut u8, DioError> {
    match array.data_type() {
        ArrowDataType::UInt64 => {
            let typed_array = array
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| DioError::Runtime("Failed to downcast to UInt64Array".to_string()))?;
            
            // SAFETY: This is unsafe because we're getting a mutable pointer to array data
            // The caller must ensure exclusive access and proper lifetime management
            let ptr = typed_array.values().as_ptr() as *mut u8;
            Ok(ptr)
        }
        ArrowDataType::Int64 => {
            let typed_array = array
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| DioError::Runtime("Failed to downcast to Int64Array".to_string()))?;
            
            let ptr = typed_array.values().as_ptr() as *mut u8;
            Ok(ptr)
        }
        _ => Err(DioError::Runtime(format!(
            "Cannot extract mutable pointer for type: {:?}",
            array.data_type()
        ))),
    }
}

/// Create Arrow array from Vec<u64> for testing and compatibility  
pub fn create_u64_array_from_vec(data: Vec<u64>) -> Result<ArrayRef, DioError> {
    Ok(Arc::new(UInt64Array::from(data)))
}

/// Create Arrow array from Vec<i64> for testing and compatibility
pub fn create_i64_array_from_vec(data: Vec<i64>) -> Result<ArrayRef, DioError> {
    Ok(Arc::new(Int64Array::from(data)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::UInt64Array;
    
    #[test]
    fn test_array_metadata_extraction() {
        let data = vec![1u64, 2, 3, 4, 5];
        let array: ArrayRef = Arc::new(UInt64Array::from(data.clone()));
        
        let metadata = ArrayMetadata::from_array_ref(&array).unwrap();
        
        assert_eq!(metadata.length, 5);
        assert_eq!(metadata.data_type, ArrowDataType::UInt64);
        assert_eq!(metadata.element_size().unwrap(), 8);
        assert!(!metadata.has_nulls());
        
        // Verify data pointer is valid
        unsafe {
            let ptr = metadata.data_ptr as *const u64;
            let slice = std::slice::from_raw_parts(ptr, metadata.length);
            assert_eq!(slice, &data);
        }
    }
    
    #[test]
    fn test_type_conversion() {
        assert_eq!(dio_type_to_arrow(&Type::U64Array).unwrap(), ArrowDataType::UInt64);
        assert_eq!(dio_type_to_arrow(&Type::I64Array).unwrap(), ArrowDataType::Int64);
        
        assert_eq!(arrow_type_to_dio_array(&ArrowDataType::UInt64).unwrap(), Type::U64Array);
        assert_eq!(arrow_type_to_dio_array(&ArrowDataType::Int64).unwrap(), Type::I64Array);
    }
    
    #[test]
    fn test_output_array_creation() {
        let array = create_output_array(&ArrowDataType::UInt64, 10).unwrap();
        assert_eq!(array.len(), 10);
        assert_eq!(array.data_type(), &ArrowDataType::UInt64);
    }
}