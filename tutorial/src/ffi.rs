//! Foreign Function Interface (FFI) support for calling Rust functions from JIT code.
//!
//! This module provides:
//! - External function registration and type checking
//! - Iterator support (boxed iterators called via pointers)
//! - Option types for FFI boundaries

use std::collections::HashMap;

use crate::DataType;

// =============================================================================
// ITERATOR SUPPORT
// =============================================================================

/// A boxed iterator over i64 values, exposed via pointer to JIT code
pub struct IterBoxI64 {
    inner: Box<dyn Iterator<Item = i64>>,
}

/// FFI-safe Option<i64> with explicit layout
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OptionI64 {
    /// 1 if Some, 0 if None
    pub has_value: i8,
    /// The value (only valid if has_value == 1)
    pub value: i64,
}

impl OptionI64 {
    pub fn some(value: i64) -> Self {
        OptionI64 {
            has_value: 1,
            value,
        }
    }

    pub fn none() -> Self {
        OptionI64 {
            has_value: 0,
            value: 0,
        }
    }

    pub fn to_option(self) -> Option<i64> {
        if self.has_value != 0 {
            Some(self.value)
        } else {
            None
        }
    }
}

/// Create a range iterator and return pointer to it
///
/// # Safety
/// Caller must eventually call iter_drop with this pointer
#[no_mangle]
pub extern "C" fn iter_create_range(start: i64, end: i64) -> *mut IterBoxI64 {
    Box::into_raw(Box::new(IterBoxI64 {
        inner: Box::new(start..end),
    }))
}

/// Get next value from iterator
///
/// # Safety
/// ptr must be a valid pointer from iter_create_range
#[no_mangle]
pub extern "C" fn iter_next_i64(ptr: *mut IterBoxI64) -> OptionI64 {
    if ptr.is_null() {
        return OptionI64::none();
    }

    let iter_box = unsafe { &mut *ptr };
    match iter_box.inner.next() {
        Some(val) => OptionI64::some(val),
        None => OptionI64::none(),
    }
}

/// Drop an iterator
///
/// # Safety
/// ptr must be a valid pointer from iter_create_range and must not be used after this call
#[no_mangle]
pub extern "C" fn iter_drop(ptr: *mut IterBoxI64) {
    if !ptr.is_null() {
        unsafe {
            drop(Box::from_raw(ptr));
        }
    }
}

// =============================================================================
// FUNCTION SIGNATURE REGISTRY
// =============================================================================

/// Signature of an external function that can be called from JIT code
#[derive(Debug, Clone)]
pub struct FunctionSignature {
    pub name: String,
    pub params: Vec<DataType>,
    pub return_type: DataType,
}

/// Registry of external functions with their signatures and pointers
#[derive(Default)]
pub struct ExternalFunctionRegistry {
    functions: HashMap<String, (FunctionSignature, *const u8)>,
}

impl ExternalFunctionRegistry {
    /// Create a new empty registry
    pub fn new() -> Self {
        Self {
            functions: HashMap::new(),
        }
    }

    /// Register an external function
    pub fn register(
        &mut self,
        name: String,
        params: Vec<DataType>,
        return_type: DataType,
        fn_ptr: *const u8,
    ) {
        let signature = FunctionSignature {
            name: name.clone(),
            params,
            return_type,
        };
        self.functions.insert(name, (signature, fn_ptr));
    }

    /// Look up a function signature by name
    pub fn get_signature(&self, name: &str) -> Option<&FunctionSignature> {
        self.functions.get(name).map(|(sig, _)| sig)
    }

    /// Look up a function pointer by name
    pub fn get_function_ptr(&self, name: &str) -> Option<*const u8> {
        self.functions.get(name).map(|(_, ptr)| *ptr)
    }

    /// Check if a function is registered
    pub fn has_function(&self, name: &str) -> bool {
        self.functions.contains_key(name)
    }
}

// =============================================================================
// HELPER MACROS
// =============================================================================

/// Helper macro to extract function signature information
///
/// Usage:
/// ```ignore
/// extern "C" fn my_add(x: i64, y: i64) -> i64 { x + y }
/// compiler.register_extern_fn!(my_add, [I64, I64] -> I64);
/// ```
#[macro_export]
macro_rules! register_extern_fn {
    ($compiler:expr, $fn_name:ident, [$($param:ident),*] -> $ret:ident) => {
        $compiler.register_external_function(
            stringify!($fn_name),
            $fn_name as *const u8,
            vec![$(DataType::$param),*],
            DataType::$ret,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_option_i64_some() {
        let opt = OptionI64::some(42);
        assert_eq!(opt.has_value, 1);
        assert_eq!(opt.value, 42);
        assert_eq!(opt.to_option(), Some(42));
    }

    #[test]
    fn test_option_i64_none() {
        let opt = OptionI64::none();
        assert_eq!(opt.has_value, 0);
        assert_eq!(opt.to_option(), None);
    }

    #[test]
    fn test_iter_create_and_iterate() {
        let iter_ptr = iter_create_range(0, 3);

        let val1 = iter_next_i64(iter_ptr);
        assert_eq!(val1.to_option(), Some(0));

        let val2 = iter_next_i64(iter_ptr);
        assert_eq!(val2.to_option(), Some(1));

        let val3 = iter_next_i64(iter_ptr);
        assert_eq!(val3.to_option(), Some(2));

        let val4 = iter_next_i64(iter_ptr);
        assert_eq!(val4.to_option(), None);

        iter_drop(iter_ptr);
    }

    #[test]
    fn test_function_registry() {
        let mut registry = ExternalFunctionRegistry::new();

        extern "C" fn test_fn(x: i64) -> i64 {
            x + 1
        }

        registry.register(
            "test_fn".to_string(),
            vec![DataType::I64],
            DataType::I64,
            test_fn as *const u8,
        );

        assert!(registry.has_function("test_fn"));
        assert!(!registry.has_function("other_fn"));

        let sig = registry.get_signature("test_fn").unwrap();
        assert_eq!(sig.name, "test_fn");
        assert_eq!(sig.params, vec![DataType::I64]);
        assert_eq!(sig.return_type, DataType::I64);
    }
}
