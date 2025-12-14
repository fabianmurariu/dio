//! Runtime execution components for compiled staged code.
//!
//! This module contains types used for executing compiled functions and managing
//! runtime values. It bridges the gap between compile-time staged values and
//! runtime execution.

use crate::{DataType, PrimType, StagingError};
use cranelift_jit::JITModule;
use std::marker::PhantomData;
use std::sync::Arc;

/// A scalar runtime value that can be passed to and returned from compiled functions
///
/// This represents actual runtime values (not staged/compile-time computations).
/// Similar to how dio3/dio4 use ScalarValue for literals and results.
///
/// # Example
///
/// ```
/// use tutorial::{Compiler, DataType, StagedU64, Expr, ScalarValue};
///
/// let mut compiler = Compiler::new().unwrap();
/// let mut compiled = compiler.compile_nary(
///     vec![DataType::U64, DataType::U64],
///     DataType::U64,
///     |_, vars| {
///         let x = StagedU64::variable(vars[0]);
///         let y = StagedU64::variable(vars[1]);
///         Expr::U64(x + y)
///     }
/// ).unwrap();
///
/// let result = compiled.call(&[
///     ScalarValue::U64(10),
///     ScalarValue::U64(5)
/// ]).unwrap();
///
/// assert_eq!(result, ScalarValue::U64(15));
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ScalarValue {
    I64(i64),
    U64(u64),
    Bool(bool),
}

impl ScalarValue {
    /// Get the data type of this scalar value
    pub fn data_type(&self) -> DataType {
        match self {
            ScalarValue::I64(_) => DataType::I64,
            ScalarValue::U64(_) => DataType::U64,
            ScalarValue::Bool(_) => DataType::Bool,
        }
    }

    /// Convert this scalar value to u64 bit representation for passing to compiled functions
    ///
    /// This is more efficient than byte-level operations and maintains proper alignment.
    /// Similar to how dio3/dio4 work with raw pointers, but using u64 slots.
    pub fn to_u64_bits(&self) -> u64 {
        match self {
            ScalarValue::I64(v) => *v as u64,      // Reinterpret i64 as u64
            ScalarValue::U64(v) => *v,              // Already u64
            ScalarValue::Bool(v) => if *v { 1 } else { 0 }, // 0 or 1
        }
    }

    /// Convert from u64 bit representation back to typed ScalarValue
    pub fn from_u64_bits(bits: u64, data_type: &DataType) -> Self {
        match data_type {
            DataType::Prim(PrimType::I64) => ScalarValue::I64(bits as i64),
            DataType::Prim(PrimType::U64) => ScalarValue::U64(bits),
            DataType::Bool => ScalarValue::Bool(bits != 0),
            DataType::Prim(_) => panic!("ScalarValue only supports I64/U64/Bool, got {:?}", data_type),
            DataType::Array { .. } => panic!("Cannot convert bits to array type"),
            DataType::ExtPtr(_) => panic!("Cannot convert bits to external pointer"),
            DataType::Unit => panic!("Cannot convert bits to unit type"),
            DataType::Struct(_) => panic!("Cannot convert bits to struct type"),
            DataType::Slice { .. } => panic!("Cannot convert bits to slice type"),
        }
    }

    /// Unwrap as i64, panics if not I64
    pub fn as_i64_unchecked(&self) -> i64 {
        match self {
            ScalarValue::I64(v) => *v,
            _ => panic!("Expected I64, got {:?}", self.data_type()),
        }
    }

    /// Unwrap as u64, panics if not U64
    pub fn as_u64_unchecked(&self) -> u64 {
        match self {
            ScalarValue::U64(v) => *v,
            _ => panic!("Expected U64, got {:?}", self.data_type()),
        }
    }

    /// Unwrap as bool, panics if not Bool
    pub fn as_bool_unchecked(&self) -> bool {
        match self {
            ScalarValue::Bool(v) => *v,
            _ => panic!("Expected Bool, got {:?}", self.data_type()),
        }
    }
}

/// A compiled generic n-ary function
///
/// This is similar to how dio3/dio4 handle compiled functions with type information
/// preserved for proper execution.
///
/// # Lifetime Safety
///
/// CompiledNary holds an `Arc<JITModule>` to ensure the compiled code remains valid
/// for the lifetime of this struct. This prevents use-after-free bugs that would occur
/// if the JITModule were dropped while CompiledNary still holds a code pointer.
pub struct CompiledNary {
    /// Shared ownership of the JIT module to keep compiled code alive
    _module: Arc<JITModule>,
    code_ptr: *const u8,
    param_types: Vec<DataType>,
    return_type: DataType,
    /// Reusable buffer for argument passing to avoid allocation on every call
    arg_buffer: Vec<u64>,
}

impl CompiledNary {
    /// Create a new compiled function
    ///
    /// # Safety
    /// The caller must ensure that code_ptr points to valid compiled machine code
    /// within the provided module, and that it matches the provided parameter and return types.
    pub(crate) fn new(
        module: Arc<JITModule>,
        code_ptr: *const u8,
        param_types: Vec<DataType>,
        return_type: DataType,
    ) -> Self {
        Self {
            _module: module,
            code_ptr,
            param_types,
            return_type,
            arg_buffer: Vec::new(), // Will be resized on first call
        }
    }

    /// Execute the compiled function with type-safe ScalarValue arguments
    ///
    /// This is the primary calling interface, similar to how dio4's execute()
    /// takes ArrayRef and extracts raw pointers. Here we take ScalarValues,
    /// perform type checking, extract raw values, and call the compiled function.
    ///
    /// # Example
    ///
    /// ```
    /// use tutorial::{Compiler, DataType, StagedU64, Expr, ScalarValue};
    ///
    /// let mut compiler = Compiler::new().unwrap();
    /// let mut compiled = compiler.compile_nary(
    ///     vec![DataType::U64, DataType::U64],
    ///     DataType::U64,
    ///     |_, vars| {
    ///         let x = StagedU64::variable(vars[0]);
    ///         let y = StagedU64::variable(vars[1]);
    ///         Expr::U64(x + y)
    ///     }
    /// ).unwrap();
    ///
    /// let result = compiled.call(&[
    ///     ScalarValue::U64(10),
    ///     ScalarValue::U64(20)
    /// ]).unwrap();
    ///
    /// assert_eq!(result, ScalarValue::U64(30));
    /// ```
    pub fn call(&mut self, args: &[ScalarValue]) -> Result<ScalarValue, StagingError> {
        // Verify argument count
        if args.len() != self.param_types.len() {
            return Err(StagingError::ExecutionFailed {
                reason: format!(
                    "Expected {} arguments, got {}",
                    self.param_types.len(),
                    args.len()
                ),
            });
        }

        // Verify argument types match expected parameter types
        for (i, (arg, expected_type)) in args.iter().zip(&self.param_types).enumerate() {
            if arg.data_type() != *expected_type {
                return Err(StagingError::TypeMismatch {
                    expected: format!("argument {} type {:?}", i, expected_type),
                    actual: format!("got {:?}", arg.data_type()),
                });
            }
        }

        // Reuse the buffer, clear and fill with new args
        // Using Vec<u64> is more efficient than Vec<u8> with byte-level copies
        self.arg_buffer.clear();
        self.arg_buffer.reserve(args.len());
        for arg in args {
            self.arg_buffer.push(arg.to_u64_bits());
        }

        // Call the compiled function with different signatures based on return type
        // The Cranelift function receives *const u64 and loads values at u64 offsets
        unsafe {
            match &self.return_type {
                DataType::Prim(PrimType::I64) => {
                    type Fn = extern "C" fn(*const u64) -> i64;
                    let func: Fn = std::mem::transmute(self.code_ptr);
                    let result = func(self.arg_buffer.as_ptr());
                    Ok(ScalarValue::I64(result))
                }
                DataType::Prim(PrimType::U64) => {
                    type Fn = extern "C" fn(*const u64) -> i64;
                    let func: Fn = std::mem::transmute(self.code_ptr);
                    let result = func(self.arg_buffer.as_ptr());
                    Ok(ScalarValue::U64(result as u64))
                }
                DataType::Prim(_) => {
                    Err(StagingError::ExecutionFailed {
                        reason: format!("ScalarValue only supports I64/U64, got {:?}", self.return_type),
                    })
                }
                DataType::Bool => {
                    type Fn = extern "C" fn(*const u64) -> i8;
                    let func: Fn = std::mem::transmute(self.code_ptr);
                    let result = func(self.arg_buffer.as_ptr());
                    Ok(ScalarValue::Bool(result != 0))
                }
                DataType::Array { .. } => {
                    Err(StagingError::ExecutionFailed {
                        reason: "Cannot return array from compiled function via call()".to_string(),
                    })
                }
                DataType::ExtPtr(_) => {
                    Err(StagingError::ExecutionFailed {
                        reason: "Cannot return external pointer from compiled function via call()".to_string(),
                    })
                }
                DataType::Unit => {
                    // Unit type - just call the function and ignore result
                    type Fn = extern "C" fn(*const u64) -> i64;
                    let func: Fn = std::mem::transmute(self.code_ptr);
                    let _result = func(self.arg_buffer.as_ptr());
                    Ok(ScalarValue::U64(0))  // Return dummy value for unit
                }
                DataType::Struct(_) => {
                    Err(StagingError::ExecutionFailed {
                        reason: "Cannot return struct from compiled function via call() - use pointer return".to_string(),
                    })
                }
                DataType::Slice { .. } => {
                    Err(StagingError::ExecutionFailed {
                        reason: "Cannot return slice from compiled function via call() - use pointer return".to_string(),
                    })
                }
            }
        }
    }

    /// Execute the compiled function with i64 arguments (low-level interface)
    ///
    /// Similar to how dio3/dio4's call_nary_op works, we pass a pointer to
    /// a byte array. The types are interpreted based on param_types.
    ///
    /// # Safety
    /// The caller must ensure that `args.len() >= param_types.len()`
    pub fn call_i64(&self, args: &[i64]) -> i64 {
        assert!(
            args.len() >= self.param_types.len(),
            "Expected at least {} arguments, got {}",
            self.param_types.len(),
            args.len()
        );

        // Pack i64 arguments into byte buffer
        let mut arg_buffer = vec![0u8; args.len() * 8];
        for (i, &arg) in args.iter().enumerate() {
            let bytes = arg.to_ne_bytes();
            let offset = i * 8;
            arg_buffer[offset..offset + 8].copy_from_slice(&bytes);
        }

        unsafe {
            let func: extern "C" fn(*const u8) -> i64 = std::mem::transmute(self.code_ptr);
            func(arg_buffer.as_ptr())
        }
    }

    /// Execute the compiled function with u64 arguments (low-level interface)
    ///
    /// # Safety
    /// The caller must ensure that `args.len() >= param_types.len()`
    pub fn call_u64(&self, args: &[u64]) -> u64 {
        assert!(
            args.len() >= self.param_types.len(),
            "Expected at least {} arguments, got {}",
            self.param_types.len(),
            args.len()
        );

        // Pack u64 arguments into byte buffer (reinterpret as i64 for storage)
        let mut arg_buffer = vec![0u8; args.len() * 8];
        for (i, &arg) in args.iter().enumerate() {
            let bytes = (arg as i64).to_ne_bytes();
            let offset = i * 8;
            arg_buffer[offset..offset + 8].copy_from_slice(&bytes);
        }

        unsafe {
            // Note: return type is still i64 in Cranelift, we reinterpret as u64
            let func: extern "C" fn(*const u8) -> i64 = std::mem::transmute(self.code_ptr);
            let result = func(arg_buffer.as_ptr());
            result as u64
        }
    }

    /// Execute with mixed i64/u64 arguments based on parameter types (low-level interface)
    ///
    /// This is the most flexible calling convention - values are passed
    /// as bytes but interpreted according to their declared types.
    pub fn call_mixed(&self, args: &[i64]) -> i64 {
        self.call_i64(args)
    }

    /// Get the parameter types
    pub fn param_types(&self) -> &[DataType] {
        &self.param_types
    }

    /// Get the return type
    pub fn return_type(&self) -> &DataType {
        &self.return_type
    }
}

/// Safe wrapper for immutable array arguments passed to JIT code
///
/// This wrapper encapsulates the lifetime safety invariants for array pointers,
/// ensuring that the JIT code cannot outlive the array data.
///
/// # Example
///
/// ```
/// use tutorial::{ArrayArg, Compiler, DataType, StagedArray};
///
/// let input_data: Vec<u64> = vec![1, 2, 3, 4, 5];
/// let array_arg = ArrayArg::from(&input_data);
///
/// // The array_arg can now be safely passed to JIT code
/// // and the lifetime system ensures input_data outlives array_arg
/// ```
pub struct ArrayArg<'a, T> {
    ptr: *const T,
    len: usize,
    _lifetime: PhantomData<&'a [T]>,
}

impl<'a, T> ArrayArg<'a, T> {
    /// Create an ArrayArg from a slice
    pub fn from(slice: &'a [T]) -> Self {
        Self {
            ptr: slice.as_ptr(),
            len: slice.len(),
            _lifetime: PhantomData,
        }
    }

    /// Get the pointer as u64 (for passing to JIT functions)
    pub fn ptr_as_u64(&self) -> u64 {
        self.ptr as u64
    }

    /// Get the length as u64 (for passing to JIT functions)
    pub fn len_as_u64(&self) -> u64 {
        self.len as u64
    }

    /// Get the raw pointer
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }
}

/// Safe wrapper for mutable array arguments passed to JIT code
///
/// This wrapper encapsulates the lifetime safety invariants for mutable array pointers,
/// ensuring that the JIT code cannot outlive the array data and that there is exclusive
/// mutable access.
///
/// # Example
///
/// ```
/// use tutorial::{MutableArrayArg, Compiler, DataType, StagedArray};
///
/// let mut output_data: Vec<u64> = vec![0; 5];
/// let mut_array_arg = MutableArrayArg::from(&mut output_data);
///
/// // The mut_array_arg can now be safely passed to JIT code
/// // and the lifetime system ensures output_data outlives mut_array_arg
/// ```
pub struct MutableArrayArg<'a, T> {
    ptr: *mut T,
    len: usize,
    _lifetime: PhantomData<&'a mut [T]>,
}

impl<'a, T> MutableArrayArg<'a, T> {
    /// Create a MutableArrayArg from a mutable slice
    pub fn from(slice: &'a mut [T]) -> Self {
        Self {
            ptr: slice.as_mut_ptr(),
            len: slice.len(),
            _lifetime: PhantomData,
        }
    }

    /// Get the pointer as u64 (for passing to JIT functions)
    pub fn ptr_as_u64(&self) -> u64 {
        self.ptr as u64
    }

    /// Get the length as u64 (for passing to JIT functions)
    pub fn len_as_u64(&self) -> u64 {
        self.len as u64
    }

    /// Get the raw pointer
    pub fn as_mut_ptr(&self) -> *mut T {
        self.ptr
    }

    /// Get the length
    pub fn len(&self) -> usize {
        self.len
    }
}
