//! Runtime execution components for compiled staged code.
//!
//! This module contains types used for executing compiled functions and managing
//! runtime values. It bridges the gap between compile-time staged values and
//! runtime execution.

use crate::{DataType, PrimType, StagingError};
use cranelift_jit::JITModule;
use std::marker::PhantomData;
use std::sync::Arc;

/// Runtime argument that can be passed to compiled JIT functions.
///
/// This enum is Copy and uses PhantomData to track lifetimes, ensuring that
/// the JIT code cannot outlive the data it references.
#[derive(Copy, Clone, Debug)]
pub enum Arg<'a> {
    /// A scalar value (i64, u64, bool) stored as raw u64 bits
    Scalar(u64),
    /// An immutable array/slice reference
    Array {
        ptr: *const u8,
        len: usize,
        _lifetime: PhantomData<&'a [u8]>,
    },
    /// A mutable array/slice reference
    ArrayMut {
        ptr: *mut u8,
        len: usize,
        _lifetime: PhantomData<&'a mut [u8]>,
    },
}

impl<'a> Arg<'a> {
    /// Convert to u64 slots for passing to JIT code
    /// Returns (slot1, slot2_opt) - scalars use 1 slot, arrays use 2 (ptr, len)
    fn to_u64_slots(&self) -> (u64, Option<u64>) {
        match self {
            Arg::Scalar(val) => (*val, None),
            Arg::Array { ptr, len, .. } => (*ptr as u64, Some(*len as u64)),
            Arg::ArrayMut { ptr, len, .. } => (*ptr as u64, Some(*len as u64)),
        }
    }
}

/// Trait for types that can be converted into runtime arguments.
///
/// This allows convenient calling syntax where you can pass primitives,
/// slices, and mutable slices directly without manually constructing `Arg`.
pub trait ArgLike<'a> {
    fn into_arg(self) -> Arg<'a>;
}

// Scalar implementations
impl<'a> ArgLike<'a> for i64 {
    fn into_arg(self) -> Arg<'a> {
        Arg::Scalar(self as u64)
    }
}

impl<'a> ArgLike<'a> for u64 {
    fn into_arg(self) -> Arg<'a> {
        Arg::Scalar(self)
    }
}

impl<'a> ArgLike<'a> for bool {
    fn into_arg(self) -> Arg<'a> {
        Arg::Scalar(if self { 1 } else { 0 })
    }
}

// Immutable slice implementations
impl<'a> ArgLike<'a> for &'a [i64] {
    fn into_arg(self) -> Arg<'a> {
        Arg::Array {
            ptr: self.as_ptr() as *const u8,
            len: self.len(),
            _lifetime: PhantomData,
        }
    }
}

impl<'a> ArgLike<'a> for &'a [u64] {
    fn into_arg(self) -> Arg<'a> {
        Arg::Array {
            ptr: self.as_ptr() as *const u8,
            len: self.len(),
            _lifetime: PhantomData,
        }
    }
}

// Mutable slice implementations
impl<'a> ArgLike<'a> for &'a mut [i64] {
    fn into_arg(self) -> Arg<'a> {
        Arg::ArrayMut {
            ptr: self.as_mut_ptr() as *mut u8,
            len: self.len(),
            _lifetime: PhantomData,
        }
    }
}

impl<'a> ArgLike<'a> for &'a mut [u64] {
    fn into_arg(self) -> Arg<'a> {
        Arg::ArrayMut {
            ptr: self.as_mut_ptr() as *mut u8,
            len: self.len(),
            _lifetime: PhantomData,
        }
    }
}

// ScalarValue implementation for backward compatibility
impl<'a> ArgLike<'a> for ScalarValue {
    fn into_arg(self) -> Arg<'a> {
        Arg::Scalar(self.to_u64_bits())
    }
}

impl<'a> ArgLike<'a> for &'a ScalarValue {
    fn into_arg(self) -> Arg<'a> {
        Arg::Scalar(self.to_u64_bits())
    }
}

// Arg implements ArgLike (identity conversion)
impl<'a> ArgLike<'a> for Arg<'a> {
    fn into_arg(self) -> Arg<'a> {
        self
    }
}

/// Result of a JIT function call with lifetime tracking.
///
/// The return value is always i64 (raw bits), but the lifetime is tracked
/// to ensure it doesn't outlive any input references.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct Res<'a> {
    value: i64,
    _lifetime: PhantomData<&'a ()>,
}

impl<'a> Res<'a> {
    /// Create a new result with lifetime tracking
    fn new(value: i64) -> Self {
        Self {
            value,
            _lifetime: PhantomData,
        }
    }

    /// Get the raw i64 value
    pub fn as_i64(self) -> i64 {
        self.value
    }

    /// Get the value reinterpreted as u64
    pub fn as_u64(self) -> u64 {
        self.value as u64
    }

    /// Get the value as bool (non-zero = true)
    pub fn as_bool(self) -> bool {
        self.value != 0
    }

    /// Unchecked conversion to i64 (for backward compatibility)
    ///
    /// This is the same as `as_i64()` but matches the ScalarValue API.
    pub fn as_i64_unchecked(self) -> i64 {
        self.as_i64()
    }

    /// Unchecked conversion to u64 (for backward compatibility)
    ///
    /// This is the same as `as_u64()` but matches the ScalarValue API.
    pub fn as_u64_unchecked(self) -> u64 {
        self.as_u64()
    }

    /// Unchecked conversion to bool (for backward compatibility)
    ///
    /// This is the same as `as_bool()` but matches the ScalarValue API.
    pub fn as_bool_unchecked(self) -> bool {
        self.as_bool()
    }
}

// PartialEq with ScalarValue for backward compatibility
impl<'a> PartialEq<ScalarValue> for Res<'a> {
    fn eq(&self, other: &ScalarValue) -> bool {
        match other {
            ScalarValue::I64(v) => self.value == *v,
            ScalarValue::U64(v) => self.value == (*v as i64),
            ScalarValue::Bool(v) => (self.value != 0) == *v,
        }
    }
}

impl<'a> PartialEq<Res<'a>> for ScalarValue {
    fn eq(&self, other: &Res<'a>) -> bool {
        other.eq(self)
    }
}

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

    /// Execute the compiled function with lifetime-tracked arguments
    ///
    /// This is the primary calling interface that accepts an iterator of `Arg` values.
    /// The lifetime tracking ensures that the JIT code cannot outlive any referenced data.
    ///
    /// # Example
    ///
    /// ```
    /// use tutorial::{Compiler, DataType, StagedU64, Expr, Arg};
    ///
    /// let compiler = Compiler::new().unwrap();
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
    /// let result = compiled.call_with(&[10u64, 20u64]).unwrap();
    /// assert_eq!(result.as_u64(), 30);
    /// ```
    pub fn call_with<'a, A: ArgLike<'a>>(&mut self, args: impl IntoIterator<Item = A>) -> Result<Res<'a>, StagingError> {
        // Clear and build arg buffer
        self.arg_buffer.clear();

        for arg_like in args {
            let arg = arg_like.into_arg();
            let (slot1, slot2_opt) = arg.to_u64_slots();
            self.arg_buffer.push(slot1);
            if let Some(slot2) = slot2_opt {
                self.arg_buffer.push(slot2);
            }
        }

        // Call the compiled function - always returns i64
        unsafe {
            type Fn = extern "C" fn(*const u64) -> i64;
            let func: Fn = std::mem::transmute(self.code_ptr);
            let result = func(self.arg_buffer.as_ptr());
            Ok(Res::new(result))
        }
    }

    /// Execute the compiled function with any ArgLike values
    ///
    /// This is an alias for `call_with` that accepts any type implementing `ArgLike`.
    /// Works with `Arg`, `ScalarValue`, primitives (i64, u64, bool), and slices.
    ///
    /// # Example
    ///
    /// ```
    /// use tutorial::{Compiler, DataType, StagedU64, Expr, ScalarValue};
    ///
    /// let compiler = Compiler::new().unwrap();
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
    /// // Works with ScalarValue (backward compat)
    /// let result = compiled.call(&[ScalarValue::U64(10), ScalarValue::U64(20)]).unwrap();
    /// assert_eq!(result.as_u64(), 30);
    /// ```
    pub fn call<'a, A: ArgLike<'a>>(&mut self, args: impl IntoIterator<Item = A>) -> Result<Res<'a>, StagingError> {
        self.call_with(args)
    }

    /// Legacy method: Execute with i64 arguments and return i64
    ///
    /// This is kept for backward compatibility with existing tests.
    /// Prefer using `call` or `call_with` for new code.
    pub fn call_i64(&mut self, args: &[i64]) -> i64 {
        let result = self.call_with(args.iter().copied()).unwrap();
        result.as_i64()
    }

    /// Legacy method: Execute with u64 arguments and return u64
    ///
    /// This is kept for backward compatibility with existing tests.
    /// Prefer using `call` or `call_with` for new code.
    pub fn call_u64(&mut self, args: &[u64]) -> u64 {
        let result = self.call_with(args.iter().copied()).unwrap();
        result.as_u64()
    }

    /// Legacy method: Execute with mixed arguments
    ///
    /// This is kept for backward compatibility with existing tests.
    /// Prefer using `call` or `call_with` for new code.
    pub fn call_mixed(&mut self, args: &[i64]) -> i64 {
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
