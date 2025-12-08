//! Iterator support for staged computations.
//!
//! This module provides iterator abstractions that work within the Expr API:
//! - `StagedArrayIter`: Zero-copy iteration over `StagedArray` elements
//! - `StagedExternIter`: Iteration over external (Rust) iterators via FFI
//!
//! Both iterator types support the `for_each` pattern for generating efficient
//! loop code in the JIT-compiled output.
//!
//! # Design Philosophy
//!
//! Iterators in JIT code need special handling because:
//! 1. We can't use Rust's `Iterator` trait directly (it's a runtime concept)
//! 2. We need to generate loop code at compile time
//! 3. External iterators require FFI calls for each `next()`
//!
//! This module provides two approaches:
//! - **Array iterators**: Fully inlined loops with index variables
//! - **External iterators**: FFI-based iteration with automatic cleanup

use crate::expr::{Expr, StagedBuilder, Var};
use crate::ffi::OptionI64;
use crate::num::{StagedI64, StagedU64};
use crate::{DataType, StagedArray, StagedBool};
use cranelift_frontend::Variable;

// =============================================================================
// STAGED ARRAY ITERATOR
// =============================================================================

/// An iterator over elements of a `StagedArray`.
///
/// This iterator generates efficient loop code that iterates over array elements
/// using an index variable. It's zero-copy - elements are loaded directly from
/// the array without any intermediate allocations.
///
/// # Example
///
/// ```ignore
/// use tutorial::*;
///
/// let mut compiler = Compiler::new().unwrap();
/// let mut compiled = compiler.compile_nary(
///     vec![DataType::arr(DataType::U64)],
///     DataType::U64,
///     |builder, vars| {
///         let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
///
///         // Create iterator and sum all elements
///         let iter = StagedArrayIter::new(input.clone());
///
///         builder.let_mut(Expr::U64(StagedU64::constant(0)), |builder, sum_var| {
///             let loop_expr = iter.for_each(builder, |builder, elem| {
///                 let sum = sum_var.clone().to_u64();
///                 let elem_u64 = elem.to_u64();
///                 builder.set_var(sum_var.clone(), Expr::U64(sum + elem_u64))
///             });
///
///             builder.let1(loop_expr, |_, _| Expr::U64(sum_var.to_u64()))
///         })
///     },
/// ).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StagedArrayIter {
    array: StagedArray,
}

impl StagedArrayIter {
    /// Create a new iterator over the given array.
    pub fn new(array: StagedArray) -> Self {
        StagedArrayIter { array }
    }

    /// Get the element type of this iterator.
    pub fn element_type(&self) -> &DataType {
        self.array.element_type()
    }

    /// Get the underlying array.
    pub fn array(&self) -> &StagedArray {
        &self.array
    }

    /// Generate a for-each loop over all elements.
    ///
    /// This creates a while loop that:
    /// 1. Declares a mutable index variable initialized to 0
    /// 2. Loops while index < array.len()
    /// 3. Loads each element and passes it to the body closure
    /// 4. Increments the index
    ///
    /// The body closure receives a `Var` containing the current element.
    ///
    /// # Returns
    ///
    /// An `Expr` representing the complete loop. The loop itself returns `Unit`,
    /// but you can use mutable variables (via `let_mut`) to accumulate results.
    pub fn for_each<F>(&self, builder: &mut StagedBuilder, body: F) -> Expr
    where
        F: FnOnce(&mut StagedBuilder, Var) -> Expr,
    {
        let array = self.array.clone();
        let len = array.len();
        let element_type = array.element_type().clone();

        // Create mutable index: let mut i = 0
        builder.let_mut(Expr::U64(StagedU64::constant(0)), move |builder, i_var| {
            builder.while_loop(
                |_builder| {
                    // Condition: i < len
                    let i = i_var.clone().to_u64();
                    Expr::Bool(i.lt(&len))
                },
                |builder| {
                    let i = i_var.clone().to_u64();

                    // Load element at index i
                    let elem_expr = builder.array_get(array.clone(), i.clone());

                    // Bind element to variable and execute body
                    builder.let1(elem_expr, |builder, elem_var| {
                        // Execute user's body
                        let body_result = body(builder, elem_var);

                        // Increment index: i = i + 1
                        let i_next = i_var.clone().to_u64() + StagedU64::constant(1);
                        builder.let1(body_result, |builder, _| {
                            builder.set_var(i_var.clone(), Expr::U64(i_next))
                        })
                    })
                },
            )
        })
    }

    /// Generate a for-each loop with index.
    ///
    /// Like `for_each`, but the body closure also receives the current index.
    ///
    /// # Example
    ///
    /// ```ignore
    /// iter.for_each_with_index(builder, |builder, idx, elem| {
    ///     // idx is the current index (Var of type U64)
    ///     // elem is the current element
    ///     Expr::Unit
    /// });
    /// ```
    pub fn for_each_with_index<F>(&self, builder: &mut StagedBuilder, body: F) -> Expr
    where
        F: FnOnce(&mut StagedBuilder, Var, Var) -> Expr,
    {
        let array = self.array.clone();
        let len = array.len();

        // Create mutable index: let mut i = 0
        builder.let_mut(Expr::U64(StagedU64::constant(0)), move |builder, i_var| {
            builder.while_loop(
                |_builder| {
                    // Condition: i < len
                    let i = i_var.clone().to_u64();
                    Expr::Bool(i.lt(&len))
                },
                |builder| {
                    let i = i_var.clone().to_u64();

                    // Load element at index i
                    let elem_expr = builder.array_get(array.clone(), i.clone());

                    // Bind index to a new variable for the user
                    builder.let1(Expr::U64(i.clone()), |builder, idx_var| {
                        // Bind element to variable
                        builder.let1(elem_expr, |builder, elem_var| {
                            // Execute user's body with both index and element
                            let body_result = body(builder, idx_var, elem_var);

                            // Increment index: i = i + 1
                            let i_next = i_var.clone().to_u64() + StagedU64::constant(1);
                            builder.let1(body_result, |builder, _| {
                                builder.set_var(i_var.clone(), Expr::U64(i_next))
                            })
                        })
                    })
                },
            )
        })
    }

    /// Generate a map operation that writes results to an output array.
    ///
    /// This creates a loop that applies a transformation to each element
    /// and writes the result to the corresponding position in the output array.
    ///
    /// # Panics
    ///
    /// Panics if the output array is not mutable.
    pub fn map_to<F>(&self, builder: &mut StagedBuilder, output: StagedArray, transform: F) -> Expr
    where
        F: FnOnce(&mut StagedBuilder, Var) -> Expr,
    {
        assert!(output.is_mutable(), "Output array must be mutable for map_to");

        let input = self.array.clone();
        let len = input.len();

        builder.let_mut(Expr::U64(StagedU64::constant(0)), move |builder, i_var| {
            builder.while_loop(
                |_builder| {
                    let i = i_var.clone().to_u64();
                    Expr::Bool(i.lt(&len))
                },
                |builder| {
                    let i = i_var.clone().to_u64();

                    // Load input element
                    let elem_expr = builder.array_get(input.clone(), i.clone());

                    builder.let1(elem_expr, |builder, elem_var| {
                        // Apply transformation
                        let transformed = transform(builder, elem_var);

                        // Store to output
                        let store_expr = builder.array_set(output.clone(), i.clone(), transformed);

                        // Increment index
                        let i_next = i_var.clone().to_u64() + StagedU64::constant(1);
                        builder.let1(store_expr, |builder, _| {
                            builder.set_var(i_var.clone(), Expr::U64(i_next))
                        })
                    })
                },
            )
        })
    }

    /// Generate a fold/reduce operation over the array.
    ///
    /// This creates a loop that accumulates a result by applying a function
    /// to each element combined with the current accumulator value.
    ///
    /// # Parameters
    ///
    /// - `initial`: The initial accumulator value
    /// - `fold_fn`: A function that takes (builder, accumulator, element) and returns the new accumulator
    ///
    /// # Returns
    ///
    /// An `Expr` that evaluates to the final accumulated value.
    pub fn fold<F>(&self, builder: &mut StagedBuilder, initial: Expr, fold_fn: F) -> Expr
    where
        F: Fn(&mut StagedBuilder, Var, Var) -> Expr,
    {
        let acc_type = initial.data_type();
        let array = self.array.clone();
        let len = array.len();

        // Create mutable accumulator and index
        builder.let_mut(initial, move |builder, acc_var| {
            builder.let_mut(Expr::U64(StagedU64::constant(0)), |builder, i_var| {
                let loop_expr = builder.while_loop(
                    |_builder| {
                        let i = i_var.clone().to_u64();
                        Expr::Bool(i.lt(&len))
                    },
                    |builder| {
                        let i = i_var.clone().to_u64();

                        // Load element
                        let elem_expr = builder.array_get(array.clone(), i.clone());

                        builder.let1(elem_expr, |builder, elem_var| {
                            // Apply fold function
                            let new_acc = fold_fn(builder, acc_var.clone(), elem_var);

                            // Update accumulator
                            let update_expr = builder.set_var(acc_var.clone(), new_acc);

                            // Increment index
                            let i_next = i_var.clone().to_u64() + StagedU64::constant(1);
                            builder.let1(update_expr, |builder, _| {
                                builder.set_var(i_var.clone(), Expr::U64(i_next))
                            })
                        })
                    },
                );

                // After loop, return the accumulator value
                builder.let1(loop_expr, |_builder, _| {
                    // Return the final accumulator value based on its type
                    match acc_type {
                        DataType::Prim(crate::num::PrimType::U64) => {
                            Expr::U64(acc_var.clone().to_u64())
                        }
                        DataType::Prim(crate::num::PrimType::I64) => {
                            Expr::I64(acc_var.clone().to_i64())
                        }
                        DataType::Bool => Expr::Bool(acc_var.clone().to_bool()),
                        _ => panic!("Unsupported accumulator type: {:?}", acc_type),
                    }
                })
            })
        })
    }
}

// =============================================================================
// EXTERNAL ITERATOR (FFI-BASED)
// =============================================================================

/// An iterator over values produced by an external Rust iterator via FFI.
///
/// This type wraps a pointer to a boxed iterator allocated on the Rust heap.
/// It calls external functions (`iter_next_*`, `iter_drop`) to advance and
/// clean up the iterator.
///
/// # Memory Safety
///
/// The iterator is automatically dropped when `for_each` completes. This is
/// implemented by generating code that calls `iter_drop` after the loop.
///
/// # Example
///
/// ```ignore
/// use tutorial::*;
/// use tutorial::ffi::*;
///
/// // Register FFI functions
/// let builder = CompilerBuilder::new().unwrap();
/// let builder = register_symbols!(builder, iter_create_range, iter_next_i64, iter_drop);
/// let mut compiler = builder.build().unwrap();
///
/// // Register signatures
/// compiler.register_external_signature(
///     "iter_create_range",
///     vec![DataType::I64, DataType::I64],
///     DataType::ExtPtr("IterBoxI64".to_string()),
/// );
/// compiler.register_external_signature(
///     "iter_next_i64",
///     vec![DataType::ExtPtr("IterBoxI64".to_string())],
///     DataType::I64,  // Returns packed OptionI64 as i64
/// );
/// compiler.register_external_signature(
///     "iter_drop",
///     vec![DataType::ExtPtr("IterBoxI64".to_string())],
///     DataType::Unit,
/// );
///
/// // Compile a function that sums a range
/// let mut compiled = compiler.compile_nary(
///     vec![DataType::I64, DataType::I64],
///     DataType::I64,
///     |builder, vars| {
///         let start = Expr::I64(StagedI64::variable(vars[0]));
///         let end = Expr::I64(StagedI64::variable(vars[1]));
///
///         StagedExternIter::range_i64(builder, start, end)
///             .sum(builder)
///     },
/// ).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct StagedExternIter {
    /// The variable holding the iterator pointer
    iter_ptr_var: Variable,
    /// Element type
    element_type: DataType,
    /// Name of the next function
    next_fn: String,
    /// Name of the drop function
    drop_fn: String,
}

impl StagedExternIter {
    /// Create a new external iterator from raw components.
    ///
    /// # Parameters
    ///
    /// - `iter_ptr_var`: Variable holding the iterator pointer
    /// - `element_type`: Type of elements yielded
    /// - `next_fn`: Name of the FFI function to get next element
    /// - `drop_fn`: Name of the FFI function to drop the iterator
    pub fn new(
        iter_ptr_var: Variable,
        element_type: DataType,
        next_fn: impl Into<String>,
        drop_fn: impl Into<String>,
    ) -> Self {
        StagedExternIter {
            iter_ptr_var,
            element_type,
            next_fn: next_fn.into(),
            drop_fn: drop_fn.into(),
        }
    }

    /// Create a range iterator from start to end (exclusive).
    ///
    /// This generates code that:
    /// 1. Calls `iter_create_range(start, end)` to create the iterator
    /// 2. Returns a `StagedExternIter` configured for i64 iteration
    ///
    /// # Note
    ///
    /// The FFI functions must be registered before compilation:
    /// - `iter_create_range`: (i64, i64) -> ExtPtr
    /// - `iter_next_i64`: (ExtPtr) -> i64 (packed OptionI64)
    /// - `iter_drop`: (ExtPtr) -> Unit
    pub fn create_range_i64(
        builder: &mut StagedBuilder,
        start: Expr,
        end: Expr,
    ) -> (Expr, StagedExternIterConfig) {
        // Call iter_create_range to get iterator pointer
        let create_expr = builder.call_external(
            "iter_create_range",
            vec![start, end],
            DataType::ExtPtr("IterBoxI64".to_string()),
        );

        let config = StagedExternIterConfig {
            element_type: DataType::I64,
            next_fn: "iter_next_i64".to_string(),
            drop_fn: "iter_drop".to_string(),
        };

        (create_expr, config)
    }

    /// Generate a for-each loop over the external iterator.
    ///
    /// This creates code that:
    /// 1. Repeatedly calls `next_fn` to get elements
    /// 2. Checks if the result is Some/None using the has_value field
    /// 3. Executes the body for each Some value
    /// 4. Calls `drop_fn` when iteration is complete
    ///
    /// # Memory Safety
    ///
    /// The drop function is always called after the loop, ensuring
    /// proper cleanup of the iterator.
    pub fn for_each<F>(self, builder: &mut StagedBuilder, body: F) -> Expr
    where
        F: FnOnce(&mut StagedBuilder, Var) -> Expr,
    {
        let iter_ptr = Var::new(
            self.iter_ptr_var,
            DataType::ExtPtr("IterBoxI64".to_string()),
        );
        let next_fn = self.next_fn.clone();
        let drop_fn = self.drop_fn.clone();
        let element_type = self.element_type.clone();

        // For OptionI64, the return is packed as: has_value (i8) at offset 0, value (i64) at offset 8
        // But since we return as a single i64, we use a different approach:
        // The next function returns the packed representation

        // Create a flag variable to track if we should continue
        builder.let_mut(Expr::Bool(StagedBool::constant(true)), move |builder, continue_var| {
            let loop_expr = builder.while_loop(
                |_builder| {
                    // Check continue flag
                    Expr::Bool(continue_var.clone().to_bool())
                },
                |builder| {
                    // Call next function - returns packed OptionI64
                    // For simplicity, we'll use a two-call approach:
                    // 1. iter_next_i64 returns OptionI64 which we handle in Rust

                    // Actually, we need to handle this differently.
                    // The OptionI64 struct is 16 bytes (i8 has_value + padding + i64 value)
                    // We can't easily decompose this in Cranelift without struct support.

                    // Alternative approach: Use a helper function that returns (has_value, value) separately
                    // For now, let's use a simpler approach with iter_next_i64_value and iter_has_next

                    // Simple approach: call iter_next_i64 which returns packed OptionI64
                    // Then extract has_value and value
                    let next_call = builder.call_external(
                        &next_fn,
                        vec![iter_ptr.clone().to_expr()],
                        DataType::ExtPtr("OptionI64".to_string()),
                    );

                    // Bind the result to extract fields
                    // For OptionI64, we need to check has_value field
                    // This requires loading from the struct offset

                    // Since direct struct field access isn't implemented, we'll use
                    // helper functions: iter_option_has_value and iter_option_value
                    builder.let1(next_call, |builder, option_var| {
                        // Check if has_value using helper
                        let has_value_call = builder.call_external(
                            "iter_option_has_value",
                            vec![option_var.clone().to_expr()],
                            DataType::Bool,
                        );

                        builder.let1(has_value_call, |builder, has_value_var| {
                            builder.if_then_else(
                                Expr::Bool(has_value_var.to_bool()),
                                |builder| {
                                    // Get the value
                                    let value_call = builder.call_external(
                                        "iter_option_value",
                                        vec![option_var.clone().to_expr()],
                                        element_type.clone(),
                                    );

                                    builder.let1(value_call, |builder, value_var| {
                                        // Execute user body
                                        body(builder, value_var)
                                    })
                                },
                                |builder| {
                                    // No more elements, set continue = false
                                    builder.set_var(continue_var.clone(), Expr::Bool(StagedBool::constant(false)))
                                },
                            )
                        })
                    })
                },
            );

            // After loop, drop the iterator
            builder.let1(loop_expr, |builder, _| {
                builder.call_external(
                    &drop_fn,
                    vec![iter_ptr.clone().to_expr()],
                    DataType::Unit,
                )
            })
        })
    }
}

/// Configuration for creating an external iterator.
#[derive(Debug, Clone)]
pub struct StagedExternIterConfig {
    pub element_type: DataType,
    pub next_fn: String,
    pub drop_fn: String,
}

// =============================================================================
// HELPER METHODS ON STAGEDBUILDER
// =============================================================================

impl StagedBuilder {
    /// Create an iterator over a staged array.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let iter = builder.iter_array(array);
    /// iter.for_each(builder, |builder, elem| {
    ///     // process elem
    ///     Expr::Unit
    /// });
    /// ```
    pub fn iter_array(&self, array: StagedArray) -> StagedArrayIter {
        StagedArrayIter::new(array)
    }

    /// Create a range iterator from start to end.
    ///
    /// This is a convenience method that creates an external iterator
    /// over the range [start, end).
    ///
    /// # Note
    ///
    /// Requires FFI functions to be registered:
    /// - `iter_create_range`
    /// - `iter_next_i64`
    /// - `iter_option_has_value`
    /// - `iter_option_value`
    /// - `iter_drop`
    pub fn iter_range_i64(
        &mut self,
        start: Expr,
        end: Expr,
    ) -> (Expr, StagedExternIterConfig) {
        StagedExternIter::create_range_i64(self, start, end)
    }
}

// =============================================================================
// FFI HELPER FUNCTIONS
// =============================================================================

/// Check if an OptionI64 has a value.
///
/// This is a helper function for the external iterator to check
/// if the option contains a value.
#[no_mangle]
pub extern "C" fn iter_option_has_value(option: OptionI64) -> i8 {
    option.has_value
}

/// Get the value from an OptionI64.
///
/// # Safety
///
/// This should only be called when `iter_option_has_value` returns non-zero.
#[no_mangle]
pub extern "C" fn iter_option_value(option: OptionI64) -> i64 {
    option.value
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffi::{iter_create_range, iter_drop, iter_next_i64};
    use crate::{register_symbols, Compiler, CompilerBuilder, ScalarValue};
    use std::sync::atomic::{AtomicUsize, Ordering};

    // Counter to track iterator drops for testing
    static DROP_COUNT: AtomicUsize = AtomicUsize::new(0);

    // Test iterator that increments drop counter
    pub struct TestIterBoxI64 {
        inner: Box<dyn Iterator<Item = i64>>,
    }

    #[no_mangle]
    pub extern "C" fn test_iter_create_range(start: i64, end: i64) -> *mut TestIterBoxI64 {
        Box::into_raw(Box::new(TestIterBoxI64 {
            inner: Box::new(start..end),
        }))
    }

    #[no_mangle]
    pub extern "C" fn test_iter_next_i64(ptr: *mut TestIterBoxI64) -> OptionI64 {
        if ptr.is_null() {
            return OptionI64::none();
        }
        let iter_box = unsafe { &mut *ptr };
        match iter_box.inner.next() {
            Some(val) => OptionI64::some(val),
            None => OptionI64::none(),
        }
    }

    #[no_mangle]
    pub extern "C" fn test_iter_drop(ptr: *mut TestIterBoxI64) {
        if !ptr.is_null() {
            DROP_COUNT.fetch_add(1, Ordering::SeqCst);
            unsafe {
                drop(Box::from_raw(ptr));
            }
        }
    }

    fn get_drop_count() -> usize {
        DROP_COUNT.load(Ordering::SeqCst)
    }

    // -------------------------------------------------------------------------
    // STAGED ARRAY ITERATOR TESTS
    // -------------------------------------------------------------------------

    #[test]
    fn test_array_iter_sum() {
        // Test summing array elements using for_each
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::arr(DataType::U64)],
                DataType::U64,
                |builder, vars| {
                    let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
                    let iter = StagedArrayIter::new(input);

                    // Sum using fold
                    iter.fold(
                        builder,
                        Expr::U64(StagedU64::constant(0)),
                        |_builder, acc_var, elem_var| {
                            let acc = acc_var.to_u64();
                            let elem = elem_var.to_u64();
                            Expr::U64(acc + elem)
                        },
                    )
                },
            )
            .unwrap();

        // Test with actual data
        let input_data: Vec<u64> = vec![1, 2, 3, 4, 5];
        let args = vec![
            input_data.as_ptr() as u64,
            input_data.len() as u64,
        ];

        let result = compiled.call_u64(&args);
        assert_eq!(result, 15); // 1 + 2 + 3 + 4 + 5 = 15
    }

    #[test]
    fn test_array_iter_empty() {
        // Test iterating over empty array
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::arr(DataType::U64)],
                DataType::U64,
                |builder, vars| {
                    let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
                    let iter = StagedArrayIter::new(input);

                    iter.fold(
                        builder,
                        Expr::U64(StagedU64::constant(42)),
                        |_builder, acc_var, elem_var| {
                            let acc = acc_var.to_u64();
                            let elem = elem_var.to_u64();
                            Expr::U64(acc + elem)
                        },
                    )
                },
            )
            .unwrap();

        // Empty array
        let input_data: Vec<u64> = vec![];
        let args = vec![
            input_data.as_ptr() as u64,
            input_data.len() as u64,
        ];

        let result = compiled.call_u64(&args);
        assert_eq!(result, 42); // Initial value unchanged
    }

    #[test]
    fn test_array_iter_map_to() {
        // Test doubling each element using map_to
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![
                    DataType::arr(DataType::U64),      // input
                    DataType::mut_arr(DataType::U64),  // output
                ],
                DataType::U64,
                |builder, vars| {
                    let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
                    let output = StagedArray::new(vars[2], vars[3], DataType::U64, true);
                    let len = input.len();

                    let iter = StagedArrayIter::new(input);

                    let map_expr = iter.map_to(builder, output, |_builder, elem_var| {
                        let elem = elem_var.to_u64();
                        Expr::U64(elem * StagedU64::constant(2))
                    });

                    builder.let1(map_expr, |_, _| Expr::U64(len))
                },
            )
            .unwrap();

        let input_data: Vec<u64> = vec![1, 2, 3, 4, 5];
        let mut output_data: Vec<u64> = vec![0; 5];

        let args = vec![
            input_data.as_ptr() as u64,
            input_data.len() as u64,
            output_data.as_mut_ptr() as u64,
            output_data.len() as u64,
        ];

        let result = compiled.call_u64(&args);
        assert_eq!(result, 5);
        assert_eq!(output_data, vec![2, 4, 6, 8, 10]);
    }

    #[test]
    fn test_array_iter_for_each() {
        // Test counting elements greater than 3
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary(
                vec![DataType::arr(DataType::U64)],
                DataType::U64,
                |builder, vars| {
                    let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
                    let iter = StagedArrayIter::new(input);

                    builder.let_mut(Expr::U64(StagedU64::constant(0)), |builder, count_var| {
                        let loop_expr = iter.for_each(builder, |builder, elem_var| {
                            let elem = elem_var.to_u64();
                            let three = StagedU64::constant(3);
                            let cond = elem.gt(&three);

                            builder.if_then_else(
                                Expr::Bool(cond),
                                |builder| {
                                    let count = count_var.clone().to_u64();
                                    builder.set_var(
                                        count_var.clone(),
                                        Expr::U64(count + StagedU64::constant(1)),
                                    )
                                },
                                |builder| {
                                    // Both branches must have the same type - return Unit
                                    builder.set_var(count_var.clone(), Expr::U64(count_var.clone().to_u64()))
                                },
                            )
                        });

                        builder.let1(loop_expr, |_, _| Expr::U64(count_var.to_u64()))
                    })
                },
            )
            .unwrap();

        let input_data: Vec<u64> = vec![1, 5, 2, 7, 3, 9, 4];
        let args = vec![
            input_data.as_ptr() as u64,
            input_data.len() as u64,
        ];

        let result = compiled.call_u64(&args);
        assert_eq!(result, 4); // 5, 7, 9, 4 are > 3
    }

    #[test]
    fn test_array_iter_with_index() {
        // Test that for_each_with_index provides correct indices
        let mut compiler = Compiler::new().unwrap();
        let compiled = compiler
            .compile_nary(
                vec![
                    DataType::arr(DataType::U64),      // input
                    DataType::mut_arr(DataType::U64),  // output: will store indices of elements > 3
                ],
                DataType::U64,
                |builder, vars| {
                    let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
                    let _output = StagedArray::new(vars[2], vars[3], DataType::U64, true);

                    // Sum of all indices where element > 3
                    builder.let_mut(Expr::U64(StagedU64::constant(0)), |builder, sum_var| {
                        let iter = StagedArrayIter::new(input);
                        let loop_expr = iter.for_each_with_index(builder, |builder, idx_var, elem_var| {
                            let elem = elem_var.to_u64();
                            let idx = idx_var.to_u64();
                            let three = StagedU64::constant(3);

                            builder.if_then_else(
                                Expr::Bool(elem.gt(&three)),
                                |builder| {
                                    let sum = sum_var.clone().to_u64();
                                    builder.set_var(sum_var.clone(), Expr::U64(sum + idx))
                                },
                                |builder| {
                                    // Both branches must have the same type - return Unit
                                    builder.set_var(sum_var.clone(), Expr::U64(sum_var.clone().to_u64()))
                                },
                            )
                        });

                        builder.let1(loop_expr, |_, _| Expr::U64(sum_var.to_u64()))
                    })
                },
            )
            .unwrap();

        // Elements: [1, 5, 2, 7, 3, 9]
        // Indices:   0  1  2  3  4  5
        // > 3:          1     3     5
        // Sum of indices: 1 + 3 + 5 = 9
        let input_data: Vec<u64> = vec![1, 5, 2, 7, 3, 9];
        let mut output_data: Vec<u64> = vec![0; 6];

        let args = vec![
            input_data.as_ptr() as u64,
            input_data.len() as u64,
            output_data.as_mut_ptr() as u64,
            output_data.len() as u64,
        ];

        let result = compiled.call_u64(&args);
        assert_eq!(result, 9);
    }

    #[test]
    fn test_array_iter_product() {
        // Test computing product of array elements
        let mut compiler = Compiler::new().unwrap();
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::arr(DataType::U64)],
                DataType::U64,
                |builder, vars| {
                    let input = StagedArray::new(vars[0], vars[1], DataType::U64, false);
                    let iter = StagedArrayIter::new(input);

                    iter.fold(
                        builder,
                        Expr::U64(StagedU64::constant(1)),
                        |_builder, acc_var, elem_var| {
                            let acc = acc_var.to_u64();
                            let elem = elem_var.to_u64();
                            Expr::U64(acc * elem)
                        },
                    )
                },
            )
            .unwrap();

        let input_data: Vec<u64> = vec![2, 3, 4];
        let args = vec![
            input_data.as_ptr() as u64,
            input_data.len() as u64,
        ];

        let result = compiled.call_u64(&args);
        assert_eq!(result, 24); // 2 * 3 * 4 = 24
    }

    // -------------------------------------------------------------------------
    // EXTERNAL ITERATOR TESTS
    // -------------------------------------------------------------------------

    #[test]
    fn test_ffi_option_helpers() {
        // Test the option helper functions directly
        let some_val = OptionI64::some(42);
        assert_eq!(iter_option_has_value(some_val), 1);
        assert_eq!(iter_option_value(some_val), 42);

        let none_val = OptionI64::none();
        assert_eq!(iter_option_has_value(none_val), 0);
    }

    #[test]
    fn test_external_iter_drop_called() {
        // Test that the drop function is called when using external iterator
        // Since tests run in parallel, we verify drop increments the counter

        // Create an iterator
        let iter_ptr = test_iter_create_range(0, 5);

        // Iterate through all elements
        loop {
            let option = test_iter_next_i64(iter_ptr);
            if option.to_option().is_none() {
                break;
            }
        }

        // Capture count before drop
        let before_drop = get_drop_count();

        // Drop the iterator
        test_iter_drop(iter_ptr);

        let after_drop = get_drop_count();
        // Verify the drop incremented the counter
        assert!(after_drop > before_drop, "Drop should have incremented the counter");
    }

    #[test]
    fn test_external_iter_multiple_drops() {
        // Test that multiple iterators are tracked correctly
        // Verify that each drop increments the counter
        for _ in 0..5 {
            let iter_ptr = test_iter_create_range(0, 3);

            // Just drain the iterator
            while test_iter_next_i64(iter_ptr).to_option().is_some() {}

            let before = get_drop_count();
            test_iter_drop(iter_ptr);
            let after = get_drop_count();

            assert!(after > before, "Each drop should increment the counter");
        }
    }

    #[test]
    fn test_external_iter_early_drop() {
        // Test that iterator is properly dropped even without full iteration
        let iter_ptr = test_iter_create_range(0, 1000);

        // Only read a few elements - don't exhaust the iterator
        for _ in 0..5 {
            let _ = test_iter_next_i64(iter_ptr);
        }

        // Capture count before drop
        let before_drop = get_drop_count();

        // Drop early (without exhausting the iterator)
        test_iter_drop(iter_ptr);

        let after_drop = get_drop_count();
        // Verify the drop incremented the counter
        assert!(after_drop > before_drop, "Iterator should be dropped even with early exit");
    }

    #[test]
    fn test_staged_extern_iter_config() {
        // Test the configuration struct
        let config = StagedExternIterConfig {
            element_type: DataType::I64,
            next_fn: "iter_next_i64".to_string(),
            drop_fn: "iter_drop".to_string(),
        };

        assert_eq!(config.element_type, DataType::I64);
        assert_eq!(config.next_fn, "iter_next_i64");
        assert_eq!(config.drop_fn, "iter_drop");
    }

    // Note: Full JIT compilation tests for external iterators require
    // registering all FFI functions with the compiler. These are more
    // integration tests and depend on the full compilation pipeline.

    #[test]
    fn test_extern_iter_jit_sum() {
        // Test external iterator with full JIT compilation
        // Sum the range [0, 5) = 0 + 1 + 2 + 3 + 4 = 10

        let builder = CompilerBuilder::new().unwrap();
        let builder = register_symbols!(
            builder,
            iter_create_range,
            iter_next_i64,
            iter_drop,
            iter_option_has_value,
            iter_option_value
        );
        let mut compiler = builder.build().unwrap();

        // Register signatures
        compiler.register_external_signature(
            "iter_create_range",
            vec![DataType::I64, DataType::I64],
            DataType::ExtPtr("IterBoxI64".to_string()),
        );
        compiler.register_external_signature(
            "iter_next_i64",
            vec![DataType::ExtPtr("IterBoxI64".to_string())],
            DataType::ExtPtr("OptionI64".to_string()),
        );
        compiler.register_external_signature(
            "iter_option_has_value",
            vec![DataType::ExtPtr("OptionI64".to_string())],
            DataType::Bool,
        );
        compiler.register_external_signature(
            "iter_option_value",
            vec![DataType::ExtPtr("OptionI64".to_string())],
            DataType::I64,
        );
        compiler.register_external_signature(
            "iter_drop",
            vec![DataType::ExtPtr("IterBoxI64".to_string())],
            DataType::Unit,
        );

        // Compile a function that sums a range
        let mut compiled = compiler
            .compile_nary(
                vec![DataType::I64, DataType::I64],
                DataType::I64,
                |builder, vars| {
                    let start = Expr::I64(StagedI64::variable(vars[0]));
                    let end = Expr::I64(StagedI64::variable(vars[1]));

                    // Create iterator
                    let (create_expr, config) = StagedExternIter::create_range_i64(builder, start, end);

                    // Bind iterator pointer and sum elements
                    builder.let1(create_expr, |builder, iter_ptr_var| {
                        let iter = StagedExternIter::new(
                            iter_ptr_var.var,
                            config.element_type,
                            config.next_fn,
                            config.drop_fn,
                        );

                        // Sum using mutable accumulator
                        builder.let_mut(Expr::I64(StagedI64::constant(0)), |builder, sum_var| {
                            let loop_expr = iter.for_each(builder, |builder, elem_var| {
                                let sum = sum_var.clone().to_i64();
                                let elem = elem_var.to_i64();
                                builder.set_var(sum_var.clone(), Expr::I64(sum + elem))
                            });

                            builder.let1(loop_expr, |_, _| Expr::I64(sum_var.to_i64()))
                        })
                    })
                },
            )
            .unwrap();

        // Test: sum of range [0, 5) = 10
        let result = compiled.call(&[ScalarValue::I64(0), ScalarValue::I64(5)]).unwrap();
        assert_eq!(result.as_i64_unchecked(), 10);

        // Test: sum of range [1, 4) = 1 + 2 + 3 = 6
        let result = compiled.call(&[ScalarValue::I64(1), ScalarValue::I64(4)]).unwrap();
        assert_eq!(result.as_i64_unchecked(), 6);

        // Test: empty range [5, 5) = 0
        let result = compiled.call(&[ScalarValue::I64(5), ScalarValue::I64(5)]).unwrap();
        assert_eq!(result.as_i64_unchecked(), 0);
    }
}
