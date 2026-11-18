//! First-class function support for staged computations.
//!
//! This module provides:
//! - `Compiler`: The central coordinator that owns function and variable definitions
//! - `FunRefN<T0, ..., OUT>`: Type-safe handles to function definitions (N = 0..8)
//! - `CallN`: Function call expressions
//! - `Compiled<T>`: The result of compilation, owns the JIT module
//!
//! # Multi-Parameter Functions
//!
//! Functions with 0-8 parameters are supported via `fun0`, `fun1`, ..., `fun8`.
//! Each returns a type-safe `FunRefN` that encodes the parameter and return types.
//!
//! # Aggregate Values
//!
//! Generated functions use a private, platform-independent storage-pointer ABI.
//! Rust-facing wrappers preserve value semantics without exposing platform
//! aggregate classification to Cranelift.

pub use crate::func_impl::*;
use crate::{
    staged::{CompilationContext, Value, ValueId},
    types::{RuntimeParam, RuntimeResult, StagedType},
};
use cranelift_jit::JITModule;
use std::{marker::PhantomData, mem::MaybeUninit, sync::Arc};

// =============================================================================
// Internal: FunDef - Stored function definition
// =============================================================================

/// Internal storage for a function definition (type-erased body)
pub(crate) struct FunDef {
    pub name: String,
    /// The body expression, type-erased but we know its signature
    pub body: Box<dyn FnOnce(&mut CompilationContext) -> Value>,
    /// Type info for each parameter (supports 0..N parameters)
    pub param_infos: Vec<TypeInfo>,
    /// Return type info
    pub return_info: TypeInfo,
    /// Variable IDs for each parameter (one per logical parameter)
    pub param_var_ids: Vec<usize>,
}

/// Stored metadata for an external function
pub(crate) struct ExternFnDef {
    /// Unique link name (`NAME` + the fn pointer), so distinct monomorphizations
    /// of a generic extern fn don't collide in cranelift's name-keyed symbol map.
    pub name: String,
    pub num_params: usize,
    pub fn_ptr: *const u8,
}

/// Emit one function body under the private storage-pointer ABI, **backend-neutrally**.
///
/// The ABI is uniform: `params` is one storage pointer per logical argument followed by a
/// single output pointer, and the function returns `void`. This unpacks each argument from
/// its storage pointer into a shape-aware variable binding,
/// runs `body` to produce the result, and writes the result back through the output pointer.
/// Every step goes through the neutral [`Backend`](crate::staged::Backend) ops on `ctx`, so
/// the same code drives Cranelift and MLIR; the caller supplies the (backend-specific)
/// function entry and terminating `return`.
pub(crate) fn emit_function_body(
    ctx: &mut CompilationContext,
    params: &[ValueId],
    param_infos: &[TypeInfo],
    param_var_ids: &[usize],
    body: impl FnOnce(&mut CompilationContext) -> Value,
    return_info: &TypeInfo,
) {
    for (index, info) in param_infos.iter().enumerate() {
        let var_id = param_var_ids[index];
        let storage_ptr = params[index];
        // A value held in memory stays at its storage; any other is loaded once into its
        // leaves — the one boundary where an argument touches memory on the way in.
        let value = ctx.load_shape(info.shape, storage_ptr);
        ctx.assign_var_layout(var_id, value, info.shape, "parameter", false);
    }

    let result = body(ctx);

    let output_ptr = params[param_infos.len()];
    ctx.store_shape(return_info.shape, output_ptr, result);
}

// =============================================================================
// Compiled<T>: The result of compilation
// =============================================================================

/// A compiled expression that owns its JIT module and executable memory.
///
/// Use [`run`](Self::run) for a plain expression or the arity-specific `call`
/// methods for a compiled staged function. Executable memory is reclaimed after
/// this value and every [`CompiledFn`] cloned from it have been dropped.
pub struct Compiled<T: StagedType> {
    executable: Arc<FrozenExecutable>,
    main: unsafe extern "C" fn(*mut u8),
    /// Each function's optimised size; see [`Compiled::sizes`].
    sizes: Vec<(String, usize)>,
    _signature: PhantomData<T>,
}

/// The backend-specific JIT resource kept alive by the shared frozen owner.
pub(crate) enum Executable {
    Cranelift(Box<JITModule>),
    #[cfg(feature = "llvm")]
    Mlir(crate::llvm::MlirExecutable),
}

/// A finalized JIT resource. Once wrapped here, no module-building, symbol-registration,
/// lookup, or packed-invocation API is accessible: callers can only execute already-resolved
/// native entry points. The `Arc` shared by `Compiled` and `CompiledFn` prevents destruction
/// while any entry point can still be called.
pub(crate) struct FrozenExecutable {
    executable: Option<Executable>,
}

impl FrozenExecutable {
    pub(crate) fn new(executable: Executable) -> Self {
        Self {
            executable: Some(executable),
        }
    }
}

// SAFETY: publication happens only after finalization and native entry-point lookup. The
// wrapped JIT objects are never accessed through `FrozenExecutable`; they solely own the
// executable allocation until exclusive `Drop`. Calls go directly through immutable native
// function pointers, and `Arc` ensures `Drop` cannot race a call. This additionally relies on
// the Cranelift and LLVM/MLIR JIT destructors being valid on a thread other than the creator.
// The parallel LLVM property test exercises concurrent execution and last-owner destruction
// on worker threads; the full differential suite exercises the same wrapper for Cranelift.
unsafe impl Send for FrozenExecutable {}
// SAFETY: see the `Send` implementation. Sharing this wrapper does not share access to either
// JIT API; it shares only ownership of finalized executable memory.
unsafe impl Sync for FrozenExecutable {}

impl Drop for FrozenExecutable {
    fn drop(&mut self) {
        // `Arc` gives the last owner exclusive destruction, so no safe callable can remain.
        // Escaped pointers are still governed by `as_fn_unchecked`'s safety contract.
        match self.executable.take() {
            Some(Executable::Cranelift(module)) => unsafe { (*module).free_memory() },
            #[cfg(feature = "llvm")]
            Some(Executable::Mlir(executable)) => drop(executable),
            None => {}
        }
    }
}

impl<T: StagedType> Compiled<T> {
    pub(crate) fn new(
        executable: Arc<FrozenExecutable>,
        main: unsafe extern "C" fn(*mut u8),
        sizes: Vec<(String, usize)>,
    ) -> Self {
        Self {
            executable,
            main,
            sizes,
            _signature: PhantomData,
        }
    }

    /// Each function's name and its number of instructions after optimisation: a
    /// measure of code size that tests can bound. Cranelift only; empty for LLVM.
    pub fn sizes(&self) -> &[(String, usize)] {
        &self.sizes
    }

    /// Execute the compiled code and return the result.
    pub fn run(&self) -> T::RuntimeValue {
        let mut output = MaybeUninit::<T::RuntimeValue>::uninit();
        // SAFETY: `compile` creates `__main__` with the canonical one-output-
        // pointer signature and writes a valid `T::RuntimeValue` before return.
        unsafe {
            (self.main)(output.as_mut_ptr().cast());
            output.assume_init()
        }
    }

    /// Read the generated function address returned by a function-valued
    /// `__main__` trampoline.
    unsafe fn function_entry_ptr(&self) -> unsafe extern "C" fn() {
        let mut output = MaybeUninit::<unsafe extern "C" fn()>::uninit();
        unsafe {
            (self.main)(output.as_mut_ptr().cast());
            output.assume_init()
        }
    }
}

/// A callable entry point that shares ownership of its finalized executable module.
///
/// Cloning this value clones an internal [`Arc`] lease, so it can be sent to worker threads
/// and remains callable after the original [`Compiled`] is dropped. The function pointer is
/// deliberately private and this type does not implement `Deref`; use the arity-specific
/// `call` method, or `Compiled::as_fn_unchecked` when a foreign API genuinely requires a bare
/// untracked pointer.
///
/// Reference results are bounded by both the invocation arguments and the
/// compiled module. They cannot escape a shorter-lived argument:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
///
/// let mut compiler = Compiler::new();
/// let identity = compiler.fun1("identity", |_ctx, value: Var<SRef<i64>>| value);
/// let compiled = compiler.compile(identity).unwrap();
/// let escaped = {
///     let value = 42i64;
///     compiled.call(&value)
/// };
/// assert_eq!(*escaped, 42);
/// ```
///
/// Like a Rust `fn`, a compiled function is contravariant in its arguments. A
/// function that accepts shorter-lived data can stand in for one that needs
/// longer-lived data:
///
/// ```
/// use rust_lms::prelude::*;
/// use rust_lms::slice::BorrowedSlice;
///
/// type Kernel<'arg> = CompiledFn<FunType1<BorrowedSlice<'arg, u64>, BorrowedSlice<'static, u64>>>;
///
/// fn lengthen_argument<'a>(f: Kernel<'a>) -> Kernel<'static> {
///     f
/// }
/// ```
///
/// but not the reverse: a kernel staged for `'static` input, whose result may
/// be that input, cannot be called with shorter-lived data:
///
/// ```compile_fail
/// use rust_lms::prelude::*;
/// use rust_lms::slice::BorrowedSlice;
///
/// type Kernel<'arg> = CompiledFn<FunType1<BorrowedSlice<'arg, u64>, BorrowedSlice<'static, u64>>>;
///
/// fn shorten_argument<'a>(f: Kernel<'static>) -> Kernel<'a> {
///     f
/// }
/// ```
#[must_use = "a compiled entry point does nothing until it is called"]
pub struct CompiledFn<F> {
    function: unsafe extern "C" fn(),
    executable: Arc<FrozenExecutable>,
    _signature: PhantomData<F>,
}

impl<F> Clone for CompiledFn<F> {
    fn clone(&self) -> Self {
        Self {
            function: self.function,
            executable: Arc::clone(&self.executable),
            _signature: PhantomData,
        }
    }
}

macro_rules! raw_argument_pointer {
    ($type:ident) => {
        *const u8
    };
}

// Generate owning entry points and direct calls for every function arity.
macro_rules! impl_compiled_fn {
    // Base case: zero parameters
    (0, $FunType:ident) => {
        impl<OUT: RuntimeResult> CompiledFn<$FunType<OUT>> {
            /// Invoke the entry point with a result lifetime bounded by this owning handle.
            pub fn call<'call>(&'call self) -> OUT::Output<'call>
            {
                let mut output = MaybeUninit::<OUT::Output<'call>>::uninit();
                // SAFETY: RuntimeResult witnesses the output storage layout,
                // and the wrapper keeps the generated code live for 'call.
                let function: unsafe extern "C" fn(*mut u8) =
                    unsafe { std::mem::transmute(self.function) };
                unsafe {
                    function(output.as_mut_ptr().cast());
                    output.assume_init()
                }
            }
        }

        impl<OUT: RuntimeResult> Compiled<$FunType<OUT>> {
            /// Create an owning, cloneable callable entry point.
            ///
            /// The internal executable lease makes temporaries and worker-thread use safe:
            ///
            /// ```
            /// use rust_lms::prelude::*;
            ///
            /// let mut compiler = Compiler::new();
            /// let function = compiler.fun0("one", |_ctx| Const::<i64>::new(1));
            /// let entry = compiler.compile(function).unwrap().as_fn();
            /// assert_eq!(entry.call(), 1);
            /// ```
            pub fn as_fn(&self) -> CompiledFn<$FunType<OUT>> {
                CompiledFn {
                    // SAFETY: the returned wrapper owns an executable lease and never exposes
                    // the bare pointer.
                    function: unsafe { self.function_entry_ptr() },
                    executable: Arc::clone(&self.executable),
                    _signature: PhantomData,
                }
            }

            /// Invoke the compiled function while borrowing its owner.
            pub fn call<'call>(&'call self) -> OUT::Output<'call> {
                let mut output = MaybeUninit::<OUT::Output<'call>>::uninit();
                // SAFETY: `self` keeps the executable live for `'call`, and RuntimeResult
                // witnesses the output storage layout.
                let function: unsafe extern "C" fn(*mut u8) =
                    unsafe { std::mem::transmute(self.function_entry_ptr()) };
                unsafe {
                    function(output.as_mut_ptr().cast());
                    output.assume_init()
                }
            }

            /// Extract the compiled function as an untracked function pointer.
            ///
            /// # Safety
            ///
            /// The returned pointer must never be invoked after the executable allocation is
            /// released; keeping `self` or an owning [`CompiledFn`] alive is sufficient. Its
            /// argument must point to writable, properly aligned storage for
            /// `OUT::RuntimeValue`.
            pub unsafe fn as_fn_unchecked(
                &self,
            ) -> unsafe extern "C" fn(*mut u8) {
                unsafe { std::mem::transmute(self.function_entry_ptr()) }
            }
        }
    };
    // N parameters (N >= 1)
    ($n:tt, $FunType:ident, [$($T:ident : $arg:ident),+]) => {
        impl<$($T: RuntimeParam,)+ OUT: RuntimeResult>
            CompiledFn<$FunType<$($T,)+ OUT>>
        {
            /// Invoke the entry point with one fresh lifetime shared by its reference
            /// arguments, result, and this owning handle.
            #[allow(clippy::too_many_arguments)]
            pub fn call<'call>(&'call self, $($arg: $T::Arg<'call>),+) -> OUT::Output<'call>
            {
                let mut output = MaybeUninit::<OUT::Output<'call>>::uninit();
                // SAFETY: RuntimeParam and RuntimeResult witness the storage
                // layouts. The shared 'call lifetime enforces the staged
                // reference contract at each invocation.
                let function: unsafe extern "C" fn(
                    $(raw_argument_pointer!($T)),+,
                    *mut u8,
                ) =
                    unsafe { std::mem::transmute(self.function) };
                unsafe {
                    function(
                        $(std::ptr::from_ref(&$arg).cast::<u8>()),+,
                        output.as_mut_ptr().cast(),
                    );
                    output.assume_init()
                }
            }
        }

        impl<$($T: RuntimeParam,)+ OUT: RuntimeResult>
            Compiled<$FunType<$($T,)+ OUT>>
        {
            /// Create an owning, cloneable callable entry point.
            pub fn as_fn(&self) -> CompiledFn<$FunType<$($T,)+ OUT>> {
                CompiledFn {
                    // SAFETY: the returned wrapper owns an executable lease and never exposes
                    // the bare pointer.
                    function: unsafe { self.function_entry_ptr() },
                    executable: Arc::clone(&self.executable),
                    _signature: PhantomData,
                }
            }

            /// Invoke the compiled function while borrowing its owner.
            #[allow(clippy::too_many_arguments)]
            pub fn call<'call>(&'call self, $($arg: $T::Arg<'call>),+) -> OUT::Output<'call> {
                let mut output = MaybeUninit::<OUT::Output<'call>>::uninit();
                // SAFETY: `self` keeps the executable live for `'call`; RuntimeParam and
                // RuntimeResult witness the argument and output storage layouts.
                let function: unsafe extern "C" fn(
                    $(raw_argument_pointer!($T)),+,
                    *mut u8,
                ) = unsafe { std::mem::transmute(self.function_entry_ptr()) };
                unsafe {
                    function(
                        $(std::ptr::from_ref(&$arg).cast::<u8>()),+,
                        output.as_mut_ptr().cast(),
                    );
                    output.assume_init()
                }
            }

            /// Extract the compiled function as an untracked function pointer.
            ///
            /// # Safety
            ///
            /// The returned pointer must never be invoked after the executable allocation is
            /// released; keeping `self` or an owning [`CompiledFn`] alive is sufficient. Each
            /// input must point to the exact runtime representation of its corresponding staged
            /// parameter, and the final pointer must reference writable, properly aligned output
            /// storage.
            pub unsafe fn as_fn_unchecked(
                &self,
            ) -> unsafe extern "C" fn($(raw_argument_pointer!($T)),+, *mut u8) {
                unsafe { std::mem::transmute(self.function_entry_ptr()) }
            }
        }
    };
}

impl_compiled_fn!(0, FunType0);
impl_compiled_fn!(1, FunType1, [A: a]);
impl_compiled_fn!(2, FunType2, [A: a, B: b]);
impl_compiled_fn!(3, FunType3, [A: a, B: b, C: c]);
impl_compiled_fn!(4, FunType4, [A: a, B: b, C: c, D: d]);
impl_compiled_fn!(5, FunType5, [A: a, B: b, C: c, D: d, E: e]);
impl_compiled_fn!(6, FunType6, [A: a, B: b, C: c, D: d, E: e, F: f]);
impl_compiled_fn!(7, FunType7, [A: a, B: b, C: c, D: d, E: e, F: f, G: g]);
impl_compiled_fn!(8, FunType8, [A: a, B: b, C: c, D: d, E: e, F: f, G: g, H: h]);
