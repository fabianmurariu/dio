//! # rust-lms: Type-Safe Staged Computation in Rust
//!
//! A Rust implementation of multi-stage programming inspired by Scala LMS (Lightweight Modular Staging).
//!
//! ## Overview
//!
//! This library provides a type-safe framework for building staged computations that compile
//! to efficient machine code via Cranelift. Key features:
//!
//! - **Compile-time type safety**: Invalid operations are caught at compile time
//! - **Zero-cost abstractions**: `Var<T>` and `Const<T>` are Copy when possible
//! - **Heterogeneous operations**: Operations can change output types (e.g., comparison → bool)
//! - **Full composability**: Any `Staged` value works anywhere a `Staged` value is expected
//! - **Dynamic dispatch support**: Boxing via `.boxed()` when needed
//!
//! ## Quick Example
//!
//! ```ignore
//! use rust_lms::prelude::*;
//! use cranelift_frontend::Variable;
//!
//! // Create variables and constants
//! let x = Var::<i64>::new(Variable::from_u32(0));
//! let five = Const::<i64>::new(5);
//! let two = Const::<i64>::new(2);
//!
//! // Build expressions: (x + 5) * 2
//! let expr = mul(add(x, five), two);
//!
//! // x is Copy, so we can reuse it!
//! let expr2 = add(x, x);
//!
//! // Comparisons change type to Bool
//! let comparison = lt(x, Const::new(100));
//!
//! // This won't compile - type mismatch caught at compile time!
//! // let bad = add(x, comparison);  // ERROR: can't add I64 and Bool
//! ```
//!
//! ## Architecture
//!
//! ### Core Traits
//!
//! - [`Staged`](staged::Staged): Anything that can generate runtime code
//! - [`StagedType`](types::StagedType): Types that can participate in staged computation
//!
//! ### Value Types
//!
//! - [`Var<T>`](staged::Var): Typed variable references (Copy-able)
//! - [`Const<T>`](staged::Const): Typed constants (Copy-able)
//!
//! ### Type Markers
//!
//! - [`i64`](types::i64), [`u64`](u64): Integer types
//! - [`f64`](f64): Floating-point type
//! - [`bool`](bool): Boolean type
//!
//! ### Operations
//!
//! - Arithmetic: [`Add`](num::Add), [`Sub`](num::Sub), [`Mul`](num::Mul), [`Div`](num::Div)
//! - Comparison: [`Lt`](num::Lt), [`Eq`](num::Eq)
//!
//! ## Design Principles
//!
//! ### 1. Separation of Values and Operations
//!
//! Unlike traditional expression trees where operations are part of the value enum,
//! this design separates:
//! - **Values**: `Var<T>`, `Const<T>` - only represent "pure" values
//! - **Operations**: `Add<L,R>`, `Lt<L,R>` - separate structs that also implement `Staged`
//!
//! ### 2. Type-Level Constraints
//!
//! Operations use trait bounds to ensure type safety:
//!
//! ```ignore
//! impl<L, R, T> Staged for Add<L, R>
//! where
//!     L: Staged<Out = T>,  // Left must produce type T
//!     R: Staged<Out = T>,  // Right must produce type T
//!     T: StagedType + SupportsAdd,  // T must support addition
//! {
//!     type Out = T;  // Result is also type T
//! }
//! ```
//!
//! ### 3. Heterogeneous Operations
//!
//! Some operations change types:
//!
//! ```ignore
//! impl<L, R, T> Staged for Lt<L, R>
//! where
//!     L: Staged<Out = T>,
//!     R: Staged<Out = T>,
//!     T: StagedType + SupportsComparison,
//! {
//!     type Out = bool;  // Always returns Bool, not T!
//! }
//! ```

// Let `#[derive(StagedType)]`'s absolute `::rust_lms::…` paths resolve when the
// derive is used *inside* this crate (e.g. iter::zip::ZipItem), not just downstream.
extern crate self as rust_lms;

pub mod control;
pub mod ffi;
pub mod func;
pub(crate) mod func_def;
pub(crate) mod func_impl;
pub mod iter;
pub mod num;
pub mod opaque;
pub mod option;
pub mod pool;
pub mod refer;
pub mod slice;
pub mod staged;
pub mod staged_opt;
pub mod r#struct;
pub mod tuple;
pub mod types;

/// Internal utilities used by proc macros.
/// Do not use directly.
#[doc(hidden)]
pub mod _internal {
    pub use crate::r#struct::Field;
}

/// Commonly used types and traits
pub mod prelude {
    pub use crate::control::{
        if_then, if_then_else, not, while_loop, IfThen, IfThenElse, Not, While,
    };
    pub use crate::ffi::{
        call_extern0, call_extern0_unchecked, call_extern1, call_extern1_unchecked, call_extern2,
        call_extern2_unchecked, call_extern3, call_extern3_unchecked, call_extern4,
        call_extern4_unchecked, slice_from_raw_parts, slice_ref_from_raw_parts, stack_alloc,
        stack_bytes, ExternArgs, ExternFn, ExternRef, FatSlice, FatSliceMut, FatSliceMutType,
        FatSliceType, FfiSlice, FfiSliceMut, FfiSliceMutType, FfiSliceType, SafeExternFn,
        SliceFromRawParts, SliceRefFromRawParts, StackAlloc, StackBytes,
    };
    pub use crate::func::{
        call0, call1, call2, call3, Compiled, CompiledFn, Compiler, Ctx, FunRef0, FunRef1, FunRef2,
        FunRef3, FunType0, FunType1, FunType2, FunType3,
    };
    pub use crate::iter::{
        box_dyn_exact_iter, box_dyn_iter, emplace_iter, range, range_step, DynExactIter, DynIter,
        ExactSizeOpaqueIter, ExactSizeOpaqueIterFns, ExactSizeOpaqueIterKind, Filter, FilterMap,
        IndexedSource, IndexedStagedIterator, IntoStagedIterator, Map, MinMax, OpaqueHandle,
        OpaqueIter, OpaqueIterFns, OpaqueIterKind, OpaqueIterSlot, RangeIter, RangeStep,
        RegisterScalar, ReusedOpaqueIter, ReusedOpaqueIterFns, ReusedOpaqueIterKind, Scan,
        SkipWhile, SliceIter, StagedIterator, TakeWhile, Zip, ZipGetAt, ZipItem, ZipItemAccess,
        ZipItemType, OPAQUE_ITER_INLINE_CAP,
    };
    pub use crate::num::{
        add, bitand, bitcast, bitor, bitxor, div, eq, gt, int_cast, int_to_float, lt, max, min,
        mul, rem, select, shl, shr, sub, BitAnd, BitOr, BitXor, Bitcast, FloatNum, IntCast, IntNum,
        IntToFloat, Num, Shl, Shr,
    };
    pub use crate::opaque::{
        const_opaque, const_opaque_mut, opaque_ref, opaque_ref_mut, Opaque, OpaqueRef, OpaqueRefMut,
    };
    pub use crate::option::{
        c_none, c_some, is_mut_ref_none, is_mut_ref_some, is_none, is_ref_none, is_ref_some,
        is_some, match_opt, match_opt_mut_ref, match_opt_ref, opt_mut_ref_none, opt_mut_ref_some,
        opt_ref_none, opt_ref_some, unwrap_or, COption, COptionType, OptMutRefType, OptRefType,
    };
    pub use crate::pool::{pool_append, BytesPool, PoolAppendExtern};
    pub use crate::r#struct::{
        field_addr, field_path, load_field, CopyFieldAccess, Field, FieldAddr, FieldPath,
        FieldRefOf, LoadField, OwnedFieldAccess, PointerLike, RefFieldAccess,
    };
    pub use crate::refer::{
        array_index, const_mut_ptr, const_ptr, load, load_mut, load_ref, load_ref_mut,
        ptr_as_const, ptr_cast, ptr_cast_mut, ptr_is_null, ptr_offset, ptr_offset_mut,
        ref_as_const, store, store_ref, ArrayIndex, ConstPtr, LoadMutRef, LoadRef, PtrCast,
        PtrIsNull, PtrOffset, PtrOffsetMut, RustPtr, RustRef, SMutPtr, SPtr, SRef, SRefMut, Store,
    };
    pub use crate::slice::{
        AsMutSlice, AsSlice, MutSliceRepr, ReprSliceMutOps, ReprSliceOps, Slice, SliceLen,
        SliceMutOps, SliceRefOps, SliceRepr, SliceType,
    };
    pub use crate::staged::{
        assign, unit, Assign, BoxableStaged, CompilationContext, Const, IntoStaged, LetVar, Staged,
        Var,
    };
    pub use crate::staged_opt::{s_none, s_some, SNone, SSome, StagedOpt, ThenSome, When};
    pub use crate::types::{ConstantType, CopyType, StagedType};
    // Re-export derive macro
    #[cfg(feature = "derive")]
    pub use rust_lms_derive::{extern_fn, StagedType};
}
