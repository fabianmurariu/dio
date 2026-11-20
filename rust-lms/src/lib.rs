//! # rust-lms: type-safe staged computation in Rust
//!
//! Multi-stage programming in the spirit of Scala LMS (Lightweight Modular
//! Staging). You build a *description* of a computation out of ordinary,
//! strongly-typed Rust values — [`Staged`](staged::Staged) values like
//! [`Var<T>`](staged::Var), `add(x, y)`, and staged iterators —
//! [`Compiler::compile`](compiler::Compiler::compile) lowers it to Cranelift IR and
//! JIT-compiles it to a native function you can call.
//!
//! **A value's Rust type encodes its staged type**, so the Rust compiler *is* the
//! staged type checker: an `i64`/`bool` mix-up, a write through a `&[T]`, or a
//! returned dangling field reference simply doesn't type-check — there is no
//! separate runtime type system to get wrong.
//!
//! ```
//! use rust_lms::prelude::*;
//!
//! let mut compiler = Compiler::new();
//! // square(x) = x * x
//! let square = compiler.fun1("square", |_ctx, x: Var<i64>| mul(x, x));
//! let compiled = compiler.compile(square).expect("compile");
//! assert_eq!(compiled.as_fn().call(7), 49); // owner-checked callable
//! ```
//!
//! ## Two phases
//!
//! - **Stage 0 ("now"):** plain Rust that *builds* the computation. Nothing runs
//!   on data — you are assembling a typed `Staged` tree.
//! - **Stage 1 ("later"):** `compile` JIT-compiles that tree into a native
//!   function; call it as many times as you like.
//!
//! ## Design in one breath
//!
//! - **Values vs operations.** [`Var<T>`](staged::Var) / [`Const<T>`](staged::Const)
//!   are pure values; operations (`Add<L,R>`, `Lt<L,R>`, …) are *separate* structs
//!   that also implement [`Staged`](staged::Staged). Their trait bounds are the
//!   type system — `Add` requires both sides share a `Num` type; a comparison's
//!   `Out` is `bool`, not the input type.
//! - **[`StagedType`](types::StagedType)** is the contract a type satisfies to be
//!   staged: its Cranelift representation plus ABI (how it passes as an argument /
//!   return value / struct field).
//!
//! ## Where to look
//!
//! - [`prelude`] re-exports the everyday surface — numbers & operators, control
//!   flow, references/pointers/slices, structs & tuples, iterators, optionals,
//!   FFI. Glob-import it.
//! - `docs/deep_dive.md` is the full architecture walkthrough: the mental model,
//!   every subsystem, the ABI, and the invariants to respect.

// Let `#[derive(StagedType)]`'s absolute `::rust_lms::…` paths resolve when the
// derive is used *inside* this crate (e.g. iter::zip::ZipItem), not just downstream.
extern crate self as rust_lms;

pub mod byte_view;
pub mod compiler;

pub mod control;
/// Cranelift backend: the default code generator (`impl Backend`), behind `func::compile`.
pub(crate) mod cranelift;
pub mod ctx;
pub mod ffi;
pub mod func;
pub(crate) mod func_def;
pub(crate) mod func_impl;
pub mod iter;
pub mod label;
/// LLVM/MLIR backend (docs/llvm.md, Phase 1+). Compiled only with `--features llvm`.
#[cfg(feature = "llvm")]
pub mod llvm;
pub mod num;
pub mod opaque;
pub mod option;
pub mod pool;
pub mod refer;
pub mod slice;
pub mod staged;
pub mod staged_opt;
pub mod r#struct;
pub mod types;

/// Internal utilities used by proc macros.
/// Do not use directly.
#[doc(hidden)]
pub mod _internal {
    pub use crate::r#struct::{DisjointField, Field};
}

/// Commonly used types and traits
pub mod prelude {
    pub use crate::{
        byte_view::{
            Bin, Binary, ByteBuffer, ByteKind, ByteView, MAX_INLINE_LEN, MakeByteView, RawByteView,
            Str, Utf8, Utf8LenExtern, make_view, relocate_view, utf8_len, view_len,
        },
        control::{IfThen, IfThenElse, Not, While, if_then, if_then_else, not, while_loop},
        ffi::{
            ExternArgs, ExternFn, ExternRef, FatSlice, FatSliceMut, FfiSlice, IntoExternArg,
            RawSlice, RawSliceMut, SafeExternFn, SliceFromRawParts, StackAlloc, StackBytes,
            call_extern0, call_extern0_unchecked, call_extern1, call_extern1_unchecked,
            call_extern2, call_extern2_unchecked, call_extern3, call_extern3_unchecked,
            call_extern4, call_extern4_unchecked, slice_from_raw_parts, slice_from_raw_parts_mut,
            stack_alloc, stack_array, stack_bytes,
        },
        func::{
            Compiled, CompiledFn, FunRef0, FunRef1, FunRef2, FunRef3, FunType0, FunType1, FunType2,
            FunType3, call0, call1, call2, call3,
        },
        iter::{
            Chain, Close, Cursor, DynExactIter, DynIter, Enumerate, ExactOpaqueIterOwner,
            ExactSizeOpaqueIter, ExactSizeOpaqueIterFns, ExactSizeOpaqueIterKind, Filter,
            FilterMap, FlatMap, FromFn, IndexedSource, IndexedStagedIterator, IntoStagedIterator,
            KMergeBy, Map, MergeBy, MinMax, OPAQUE_ITER_INLINE_CAP, OpaqueHandle, OpaqueIter,
            OpaqueIterFns, OpaqueIterItem, OpaqueIterKind, OpaqueIterOwner, OpaqueIterSlot, Pair,
            RangeIter, RangeStep, ReusedOpaqueIter, ReusedOpaqueIterFns, ReusedOpaqueIterKind, Rev,
            Scan, SetBits, Skip, SkipWhile, SliceIter, StagedIterator, Take, TakeWhile, Zip,
            ZipGetAt, ZipItem, ZipItemAccess, ZipItemType, ZipLen, box_dyn_exact_iter,
            box_dyn_iter, emplace_iter, empty, from_fn, kmerge_by, range, range_step, set_bits,
        },
        label::{Again, Label},
        num::{
            BitAnd, BitOp, BitOpKind, BitOr, BitXor, Bitcast, Bits, FloatNum, FloatPromote, Ge,
            Int, IntCast, IntNum, IntToFloat, Le, Ne, Num, Ordered, Shl, Shr, add, bitand, bitcast,
            bitor, bitxor, count_ones, div, eq, float_promote, ge, gt, i128_hi, i128_lo, int_cast,
            int_to_float, le, leading_zeros, lt, max, min, mul, mul_high, ne, pack_i128, rem,
            select, shl, shr, sub, swap_bytes, trailing_zeros,
        },
        opaque::Opaque,
        option::{
            COption, COptionType, OptMutRefType, OptRefType, c_none, c_some, is_mut_ref_none,
            is_mut_ref_some, is_none, is_ref_none, is_ref_some, is_some, match_opt,
            match_opt_mut_ref, match_opt_ref, opt_mut_ref_none, opt_mut_ref_some, opt_ref_none,
            opt_ref_some, unwrap_or,
        },
        pool::{
            ByteLiterals, BytesPool, BytesPoolNewBlockExtern, PoolRef, StagedPool,
            bytes_pool_new_block,
        },
        refer::{
            ArrayIndex, ConstPtr, IntoMutRef, LoadMutPtr, LoadMutRef, LoadPtr, LoadRef, PtrCast,
            PtrIsNull, PtrOffset, PtrOffsetMut, ReadUnaligned, SMutPtr, SPtr, SRef, SRefMut, Store,
            StoreRef, WriteUnaligned, array_index, const_mut_ptr, const_ptr, copy_bytes, load,
            load_mut, load_ref, load_ref_mut, ptr_as_const, ptr_as_ref, ptr_cast, ptr_cast_mut,
            ptr_is_null, ptr_offset, ptr_offset_mut, read_unaligned, ref_as_const, ref_as_ptr,
            ref_mut_as_ptr, store, store_ref, write_unaligned,
        },
        slice::{
            AsRawSlice, AsSlice, BorrowedSlice, BorrowedSliceMut, MutSliceType, RawSliceOps,
            RawSliceType, ReprRawSliceOps, ReprSliceOps, Slice, SliceGetOr, SliceGetPtrUnchecked,
            SliceGetRange, SliceLen, SliceMutOps, SliceOps, SliceRepr, SliceSet,
            SliceSliceUnchecked, SliceType, TrustedSliceOps, TrustedSliceType,
        },
        staged::{
            Assign, BoxableStaged, CompilationContext, Const, IntoStaged, LifetimeErased, Staged,
            Value, ValueId, Var, VarUse, assign, unit,
        },
        staged_opt::{SNone, SSome, StagedOpt, ThenSome, When, s_none, s_some},
        r#struct::{
            CopyFieldAccess, DisjointField, Field, FieldAddr, FieldPath, FieldRefOf, LoadField,
            MutField, OwnedFieldAccess, PointerLike, RefFieldAccess, SplitFieldsMut, field_addr,
            field_mut, field_path, load_field_mut, load_field_unchecked, split_fields_mut,
        },
        types::{
            ConstantType, CopyType, DirectValue, FloatCmp, IntCmp, Leaves, RuntimeParam,
            RuntimeResult, ScalarType, Shape, StagedType,
        },
    };
    // Re-export derive macro
    pub use crate::{
        compiler::{Compiler, JitBackend},
        ctx::Ctx,
    };
    #[cfg(feature = "derive")]
    pub use rust_lms_derive::{StagedType, extern_fn};
}
