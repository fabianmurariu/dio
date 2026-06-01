//! Staged views over an Arrow column and the iterator-source glue.
//!
//! [`StagedArrowArrayI32`] binds an `&FfiArray` parameter's buffers into local
//! variables and hands out [`ValuesI32`], a `rust-lms` iterator source over the
//! values buffer. Because it implements the standard iterator traits, every
//! combinator and terminal (`map`/`filter`/`sum`/`count_if`/`zip`/…) works
//! against an Arrow column with no extra code here.

use cranelift_codegen::ir::{types, InstBuilder, MemFlags, Value};
use rust_lms::prelude::*;

use crate::ffi::{FfiArray, FfiArrayType};

/// Size of an `i32` value, in bytes (the stride between elements).
const I32_STRIDE: i64 = 4;

// =============================================================================
// StagedNullBuffer
// =============================================================================

/// Staged view of an Arrow validity bitmap.
///
/// All three fields are bound into locals by [`StagedArrowArrayI32::load`]. A
/// `validity` address of `0` means the array has no nulls (every element valid).
/// The per-element validity test is not wired up yet — this carries the metadata
/// end-to-end so null-aware operations can be layered on next.
#[derive(Clone, Copy)]
pub struct StagedNullBuffer {
    /// Address of the bitmap's first byte, or `0` when the array has no nulls.
    pub validity: Var<U64Type>,
    /// Bit offset into the bitmap (Arrow slices validity by bit, not byte).
    pub bit_offset: Var<U64Type>,
    /// Number of null entries.
    pub null_count: Var<U64Type>,
}

// =============================================================================
// StagedArrowArrayI32
// =============================================================================

/// Staged view of an Arrow `Int32Array`: a values buffer plus a null buffer.
///
/// Construct with [`StagedArrowArrayI32::load`] from a `Var<SRef<FfiArray>>`
/// function parameter, then iterate via [`StagedArrowArrayI32::values`].
#[derive(Clone, Copy)]
pub struct StagedArrowArrayI32 {
    values: Var<U64Type>,
    len: Var<U64Type>,
    nulls: StagedNullBuffer,
}

impl StagedArrowArrayI32 {
    /// Bind an `&FfiArray` parameter's fields into local variables.
    pub fn load<'a>(ctx: &mut Ctx, arr: Var<SRef<'a, FfiArray>>) -> Self
    where
        'a: 'static,
    {
        let values = ctx.var(arr.get(FfiArrayType::values));
        let len = ctx.var(arr.get(FfiArrayType::len));
        let validity = ctx.var(arr.get(FfiArrayType::validity));
        let bit_offset = ctx.var(arr.get(FfiArrayType::validity_bit_offset));
        let null_count = ctx.var(arr.get(FfiArrayType::null_count));

        StagedArrowArrayI32 {
            values,
            len,
            nulls: StagedNullBuffer {
                validity,
                bit_offset,
                null_count,
            },
        }
    }

    /// Logical element count as a staged value.
    pub fn len(&self) -> Var<U64Type> {
        self.len
    }

    /// The null buffer (validity bitmap address + null count).
    pub fn null_buffer(&self) -> StagedNullBuffer {
        self.nulls
    }

    /// An iterator source over the values buffer (does not yet skip nulls).
    pub fn values(&self) -> ValuesI32 {
        ValuesI32 {
            base: self.values,
            len: self.len,
        }
    }
}

// =============================================================================
// ArrowGetI32: load element `index` from a raw values base address
// =============================================================================

/// Load the `i32` at `base + index * 4`. The single piece of custom codegen in
/// this crate; everything else is built from `rust-lms` primitives.
pub struct ArrowGetI32 {
    base: Var<U64Type>,
    index: Var<U64Type>,
}

impl Staged for ArrowGetI32 {
    type Out = I32Type;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        let base = self.base.codegen(ctx);
        let index = self.index.codegen(ctx);
        let stride = ctx.builder.ins().iconst(types::I64, I32_STRIDE);
        let byte_offset = ctx.builder.ins().imul(index, stride);
        let addr = ctx.builder.ins().iadd(base, byte_offset);
        ctx.builder
            .ins()
            .load(types::I32, MemFlags::trusted(), addr, 0)
    }
}

// =============================================================================
// ValuesI32: the iterator source
// =============================================================================

/// Iterator source over the `i32` values of an Arrow column.
///
/// Mirrors `rust-lms`'s `SliceIter`, but loads from a raw base address rather
/// than a fat-pointer slice. Implements [`StagedIterator`],
/// [`IndexedStagedIterator`], and [`IndexedSource`], so all combinators,
/// terminals, and `zip` apply.
#[derive(Clone, Copy)]
pub struct ValuesI32 {
    base: Var<U64Type>,
    len: Var<U64Type>,
}

impl StagedIterator for ValuesI32 {
    type Item = I32Type;

    fn for_each<F>(self, ctx: &mut Ctx, consumer: F)
    where
        F: FnOnce(&mut Ctx, Var<I32Type>) + 'static,
    {
        let i = ctx.var(0u64);
        let me = self;
        ctx.while_loop(lt(i, me.len), move |ctx| {
            // Bind the element inside the loop so the frontend resolves it to the
            // load with no extra copy (same shape SliceIter emits).
            let elem = ctx.bind(ArrowGetI32 {
                base: me.base,
                index: i,
            });
            consumer(ctx, elem);
            ctx.store(i, add(i, 1u64));
        });
    }
}

impl IndexedStagedIterator for ValuesI32 {
    type LenExpr = Var<U64Type>;

    fn len(&self) -> Self::LenExpr {
        self.len
    }
}

impl IndexedSource for ValuesI32 {
    type Item = I32Type;
    type LenExpr = Var<U64Type>;
    type GetExpr = ArrowGetI32;

    fn len(&self) -> Self::LenExpr {
        self.len
    }

    fn get_at(self, index: Var<U64Type>) -> Self::GetExpr {
        ArrowGetI32 {
            base: self.base,
            index,
        }
    }
}
