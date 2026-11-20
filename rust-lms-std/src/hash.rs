//! Staged hashing for [`ChainedTable`](crate::ChainedTable) keys: a **folded multiply**,
//! the mixer of wyhash and foldhash. Multiplying two 64-bit words gives a 128-bit product
//! whose bits each depend on many input bits; xor-ing its two halves folds that back into
//! 64. It needs only `mul` and [`mul_high`], so it compiles to the same few instructions on
//! every backend (no CRC32, which Cranelift lacks).
//!
//! A key of several values hashes them one 64-bit word at a time:
//!
//! ```ignore
//! let h = hash_word(ctx, SEED, segment_and_pos);
//! let h = hash_word(ctx, h, t);
//! ```

use rust_lms::prelude::*;

/// The hash a key starts from (the first digits of π): any fixed value, so hashes do not
/// change between runs.
pub const SEED: u64 = 0x243f_6a88_85a3_08d3;

/// An odd constant with its bits spread (the golden ratio's): multiplying by it spreads
/// every bit of the other word across the product.
const SPREAD: u64 = 0x9e37_79b9_7f4a_7c15;

/// The 128-bit product of `a` and `b`, its two halves xor-ed.
pub fn fold_mul(ctx: &Ctx, a: Var<u64>, b: impl IntoStaged<u64, Staged: 'static>) -> Var<u64> {
    let b = ctx.bind(b.into_staged());
    ctx.bind(bitxor(mul(a, b), mul_high(a, b)))
}

/// `hash` with `word` mixed in.
pub fn hash_word(
    ctx: &Ctx,
    hash: impl IntoStaged<u64, Staged: 'static>,
    word: impl IntoStaged<u64, Staged: 'static>,
) -> Var<u64> {
    let mixed = ctx.bind(bitxor(hash, word));
    fold_mul(ctx, mixed, SPREAD)
}
