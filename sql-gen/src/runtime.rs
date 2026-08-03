//! Runtime primitives — Rust `extern "C"` functions the generated kernel calls
//! for string semantics we deliberately don't emit in codegen (UTF-8, byte
//! access). The paper's `printf`/`Scanner` equivalents.
//!
//! The generated code stays fast for the common paths (short-literal equality is
//! a pure view compare); these are the fallbacks for the rest.

use arrow::array::{Array, StringViewArray};
use rust_lms::prelude::*;

/// Full byte-equality of a `Utf8View` row against a literal, used as the
/// fallback when the view fast-path (length + prefix) is inconclusive (long
/// strings). `arr` is the originating array (via `FfiArray.array`); `lit` is the
/// baked literal bytes.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn string_view_eq(arr: &StringViewArray, row: usize, lit: FatSlice<u8>) -> bool {
    arr.is_valid(row) && arr.value(row).as_bytes() == unsafe { lit.as_slice() }
}

/// Full byte-equality of two `Utf8View` rows (`WHERE a.name = b.name` style),
/// the fallback when neither side is an inline literal so the staged view compare
/// can't apply. Null rows compare unequal; the staged side also masks the result
/// by both columns' validity.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn string_view_eq_cols(
    a: &StringViewArray,
    row_a: usize,
    b: &StringViewArray,
    row_b: usize,
) -> bool {
    a.is_valid(row_a) && b.is_valid(row_b) && a.value(row_a) == b.value(row_b)
}

/// Handles to the runtime externs, registered once per compile and threaded
/// through codegen so `str_eq` etc. can call them.
#[derive(Clone, Copy)]
pub struct Runtime {
    pub string_view_eq: ExternRef<StringViewEqExtern>,
    pub string_view_eq_cols: ExternRef<StringViewEqColsExtern>,
}

impl Runtime {
    /// Register every runtime extern with `compiler` and collect their handles.
    pub fn register(compiler: &mut Compiler) -> Self {
        Self {
            string_view_eq: compiler.extern_fn::<StringViewEqExtern>(),
            string_view_eq_cols: compiler.extern_fn::<StringViewEqColsExtern>(),
        }
    }
}
