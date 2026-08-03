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

/// Handles to the runtime externs, registered once per compile and threaded
/// through codegen so `str_eq` etc. can call them.
#[derive(Clone, Copy)]
pub struct Runtime {
    pub string_view_eq: ExternRef<StringViewEqExtern>,
}

impl Runtime {
    /// Register every runtime extern with `compiler` and collect their handles.
    pub fn register(compiler: &mut Compiler) -> Self {
        Self {
            string_view_eq: compiler.extern_fn::<StringViewEqExtern>(),
        }
    }
}
