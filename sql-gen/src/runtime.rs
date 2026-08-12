//! Runtime primitives — Rust `extern "C"` functions the generated kernel calls
//! for string semantics we deliberately don't emit in codegen (UTF-8, byte
//! access). The paper's `printf`/`Scanner` equivalents.
//!
//! A string reaches the runtime as a resolved byte reference (`&[u8]` /
//! `FatSlice<u8>`) — the *generic* representation every string container produces
//! (see `codegen::resolve`). So these ops are a flat library over bytes: compare,
//! append to output. The generated code keeps the fast paths (short-literal `=` is
//! a pure staged view compare) and only falls back here for real byte work.

use arrow::array::{StringViewArray, StringViewBuilder};
use rust_lms::prelude::*;

use crate::group::{
    GroupLenExtern, GroupRecordsBaseExtern, GroupUpsertExtern, GroupUpsertNullExtern,
    GroupUpsertStrExtern,
};

/// The byte pointer of a `Utf8View` row: `arr.value(row).as_ptr()`. Valid for the
/// life of the source array (points into its views/data buffers); the length
/// comes free from the staged view (`lo & 0xFFFF_FFFF`), so it needs no extern.
/// A null row reads as an empty inline view — safe, never out of bounds.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn str_ptr(arr: &StringViewArray, row: usize) -> *const u8 {
    arr.value(row).as_ptr()
}

/// Full byte-equality of two resolved strings — the uniform fallback for every
/// source combination that isn't a pure staged view compare.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn bytes_eq(a: FatSlice<u8>, b: FatSlice<u8>) -> bool {
    unsafe { a.as_slice() == b.as_slice() }
}

/// Append a (non-null) resolved string to an output `StringViewBuilder`. The bytes
/// are always valid UTF-8 (from an arrow string or a checked literal).
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn strview_append_bytes(sink: &mut StringViewBuilder, bytes: FatSlice<u8>) {
    let s = unsafe { std::str::from_utf8_unchecked(bytes.as_slice()) };
    sink.append_value(s);
}

/// Append a null to an output `StringViewBuilder`.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn strview_append_null(sink: &mut StringViewBuilder) {
    sink.append_null();
}

/// Handles to the runtime externs, registered once per compile and threaded
/// through codegen so `resolve`/`str_eq`/`write_str_col` can call them.
#[derive(Clone, Copy)]
pub struct Runtime {
    pub str_ptr: ExternRef<StrPtrExtern>,
    pub bytes_eq: ExternRef<BytesEqExtern>,
    pub strview_append_bytes: ExternRef<StrviewAppendBytesExtern>,
    pub strview_append_null: ExternRef<StrviewAppendNullExtern>,
    pub group_upsert: ExternRef<GroupUpsertExtern>,
    pub group_upsert_str: ExternRef<GroupUpsertStrExtern>,
    pub group_upsert_null: ExternRef<GroupUpsertNullExtern>,
    pub group_records_base: ExternRef<GroupRecordsBaseExtern>,
    pub group_len: ExternRef<GroupLenExtern>,
}

impl Runtime {
    /// Register every runtime extern with `compiler` and collect their handles.
    pub fn register(compiler: &mut Compiler) -> Self {
        Self {
            str_ptr: compiler.extern_fn::<StrPtrExtern>(),
            bytes_eq: compiler.extern_fn::<BytesEqExtern>(),
            strview_append_bytes: compiler.extern_fn::<StrviewAppendBytesExtern>(),
            strview_append_null: compiler.extern_fn::<StrviewAppendNullExtern>(),
            group_upsert: compiler.extern_fn::<GroupUpsertExtern>(),
            group_upsert_str: compiler.extern_fn::<GroupUpsertStrExtern>(),
            group_upsert_null: compiler.extern_fn::<GroupUpsertNullExtern>(),
            group_records_base: compiler.extern_fn::<GroupRecordsBaseExtern>(),
            group_len: compiler.extern_fn::<GroupLenExtern>(),
        }
    }
}
