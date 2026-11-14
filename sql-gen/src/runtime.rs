//! Runtime primitives — Rust `extern "C"` functions the generated kernel calls for
//! what lives on the host: output builders, the GROUP BY and join state, the input
//! streams. The paper's `printf`/`Scanner` equivalents. String operations are staged
//! (`rust-lms`'s `ByteView`); a string reaches an extern as its bytes, `FatSlice<u8>`.

use arrow::array::StringViewBuilder;
use rust_lms::prelude::*;
use rust_lms_std::SvecGrowExtern;

use crate::group::{
    GroupInternExtern, GroupLenExtern, GroupRecordsBaseExtern, GroupStringsExtern,
    GroupUpsertExtern, GroupUpsertNullExtern, GroupUpsertPackedExtern,
};
use crate::join::{
    JoinInsertExtern, JoinLeftBatchExtern, JoinProbeBaseExtern, JoinProbeCountExtern,
    JoinRelCountExtern,
};
use crate::scan::ScanNextExtern;

/// Append a (non-null) string's bytes to an output `StringViewBuilder`. They are valid
/// UTF-8: a `Utf8View` row, a literal or a group key that was one.
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
/// through codegen.
#[derive(Clone, Copy)]
pub struct Runtime {
    pub strview_append_bytes: ExternRef<StrviewAppendBytesExtern>,
    pub strview_append_null: ExternRef<StrviewAppendNullExtern>,
    pub group_upsert: ExternRef<GroupUpsertExtern>,
    pub group_upsert_null: ExternRef<GroupUpsertNullExtern>,
    pub group_upsert_packed: ExternRef<GroupUpsertPackedExtern>,
    pub group_intern: ExternRef<GroupInternExtern>,
    pub group_strings: ExternRef<GroupStringsExtern>,
    pub group_records_base: ExternRef<GroupRecordsBaseExtern>,
    pub group_len: ExternRef<GroupLenExtern>,
    /// Grow an output column's `SVec` (cold path of `SVec::push`). See `output.rs`.
    pub svec_grow: ExternRef<SvecGrowExtern>,
    /// Pull the next input batch of a table (null = exhausted). See `scan.rs`.
    pub scan_next: ExternRef<ScanNextExtern>,
    /// Hash-join build index: relation batch count / per-row locator insert.
    pub join_rel_count: ExternRef<JoinRelCountExtern>,
    pub join_insert: ExternRef<JoinInsertExtern>,
    /// Hash-join probe: match count / locator-run base / build-batch descriptors.
    pub join_probe_count: ExternRef<JoinProbeCountExtern>,
    pub join_probe_base: ExternRef<JoinProbeBaseExtern>,
    pub join_left_batch: ExternRef<JoinLeftBatchExtern>,
}

impl Runtime {
    /// Register every runtime extern with `compiler` and collect their handles.
    pub fn register(compiler: &mut Compiler) -> Self {
        Self {
            strview_append_bytes: compiler.extern_fn::<StrviewAppendBytesExtern>(),
            strview_append_null: compiler.extern_fn::<StrviewAppendNullExtern>(),
            group_upsert: compiler.extern_fn::<GroupUpsertExtern>(),
            group_upsert_null: compiler.extern_fn::<GroupUpsertNullExtern>(),
            group_upsert_packed: compiler.extern_fn::<GroupUpsertPackedExtern>(),
            group_intern: compiler.extern_fn::<GroupInternExtern>(),
            group_strings: compiler.extern_fn::<GroupStringsExtern>(),
            group_records_base: compiler.extern_fn::<GroupRecordsBaseExtern>(),
            group_len: compiler.extern_fn::<GroupLenExtern>(),
            svec_grow: compiler.extern_fn::<SvecGrowExtern>(),
            scan_next: compiler.extern_fn::<ScanNextExtern>(),
            join_rel_count: compiler.extern_fn::<JoinRelCountExtern>(),
            join_insert: compiler.extern_fn::<JoinInsertExtern>(),
            join_probe_count: compiler.extern_fn::<JoinProbeCountExtern>(),
            join_probe_base: compiler.extern_fn::<JoinProbeBaseExtern>(),
            join_left_batch: compiler.extern_fn::<JoinLeftBatchExtern>(),
        }
    }
}
