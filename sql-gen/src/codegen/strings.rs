//! String operations over `Utf8View`: resolving to bytes, view-based equality, and
//! staged string literals.

use rust_lms::prelude::*;

use crate::runtime::StrPtrExtern;
use crate::value::{ColVal, Nullness, StrVal};

use super::CodegenCtx;
use super::numeric::tag;

/// Resolve a string to its bytes `(ptr, len)` — the generic representation every
/// container produces. A `Column` reads its byte pointer via the `str_ptr` extern
/// (its length comes free from the view, `lo & 0xFFFF_FFFF`); `Bytes` (interned
/// literal or produced string) is already resolved. Takes the extern handle (not
/// `&CodegenCtx`) so it can be called from `'static` branch closures.
pub(crate) fn resolve(
    ctx: &mut Ctx,
    sv: StrVal,
    str_ptr: ExternRef<StrPtrExtern>,
) -> (Var<SPtr<u8>>, Var<u64>) {
    match sv {
        StrVal::Column { lo, array, row, .. } => {
            let ptr = ctx.bind(call_extern2(str_ptr, array, row));
            let len = ctx.bind(bitand::<u64, _, _>(lo, 0xFFFF_FFFFu64));
            (ptr, len)
        }
        StrVal::Bytes { ptr, len, .. } => (ptr, len),
    }
}
/// The 16-byte view halves of a string, as staged `u64`s — for the equality fast
/// path. A `Column` has them as `Var`s; a literal's are host constants baked in.
/// A produced `Bytes` (no precomputed view) has none → resolve to bytes instead.
fn view_of(ctx: &mut Ctx, sv: StrVal) -> Option<(Var<u64>, Var<u64>)> {
    match sv {
        StrVal::Column { lo, hi, .. } => Some((lo, hi)),
        StrVal::Bytes {
            view: Some((lo, hi)),
            ..
        } => Some((ctx.var(lo), ctx.var(hi))),
        StrVal::Bytes { view: None, .. } => None,
    }
}
/// String equality over `Utf8View`, using every bit of the 16-byte view before
/// touching bytes (the view layout: `lo` = `[len:u32][first 4 content bytes]`,
/// `hi` = the inline tail for `len ≤ 12`, else buffer index/offset):
///
/// 1. `lo` differs  → **not equal** (differing length or first-4 bytes) — no resolve.
/// 2. `len ≤ 12`    → `hi` settles it (both fully inline) — no resolve.
/// 3. `len > 12`    → prefixes match but the rest is array-relative → resolve both
///    sides and `bytes_eq`.
///
/// This is correct across containers: a column row and a literal share the same
/// `lo`/`hi` encoding, and step 3 never trusts the array-relative `hi`. Only when
/// an operand has no view (a produced string) do we resolve outright.
pub(crate) fn str_eq(ctx: &mut Ctx, l: StrVal, r: StrVal, cx: &CodegenCtx) -> Var<bool> {
    let str_ptr = cx.rt.str_ptr;
    let bytes_eq = cx.rt.bytes_eq;
    match (view_of(ctx, l), view_of(ctx, r)) {
        (Some((a_lo, a_hi)), Some((b_lo, b_hi))) => {
            let result = ctx.var(false);
            let lo_eq = ctx.bind(eq(a_lo, b_lo));
            // Only when `lo` matches (same length + first 4 content bytes) is there
            // anything more to check.
            ctx.if_then(lo_eq, move |ctx| {
                let len = ctx.bind(bitand::<u64, _, _>(a_lo, 0xFFFF_FFFFu64));
                let long = ctx.bind(gt(len, 12u64));
                // len <= 12: the views are the whole strings — `hi` is definitive.
                let hi_eq = ctx.bind(eq(a_hi, b_hi));
                ctx.if_then(not(long), move |ctx| ctx.store(result, hi_eq));
                // len > 12: prefixes match; compare the full bytes.
                ctx.if_then(long, move |ctx| {
                    let (ap, al) = resolve(ctx, l, str_ptr);
                    let (bp, bl) = resolve(ctx, r, str_ptr);
                    let eq = ctx.bind(call_extern2(
                        bytes_eq,
                        slice_from_raw_parts::<u8, _, _>(ap, al),
                        slice_from_raw_parts::<u8, _, _>(bp, bl),
                    ));
                    ctx.store(result, eq);
                });
            });
            result
        }
        // At least one operand is a produced string with no view: resolve both.
        _ => {
            let (ap, al) = resolve(ctx, l, str_ptr);
            let (bp, bl) = resolve(ctx, r, str_ptr);
            ctx.bind(call_extern2(
                bytes_eq,
                slice_from_raw_parts::<u8, _, _>(ap, al),
                slice_from_raw_parts::<u8, _, _>(bp, bl),
            ))
        }
    }
}

pub(crate) fn is_str(cv: ColVal) -> bool {
    matches!(cv, ColVal::Str(..))
}
/// The `Utf8View` view halves `(lo, hi)` for a literal of *any* length, matching
/// arrow's on-wire encoding so they compare bit-identically to a column row's
/// view. `lo` = `[len:u32][first ≤4 content bytes, zero-padded]` — valid for all
/// lengths (it's the length + prefix). `hi` = the inline tail (bytes 4..12) for
/// `len ≤ 12`; for longer strings `hi` is `0` (a column's `hi` there is a
/// buffer/offset, which `str_eq` never compares — it resolves instead).
fn literal_view_halves(s: &str) -> (u64, u64) {
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut lo = [0u8; 8];
    lo[0..4].copy_from_slice(&(len as u32).to_le_bytes());
    let head = &bytes[..len.min(4)];
    lo[4..4 + head.len()].copy_from_slice(head);
    let lo = u64::from_le_bytes(lo);
    let hi = if len <= 12 {
        let mut hi = [0u8; 8];
        let tail = &bytes[len.min(4)..len]; // bytes 4..len (or empty)
        hi[..tail.len()].copy_from_slice(tail);
        u64::from_le_bytes(hi)
    } else {
        0
    };
    (lo, hi)
}
/// A string literal as resolved [`StrVal::Bytes`]: `ptr` is the literal's stable
/// address in the `BytesPool` (interned once at build time — see [`CodegenCtx::lits`]),
/// and `view` carries the view halves (for *any* length) so `str_eq`'s length +
/// prefix fast-reject and inline compare apply.
pub(crate) fn gen_str_literal(ctx: &mut Ctx, s: &str, cx: &CodegenCtx) -> ColVal {
    let ptr = *cx
        .lits
        .get(s)
        .unwrap_or_else(|| panic!("literal not interned: {s:?}"));
    ColVal::Str(
        StrVal::Bytes {
            ptr: ctx.bind(const_ptr::<u8>(ptr)),
            len: ctx.var(s.len() as u64),
            view: Some(literal_view_halves(s)),
        },
        Nullness::NonNull,
    )
}
pub(crate) fn as_str(cv: ColVal) -> StrVal {
    match cv {
        ColVal::Str(sv, _) => sv,
        other => panic!("expected string group key, got {}", tag(other)),
    }
}
