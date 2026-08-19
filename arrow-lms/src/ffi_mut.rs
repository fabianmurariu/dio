//! Write side: standalone staged validity-bitmap ops.
//!
//! The output-materialization path moved to `sql-gen`'s `output::OutCols`
//! (`SVec`-backed growable columns; see `docs/table_scan.md` §4), so there is no
//! host `PreparedOutput` here anymore. What remains is the staged twin of the
//! read-side `is_valid`: [`ValidityView`]'s `set_null` / `set_valid`, usable on
//! any mutable [`FfiValidity`] bitmap independently of a particular array.

use rust_lms::prelude::*;

use crate::array::ValidityView;
use crate::ffi::{FfiValidityMut, FfiValidityMutType};

fn bit_location_mut(
    ctx: &mut Ctx,
    validity: &mut Var<SRefMut<'static, FfiValidityMut>>,
    index: Var<u64>,
) -> (Var<u64>, Var<u64>) {
    let bit_offset = ctx.bind(load_field_mut(validity, FfiValidityMutType::bit_offset()));
    let bit = ctx.bind(add(bit_offset, index));
    let byte_index = ctx.bind(shr(bit, 3u64));
    let mask = ctx.bind(shl(1u64, bitand(bit, 7u64)));
    (byte_index, mask)
}

/// Mutable validity-bitmap ops — the write twin of the read-side `is_valid`.
/// They live on [`ValidityView`] (the staged counterpart of `FfiValidity`), so a
/// bitmap can be updated independently of any primitive array, and share
/// `bit_location` with `is_valid` so the bit arithmetic exists in one place.
impl ValidityView<Var<SRefMut<'static, FfiValidityMut>>> {
    /// Mark row `i` null (clear its validity bit): `byte &= ~mask`.
    ///
    /// # Safety
    ///
    /// At execution, `i` must be less than the prepared bitmap's bit length.
    pub unsafe fn set_null(&mut self, ctx: &mut Ctx, i: Var<u64>) {
        // SAFETY: the mutable validity view must own exclusive access to its
        // live bitmap buffer for the duration of generated execution.
        let bytes_read = unsafe {
            field_addr_mut_unchecked(&mut self.validity, FfiValidityMutType::bytes())
                .as_mut_slice::<u8>()
        };
        // SAFETY: this impl is restricted to staged mutable references to
        // `FfiValidity`.
        let (byte_index, mask) = bit_location_mut(ctx, &mut self.validity, i);
        let byte_index = ctx.bind(byte_index);
        // SAFETY: `i` is in range by this method's contract, so its computed
        // byte lies within the prepared bitmap.
        let old = ctx.bind(int_cast::<u64, u8, _>(unsafe {
            bytes_read.get_unchecked(byte_index)
        }));
        let was_valid = ctx.bind(not(eq(bitand::<u64, _, _>(old, mask), 0u64)));
        let not_mask = bitxor::<u64, _, _>(mask, Const::<u64>::new(u64::MAX));
        let cleared = int_cast::<u8, u64, _>(bitand::<u64, _, _>(old, not_mask));
        // SAFETY: the read above is complete before this fresh mutable
        // projection writes the same in-range byte.
        let bytes_write = unsafe {
            field_addr_mut_unchecked(&mut self.validity, FfiValidityMutType::bytes())
                .as_mut_slice::<u8>()
        };
        ctx.emit(unsafe { bytes_write.set_unchecked(byte_index, cleared) });
        ctx.if_then(was_valid, |ctx| {
            let count = ctx.bind(load_field_mut(
                &mut self.validity,
                FfiValidityMutType::null_count(),
            ));
            // SAFETY: this is the only access to `null_count` in this branch.
            let count_ref = unsafe {
                field_addr_mut_unchecked(&mut self.validity, FfiValidityMutType::null_count())
            };
            ctx.emit(unsafe { store_ref_unchecked(count_ref, add(count, 1u64)) });
        });
    }

    /// Mark row `i` valid (set its validity bit): `byte |= mask`.
    ///
    /// # Safety
    ///
    /// At execution, `i` must be less than the prepared bitmap's bit length.
    pub unsafe fn set_valid(&mut self, ctx: &mut Ctx, i: Var<u64>) {
        // SAFETY: the mutable validity view must own exclusive access to its
        // live bitmap buffer for the duration of generated execution.
        let bytes_read = unsafe {
            field_addr_mut_unchecked(&mut self.validity, FfiValidityMutType::bytes())
                .as_mut_slice::<u8>()
        };
        // SAFETY: this impl is restricted to staged mutable references to
        // `FfiValidity`.
        let (byte_index, mask) = bit_location_mut(ctx, &mut self.validity, i);
        let byte_index = ctx.bind(byte_index);
        // SAFETY: `i` is in range by this method's contract, so its computed
        // byte lies within the prepared bitmap.
        let old = ctx.bind(int_cast::<u64, u8, _>(unsafe {
            bytes_read.get_unchecked(byte_index)
        }));
        let was_null = ctx.bind(eq(bitand::<u64, _, _>(old, mask), 0u64));
        let set = int_cast::<u8, u64, _>(bitor::<u64, _, _>(old, mask));
        // SAFETY: the read above is complete before this fresh mutable
        // projection writes the same in-range byte.
        let bytes_write = unsafe {
            field_addr_mut_unchecked(&mut self.validity, FfiValidityMutType::bytes())
                .as_mut_slice::<u8>()
        };
        ctx.emit(unsafe { bytes_write.set_unchecked(byte_index, set) });
        ctx.if_then(was_null, |ctx| {
            let count = ctx.bind(load_field_mut(
                &mut self.validity,
                FfiValidityMutType::null_count(),
            ));
            // SAFETY: this is the only access to `null_count` in this branch.
            let count_ref = unsafe {
                field_addr_mut_unchecked(&mut self.validity, FfiValidityMutType::null_count())
            };
            ctx.emit(unsafe { store_ref_unchecked(count_ref, sub(count, 1u64)) });
        });
    }
}
