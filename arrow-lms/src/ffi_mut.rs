//! Write side: standalone staged validity-bitmap ops.
//!
//! The output-materialization path moved to `sql-gen`'s `output::OutCols`
//! (`SVec`-backed growable columns; see `docs/table_scan.md` §4), so there is no
//! host `PreparedOutput` here anymore. What remains is the staged twin of the
//! read-side `is_valid`: [`ValidityView`]'s `set_null` / `set_valid`, usable on
//! any mutable [`FfiValidity`] bitmap independently of a particular array.

use rust_lms::prelude::*;

use crate::array::{bit_location, ValidityView};
use crate::ffi::{FfiValidity, FfiValidityType};

/// Mutable validity-bitmap ops — the write twin of the read-side `is_valid`.
/// They live on [`ValidityView`] (the staged counterpart of `FfiValidity`), so a
/// bitmap can be updated independently of any primitive array, and share
/// `bit_location` with `is_valid` so the bit arithmetic exists in one place.
impl<V> ValidityView<V>
where
    V: Staged<Out = SRefMut<'static, FfiValidity>> + Clone + 'static,
{
    /// Mark row `i` null (clear its validity bit): `byte &= ~mask`.
    pub fn set_null(&self, ctx: &mut Ctx, i: Var<u64>) {
        // SAFETY: the mutable validity view must own exclusive access to its
        // live bitmap buffer for the duration of generated execution.
        let bytes = unsafe {
            field_addr(self.validity.clone(), FfiValidityType::bytes()).as_mut_slice::<u8>()
        };
        let (byte_index, mask) = bit_location(self.validity.clone(), i);
        let byte_index = ctx.bind(byte_index);
        let old = int_cast::<u64, u8, _>(bytes.clone().get_unchecked(byte_index));
        let not_mask = bitxor::<u64, _, _>(mask, Const::<u64>::new(u64::MAX));
        let cleared = int_cast::<u8, u64, _>(bitand::<u64, _, _>(old, not_mask));
        ctx.emit(bytes.set_unchecked(byte_index, cleared));
    }

    /// Mark row `i` valid (set its validity bit): `byte |= mask`.
    pub fn set_valid(&self, ctx: &mut Ctx, i: Var<u64>) {
        // SAFETY: the mutable validity view must own exclusive access to its
        // live bitmap buffer for the duration of generated execution.
        let bytes = unsafe {
            field_addr(self.validity.clone(), FfiValidityType::bytes()).as_mut_slice::<u8>()
        };
        let (byte_index, mask) = bit_location(self.validity.clone(), i);
        let byte_index = ctx.bind(byte_index);
        let old = int_cast::<u64, u8, _>(bytes.clone().get_unchecked(byte_index));
        let set = int_cast::<u8, u64, _>(bitor::<u64, _, _>(old, mask));
        ctx.emit(bytes.set_unchecked(byte_index, set));
    }
}
