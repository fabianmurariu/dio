//! Bitmaps, and Arrow validity built on one.
//!
//! An [`FfiBitmap<W>`] is a run of bits in words of `W`, least significant bit first:
//! bit `i` of the run is bit `(bit_offset + i) % W::BITS` of word
//! `(bit_offset + i) / W::BITS`. A word is one of `u8`, `u16`, `u32`, `u64`, `u128`
//! ([`BitmapWord`]), so its width is a constant. Arrow's bitmaps are `FfiBitmap<u8>`
//! (any byte length, byte aligned, sliced by moving the bit offset); wider words suit
//! bitmaps built in words, such as db4's.
//!
//! [`FfiValidity`] is an Arrow array's validity: a bitmap of valid rows and the null
//! count. With no nulls the bitmap is empty and never read.

use std::fmt;

use arrow::buffer::NullBuffer;
use rust_lms::prelude::*;

mod word {
    pub trait Sealed {}
}

/// A bitmap word: `u8`, `u16`, `u32`, `u64` or `u128`.
pub trait BitmapWord: Int + StagedType<RuntimeValue = Self> + word::Sealed {
    /// `log2(Self::BITS)`: the shift from a bit to its word.
    const SHIFT: u64;
}

macro_rules! bitmap_word {
    ($($ty:ty),*) => {$(
        impl word::Sealed for $ty {}
        impl BitmapWord for $ty {
            const SHIFT: u64 = <$ty>::BITS.trailing_zeros() as u64;
        }
    )*};
}

bitmap_word!(u8, u16, u32, u64, u128);

/// Bits `bit_offset..bit_offset + bit_len` of `words`, least significant bit first.
#[repr(C)]
#[derive(StagedType)]
pub struct FfiBitmap<W: BitmapWord> {
    words: FfiSlice<W>,
    bit_offset: u64,
    bit_len: u64,
}

impl<W: BitmapWord> Clone for FfiBitmap<W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<W: BitmapWord> Copy for FfiBitmap<W> {}

impl<W: BitmapWord> fmt::Debug for FfiBitmap<W> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FfiBitmap")
            .field("words", &self.words)
            .field("bit_offset", &self.bit_offset)
            .field("bit_len", &self.bit_len)
            .finish()
    }
}

impl<W: BitmapWord> FfiBitmap<W> {
    /// Bits `bit_offset..bit_offset + bit_len` of `words`, which must hold them.
    pub fn new(words: &[W], bit_offset: usize, bit_len: usize) -> Self {
        let bits = words.len() << W::SHIFT;
        assert!(
            bit_offset + bit_len <= bits,
            "bits {bit_offset}..{} are past the {bits} bits given",
            bit_offset + bit_len,
        );
        Self {
            words: FfiSlice::new(words),
            bit_offset: bit_offset as u64,
            bit_len: bit_len as u64,
        }
    }

    /// No bits.
    pub const fn empty() -> Self {
        Self {
            words: FfiSlice::empty(),
            bit_offset: 0,
            bit_len: 0,
        }
    }

    /// How many bits.
    pub fn len(&self) -> usize {
        self.bit_len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.bit_len == 0
    }
}

/// An Arrow array's validity: a bitmap of valid rows and how many are null. With no
/// nulls the bitmap is empty, whatever the row count, and never read.
#[repr(C)]
#[derive(Clone, Copy, Debug, StagedType)]
pub struct FfiValidity {
    bits: FfiBitmap<u8>,
    null_count: u64,
}

impl FfiValidity {
    /// Every row valid: no bits, no nulls.
    pub const fn all_valid() -> Self {
        Self {
            bits: FfiBitmap::empty(),
            null_count: 0,
        }
    }

    /// The validity of an Arrow array of `nulls`.
    pub(crate) fn from_nulls(nulls: Option<&NullBuffer>) -> Self {
        match nulls {
            Some(nulls) if nulls.null_count() != 0 => Self {
                bits: FfiBitmap::new(nulls.inner().values(), nulls.offset(), nulls.len()),
                null_count: nulls.null_count() as u64,
            },
            _ => Self::all_valid(),
        }
    }

    pub fn null_count(&self) -> u64 {
        self.null_count
    }
}

// =============================================================================
// Staged reads
// =============================================================================

/// A staged bitmap of words of `W`.
#[derive(Clone, Copy)]
pub(crate) struct BitmapView<P, W> {
    bitmap: P,
    _word: std::marker::PhantomData<W>,
}

impl<P, W> BitmapView<P, W>
where
    P: Staged<Out = SPtr<FfiBitmap<W>>> + Clone + 'static,
    W: BitmapWord,
{
    /// The bitmap at `bitmap`.
    ///
    /// # Safety
    ///
    /// The descriptor and the words it addresses are alive and unchanged while the
    /// kernel reads them.
    pub(crate) unsafe fn new(bitmap: P) -> Self {
        Self {
            bitmap,
            _word: std::marker::PhantomData,
        }
    }

    /// How many bits.
    pub(crate) fn len(&self) -> impl Staged<Out = u64> + Clone + use<P, W> {
        // SAFETY: `P` addresses an `FfiBitmap<W>` (its type).
        unsafe { load_field_unchecked(self.bitmap.clone(), FfiBitmapType::bit_len::<W>()) }
    }

    /// Bit `index` of the run.
    ///
    /// # Safety
    ///
    /// At execution, `index` is less than the bitmap's length.
    pub(crate) unsafe fn bit<I>(&self, index: I) -> impl Staged<Out = bool> + Clone + use<P, W, I>
    where
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: `P` addresses an `FfiBitmap<W>` (its type).
        let offset =
            unsafe { load_field_unchecked(self.bitmap.clone(), FfiBitmapType::bit_offset::<W>()) };
        let bit = add::<u64, _, _>(offset, index.into_staged());
        let word_index = shr::<u64, _, _>(bit.clone(), W::SHIFT);
        // SAFETY: the descriptor and its words are alive (`new`'s contract); the read is
        // shared, and `bit_offset + index` is inside the run (the caller's bound), which
        // is inside `words` (`FfiBitmap::new`).
        let word = unsafe {
            field_addr(self.bitmap.clone(), FfiBitmapType::words::<W>())
                .into_raw_slice::<W>()
                .assume_shared()
                .get_unchecked(word_index)
        };
        // The bit's place in its word, shifted down to bit 0 in the word's own width,
        // then the low byte.
        let in_word = int_cast::<W, u64, _>(bitand::<u64, _, _>(bit, (1u64 << W::SHIFT) - 1));
        let low = int_cast::<u8, W, _>(shr::<W, _, _>(word, in_word));
        ne(bitand::<u8, _, _>(low, 1u8), 0u8)
    }
}

/// A staged Arrow validity: the null count, and whether a row is valid.
#[derive(Clone, Copy)]
pub struct ValidityView<V> {
    validity: V,
}

impl<V> ValidityView<V>
where
    V: Staged<Out = SPtr<FfiValidity>> + Clone + 'static,
{
    /// The validity at `validity`.
    ///
    /// # Safety
    ///
    /// The descriptor and the bytes it addresses are alive and unchanged while the
    /// kernel reads them.
    pub(crate) unsafe fn new(validity: V) -> Self {
        Self { validity }
    }

    pub fn null_count(&self) -> impl Staged<Out = u64> + Clone + use<V> {
        // SAFETY: `V` addresses an `FfiValidity` (its type).
        unsafe { load_field_unchecked(self.validity.clone(), FfiValidityType::null_count()) }
    }

    /// Whether row `index` is valid: always, without reading the bitmap, when there are
    /// no nulls.
    ///
    /// # Safety
    ///
    /// At execution, `index` is less than the array's row count.
    pub unsafe fn is_valid<I>(&self, index: I) -> impl Staged<Out = bool> + Clone + use<V, I>
    where
        I: IntoStaged<u64>,
        I::Staged: Clone,
    {
        // SAFETY: the bitmap is part of the validity, alive as long as it is (`new`); with
        // nulls it has a bit per row and `index` is a row (the caller), and without, the
        // bit is never read.
        let bit = unsafe {
            BitmapView::<_, u8>::new(field_addr(self.validity.clone(), FfiValidityType::bits()))
                .bit(index)
        };
        if_then_else(eq(self.null_count(), 0u64), Const::<bool>::new(true), bit)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each backend: Cranelift, and LLVM with the `llvm` feature.
    fn for_each_backend(mut body: impl FnMut(Compiler)) {
        body(Compiler::new());
        #[cfg(feature = "llvm")]
        body(Compiler::new().with_backend(JitBackend::Llvm));
    }

    /// No nulls is no bitmap, whether or not Arrow has a null buffer; with nulls, the
    /// bitmap is the buffer's rows from where they start.
    #[test]
    fn a_validity_has_bits_only_with_nulls() {
        for nulls in [None, Some(NullBuffer::new_valid(9))] {
            let validity = FfiValidity::from_nulls(nulls.as_ref());
            assert_eq!(validity.null_count(), 0);
            assert_eq!(validity.bits.bit_len, 0);
            assert!(validity.bits.words.is_empty());
        }
        let nulls = NullBuffer::from(vec![true, false, true, true, false]).slice(1, 3);
        let validity = FfiValidity::from_nulls(Some(&nulls));
        assert_eq!(validity.null_count(), 1);
        assert_eq!((validity.bits.bit_offset, validity.bits.bit_len), (1, 3));
    }

    #[test]
    #[should_panic(expected = "bits 30..34 are past the 32 bits given")]
    fn a_bitmap_holds_its_bits() {
        FfiBitmap::<u32>::new(&[0], 30, 4);
    }

    /// Every bit of a run starting mid-word, in words of every width, against the
    /// same bits read on the host.
    #[test]
    fn bits_in_words_of_any_width() {
        fn check<W: BitmapWord>(words: &[W], host_bit: impl Fn(usize) -> bool) {
            let bits = words.len() << W::SHIFT;
            for_each_backend(|mut compiler| {
                let f = compiler.fun2(
                    "bit",
                    |_ctx, bitmap: Var<SRef<FfiBitmap<W>>>, i: Var<u64>| {
                        // SAFETY: the test's bitmap is alive and `i` inside it.
                        unsafe { BitmapView::<_, W>::new(ref_as_ptr(bitmap)).bit(i) }
                    },
                );
                let compiled = compiler.compile(f).expect("compile");
                for offset in [0, 3, 8, 13] {
                    let bitmap = FfiBitmap::<W>::new(words, offset, bits - offset);
                    for i in 0..bits - offset {
                        let bit = compiled.call(&bitmap, i as u64);
                        assert_eq!(bit, host_bit(offset + i), "offset {offset}, bit {i}");
                    }
                }
            });
        }
        let bytes = [0b1010_0110u8, 0xff, 0x00, 0b0001_1000];
        check::<u8>(&bytes, |i| bytes[i / 8] >> (i % 8) & 1 == 1);
        let shorts = [0x8001u16, 0x1234, 0xfe7f];
        check::<u16>(&shorts, |i| shorts[i / 16] >> (i % 16) & 1 == 1);
        let words = [0x8000_0001u32, 0x1234_5678];
        check::<u32>(&words, |i| words[i / 32] >> (i % 32) & 1 == 1);
        let longs = [0x8000_0000_0000_0001u64 | 1 << 40];
        check::<u64>(&longs, |i| longs[i / 64] >> (i % 64) & 1 == 1);
        let wide = [
            1u128 << 127 | 1 << 70 | 1 << 64 | 0x8000_0000_0000_0001,
            0xdead_beef,
        ];
        check::<u128>(&wide, |i| wide[i / 128] >> (i % 128) & 1 == 1);
    }

    /// No nulls: every row is valid, and the empty bitmap's dangling pointer, which faults
    /// if read, is never read, however far the row.
    #[test]
    fn no_nulls_never_reads_the_bitmap() {
        for_each_backend(|mut compiler| {
            let f = compiler.fun2(
                "valid",
                |_ctx, validity: Var<SRef<FfiValidity>>, row: Var<u64>| {
                    // SAFETY: the test's descriptor is alive; with no nulls any row reads.
                    let validity = unsafe { ValidityView::new(ref_as_ptr(validity)) };
                    unsafe { validity.is_valid(row) }
                },
            );
            let compiled = compiler.compile(f).expect("compile");
            let all_valid = FfiValidity::all_valid();
            for row in [0, 9, 1 << 20] {
                assert!(compiled.call(&all_valid, row));
            }
        });
    }
}
