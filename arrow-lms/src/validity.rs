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
    /// What only this crate reads of a word.
    pub trait Sealed: Sized {
        /// No bit set.
        const ZERO: Self;
        /// Every bit set.
        const ONES: Self;

        /// Whether bit `i` is set, on the host.
        fn bit(self, i: usize) -> bool;
    }
}

/// A bitmap word: `u8`, `u16`, `u32`, `u64` or `u128`.
pub trait BitmapWord: Int + StagedType<RuntimeValue = Self> + word::Sealed {
    /// `log2(Self::BITS)`: the shift from a bit to its word.
    const SHIFT: u64;
}

macro_rules! bitmap_word {
    ($($ty:ty),*) => {$(
        impl word::Sealed for $ty {
            const ZERO: Self = 0;
            const ONES: Self = <$ty>::MAX;

            fn bit(self, i: usize) -> bool {
                self >> i & 1 == 1
            }
        }
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

    /// Every bit of `words`, from bit 0.
    pub fn from_words(words: &[W]) -> Self {
        Self::new(words, 0, words.len() << W::SHIFT)
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

    /// Bit `i` of the run, read on the host; `None` past it. For tests, which check a
    /// kernel's reads against the descriptors it was given.
    ///
    /// # Safety
    ///
    /// The words are alive.
    #[cfg(any(test, feature = "test-utils"))]
    pub unsafe fn get(&self, i: usize) -> Option<bool> {
        let bit = self.bit_offset as usize + i;
        // SAFETY: the words are alive (the caller); a bit of the run is in them (`new`).
        let words = unsafe { self.words.as_slice() };
        (i < self.len()).then(|| words[bit >> W::SHIFT].bit(bit & ((1 << W::SHIFT) - 1)))
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

/// A staged [`FfiBitmap<W>`], read a bit or a word at a time. Bit `i` of the run is bit
/// `i % W::BITS` of the run's word `i / W::BITS`, wherever the run starts in its words;
/// bits past the run are not set.
pub struct Bitmap<W: BitmapWord> {
    /// From the word the run starts in.
    words: Var<SRef<Slice<W>>>,
    /// Where the run starts in its first word: `bit_offset % W::BITS`.
    shift: Var<u64>,
    len: Var<u64>,
}

impl<W: BitmapWord> Clone for Bitmap<W> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<W: BitmapWord> Copy for Bitmap<W> {}

impl<W: BitmapWord> Bitmap<W> {
    const BITS: u64 = 1 << W::SHIFT;

    /// The bitmap `bitmap` refers to.
    pub fn new<B>(ctx: &Ctx, bitmap: B) -> Self
    where
        B: Staged<Out = SRef<FfiBitmap<W>>> + Clone + 'static,
    {
        let bitmap = ref_as_ptr(bitmap);
        // SAFETY: a shared reference to a live descriptor, whose owner keeps the words it
        // addresses alive and unchanged while the kernel reads them.
        let (offset, len, words) = unsafe {
            (
                load_field_unchecked(bitmap.clone(), FfiBitmapType::bit_offset::<W>()),
                load_field_unchecked(bitmap.clone(), FfiBitmapType::bit_len::<W>()),
                field_addr(bitmap, FfiBitmapType::words::<W>())
                    .into_raw_slice::<W>()
                    .assume_shared(),
            )
        };
        let offset = ctx.bind(offset);
        let words = ctx.bind(words);
        Self {
            // SAFETY: the run is inside its words (`FfiBitmap::new`), so the word it
            // starts in is too.
            words: ctx.bind(unsafe {
                words.subslice_unchecked(shr::<u64, _, _>(offset, W::SHIFT), words.len())
            }),
            shift: ctx.bind(bitand::<u64, _, _>(offset, Self::BITS - 1)),
            len: ctx.bind(len),
        }
    }

    /// How many bits.
    pub fn len(self) -> Var<u64> {
        self.len
    }

    /// Whether bit `i` is set.
    pub fn contains(self, ctx: &Ctx, i: Var<u64>) -> Var<bool> {
        let set = ctx.var(false);
        ctx.if_then(lt(i, self.len), |ctx| {
            let bit = ctx.bind(add::<u64, _, _>(i, self.shift));
            // SAFETY: `i < len`, so `shift + i` is a bit of the run, inside `words`.
            let word = unsafe { self.words.get_unchecked(shr::<u64, _, _>(bit, W::SHIFT)) };
            let in_word = int_cast::<W, u64, _>(bitand::<u64, _, _>(bit, Self::BITS - 1));
            // Shifted down to bit 0 in the word's own width, then the low byte.
            let low = int_cast::<u8, W, _>(shr::<W, _, _>(word, in_word));
            ctx.store(set, ne(bitand::<u8, _, _>(low, 1u8), 0u8));
        });
        set
    }

    /// Word `w` of the run: its bits `w * W::BITS..(w + 1) * W::BITS`, 0 past the run.
    pub fn word(self, ctx: &Ctx, w: Var<u64>) -> Var<W> {
        let word = ctx.var(Const::<W>::new(W::ZERO));
        let start = ctx.bind(shl::<u64, _, _>(w, W::SHIFT));
        ctx.if_then(lt(start, self.len), |ctx| {
            // SAFETY: `start < len`, so bit `shift + start` is in the run and its word,
            // at or after word `w`, is in `words`.
            let low = unsafe { self.words.get_unchecked(w) };
            ctx.store(word, shr::<W, _, _>(low, int_cast::<W, u64, _>(self.shift)));
            // A run that starts mid-word takes its word's high bits from the next.
            let next = ctx.bind(add::<u64, _, _>(w, 1u64));
            ctx.if_then(ne(self.shift, 0u64), |ctx| {
                ctx.if_then(lt(next, self.words.len()), |ctx| {
                    // SAFETY: `next < words.len()`.
                    let high = unsafe { self.words.get_unchecked(next) };
                    let up = int_cast::<W, u64, _>(sub::<u64, _, _>(Self::BITS, self.shift));
                    ctx.store(word, bitor::<W, _, _>(word, shl::<W, _, _>(high, up)));
                });
            });
            // Keep the run's bits of the last word: `1 <= len - start`.
            let past = sub::<u64, _, _>(
                Self::BITS,
                min::<u64, _, _>(sub::<u64, _, _>(self.len, start), Self::BITS),
            );
            let keep = shr::<W, _, _>(Const::<W>::new(W::ONES), int_cast::<W, u64, _>(past));
            ctx.store(word, bitand::<W, _, _>(word, keep));
        });
        word
    }
}

impl<W: BitmapWord + IntNum + RangeStep> Bitmap<W>
where
    W::RuntimeValue: Default,
{
    /// The bits in `start..end` set in any of `bitmaps`, in order. Each word visits
    /// only its set bits, so a zero word costs one compare.
    pub fn set_in(
        ctx: &Ctx,
        bitmaps: &[Self],
        start: Var<u64>,
        end: Var<u64>,
    ) -> impl StagedIterator<Item = u64> + use<W> {
        let bits = bitmaps
            .iter()
            .map(|bitmap| bitmap.len)
            .reduce(|a, b| ctx.bind(max::<u64, _, _>(a, b)))
            .expect("at least one bitmap");
        let bitmaps = bitmaps.to_vec();
        Self::set_in_words(ctx, start, end, bits, move |ctx, w| {
            let word = ctx.var(Const::<W>::new(W::ZERO));
            for bitmap in &bitmaps {
                ctx.store(word, bitor::<W, _, _>(word, bitmap.word(ctx, w)));
            }
            word
        })
    }

    /// The bits in `start..end`, below `bits`, set in `word(w)`, the `w`-th word of
    /// some union: [`set_in`](Self::set_in) for a union staged as a loop.
    pub fn set_in_words<F>(
        ctx: &Ctx,
        start: Var<u64>,
        end: Var<u64>,
        bits: Var<u64>,
        word: F,
    ) -> impl StagedIterator<Item = u64> + use<W, F>
    where
        F: Fn(&Ctx, Var<u64>) -> Var<W> + 'static,
    {
        let end = ctx.bind(min::<u64, _, _>(end, bits));
        let start = ctx.bind(min::<u64, _, _>(start, end));
        let first = ctx.bind(shr::<u64, _, _>(start, W::SHIFT));
        let partial = select(
            ne(bitand::<u64, _, _>(end, Self::BITS - 1), 0u64),
            1u64,
            0u64,
        );
        let last = ctx.bind(add::<u64, _, _>(shr::<u64, _, _>(end, W::SHIFT), partial));
        range(first, last).flat_map(move |ctx, w| {
            let word = word(ctx, w);
            let base = ctx.bind(shl::<u64, _, _>(w, W::SHIFT));
            // Keep bits `lo..hi` of this word: `lo < BITS` and `0 < hi <= BITS`.
            let lo = sub::<u64, _, _>(max::<u64, _, _>(start, base), base);
            let hi = ctx.bind(min::<u64, _, _>(sub::<u64, _, _>(end, base), Self::BITS));
            let ones = || Const::<W>::new(W::ONES);
            let mask = bitand::<W, _, _>(
                shl::<W, _, _>(ones(), int_cast::<W, u64, _>(lo)),
                shr::<W, _, _>(
                    ones(),
                    int_cast::<W, u64, _>(sub::<u64, _, _>(Self::BITS, hi)),
                ),
            );
            set_bits::<W, _>(bitand::<W, _, _>(word, mask))
                .map(move |_, bit| add::<u64, _, _>(base, int_cast::<u64, W, _>(bit)))
        })
    }
}

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

    /// Bits `offset..offset + len` of `words` in every place a run can start and end:
    /// at a word boundary, mid-word, and a word or more in.
    fn runs<W: BitmapWord>(words: &[W]) -> Vec<FfiBitmap<W>> {
        let bits = words.len() << W::SHIFT;
        let word = 1usize << W::SHIFT;
        let mut runs = vec![FfiBitmap::from_words(words), FfiBitmap::empty()];
        for offset in [0, 3, word - 1, word + 5].into_iter().filter(|&o| o < bits) {
            let lens = [0, 1, word - 3, word, bits - offset - 1, bits - offset];
            for len in lens.into_iter().filter(|&len| offset + len <= bits) {
                runs.push(FfiBitmap::new(words, offset, len));
            }
        }
        runs
    }

    /// Whether each bit is set, from below the run's start to past its end, against
    /// the host's reads of the same bits, in words of every width.
    #[test]
    fn a_bitmap_reads_its_bits_from_any_offset() {
        fn check<W: BitmapWord>(words: &[W]) {
            for_each_backend(|mut compiler| {
                let f = compiler.fun2(
                    "contains",
                    |ctx, bitmap: Var<SRef<FfiBitmap<W>>>, i: Var<u64>| {
                        Bitmap::new(ctx, bitmap).contains(ctx, i)
                    },
                );
                let compiled = compiler.compile(f).expect("compile");
                for bitmap in runs(words) {
                    for i in 0..bitmap.len() + 3 {
                        // SAFETY: `words` is alive.
                        let want = unsafe { bitmap.get(i) }.unwrap_or(false);
                        assert_eq!(compiled.call(&bitmap, i as u64), want, "{bitmap:?} bit {i}");
                    }
                }
            });
        }
        check::<u8>(&[0b1010_0110, 0xff, 0x00, 0b0001_1000]);
        check::<u16>(&[0x8001, 0x1234, 0xfe7f]);
        check::<u32>(&[0x8000_0001, 0x1234_5678, 0xdead_beef]);
        check::<u64>(&[0x8000_0000_0000_0001 | 1 << 40, 0x0123_4567_89ab_cdef]);
        check::<u128>(&[
            1 << 127 | 1 << 70 | 1 << 64 | 0x8000_0000_0000_0001,
            0xdead_beef,
        ]);
    }

    /// Each word of a run is its bits in order, the high ones from the next stored
    /// word when the run starts mid-word, and none past the run's end.
    #[test]
    fn a_bitmap_reads_its_words_from_any_offset() {
        fn check<W: BitmapWord>(words: &[W]) {
            let width = 1usize << W::SHIFT;
            for_each_backend(|mut compiler| {
                let f = compiler.fun2(
                    "word",
                    |ctx, bitmap: Var<SRef<FfiBitmap<W>>>, w: Var<u64>| {
                        ctx.bind(int_cast::<u64, W, _>(Bitmap::new(ctx, bitmap).word(ctx, w)))
                    },
                );
                let compiled = compiler.compile(f).expect("compile");
                for bitmap in runs(words) {
                    for w in 0..words.len() + 1 {
                        let want = (0..width)
                            // SAFETY: `words` is alive.
                            .filter(|k| unsafe { bitmap.get(w * width + k) } == Some(true))
                            .fold(0u64, |word, k| word | 1 << k);
                        assert_eq!(
                            compiled.call(&bitmap, w as u64),
                            want,
                            "{bitmap:?} word {w}"
                        );
                    }
                }
            });
        }
        check::<u8>(&[0b1010_0110, 0xff, 0x00, 0b0001_1000]);
        check::<u16>(&[0x8001, 0x1234, 0xfe7f]);
        check::<u32>(&[0x8000_0001, 0x1234_5678, 0xdead_beef]);
        check::<u64>(&[0x8000_0000_0000_0001 | 1 << 40, 0x0123_4567_89ab_cdef]);
    }

    /// The set bits of two runs in a range, in order, as one hash of the sequence:
    /// every pair of their runs, over ranges that start and end inside and outside them.
    #[test]
    fn set_in_is_the_union_of_set_bits_in_a_range() {
        fn hash(bits: impl IntoIterator<Item = u64>) -> u64 {
            bits.into_iter().fold(0, |h: u64, bit| {
                h.wrapping_mul(1_000_003).wrapping_add(bit + 1)
            })
        }
        fn check<W: BitmapWord + IntNum + RangeStep>(a: &[W], b: &[W])
        where
            W::RuntimeValue: Default,
        {
            let width = 1u64 << W::SHIFT;
            for_each_backend(|mut compiler| {
                let f = compiler.fun4(
                    "set_in",
                    |ctx,
                     a: Var<SRef<FfiBitmap<W>>>,
                     b: Var<SRef<FfiBitmap<W>>>,
                     start: Var<u64>,
                     end: Var<u64>| {
                        let bitmaps = [Bitmap::new(ctx, a), Bitmap::new(ctx, b)];
                        Bitmap::set_in(ctx, &bitmaps, start, end).fold(ctx, 0u64, |_, h, bit| {
                            add::<u64, _, _>(
                                mul::<u64, _, _>(h, 1_000_003u64),
                                add::<u64, _, _>(bit, 1u64),
                            )
                        })
                    },
                );
                let compiled = compiler.compile(f).expect("compile");
                for (x, y) in runs(a).into_iter().zip(runs(b).into_iter().rev()) {
                    let bits = x.len().max(y.len()) as u64;
                    for (start, end) in [(0, bits), (3, width + 2), (width - 1, bits + 9), (5, 2)] {
                        // SAFETY: `a` and `b` are alive.
                        let set = |i: u64| unsafe {
                            x.get(i as usize) == Some(true) || y.get(i as usize) == Some(true)
                        };
                        let want = hash((start..end.min(bits)).filter(|&i| set(i)));
                        let got = compiled.call(&x, &y, start, end);
                        assert_eq!(got, want, "{x:?} {y:?} {start}..{end}");
                    }
                }
            });
        }
        check::<u32>(
            &[0x8000_0001, 0x1234_5678, 0xdead_beef],
            &[0x00ff_0000, 0, 0xf000_000f],
        );
        check::<u64>(
            &[0x8000_0000_0000_0001 | 1 << 40, 0x0123_4567_89ab_cdef],
            &[0xf0, 1 << 63],
        );
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
