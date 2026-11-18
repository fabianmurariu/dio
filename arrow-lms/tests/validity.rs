//! Validity read through an array's `validity()`, against Arrow's own `is_valid`:
//! sliced arrays whose bitmaps start mid-byte, and arrays with no nulls. The bitmap
//! itself is tested in `src/validity.rs`.

use arrow::{
    array::{Array, Int32Array},
    buffer::NullBuffer,
};

use arrow_lms::{ArrayBatchOps, ArrowArrayOps, FfiArray, FfiPrimArray, prepare_dyn_arrays};
use rust_lms::prelude::*;

mod common;
use common::for_each_backend;

type ValidityFn = FunType2<SRef<Slice<FfiArray>>, SRefMut<Slice<u64>>, ()>;

/// A kernel reading up to 64 rows' validity, one bit per row, and the null count, from
/// its batch's only column, an `Int32` one.
fn validity_kernel(mut compiler: Compiler) -> Compiled<ValidityFn> {
    let f = compiler.fun2(
        "validity",
        |ctx, batch: Var<SRef<Slice<FfiArray>>>, out: Var<SRefMut<Slice<u64>>>| {
            // SAFETY: every batch the test passes is one Int32 column.
            let column = ctx.bind(unsafe { batch.column::<FfiPrimArray<i32>>(0) });
            let validity = column.validity();
            let len = ctx.bind(column.len());
            let mask = ctx.var(0u64);
            range(0u64, len).for_each(ctx, |ctx, i| {
                // SAFETY: `i` is a row.
                let valid = unsafe { validity.is_valid(i) };
                let bit = shl::<u64, _, _>(Const::<u64>::new(1), i);
                ctx.store(mask, bitor::<u64, _, _>(mask, select(valid, bit, 0u64)));
            });
            let mut out = out;
            // SAFETY: the test passes two `u64`s.
            ctx.emit(unsafe { out.reborrow().set_unchecked(0u64, mask) });
            ctx.emit(unsafe { out.set_unchecked(1u64, validity.null_count()) });
            unit()
        },
    );
    compiler.compile(f).expect("compile")
}

/// What `kernel` reads of `array`.
fn read_validity(kernel: &Compiled<ValidityFn>, array: &Int32Array) -> (u64, u64) {
    let prepared = prepare_dyn_arrays(&[array]).unwrap();
    let mut out = [0u64; 2];
    kernel.call(prepared.arrays(), &mut out);
    (out[0], out[1])
}

/// What Arrow says: one bit per valid row, and the null count.
fn expected(array: &Int32Array) -> (u64, u64) {
    let mask = (0..array.len())
        .filter(|&i| array.is_valid(i))
        .fold(0u64, |mask, i| mask | (1 << i));
    (mask, array.null_count() as u64)
}

/// 40 rows, nulls in an irregular pattern crossing byte boundaries.
fn nullable() -> Int32Array {
    let valid: Vec<bool> = (0..40).map(|i| i % 3 != 0 && i % 7 != 5).collect();
    Int32Array::new(
        (0..40).collect::<Vec<i32>>().into(),
        Some(NullBuffer::from(valid)),
    )
}

/// A slice at bit offset 0, mid-byte (3), and on the next byte (8): each row is valid
/// exactly when Arrow says so.
#[test]
fn is_valid_matches_arrow_at_any_bit_offset() {
    for_each_backend(|compiler| {
        let kernel = validity_kernel(compiler);
        for offset in [0, 3, 8] {
            let sliced = nullable().slice(offset, 29);
            assert_eq!(sliced.nulls().unwrap().offset(), offset);
            assert_eq!(
                read_validity(&kernel, &sliced),
                expected(&sliced),
                "offset {offset}"
            );
        }
    });
}

/// Arrow arrays with no nulls, with and without a null buffer: every row valid.
#[test]
fn no_nulls_is_every_row_valid() {
    for_each_backend(|compiler| {
        let kernel = validity_kernel(compiler);
        let plain = Int32Array::from(vec![1, 2, 3]);
        assert_eq!(read_validity(&kernel, &plain), (0b111, 0));
        let no_nulls = Int32Array::new(vec![1, 2, 3].into(), Some(NullBuffer::new_valid(3)));
        assert_eq!(read_validity(&kernel, &no_nulls), (0b111, 0));
    });
}
