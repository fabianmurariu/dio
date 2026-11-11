//! `FfiByteViewArray<K>`: every row a kernel reads is `GenericByteViewArray::value(i)`.
//! Arrays are built with small blocks, so most long views have `buf_idx > 0`, and are
//! described together, so all but the first array's buffers start part-way into the
//! shared table. Nulls, sliced arrays and missing columns included.

use arrow::array::{Array, BinaryViewArray, BinaryViewBuilder, StringViewArray, StringViewBuilder};
use arrow_lms::{ByteViewDescriptors, FfiByteViewArray, FfiByteViewArrayOps};
use rust_lms::prelude::*;

/// Strings of every length class, several to a 32-byte block.
fn strings(seed: usize) -> Vec<Option<String>> {
    (0..40)
        .map(|i| {
            let n = (i * 7 + seed) % 23;
            (i % 9 != 4).then(|| "abcdefghijklmnopqrstuvwxyz"[..n].to_owned())
        })
        .collect()
}

fn string_array(values: &[Option<String>]) -> StringViewArray {
    let mut builder = StringViewBuilder::new().with_fixed_block_size(32);
    for v in values {
        builder.append_option(v.as_deref());
    }
    builder.finish()
}

/// Three arrays, one sliced, with a missing column between them.
fn columns() -> Vec<Option<StringViewArray>> {
    let sliced = string_array(&strings(2));
    let sliced = sliced.slice(5, 30);
    vec![
        Some(string_array(&strings(0))),
        None,
        Some(string_array(&strings(1))),
        Some(sliced),
    ]
}

/// Row `(a, i)`, read in a kernel: its value, or `None` when the row is null.
fn read_rows(
    descriptors: &ByteViewDescriptors<Utf8>,
    a: usize,
    rows: usize,
) -> Vec<Option<Vec<u8>>> {
    let mut compiler = Compiler::new();
    let value = compiler.fun3(
        "value",
        |ctx, arrays: Var<SRef<Slice<FfiByteViewArray<Utf8>>>>, a: Var<u64>, i: Var<u64>| {
            // SAFETY: the host passes an array index and a valid row inside it.
            ctx.bind(unsafe { arrays.get_ref_unchecked(a).byte_views().value_unchecked(i) })
        },
    );
    let compiled = compiler.compile(value).expect("compile");
    let mut compiler = Compiler::new();
    let valid = compiler.fun3(
        "valid",
        |ctx, arrays: Var<SRef<Slice<FfiByteViewArray<Utf8>>>>, a: Var<u64>, i: Var<u64>| {
            // SAFETY: the host passes an array index and a row inside it.
            ctx.bind(unsafe {
                arrays
                    .get_ref_unchecked(a)
                    .byte_views()
                    .validity()
                    .is_valid(i)
            })
        },
    );
    let valid = compiler.compile(valid).expect("compile");
    (0..rows)
        .map(|i| {
            let args = (descriptors.arrays(), a as u64, i as u64);
            valid.as_fn().call(args.0, args.1, args.2).then(|| {
                let view = compiled.as_fn().call(args.0, args.1, args.2);
                // SAFETY: the arrays and the descriptors are alive.
                unsafe { view.bytes().to_vec() }
            })
        })
        .collect()
}

#[test]
fn rows_are_the_arrays_values() {
    let columns = columns();
    let descriptors = ByteViewDescriptors::<Utf8>::new(columns.iter().map(Option::as_ref));
    assert_eq!(descriptors.arrays().len(), columns.len());
    for (a, column) in columns.iter().enumerate() {
        let ffi = &descriptors.arrays()[a];
        let Some(column) = column else {
            assert!(ffi.is_empty());
            continue;
        };
        assert!(column.data_buffers().len() > 1, "views with buf_idx > 0");
        assert_eq!(ffi.len(), column.len());
        assert_eq!(ffi.num_buffers(), column.data_buffers().len());
        assert_eq!(ffi.null_count(), column.null_count() as u64);
        let want: Vec<_> = (0..column.len())
            .map(|i| {
                column
                    .is_valid(i)
                    .then(|| column.value(i).as_bytes().to_vec())
            })
            .collect();
        assert_eq!(read_rows(&descriptors, a, column.len()), want, "array {a}");
    }
}

/// The empty column has no rows: a kernel's length check finds nothing there.
#[test]
fn a_missing_column_has_no_rows() {
    let descriptors = ByteViewDescriptors::<Utf8>::new([None]);
    let mut compiler = Compiler::new();
    let len = compiler.fun1(
        "len",
        |ctx, arrays: Var<SRef<Slice<FfiByteViewArray<Utf8>>>>| {
            // SAFETY: one array.
            ctx.bind(unsafe { arrays.get_ref_unchecked(0u64).byte_views().len() })
        },
    );
    let compiled = compiler.compile(len).expect("compile");
    assert_eq!(compiled.as_fn().call(descriptors.arrays()), 0);
}

/// Equal strings stored in different arrays, so different tables and places, are equal,
/// and order across arrays is byte order.
#[test]
fn rows_of_different_arrays_compare_by_bytes() {
    let columns = columns();
    let descriptors = ByteViewDescriptors::<Utf8>::new(columns.iter().map(Option::as_ref));
    let mut compiler = Compiler::new();
    let cmp = compiler.fun5(
        "cmp",
        |ctx,
         arrays: Var<SRef<Slice<FfiByteViewArray<Utf8>>>>,
         a: Var<u64>,
         i: Var<u64>,
         b: Var<u64>,
         j: Var<u64>| {
            // SAFETY: the host passes valid rows of existing arrays.
            let (x, y) = unsafe {
                (
                    ctx.bind(arrays.get_ref_unchecked(a).byte_views().value_unchecked(i)),
                    ctx.bind(arrays.get_ref_unchecked(b).byte_views().value_unchecked(j)),
                )
            };
            x.compare(ctx, y)
        },
    );
    let compiled = compiler.compile(cmp).expect("compile");
    let cmp = compiled.as_fn();
    let present: Vec<(usize, &StringViewArray)> = columns
        .iter()
        .enumerate()
        .filter_map(|(a, c)| c.as_ref().map(|c| (a, c)))
        .collect();
    let mut equal_across = 0;
    for &(a, x) in &present {
        for &(b, y) in &present {
            for i in (0..x.len()).filter(|&i| x.is_valid(i)) {
                for j in (0..y.len()).filter(|&j| y.is_valid(j)) {
                    let want = x.value(i).as_bytes().cmp(y.value(j).as_bytes()) as i8;
                    equal_across += (a != b && want == 0) as usize;
                    assert_eq!(
                        cmp.call(descriptors.arrays(), a as u64, i as u64, b as u64, j as u64),
                        want
                    );
                }
            }
        }
    }
    assert!(equal_across > 0, "some value is in two arrays");
}

#[test]
fn binary_rows_are_the_arrays_values() {
    let values: Vec<Option<Vec<u8>>> = (0..30u8)
        .map(|i| (i % 7 != 3).then(|| (0..i).map(|b| b.wrapping_mul(37) ^ 0x80).collect()))
        .collect();
    let mut builder = BinaryViewBuilder::new().with_fixed_block_size(32);
    for v in &values {
        builder.append_option(v.as_deref());
    }
    let array: BinaryViewArray = builder.finish();
    let descriptors = ByteViewDescriptors::<Binary>::new([Some(&array)]);

    let mut compiler = Compiler::new();
    let value = compiler.fun2(
        "value",
        |ctx, arrays: Var<SRef<Slice<FfiByteViewArray<Binary>>>>, i: Var<u64>| {
            // SAFETY: the host passes a valid row of the one array.
            ctx.bind(unsafe {
                arrays
                    .get_ref_unchecked(0u64)
                    .byte_views()
                    .value_unchecked(i)
            })
        },
    );
    let compiled = compiler.compile(value).expect("compile");
    for (i, want) in values.iter().enumerate() {
        if let Some(want) = want {
            let view = compiled.as_fn().call(descriptors.arrays(), i as u64);
            // SAFETY: the array and the descriptors are alive.
            assert_eq!(unsafe { view.bytes() }, want.as_slice(), "row {i}");
        }
    }
}

/// The descriptors own the table: moving them moves nothing a kernel reads.
#[test]
fn descriptors_can_move() {
    let columns = columns();
    let descriptors = ByteViewDescriptors::<Utf8>::new(columns.iter().map(Option::as_ref));
    let before = read_rows(&descriptors, 2, columns[2].as_ref().unwrap().len());
    let moved = Box::new(descriptors);
    assert_eq!(
        read_rows(&moved, 2, columns[2].as_ref().unwrap().len()),
        before
    );
}
