//! `FfiSlice<T>`, the typed `(ptr, len)` field of a `#[repr(C)]` descriptor, in the
//! three shapes a data layer builds from it (`relational/jit_stage6a.md` M0):
//!
//! - a field of a descriptor generic over its element type, not at offset 0;
//! - a slice of descriptors, each holding slices of its own;
//! - a slice of opaque fixed-size slots, each cast to the descriptor the kernel was
//!   compiled to find there: one runtime container for columns of different types.

use std::mem::{align_of, size_of};

use rust_lms::{
    prelude::*,
    types::{Leaves, ScalarType},
};

mod common;
use common::for_each_backend;

/// A column of `T`s with a field before its values, so `values` is not at offset 0.
#[repr(C)]
#[derive(StagedType)]
pub struct Column<T: StagedType> {
    nulls: u64,
    values: FfiSlice<T>,
}

impl<T: StagedType> Clone for Column<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T: StagedType> Copy for Column<T> {}

impl<T: StagedType> Column<T> {
    fn new(nulls: u64, values: &[T::RuntimeValue]) -> Self {
        Self {
            nulls,
            values: FfiSlice::new(values),
        }
    }
}

/// A larger descriptor of a different shape: three slices and three counts, 72 bytes.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct Wide {
    a: FfiSlice<u64>,
    b: FfiSlice<u32>,
    c: FfiSlice<u16>,
    x: u64,
    y: u64,
    z: u64,
}

/// One descriptor of any of the types above, in a fixed-size slot: what a batch holds
/// per column. Kernels never read a slot; they cast it to the descriptor it holds.
#[repr(C, align(8))]
#[derive(Clone, Copy)]
pub struct SlotBytes([u8; 72]);

#[derive(Clone, Copy)]
pub struct Slot;

// SAFETY: a 72-byte, 8-aligned value held by a pointer to its storage (no leaves), as a
// struct is; its runtime value is exactly that.
unsafe impl StagedType for Slot {
    type RuntimeValue = SlotBytes;

    fn scalar_type() -> ScalarType {
        ScalarType::Ptr
    }

    fn size_of() -> usize {
        size_of::<SlotBytes>()
    }

    fn align_of() -> usize {
        align_of::<SlotBytes>()
    }

    fn is_copy_struct() -> bool {
        true
    }

    fn leaves() -> Option<Leaves> {
        None
    }
}

unsafe impl CopyType for Slot {}

const _: () = {
    assert!(size_of::<Column<i64>>() == 24);
    assert!(size_of::<Wide>() == 72);
    assert!(size_of::<SlotBytes>() == 72 && align_of::<SlotBytes>() == 8);
};

/// The slot holding `descriptor`, which must fit it.
fn slot<D: Copy>(descriptor: D) -> SlotBytes {
    assert!(size_of::<D>() <= size_of::<SlotBytes>() && align_of::<D>() <= 8);
    let mut slot = SlotBytes([0; 72]);
    // SAFETY: `D` fits the slot and its alignment divides the slot's, checked above.
    unsafe { std::ptr::write(slot.0.as_mut_ptr().cast::<D>(), descriptor) };
    slot
}

/// A column's values, read as a trusted slice.
///
/// # Safety
///
/// The test keeps the values alive and unchanged for the call.
unsafe fn values<T: StagedType + 'static>(
    ctx: &Ctx,
    column: Var<SRef<Column<T>>>,
) -> Var<SRef<Slice<T>>> {
    let field = column.get_ref(ColumnType::values::<T>());
    // SAFETY: forwarded from the caller.
    ctx.bind(unsafe { field.into_raw_slice::<T>().assume_shared() })
}

/// The field of a generic descriptor, behind another field, reads as `Slice<T>` for
/// two element sizes; `len` counts `T`s.
#[test]
fn a_slice_field_of_a_generic_descriptor() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun2(
            "sum",
            |ctx, wide: Var<SRef<Column<i64>>>, narrow: Var<SRef<Column<u8>>>| {
                // SAFETY: the test keeps both columns' values alive.
                let (wide, narrow) = unsafe { (values(ctx, wide), values(ctx, narrow)) };
                let narrow_sum =
                    narrow
                        .staged_iter()
                        .fold(ctx, Const::<u64>::new(0), |_, acc, x| {
                            add(acc, int_cast::<u64, u8, _>(x))
                        });
                let lens = mul(add(wide.len(), narrow.len()), 1000u64);
                add(
                    add(
                        wide.staged_iter().sum(ctx),
                        int_cast::<i64, u64, _>(narrow_sum),
                    ),
                    int_cast::<i64, u64, _>(lens),
                )
            },
        );
        let compiled = compiler.compile(f).expect("compile");
        let (wide, narrow) = ([10i64, 20, 30], [1u8, 2, 3, 4, 250]);
        let result = compiled.call(
            &Column::<i64>::new(0, &wide),
            &Column::<u8>::new(0, &narrow),
        );
        assert_eq!(result, 60 + 260 + 8 * 1000);
        let empty = compiled.call(&Column::<i64>::new(0, &[]), &Column::<u8>::new(0, &[]));
        assert_eq!(empty, 0);
    });
}

/// A slice of descriptors, each with a slice of its own: db4's `FfiPropColumns`, an
/// arrow-lms batch of one column type.
#[test]
fn a_slice_of_descriptors() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("sum", |ctx, columns: Var<SRef<FfiSlice<Column<i64>>>>| {
            // SAFETY: the test keeps the descriptors and their values alive.
            let columns =
                ctx.bind(unsafe { columns.into_raw_slice::<Column<i64>>().assume_shared() });
            let total = ctx.var(0i64);
            range(0u64, columns.len()).for_each(ctx, |ctx, i| {
                // SAFETY: `i` is inside the slice; the values are alive, as above.
                let column = ctx.bind(unsafe { columns.get_ref_unchecked(i) });
                let values = unsafe { values(ctx, column) };
                let nulls = column.get(ColumnType::nulls::<i64>());
                ctx.store(
                    total,
                    add(
                        add(total, values.staged_iter().sum(ctx)),
                        int_cast::<i64, u64, _>(mul(nulls, 100u64)),
                    ),
                );
            });
            total
        });
        let compiled = compiler.compile(f).expect("compile");
        let (a, b, c) = ([1i64, 2], [], [3i64, 4, 5]);
        let columns = [
            Column::<i64>::new(1, &a),
            Column::new(0, &b),
            Column::new(2, &c),
        ];
        assert_eq!(compiled.call(&FfiSlice::new(&columns)), 15 + 300);
    });
}

/// Columns of three descriptor types in one slice of slots, each cast to the type the
/// kernel was compiled to find at its index. Slots are 72 bytes apart, so a wrong stride
/// or a cast at the wrong address reads the wrong column.
#[test]
fn slots_cast_to_the_descriptor_each_holds() {
    for_each_backend(|mut compiler| {
        let f = compiler.fun1("sum", |ctx, slots: Var<SRef<FfiSlice<Slot>>>| {
            // SAFETY: the test keeps the slots and everything they point to alive.
            let slots = ctx.bind(unsafe { slots.into_raw_slice::<Slot>().assume_shared() });
            // SAFETY (each cast): the test puts these descriptor types at these indices.
            let at = |i: u64| unsafe { slots.get_ptr_unchecked(i) };
            let longs = ctx.bind(unsafe {
                ptr_as_ref::<Column<i64>, _>(ptr_cast::<Column<i64>, Slot, _>(at(0)))
            });
            let wide = unsafe { ptr_as_ref::<Wide, _>(ptr_cast::<Wide, Slot, _>(at(1))) };
            let bytes = ctx.bind(unsafe {
                ptr_as_ref::<Column<u8>, _>(ptr_cast::<Column<u8>, Slot, _>(at(2)))
            });
            let wide = ctx.bind(wide);
            // SAFETY: as above.
            let (longs, bytes) = unsafe { (values(ctx, longs), values(ctx, bytes)) };
            let c = ctx.bind(unsafe {
                wide.get_ref(WideType::c())
                    .into_raw_slice::<u16>()
                    .assume_shared()
            });
            let c_sum = c
                .staged_iter()
                .fold(ctx, Const::<u64>::new(0), |_, acc, x| {
                    add(acc, int_cast::<u64, u16, _>(x))
                });
            let b_len = unsafe {
                wide.get_ref(WideType::b())
                    .into_raw_slice::<u32>()
                    .assume_shared()
            }
            .len();
            let z = wide.get(WideType::z());
            let bytes_len = bytes.len();
            let wide_part = add(add(c_sum, mul(b_len, 10u64)), mul(z, 100u64));
            add(
                longs.staged_iter().sum(ctx),
                int_cast::<i64, u64, _>(add(wide_part, mul(bytes_len, 10_000u64))),
            )
        });
        let compiled = compiler.compile(f).expect("compile");
        let longs = [1i64, 2, 3];
        let (a, b, c) = ([7u64], [1u32, 2], [5u16, 6, 7]);
        let bytes = [9u8; 4];
        let slots = [
            slot(Column::<i64>::new(0, &longs)),
            slot(Wide {
                a: FfiSlice::new(&a),
                b: FfiSlice::new(&b),
                c: FfiSlice::new(&c),
                x: 0,
                y: 0,
                z: 3,
            }),
            slot(Column::<u8>::new(0, &bytes)),
        ];
        // 6 (longs) + 18 (c) + 20 (b's len) + 300 (z) + 40_000 (bytes' len)
        assert_eq!(
            compiled.call(&FfiSlice::new(&slots)),
            6 + 18 + 20 + 300 + 40_000
        );
    });
}
