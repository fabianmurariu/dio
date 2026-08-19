//! `ValidityView` used on its own — a mutable validity bitmap manipulated with
//! no primitive array involved (the staged counterpart of `FfiValidity` owns its
//! own bit get/set).

use arrow_lms::{prepare_validity_mut, FfiValidityMut, ValidityView};
use rust_lms::prelude::*;

#[test]
fn set_null_clears_bits_standalone() {
    let mut bitmap = vec![0xFFu8]; // 8 valid bits
    let mut validity = prepare_validity_mut(&mut bitmap, 8).unwrap();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("clear", |ctx, v: Var<SRefMut<FfiValidityMut>>| {
        let mut view = ValidityView::new(v);
        for bit in [1u64, 3, 6, 3] {
            let i = ctx.var(bit);
            // SAFETY: every constant index is below the prepared bit length 8.
            unsafe { view.set_null(ctx, i) };
        }
        Const::<()>::new(())
    });
    let compiled = compiler.compile(f).unwrap();
    let clear = compiled.as_fn();
    clear.call(validity.descriptor_mut());

    assert_eq!(validity.null_count(), 3);
    drop(validity);
    // bits 1, 3, 6 cleared
    assert_eq!(bitmap[0], 0b1011_0101);
}

#[test]
fn set_valid_sets_bits_standalone() {
    let mut bitmap = vec![0x00u8]; // all null
    let mut validity = prepare_validity_mut(&mut bitmap, 8).unwrap();

    let mut compiler = Compiler::new();
    let f = compiler.fun1("mark", |ctx, v: Var<SRefMut<FfiValidityMut>>| {
        let mut view = ValidityView::new(v);
        for bit in [0u64, 2, 5, 2] {
            let i = ctx.var(bit);
            // SAFETY: every constant index is below the prepared bit length 8.
            unsafe { view.set_valid(ctx, i) };
        }
        Const::<()>::new(())
    });
    let compiled = compiler.compile(f).unwrap();
    let mark = compiled.as_fn();
    mark.call(validity.descriptor_mut());

    assert_eq!(validity.null_count(), 5);
    drop(validity);
    // bits 0, 2, 5 set
    assert_eq!(bitmap[0], 0b0010_0101);
}
