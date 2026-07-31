//! `ValidityView` used on its own — a mutable validity bitmap manipulated with
//! no primitive array involved (the staged counterpart of `FfiValidity` owns its
//! own bit get/set).

use arrow_lms::{FfiBuffer, FfiValidity, ValidityView};
use rust_lms::prelude::*;

/// An `FfiValidity` over a host-owned bitmap of `len` bits, all-valid.
fn validity_over(bitmap: &mut [u8], bit_len: u64) -> FfiValidity {
    FfiValidity {
        bytes: unsafe { FfiBuffer::from_raw_parts(bitmap.as_mut_ptr(), bitmap.len()) },
        bit_offset: 0,
        bit_len,
        null_count: 0,
    }
}

#[test]
fn set_null_clears_bits_standalone() {
    let mut bitmap = vec![0xFFu8]; // 8 valid bits
    let mut validity = validity_over(&mut bitmap, 8);

    let mut compiler = Compiler::new();
    let f = compiler.fun1("clear", |ctx, v: Var<SRefMut<FfiValidity>>| {
        let view = ValidityView::new(v);
        for bit in [1u64, 3, 6] {
            let i = ctx.var(bit);
            view.set_null(ctx, i);
        }
        Const::<()>::new(())
    });
    let clear = compiler.compile(f).unwrap().as_fn();
    clear(&mut validity);

    // bits 1, 3, 6 cleared
    assert_eq!(bitmap[0], 0b1011_0101);
}

#[test]
fn set_valid_sets_bits_standalone() {
    let mut bitmap = vec![0x00u8]; // all null
    let mut validity = validity_over(&mut bitmap, 8);

    let mut compiler = Compiler::new();
    let f = compiler.fun1("mark", |ctx, v: Var<SRefMut<FfiValidity>>| {
        let view = ValidityView::new(v);
        for bit in [0u64, 2, 5] {
            let i = ctx.var(bit);
            view.set_valid(ctx, i);
        }
        Const::<()>::new(())
    });
    let mark = compiler.compile(f).unwrap().as_fn();
    mark(&mut validity);

    // bits 0, 2, 5 set
    assert_eq!(bitmap[0], 0b0010_0101);
}
