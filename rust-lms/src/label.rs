//! Scoped jump targets: the handles behind [`Ctx::block`](crate::func::Ctx::block),
//! [`Ctx::join`](crate::func::Ctx::join), [`Ctx::repeat`](crate::func::Ctx::repeat) and
//! [`Ctx::iterate`](crate::func::Ctx::iterate).
//!
//! These are Rust's labelled `break` and `continue`, staged:
//!
//! | `Ctx`                              | Rust                          |
//! |------------------------------------|-------------------------------|
//! | `ctx.block(\|ctx, out\| …)`        | `'out: { … }`                 |
//! | `ctx.join(\|ctx, out\| … v)`       | `'out: { … v }`               |
//! | `ctx.goto(out, x)` / `ctx.exit(out)` | `break 'out x` / `break 'out` |
//! | `ctx.repeat(\|ctx, again\| … v)`   | `'again: loop { …; break v }` |
//! | `ctx.again(again)`                 | `continue 'again`             |
//! | `ctx.iterate(\|ctx, done\| …)`     | `'done: loop { … }`           |
//!
//! `block`, `join` and `repeat` are *not* loop scopes: a
//! [`break_loop`](crate::func::Ctx::break_loop) inside them passes straight
//! through to the enclosing `iterate`/`while_loop`. That is what lets a
//! combinator retry or merge locally without capturing a terminal's early exit.
//!
//! # The brand
//!
//! Each scope hands its closure a handle whose lifetime `'s` is universally
//! quantified (`for<'s> FnOnce(&mut Ctx, Label<'s, _>)`), so the handle cannot
//! leave the closure. The target block is sealed when the closure's code has
//! been emitted, so a jump staged after that would add a predecessor to a
//! sealed block. The brand makes that a compile error:
//!
//! ```compile_fail
//! use rust_lms::prelude::*;
//!
//! fn leak(ctx: &mut Ctx) {
//!     let mut leaked = None;
//!     ctx.block(|_ctx, out| leaked = Some(out));
//!     ctx.exit(leaked.unwrap());
//! }
//! ```
//!
//! Only the label's plain id enters the `'static` codegen queue; the brand exists
//! at stage 0 alone, as with [`Ctx::bind_lt`](crate::func::Ctx::bind_lt).

use std::marker::PhantomData;

/// An invariant lifetime: `'s` can be neither shortened nor lengthened, so a
/// handle branded with one scope never unifies with another scope's.
type Brand<'s> = PhantomData<fn(&'s ()) -> &'s ()>;

/// A forward jump target that receives a `T` — the parameters of a merge block,
/// i.e. a phi. `Label<'s>` (`T = ()`) is a plain target, jumped to with
/// [`Ctx::exit`](crate::func::Ctx::exit).
pub struct Label<'s, T = ()> {
    pub(crate) id: usize,
    _brand: Brand<'s>,
    _value: PhantomData<fn(T)>,
}

impl<T> Clone for Label<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Label<'_, T> {}

impl<T> Label<'_, T> {
    pub(crate) fn new(id: usize) -> Self {
        Label {
            id,
            _brand: PhantomData,
            _value: PhantomData,
        }
    }
}

/// A backward jump target: [`Ctx::again`](crate::func::Ctx::again) re-runs the
/// body of the [`repeat`](crate::func::Ctx::repeat) that created it.
pub struct Again<'s> {
    pub(crate) id: usize,
    _brand: Brand<'s>,
}

impl Clone for Again<'_> {
    fn clone(&self) -> Self {
        *self
    }
}
impl Copy for Again<'_> {}

impl Again<'_> {
    pub(crate) fn new(id: usize) -> Self {
        Again {
            id,
            _brand: PhantomData,
        }
    }
}

// =============================================================================
// Dead: the fall-through value of a join whose every path already jumped
// =============================================================================

use crate::staged::{CompilationContext, Staged, Value, ValueId};
use crate::types::{ScalarType, StagedType};

/// A placeholder `T` for code that is never reached — the fall-through of a
/// [`join`](crate::func::Ctx::join) body whose every path ends in a jump.
/// Emitted into a dead block, so its (zero) value is never observed.
pub(crate) struct Dead<T>(PhantomData<fn() -> T>);

pub(crate) fn dead<T>() -> Dead<T> {
    Dead(PhantomData)
}

fn zero_leaf(ctx: &mut CompilationContext, ty: ScalarType) -> ValueId {
    match ty {
        ScalarType::F32 => ctx.f32const(0.0),
        ScalarType::F64 => ctx.f64const(0.0),
        ScalarType::Ptr => ctx.null_ptr(),
        ty => ctx.iconst(ty, 0),
    }
}

// SAFETY: the value is only ever produced in unreachable code; it has `T`'s
// leaf layout (fat → ptr + len, otherwise `T::scalar_type`).
unsafe impl<T: StagedType> Staged for Dead<T> {
    type Out = T;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        if T::is_fat_pointer() {
            Value::fat(ctx.null_ptr(), ctx.iconst(ScalarType::I64, 0))
        } else {
            Value::scalar(zero_leaf(ctx, T::scalar_type()))
        }
    }
}
