//! [`SVec`] — a growable dynamic array proving **handle indirection**: bake a
//! pointer to a stable [`RawVec`] control block, never to the movable buffer;
//! every access reloads the buffer pointer from the block, and growth (the
//! [`svec_grow`] extern) reallocs and writes the new pointer back. See the crate
//! docs for the why.

use std::alloc::{Layout, alloc, dealloc, handle_alloc_error, realloc};
use std::marker::PhantomData;

use rust_lms::prelude::*;

/// The stable control block: buffer pointer, element length, element capacity.
/// `#[repr(C)]` + `StagedType` so the kernel reads/writes `ptr`/`len`/`cap` by
/// field; `elem_size`/`elem_align` are host bookkeeping (the monomorphic
/// [`svec_grow`] extern and [`HostVec`]'s allocation math), never touched by the
/// kernel. Lives at a fixed host address ([`HostVec`] boxes it); its address is
/// baked into the kernel.
#[repr(C)]
#[derive(Clone, Copy, StagedType)]
pub struct RawVec {
    #[staged(SMutPtr<u8>)]
    ptr: *mut u8,
    #[staged(u64)]
    len: usize,
    #[staged(u64)]
    cap: usize,
    #[staged(u64)]
    elem_size: usize,
    #[staged(u64)]
    elem_align: usize,
}

fn allocation_layout(cap: usize, elem_size: usize, elem_align: usize) -> Layout {
    let Some(bytes) = cap.checked_mul(elem_size) else {
        std::process::abort();
    };
    let Ok(layout) = Layout::from_size_align(bytes, elem_align) else {
        std::process::abort();
    };
    layout
}

/// An element-typed handle to a [`HostVec`]'s stable control block.
///
/// This prevents the ordinary [`SVec`] constructor from pairing a host vector
/// with the wrong staged runtime element type. It deliberately does not borrow
/// the host owner because staged values are retained by [`Ctx`]; the lifetime
/// requirement therefore remains part of [`SVec::new`]'s safety contract.
pub struct HostVecHandle<R> {
    ctrl: *mut RawVec,
    _r: PhantomData<fn() -> R>,
}

impl<R> Clone for HostVecHandle<R> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<R> Copy for HostVecHandle<R> {}

/// Host owner of an [`SVec`]'s storage: allocates the buffer, keeps the control
/// block at a stable address (in a `Box`), and frees on drop. The host allocates
/// it before the kernel is compiled and keeps it alive across the run — the same
/// "host outlives the kernel" contract as the string [`BytesPool`] and the GROUP
/// BY state. `R` is the **runtime** element type; the kernel-side [`SVec<T>`] uses
/// the matching staged type `T` (with `T::size_of() == size_of::<R>()`).
pub struct HostVec<R> {
    raw: Box<RawVec>,
    _r: PhantomData<R>,
}

impl<R> HostVec<R> {
    /// An empty vec (no allocation until the first `push` grows it).
    pub fn new() -> Self {
        assert!(
            std::mem::size_of::<R>() != 0,
            "HostVec does not support zero-sized element types"
        );
        HostVec {
            raw: Box::new(RawVec {
                ptr: std::ptr::null_mut(),
                len: 0,
                cap: 0,
                elem_size: std::mem::size_of::<R>(),
                elem_align: std::mem::align_of::<R>(),
            }),
            _r: PhantomData,
        }
    }

    /// A typed handle to the stable control block for [`SVec::new`].
    pub fn handle(&mut self) -> HostVecHandle<R> {
        HostVecHandle {
            ctrl: &mut *self.raw,
            _r: PhantomData,
        }
    }

    /// The untyped control-block address for runtime-selected staged types.
    ///
    /// Pairing this pointer with an [`SVec`] requires
    /// [`SVec::from_raw_unchecked`], which makes the lost type relationship
    /// explicit at the call site.
    pub fn as_raw_control_ptr(&mut self) -> *mut RawVec {
        &mut *self.raw
    }

    /// The current element count (as the kernel last left it).
    pub fn len(&self) -> usize {
        self.raw.len
    }

    pub fn is_empty(&self) -> bool {
        self.raw.len == 0
    }

    /// The populated elements, for host-side read-back.
    pub fn as_slice(&self) -> &[R] {
        if self.raw.ptr.is_null() {
            &[]
        } else {
            // SAFETY: `ptr` is valid for `len` elements of `R` (the kernel only
            // wrote indices `< len`, each sized `size_of::<R>()`).
            unsafe { std::slice::from_raw_parts(self.raw.ptr as *const R, self.raw.len) }
        }
    }
}

impl<R> Default for HostVec<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> Drop for HostVec<R> {
    fn drop(&mut self) {
        if !self.raw.ptr.is_null() {
            // SAFETY: `ptr`/`cap` came from `svec_grow` using this same
            // `elem_size`/`elem_align`, so the layout matches the allocation.
            unsafe {
                let layout =
                    allocation_layout(self.raw.cap, self.raw.elem_size, self.raw.elem_align);
                dealloc(self.raw.ptr, layout);
            }
        }
    }
}

/// Grow `v`'s buffer (double the capacity, or 4 from empty), reallocating and
/// writing the new `ptr`/`cap` back into the control block. The **cold path** of
/// [`SVec::push`] — the kernel calls this only when `len == cap`. Monomorphic
/// (element size/align ride in the control block), so it serves any `SVec<T>`.
#[extern_fn]
#[unsafe(no_mangle)]
pub extern "C" fn svec_grow(v: &mut RawVec) {
    let new_cap = if v.cap == 0 {
        4
    } else {
        let Some(new_cap) = v.cap.checked_mul(2) else {
            std::process::abort();
        };
        new_cap
    };
    // SAFETY: sizes/aligns are the ones the buffer was (or will be) allocated with.
    unsafe {
        let new_layout = allocation_layout(new_cap, v.elem_size, v.elem_align);
        let new_ptr = if v.ptr.is_null() {
            alloc(new_layout)
        } else {
            let old_layout = allocation_layout(v.cap, v.elem_size, v.elem_align);
            realloc(v.ptr, old_layout, new_layout.size())
        };
        if new_ptr.is_null() {
            handle_alloc_error(new_layout);
        }
        v.ptr = new_ptr;
        v.cap = new_cap;
    }
}

/// A shared borrow of an [`SVec`]'s storage.
///
/// The borrow is an ordinary Rust one: holding this blocks growth through the
/// owning handle until non-lexical lifetimes end it at the view's last use.
///
/// The lifetime lives on the *view*, never on the staged expression it derefs
/// to: `Ctx` retains staged expressions under a `'static` bound, so a view that
/// itself implemented `Staged` would have its borrow pinned to `'static` and
/// could never be released.
pub struct SVecSlice<'a, T> {
    expr: SVecSliceExpr<T>,
    _borrow: PhantomData<&'a T>,
}

// Manual, so the derives do not impose a spurious `T: Copy` (the element is a
// staged *marker*, not a runtime value).
impl<T> Copy for SVecSlice<'_, T> {}

impl<T> Clone for SVecSlice<'_, T> {
    /// Shared views coexist, like `&[T]`.
    fn clone(&self) -> Self {
        *self
    }
}

// SAFETY: lowers exactly as its erased twin does — same reloading `(ptr, len)`.
unsafe impl<'a, T: StagedType + 'static> Staged for SVecSlice<'a, T> {
    type Out = BorrowedSlice<'a, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.expr.codegen(ctx)
    }
}

impl<'a, T: StagedType + 'static> LifetimeErased for SVecSlice<'a, T> {
    type Out = BorrowedSlice<'a, T>;
    type ErasedOut = BorrowedSlice<'static, T>;

    fn erase_lifetime(self) -> Box<dyn Staged<Out = BorrowedSlice<'static, T>>> {
        Box::new(self.expr)
    }
}



/// A unique borrow of an [`SVec`]'s storage. Neither `Copy` nor `Clone`:
/// duplicating it would duplicate the exclusive capability.
pub struct SVecSliceMut<'a, T> {
    expr: SVecSliceExprMut<T>,
    _borrow: PhantomData<&'a mut T>,
}

/// The staged `(ptr, len)` of an `SVec`'s storage, **reloaded** from the control
/// block. A raw descriptor: it says nothing about validity, which is what the
/// promotions in [`SVecSliceExpr`]/[`SVecSliceExprMut`] assert.
fn raw_svec_slice<T: StagedType + 'static>(
    ctrl: *mut RawVec,
) -> impl Staged<Out = RawSliceMut<T>> + Copy {
    let block = const_mut_ptr::<RawVec>(ctrl);
    // SAFETY: every view/handle constructor guarantees `ctrl` addresses a live
    // control block, and `ptr`/`len` are its declared staged fields.
    let data = ptr_cast_mut::<T, u8, _>(unsafe { load_field_unchecked(block, RawVecType::ptr()) });
    let len = unsafe { load_field_unchecked(block, RawVecType::len()) };
    // SAFETY: `ptr`/`len` are exactly the descriptor the host maintains.
    unsafe { slice_from_raw_parts_mut::<T, _, _>(data, len) }
}

/// A shared staged slice over an [`SVec`]'s storage — the `Deref` target of
/// [`SVecSlice`], and an ordinary [`SRef<Slice<T>>`] expression, so it carries
/// the whole common slice API.
///
/// **Lifetime-free on purpose.** `Ctx` retains staged expressions under a
/// `'static` bound, so a lifetime here would be forced to `'static` and would
/// pin the view's borrow forever. The borrow lives on the guard instead.
///
/// **Reloading on purpose.** Each use re-reads `(ptr, len)` from the control
/// block, so even a copy that outlives its guard observes the current buffer
/// after a growth rather than a stale pointer.
pub(crate) struct SVecSliceExpr<T> {
    ctrl: *mut RawVec,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for SVecSliceExpr<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SVecSliceExpr<T> {}

// SAFETY: lowers to the `(ptr, len)` pair the host control block maintains.
unsafe impl<T: StagedType + 'static> Staged for SVecSliceExpr<T> {
    type Out = BorrowedSlice<'static, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // SAFETY: `SVec`'s constructors require the host storage to outlive every
        // generated use, and the guard that produced this expression borrows the
        // handle, so no growth can be emitted while it is live.
        unsafe { raw_svec_slice::<T>(self.ctrl).assume_shared() }.codegen(ctx)
    }
}

/// The unique twin of [`SVecSliceExpr`] — an [`SRefMut<Slice<T>>`] expression,
/// so it carries the writing ops as well.
pub(crate) struct SVecSliceExprMut<T> {
    ctrl: *mut RawVec,
    _t: PhantomData<fn() -> T>,
}

impl<T> Clone for SVecSliceExprMut<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SVecSliceExprMut<T> {}

// SAFETY: as `SVecSliceExpr`, and the guard that produced it borrows the handle
// mutably, so no other view of the same handle can coexist.
unsafe impl<T: StagedType + 'static> Staged for SVecSliceExprMut<T> {
    type Out = BorrowedSliceMut<'static, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        // SAFETY: see the type-level note; exclusivity comes from the guard.
        unsafe { raw_svec_slice::<T>(self.ctrl).assume_unique() }.codegen(ctx)
    }
}

/// Load `len` from a control block. Shared by the views and [`SVec::len`] so the
/// field offset lives in exactly one place.
fn raw_vec_len(ctx: &mut Ctx, ctrl: *mut RawVec) -> Var<u64> {
    // SAFETY: every constructor of a view or handle guarantees `ctrl` points at
    // a live control block; `len` is one of its declared staged fields.
    ctx.bind(unsafe { load_field_unchecked(const_mut_ptr::<RawVec>(ctrl), RawVecType::len()) })
}

// SAFETY: as `SVecSlice`, and the guard's `&mut` borrow gives exclusivity.
unsafe impl<'a, T: StagedType + 'static> Staged for SVecSliceMut<'a, T> {
    type Out = BorrowedSliceMut<'a, T>;

    fn codegen(&self, ctx: &mut CompilationContext) -> Value {
        self.expr.codegen(ctx)
    }
}

impl<'a, T: StagedType + 'static> LifetimeErased for SVecSliceMut<'a, T> {
    type Out = BorrowedSliceMut<'a, T>;
    type ErasedOut = BorrowedSliceMut<'static, T>;

    fn erase_lifetime(self) -> Box<dyn Staged<Out = BorrowedSliceMut<'static, T>>> {
        Box::new(self.expr)
    }
}

/// The kernel-side, typed handle to a [`HostVec`]'s storage. Generic over the
/// **staged** element type `T`; carries the baked control-block address and the
/// registered [`svec_grow`] extern handle. All ops emit staged code; only `push`'s
/// grow branch calls the extern.
pub struct SVec<T> {
    ctrl: *mut RawVec,
    grow: ExternRef<SvecGrowExtern>,
    _t: PhantomData<T>,
}

impl<T: StagedType + CopyType + 'static> SVec<T> {
    /// Build a handle from a typed host-vector handle and the registered grow
    /// extern (`compiler.extern_fn::<SvecGrowExtern>()`).
    ///
    /// # Safety
    ///
    /// The source [`HostVec`] must remain live and exclusively available to
    /// generated code for every use of the returned handle.
    pub unsafe fn new(
        handle: HostVecHandle<T::RuntimeValue>,
        grow: ExternRef<SvecGrowExtern>,
    ) -> Self {
        // SAFETY: the typed handle establishes the element layout relationship;
        // the caller supplies the remaining lifetime and exclusivity guarantee.
        unsafe { Self::from_raw_unchecked(handle.ctrl, grow) }
    }

    /// Build a handle from an untyped, baked control-block pointer.
    ///
    /// This is the one documented escape hatch, for SQL code generation, where
    /// the staged type is selected dynamically alongside the matching host
    /// vector.
    ///
    /// # Safety
    ///
    /// `ctrl` must point to a live [`RawVec`] whose element layout exactly
    /// matches `T::RuntimeValue`. The control block and its allocation must
    /// remain live and exclusively available to generated code for every use
    /// of the returned handle.
    ///
    /// **Borrow checking is per-handle.** Views borrow the handle they came
    /// from, so a second handle reconstructed over the same control block
    /// borrows independently and the compiler cannot relate the two. Callers
    /// must therefore not reconstruct a handle while a view from an existing one
    /// is live — that is the one aliasing rule this API cannot check for you.
    /// (Reconstructing a handle *to push*, with no view outstanding, is fine and
    /// is what `sql-gen` does per output column.)
    pub unsafe fn from_raw_unchecked(ctrl: *mut RawVec, grow: ExternRef<SvecGrowExtern>) -> Self {
        SVec {
            ctrl,
            grow,
            _t: PhantomData,
        }
    }

    /// Borrow the storage as a shared view. Blocks growth through this handle
    /// until the view is dropped.
    pub fn as_slice(&self) -> SVecSlice<'_, T> {
        SVecSlice {
            expr: SVecSliceExpr {
                ctrl: self.ctrl,
                _t: PhantomData,
            },
            _borrow: PhantomData,
        }
    }

    /// Borrow the storage as a unique view. Blocks growth *and* any further
    /// view through this handle until it is dropped.
    pub fn as_mut_slice(&mut self) -> SVecSliceMut<'_, T> {
        SVecSliceMut {
            expr: SVecSliceExprMut {
                ctrl: self.ctrl,
                _t: PhantomData,
            },
            _borrow: PhantomData,
        }
    }

    /// A staged `*mut RawVec` to the control block (baked address, typed pointee).
    fn ctrl(&self) -> ConstPtr<SMutPtr<RawVec>> {
        const_mut_ptr::<RawVec>(self.ctrl)
    }

    /// The buffer pointer, typed to `T` — reloaded from the control block so it
    /// reflects the latest growth. `*(ctrl.ptr) as *mut T`.
    fn data(&self) -> impl Staged<Out = SMutPtr<T>> + Copy + use<T> {
        // SAFETY: guaranteed by `SVec::new`; `ptr` is a field of the live
        // control block and has the declared staged type.
        ptr_cast_mut::<T, u8, _>(unsafe { load_field_unchecked(self.ctrl(), RawVecType::ptr()) })
    }

    /// Current element count.
    pub fn len(&self, ctx: &mut Ctx) -> Var<u64> {
        raw_vec_len(ctx, self.ctrl)
    }

    /// Element `i` (unchecked). `*(data + i)`.
    ///
    /// # Safety
    ///
    /// At execution, `i` must be less than the vector's initialized length.
    pub unsafe fn get(&self, ctx: &mut Ctx, i: Var<u64>) -> Var<T> {
        let idx = ctx.bind(int_cast::<i64, u64, _>(i));
        // SAFETY: callers of this unchecked operation must keep `i < len`;
        // `SVec::new` guarantees the allocation and element layout.
        ctx.bind(unsafe { load_mut(ptr_offset_mut(self.data(), idx)) })
    }

    /// Store `v` at element `i` (unchecked). `*(data + i) = v`.
    ///
    /// # Safety
    ///
    /// At execution, `i` must be less than the vector's allocated capacity. If
    /// it is beyond the current length, callers must also maintain the vector's
    /// initialized-length invariant before the vector is read or dropped.
    pub unsafe fn set(&mut self, ctx: &mut Ctx, i: Var<u64>, v: Var<T>) {
        let idx = ctx.bind(int_cast::<i64, u64, _>(i));
        // SAFETY: callers of this unchecked operation must keep `i < cap`;
        // `SVec::new` guarantees the allocation and element layout.
        ctx.emit(unsafe { store(ptr_offset_mut(self.data(), idx), v) });
    }

    /// Append `v`, growing the buffer if full. `if len==cap { grow }; data[len]=v; len++`.
    ///
    /// Requires `&mut self`, so growth cannot overlap a view of the same handle:
    /// growth may move the buffer, which would invalidate a `(ptr, len)` a view
    /// had already materialized. A shared view blocks it:
    ///
    /// ```compile_fail
    /// use rust_lms::prelude::*;
    /// use rust_lms_std::{HostVec, SVec, SvecGrowExtern};
    ///
    /// let mut host = HostVec::<i64>::new();
    /// let mut compiler = Compiler::new();
    /// let grow = compiler.extern_fn::<SvecGrowExtern>();
    /// let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };
    /// let _f = compiler.fun0("k", |ctx| {
    ///     let view = svec.as_slice();
    ///     let n = ctx.bind(view.len());
    ///     let v = ctx.bind(int_cast::<i64, u64, _>(n));
    ///     svec.push(ctx, v);
    ///     let _still_live = view.len();
    ///     n
    /// });
    /// ```
    ///
    /// And a unique view excludes every other borrow:
    ///
    /// ```compile_fail
    /// use rust_lms::prelude::*;
    /// use rust_lms_std::{HostVec, SVec, SvecGrowExtern};
    ///
    /// let mut host = HostVec::<i64>::new();
    /// let mut compiler = Compiler::new();
    /// let grow = compiler.extern_fn::<SvecGrowExtern>();
    /// let mut svec = unsafe { SVec::<i64>::new(host.handle(), grow) };
    /// let unique = svec.as_mut_slice();
    /// let shared = svec.as_slice();
    /// drop((unique, shared));
    /// ```
    pub fn push(&mut self, ctx: &mut Ctx, v: Var<T>) {
        let len = self.len(ctx);
        // SAFETY: guaranteed by `SVec::new`; `cap` is a field of the live
        // control block.
        let cap = ctx.bind(unsafe { load_field_unchecked(self.ctrl(), RawVecType::cap()) });
        let ctrl = self.ctrl;
        let grow = self.grow;
        ctx.if_then(eq(len, cap), move |ctx| {
            let ctrl_ptr = const_mut_ptr::<Opaque<RawVec>>(ctrl);
            // SAFETY: `HostVec` owns this stable control block for the kernel
            // call, and generated code is its only accessor during growth.
            ctx.emit(unsafe { call_extern1_unchecked(grow, ctrl_ptr) });
        });
        // `grow` leaves `len` unchanged but may move the buffer — reload `data`.
        let idx = ctx.bind(int_cast::<i64, u64, _>(len));
        // SAFETY: growth establishes `len < cap`, so `data[len]` is writable.
        ctx.emit(unsafe { store(ptr_offset_mut(self.data(), idx), v) });
        let next = ctx.bind(add(len, 1u64));
        // SAFETY: `len` is a writable field of the live control block.
        ctx.emit(unsafe { store(field_addr(self.ctrl(), RawVecType::len()), next) });
    }
}
