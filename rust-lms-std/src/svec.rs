//! [`SVec`] — a growable dynamic array proving **handle indirection**: bake a
//! pointer to a stable [`RawVec`] control block, never to the movable buffer;
//! every access reloads the buffer pointer from the block, and growth (the
//! [`svec_grow`] extern) reallocs and writes the new pointer back. See the crate
//! docs for the why.

use std::alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout};
use std::cell::Cell;
use std::marker::PhantomData;
use std::rc::Rc;

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
#[no_mangle]
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

/// Stage-0 borrow state for one [`SVec`] handle: `> 0` = that many shared
/// views, `-1` = one unique view, `0` = unborrowed.
///
/// This never reaches codegen. Staging *is* an ordinary Rust program, and it
/// runs in exactly emission order — the timeline on which a reallocation
/// invalidates a previously materialized `(ptr, len)`. So a `RefCell`-style
/// counter checked while the kernel is being *built* is strictly more precise
/// than a type-level borrow, which Rust can only scope to the whole staging
/// region (a view retained in the `Ctx` graph keeps its borrow alive until
/// compilation, so lexical reborrow recovery is unreachable).
///
/// A violation panics while the kernel is under construction — a build-time
/// failure for a staging library, not a production one.
#[derive(Clone, Default)]
struct StageBorrows(Rc<Cell<i64>>);

impl StageBorrows {
    fn acquire_shared(&self) {
        let n = self.0.get();
        assert!(
            n >= 0,
            "staged: cannot take a shared view of an SVec while a unique view is outstanding"
        );
        self.0.set(n + 1);
    }

    fn acquire_unique(&self) {
        assert_eq!(
            self.0.get(),
            0,
            "staged: cannot take a unique view of an SVec while another view is outstanding"
        );
        self.0.set(-1);
    }

    fn release_shared(&self) {
        self.0.set(self.0.get() - 1);
    }

    fn release_unique(&self) {
        self.0.set(0);
    }

    /// Growth may move the buffer, so it is forbidden while any view is live.
    fn assert_unborrowed(&self, what: &str) {
        assert_eq!(
            self.0.get(),
            0,
            "staged: cannot {what} an SVec while a view of it is outstanding \
             (drop the view first; see docs/refactor_slice_api.md rows 7-8)"
        );
    }
}

/// A shared borrow of an [`SVec`]'s storage, tracked at stage 0.
///
/// Holding one blocks growth through the owning handle until it is dropped.
/// Row 8 gives this a `Deref` to a reloading staged slice expression; for now it
/// carries the borrow and the length read that borrow licenses.
pub struct SVecSlice<T> {
    ctrl: *mut RawVec,
    borrows: StageBorrows,
    _t: PhantomData<T>,
}

impl<T> Drop for SVecSlice<T> {
    fn drop(&mut self) {
        self.borrows.release_shared();
    }
}

impl<T: StagedType + CopyType + 'static> SVecSlice<T> {
    /// Current element count, reloaded from the control block.
    pub fn len(&self, ctx: &mut Ctx) -> Var<u64> {
        raw_vec_len(ctx, self.ctrl)
    }
}

impl<T> Clone for SVecSlice<T> {
    /// Shared views coexist, like `&[T]` — cloning takes another borrow.
    fn clone(&self) -> Self {
        self.borrows.acquire_shared();
        SVecSlice {
            ctrl: self.ctrl,
            borrows: self.borrows.clone(),
            _t: PhantomData,
        }
    }
}

/// A unique borrow of an [`SVec`]'s storage, tracked at stage 0. Neither `Copy`
/// nor `Clone`: duplicating it would duplicate the exclusive capability.
pub struct SVecSliceMut<T> {
    ctrl: *mut RawVec,
    borrows: StageBorrows,
    _t: PhantomData<T>,
}

impl<T> Drop for SVecSliceMut<T> {
    fn drop(&mut self) {
        self.borrows.release_unique();
    }
}

impl<T: StagedType + CopyType + 'static> SVecSliceMut<T> {
    /// Current element count, reloaded from the control block.
    pub fn len(&self, ctx: &mut Ctx) -> Var<u64> {
        raw_vec_len(ctx, self.ctrl)
    }
}

/// Load `len` from a control block. Shared by the views and [`SVec::len`] so the
/// field offset lives in exactly one place.
fn raw_vec_len(ctx: &mut Ctx, ctrl: *mut RawVec) -> Var<u64> {
    // SAFETY: every constructor of a view or handle guarantees `ctrl` points at
    // a live control block; `len` is one of its declared staged fields.
    ctx.bind(unsafe { load_field_unchecked(const_mut_ptr::<RawVec>(ctrl), RawVecType::len()) })
}

/// The kernel-side, typed handle to a [`HostVec`]'s storage. Generic over the
/// **staged** element type `T`; carries the baked control-block address and the
/// registered [`svec_grow`] extern handle. All ops emit staged code; only `push`'s
/// grow branch calls the extern.
pub struct SVec<T> {
    ctrl: *mut RawVec,
    grow: ExternRef<SvecGrowExtern>,
    borrows: StageBorrows,
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
    /// **The borrow tracker is per-handle.** Each handle carries its own
    /// counter, so views taken from *this* handle are invisible to any other
    /// handle over the same control block. Callers must therefore not
    /// reconstruct a second handle while a view from an existing one is live —
    /// that is the one aliasing rule stage-0 tracking cannot catch for you.
    /// (Reconstructing a handle to push, with no view outstanding, is fine and
    /// is what `sql-gen` does per output column.)
    pub unsafe fn from_raw_unchecked(ctrl: *mut RawVec, grow: ExternRef<SvecGrowExtern>) -> Self {
        SVec {
            ctrl,
            grow,
            borrows: StageBorrows::default(),
            _t: PhantomData,
        }
    }

    /// Borrow the storage as a shared view. Blocks growth through this handle
    /// until the view is dropped.
    pub fn as_slice(&self) -> SVecSlice<T> {
        self.borrows.acquire_shared();
        SVecSlice {
            ctrl: self.ctrl,
            borrows: self.borrows.clone(),
            _t: PhantomData,
        }
    }

    /// Borrow the storage as a unique view. Blocks growth *and* any further
    /// view through this handle until it is dropped.
    pub fn as_mut_slice(&mut self) -> SVecSliceMut<T> {
        self.borrows.acquire_unique();
        SVecSliceMut {
            ctrl: self.ctrl,
            borrows: self.borrows.clone(),
            _t: PhantomData,
        }
    }

    /// A staged `*mut RawVec` to the control block (baked address, typed pointee).
    fn ctrl(&self) -> ConstPtr<SMutPtr<RawVec>> {
        const_mut_ptr::<RawVec>(self.ctrl)
    }

    /// The buffer pointer, typed to `T` — reloaded from the control block so it
    /// reflects the latest growth. `*(ctrl.ptr) as *mut T`.
    fn data(&self) -> impl Staged<Out = SMutPtr<T>> + Copy {
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
    /// Requires `&mut self`, and panics at kernel-build time if any view of this
    /// handle is outstanding: growth may move the buffer, which would invalidate
    /// a `(ptr, len)` the view had already materialized.
    pub fn push(&mut self, ctx: &mut Ctx, v: Var<T>) {
        self.borrows.assert_unborrowed("grow");
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
