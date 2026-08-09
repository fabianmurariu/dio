//! [`SVec`] — a growable dynamic array proving **handle indirection**: bake a
//! pointer to a stable [`RawVec`] control block, never to the movable buffer;
//! every access reloads the buffer pointer from the block, and growth (the
//! [`svec_grow`] extern) reallocs and writes the new pointer back. See the crate
//! docs for the why.

use std::alloc::{alloc, dealloc, realloc, Layout};
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
    pub ptr: *mut u8,
    #[staged(u64)]
    pub len: usize,
    #[staged(u64)]
    pub cap: usize,
    #[staged(u64)]
    pub elem_size: usize,
    #[staged(u64)]
    pub elem_align: usize,
}

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

    /// The stable address of the control block — bake this into the kernel (via
    /// [`SVec::new`]). Valid until this `HostVec` is dropped.
    pub fn control_ptr(&mut self) -> *mut RawVec {
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
                    Layout::from_size_align(self.raw.cap * self.raw.elem_size, self.raw.elem_align)
                        .expect("valid layout");
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
    let new_cap = if v.cap == 0 { 4 } else { v.cap * 2 };
    // SAFETY: sizes/aligns are the ones the buffer was (or will be) allocated with.
    unsafe {
        let new_bytes = new_cap * v.elem_size;
        let new_ptr = if v.ptr.is_null() {
            let layout = Layout::from_size_align(new_bytes, v.elem_align).expect("valid layout");
            alloc(layout)
        } else {
            let old_layout =
                Layout::from_size_align(v.cap * v.elem_size, v.elem_align).expect("valid layout");
            realloc(v.ptr, old_layout, new_bytes)
        };
        assert!(!new_ptr.is_null(), "svec_grow: allocation failed");
        v.ptr = new_ptr;
        v.cap = new_cap;
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

impl<T> Clone for SVec<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for SVec<T> {}

impl<T: StagedType + CopyType + 'static> SVec<T> {
    /// Build a handle from a baked control-block pointer ([`HostVec::control_ptr`])
    /// and the registered grow extern (`compiler.extern_fn::<SvecGrowExtern>()`).
    pub fn new(ctrl: *mut RawVec, grow: ExternRef<SvecGrowExtern>) -> Self {
        SVec {
            ctrl,
            grow,
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
        ptr_cast_mut::<T, u8, _>(load_field(self.ctrl(), RawVecType::ptr()))
    }

    /// Current element count.
    pub fn len(&self, ctx: &mut Ctx) -> Var<u64> {
        ctx.bind(load_field(self.ctrl(), RawVecType::len()))
    }

    /// Element `i` (unchecked). `*(data + i)`.
    pub fn get(&self, ctx: &mut Ctx, i: Var<u64>) -> Var<T> {
        let idx = ctx.bind(int_cast::<i64, u64, _>(i));
        ctx.bind(load_ref_mut(ptr_offset_mut(self.data(), idx)))
    }

    /// Store `v` at element `i` (unchecked). `*(data + i) = v`.
    pub fn set(&self, ctx: &mut Ctx, i: Var<u64>, v: Var<T>) {
        let idx = ctx.bind(int_cast::<i64, u64, _>(i));
        ctx.emit(store(ptr_offset_mut(self.data(), idx), v));
    }

    /// Append `v`, growing the buffer if full. `if len==cap { grow }; data[len]=v; len++`.
    pub fn push(&self, ctx: &mut Ctx, v: Var<T>) {
        let len = self.len(ctx);
        let cap = ctx.bind(load_field(self.ctrl(), RawVecType::cap()));
        let ctrl = self.ctrl;
        let grow = self.grow;
        ctx.if_then(eq(len, cap), move |ctx| {
            ctx.emit(call_extern1(grow, const_opaque_mut::<RawVec>(ctrl)));
        });
        // `grow` leaves `len` unchanged but may move the buffer — reload `data`.
        let idx = ctx.bind(int_cast::<i64, u64, _>(len));
        ctx.emit(store(ptr_offset_mut(self.data(), idx), v));
        let next = ctx.bind(add(len, 1u64));
        ctx.emit(store_ref(field_addr(self.ctrl(), RawVecType::len()), next));
    }
}
