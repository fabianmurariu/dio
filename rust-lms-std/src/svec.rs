//! [`SVec`] — a growable dynamic array proving **handle indirection**: bake a
//! pointer to a stable [`RawVec`] control block, never to the movable buffer;
//! every access reloads the buffer pointer from the block, and growth (the
//! [`svec_grow`] extern) reallocs and writes the new pointer back. See the crate
//! docs for the why.

use std::alloc::{alloc, dealloc, handle_alloc_error, realloc, Layout};
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
    /// This is the escape hatch for SQL code generation, where the staged type
    /// is selected dynamically alongside the matching host vector.
    ///
    /// # Safety
    ///
    /// `ctrl` must point to a live [`RawVec`] whose element layout exactly
    /// matches `T::RuntimeValue`. The control block and its allocation must
    /// remain live and exclusively available to generated code for every use
    /// of the returned handle.
    pub unsafe fn from_raw_unchecked(ctrl: *mut RawVec, grow: ExternRef<SvecGrowExtern>) -> Self {
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
        // SAFETY: guaranteed by `SVec::new`; `ptr` is a field of the live
        // control block and has the declared staged type.
        ptr_cast_mut::<T, u8, _>(unsafe { load_field_unchecked(self.ctrl(), RawVecType::ptr()) })
    }

    /// Current element count.
    pub fn len(&self, ctx: &mut Ctx) -> Var<u64> {
        // SAFETY: guaranteed by `SVec::new`; `len` is a field of the live
        // control block.
        ctx.bind(unsafe { load_field_unchecked(self.ctrl(), RawVecType::len()) })
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
    pub unsafe fn set(&self, ctx: &mut Ctx, i: Var<u64>, v: Var<T>) {
        let idx = ctx.bind(int_cast::<i64, u64, _>(i));
        // SAFETY: callers of this unchecked operation must keep `i < cap`;
        // `SVec::new` guarantees the allocation and element layout.
        ctx.emit(unsafe { store(ptr_offset_mut(self.data(), idx), v) });
    }

    /// Append `v`, growing the buffer if full. `if len==cap { grow }; data[len]=v; len++`.
    pub fn push(&self, ctx: &mut Ctx, v: Var<T>) {
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
