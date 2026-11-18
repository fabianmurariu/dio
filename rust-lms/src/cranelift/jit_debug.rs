//! Hands each JIT function's ELF image ([`super::dwarf::image`]) to a debugger, through the
//! GDB JIT interface: a list of images the debugger reads whenever the process calls
//! `__jit_debug_register_code`, which it breaks on. GDB reads it as is; LLDB needs
//! `settings set plugin.jit-loader.gdb.enable on`. Off with the `llvm` feature: LLVM's JIT
//! defines the same two symbols.

use std::sync::Mutex;

#[repr(C)]
struct Entry {
    next: *mut Entry,
    prev: *mut Entry,
    image: *const u8,
    size: u64,
}

#[repr(C)]
struct Descriptor {
    version: u32,
    action: u32,
    relevant: *mut Entry,
    first: *mut Entry,
}

const REGISTER: u32 = 1;
const UNREGISTER: u32 = 2;

// The debugger finds both by name; the names and layout are the interface's.
#[unsafe(no_mangle)]
static mut __jit_debug_descriptor: Descriptor = Descriptor {
    version: 1,
    action: 0,
    relevant: std::ptr::null_mut(),
    first: std::ptr::null_mut(),
};

#[unsafe(no_mangle)]
#[inline(never)]
extern "C" fn __jit_debug_register_code() {
    // Not empty, so it is not merged with another function or optimised away.
    std::hint::black_box(());
}

/// Guards the list: compiles and drops on other threads change it too.
static LIST: Mutex<()> = Mutex::new(());

/// One image the debugger knows of; drop it before the code it describes is freed.
pub(crate) struct Registration {
    entry: *mut Entry,
    _image: Box<[u8]>,
}

/// Tells the debugger of `image`, until the returned registration is dropped.
pub(crate) fn register(image: Vec<u8>) -> Registration {
    let image = image.into_boxed_slice();
    let entry = Box::into_raw(Box::new(Entry {
        next: std::ptr::null_mut(),
        prev: std::ptr::null_mut(),
        image: image.as_ptr(),
        size: image.len() as u64,
    }));
    let _list = LIST.lock().unwrap_or_else(|e| e.into_inner());
    // SAFETY: the list is only changed under `LIST`; `entry` is ours until unregistered.
    unsafe {
        let descriptor = &raw mut __jit_debug_descriptor;
        (*entry).next = (*descriptor).first;
        if let Some(first) = (*descriptor).first.as_mut() {
            first.prev = entry;
        }
        (*descriptor).first = entry;
        (*descriptor).relevant = entry;
        (*descriptor).action = REGISTER;
        __jit_debug_register_code();
    }
    Registration {
        entry,
        _image: image,
    }
}

impl Drop for Registration {
    fn drop(&mut self) {
        let entry = self.entry;
        let _list = LIST.lock().unwrap_or_else(|e| e.into_inner());
        // SAFETY: as in `register`; `entry` is in the list until here.
        unsafe {
            let descriptor = &raw mut __jit_debug_descriptor;
            match (*entry).prev.as_mut() {
                Some(prev) => prev.next = (*entry).next,
                None => (*descriptor).first = (*entry).next,
            }
            if let Some(next) = (*entry).next.as_mut() {
                next.prev = (*entry).prev;
            }
            (*descriptor).relevant = entry;
            (*descriptor).action = UNREGISTER;
            __jit_debug_register_code();
            drop(Box::from_raw(entry));
        }
    }
}

/// A copy of each image in the list, as the debugger sees it.
#[cfg(test)]
pub(crate) fn registered() -> Vec<Vec<u8>> {
    let _list = LIST.lock().unwrap_or_else(|e| e.into_inner());
    let mut images = Vec::new();
    // SAFETY: as in `register`; every entry's image lives as long as the entry.
    unsafe {
        let mut entry = __jit_debug_descriptor.first;
        while let Some(e) = entry.as_ref() {
            images.push(std::slice::from_raw_parts(e.image, e.size as usize).to_vec());
            entry = e.next;
        }
    }
    images
}
