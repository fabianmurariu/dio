"""When a breakpoint is set on a rust-lms `.ir` line, also set one on the Rust line that
generated it (the innermost Rust line in the JIT code's DWARF). Load from ~/.lldbinit:

    command script import /path/to/dio/rust-lms/lldb/ir_rust_bp.py

Needs `RUST_LMS_DEBUG_IR_SOURCE=1` and `settings set plugin.jit-loader.gdb.enable on`.
The Rust breakpoint is added once the kernel is registered, after stage 0 ran: only its
JIT locations stop.
"""
import sys
import threading

import lldb

# ponytail: never removes a Rust breakpoint it added; delete it by hand.
_done = set()  # (IR breakpoint id, Rust file, line) already added


def _ir_file(bp):
    s = lldb.SBStream()
    bp.GetDescription(s)
    return ".ir'" in s.GetData()


def _add_rust(target, bp):
    for loc in bp:
        line = loc.GetAddress().GetLineEntry()
        path = str(line.GetFileSpec())
        key = (bp.GetID(), path, line.GetLine())
        if not line.IsValid() or path.endswith(".ir") or key in _done:
            continue
        _done.add(key)
        rust = target.BreakpointCreateByLocation(path, line.GetLine())
        print(f"IR breakpoint {bp.GetID()}: added breakpoint {rust.GetID()} at {path}:{line.GetLine()}", file=sys.__stdout__, flush=True)


def _listen(debugger):
    listener = lldb.SBListener("ir_rust_bp")
    event = lldb.SBEvent()
    targets = []
    while True:
        # ponytail: looks for new targets every second; listening by event class misses them.
        for target in debugger:
            if target not in targets:
                targets.append(target)
                target.GetBroadcaster().AddListener(listener, lldb.SBTarget.eBroadcastBitBreakpointChanged)
                for bp in target.breakpoint_iter():  # resolved before we listened
                    if _ir_file(bp):
                        _add_rust(target, bp)
        if not listener.WaitForEvent(1, event) or not lldb.SBBreakpoint.EventIsBreakpointEvent(event):
            continue
        kind = lldb.SBBreakpoint.GetBreakpointEventTypeFromEvent(event)
        if kind in (lldb.eBreakpointEventTypeAdded, lldb.eBreakpointEventTypeLocationsAdded):
            bp = lldb.SBBreakpoint.GetBreakpointFromEvent(event)
            if _ir_file(bp):
                _add_rust(bp.GetTarget(), bp)


def __lldb_init_module(debugger, _):
    threading.Thread(target=_listen, args=(debugger,), daemon=True).start()
