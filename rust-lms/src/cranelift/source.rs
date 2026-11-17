//! Cranelift IR as readable C-like text (see [`crate::compiler::DEBUG_IR_SOURCE`]).
//!
//! One line per instruction, with Cranelift's own value and block numbers (`v5`, `b3`), as
//! `RUST_LMS_DEBUG_IR_OPT` prints them. A jump assigns the target
//! block's parameters as one tuple, `(v5, v8) = (v14, v9)`, since they are assigned at
//! once. Unsigned operators carry a `u` (`<u`, `/u`, `>>u`); other operations read as calls,
//! `umin(v1, v2)`, with the result's type shown when it differs from the operand's
//! (`v9: i64 = uextend(v3)`). An instruction not handled here is printed as Cranelift
//! prints it. With ids kept ([`InstIds`]), each line ends with its instruction's id,
//! `// @12`.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::Write;
use std::sync::Arc;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    Block, BlockArg, BlockCall, ExternalName, FuncRef, Function, Inst, InstructionData, Opcode,
    SourceLoc, Value,
};

/// A call stack, as return addresses, innermost first.
pub(crate) type Stack = Arc<[usize]>;

thread_local! {
    /// The call stack of the statement being replayed, if it was recorded with one.
    static ORIGIN: RefCell<Option<Stack>> = const { RefCell::new(None) };
}

/// The current call stack.
pub(crate) fn capture() -> Stack {
    let mut stack = Vec::new();
    backtrace::trace(|frame| {
        stack.push(frame.ip() as usize);
        true
    });
    stack.into()
}

/// Runs `f` with `origin` as the origin of the instructions it emits: a statement replayed
/// in `compile`, under the call stack it was recorded on.
pub(crate) fn with_origin<R>(origin: Stack, f: impl FnOnce() -> R) -> R {
    /// Puts the outer origin back, also when `f` panics.
    struct Restore(Option<Stack>);
    impl Drop for Restore {
        fn drop(&mut self) {
            ORIGIN.with(|current| current.replace(self.0.take()));
        }
    }
    let _restore = Restore(ORIGIN.with(|current| current.replace(Some(origin))));
    f()
}

/// Hands out instruction ids, unique across one compile, kept by Cranelift as each
/// instruction's `SourceLoc`, with each one's origin: the call stack its statement was
/// recorded on ([`with_origin`]), or else the one it is emitted on.
#[derive(Default)]
pub(crate) struct InstIds {
    /// Each id's call stack (id `n` at `n - 1`), as an index into `stacks`.
    of_id: Vec<u32>,
    /// Each distinct call stack; most instructions share a few.
    stacks: Vec<Stack>,
    stack_index: HashMap<Stack, u32>,
    /// Each address's frames (several when calls were inlined), resolved on first use.
    places: HashMap<usize, Vec<(String, u32)>>,
}

impl InstIds {
    /// A fresh id, with its origin.
    pub(crate) fn next(&mut self) -> SourceLoc {
        let stack = ORIGIN
            .with(|current| current.borrow().clone())
            .unwrap_or_else(capture);
        let index = match self.stack_index.get(&stack) {
            Some(&index) => index,
            None => {
                let index = self.stacks.len() as u32;
                self.stacks.push(stack.clone());
                self.stack_index.insert(stack, index);
                index
            }
        };
        self.of_id.push(index);
        SourceLoc::new(self.of_id.len() as u32)
    }

    /// Where `id` was generated: `file:line` per frame, innermost first (empty for an
    /// unknown id). Resolves every frame, uncached: for looking at a few ids.
    pub(crate) fn origin(&self, id: u32) -> Vec<String> {
        let stack = self.stack(id).unwrap_or_default();
        (stack.iter())
            .flat_map(|&ip| resolve(ip))
            .map(|(file, line)| format!("{}:{line}", short(&file)))
            .collect()
    }

    /// The first frame of `id`'s origin outside rust-lms and the standard library: the
    /// code that asked for the instruction, as its shortened file and line. Caches each
    /// address.
    pub(crate) fn caller(&mut self, id: u32) -> Option<(String, u32)> {
        let stack = self.stack(id)?.to_vec();
        stack.into_iter().find_map(|ip| {
            let frames = self.places.entry(ip).or_insert_with(|| resolve(ip));
            (frames.iter())
                .find(|(file, _)| !is_internal(file))
                .map(|(file, line)| (short(file), *line))
        })
    }

    fn stack(&self, id: u32) -> Option<&[usize]> {
        let index = *self.of_id.get(id.checked_sub(1)? as usize)?;
        Some(&self.stacks[index as usize])
    }
}

/// `ip`'s frames as file and line: several when calls were inlined, innermost first.
fn resolve(ip: usize) -> Vec<(String, u32)> {
    let mut frames = Vec::new();
    backtrace::resolve(ip as *mut std::ffi::c_void, |symbol| {
        if let (Some(file), Some(line)) = (symbol.filename(), symbol.lineno()) {
            frames.push((file.to_string_lossy().into_owned(), line));
        }
    });
    frames
}

/// Whether `file` is in rust-lms itself, the standard library (`/rustc/<hash>/…`, or
/// `rust-src` under `lib/rustlib`) or `backtrace`.
fn is_internal(file: &str) -> bool {
    [
        "/rust-lms/src/",
        "/rustc/",
        "/lib/rustlib/",
        "/backtrace-0.",
    ]
    .iter()
    .any(|dir| file.contains(dir))
}

/// `file` from its crate's directory on: `db4-execution/src/codegen.rs`,
/// `rust-lms/tests/test_source.rs`.
fn short(file: &str) -> String {
    let parts: Vec<_> = file.split('/').collect();
    let dir = |names: &[&str]| parts.iter().rposition(|p| names.contains(p));
    let start = (dir(&["src"]).or_else(|| dir(&["tests", "benches", "examples"])))
        .map_or(parts.len().saturating_sub(2), |i| i.saturating_sub(1));
    parts[start..].join("/")
}

/// `func`, named `name`, as readable text, and each instruction id's line in it (1-based;
/// an instruction printed on several lines is on its last, where its id is); `names` holds
/// each called function's name by its Cranelift id.
pub(crate) fn render(
    name: &str,
    func: &Function,
    names: &HashMap<u32, String>,
) -> (String, Vec<(u32, u32)>) {
    let mut out = String::new();
    let mut lines = Vec::new();
    // The line `out` ends on, counted up to `counted` bytes.
    let (mut at, mut counted) = (1, 0);
    let entry = func.layout.entry_block();
    let params = entry.map_or(&[][..], |block| func.dfg.block_params(block));
    let _ = writeln!(out, "fn {name}({}) {{", typed(func, params));
    for block in func.layout.blocks() {
        let params = func.dfg.block_params(block);
        if Some(block) == entry || params.is_empty() {
            let _ = writeln!(out, "{}:", label(block));
        } else {
            let _ = writeln!(out, "{}({}):", label(block), typed(func, params));
        }
        for inst in func.layout.block_insts(block) {
            let text = (line(func, inst, names))
                .unwrap_or_else(|| func.dfg.display_inst(inst).to_string());
            let _ = write!(out, "    {}", text.replace('\n', "\n    "));
            let id = func.srcloc(inst);
            if !id.is_default() {
                let _ = write!(out, "  // @{}", id.bits());
                at += out[counted..].bytes().filter(|&b| b == b'\n').count() as u32;
                counted = out.len();
                lines.push((id.bits(), at));
            }
            out.push('\n');
        }
    }
    out.push('}');
    (out, lines)
}

/// The size of `ctx`'s compiled code, and each range of it (offsets from its start) with the
/// id of the instruction it came from: `None` for code no instruction asked for (the
/// prologue, the driver's return).
pub(crate) fn code_ranges(ctx: &cranelift_codegen::Context) -> (u32, Vec<crate::func::CodeRange>) {
    let code = ctx.compiled_code().expect("the function was compiled");
    let base = ctx.func.params.base_srcloc();
    let ranges = (code.buffer.get_srclocs_sorted().iter())
        .map(|range| {
            let loc = match range.loc {
                loc if loc.is_rel() => loc.relocate(base),
                loc => loc.as_abs(),
            };
            (
                range.start..range.end,
                (!loc.is_default()).then(|| loc.bits()),
            )
        })
        .collect();
    (code.code_buffer().len() as u32, ranges)
}

/// The code that generated each instruction id in `func`: see [`layout`].
pub(crate) fn origins(func: &Function, ids: &mut InstIds) -> String {
    let in_func: std::collections::BTreeSet<u32> = (func.layout.blocks())
        .flat_map(|block| func.layout.block_insts(block))
        .map(|inst| func.srcloc(inst))
        .filter(|id| !id.is_default())
        .map(|id| id.bits())
        .collect();
    let callers: Vec<_> = in_func.into_iter().map(|id| (id, ids.caller(id))).collect();
    layout(&callers)
}

/// Each id's origin (`ids`: in increasing order, `None` when unknown), grouped by file, in
/// order of first appearance, then by line: one line per origin, listing its ids as ranges,
/// and the unknown ones last, after `?`. A range runs over ids that are next to each other
/// in `ids`, so an id missing from the function does not break it.
///
/// ```text
/// db4-disk-storage/src/ffi/staged/graph.rs
///     :45  @7–8, @21–22
///     :62  @54–55
/// ```
fn layout(ids: &[(u32, Option<(String, u32)>)]) -> String {
    type Ranges = Vec<(usize, usize)>;
    // Positions in `ids`, as ranges (first, last): per file, per line, and the unknown ones.
    let mut files: Vec<(&str, std::collections::BTreeMap<u32, Ranges>)> = Vec::new();
    let mut unknown = Ranges::new();
    let extend = |ranges: &mut Ranges, at: usize| match ranges.last_mut() {
        Some((_, last)) if *last + 1 == at => *last = at,
        _ => ranges.push((at, at)),
    };
    for (at, (_, caller)) in ids.iter().enumerate() {
        let Some((file, line)) = caller else {
            extend(&mut unknown, at);
            continue;
        };
        let lines = match files.iter().position(|(f, _)| f == file) {
            Some(i) => &mut files[i].1,
            None => {
                files.push((file, Default::default()));
                &mut files.last_mut().expect("just pushed").1
            }
        };
        extend(lines.entry(*line).or_default(), at);
    }
    let show = |ranges: &Ranges| {
        (ranges.iter())
            .map(|&(first, last)| match (ids[first].0, ids[last].0) {
                (a, b) if a == b => format!("@{a}"),
                (a, b) => format!("@{a}–{b}"),
            })
            .collect::<Vec<_>>()
            .join(", ")
    };
    let mut out = String::new();
    for (file, lines) in &files {
        let _ = writeln!(out, "{file}");
        let width = (lines.keys()).map(|line| line.to_string().len()).max();
        for (line, ranges) in lines {
            let width = width.unwrap_or(0);
            let _ = writeln!(out, "    :{line:<width$}  {}", show(ranges));
        }
    }
    if !unknown.is_empty() {
        let _ = writeln!(out, "?  {}", show(&unknown));
    }
    out
}

/// `inst` as text, or `None` when it is not handled here.
fn line(func: &Function, inst: Inst, names: &HashMap<u32, String>) -> Option<String> {
    let result = || func.dfg.first_result(inst);
    let pool = &func.dfg.value_lists;
    Some(match func.dfg.insts[inst] {
        InstructionData::UnaryImm {
            opcode: Opcode::Iconst,
            imm,
        } => format!("{} = {}", result(), imm.bits()),
        InstructionData::UnaryIeee64 {
            opcode: Opcode::F64const,
            imm,
        } => format!("{} = {:?}", result(), f64::from_bits(imm.bits())),
        InstructionData::UnaryIeee32 {
            opcode: Opcode::F32const,
            imm,
        } => format!("{} = {:?}", result(), f32::from_bits(imm.bits())),
        InstructionData::Unary { opcode, arg } => {
            let operation = match opcode {
                Opcode::Bnot => format!("~{arg}"),
                Opcode::Ineg | Opcode::Fneg => format!("-{arg}"),
                _ => format!("{opcode}({arg})"),
            };
            format!("{} = {operation}", typed_result(func, result(), arg))
        }
        InstructionData::Binary {
            opcode,
            args: [a, b],
        } => match binary(opcode) {
            Some(op) => format!("{} = {a} {op} {b}", result()),
            None => format!("{} = {opcode}({a}, {b})", typed_result(func, result(), a)),
        },
        InstructionData::Ternary {
            opcode: Opcode::Select,
            args: [c, a, b],
        } => format!("{} = {c} ? {a} : {b}", result()),
        InstructionData::IntCompare {
            cond, args: [a, b], ..
        } => format!("{} = {a} {} {b}", result(), compare(cond)),
        InstructionData::FloatCompare {
            cond, args: [a, b], ..
        } => match float_compare(cond) {
            Some(op) => format!("{} = {a} {op} {b}", result()),
            None => format!("{} = fcmp_{cond}({a}, {b})", result()),
        },
        InstructionData::StackAddr {
            opcode: Opcode::StackAddr,
            stack_slot,
            offset,
        } => format!("{} = &{}", result(), address(stack_slot, offset.into())),
        InstructionData::FuncAddr {
            opcode: Opcode::FuncAddr,
            func_ref,
        } => format!("{} = &{}", result(), callee(func, func_ref, names)),
        InstructionData::Call {
            opcode: Opcode::Call,
            ref args,
            func_ref,
        } => {
            let call = format!(
                "{}({})",
                callee(func, func_ref, names),
                list(args.as_slice(pool))
            );
            match func.dfg.inst_results(inst) {
                [] => call,
                [value] => format!("{value} = {call}"),
                values => format!("({}) = {call}", list(values)),
            }
        }
        InstructionData::Load {
            opcode: Opcode::Load,
            arg,
            offset,
            ..
        } => {
            let ty = func.dfg.value_type(result());
            format!("{}: {ty} = *{}", result(), address(arg, offset.into()))
        }
        InstructionData::Store {
            opcode: Opcode::Store,
            args: [value, ptr],
            offset,
            ..
        } => format!("*{} = {value}", address(ptr, offset.into())),
        InstructionData::Jump { destination, .. } => jump(func, destination)?.join("\n"),
        InstructionData::Brif {
            arg,
            blocks: [then, els],
            ..
        } => {
            if then.len(pool) + els.len(pool) == 0 {
                let (then, els) = (then.block(pool), els.block(pool));
                format!("if {arg} goto {} else {}", label(then), label(els))
            } else {
                let (then, els) = (jump(func, then)?, jump(func, els)?);
                format!(
                    "if {arg} {{ {} }} else {{ {} }}",
                    then.join("; "),
                    els.join("; ")
                )
            }
        }
        InstructionData::MultiAry {
            opcode: Opcode::Return,
            ref args,
        } => match args.as_slice(&func.dfg.value_lists) {
            [] => "return".to_string(),
            values => format!("return {}", list(values)),
        },
        _ => return None,
    })
}

/// `goto` the destination, after assigning its parameters as one tuple: one statement each.
fn jump(func: &Function, destination: BlockCall) -> Option<Vec<String>> {
    let pool = &func.dfg.value_lists;
    let block = destination.block(pool);
    let args = destination
        .args(pool)
        .map(|arg| match arg {
            BlockArg::Value(value) => Some(value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let goto = format!("goto {}", label(block));
    Some(match (func.dfg.block_params(block), &args[..]) {
        ([], _) => vec![goto],
        ([param], [arg]) => vec![format!("{param} = {arg}"), goto],
        (params, args) => vec![format!("({}) = ({})", list(params), list(args)), goto],
    })
}

/// The name of the function `func_ref` refers to, or Cranelift's name for it.
fn callee(func: &Function, func_ref: FuncRef, names: &HashMap<u32, String>) -> String {
    match func.dfg.ext_funcs[func_ref].name {
        ExternalName::User(name) => {
            let id = func.params.user_named_funcs()[name].index;
            (names.get(&id).cloned()).unwrap_or_else(|| func_ref.to_string())
        }
        _ => func_ref.to_string(),
    }
}

/// `result`, with its type when that differs from `operand`'s: a cast or a widening.
fn typed_result(func: &Function, result: Value, operand: Value) -> String {
    let ty = func.dfg.value_type(result);
    if ty == func.dfg.value_type(operand) {
        result.to_string()
    } else {
        format!("{result}: {ty}")
    }
}

fn float_compare(cond: FloatCC) -> Option<&'static str> {
    Some(match cond {
        FloatCC::Equal => "==",
        FloatCC::NotEqual => "!=",
        FloatCC::LessThan => "<",
        FloatCC::LessThanOrEqual => "<=",
        FloatCC::GreaterThan => ">",
        FloatCC::GreaterThanOrEqual => ">=",
        _ => return None,
    })
}

fn binary(opcode: Opcode) -> Option<&'static str> {
    Some(match opcode {
        Opcode::Iadd | Opcode::Fadd => "+",
        Opcode::Isub | Opcode::Fsub => "-",
        Opcode::Imul | Opcode::Fmul => "*",
        Opcode::Sdiv | Opcode::Fdiv => "/",
        Opcode::Udiv => "/u",
        Opcode::Srem => "%",
        Opcode::Urem => "%u",
        Opcode::Band => "&",
        Opcode::Bor => "|",
        Opcode::Bxor => "^",
        Opcode::Ishl => "<<",
        Opcode::Sshr => ">>",
        Opcode::Ushr => ">>u",
        _ => return None,
    })
}

fn compare(cond: IntCC) -> &'static str {
    match cond {
        IntCC::Equal => "==",
        IntCC::NotEqual => "!=",
        IntCC::SignedLessThan => "<",
        IntCC::SignedLessThanOrEqual => "<=",
        IntCC::SignedGreaterThan => ">",
        IntCC::SignedGreaterThanOrEqual => ">=",
        IntCC::UnsignedLessThan => "<u",
        IntCC::UnsignedLessThanOrEqual => "<=u",
        IntCC::UnsignedGreaterThan => ">u",
        IntCC::UnsignedGreaterThanOrEqual => ">=u",
    }
}

/// `ptr`, or `(ptr + offset)`.
fn address(ptr: impl std::fmt::Display, offset: i32) -> String {
    match offset {
        0 => ptr.to_string(),
        offset => format!("({ptr} + {offset})"),
    }
}

/// `block3` as `b3`.
fn label(block: Block) -> String {
    format!("b{}", block.as_u32())
}

fn list(values: &[impl std::fmt::Display]) -> String {
    values
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ")
}

fn typed(func: &Function, values: &[Value]) -> String {
    values
        .iter()
        .map(|&v| format!("{v}: {}", func.dfg.value_type(v)))
        .collect::<Vec<_>>()
        .join(", ")
}

#[cfg(test)]
mod tests {
    use super::{layout, render};
    use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, SourceLoc, types::I64};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
    use std::collections::HashMap;

    #[test]
    fn each_id_is_on_its_line_and_a_jump_on_its_goto() {
        // b0(v0): @1 v2 = 0; @2 goto b1 with v2 (two lines); b1(v1): @3 return v1.
        let mut func = Function::new();
        func.signature.params.push(AbiParam::new(I64));
        func.signature.returns.push(AbiParam::new(I64));
        let mut ctx = FunctionBuilderContext::new();
        let mut b = FunctionBuilder::new(&mut func, &mut ctx);
        let (b0, b1) = (b.create_block(), b.create_block());
        b.append_block_params_for_function_params(b0);
        let p = b.append_block_param(b1, I64);
        b.switch_to_block(b0);
        b.set_srcloc(SourceLoc::new(1));
        let zero = b.ins().iconst(I64, 0);
        b.set_srcloc(SourceLoc::new(2));
        b.ins().jump(b1, &[zero.into()]);
        b.switch_to_block(b1);
        b.set_srcloc(SourceLoc::new(3));
        b.ins().return_(&[p]);
        b.seal_all_blocks();
        let flags = cranelift_codegen::settings::Flags::new(cranelift_codegen::settings::builder());
        b.finalize(
            cranelift_native::builder()
                .unwrap()
                .finish(flags)
                .unwrap()
                .frontend_config(),
        );
        let (text, lines) = render("f", &func, &HashMap::new());
        assert_eq!(
            text,
            "\
fn f(v0: i64) {
b0:
    v2 = 0  // @1
    v1 = v2
    goto b1  // @2
b1(v1: i64):
    return v1  // @3
}"
        );
        assert_eq!(lines, [(1, 3), (2, 5), (3, 7)]);
    }

    fn at(file: &str, line: u32) -> Option<(String, u32)> {
        Some((file.to_string(), line))
    }

    #[test]
    fn origins_group_by_file_then_line_as_ranges() {
        // Id 5 is missing from the function, so 4 and 6 are next to each other.
        let ids = [
            (1, at("a.rs", 10)),
            (2, at("a.rs", 10)),
            (3, at("b.rs", 7)),
            (4, at("a.rs", 10)),
            (6, at("a.rs", 10)),
            (7, at("a.rs", 9)),
            (8, None),
            (9, at("b.rs", 7)),
        ];
        assert_eq!(
            layout(&ids),
            "\
a.rs
    :9   @7
    :10  @1–2, @4–6
b.rs
    :7  @3, @9
?  @8
"
        );
    }
}
