//! Optimised Cranelift IR as readable C-like text (see [`crate::func::DEBUG_IR_SOURCE`]).
//!
//! One line per instruction, with Cranelift's own value and block numbers (`v5`, `b3`), so
//! the text reads side by side with `RUST_LMS_DEBUG_IR_OPT`. A jump assigns the target
//! block's parameters as one tuple, `(v5, v8) = (v14, v9)`, since they are assigned at
//! once. Unsigned operators carry a `u` (`<u`, `/u`, `>>u`); other operations read as calls,
//! `umin(v1, v2)`, with the result's type shown when it differs from the operand's
//! (`v9: i64 = uextend(v3)`). An instruction not handled here is printed as Cranelift
//! prints it.

use std::collections::HashMap;
use std::fmt::Write;

use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::{
    Block, BlockArg, BlockCall, ExternalName, FuncRef, Function, Inst, InstructionData, Opcode,
    Value,
};

/// `func`, named `name`, as readable text; `names` holds each called function's name by
/// its Cranelift id.
pub(crate) fn render(name: &str, func: &Function, names: &HashMap<u32, String>) -> String {
    let mut out = String::new();
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
            match line(func, inst, names) {
                Some(text) => {
                    for text in text.lines() {
                        let _ = writeln!(out, "    {text}");
                    }
                }
                None => {
                    let _ = writeln!(out, "    {}", func.dfg.display_inst(inst));
                }
            }
        }
    }
    out.push('}');
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
