//! DWARF for the JIT code, so a debugger shows the readable text (see
//! [`crate::func::DEBUG_IR_SOURCE`]): one compile unit per function, its file the function's
//! `<name>.ir`, its line table each range of code at the line of the instruction it came
//! from.

use std::collections::HashMap;
use std::path::Path;

use gimli::constants::*;
use gimli::write::{
    Address, AttributeValue, Dwarf, EndianVec, Expression, LineProgram, LineString, Unit,
};
use gimli::{Encoding, Format, LineEncoding, NativeEndian, SectionId};

use crate::func::CodeMap;

/// Each non-empty DWARF section, by its id.
pub(crate) type Sections = Vec<(SectionId, Vec<u8>)>;

/// The DWARF for `code`, each function's text at `<dir>/<name>.ir` (`lines`: each function's
/// ids and their lines in its text, in `code`'s order). Code no instruction asked for, and
/// gaps, are at line 1, the function's header.
pub(crate) fn build(dir: &Path, code: &[CodeMap], lines: &[Vec<(u32, u32)>]) -> Sections {
    let encoding = Encoding {
        format: Format::Dwarf32,
        version: 4,
        address_size: size_of::<usize>() as u8,
    };
    let string = |s: &str| s.as_bytes().to_vec();
    let dir = string(&dir.to_string_lossy());
    let mut dwarf = Dwarf::default();
    for (function, lines) in code.iter().zip(lines) {
        let file = string(&format!("{}.ir", function.name));
        let mut program = LineProgram::new(
            encoding,
            LineEncoding::default(),
            LineString::String(dir.clone()),
            None,
            LineString::String(file.clone()),
            None,
        );
        let file_id = program.add_file(
            LineString::String(file.clone()),
            program.default_directory(),
            None,
        );
        let line_of: HashMap<u32, u32> = lines.iter().copied().collect();
        let start = Address::Constant(function.address as u64);
        program.begin_sequence(Some(start));
        let row = |program: &mut LineProgram, offset: u32, line: u32| {
            let row = program.row();
            (row.address_offset, row.line, row.file) = (offset.into(), line.into(), file_id);
            program.generate_row();
        };
        let mut end = 0;
        for (range, id) in &function.ranges {
            if range.start > end {
                row(&mut program, end, 1);
            }
            let line = id.and_then(|id| line_of.get(&id).copied()).unwrap_or(1);
            row(&mut program, range.start, line);
            end = range.end;
        }
        program.end_sequence(function.size.into());

        let unit = dwarf.units.add(Unit::new(encoding, program));
        let unit = dwarf.units.get_mut(unit);
        let root = unit.root();
        let cu = unit.get_mut(root);
        cu.set(DW_AT_producer, AttributeValue::String(string("rust-lms")));
        cu.set(DW_AT_language, AttributeValue::Language(DW_LANG_C99));
        cu.set(DW_AT_name, AttributeValue::String(file));
        cu.set(DW_AT_comp_dir, AttributeValue::String(dir.clone()));
        cu.set(DW_AT_low_pc, AttributeValue::Address(start));
        cu.set(DW_AT_high_pc, AttributeValue::Udata(function.size.into()));
        let subprogram = unit.add(root, DW_TAG_subprogram);
        let subprogram = unit.get_mut(subprogram);
        let mut frame_base = Expression::new();
        frame_base.op(DW_OP_call_frame_cfa);
        subprogram.set(DW_AT_name, AttributeValue::String(string(&function.name)));
        subprogram.set(DW_AT_external, AttributeValue::Flag(true));
        subprogram.set(DW_AT_decl_file, AttributeValue::FileIndex(Some(file_id)));
        subprogram.set(DW_AT_decl_line, AttributeValue::Udata(1));
        subprogram.set(DW_AT_low_pc, AttributeValue::Address(start));
        subprogram.set(DW_AT_high_pc, AttributeValue::Udata(function.size.into()));
        subprogram.set(DW_AT_frame_base, AttributeValue::Exprloc(frame_base));
    }
    let mut sections = gimli::write::Sections::new(EndianVec::new(NativeEndian));
    dwarf.write(&mut sections).expect("DWARF of in-memory code");
    let mut out = Vec::new();
    sections
        .for_each(|id, data| {
            if !data.slice().is_empty() {
                out.push((id, data.slice().to_vec()));
            }
            Ok::<_, gimli::write::Error>(())
        })
        .expect("collecting sections");
    out
}

#[cfg(test)]
mod tests {
    use crate::prelude::*;
    use gimli::{EndianSlice, NativeEndian};

    /// `address`'s file and line, from the line tables in `sections`.
    fn line_at(sections: &super::Sections, address: u64) -> Option<(String, u64)> {
        let empty: &[u8] = &[];
        let dwarf = gimli::Dwarf::load(|id| {
            let data = sections
                .iter()
                .find(|(s, _)| *s == id)
                .map_or(empty, |(_, d)| d);
            Ok::<_, gimli::Error>(EndianSlice::new(data, NativeEndian))
        })
        .unwrap();
        let mut units = dwarf.units();
        while let Some(header) = units.next().unwrap() {
            let unit = dwarf.unit(header).unwrap();
            let program = unit.line_program.clone().unwrap();
            let mut rows = program.rows();
            // The last row at or before `address`, in a sequence that goes past it.
            let mut found = None;
            while let Some((header, row)) = rows.next_row().unwrap() {
                if row.end_sequence() {
                    if row.address() > address && found.is_some() {
                        return found;
                    }
                    found = None;
                } else if row.address() <= address {
                    let file = row.file(header).unwrap().path_name();
                    let file = dwarf.attr_string(&unit, file).unwrap();
                    let line = row.line().unwrap().get();
                    found = Some((file.to_string_lossy().into_owned(), line));
                }
            }
        }
        None
    }

    #[test]
    fn an_instructions_code_is_at_its_line_in_its_file() {
        let mut compiler = Compiler::new().with_debug_ir_source(true);
        let count_to_n = compiler.fun1("count_to_n", |ctx, n: Var<i64>| {
            let i = ctx.var(0i64);
            let sum = ctx.var(0i64);
            ctx.while_loop(lt(i, n), move |ctx| {
                ctx.store(sum, sum + i);
                ctx.store(i, i + 1i64);
            });
            sum
        });
        let compiled = compiler.compile(count_to_n).unwrap();
        let (_, text) = (compiled.sources().iter())
            .find(|(name, _)| name == "count_to_n")
            .unwrap();
        assert_eq!(text.lines().nth(11), Some("    v9 = v8 + v5  // @7"));
        let code = (compiled.code_ranges().iter())
            .find(|code| code.name == "count_to_n")
            .unwrap();
        let (range, _) = (code.ranges.iter()).find(|(_, id)| *id == Some(7)).unwrap();
        let sections = compiled.dwarf.as_ref().expect("DWARF with the text");
        // Its first byte and its last: the whole range is at line 12.
        for offset in [range.start, range.end - 1] {
            let at = (code.address + offset as usize) as u64;
            assert_eq!(line_at(sections, at), Some(("count_to_n.ir".into(), 12)));
        }
        // Code no instruction asked for (the driver's `return`): at the header, line 1.
        let (range, _) = (code.ranges.iter())
            .find(|(_, id)| id.is_none())
            .expect("the driver's return");
        let at = (code.address + range.start as usize) as u64;
        assert_eq!(line_at(sections, at), Some(("count_to_n.ir".into(), 1)));
    }
}
