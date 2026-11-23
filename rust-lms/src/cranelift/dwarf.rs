//! DWARF for the JIT code, so a debugger shows the readable text (see
//! [`crate::compiler::DEBUG_IR_SOURCE`]): per function, an ELF image whose `.text` is the
//! function's code in memory, and one compile unit: its file the function's `<name>.ir`, its
//! line table each range of code at the line of the instruction it came from.

use std::collections::HashMap;
use std::path::Path;

use gimli::constants::*;
use gimli::write::{
    Address, AttributeValue, Dwarf, EndianVec, Expression, FileId, LineProgram, LineString, Unit,
    UnitEntryId,
};
use gimli::{Encoding, Format, LineEncoding, NativeEndian, SectionId};

use crate::cranelift::source::Frame;
use crate::func::CodeMap;

/// Each non-empty DWARF section, by its id.
type Sections = Vec<(SectionId, Vec<u8>)>;

/// An ELF image of `function` for a debugger: a `.text` with no bytes at the function's
/// address, its symbol, and its DWARF (see [`build`]). `None` on a target ELF has no name for.
pub(crate) fn image(
    dir: &Path,
    function: &CodeMap,
    lines: &[(u32, u32)],
    chains: &HashMap<u32, Vec<Frame>>,
) -> Option<Vec<u8>> {
    use object::write::{Object, Symbol, SymbolSection};
    use object::{Architecture, BinaryFormat, Endianness, SectionKind, SymbolFlags, SymbolKind};
    let architecture = match std::env::consts::ARCH {
        "aarch64" => Architecture::Aarch64,
        "x86_64" => Architecture::X86_64,
        _ => return None,
    };
    let mut elf = Object::new(BinaryFormat::Elf, architecture, Endianness::default());
    // The code is already in memory: `.text` only says where, and how long.
    let text = elf.add_section(
        Vec::new(),
        b".text".to_vec(),
        SectionKind::UninitializedData,
    );
    elf.append_section_bss(text, function.size.into(), 16);
    elf.section_mut(text).flags = object::SectionFlags::Elf {
        sh_type: object::elf::SHT_NOBITS,
        sh_flags: object::elf::SHF_ALLOC | object::elf::SHF_EXECINSTR,
    };
    elf.add_symbol(Symbol {
        name: function.name.as_bytes().to_vec(),
        value: 0,
        size: function.size.into(),
        kind: SymbolKind::Text,
        scope: object::SymbolScope::Compilation,
        weak: false,
        section: SymbolSection::Section(text),
        flags: SymbolFlags::None,
    });
    for (id, data) in build(dir, function, lines, chains) {
        let section = elf.add_section(Vec::new(), id.name().into(), SectionKind::Debug);
        elf.append_section_data(section, &data, 1);
    }
    let mut image = elf.write().expect("ELF of in-memory sections");
    set_text_address(&mut image, function.address as u64);
    Some(image)
}

/// Sets `.text`'s address to `address`: `object` writes 0, as for a file not yet loaded.
fn set_text_address(image: &mut [u8], address: u64) {
    use object::Endian;
    use object::read::elf::{FileHeader, SectionHeader};
    type Header = object::elf::FileHeader64<object::Endianness>;
    let header = Header::parse(&*image).expect("the ELF just written");
    let endian = header.endian().unwrap();
    let sections = header.sections(endian, &*image).unwrap();
    let (index, text) = (sections.iter().enumerate())
        .find(|(_, s)| sections.section_name(endian, *s) == Ok(b".text"))
        .expect("a .text");
    // Where `text.sh_addr` is in `image`.
    let at = header.e_shoff(endian) as usize
        + index * header.e_shentsize(endian) as usize
        + std::mem::offset_of!(object::elf::SectionHeader64<object::Endianness>, sh_addr);
    debug_assert_eq!(text.sh_addr(endian), 0);
    image[at..at + 8].copy_from_slice(&endian.write_u64(address));
}

/// The DWARF for `function`, its text at `<dir>/<name>.ir` (`lines`: its ids and their lines in
/// the text). Two units over the same code: the first at each id's innermost Rust origin
/// (`chains`), the rest of its origin and its `.ir` line as inlined callers; the second at the
/// `.ir` lines, so a debugger can break on either. Code no instruction asked for, and gaps, are
/// at line 1, the function's header.
fn build(
    dir: &Path,
    function: &CodeMap,
    lines: &[(u32, u32)],
    chains: &HashMap<u32, Vec<Frame>>,
) -> Sections {
    let encoding = Encoding {
        format: Format::Dwarf32,
        version: 4,
        address_size: size_of::<usize>() as u8,
    };
    let dir = dir.to_string_lossy().as_bytes().to_vec();
    let mut dwarf = Dwarf::default();
    add_unit(&mut dwarf, encoding, &dir, function, lines, chains);
    add_unit(&mut dwarf, encoding, &dir, function, lines, &HashMap::new());
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

/// One unit of [`build`]: each id's code at its innermost frame in `chains`, else at its line.
fn add_unit(
    dwarf: &mut Dwarf,
    encoding: Encoding,
    dir: &[u8],
    function: &CodeMap,
    lines: &[(u32, u32)],
    chains: &HashMap<u32, Vec<Frame>>,
) {
    let string = |s: &str| s.as_bytes().to_vec();
    let file = string(&format!("{}.ir", function.name));
    let mut program = LineProgram::new(
        encoding,
        LineEncoding::default(),
        LineString::String(dir.to_vec()),
        None,
        LineString::String(file.clone()),
        None,
    );
    let directory = program.default_directory();
    let ir = program.add_file(LineString::String(file.clone()), directory, None);
    let mut files: HashMap<&str, FileId> = HashMap::new();
    for frame in chains.values().flatten() {
        files.entry(&frame.file).or_insert_with(|| {
            program.add_file(LineString::String(string(&frame.file)), directory, None)
        });
    }
    let line_of: HashMap<u32, u32> = lines.iter().copied().collect();
    let innermost = |id: Option<u32>| match id.and_then(|id| chains.get(&id)?.first()) {
        Some(frame) => (files[frame.file.as_str()], frame.line),
        None => (ir, id.and_then(|id| line_of.get(&id).copied()).unwrap_or(1)),
    };
    let start = Address::Constant(function.address as u64);
    program.begin_sequence(Some(start));
    let row = |program: &mut LineProgram, offset: u32, (file, line): (FileId, u32)| {
        let row = program.row();
        (row.address_offset, row.line, row.file) = (offset.into(), line.into(), file);
        program.generate_row();
    };
    let mut end = 0;
    for (range, id) in &function.ranges {
        if range.start > end {
            row(&mut program, end, (ir, 1));
        }
        row(&mut program, range.start, innermost(*id));
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
    cu.set(DW_AT_comp_dir, AttributeValue::String(dir.to_vec()));
    cu.set(DW_AT_low_pc, AttributeValue::Address(start));
    cu.set(DW_AT_high_pc, AttributeValue::Udata(function.size.into()));
    // One abstract subprogram per Rust function in the chains, for the inlined frames' names.
    let mut abstract_of: HashMap<&str, UnitEntryId> = HashMap::new();
    for frame in chains.values().flatten() {
        abstract_of.entry(&frame.name).or_insert_with(|| {
            let entry = unit.add(root, DW_TAG_subprogram);
            let entry_ref = unit.get_mut(entry);
            entry_ref.set(DW_AT_name, AttributeValue::String(string(&frame.name)));
            entry_ref.set(DW_AT_inline, AttributeValue::Inline(DW_INL_inlined));
            entry
        });
    }
    let subprogram = unit.add(root, DW_TAG_subprogram);
    let subprogram_ref = unit.get_mut(subprogram);
    let mut frame_base = Expression::new();
    frame_base.op(DW_OP_call_frame_cfa);
    subprogram_ref.set(DW_AT_name, AttributeValue::String(string(&function.name)));
    subprogram_ref.set(DW_AT_external, AttributeValue::Flag(true));
    subprogram_ref.set(DW_AT_decl_file, AttributeValue::FileIndex(Some(ir)));
    subprogram_ref.set(DW_AT_decl_line, AttributeValue::Udata(1));
    subprogram_ref.set(DW_AT_low_pc, AttributeValue::Address(start));
    subprogram_ref.set(DW_AT_high_pc, AttributeValue::Udata(function.size.into()));
    subprogram_ref.set(DW_AT_frame_base, AttributeValue::Exprloc(frame_base));
    // Each id's chain, outermost first, called from the id's `.ir` line, then each frame
    // from the line of the frame outside it.
    for (range, id) in &function.ranges {
        let Some((id, chain)) = id.and_then(|id| Some((id, chains.get(&id)?))) else {
            continue;
        };
        let line = line_of.get(&id).copied().unwrap_or(1);
        let (mut parent, mut call) = (subprogram, (ir, line));
        for frame in chain.iter().rev() {
            let entry = unit.add(parent, DW_TAG_inlined_subroutine);
            let entry_ref = unit.get_mut(entry);
            let low = Address::Constant((function.address + range.start as usize) as u64);
            entry_ref.set(
                DW_AT_abstract_origin,
                AttributeValue::UnitRef(abstract_of[frame.name.as_str()]),
            );
            entry_ref.set(DW_AT_low_pc, AttributeValue::Address(low));
            entry_ref.set(
                DW_AT_high_pc,
                AttributeValue::Udata((range.end - range.start).into()),
            );
            entry_ref.set(DW_AT_call_file, AttributeValue::FileIndex(Some(call.0)));
            entry_ref.set(DW_AT_call_line, AttributeValue::Udata(call.1.into()));
            (parent, call) = (entry, (files[frame.file.as_str()], frame.line));
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::cranelift::jit_debug::registered;
    use crate::prelude::*;
    use gimli::{EndianSlice, NativeEndian};
    use object::read::elf::{ElfFile64, SectionHeader};
    use object::{Object, ObjectSection, ObjectSymbol};

    /// Where `image`'s `.text` is, and the name of its symbol.
    fn text_of(image: &[u8]) -> (usize, String) {
        let elf = ElfFile64::<object::Endianness>::parse(image).unwrap();
        let text = elf.section_by_name(".text").unwrap();
        let address = text.elf_section_header().sh_addr(elf.endian()) as usize;
        let name = elf.symbols().next().unwrap().name().unwrap().to_string();
        (address, name)
    }

    /// The registered images of the compile whose text is in `dir`, by function name: its
    /// directory, a DWARF string, ends in a NUL (`.../7` is not `.../70`).
    fn registered_from(dir: &std::path::Path) -> Vec<(String, Vec<u8>)> {
        let dir = format!("{}\0", dir.display());
        let mut images: Vec<_> = (registered().into_iter())
            .filter(|image| image.windows(dir.len()).any(|w| w == dir.as_bytes()))
            .map(|image| (text_of(&image).1, image))
            .collect();
        images.sort();
        images
    }

    type Read<'a> = gimli::Dwarf<EndianSlice<'a, NativeEndian>>;

    /// `address`'s file and line, from the first line table in the ELF `image` that has it.
    fn line_at(image: &[u8], address: u64) -> Option<(String, u64)> {
        let elf = ElfFile64::<object::Endianness>::parse(image).unwrap();
        let dwarf = gimli::Dwarf::load(|id| {
            let data = elf
                .section_by_name(id.name())
                .map_or(&[][..], |s| s.data().unwrap());
            Ok::<_, gimli::Error>(EndianSlice::new(data, NativeEndian))
        })
        .unwrap();
        lines_at(&dwarf, address).into_iter().next()
    }

    /// `address`'s file and line in each unit's line table that has it, in unit order.
    fn lines_at(dwarf: &Read, address: u64) -> Vec<(String, u64)> {
        let mut lines = Vec::new();
        let mut units = dwarf.units();
        while let Some(header) = units.next().unwrap() {
            let unit = dwarf.unit(header).unwrap();
            let program = unit.line_program.clone().unwrap();
            let mut rows = program.rows();
            // The last row at or before `address`, in a sequence that goes past it.
            let mut found = None;
            while let Some((header, row)) = rows.next_row().unwrap() {
                if row.end_sequence() {
                    if row.address() > address {
                        lines.extend(found.take());
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
        lines
    }

    /// The first unit's inlined frames at `address`, outermost first: each one's function,
    /// and the file and line it is called from.
    fn inlined_at(dwarf: &Read, address: u64) -> Vec<(String, String, u64)> {
        use gimli::AttributeValue;
        let header = dwarf.units().next().unwrap().unwrap();
        let unit = dwarf.unit(header).unwrap();
        let program = unit.line_program.clone().unwrap();
        let string = |value| {
            dwarf
                .attr_string(&unit, value)
                .unwrap()
                .to_string_lossy()
                .into_owned()
        };
        let mut frames = Vec::new();
        let mut entries = unit.entries();
        while let Some(entry) = entries.next_dfs().unwrap() {
            if entry.tag() != gimli::DW_TAG_inlined_subroutine {
                continue;
            }
            let (Some(AttributeValue::Addr(low)), Some(AttributeValue::Udata(len))) = (
                entry.attr_value(gimli::DW_AT_low_pc),
                entry.attr_value(gimli::DW_AT_high_pc),
            ) else {
                panic!("an inlined frame without its code");
            };
            if !(low..low + len).contains(&address) {
                continue;
            }
            let Some(AttributeValue::UnitRef(origin)) =
                entry.attr_value(gimli::DW_AT_abstract_origin)
            else {
                panic!("an inlined frame without its function");
            };
            let name = unit
                .entry(origin)
                .unwrap()
                .attr_value(gimli::DW_AT_name)
                .unwrap();
            let Some(AttributeValue::FileIndex(file)) = entry.attr_value(gimli::DW_AT_call_file)
            else {
                panic!("an inlined frame without its call site");
            };
            let file = program.header().file(file).unwrap().path_name();
            let Some(AttributeValue::Udata(line)) = entry.attr_value(gimli::DW_AT_call_line) else {
                panic!("an inlined frame without its call line");
            };
            frames.push((string(name), string(file), line));
        }
        frames
    }

    #[test]
    fn rust_frames_are_inlined_into_the_ir_line_and_the_ir_has_its_own_unit() {
        use crate::cranelift::source::Frame;
        use crate::func::CodeMap;
        // Id 1 asked for by `inner` (inner.rs:10), called from `outer` (outer.rs:20); id 2
        // with no frames; the first 4 bytes asked for by nothing.
        let function = CodeMap {
            name: "k".into(),
            address: 0x1000,
            size: 16,
            ranges: vec![(0..4, None), (4..8, Some(1)), (8..16, Some(2))],
        };
        let frame = |name: &str, file: &str, line| Frame {
            name: name.into(),
            file: file.into(),
            line,
        };
        let chains = std::collections::HashMap::from([(
            1,
            vec![
                frame("inner", "/src/inner.rs", 10),
                frame("outer", "/src/outer.rs", 20),
            ],
        )]);
        let sections = super::build(
            std::path::Path::new("/tmp/k"),
            &function,
            &[(1, 3), (2, 4)],
            &chains,
        );
        let dwarf = gimli::Dwarf::load(|id| {
            let data = (sections.iter()).find(|(section, _)| *section == id);
            Ok::<_, gimli::Error>(EndianSlice::new(
                data.map_or(&[][..], |(_, d)| d),
                NativeEndian,
            ))
        })
        .unwrap();
        let at = |file: &str, line| (file.to_string(), line);
        // Unit 1 at the innermost Rust line, unit 2 at the `.ir` line.
        assert_eq!(
            lines_at(&dwarf, 0x1004),
            [at("/src/inner.rs", 10), at("k.ir", 3)]
        );
        assert_eq!(
            inlined_at(&dwarf, 0x1004),
            [
                ("outer".into(), "k.ir".into(), 3),
                ("inner".into(), "/src/outer.rs".into(), 20)
            ]
        );
        // No frames: at its `.ir` line in both, nothing inlined; no id: the header.
        assert_eq!(lines_at(&dwarf, 0x1008), [at("k.ir", 4), at("k.ir", 4)]);
        assert!(inlined_at(&dwarf, 0x1008).is_empty());
        assert_eq!(lines_at(&dwarf, 0x1000), [at("k.ir", 1), at("k.ir", 1)]);
    }

    fn count_to_n() -> Compiled<crate::func::FunType1<i64, i64>> {
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
        compiler.compile(count_to_n).unwrap()
    }

    #[test]
    fn an_instructions_code_is_at_its_line_in_its_file() {
        let compiled = count_to_n();
        let (_, text) = (compiled.sources().iter())
            .find(|(name, _)| name == "count_to_n")
            .unwrap();
        assert_eq!(text.lines().nth(11), Some("    v9 = v8 + v5  // @7"));
        let code = (compiled.code_ranges().iter())
            .find(|code| code.name == "count_to_n")
            .unwrap();
        // What a debugger is handed: an image of the function, where it is in memory.
        let images = registered_from(compiled.source_dir().unwrap());
        let (_, image) = (images.iter())
            .find(|(name, _)| name == "count_to_n")
            .unwrap();
        assert_eq!(text_of(image), (code.address, "count_to_n".into()));
        let (range, _) = (code.ranges.iter()).find(|(_, id)| *id == Some(7)).unwrap();
        // Its first byte and its last: the whole range is at line 12.
        for offset in [range.start, range.end - 1] {
            let at = (code.address + offset as usize) as u64;
            assert_eq!(line_at(image, at), Some(("count_to_n.ir".into(), 12)));
        }
        // Code no instruction asked for (the driver's `return`): at the header, line 1.
        let (range, _) = (code.ranges.iter())
            .find(|(_, id)| id.is_none())
            .expect("the driver's return");
        let at = (code.address + range.start as usize) as u64;
        assert_eq!(line_at(image, at), Some(("count_to_n.ir".into(), 1)));
    }

    #[test]
    fn the_debugger_forgets_the_code_when_it_is_dropped() {
        let compiled = count_to_n();
        let dir = compiled.source_dir().unwrap().to_path_buf();
        // `__main__` and `count_to_n`, each its own image.
        let names: Vec<_> = (registered_from(&dir).into_iter())
            .map(|(name, _)| name)
            .collect();
        assert_eq!(names, ["__main__", "count_to_n"]);
        drop(compiled);
        assert!(registered_from(&dir).is_empty());
    }
}
