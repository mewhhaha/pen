//! Gives a module the guest ABI's shape (blot2's `compiler/guest-abi.md`)
//! when its language cannot name `externref`, as Rust cannot: what a game's
//! `main` takes is the host as an opaque reference.
//!
//! The module in (as `guest.rs` builds it) exports `pen:main: () -> ()`,
//! the program, and its memory, `blot:allocate` and `blot:reset`, and
//! imports `pen:guest/1` `exchange: (i32) -> i32`, which takes an array's
//! address and returns the host's array's. The module out has instead:
//!
//! - the import `blot:host/1` `call_array_f32_array_f32: (externref, i32)
//!   -> i32`, first;
//! - a mutable `externref` global holding the host while `main` runs;
//! - `exchange` defined, as a call of the import with that global;
//! - `main: (externref) -> i32`, which keeps the host, runs `pen:main` and
//!   returns Unit (0);
//! - exactly the exports a Blot game has (`main`, `blot:memory`,
//!   `blot:allocate`, `blot:reset`), and its `blot:abi` manifest (ABI 2).
//!
//! Every other function keeps its index: the import section gains one
//! function and loses one, so only imports before `exchange` move, and
//! calls of `exchange` go to its definition. The name section follows.
use anyhow::{Context, Result, bail, ensure};
use wasm_encoder::reencode::{Error, Reencode, utils};
use wasm_encoder::{
    CodeSection, ConstExpr, CustomSection, EntityType, ExportKind, ExportSection, Function, FunctionSection,
    GlobalSection, GlobalType, HeapType, ImportSection, Module, NameMap, NameSection, TypeSection, ValType,
};
use wasmparser::{ExternalKind, Parser, Payload, TypeRef};

/// The import a module exchanges arrays with the host through.
pub const EXCHANGE: (&str, &str) = ("pen:guest/1", "exchange");
/// The export that runs the program.
pub const PROGRAM: &str = "pen:main";
/// The manifest: ABI 2, one function, `main: (Array F32 -> Array F32 !
/// {Foreign}) -> Unit`, no constants.
pub const MANIFEST: [u8; 12] = [0x02, 0x01, 0x04, b'm', b'a', b'i', b'n', 0x04, 0x06, 0x06, 0x00, 0x00];

struct Adapter {
    /// Where `exchange` is among the function imports.
    exchange: u32,
    /// Where the new types, the host global and the new functions go.
    types: u32,
    host: u32,
    shim: u32,
    main: u32,
    program: u32,
}

impl Reencode for Adapter {
    type Error = std::convert::Infallible;

    fn function_index(&mut self, function: u32) -> Result<u32, Error<Self::Error>> {
        Ok(match function {
            function if function == self.exchange => self.shim,
            function if function < self.exchange => function + 1,
            function => function,
        })
    }

    fn parse_type_section(
        &mut self,
        types: &mut TypeSection,
        section: wasmparser::TypeSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        utils::parse_type_section(self, types, section)?;
        // The import, main and the shim.
        types.ty().function([ValType::EXTERNREF, ValType::I32], [ValType::I32]);
        types.ty().function([ValType::EXTERNREF], [ValType::I32]);
        types.ty().function([ValType::I32], [ValType::I32]);
        Ok(())
    }

    fn parse_import_section(
        &mut self,
        imports: &mut ImportSection,
        section: wasmparser::ImportSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        imports.import("blot:host/1", "call_array_f32_array_f32", EntityType::Function(self.types));
        for import in section.into_imports() {
            let import = import?;
            if (import.module, import.name) != EXCHANGE {
                self.parse_import(imports, import)?;
            }
        }
        Ok(())
    }

    fn parse_function_section(
        &mut self,
        functions: &mut FunctionSection,
        section: wasmparser::FunctionSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        utils::parse_function_section(self, functions, section)?;
        functions.function(self.types + 2);
        functions.function(self.types + 1);
        Ok(())
    }

    fn parse_global_section(
        &mut self,
        globals: &mut GlobalSection,
        section: wasmparser::GlobalSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        utils::parse_global_section(self, globals, section)?;
        let host = GlobalType { val_type: ValType::EXTERNREF, mutable: true, shared: false };
        globals.global(host, &ConstExpr::ref_null(HeapType::EXTERN));
        Ok(())
    }

    fn parse_export_section(
        &mut self,
        exports: &mut ExportSection,
        section: wasmparser::ExportSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        exports.export("main", ExportKind::Func, self.main);
        for export in section {
            let export = export?;
            match (export.kind, export.name) {
                (ExternalKind::Memory, _) => {
                    exports.export("blot:memory", ExportKind::Memory, export.index);
                }
                (ExternalKind::Func, name @ ("blot:allocate" | "blot:reset")) => {
                    exports.export(name, ExportKind::Func, self.function_index(export.index)?);
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_code_section(
        &mut self,
        code: &mut CodeSection,
        section: wasmparser::CodeSectionReader<'_>,
    ) -> Result<(), Error<Self::Error>> {
        utils::parse_code_section(self, code, section)?;
        // exchange(packet) = call_array_f32_array_f32(host, packet)
        let mut shim = Function::new([]);
        shim.instructions().global_get(self.host).local_get(0).call(0).end();
        code.function(&shim);
        // main(host): keep it while the program runs, then let it go.
        let mut main = Function::new([]);
        main.instructions()
            .local_get(0)
            .global_set(self.host)
            .call(self.program)
            .ref_null(HeapType::EXTERN)
            .global_set(self.host)
            .i32_const(0)
            .end();
        code.function(&main);
        Ok(())
    }

    fn custom_name_section(
        &mut self,
        section: wasmparser::NameSectionReader<'_>,
    ) -> Result<NameSection, Error<Self::Error>> {
        let mut names = NameSection::new();
        for subsection in section {
            match subsection? {
                wasmparser::Name::Function(map) => {
                    // Renumbered, `exchange` moves past the rest: sort again.
                    let mut functions = vec![(self.shim, "pen:exchange".to_owned()), (self.main, "main".to_owned())];
                    for naming in map {
                        let naming = naming?;
                        if naming.index != self.exchange {
                            functions.push((self.function_index(naming.index)?, naming.name.to_owned()));
                        }
                    }
                    functions.sort_by_key(|(index, _)| *index);
                    let mut map = NameMap::new();
                    for (index, name) in &functions {
                        map.append(*index, name);
                    }
                    names.functions(&map);
                }
                subsection => utils::parse_custom_name_subsection(self, &mut names, subsection)?,
            }
        }
        Ok(names)
    }
}

/// The module `raw`, with the guest ABI's shape.
pub fn adapt(raw: &[u8]) -> Result<Vec<u8>> {
    let mut imports = 0;
    let mut exchange = None;
    let (mut types, mut globals, mut defined) = (0, 0, 0);
    let (mut program, mut memory) = (None, false);
    for payload in Parser::new(0).parse_all(raw) {
        match payload? {
            Payload::TypeSection(section) => {
                for group in section {
                    types += u32::try_from(group?.types().len())?;
                }
            }
            Payload::ImportSection(section) => {
                for import in section.into_imports() {
                    let import = import?;
                    match import.ty {
                        TypeRef::Func(_) => {
                            if (import.module, import.name) == EXCHANGE {
                                exchange = Some(imports);
                            }
                            imports += 1;
                        }
                        TypeRef::Global(_) => globals += 1,
                        _ => {}
                    }
                }
            }
            Payload::FunctionSection(section) => defined = section.count(),
            Payload::GlobalSection(section) => globals += section.count(),
            Payload::ExportSection(section) => {
                for export in section {
                    let export = export?;
                    match export.kind {
                        ExternalKind::Func if export.name == PROGRAM => program = Some(export.index),
                        ExternalKind::Memory => memory = true,
                        _ => {}
                    }
                }
            }
            Payload::CustomSection(section) if section.name() == "blot:abi" => {
                bail!("the module already has a blot:abi manifest")
            }
            _ => {}
        }
    }
    let exchange = exchange.with_context(|| format!("the module does not import {}::{}", EXCHANGE.0, EXCHANGE.1))?;
    let program = program.with_context(|| format!("the module does not export {PROGRAM}"))?;
    ensure!(memory, "the module does not export its memory");
    ensure!(globals > 0, "the module has no globals (a Rust module has its stack pointer)");
    let shim = imports + defined;
    let mut adapter = Adapter { exchange, types, host: globals, shim, main: shim + 1, program: 0 };
    adapter.program = adapter.function_index(program).map_err(|error| anyhow::anyhow!("{error}"))?;
    let mut module = Module::new();
    module.section(&CustomSection { name: "blot:abi".into(), data: MANIFEST[..].into() });
    adapter.parse_core_module(&mut module, Parser::new(0), raw).map_err(|error| anyhow::anyhow!("{error}"))?;
    let adapted = module.finish();
    wasmparser::Validator::new().validate_all(&adapted).context("the adapted module does not validate")?;
    Ok(adapted)
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_encoder::{Instruction, MemorySection, MemoryType};

    /// A module like `guest.rs` builds: one import before `exchange`, a
    /// program that calls it, and the arena.
    fn raw() -> Vec<u8> {
        let mut module = Module::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I32]);
        types.ty().function([], []);
        module.section(&types);
        let mut imports = ImportSection::new();
        imports.import("env", "other", EntityType::Function(1));
        imports.import(EXCHANGE.0, EXCHANGE.1, EntityType::Function(0));
        module.section(&imports);
        let mut functions = FunctionSection::new();
        functions.function(1).function(0).function(0);
        module.section(&functions);
        let mut memories = MemorySection::new();
        memories.memory(MemoryType { minimum: 1, maximum: None, memory64: false, shared: false, page_size_log2: None });
        module.section(&memories);
        let mut globals = GlobalSection::new();
        globals.global(GlobalType { val_type: ValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(0));
        module.section(&globals);
        let mut exports = ExportSection::new();
        exports.export("memory", ExportKind::Memory, 0);
        exports.export(PROGRAM, ExportKind::Func, 2);
        exports.export("blot:allocate", ExportKind::Func, 3);
        exports.export("blot:reset", ExportKind::Func, 4);
        exports.export("__data_end", ExportKind::Global, 0);
        module.section(&exports);
        let mut code = CodeSection::new();
        let mut program = Function::new([]);
        program.instruction(&Instruction::Call(0));
        program.instruction(&Instruction::I32Const(16));
        program.instruction(&Instruction::Call(1));
        program.instruction(&Instruction::Drop);
        program.instruction(&Instruction::End);
        code.function(&program);
        for _ in 0..2 {
            let mut arena = Function::new([]);
            arena.instruction(&Instruction::LocalGet(0));
            arena.instruction(&Instruction::End);
            code.function(&arena);
        }
        module.section(&code);
        let mut names = NameSection::new();
        let mut map = NameMap::new();
        map.append(0, "other");
        map.append(1, "exchange");
        map.append(2, "program");
        names.functions(&map);
        module.section(&names);
        module.finish()
    }

    #[test]
    fn modules_take_the_guest_abis_shape() {
        let adapted = adapt(&raw()).unwrap();
        let mut imports = Vec::new();
        let mut exports = Vec::new();
        let mut manifest = None;
        let mut names = Vec::new();
        let mut calls = Vec::new();
        for payload in Parser::new(0).parse_all(&adapted) {
            match payload.unwrap() {
                Payload::ImportSection(section) => {
                    for import in section.into_imports() {
                        let import = import.unwrap();
                        imports.push(format!("{}::{}", import.module, import.name));
                    }
                }
                Payload::ExportSection(section) => {
                    for export in section {
                        let export = export.unwrap();
                        exports.push((export.name.to_owned(), export.index));
                    }
                }
                Payload::CodeSectionEntry(body) => {
                    for operator in body.get_operators_reader().unwrap() {
                        if let wasmparser::Operator::Call { function_index } = operator.unwrap() {
                            calls.push(function_index);
                        }
                    }
                }
                Payload::CustomSection(section) => match section.as_known() {
                    wasmparser::KnownCustom::Name(reader) => {
                        for name in reader {
                            if let wasmparser::Name::Function(map) = name.unwrap() {
                                names.extend(map.into_iter().map(|naming| {
                                    let naming = naming.unwrap();
                                    (naming.index, naming.name.to_owned())
                                }));
                            }
                        }
                    }
                    _ if section.name() == "blot:abi" => manifest = Some(section.data().to_vec()),
                    _ => {}
                },
                _ => {}
            }
        }
        assert_eq!(imports, ["blot:host/1::call_array_f32_array_f32", "env::other"]);
        // Functions: 0 the host, 1 other, 2 program, 3 and 4 the arena, 5
        // exchange's definition, 6 main.
        let exported: Vec<_> = exports.iter().map(|(name, index)| (name.as_str(), *index)).collect();
        assert_eq!(exported, [("main", 6), ("blot:memory", 0), ("blot:allocate", 3), ("blot:reset", 4)]);
        assert_eq!(manifest.as_deref(), Some(&MANIFEST[..]));
        assert_eq!(calls, [1, 5, 0, 2], "program calls other and exchange; exchange the host; main the program");
        let named: Vec<_> = names.iter().map(|(index, name)| (*index, name.as_str())).collect();
        assert_eq!(named, [(1, "other"), (2, "program"), (5, "pen:exchange"), (6, "main")]);
        assert!(adapt(&adapted).is_err(), "a module adapts once");
    }

    #[test]
    fn modules_without_the_exchange_are_refused() {
        let mut module = Module::new();
        module.section(&TypeSection::new());
        let error = adapt(&module.finish()).unwrap_err();
        assert!(format!("{error:#}").contains("pen:guest/1::exchange"), "{error:#}");
    }
}
