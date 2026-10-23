//! Dev-only allocation traffic counters, inserted at Blot's allocator boundary.
//! Reused blocks count too. Normal modules are never rewritten.
use std::convert::Infallible;

use anyhow::{Result, ensure};
use wasm_encoder::reencode::{self, Reencode};
use wasm_encoder::{ConstExpr, ExportKind, GlobalType, Instruction as I, ValType};
use wasmparser::{ExternalKind, Parser, Payload, TypeRef};

pub const BYTES: &str = "pen:trace_bytes";
pub const ALLOCATIONS: &str = "pen:trace_allocations";

pub fn instrument(bytes: &[u8]) -> Result<Vec<u8>> {
    let mut allocator = None;
    let mut globals = 0;
    let mut imported_functions = 0;
    let mut signatures = Vec::new();
    let mut types = Vec::new();
    for payload in Parser::new(0).parse_all(bytes) {
        match payload? {
            Payload::TypeSection(section) => {
                types.extend(section.into_iter_err_on_gc_types().collect::<Result<Vec<_>, _>>()?)
            }
            Payload::ImportSection(section) => {
                for import in section.into_imports() {
                    match import?.ty {
                        TypeRef::Func(index) => {
                            imported_functions += 1;
                            signatures.push(index);
                        }
                        TypeRef::Global(_) => globals += 1,
                        _ => {}
                    }
                }
            }
            Payload::FunctionSection(section) => signatures.extend(section.into_iter().collect::<Result<Vec<_>, _>>()?),
            Payload::GlobalSection(section) => globals += section.count(),
            Payload::ExportSection(section) => {
                for export in section {
                    let export = export?;
                    ensure!(export.name != BYTES && export.name != ALLOCATIONS, "module is already instrumented");
                    if export.name == "blot:allocation" && export.kind == ExternalKind::Func {
                        allocator = Some(export.index);
                    }
                }
            }
            _ => {}
        }
    }
    let allocator = allocator.ok_or_else(|| {
        anyhow::anyhow!("system tracing needs a compiler exporting blot:allocation; rebuild Blot and the game")
    })?;
    ensure!(allocator >= imported_functions && globals > 0, "invalid profiling allocator");
    let signature = signatures.get(allocator as usize).and_then(|index| types.get(*index as usize));
    ensure!(
        signature
            .is_some_and(|ty| ty.params() == [wasmparser::ValType::I32] && ty.results() == [wasmparser::ValType::I32]),
        "profiling allocator must take and return i32"
    );
    let mut rewriter = Counters { allocator, globals, function: imported_functions };
    let mut module = wasm_encoder::Module::new();
    rewriter.parse_core_module(&mut module, Parser::new(0), bytes)?;
    Ok(module.finish())
}

struct Counters {
    allocator: u32,
    globals: u32,
    function: u32,
}
impl Reencode for Counters {
    type Error = Infallible;

    fn parse_global_section(
        &mut self,
        globals: &mut wasm_encoder::GlobalSection,
        section: wasmparser::GlobalSectionReader<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        reencode::utils::parse_global_section(self, globals, section)?;
        for _ in 0..2 {
            globals
                .global(GlobalType { val_type: ValType::I64, mutable: true, shared: false }, &ConstExpr::i64_const(0));
        }
        Ok(())
    }

    fn parse_export_section(
        &mut self,
        exports: &mut wasm_encoder::ExportSection,
        section: wasmparser::ExportSectionReader<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        reencode::utils::parse_export_section(self, exports, section)?;
        exports.export(BYTES, ExportKind::Global, self.globals);
        exports.export(ALLOCATIONS, ExportKind::Global, self.globals + 1);
        Ok(())
    }

    fn parse_function_body(
        &mut self,
        code: &mut wasm_encoder::CodeSection,
        body: wasmparser::FunctionBody<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        let mut function = self.new_function_with_parsed_locals(&body)?;
        if self.function == self.allocator {
            for instruction in [
                I::LocalGet(0),
                I::If(wasm_encoder::BlockType::Empty),
                I::GlobalGet(self.globals),
                I::LocalGet(0),
                I::I64ExtendI32U,
                I::I64Add,
                I::GlobalSet(self.globals),
                I::GlobalGet(self.globals + 1),
                I::I64Const(1),
                I::I64Add,
                I::GlobalSet(self.globals + 1),
                I::End,
            ] {
                function.instruction(&instruction);
            }
        }
        let mut reader = body.get_operators_reader()?;
        while !reader.eof() {
            function.instruction(&self.parse_instruction(&mut reader)?);
        }
        code.function(&function);
        self.function += 1;
        Ok(())
    }

    fn parse_custom_section(
        &mut self,
        module: &mut wasm_encoder::Module,
        section: wasmparser::CustomSectionReader<'_>,
    ) -> Result<(), reencode::Error<Self::Error>> {
        // Rewriting moves instruction offsets. Never publish stale source positions.
        if section.name() != "sourceMappingURL" {
            reencode::utils::parse_custom_section(self, module, section)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wasm_encoder::*;

    fn allocator() -> Vec<u8> {
        let mut module = Module::new();
        let mut types = TypeSection::new();
        types.ty().function([ValType::I32], [ValType::I32]);
        module.section(&types);
        let mut functions = FunctionSection::new();
        functions.function(0);
        module.section(&functions);
        let mut globals = GlobalSection::new();
        globals.global(GlobalType { val_type: ValType::I32, mutable: true, shared: false }, &ConstExpr::i32_const(0));
        module.section(&globals);
        let mut exports = ExportSection::new();
        exports.export("blot:allocation", ExportKind::Func, 0);
        module.section(&exports);
        // Always returns the same block, without growing memory.
        let mut function = Function::new([]);
        function.instruction(&Instruction::I32Const(65536)).instruction(&Instruction::End);
        let mut code = CodeSection::new();
        code.function(&function);
        module.section(&code);
        module.finish()
    }

    #[test]
    fn counts_reused_blocks_and_more_than_four_gib_without_counting_zero_size_queries() {
        let original = allocator();
        let instrumented = instrument(&original).unwrap();
        let engine = crate::engine().unwrap();
        let module = wasmtime::Module::new(&engine, &instrumented).unwrap();
        let mut store = wasmtime::Store::new(&engine, ());
        let instance = wasmtime::Instance::new(&mut store, &module, &[]).unwrap();
        let allocate = instance.get_typed_func::<i32, i32>(&mut store, "blot:allocation").unwrap();
        for size in [32, 32, 0, -1, -1] {
            assert_eq!(allocate.call(&mut store, size).unwrap(), 65536);
        }
        let bytes = instance.get_global(&mut store, BYTES).unwrap().get(&mut store).i64().unwrap();
        let count = instance.get_global(&mut store, ALLOCATIONS).unwrap().get(&mut store).i64().unwrap();
        assert_eq!(bytes, 64 + 2 * i64::from(u32::MAX));
        assert_eq!(count, 4);
        assert!(instrument(&instrumented).is_err());
        let ordinary = wasmtime::Module::new(&engine, &original).unwrap();
        assert!(ordinary.get_export(BYTES).is_none(), "normal modules have no counter overhead");
    }

    #[test]
    fn old_compilers_fail_explicitly_instead_of_reporting_heap_growth_as_allocation() {
        let module = Module::new().finish();
        assert!(instrument(&module).unwrap_err().to_string().contains("blot:allocation"));
    }
}
