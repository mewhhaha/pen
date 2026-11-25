//! Blot's guest ABI manifest (`blot2/docs/host/guest-abi.md`): the
//! `blot:abi` custom section a guest module declares its exports in, and
//! the entry a game must have. Native pen (blot-host) and the browser read
//! it alike, before a module runs. ABI 3's ownership hooks are specified in
//! `blot2/docs/compiler/SHARED_OWNERSHIP.md`.
use std::fmt;

use anyhow::{Result, anyhow, bail, ensure};

/// The custom section the manifest is in.
pub const ABI_SECTION: &str = "blot:abi";
/// The custom section declaring `#kernel` manifests (version 1).
pub const KERNELS_SECTION: &str = "blot:kernels";
/// The custom section declaring `#process` manifests (version 1).
pub const PROCESSES_SECTION: &str = "blot:processes";
/// The custom section declaring first-class tests (version 1).
pub const TESTS_SECTION: &str = "blot:tests";
/// Default logical transport packet ceiling in bytes (2 MiB).
pub const DEFAULT_TRANSPORT_BUDGET: u32 = 2_097_152;
/// `blot:kernels` manifest flag bit 0: kernel has a prepared context (`prepare`).
pub const KERNEL_FLAG_HAS_CONTEXT: u8 = 1 << 0;
/// `blot:kernels` manifest flag bit 1: kernel emits variable-length `[len, words...]` output.
pub const KERNEL_FLAG_VARIABLE_OUTPUT: u8 = 1 << 1;
/// ABI 8 identifies data-stack storage, including scalar-only modules.
/// Earlier profiles remain readable only during the compiler cutover.
pub const ABI_VERSIONS: [u32; 3] = [2, 3, 8];

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum WireSchemaKind {
    Unit = 0,
    FlatFixed = 1,
    VarU32 = 2,
    VarF32 = 3,
}

impl WireSchemaKind {
    fn from_byte(byte: u8) -> Result<Self> {
        Ok(match byte {
            0 => WireSchemaKind::Unit,
            1 => WireSchemaKind::FlatFixed,
            2 => WireSchemaKind::VarU32,
            3 => WireSchemaKind::VarF32,
            _ => bail!("invalid_abi: unknown wire schema kind {byte}"),
        })
    }
}

/// Canonical wire layout of a kernel or process boundary type (`blot:kernels` / `blot:processes` v1).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WireSchema {
    pub kind: WireSchemaKind,
    pub wire_tag: ValueType,
    pub stride: u32,
    pub word_tags: Vec<ValueType>,
    pub schema_hash: u64,
}

/// One `#kernel` descriptor from `blot:kernels` v1 (0-based index == `kernel_id`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelManifest {
    pub id: u32,
    pub name: String,
    pub flags: u8,
    pub in_schema: WireSchema,
    pub out_schema: WireSchema,
    pub io_schema_hash: u64,
    pub impl_hash: u64,
    pub binding_hash: u64,
    pub ctx_schema: Option<WireSchema>,
    pub ctx_export: Option<String>,
    pub clear_export: Option<String>,
}

impl KernelManifest {
    pub fn has_context(&self) -> bool {
        (self.flags & KERNEL_FLAG_HAS_CONTEXT) != 0
    }

    pub fn variable_output(&self) -> bool {
        (self.flags & KERNEL_FLAG_VARIABLE_OUTPUT) != 0
    }
}

/// One `#process` descriptor from `blot:processes` v1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProcessManifest {
    pub name: String,
    pub init_export: String,
    pub step_export: String,
    pub config_schema: WireSchema,
    pub cmd_schema: WireSchema,
    pub reply_schema: WireSchema,
}

/// Decoded Wasm Section ID 5 (`MemorySection`) shape, parsed with zero external crates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MemoryShape {
    /// Whether Wasm Section ID 5 was present in the bytes.
    pub section_present: bool,
    /// Whether the module defines a linear memory (count == 1).
    pub has_memory: bool,
    pub memory64: bool,
    pub shared: bool,
    pub page_size_log2: u32,
    pub initial_pages: u64,
    pub maximum_pages: Option<u64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    Unit,
    U32,
    Bool,
    F32,
    ArrayU32,
    ArrayF32,
}

impl ValueType {
    pub fn is_array(self) -> bool {
        matches!(self, ValueType::ArrayU32 | ValueType::ArrayF32)
    }

    pub fn is_scalar(self) -> bool {
        !self.is_array()
    }

    /// Its name in a callback adapter's import name.
    pub fn import_name(self) -> &'static str {
        match self {
            ValueType::Unit => "unit",
            ValueType::U32 => "u32",
            ValueType::Bool => "bool",
            ValueType::F32 => "f32",
            ValueType::ArrayU32 => "array_u32",
            ValueType::ArrayF32 => "array_f32",
        }
    }

    fn from_tag(tag: u8) -> Result<ValueType> {
        Ok(match tag {
            0 => ValueType::Unit,
            1 => ValueType::U32,
            2 => ValueType::Bool,
            3 => ValueType::F32,
            5 => ValueType::ArrayU32,
            6 => ValueType::ArrayF32,
            _ => bail!("invalid_abi: unknown value type tag {tag}"),
        })
    }
}

impl fmt::Display for ValueType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            ValueType::Unit => "Unit",
            ValueType::U32 => "U32",
            ValueType::Bool => "Bool",
            ValueType::F32 => "F32",
            ValueType::ArrayU32 => "Array U32",
            ValueType::ArrayF32 => "Array F32",
        })
    }
}

/// A host callback's signature: one value in, one value out, `! {Foreign}`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CallbackType {
    pub parameter: ValueType,
    pub result: ValueType,
}

impl CallbackType {
    /// `call_<parameter>_<result>`, the adapter this signature imports.
    pub fn import_name(self) -> String {
        format!("call_{}_{}", self.parameter.import_name(), self.result.import_name())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Parameter {
    Value(ValueType),
    Callback(CallbackType),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestFunction {
    pub name: String,
    pub parameter: Parameter,
    pub result: ValueType,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestConstant {
    pub name: String,
    pub value: ValueType,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TestKind {
    Check,
    Trap,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TestMetadata {
    pub kind: TestKind,
    pub export_name: String,
    pub display_name: String,
    pub filename: String,
    pub line: u32,
}

/// The `blot:abi` manifest: the module's exports, in declaration order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GuestAbi {
    pub functions: Vec<GuestFunction>,
    pub constants: Vec<GuestConstant>,
    /// First-class test declarations, in metadata order.
    pub tests: Vec<TestMetadata>,
    /// ABI 8: named data-stack owners and copied numeric packets, with no
    /// counted ownership hooks, resource adapters or language tasks.
    pub data_stacks: bool,
    /// Optional ABI 8 post-invocation hook for scoped owners and callbacks.
    pub invocation_finish: bool,
    /// ABI 3: the module exports `blot:ownership_enter`, which a host calls
    /// before each call's inputs, and `blot:ownership_leave(outcome)`, which
    /// it calls after copying the result out (outcome 0) or after a failure
    /// (1), before `blot:reset`.
    pub ownership: bool,
    /// Blot's structured tasks (guest-abi.md, "Structured task
    /// extension"): the `blot:tasks` section's descriptor count.
    pub tasks: Option<u32>,
    /// Functions whose callback is `concurrent` (manifest tag 13): tasks
    /// may suspend in it side by side.
    pub concurrent: Vec<String>,
    /// `#kernel` descriptors parsed from `blot:kernels` v1 (index == `kernel_id`).
    pub kernels: Vec<KernelManifest>,
    /// `#process` descriptors parsed from `blot:processes` v1.
    pub processes: Vec<ProcessManifest>,
    /// Declared logical transport ceiling in bytes (`DEFAULT_TRANSPORT_BUDGET` if absent).
    pub transport_budget: u32,
    /// Host-internal wrapper exports (`blot:kernel_ctx:*`, `blot:kernel_clear:*`,
    /// `blot:proc_init:*`, `blot:proc_step:*`).
    pub host_internal: Vec<String>,
}

impl Default for GuestAbi {
    fn default() -> Self {
        Self {
            functions: Vec::new(),
            constants: Vec::new(),
            tests: Vec::new(),
            data_stacks: false,
            invocation_finish: false,
            ownership: false,
            tasks: None,
            concurrent: Vec::new(),
            kernels: Vec::new(),
            processes: Vec::new(),
            transport_budget: DEFAULT_TRANSPORT_BUDGET,
            host_internal: Vec::new(),
        }
    }
}

/// The custom section a task module declares its task descriptors in.
pub const TASKS_SECTION: &str = "blot:tasks";

/// Optional ABI 8 invocation finish extension, version 1.
pub const INVOCATION_SECTION: &str = "blot:invocation";
pub const INVOCATION_FINISH: &str = "blot:finish_invocation";

/// What a game's `main` takes: the host, which answers each frame's
/// request with the next frame's input.
pub const GAME_HOST: CallbackType = CallbackType { parameter: ValueType::ArrayF32, result: ValueType::ArrayF32 };

impl GuestAbi {
    pub fn function(&self, name: &str) -> Option<&GuestFunction> {
        self.functions.iter().find(|function| function.name == name)
    }

    pub fn kernel(&self, name: &str) -> Option<&KernelManifest> {
        self.kernels.iter().find(|kernel| kernel.name == name)
    }

    pub fn process(&self, name: &str) -> Option<&ProcessManifest> {
        self.processes.iter().find(|process| process.name == name)
    }

    /// Whether `name` is a host-internal kernel or process wrapper export.
    pub fn is_host_internal(&self, name: &str) -> bool {
        self.host_internal.iter().any(|entry| entry == name)
    }

    /// Callback signatures, deduplicated in first-export order.
    pub fn callbacks(&self) -> Vec<CallbackType> {
        let mut found = Vec::new();
        for function in &self.functions {
            if let Parameter::Callback(signature) = function.parameter
                && !found.contains(&signature)
            {
                found.push(signature);
            }
        }
        found
    }

    /// Whether arrays cross the boundary, so the module needs its array arena.
    pub fn uses_arrays(&self) -> bool {
        self.functions.iter().any(|function| {
            function.result.is_array()
                || match function.parameter {
                    Parameter::Value(value) => value.is_array(),
                    Parameter::Callback(signature) => signature.parameter.is_array() || signature.result.is_array(),
                }
        })
    }

    /// Checks native admission. Pure task scopes run synchronously; concurrent
    /// callbacks require suspended sibling stacks, which this embedding does
    /// not supply (docs/services-as-tasks.md).
    pub fn check_hostable(&self) -> Result<()> {
        if let Some(descriptors) = self.tasks
            && !self.concurrent.is_empty()
        {
            bail!(
                "unsupported_abi: the module uses Blot's structured tasks ({TASKS_SECTION}, {descriptors} \
                 descriptors; concurrent callbacks for {}), which require suspended sibling stacks; native pen currently hosts synchronous tasks (docs/services-as-tasks.md)",
                self.concurrent.join(", ")
            );
        }
        Ok(())
    }

    /// Checks the module is a game: it exports `main`, which takes the host
    /// ([`GAME_HOST`]) and runs until the host stops it.
    pub fn check_game(&self) -> Result<()> {
        let main = self.function("main").ok_or_else(|| anyhow!("game must export main"))?;
        ensure!(
            main.result == ValueType::Unit && main.parameter == Parameter::Callback(GAME_HOST),
            "game must export main: (Array F32 -> Array F32 ! {{Foreign}}) -> Unit"
        );
        Ok(())
    }
}

pub(crate) struct Reader<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) at: usize,
}

impl Reader<'_> {
    pub(crate) fn byte(&mut self) -> Result<u8> {
        let byte = *self.bytes.get(self.at).ok_or_else(|| anyhow!("invalid_abi: truncated manifest"))?;
        self.at += 1;
        Ok(byte)
    }

    /// A canonical unsigned LEB128 u32.
    fn u32(&mut self) -> Result<u32> {
        self.leb(true)
    }

    /// An unsigned LEB128 u32, canonical or (as Wasm's own framing may be)
    /// padded.
    pub(crate) fn leb(&mut self, canonical: bool) -> Result<u32> {
        let mut value = 0u64;
        for index in 0..5 {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << (index * 7);
            if byte & 0x80 == 0 {
                ensure!(!canonical || index == 0 || byte != 0, "invalid_abi: non-canonical LEB128");
                return u32::try_from(value).map_err(|_| anyhow!("invalid_abi: LEB128 exceeds u32"));
            }
        }
        bail!("invalid_abi: LEB128 too long")
    }

    /// An unsigned LEB128 u64, canonical or padded (for Wasm MemorySection limits).
    fn leb_u64(&mut self, canonical: bool) -> Result<u64> {
        let mut value = 0u64;
        for index in 0..10 {
            let byte = self.byte()?;
            let shift = index * 7;
            if shift == 63 && (byte & 0x7e) != 0 {
                bail!("invalid_abi: LEB128 exceeds u64");
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                ensure!(!canonical || index == 0 || byte != 0, "invalid_abi: non-canonical LEB128");
                return Ok(value);
            }
        }
        bail!("invalid_abi: LEB128 too long")
    }

    fn u64_le(&mut self) -> Result<u64> {
        let end = self.at.checked_add(8).filter(|end| *end <= self.bytes.len());
        let end = end.ok_or_else(|| anyhow!("invalid_abi: truncated u64"))?;
        let raw: [u8; 8] = self.bytes[self.at..end].try_into().unwrap();
        self.at = end;
        Ok(u64::from_le_bytes(raw))
    }

    fn name(&mut self) -> Result<String> {
        let length = self.u32()? as usize;
        self.text(length)
    }

    /// `length` bytes of UTF-8.
    pub(crate) fn text(&mut self, length: usize) -> Result<String> {
        let end = self.at.checked_add(length).filter(|end| *end <= self.bytes.len());
        let end = end.ok_or_else(|| anyhow!("invalid_abi: truncated name"))?;
        let name =
            std::str::from_utf8(&self.bytes[self.at..end]).map_err(|_| anyhow!("invalid_abi: name is not UTF-8"))?;
        self.at = end;
        Ok(name.to_owned())
    }
}

/// Parses a `blot:abi` custom section's payload.
pub fn parse_abi(payload: &[u8]) -> Result<GuestAbi> {
    let mut reader = Reader { bytes: payload, at: 0 };
    let version = reader.u32()?;
    ensure!(ABI_VERSIONS.contains(&version), "invalid_abi: unsupported ABI version {version}");
    let mut abi = GuestAbi { ownership: version == 3, data_stacks: version == 8, ..GuestAbi::default() };
    let mut names = std::collections::HashSet::new();
    for _ in 0..reader.u32()? {
        let name = reader.name()?;
        let tag = reader.byte()?;
        ensure!(version != 8 || tag != 13, "invalid_abi: data-stack ABI does not support language tasks");
        // 13: a concurrent callback, valid only beside a task section.
        if tag == 13 {
            abi.concurrent.push(name.clone());
        }
        let parameter = if tag == 4 || tag == 13 {
            Parameter::Callback(CallbackType {
                parameter: ValueType::from_tag(reader.byte()?)?,
                result: ValueType::from_tag(reader.byte()?)?,
            })
        } else {
            Parameter::Value(ValueType::from_tag(tag)?)
        };
        let result = ValueType::from_tag(reader.byte()?)?;
        ensure!(names.insert(name.clone()), "invalid_abi: duplicate export {name}");
        abi.functions.push(GuestFunction { name, parameter, result });
    }
    for _ in 0..reader.u32()? {
        let name = reader.name()?;
        let value = ValueType::from_tag(reader.byte()?)?;
        ensure!(value.is_scalar(), "invalid_abi: constant {name} is not a scalar");
        ensure!(names.insert(name.clone()), "invalid_abi: duplicate export {name}");
        abi.constants.push(GuestConstant { name, value });
    }
    ensure!(reader.at == payload.len(), "invalid_abi: trailing manifest bytes");
    Ok(abi)
}

/// A module's sections, in order: each one's id and contents.
pub fn sections(wasm: &[u8]) -> Result<Vec<(u8, &[u8])>> {
    ensure!(wasm.len() >= 8 && wasm[..4] == *b"\0asm", "not a Wasm module");
    let mut reader = Reader { bytes: wasm, at: 8 };
    let mut sections = Vec::new();
    while reader.at < wasm.len() {
        let id = reader.byte()?;
        let size = reader.leb(false)? as usize;
        let end = reader.at.checked_add(size).filter(|end| *end <= wasm.len());
        let end = end.ok_or_else(|| anyhow!("truncated Wasm section"))?;
        sections.push((id, &wasm[reader.at..end]));
        reader.at = end;
    }
    Ok(sections)
}

/// A module's custom sections, in order: each one's name and payload.
pub fn custom_sections(wasm: &[u8]) -> Result<Vec<(String, &[u8])>> {
    let mut custom = Vec::new();
    for (id, contents) in sections(wasm)? {
        if id == 0 {
            let mut section = Reader { bytes: contents, at: 0 };
            let length = section.leb(false)? as usize;
            let name = section.text(length)?;
            custom.push((name, &contents[section.at..]));
        }
    }
    Ok(custom)
}

/// Decodes Wasm Section ID 5 (`MemorySection`) using zero external crates.
/// Supports standard 32-bit, shared (`0x02`), memory64 (`0x04`), and custom
/// page-size (`0x08`) flags so hosts can validate memory shape even when built
/// without `wasmparser`.
pub fn read_memory_shape(wasm: &[u8]) -> Result<MemoryShape> {
    let mut found: Option<MemoryShape> = None;
    for (id, contents) in sections(wasm)? {
        if id != 5 {
            continue;
        }
        ensure!(found.is_none(), "invalid_abi: more than one Wasm memory section");
        let mut reader = Reader { bytes: contents, at: 0 };
        let count = reader.leb(false)?;
        ensure!(count <= 1, "invalid_abi: module declares {count} linear memories (at most 1)");
        if count == 0 {
            ensure!(reader.at == contents.len(), "invalid_abi: trailing memory section bytes");
            found = Some(MemoryShape {
                section_present: true,
                has_memory: false,
                memory64: false,
                shared: false,
                page_size_log2: 16,
                initial_pages: 0,
                maximum_pages: None,
            });
            continue;
        }
        let flags = reader.byte()?;
        ensure!(flags & !0x0f == 0, "invalid_abi: unknown Wasm memory flags {flags:#x}");
        let has_max = (flags & 0x01) != 0;
        let shared = (flags & 0x02) != 0;
        let memory64 = (flags & 0x04) != 0;
        let custom_page_size = (flags & 0x08) != 0;
        let initial_pages = if memory64 { reader.leb_u64(false)? } else { u64::from(reader.leb(false)?) };
        let maximum_pages = if has_max {
            Some(if memory64 { reader.leb_u64(false)? } else { u64::from(reader.leb(false)?) })
        } else {
            None
        };
        if let Some(max_pages) = maximum_pages {
            ensure!(
                max_pages >= initial_pages,
                "invalid_abi: memory maximum pages ({max_pages}) < initial pages ({initial_pages})"
            );
        }
        let page_size_log2 = if custom_page_size { reader.leb(false)? } else { 16 };
        ensure!(reader.at == contents.len(), "invalid_abi: trailing memory section bytes");
        found = Some(MemoryShape {
            section_present: true,
            has_memory: true,
            memory64,
            shared,
            page_size_log2,
            initial_pages,
            maximum_pages,
        });
    }
    Ok(found.unwrap_or(MemoryShape {
        section_present: false,
        has_memory: false,
        memory64: false,
        shared: false,
        page_size_log2: 16,
        initial_pages: 0,
        maximum_pages: None,
    }))
}

/// Finds the single `blot:abi` custom section in a module's bytes, plus
/// optional tasks, invocation, kernel, process, and test sections.
pub fn read_abi(wasm: &[u8]) -> Result<GuestAbi> {
    let mut found = None;
    let mut tasks = None;
    let mut counted_section = false;
    let mut invocation_finish = false;
    let mut kernels_payload: Option<&[u8]> = None;
    let mut processes_payload: Option<&[u8]> = None;
    let mut tests_payload: Option<&[u8]> = None;
    for (name, payload) in custom_sections(wasm)? {
        if matches!(name.as_str(), "blot:ownership" | "blot:cycles" | "blot:memory-profile" | "blot:cleanup_data") {
            counted_section = true;
        }
        if name == ABI_SECTION {
            ensure!(found.is_none(), "invalid_abi: more than one {ABI_SECTION} section");
            found = Some(parse_abi(payload)?);
        } else if name == TASKS_SECTION {
            ensure!(tasks.is_none(), "invalid_abi: more than one {TASKS_SECTION} section");
            tasks = Some(parse_tasks(payload)?);
        } else if name == INVOCATION_SECTION {
            ensure!(!invocation_finish && payload == [1], "invalid_abi: invalid {INVOCATION_SECTION} section");
            invocation_finish = true;
        } else if name == KERNELS_SECTION {
            ensure!(kernels_payload.is_none(), "invalid_abi: more than one {KERNELS_SECTION} section");
            kernels_payload = Some(payload);
        } else if name == PROCESSES_SECTION {
            ensure!(processes_payload.is_none(), "invalid_abi: more than one {PROCESSES_SECTION} section");
            processes_payload = Some(payload);
        } else if name == TESTS_SECTION {
            ensure!(tests_payload.is_none(), "invalid_abi: more than one {TESTS_SECTION} section");
            tests_payload = Some(payload);
        }
    }
    let mut abi = found.ok_or_else(|| anyhow!("invalid_abi: missing {ABI_SECTION} section"))?;
    ensure!(
        !abi.data_stacks || !counted_section,
        "invalid_abi: data-stack ABI cannot declare counted runtime sections"
    );
    ensure!(tasks.is_none() || abi.ownership, "invalid_abi: tasks need ABI 3's ownership");
    ensure!(tasks.is_some() || abi.concurrent.is_empty(), "invalid_abi: concurrent callbacks need {TASKS_SECTION}");
    ensure!(!invocation_finish || abi.data_stacks, "invalid_abi: invocation finish requires ABI 8");
    ensure!(
        kernels_payload.is_none() || abi.data_stacks,
        "invalid_abi: {KERNELS_SECTION} requires ABI 8"
    );
    ensure!(
        processes_payload.is_none() || abi.data_stacks,
        "invalid_abi: {PROCESSES_SECTION} requires ABI 8"
    );
    abi.tasks = tasks;
    abi.invocation_finish = invocation_finish;
    if let Some(payload) = tests_payload {
        abi.tests = parse_tests(payload)?;
    }
    let mut declared_budget: Option<u32> = None;
    if let Some(payload) = kernels_payload {
        let (budget, kernels) = parse_kernels(payload, &abi)?;
        declared_budget = Some(budget);
        for kernel in &kernels {
            if let Some(ctx_export) = &kernel.ctx_export {
                abi.host_internal.push(ctx_export.clone());
            }
            if let Some(clear_export) = &kernel.clear_export {
                abi.host_internal.push(clear_export.clone());
            }
        }
        abi.kernels = kernels;
    }
    if let Some(payload) = processes_payload {
        let (budget, processes) = parse_processes(payload, &abi)?;
        if let Some(existing) = declared_budget {
            ensure!(
                existing == budget,
                "invalid_abi: {KERNELS_SECTION} transport_budget ({existing}) != {PROCESSES_SECTION} transport_budget ({budget})"
            );
        }
        declared_budget = Some(budget);
        for process in &processes {
            abi.host_internal.push(process.init_export.clone());
            abi.host_internal.push(process.step_export.clone());
        }
        abi.processes = processes;
    }
    if let Some(budget) = declared_budget {
        abi.transport_budget = budget;
    }
    Ok(abi)
}

fn parse_tests(payload: &[u8]) -> Result<Vec<TestMetadata>> {
    fn name(reader: &mut Reader<'_>) -> Result<String> {
        let length = reader.leb(false)? as usize;
        reader.text(length)
    }

    let mut reader = Reader { bytes: payload, at: 0 };
    let version = reader.byte()?;
    ensure!(version == 1, "invalid_abi: unsupported {TESTS_SECTION} version {version}");
    let count = reader.leb(false)? as usize;
    ensure!(count <= (payload.len() - reader.at) / 5, "invalid_abi: truncated {TESTS_SECTION} entries");
    let mut tests = Vec::with_capacity(count);
    for _ in 0..count {
        let kind = match reader.byte()? {
            0 => TestKind::Check,
            1 => TestKind::Trap,
            kind => bail!("invalid_abi: unknown {TESTS_SECTION} kind {kind}"),
        };
        tests.push(TestMetadata {
            kind,
            export_name: name(&mut reader)?,
            display_name: name(&mut reader)?,
            filename: name(&mut reader)?,
            line: reader.leb(false)?,
        });
    }
    ensure!(reader.at == payload.len(), "invalid_abi: trailing {TESTS_SECTION} bytes");
    Ok(tests)
}

fn parse_wire_schema(reader: &mut Reader<'_>, transport_budget: u32) -> Result<WireSchema> {
    let kind = WireSchemaKind::from_byte(reader.byte()?)?;
    let wire_tag = ValueType::from_tag(reader.byte()?)?;
    let stride = reader.u32()?;
    let mut word_tags = Vec::new();
    match kind {
        WireSchemaKind::Unit => {
            ensure!(wire_tag == ValueType::Unit && stride == 0, "invalid_abi: malformed Unit wire schema");
        }
        WireSchemaKind::FlatFixed => {
            ensure!(stride >= 1, "invalid_abi: FlatFixed wire schema requires stride >= 1");
            let max_bytes = u64::from(stride).checked_mul(4).and_then(|bytes| bytes.checked_add(4));
            ensure!(
                max_bytes.is_some_and(|bytes| bytes <= u64::from(transport_budget)),
                "invalid_abi: FlatFixed stride {stride} exceeds transport_budget {transport_budget}"
            );
            word_tags.reserve(stride as usize);
            for _ in 0..stride {
                let tag = ValueType::from_tag(reader.byte()?)?;
                ensure!(
                    matches!(tag, ValueType::U32 | ValueType::Bool | ValueType::F32),
                    "invalid_abi: FlatFixed word_tag must be U32, Bool, or F32, got {tag}"
                );
                word_tags.push(tag);
            }
            if stride == 1 {
                ensure!(
                    wire_tag == word_tags[0] || wire_tag.is_array(),
                    "invalid_abi: 1-word FlatFixed wire_tag {wire_tag} incompatible with word tag {}",
                    word_tags[0]
                );
            } else {
                ensure!(wire_tag.is_array(), "invalid_abi: multi-word FlatFixed wire_tag must be an array");
            }
        }
        WireSchemaKind::VarU32 => {
            ensure!(stride == 0 && wire_tag == ValueType::ArrayU32, "invalid_abi: malformed VarU32 wire schema");
        }
        WireSchemaKind::VarF32 => {
            ensure!(
                stride == 0 && matches!(wire_tag, ValueType::ArrayU32 | ValueType::ArrayF32),
                "invalid_abi: malformed VarF32 wire schema"
            );
        }
    }
    let schema_hash = reader.u64_le()?;
    Ok(WireSchema { kind, wire_tag, stride, word_tags, schema_hash })
}

fn parse_kernels(payload: &[u8], abi: &GuestAbi) -> Result<(u32, Vec<KernelManifest>)> {
    let mut reader = Reader { bytes: payload, at: 0 };
    let version = reader.u32()?;
    ensure!(version == 1, "invalid_abi: unsupported {KERNELS_SECTION} version {version}");
    let transport_budget = reader.u32()?;
    ensure!(
        transport_budget >= 8 && transport_budget.is_multiple_of(4),
        "invalid_abi: invalid {KERNELS_SECTION} transport_budget {transport_budget}"
    );
    let count = reader.u32()?;
    let mut kernels = Vec::with_capacity(usize::try_from(count).unwrap_or(0).min(1024));
    let mut names = std::collections::HashSet::new();
    for id in 0..count {
        let name = reader.name()?;
        ensure!(names.insert(name.clone()), "invalid_abi: duplicate kernel {name}");
        let flags = reader.byte()?;
        ensure!(flags & !0x03 == 0, "invalid_abi: kernel {name} has unknown flags {flags:#x}");
        let has_context = (flags & KERNEL_FLAG_HAS_CONTEXT) != 0;
        let variable_output = (flags & KERNEL_FLAG_VARIABLE_OUTPUT) != 0;
        let in_schema = parse_wire_schema(&mut reader, transport_budget)?;
        ensure!(
            in_schema.kind == WireSchemaKind::FlatFixed && in_schema.wire_tag.is_array(),
            "invalid_abi: kernel {name} input must be FlatFixed array wire"
        );
        let out_schema = parse_wire_schema(&mut reader, transport_budget)?;
        if variable_output {
            ensure!(
                matches!(out_schema.kind, WireSchemaKind::VarU32 | WireSchemaKind::VarF32)
                    && out_schema.wire_tag == ValueType::ArrayU32,
                "invalid_abi: variable-output kernel {name} must have VarU32/VarF32 schema with Array U32 wire"
            );
        } else {
            ensure!(
                out_schema.kind == WireSchemaKind::FlatFixed && out_schema.wire_tag.is_array(),
                "invalid_abi: fixed-output kernel {name} must have FlatFixed array wire"
            );
        }
        let entry = abi
            .function(&name)
            .ok_or_else(|| anyhow!("invalid_abi: kernel {name} missing from {ABI_SECTION}"))?;
        ensure!(
            entry.parameter == Parameter::Value(in_schema.wire_tag) && entry.result == out_schema.wire_tag,
            "invalid_abi: kernel {name} signature mismatch against {ABI_SECTION}"
        );
        let io_schema_hash = reader.u64_le()?;
        let impl_hash = reader.u64_le()?;
        let binding_hash = reader.u64_le()?;
        let (ctx_schema, ctx_export, clear_export) = if has_context {
            let ctx = parse_wire_schema(&mut reader, transport_budget)?;
            ensure!(
                matches!(ctx.kind, WireSchemaKind::FlatFixed | WireSchemaKind::VarU32 | WireSchemaKind::VarF32),
                "invalid_abi: kernel {name} context schema cannot be Unit"
            );
            let ctx_name = reader.name()?;
            let clear_name = reader.name()?;
            let ctx_fn = abi
                .function(&ctx_name)
                .ok_or_else(|| anyhow!("invalid_abi: kernel {name} ctx_export {ctx_name} missing from {ABI_SECTION}"))?;
            ensure!(
                ctx_fn.parameter == Parameter::Value(ctx.wire_tag) && ctx_fn.result == ValueType::Unit,
                "invalid_abi: kernel {name} ctx_export {ctx_name} signature mismatch"
            );
            let clear_fn = abi.function(&clear_name).ok_or_else(|| {
                anyhow!("invalid_abi: kernel {name} clear_export {clear_name} missing from {ABI_SECTION}")
            })?;
            ensure!(
                clear_fn.parameter == Parameter::Value(ValueType::Unit) && clear_fn.result == ValueType::Unit,
                "invalid_abi: kernel {name} clear_export {clear_name} signature mismatch"
            );
            (Some(ctx), Some(ctx_name), Some(clear_name))
        } else {
            (None, None, None)
        };
        kernels.push(KernelManifest {
            id,
            name,
            flags,
            in_schema,
            out_schema,
            io_schema_hash,
            impl_hash,
            binding_hash,
            ctx_schema,
            ctx_export,
            clear_export,
        });
    }
    ensure!(reader.at == payload.len(), "invalid_abi: trailing {KERNELS_SECTION} bytes");
    Ok((transport_budget, kernels))
}

fn parse_processes(payload: &[u8], abi: &GuestAbi) -> Result<(u32, Vec<ProcessManifest>)> {
    let mut reader = Reader { bytes: payload, at: 0 };
    let version = reader.u32()?;
    ensure!(version == 1, "invalid_abi: unsupported {PROCESSES_SECTION} version {version}");
    let transport_budget = reader.u32()?;
    ensure!(
        transport_budget >= 8 && transport_budget.is_multiple_of(4),
        "invalid_abi: invalid {PROCESSES_SECTION} transport_budget {transport_budget}"
    );
    let count = reader.u32()?;
    let mut processes = Vec::with_capacity(usize::try_from(count).unwrap_or(0).min(1024));
    let mut names = std::collections::HashSet::new();
    for _ in 0..count {
        let name = reader.name()?;
        ensure!(names.insert(name.clone()), "invalid_abi: duplicate process {name}");
        let init_export = reader.name()?;
        let step_export = reader.name()?;
        let config_schema = parse_wire_schema(&mut reader, transport_budget)?;
        let cmd_schema = parse_wire_schema(&mut reader, transport_budget)?;
        let reply_schema = parse_wire_schema(&mut reader, transport_budget)?;
        let init_fn = abi
            .function(&init_export)
            .ok_or_else(|| anyhow!("invalid_abi: process {name} init_export {init_export} missing from {ABI_SECTION}"))?;
        ensure!(
            init_fn.parameter == Parameter::Value(config_schema.wire_tag) && init_fn.result == ValueType::Unit,
            "invalid_abi: process {name} init_export {init_export} signature mismatch"
        );
        let step_fn = abi
            .function(&step_export)
            .ok_or_else(|| anyhow!("invalid_abi: process {name} step_export {step_export} missing from {ABI_SECTION}"))?;
        ensure!(
            step_fn.parameter == Parameter::Value(cmd_schema.wire_tag) && step_fn.result == reply_schema.wire_tag,
            "invalid_abi: process {name} step_export {step_export} signature mismatch"
        );
        processes.push(ProcessManifest { name, init_export, step_export, config_schema, cmd_schema, reply_schema });
    }
    ensure!(reader.at == payload.len(), "invalid_abi: trailing {PROCESSES_SECTION} bytes");
    Ok((transport_budget, processes))
}

/// A `blot:tasks` section's payload: version 2, then the descriptor count
/// (at most 4096), and nothing after.
fn parse_tasks(payload: &[u8]) -> Result<u32> {
    let mut reader = Reader { bytes: payload, at: 0 };
    let version = reader.u32()?;
    ensure!(version == 2, "invalid_abi: unsupported task contract {version}");
    let count = reader.u32()?;
    ensure!(count <= 4096, "invalid_abi: {count} task descriptors (at most 4096)");
    ensure!(reader.at == payload.len(), "invalid_abi: trailing task section bytes");
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn section(name: &str, payload: &[u8]) -> Vec<u8> {
        let contents = [&[u8::try_from(name.len()).unwrap()][..], name.as_bytes(), payload].concat();
        [vec![0, u8::try_from(contents.len()).unwrap()], contents].concat()
    }

    fn module(sections: &[Vec<u8>]) -> Vec<u8> {
        [b"\0asm\x01\0\0\0".to_vec(), sections.concat()].concat()
    }

    #[test]
    fn reads_test_metadata_v1() {
        let abi = section(ABI_SECTION, &[8, 0, 0]);
        let payload = b"\x01\x82\x00\x00\x04test\x02\xc3\xa9\x06a.blot\xac\x02\x01\x04trap\x00\x00\x01";
        let wasm = module(&[section(TESTS_SECTION, payload), abi.clone()]);
        assert_eq!(
            read_abi(&wasm).unwrap().tests,
            [
                TestMetadata {
                    kind: TestKind::Check,
                    export_name: "test".into(),
                    display_name: "é".into(),
                    filename: "a.blot".into(),
                    line: 300,
                },
                TestMetadata {
                    kind: TestKind::Trap,
                    export_name: "trap".into(),
                    display_name: String::new(),
                    filename: String::new(),
                    line: 1,
                },
            ]
        );
        assert!(read_abi(&module(&[abi.clone(), section(TESTS_SECTION, &[1, 0])])).unwrap().tests.is_empty());
        assert!(read_abi(&module(&[abi])).unwrap().tests.is_empty());
    }

    #[test]
    fn rejects_invalid_test_metadata() {
        let abi = section(ABI_SECTION, &[8, 0, 0]);
        let empty = section(TESTS_SECTION, &[1, 0]);
        let duplicates = read_abi(&module(&[abi.clone(), empty.clone(), empty])).unwrap_err();
        assert!(duplicates.to_string().contains("more than one blot:tests section"));
        for (payload, message) in [
            (&b""[..], "truncated"),
            (&b"\x01"[..], "truncated"),
            (&b"\x02\x00"[..], "unsupported blot:tests version"),
            (&b"\x01\x01\x02\x00\x00\x00\x00"[..], "unknown blot:tests kind"),
            (&b"\x01\x01\x00\x00\x00\x00"[..], "truncated blot:tests entries"),
            (&b"\x01\x01\x00\x05a\x00\x00"[..], "truncated name"),
            (&b"\x01\x01\x00\x01\xff\x00\x00\x01"[..], "not UTF-8"),
            (&b"\x01\x01\x00\x00\x00\x00\x80"[..], "truncated"),
            (&b"\x01\x01\x00\x00\x00\x00\xff\xff\xff\xff\x10"[..], "LEB128 exceeds u32"),
            (&b"\x01\x00\xff"[..], "trailing blot:tests bytes"),
        ] {
            let error = read_abi(&module(&[abi.clone(), section(TESTS_SECTION, payload)])).unwrap_err().to_string();
            assert!(error.starts_with("invalid_abi:") && error.contains(message), "{payload:?}: {error}");
        }
    }

    #[test]
    fn manifest_test_vectors() {
        assert_eq!(parse_abi(&[0x02, 0x00, 0x00]).unwrap(), GuestAbi::default());
        let abi = parse_abi(&[0x02, 0x01, 0x04, 0x6d, 0x61, 0x69, 0x6e, 0x04, 0x01, 0x01, 0x01, 0x00]).unwrap();
        assert_eq!(
            abi.functions,
            vec![GuestFunction {
                name: "main".into(),
                parameter: Parameter::Callback(CallbackType { parameter: ValueType::U32, result: ValueType::U32 }),
                result: ValueType::U32,
            }]
        );
        assert!(parse_abi(&[0x02, 0x00, 0x00, 0x00]).is_err());
        assert_eq!(parse_abi(&[0x03, 0x00, 0x00]).unwrap(), GuestAbi { ownership: true, ..GuestAbi::default() });
        assert!(parse_abi(&[0x04, 0x00, 0x00]).is_err());
        assert_eq!(parse_abi(&[0x08, 0x00, 0x00]).unwrap(), GuestAbi { data_stacks: true, ..GuestAbi::default() });
    }

    #[test]
    fn task_modules_are_read_and_refused() {
        // main: (Array F32 -> Array F32, concurrent) -> Unit, ABI 3.
        let manifest = [0x03, 0x01, 0x04, b'm', b'a', b'i', b'n', 13, 0x06, 0x06, 0x00, 0x00];
        let abi = read_abi(&module(&[section(ABI_SECTION, &manifest), section(TASKS_SECTION, &[2, 3])])).unwrap();
        assert_eq!((abi.tasks, abi.concurrent.as_slice()), (Some(3), &["main".to_owned()][..]));
        assert!(abi.check_game().is_ok(), "its main still takes the host");
        let refused = format!("{:#}", abi.check_hostable().unwrap_err());
        assert!(
            refused.contains("structured tasks (blot:tasks, 3 descriptors; concurrent callbacks for main)"),
            "{refused}"
        );
        let error = |sections: &[Vec<u8>]| format!("{:#}", read_abi(&module(sections)).unwrap_err());
        assert!(error(&[section(ABI_SECTION, &manifest)]).contains("concurrent callbacks need blot:tasks"));
        let mut version_2 = manifest;
        version_2[0] = 2;
        assert!(error(&[section(ABI_SECTION, &version_2), section(TASKS_SECTION, &[2, 3])]).contains("ownership"));
        assert!(error(&[section(ABI_SECTION, &manifest), section(TASKS_SECTION, &[1, 3])]).contains("contract 1"));
        assert!(GuestAbi::default().check_hostable().is_ok());
        let stack = section(ABI_SECTION, &[8, 0, 0]);
        assert!(read_abi(&module(std::slice::from_ref(&stack))).unwrap().data_stacks);
        assert!(error(&[stack.clone(), section("blot:ownership", &[1, 0, 1])]).contains("counted runtime"));
        assert!(error(&[stack, section(TASKS_SECTION, &[2, 0])]).contains("ownership"));
    }

    #[test]
    fn invocation_finish_is_optional_and_requires_abi_8() {
        let manifest = section(ABI_SECTION, &[8, 0, 0]);
        let finish = section(INVOCATION_SECTION, &[1]);
        assert!(!read_abi(&module(std::slice::from_ref(&manifest))).unwrap().invocation_finish);
        assert!(read_abi(&module(&[manifest.clone(), finish.clone()])).unwrap().invocation_finish);
        assert!(read_abi(&module(&[manifest.clone(), finish.clone(), finish.clone()])).is_err());
        assert!(read_abi(&module(&[manifest, section(INVOCATION_SECTION, &[2])])).is_err());
        assert!(read_abi(&module(&[section(ABI_SECTION, &[2, 0, 0]), finish])).is_err());
    }

    #[test]
    fn games_need_main_taking_the_host() {
        // main: (Array F32 -> Array F32) -> Unit, and a constant.
        let game = [0x02, 0x01, 0x04, b'm', b'a', b'i', b'n', 0x04, 0x06, 0x06, 0x00, 0x01, 0x01, b'k', 0x03];
        let abi = parse_abi(&game).unwrap();
        assert!(abi.check_game().is_ok() && abi.uses_arrays());
        assert_eq!(abi.callbacks()[0].import_name(), "call_array_f32_array_f32");
        let mut wrong = game;
        wrong[10] = 0x01;
        assert!(parse_abi(&wrong).unwrap().check_game().is_err(), "main returns U32");
        assert!(GuestAbi::default().check_game().is_err(), "no main");
        // In a module: the custom section, found among others.
        let section = [&[8u8][..], &b"blot:abi"[..], &game[..]].concat();
        let mut module = b"\0asm\x01\0\0\0".to_vec();
        module.extend([1, 0, 0, u8::try_from(section.len()).unwrap()]);
        module.extend(&section);
        assert_eq!(read_abi(&module).unwrap(), abi);
        assert!(read_abi(&module[..8]).is_err(), "no manifest");
    }

    #[test]
    fn memory_shape_decodes_standard_shared_memory64_and_custom_page_size() {
        // No memory section
        let none = read_memory_shape(&module(&[])).unwrap();
        assert!(!none.section_present && !none.has_memory);

        // Standard 32-bit bounded: count=1, flags=0x01, min=2, max=16
        let bounded = read_memory_shape(&module(&[vec![5, 4, 1, 0x01, 2, 16]])).unwrap();
        assert_eq!(
            bounded,
            MemoryShape {
                section_present: true,
                has_memory: true,
                memory64: false,
                shared: false,
                page_size_log2: 16,
                initial_pages: 2,
                maximum_pages: Some(16),
            }
        );

        // Unbounded 32-bit: count=1, flags=0x00, min=1
        let unbounded = read_memory_shape(&module(&[vec![5, 3, 1, 0x00, 1]])).unwrap();
        assert!(unbounded.has_memory && unbounded.maximum_pages.is_none() && unbounded.initial_pages == 1);

        // Shared: flags=0x03, min=1, max=4
        let shared = read_memory_shape(&module(&[vec![5, 4, 1, 0x03, 1, 4]])).unwrap();
        assert!(shared.shared && !shared.memory64 && shared.maximum_pages == Some(4));

        // Memory64: flags=0x04, min=1
        let mem64 = read_memory_shape(&module(&[vec![5, 3, 1, 0x04, 1]])).unwrap();
        assert!(mem64.memory64 && !mem64.shared);

        // Custom page size: flags=0x08, min=1, page_size_log2=0 (1-byte pages)
        let custom = read_memory_shape(&module(&[vec![5, 4, 1, 0x08, 1, 0]])).unwrap();
        assert_eq!(custom.page_size_log2, 0);

        // Rejects count > 1, unknown flags, trailing bytes, max < initial
        assert!(read_memory_shape(&module(&[vec![5, 5, 2, 0x00, 1, 0x00, 1]])).is_err());
        assert!(read_memory_shape(&module(&[vec![5, 3, 1, 0x10, 1]])).is_err());
        assert!(read_memory_shape(&module(&[vec![5, 4, 1, 0x00, 1, 0]])).is_err());
        assert!(read_memory_shape(&module(&[vec![5, 4, 1, 0x01, 4, 2]])).is_err());
    }

    #[test]
    fn kernels_and_processes_sections_validate_against_abi_8() {
        let str_bytes = |s: &str| [&[u8::try_from(s.len()).unwrap()][..], s.as_bytes()].concat();
        let uleb = |mut v: u32| -> Vec<u8> {
            let mut out = Vec::new();
            loop {
                let mut b = u8::try_from(v & 0x7f).unwrap();
                v >>= 7;
                if v != 0 {
                    b |= 0x80;
                }
                out.push(b);
                if v == 0 {
                    return out;
                }
            }
        };
        let wire_fixed = |wire_tag: u8, word_tags: &[u8], hash: u64| -> Vec<u8> {
            let mut out = vec![1, wire_tag];
            out.extend(uleb(u32::try_from(word_tags.len()).unwrap()));
            out.extend_from_slice(word_tags);
            out.extend_from_slice(&hash.to_le_bytes());
            out
        };
        // ABI 8 exports: k(ArrayU32->ArrayU32), blot:kernel_ctx:k(ArrayF32->Unit), blot:kernel_clear:k(Unit->Unit),
        // blot:proc_init:p(U32->Unit), blot:proc_step:p(U32->U32)
        let abi_payload = [
            vec![8, 5],
            str_bytes("k"),
            vec![5, 5],
            str_bytes("blot:kernel_ctx:k"),
            vec![6, 0],
            str_bytes("blot:kernel_clear:k"),
            vec![0, 0],
            str_bytes("blot:proc_init:p"),
            vec![1, 0],
            str_bytes("blot:proc_step:p"),
            vec![1, 1],
            vec![0],
        ]
        .concat();
        let kernels_payload = [
            vec![1],
            uleb(DEFAULT_TRANSPORT_BUDGET),
            vec![1],
            str_bytes("k"),
            vec![KERNEL_FLAG_HAS_CONTEXT],
            wire_fixed(5, &[1], 0x11),
            wire_fixed(5, &[1, 1], 0x22),
            0x33u64.to_le_bytes().to_vec(),
            0x44u64.to_le_bytes().to_vec(),
            0x55u64.to_le_bytes().to_vec(),
            vec![3, 6, 0], // VarF32 ctx
            0x66u64.to_le_bytes().to_vec(),
            str_bytes("blot:kernel_ctx:k"),
            str_bytes("blot:kernel_clear:k"),
        ]
        .concat();
        let processes_payload = [
            vec![1],
            uleb(DEFAULT_TRANSPORT_BUDGET),
            vec![1],
            str_bytes("p"),
            str_bytes("blot:proc_init:p"),
            str_bytes("blot:proc_step:p"),
            wire_fixed(1, &[1], 0x77),
            wire_fixed(1, &[1], 0x88),
            wire_fixed(1, &[1], 0x99),
        ]
        .concat();
        let custom_section = |name: &str, payload: &[u8]| -> Vec<u8> {
            let mut contents = str_bytes(name);
            contents.extend_from_slice(payload);
            let mut out = vec![0];
            out.extend(uleb(u32::try_from(contents.len()).unwrap()));
            out.extend(contents);
            out
        };
        let wasm = module(&[
            custom_section(ABI_SECTION, &abi_payload),
            custom_section(KERNELS_SECTION, &kernels_payload),
            custom_section(PROCESSES_SECTION, &processes_payload),
        ]);
        let abi = read_abi(&wasm).unwrap();
        assert_eq!(abi.kernels.len(), 1);
        assert_eq!(abi.processes.len(), 1);
        assert_eq!(abi.transport_budget, DEFAULT_TRANSPORT_BUDGET);
        assert!(abi.is_host_internal("blot:kernel_ctx:k"));
        assert!(abi.is_host_internal("blot:kernel_clear:k"));
        assert!(abi.is_host_internal("blot:proc_init:p"));
        assert!(abi.is_host_internal("blot:proc_step:p"));
        assert!(!abi.is_host_internal("k"));

        // Mismatched transport budgets between kernels and processes rejected
        let bad_processes = [
            vec![1],
            uleb(1024),
            vec![0],
        ]
        .concat();
        let bad_wasm = module(&[
            custom_section(ABI_SECTION, &abi_payload),
            custom_section(KERNELS_SECTION, &kernels_payload),
            custom_section(PROCESSES_SECTION, &bad_processes),
        ]);
        assert!(read_abi(&bad_wasm).is_err());
    }
}
