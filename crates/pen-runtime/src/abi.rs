//! Blot's guest ABI manifest (`blot2/compiler/guest-abi.md`): the
//! `blot:abi` custom section a guest module declares its exports in, and
//! the entry a game must have. Native pen (blot-host) and the browser read
//! it alike, before a module runs. ABI 3's ownership hooks are specified in
//! `blot2/zig-native/SHARED_OWNERSHIP.md`.
use std::fmt;

use anyhow::{Result, anyhow, bail, ensure};

/// The custom section the manifest is in.
pub const ABI_SECTION: &str = "blot:abi";
/// The manifest versions this reads: 2, and 3, whose guests also track what
/// they own across each call (see [`GuestAbi::ownership`]).
pub const ABI_VERSIONS: std::ops::RangeInclusive<u32> = 2..=3;

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

/// The `blot:abi` manifest: the module's exports, in declaration order.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GuestAbi {
    pub functions: Vec<GuestFunction>,
    pub constants: Vec<GuestConstant>,
    /// ABI 3: the module exports `blot:ownership_enter`, which a host calls
    /// before each call's inputs, and `blot:ownership_leave(outcome)`, which
    /// it calls after copying the result out (outcome 0) or after a failure
    /// (1), before `blot:reset`.
    pub ownership: bool,
}

/// What a game's `main` takes: the host, which answers each frame's
/// request with the next frame's input.
pub const GAME_HOST: CallbackType = CallbackType { parameter: ValueType::ArrayF32, result: ValueType::ArrayF32 };

impl GuestAbi {
    pub fn function(&self, name: &str) -> Option<&GuestFunction> {
        self.functions.iter().find(|function| function.name == name)
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

struct Reader<'a> {
    bytes: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn byte(&mut self) -> Result<u8> {
        let byte = *self.bytes.get(self.at).ok_or_else(|| anyhow!("invalid_abi: truncated manifest"))?;
        self.at += 1;
        Ok(byte)
    }

    /// A canonical unsigned LEB128 u32.
    fn u32(&mut self) -> Result<u32> {
        let mut value = 0u64;
        for index in 0..5 {
            let byte = self.byte()?;
            value |= u64::from(byte & 0x7f) << (index * 7);
            if byte & 0x80 == 0 {
                ensure!(index == 0 || byte != 0, "invalid_abi: non-canonical LEB128");
                return u32::try_from(value).map_err(|_| anyhow!("invalid_abi: LEB128 exceeds u32"));
            }
        }
        bail!("invalid_abi: LEB128 too long")
    }

    fn name(&mut self) -> Result<String> {
        let length = self.u32()? as usize;
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
    let mut abi = GuestAbi { ownership: version == 3, ..GuestAbi::default() };
    let mut names = std::collections::HashSet::new();
    for _ in 0..reader.u32()? {
        let name = reader.name()?;
        let tag = reader.byte()?;
        let parameter = if tag == 4 {
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

/// Finds the single `blot:abi` custom section in a module's bytes.
pub fn read_abi(wasm: &[u8]) -> Result<GuestAbi> {
    ensure!(wasm.len() >= 8 && wasm[..4] == *b"\0asm", "not a Wasm module");
    let mut reader = Reader { bytes: wasm, at: 8 };
    let mut found = None;
    while reader.at < wasm.len() {
        let id = reader.byte()?;
        let size = reader.u32()? as usize;
        let end = reader.at.checked_add(size).filter(|end| *end <= wasm.len());
        let end = end.ok_or_else(|| anyhow!("truncated Wasm section"))?;
        if id == 0 {
            let mut section = Reader { bytes: &wasm[..end], at: reader.at };
            if section.name()? == ABI_SECTION {
                ensure!(found.is_none(), "invalid_abi: more than one {ABI_SECTION} section");
                found = Some(parse_abi(&wasm[section.at..end])?);
            }
        }
        reader.at = end;
    }
    found.ok_or_else(|| anyhow!("invalid_abi: missing {ABI_SECTION} section"))
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
