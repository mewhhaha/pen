//! The shaders' parser: a shader's `Parameters` struct, if it has one, as a
//! record type of the same name (the up to four f32s a draw passes it), and
//! its module-scope scalar constants (f32, u32 and bool), so Blot and the
//! shader share them. gdev's asset_parsers.ts read these with the npm package
//! wgsl_reflect; this reads them with naga, which also checks the whole
//! shader as wgpu will.
//!
//! naga keeps a constant of a concrete type (`const count = 9u;`) in the
//! module it makes, but drops one of an abstract type (`const width = 0.16;`,
//! most of gdev's) once it has used it. So each module-scope constant, found
//! by its `const` keyword outside comments and braces, is also read through a
//! probe appended to the shader, `var<private> PROBE = NAME;`, where naga
//! gives an abstract float the f32 that wgsl_reflect gave it. An abstract
//! integer, which wgsl_reflect left out as no constant of Blot's, becomes an
//! i32 or a u32, or beyond those an f32 that only it can also pass as an i64.
//! A float beyond f32 fails its probe, and is probed again as an f64, so that
//! it fails as it did with assets.ts.
use naga::front::wgsl::ParseError;
use naga::{Constant, Expression, Handle, Literal, Module, Scalar, TypeInner};

use crate::module::{Description, Record, Type, Value};

/// What the shader `code` gives Blot, or why it does not parse.
pub(crate) fn reflect(code: &str) -> Result<Description, String> {
    let names = constant_names(code);
    let probed = probe(code, &names)?;
    let module = &probed.module;
    let mut types = Vec::new();
    let parameters = module.types.iter().find_map(|(_, ty)| match &ty.inner {
        TypeInner::Struct { members, .. } if ty.name.as_deref() == Some("Parameters") => Some(members),
        _ => None,
    });
    if let Some(members) = parameters {
        let f32 = |ty| matches!(module.types[ty].inner, TypeInner::Scalar(Scalar::F32));
        if members.len() > 4 || !members.iter().all(|member| f32(member.ty)) {
            return Err("Parameters must be up to four f32s, a draw's parameters".to_string());
        }
        let fields = members.iter().map(|member| (member.name.clone().unwrap_or_default(), Type::F32)).collect();
        types.push(Record { name: "Parameters".to_string(), fields });
    }
    let mut values = Vec::new();
    for (index, name) in names.iter().enumerate() {
        let concrete = module.constants.iter().find(|(_, constant)| constant.name.as_deref() == Some(*name));
        let value = match concrete {
            Some((_, constant)) => concrete_value(module, constant, name)?,
            None => abstract_value(code, &probed, index, name),
        };
        if let Some((shape, value)) = value {
            values.push((name.to_string(), shape, value));
        }
    }
    Ok(Description { types, values })
}

/// A concrete constant's value, when it is an f32, a u32 or a bool.
fn concrete_value(module: &Module, constant: &Constant, name: &str) -> Result<Option<(Type, Value)>, String> {
    Ok(match literal(module, constant.init) {
        Some(Literal::F32(value)) => Some((Type::F32, Value::F32(value.into()))),
        Some(Literal::U32(value)) => Some((Type::U32, Value::U32(value))),
        Some(Literal::Bool(value)) => Some((Type::Bool, Value::Bool(value))),
        _ if [Scalar::F32, Scalar::U32, Scalar::BOOL]
            .iter()
            .any(|scalar| module.types[constant.ty].inner == TypeInner::Scalar(*scalar)) =>
        {
            return Err(format!("{name}: naga left its value uncomputed"));
        }
        _ => None,
    })
}

/// An abstract constant's value, through its probe, when it is a float.
fn abstract_value(code: &str, probed: &Probed, index: usize, name: &str) -> Option<(Type, Value)> {
    let probe = format!("{}{index}", probed.prefix);
    let module = &probed.module;
    let (_, global) = module.global_variables.iter().find(|(_, global)| global.name.as_ref() == Some(&probe))?;
    match (probed.kinds[index], literal(module, global.init?)?) {
        (Probe::Untyped, Literal::F32(value)) => {
            // An integer becomes an f32 only beyond i32 and u32.
            let beyond = value.fract() == 0.0 && !(-2_147_483_648.0..4_294_967_296.0).contains(&value);
            let integer = beyond
                && naga::front::wgsl::parse_str(&format!("{code}\nvar<private> {probe}: i64 = {name};\n")).is_ok();
            (!integer).then_some((Type::F32, Value::F32(value.into())))
        }
        (Probe::Wide, Literal::F64(value)) => Some((Type::F32, Value::F32(value))),
        _ => None,
    }
}

/// How a constant is probed: untyped first; as an f64 when that fails, which
/// only a float beyond f32 passes; and not at all when that fails too (a
/// vector or array of those).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Probe {
    Untyped,
    Wide,
    Skipped,
}

struct Probed {
    module: Module,
    /// The probes' names: this, then the constant's index.
    prefix: String,
    kinds: Vec<Probe>,
}

/// The shader `code` with a probe for each of the constants `names`.
fn probe(code: &str, names: &[&str]) -> Result<Probed, String> {
    let mut prefix = "pen_probe".to_string();
    while code.contains(&prefix) {
        prefix.push('_');
    }
    prefix.push('_');
    let mut kinds = vec![Probe::Untyped; names.len()];
    // The shader with the probes of the constants `probed`.
    let source = |kinds: &[Probe], probed: &[usize]| {
        let mut source = format!("{code}\n");
        for &index in probed {
            let annotation = match kinds[index] {
                Probe::Untyped => "",
                Probe::Wide => ": f64",
                Probe::Skipped => continue,
            };
            source.push_str(&format!("var<private> {prefix}{index}{annotation} = {};\n", names[index]));
        }
        source
    };
    let parses = |kinds: &[Probe], probed: &[usize]| naga::front::wgsl::parse_str(&source(kinds, probed)).is_ok();
    let mut original_parses = false;
    loop {
        let probed: Vec<usize> = (0..names.len()).filter(|index| kinds[*index] != Probe::Skipped).collect();
        let all = source(&kinds, &probed);
        let error = match naga::front::wgsl::parse_str(&all) {
            Ok(module) => return Ok(Probed { module, prefix, kinds }),
            Err(error) => error,
        };
        if !original_parses {
            naga::front::wgsl::parse_str(code).map_err(|error| describe(&error, code))?;
            original_parses = true;
        }
        // A probe that fails does so on its own (naga places the failure at
        // the constant it probes, or within it), so halve the probes until
        // one is left.
        let mut suspects = &probed[..];
        while suspects.len() > 1 {
            let (first, second) = suspects.split_at(suspects.len() / 2);
            suspects = if parses(&kinds, first) { second } else { first };
        }
        let Some(&index) = suspects.first().filter(|index| !parses(&kinds, &[**index])) else {
            return Err(format!("its constants cannot be read: {}", describe(&error, &all)));
        };
        kinds[index] = if kinds[index] == Probe::Untyped { Probe::Wide } else { Probe::Skipped };
    }
}

/// A naga error as wgsl_reflect's were: the message, then its line.
fn describe(error: &ParseError, code: &str) -> String {
    match error.location(code) {
        Some(location) => format!("{}. Line: {}", error.message(), location.line_number),
        None => error.message().to_string(),
    }
}

/// The scalar a module-scope expression comes to, if it is one.
fn literal(module: &Module, expression: Handle<Expression>) -> Option<Literal> {
    match module.global_expressions[expression] {
        Expression::Literal(literal) => Some(literal),
        Expression::Constant(constant) => literal(module, module.constants[constant].init),
        Expression::ZeroValue(ty) => match module.types[ty].inner {
            TypeInner::Scalar(scalar) => Literal::zero(scalar),
            _ => None,
        },
        _ => None,
    }
}

/// The names of the module-scope constants in `code`, in order: each word
/// after a `const` outside comments and braces.
fn constant_names(code: &str) -> Vec<&str> {
    let mut names = Vec::new();
    let mut depth = 0_usize;
    let mut after_const = false;
    let mut rest = code;
    while let Some(next) = rest.chars().next() {
        if let Some(comment) = rest.strip_prefix("//") {
            rest = comment
                .find(['\n', '\u{b}', '\u{c}', '\r', '\u{85}', '\u{2028}', '\u{2029}'])
                .map_or("", |end| &comment[end..]);
        } else if rest.starts_with("/*") {
            rest = after_block_comment(rest);
        } else if is_word(next) {
            let end = rest.find(|character| !is_word(character)).unwrap_or(rest.len());
            let word = &rest[..end];
            if after_const {
                names.push(word);
                after_const = false;
            } else if depth == 0 && word == "const" {
                after_const = true;
            }
            rest = &rest[end..];
        } else {
            match next {
                '{' => depth += 1,
                '}' => depth = depth.saturating_sub(1),
                _ => {}
            }
            after_const &= next.is_whitespace();
            rest = &rest[next.len_utf8()..];
        }
    }
    names
}

/// What follows the block comment `text` starts with (WGSL's nest).
fn after_block_comment(text: &str) -> &str {
    let mut depth = 0_usize;
    let mut rest = text;
    while let Some(next) = rest.chars().next() {
        if rest.starts_with("/*") {
            depth += 1;
            rest = &rest[2..];
        } else if rest.starts_with("*/") {
            depth -= 1;
            rest = &rest[2..];
            if depth == 0 {
                return rest;
            }
        } else {
            rest = &rest[next.len_utf8()..];
        }
    }
    rest
}

/// Whether `character` can be in a WGSL identifier or number.
fn is_word(character: char) -> bool {
    character == '_' || unicode_ident::is_xid_continue(character)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_module_scope_constants() {
        let code = "// const a = 1.0;\n/* const b /* nested */ const c */ const d = 1.0;\nconst_assert true;\n\
                    fn f() -> f32 { const e = 2.0; return e; }\nstruct S { x: f32 }\nconst\n  f: f32 = 3.0;\nconst g=1;";
        assert_eq!(constant_names(code), ["d", "f", "g"]);
    }
}
