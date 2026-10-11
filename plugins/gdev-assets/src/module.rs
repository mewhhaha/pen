//! What blot2's compiler/assets.ts makes of a parser's description: a Blot
//! module that declares each record type as nominal data, `type NAME is data
//! = #NAME { field: Type }`, and each value as a constant of its type, every
//! name checked and the module's size and nesting bounded. Only the types
//! gdev's parsers describe are here: Bool, U32, F32, arrays, tuples and
//! records.
use crate::LIMIT;
use crate::number;

/// A value's type.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Type {
    Bool,
    U32,
    F32,
    Array(Box<Type>),
    Tuple(Vec<Type>),
    Record(Record),
}

/// A nominal record type: its name and its fields, in order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Record {
    pub(crate) name: String,
    pub(crate) fields: Vec<(String, Type)>,
}

/// A value of a [`Type`]. A record's values are in its type's field order.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Value {
    Bool(bool),
    U32(u32),
    /// Any number; it is rounded to f32, and must stay finite.
    F32(f64),
    Array(Vec<Value>),
    Tuple(Vec<Value>),
    Record(Vec<Value>),
}

/// What a parser makes of a file: record types the module exports without a
/// value, and its named values.
#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Description {
    pub(crate) types: Vec<Record>,
    pub(crate) values: Vec<(String, Type, Value)>,
}

/// The Blot source of `description`: its record types' declarations, each
/// once and in the order first met (a record's own field types before it),
/// then its values' constants.
pub(crate) fn lower(description: &Description) -> Result<String, String> {
    let mut lowering = Lowering::default();
    for record in &description.types {
        lowering.visit(0)?;
        lowering.declare(record, 0)?;
    }
    let mut body = String::new();
    for (name, shape, value) in &description.values {
        identifier(name, false)?;
        let annotation = lowering.shape(shape, 0)?;
        body.push_str(&format!("const {name}: {annotation} = "));
        lowering.value(shape, value, 0, &mut body)?;
        body.push('\n');
    }
    let mut source: String = lowering.declarations.into_iter().map(|(_, declaration)| declaration).collect();
    if source.len() + body.len() > LIMIT {
        return Err("Generated asset module exceeds 16 MiB".to_string());
    }
    source.push_str(&body);
    Ok(source)
}

#[derive(Default)]
struct Lowering {
    /// Each record type's declaration, by name, in the order first met.
    declarations: Vec<(String, String)>,
    /// The types and values visited, against the structural budget.
    nodes: usize,
}

impl Lowering {
    fn visit(&mut self, depth: usize) -> Result<(), String> {
        self.nodes += 1;
        if self.nodes > 1_000_000 || depth > 128 {
            return Err("Asset description exceeds the structural budget".to_string());
        }
        Ok(())
    }

    /// The Blot type of `shape`, declaring the records in it.
    fn shape(&mut self, shape: &Type, depth: usize) -> Result<String, String> {
        self.visit(depth)?;
        Ok(match shape {
            Type::Bool => "Bool".to_string(),
            Type::U32 => "U32".to_string(),
            Type::F32 => "F32".to_string(),
            Type::Array(element) => format!("(Array {})", self.shape(element, depth + 1)?),
            Type::Tuple(elements) => {
                let elements =
                    elements.iter().map(|element| self.shape(element, depth + 1)).collect::<Result<Vec<_>, _>>()?;
                format!("({}{})", elements.join(", "), if elements.len() == 1 { "," } else { "" })
            }
            Type::Record(record) => self.declare(record, depth)?,
        })
    }

    /// Declares `record` (and the records in its fields), and names it.
    fn declare(&mut self, record: &Record, depth: usize) -> Result<String, String> {
        let name = &record.name;
        identifier(name, true)?;
        let mut fields = Vec::with_capacity(record.fields.len());
        for (field, shape) in &record.fields {
            identifier(field, false)?;
            fields.push(format!("{field}: {}", self.shape(shape, depth + 1)?));
        }
        let declaration = format!("type {name} is data = #{name} {{ {} }}\n", fields.join(", "));
        match self.declarations.iter().find(|(declared, _)| declared == name) {
            Some((_, previous)) if *previous != declaration => return Err(format!("Conflicting asset type: {name}")),
            Some(_) => {}
            None => self.declarations.push((name.clone(), declaration)),
        }
        Ok(name.clone())
    }

    /// Writes `value`, of type `shape`, as a Blot expression.
    fn value(&mut self, shape: &Type, value: &Value, depth: usize, out: &mut String) -> Result<(), String> {
        self.visit(depth)?;
        match (shape, value) {
            (Type::Bool, Value::Bool(value)) => out.push_str(if *value { "#True" } else { "#False" }),
            (Type::U32, Value::U32(value)) => out.push_str(&value.to_string()),
            (Type::F32, Value::F32(value)) => out.push_str(&f32_literal(*value)?),
            (Type::Array(element), Value::Array(items)) => {
                out.push_str("#[");
                for (index, item) in items.iter().enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    self.value(element, item, depth + 1, out)?;
                }
                out.push(']');
            }
            (Type::Tuple(elements), Value::Tuple(items)) => {
                if elements.len() != items.len() {
                    return Err("Asset tuple arity mismatch".to_string());
                }
                out.push('(');
                for (index, (element, item)) in elements.iter().zip(items).enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    self.value(element, item, depth + 1, out)?;
                }
                out.push_str(if items.len() == 1 { ",)" } else { ")" });
            }
            (Type::Record(record), Value::Record(items)) => {
                if record.fields.len() != items.len() {
                    return Err(format!("Fields do not match asset type {}", record.name));
                }
                out.push_str(&format!("(#{} {{ ", record.name));
                for (index, ((field, element), item)) in record.fields.iter().zip(items).enumerate() {
                    if index > 0 {
                        out.push_str(", ");
                    }
                    out.push_str(field);
                    out.push_str(": ");
                    self.value(element, item, depth + 1, out)?;
                }
                out.push_str(" })");
            }
            _ => return Err("Asset value does not match its type".to_string()),
        }
        Ok(())
    }
}

/// `value` rounded to f32, as a Blot F32 literal: JavaScript's digits with
/// `.0` when they have no point or exponent, and in parentheses when negative
/// (negative zero too).
fn f32_literal(value: f64) -> Result<String, String> {
    let rounded = value as f32;
    if !value.is_finite() || !rounded.is_finite() {
        return Err("Asset number does not fit finite F32".to_string());
    }
    let mut text = number::javascript(f64::from(rounded).abs());
    if !text.contains(['.', 'e']) {
        text.push_str(".0");
    }
    Ok(if rounded.is_sign_negative() { format!("(-{text})") } else { text })
}

/// The words Blot keeps for itself.
const KEYWORDS: [&str; 27] = [
    "const", "let", "type", "data", "effect", "is", "fn", "do", "use", "return", "yield", "break", "for", "ever", "in",
    "case", "of", "if", "then", "else", "as", "import", "infix", "infixl", "infixr", "self", "where",
];

/// Checks `name` as a Blot type name (`upper`) or value or field name.
fn identifier(name: &str, upper: bool) -> Result<(), String> {
    let mut characters = name.chars();
    let first = characters.next().is_some_and(|first| {
        if upper { first.is_ascii_uppercase() } else { first.is_ascii_lowercase() || first == '_' }
    });
    let valid = first
        && characters.all(|character| character.is_ascii_alphanumeric() || character == '_')
        && !KEYWORDS.contains(&name)
        && name != "_"
        && !(upper && ["Unit", "Bool", "U32", "F32", "Array", "List"].contains(&name));
    if valid {
        Ok(())
    } else {
        Err(format!("Invalid Blot {} name: {name}", if upper { "type" } else { "value/field" }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn record(name: &str, fields: &[(&str, Type)]) -> Record {
        Record { name: name.to_string(), fields: fields.iter().map(|(n, t)| (n.to_string(), t.clone())).collect() }
    }

    #[test]
    fn lowers_types_and_values() {
        let inner = record("Inner", &[("x", Type::F32)]);
        let outer = record(
            "Outer",
            &[
                ("inner", Type::Record(inner.clone())),
                ("pair", Type::Tuple(vec![Type::U32, Type::Bool])),
                ("one", Type::Tuple(vec![Type::F32])),
            ],
        );
        let description = Description {
            types: vec![record("Shown", &[])],
            values: vec![
                (
                    "outer".to_string(),
                    Type::Array(Box::new(Type::Record(outer))),
                    Value::Array(vec![Value::Record(vec![
                        Value::Record(vec![Value::F32(-0.5)]),
                        Value::Tuple(vec![Value::U32(7), Value::Bool(false)]),
                        Value::Tuple(vec![Value::F32(-0.0)]),
                    ])]),
                ),
                ("inner".to_string(), Type::Record(inner), Value::Record(vec![Value::F32(1e-7)])),
                ("none".to_string(), Type::Array(Box::new(Type::U32)), Value::Array(vec![])),
            ],
        };
        assert_eq!(
            lower(&description).unwrap(),
            "type Shown is data = #Shown {  }\n\
             type Inner is data = #Inner { x: F32 }\n\
             type Outer is data = #Outer { inner: Inner, pair: (U32, Bool), one: (F32,) }\n\
             const outer: (Array Outer) = #[(#Outer { inner: (#Inner { x: (-0.5) }), pair: (7, #False), one: ((-0.0),) })]\n\
             const inner: Inner = (#Inner { x: 1.0000000116860974e-7 })\n\
             const none: (Array U32) = #[]\n"
        );
    }

    #[test]
    fn checks_names_numbers_and_conflicts() {
        let value = |name: &str, shape: Type, value: Value| Description {
            types: vec![],
            values: vec![(name.to_string(), shape, value)],
        };
        assert_eq!(lower(&value("Big", Type::U32, Value::U32(1))).unwrap_err(), "Invalid Blot value/field name: Big");
        assert_eq!(lower(&value("fn", Type::U32, Value::U32(1))).unwrap_err(), "Invalid Blot value/field name: fn");
        assert_eq!(lower(&value("_", Type::U32, Value::U32(1))).unwrap_err(), "Invalid Blot value/field name: _");
        assert!(lower(&value("_ok9", Type::U32, Value::U32(1))).is_ok());
        let typed = |name: &str| value("x", Type::Record(record(name, &[])), Value::Record(vec![]));
        assert_eq!(lower(&typed("List")).unwrap_err(), "Invalid Blot type name: List");
        assert_eq!(lower(&typed("lower")).unwrap_err(), "Invalid Blot type name: lower");
        assert_eq!(
            lower(&value("x", Type::F32, Value::F32(1e39))).unwrap_err(),
            "Asset number does not fit finite F32"
        );
        assert_eq!(
            lower(&value("x", Type::F32, Value::F32(f64::NAN))).unwrap_err(),
            "Asset number does not fit finite F32"
        );
        let conflict =
            Description { types: vec![record("Same", &[]), record("Same", &[("x", Type::U32)])], values: vec![] };
        assert_eq!(lower(&conflict).unwrap_err(), "Conflicting asset type: Same");
        let mut deep = Type::U32;
        for _ in 0..130 {
            deep = Type::Array(Box::new(deep));
        }
        assert_eq!(
            lower(&value("x", deep, Value::Array(vec![]))).unwrap_err(),
            "Asset description exceeds the structural budget"
        );
    }
}
