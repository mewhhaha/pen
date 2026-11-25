//! Compile and watch checkout programs through Wasm components.
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, channel};

use anyhow::{Result, bail};
use pen_plugins::{Diagnostic, Level, Outcome, Plugins};

use crate::dev::watch;
use crate::devlog::{self, Source};

/// The checkout's plugins, if its `pen.json` has a compiler.
pub fn load(root: &Path) -> Result<Option<Arc<Plugins>>> {
    let log: pen_plugins::Logger = Arc::new(|level, message| match level {
        Level::Info => devlog::info(Source::Build, message),
        Level::Error => devlog::error(Source::Build, message),
    });
    Ok(Plugins::load(root, log)?.filter(|plugins| plugins.compiles()))
}

/// Writes `bytes` to `path` in one rename, unless it holds them already;
/// whether it changed.
fn write(path: &Path, bytes: &[u8]) -> Result<bool> {
    if std::fs::read(path).is_ok_and(|old| old == bytes) {
        return Ok(false);
    }
    let name = path.file_name().map_or_else(|| "build".into(), |name| name.to_string_lossy().into_owned());
    let temporary = path.with_file_name(format!(".{name}.{}.tmp", std::process::id()));
    std::fs::write(&temporary, bytes)?;
    std::fs::rename(&temporary, path)?;
    Ok(true)
}

/// Renders a Solid-2-style diagnostic with canonical `[CODE]`, enclosing scope,
/// `at file:line:col`, multi-span source excerpt (primary `^` + secondary `-`
/// blame from JSON `details`), actionable `fix:`, and `doc:` pointer while
/// keeping the raw compiler `code` and `message` visible for test assertions.
pub fn describe(diagnostic: &Diagnostic, standing_in: Option<(&str, &str)>) -> String {
    let source = match standing_in {
        Some((path, text)) if path == diagnostic.file => Some(text.to_owned()),
        _ => std::fs::read_to_string(&diagnostic.file).ok(),
    };
    let canonical = canonical_code(&diagnostic.code);
    let parsed_details: Option<serde_json::Value> =
        diagnostic.details.as_deref().and_then(|raw| serde_json::from_str(raw).ok());

    let Some(source_text) = source else {
        let mut out = format!(
            "{}:@{}: [{canonical}] ({}) {}",
            diagnostic.file, diagnostic.start, diagnostic.code, diagnostic.message
        );
        if let Some(details) = &diagnostic.details {
            out.push('\n');
            out.push_str(details);
        }
        return out;
    };

    let bytes = source_text.as_bytes();
    let lines: Vec<&str> = source_text.split('\n').map(|line| line.strip_suffix('\r').unwrap_or(line)).collect();
    let line_starts = compute_line_starts(bytes);

    let (p_start, p_end) = expand_point_span(bytes, diagnostic.start as usize, diagnostic.end as usize);
    let (p_line, p_col) = byte_to_line_col(bytes, &line_starts, diagnostic.start as usize);
    let snippet = slice_utf8(bytes, p_start, p_end).trim().to_owned();
    let primary_line_text = lines.get(p_line.saturating_sub(1)).copied().unwrap_or("");

    let blame = extract_blame(parsed_details.as_ref());
    let subject_raw = if blame.subject_end > blame.subject_start {
        slice_utf8(bytes, blame.subject_start, blame.subject_end).to_owned()
    } else {
        String::new()
    };
    let subject_name = extract_binding_ident(&subject_raw);
    let holder_name = if blame.holder_end > blame.holder_start {
        extract_binding_ident(slice_utf8(bytes, blame.holder_start, blame.holder_end))
    } else {
        String::new()
    };
    let has_origin = !blame.origin_kind.is_empty() && blame.origin_end > blame.origin_start;
    let origin_line = if has_origin { Some(byte_to_line_col(bytes, &line_starts, blame.origin_start).0) } else { None };

    let (headline, primary_label, fix) = build_headline_and_fix(
        &canonical,
        diagnostic,
        &snippet,
        primary_line_text,
        &blame.origin_kind,
        &subject_name,
        &subject_raw,
        &holder_name,
        origin_line,
        parsed_details.as_ref(),
    );

    let mut out = Vec::new();
    out.push(format!("{}:{p_line}:{p_col}: [{canonical}] ({}) {headline}", diagnostic.file, diagnostic.code));

    let base_name = Path::new(&diagnostic.file).file_name().and_then(|name| name.to_str()).unwrap_or(&diagnostic.file);
    if let Some(scope) = find_enclosing_scope(&lines, p_line) {
        out.push(format!("in {base_name} › {scope}"));
    } else {
        out.push(format!("in {base_name}"));
    }

    let mut spans: Vec<(usize, usize, char, String)> = Vec::new();
    if has_origin {
        let label = match blame.origin_kind.as_str() {
            "moved" if !subject_name.is_empty() => format!("`{subject_name}` moved here"),
            "moved" => "moved here".to_owned(),
            "borrowed" if !holder_name.is_empty() => format!("borrowed by `{holder_name}` here"),
            "borrowed" => "borrowed here".to_owned(),
            "aliased" => "aliased here".to_owned(),
            "parameter" => "borrowed parameter (not `own`)".to_owned(),
            _ => "related here".to_owned(),
        };
        spans.push((blame.origin_start, blame.origin_end, '-', label));
    }
    spans.push((p_start, p_end, '^', primary_label));
    out.extend(format_excerpt(bytes, &lines, &line_starts, &spans));

    if let Some(fix_text) = fix {
        out.push(format!("fix: {fix_text}"));
    }
    out.push(format!("doc: blot explain {canonical} (skills/blot-diagnostics/SKILL.md#{canonical})"));
    out.join("\n")
}

#[derive(Default)]
struct BlameOrigin {
    origin_kind: String,
    origin_start: usize,
    origin_end: usize,
    subject_start: usize,
    subject_end: usize,
    holder_start: usize,
    holder_end: usize,
}

fn canonical_code(raw: &str) -> String {
    match raw {
        "GPU_FRONTEND_SYNTAX_ERROR" => "SYNTAX_ERROR".to_owned(),
        "GPU_FRONTEND_MALFORMED_DELIMITER" => "UNMATCHED_DELIMITER".to_owned(),
        "GPU_FRONTEND_LEXICAL_ERROR" => "LEXICAL_ERROR".to_owned(),
        "unknown_name" | "unknown_value" => "UNKNOWN_VALUE".to_owned(),
        other => other
            .strip_prefix("GPU_FRONTEND_")
            .unwrap_or(other)
            .chars()
            .map(|c| if c.is_ascii_alphanumeric() { c.to_ascii_uppercase() } else { '_' })
            .collect(),
    }
}

fn extract_blame(details: Option<&serde_json::Value>) -> BlameOrigin {
    let Some(obj) = details.and_then(serde_json::Value::as_object) else {
        return BlameOrigin::default();
    };
    let get_usize = |key: &str| obj.get(key).and_then(serde_json::Value::as_u64).and_then(|v| usize::try_from(v).ok()).unwrap_or(0);
    BlameOrigin {
        origin_kind: obj.get("origin_kind").and_then(serde_json::Value::as_str).unwrap_or("").to_owned(),
        origin_start: get_usize("origin_start"),
        origin_end: get_usize("origin_end"),
        subject_start: get_usize("subject_start"),
        subject_end: get_usize("subject_end"),
        holder_start: get_usize("holder_start"),
        holder_end: get_usize("holder_end"),
    }
}

fn compute_line_starts(bytes: &[u8]) -> Vec<usize> {
    let mut starts = vec![0];
    for (idx, &byte) in bytes.iter().enumerate() {
        if byte == b'\n' {
            starts.push(idx + 1);
        }
    }
    starts
}

fn byte_to_line_col(bytes: &[u8], line_starts: &[usize], offset: usize) -> (usize, usize) {
    let clamped = offset.min(bytes.len());
    let line_idx = line_starts.partition_point(|&start| start <= clamped).saturating_sub(1);
    let line_start = line_starts[line_idx];
    let col = String::from_utf8_lossy(&bytes[line_start..clamped]).chars().count() + 1;
    (line_idx + 1, col)
}

fn slice_utf8(bytes: &[u8], start: usize, end: usize) -> &str {
    let s = start.min(bytes.len());
    let e = end.max(s).min(bytes.len());
    std::str::from_utf8(&bytes[s..e]).unwrap_or("")
}

fn is_ident_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || matches!(b, b'_' | b'$' | b'@')
}

fn expand_point_span(bytes: &[u8], start: usize, end: usize) -> (usize, usize) {
    let s = start.min(bytes.len());
    let e = end.min(bytes.len());
    if e > s {
        return (s, e);
    }
    if s >= bytes.len() {
        return (s, s);
    }
    if is_ident_byte(bytes[s]) {
        let mut cur = s;
        while cur < bytes.len() {
            if is_ident_byte(bytes[cur])
                || (bytes[cur] == b'.' && cur + 1 < bytes.len() && is_ident_byte(bytes[cur + 1]))
            {
                cur += 1;
            } else {
                break;
            }
        }
        return (s, cur);
    }
    if bytes[s] != b'\n' && bytes[s] != b'\r' {
        return (s, (s + 1).min(bytes.len()));
    }
    (s, s)
}

fn find_enclosing_scope(lines: &[&str], line_1_based: usize) -> Option<String> {
    let prefixes = [
        "#entry func ",
        "#entry const ",
        "#entry let ",
        "#kernel func ",
        "#kernel const ",
        "#process const ",
        "private func ",
        "private const ",
        "func ",
        "const ",
        "type ",
    ];
    let mut idx = line_1_based.saturating_sub(1).min(lines.len().saturating_sub(1));
    loop {
        if let Some(&line) = lines.get(idx)
            && !line.starts_with(' ')
            && !line.starts_with('\t')
        {
            for prefix in prefixes {
                if let Some(rest) = line.strip_prefix(prefix) {
                    let name: String =
                        rest.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '.').collect();
                    if !name.is_empty() {
                        let kind = prefix.trim().strip_prefix("private ").unwrap_or(prefix.trim());
                        return Some(format!("{kind} {name}"));
                    }
                }
            }
        }
        if idx == 0 {
            break;
        }
        idx -= 1;
    }
    None
}

fn extract_binding_ident(raw: &str) -> String {
    let trimmed = raw.trim().trim_start_matches('(').trim_start();
    let trimmed = trimmed.strip_prefix("own ").map_or(trimmed, str::trim_start);
    let ident: String = trimmed.chars().take_while(|c| c.is_ascii_alphanumeric() || *c == '_').collect();
    if ident.is_empty() { raw.trim().to_owned() } else { ident }
}

fn format_own_param_suggestion(raw_subject: &str, name: &str) -> String {
    let trimmed = raw_subject.trim();
    if let Some(inner) = trimmed.strip_prefix('(').and_then(|s| s.strip_suffix(')'))
        && let Some((lhs, rhs)) = inner.split_once(':')
    {
        return format!("({}: own {})", lhs.trim(), rhs.trim());
    }
    format!("({name}: own #[T])")
}

fn format_excerpt(
    bytes: &[u8],
    lines: &[&str],
    line_starts: &[usize],
    spans: &[(usize, usize, char, String)],
) -> Vec<String> {
    if spans.is_empty() || lines.is_empty() {
        return Vec::new();
    }
    let mut located: Vec<(usize, usize, usize, char, &str)> = spans
        .iter()
        .map(|(s, e, marker, label)| {
            let (es, ee) = expand_point_span(bytes, *s, *e);
            let (start_line, start_col) = byte_to_line_col(bytes, line_starts, es);
            let (end_line, end_col) = byte_to_line_col(bytes, line_starts, ee.saturating_sub(1).max(es));
            let line_len = lines.get(start_line.saturating_sub(1)).map_or(0, |l| l.chars().count());
            let end_col_on_line = if end_line == start_line { end_col } else { line_len };
            let width = end_col_on_line.saturating_sub(start_col).saturating_add(1).max(1);
            (start_line, start_col, width, *marker, label.as_str())
        })
        .collect();
    located.sort_by_key(|&(line, col, _, marker, _)| (line, usize::from(marker == '^'), col));

    let mut unique_lines: Vec<usize> = located.iter().map(|&(l, ..)| l).collect();
    unique_lines.sort_unstable();
    unique_lines.dedup();
    let mut display_lines = Vec::new();
    for (idx, &ln) in unique_lines.iter().enumerate() {
        if idx > 0 {
            let prev = unique_lines[idx - 1];
            if ln.saturating_sub(prev) <= 3 {
                for between in (prev + 1)..ln {
                    display_lines.push(between);
                }
            }
        }
        display_lines.push(ln);
    }

    let max_line = display_lines.iter().copied().max().unwrap_or(1);
    let gutter_width = max_line.to_string().len().max(2);
    let blank = " ".repeat(gutter_width);
    let mut out = Vec::new();
    for (idx, &ln) in display_lines.iter().enumerate() {
        if idx > 0 && ln.saturating_sub(display_lines[idx - 1]) > 1 {
            out.push(format!("{blank} ..."));
        }
        let text = lines.get(ln.saturating_sub(1)).copied().unwrap_or("");
        out.push(format!("{ln:>gutter_width$} | {text}"));
        for &(span_line, col, width, marker, label) in &located {
            if span_line != ln {
                continue;
            }
            let pad = " ".repeat(col.saturating_sub(1));
            let carets: String = std::iter::repeat_n(marker, width).collect();
            if label.is_empty() {
                out.push(format!("{blank} | {pad}{carets}"));
            } else {
                out.push(format!("{blank} | {pad}{carets} {label}"));
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn build_headline_and_fix(
    canonical: &str,
    diagnostic: &Diagnostic,
    snippet: &str,
    primary_line: &str,
    origin_kind: &str,
    subject_name: &str,
    subject_raw: &str,
    holder_name: &str,
    origin_line: Option<usize>,
    details: Option<&serde_json::Value>,
) -> (String, String, Option<String>) {
    let trimmed_line = primary_line.trim();
    match canonical {
        "SYNTAX_ERROR" | "UNMATCHED_DELIMITER" => {
            let fix = if trimmed_line.starts_with("fn ") || snippet == "fn" {
                "Blot uses `func`, not `fn` (e.g. `func add (a: U32) -> (b: U32): U32 => return a + b`).".to_owned()
            } else if trimmed_line.starts_with("type ") && trimmed_line.contains('|') && !trimmed_line.contains("is data")
            {
                "Sum types require `is data` and `$`-prefixed constructors: `type Opt is data = $Some U32 | $Nothing`."
                    .to_owned()
            } else if trimmed_line.contains("&&") || trimmed_line.contains("||") {
                "Blot uses `and`, `or`, and `not` (not `&&`, `||`, `!`) for boolean operators.".to_owned()
            } else if trimmed_line.starts_with("mut ") || trimmed_line.starts_with("var ") {
                "Blot declares locals with `let x = ...` and updates with `x := ...` (no `mut`/`var` keyword)."
                    .to_owned()
            } else {
                "Check Blot declaration and indented suite syntax (`blot guide --cheatsheet`).".to_owned()
            };
            (diagnostic.message.clone(), "syntax error here".to_owned(), Some(fix))
        }
        "UNSUPPORTED_EXPRESSION" => {
            let is_bare_list =
                diagnostic.message.contains("`[...]`") || (snippet.starts_with('[') && !snippet.starts_with("#["));
            let fix = if is_bare_list {
                "Replace bare `[...]` with `#[...]` for Blot array literals (e.g. `#[1, 2, 3]`).".to_owned()
            } else {
                "Replace with a supported Blot vNext expression (`blot guide --cheatsheet`).".to_owned()
            };
            let label = if is_bare_list { "use `#[...]` here" } else { "unsupported expression" };
            (diagnostic.message.clone(), label.to_owned(), Some(fix))
        }
        "UNKNOWN_VALUE" => {
            if let Some(after) = diagnostic.message.split("did you mean `").nth(1)
                && let Some(suggested) = after.split('`').next()
            {
                let label = if suggested.starts_with('$') {
                    format!("add `$` -> `{suggested}`")
                } else {
                    format!("did you mean `{suggested}`?")
                };
                let fix = if suggested.starts_with('$') {
                    format!("Prefix the data constructor with `$`: use `{suggested}`.")
                } else {
                    format!("Replace `{snippet}` with `{suggested}`.")
                };
                return (diagnostic.message.clone(), label, Some(fix));
            }
            let fix = match snippet {
                "None" => "Blot's prelude option uses `$Nothing` and `$Some x` (not `None`/`Some`).",
                "String" | "str" => "Blot represents strings as byte arrays (`#[U8]` / `Array U8`).",
                _ => "Check spelling, prefix data constructors with `$`, or import the defining module.",
            };
            (diagnostic.message.clone(), "not in scope".to_owned(), Some(fix.to_owned()))
        }
        "MISSING_MEMBER" => {
            let exp = details.and_then(|v| v.get("explanation"));
            let member = exp
                .and_then(|e| e.get("member"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| snippet.rsplit('.').next().unwrap_or(""));
            let operand = exp
                .and_then(|e| e.get("candidates"))
                .and_then(serde_json::Value::as_array)
                .and_then(|arr| arr.first())
                .and_then(|c| c.get("operand"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or("");
            let headline = if !member.is_empty() && !operand.is_empty() {
                format!("Type `{operand}` has no field or member `.{member}`.")
            } else {
                diagnostic.message.clone()
            };
            let fix = if member == "len" {
                "Blot arrays use `.length` (not `.len`).".to_owned()
            } else if let Some(inner) = operand.trim().strip_prefix('{').and_then(|s| s.strip_suffix('}')) {
                let fields: Vec<String> = inner
                    .split(',')
                    .filter_map(|p| p.split(':').next().map(|n| format!("`{}`", n.trim())))
                    .filter(|s| s.len() > 2)
                    .collect();
                if fields.is_empty() {
                    "Check the receiver's fields or call a namespaced function directly.".to_owned()
                } else {
                    format!("Available fields on `{operand}`: {}.", fields.join(", "))
                }
            } else {
                "Check the receiver's fields or call a namespaced function directly.".to_owned()
            };
            let label = if member.is_empty() { "missing member".to_owned() } else { format!("unknown member `.{member}`") };
            (headline, label, Some(fix))
        }
        "SPENT_VALUE" => {
            let fallback = extract_binding_ident(snippet);
            let name = if subject_name.is_empty() { fallback.as_str() } else { subject_name };
            let copy_expr = if name.is_empty() { "value.copy".to_owned() } else { format!("{name}.copy") };
            let where_str = origin_line.map_or(String::new(), |l| format!(" at line {l}"));
            let headline = format!("`{name}` was already moved{where_str} and can no longer be used here.");
            let fix = format!(
                "Pass `{copy_expr}`{where_str} to keep `{name}` live, or use the owned value returned by that call."
            );
            (headline, format!("`{name}` used after move"), Some(fix))
        }
        "ALIASED_UPDATE" => {
            let fallback = extract_binding_ident(snippet);
            let name = if subject_name.is_empty() { fallback.as_str() } else { subject_name };
            let copy_expr = if name.is_empty() { "value.copy".to_owned() } else { format!("{name}.copy") };
            if origin_kind == "borrowed" {
                let holder = if holder_name.is_empty() { "another binding" } else { holder_name };
                let where_str = origin_line.map_or(String::new(), |l| format!(" (line {l})"));
                let headline = format!("Cannot mutate `{name}` in place while `{holder}` still borrows it.");
                let fix = if holder_name.is_empty() {
                    format!("Take `{copy_expr}` before creating the alias, or end the borrow before mutating `{name}`.")
                } else {
                    format!("Snapshot with `let {holder_name} = {copy_expr}`{where_str}, or finish reading `{holder_name}` before mutating `{name}`.")
                };
                return (headline, "mutated while borrowed".to_owned(), Some(fix));
            }
            if origin_kind == "parameter" {
                let own_param = format_own_param_suggestion(subject_raw, name);
                let headline = format!("Cannot mutate borrowed parameter `{name}` in place without `own`.");
                let fix = format!(
                    "Mark the parameter `own` (e.g. `{own_param}`) to mutate in place, or copy locally with `let local = {copy_expr}`."
                );
                return (headline, "mutates borrowed parameter".to_owned(), Some(fix));
            }
            (
                format!("Cannot mutate `{name}` in place because another alias may still observe it."),
                "aliased in-place write".to_owned(),
                Some(format!("Take an explicit copy (`let local = {copy_expr}`) before updating with `:=`, or remove the earlier alias.")),
            )
        }
        "OWN_ARGUMENT" => {
            let fallback = extract_binding_ident(snippet);
            let name = if subject_name.is_empty() { fallback.as_str() } else { subject_name };
            let copy_expr = if name.is_empty() { "value.copy".to_owned() } else { format!("{name}.copy") };
            let fix = if origin_kind == "parameter" {
                format!("Mark parameter `{name}` as `own` to forward ownership, or pass `{copy_expr}`.")
            } else {
                format!("Pass `{copy_expr}` to give the callee a fresh owned copy, or end live borrows of `{name}` first.")
            };
            (
                format!("Argument `{name}` is not uniquely owned and cannot be moved into an `own` parameter."),
                "not uniquely owned".to_owned(),
                Some(fix),
            )
        }
        "RESULT_NOT_OWNED" => {
            let fallback = extract_binding_ident(snippet);
            let name = if subject_name.is_empty() { fallback.as_str() } else { subject_name };
            let copy_expr = if name.is_empty() { "value.copy".to_owned() } else { format!("{name}.copy") };
            (
                format!("Function result `{name}` is borrowed or aliased; returned storage must be uniquely owned."),
                "borrowed result cannot escape".to_owned(),
                Some(format!("Return `{copy_expr}` to hand the caller a fresh copy, or take the source parameter with `own`.")),
            )
        }
        "ENTRY_TYPE" => {
            if let Some(after) = diagnostic.message.split("(unhandled: ").nth(1)
                && let Some(op) = after.split(')').next()
            {
                let (family, member) = op.split_once('.').unwrap_or((op, "op"));
                return (
                    format!("Entry function leaves effect `{op}` unhandled across the Wasm boundary."),
                    format!("unhandled effect `{op}`"),
                    Some(format!(
                        "Handle `{op}` inside the entry function with `with {family} = {{ {member}: func ... => ... }}:` or `do <provider>:`."
                    )),
                );
            }
            (
                diagnostic.message.clone(),
                "invalid entry signature".to_owned(),
                Some("Ensure entry functions handle all effects and use ABI-8 boundary types.".to_owned()),
            )
        }
        "TYPE_MISMATCH" => {
            let fix = if diagnostic.message.contains("F32") && diagnostic.message.contains("U32") {
                "Blot has no implicit numeric conversions; match the literal (`1` vs `1.0`) or convert explicitly with `@u32.to_f32` / `@f32.to_u32`."
            } else {
                "Align the expression type with the declared annotation or expected parameter type."
            };
            (diagnostic.message.clone(), "type mismatch here".to_owned(), Some(fix.to_owned()))
        }
        _ => (diagnostic.message.clone(), canonical.to_ascii_lowercase().replace('_', " "), None),
    }
}

/// Says what a build of `what` came to, and writes its Wasm to `out`;
/// whether it built.
fn take(what: &str, out: &Path, outcome: Outcome) -> bool {
    match outcome {
        Outcome::Built(built) => match write(out, &built.wasm) {
            Ok(changed) => {
                let unchanged = if changed { "" } else { ", unchanged" };
                devlog::info(Source::Build, format!("plugins: {what} compiled ({:.2} s{unchanged})", built.seconds));
                true
            }
            Err(error) => {
                devlog::error(Source::Build, format!("plugins: writing {}: {error:#}", out.display()));
                false
            }
        },
        Outcome::Rejected(diagnostics) => {
            devlog::error(Source::Build, format!("plugins: {what} build failed"));
            for diagnostic in &diagnostics {
                devlog::error(Source::Build, describe(diagnostic, None));
            }
            false
        }
        Outcome::Failed(error) => {
            devlog::error(Source::Build, format!("plugins: {what} build failed: {error}"));
            false
        }
    }
}

/// Builds `what`, the program at `entry`, into `out` once.
pub fn compile(plugins: Arc<Plugins>, what: &str, entry: &Path, out: &Path) -> Result<()> {
    if !take(what, out, pen_plugins::build(plugins, entry)) {
        bail!("the {what} did not compile");
    }
    Ok(())
}

/// Builds `what` into `out`, and again as its sources change, until the
/// result is dropped; the receiver hears whether each build built. The
/// sources are watched through the session's one file watcher
/// (`dev::watch`), as each build says which they are.
pub fn watch(
    plugins: Arc<Plugins>,
    what: &str,
    entry: &Path,
    out: &Path,
    ignored: Vec<PathBuf>,
) -> Result<(pen_plugins::Watching, Receiver<bool>)> {
    let (sender, built) = channel();
    let (changed, changes) = channel();
    let subscription = watch::subscribe(&[], move |event| {
        if let Ok(event) = event
            && !matches!(event.kind, notify::EventKind::Access(_))
        {
            let _ = changed.send(event.paths);
        }
    })?;
    let (what, out) = (what.to_owned(), out.to_owned());
    let watching = pen_plugins::watch_changes(plugins, entry, ignored, changes, move |outcome, sources| {
        // Before the outcome is told: an edit made on seeing it must count.
        if let Err(error) = subscription.extend(&roots(sources)) {
            devlog::error(Source::Build, format!("plugins: watching the {what}'s sources: {error:#}"));
        }
        let _ = sender.send(take(&what, &out, outcome));
    })?;
    Ok((watching, built))
}

/// What to watch for a build's sources: a directory and everything under
/// it, or a file through its directory (editors replace files they save,
/// and a file may not be there yet).
fn roots(sources: &BTreeSet<PathBuf>) -> Vec<(&Path, bool)> {
    let mut roots: Vec<(&Path, bool)> = Vec::new();
    for source in sources {
        let root = if source.is_dir() {
            Some((source.as_path(), true))
        } else {
            source.parent().filter(|parent| parent.is_dir()).map(|parent| (parent, false))
        };
        if let Some(root) = root.filter(|root| !roots.contains(root)) {
            roots.push(root);
        }
    }
    roots
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn watches_directories_whole_and_files_through_theirs() {
        let directory = std::env::temp_dir().join(format!("pen-plugin-roots-{}", std::process::id()));
        std::fs::create_dir_all(directory.join("src/lib")).unwrap();
        let sources: BTreeSet<PathBuf> = [
            directory.join("src/lib"),
            directory.join("src/main.blot"),
            directory.join("src/later.blot"),
            directory.join("gone/away.blot"),
        ]
        .into();
        let src = directory.join("src");
        let lib = directory.join("src/lib");
        assert_eq!(roots(&sources), [(src.as_path(), false), (lib.as_path(), true)]);
        let _ = std::fs::remove_dir_all(&directory);
    }

    #[test]
    fn shows_worker_declaration_names_in_diagnostics() {
        for (declaration, expected) in [
            ("#kernel func double (value: U32): U32 =>", "#kernel func double"),
            ("#kernel const plan_routes: Kernel [F32, F32, F32] = {", "#kernel const plan_routes"),
            ("#process const sim: Process [U32, U32, U32] = {", "#process const sim"),
        ] {
            let lines = ["func previous (): U32 => return 0", declaration, "  missing_name"];
            assert_eq!(find_enclosing_scope(&lines, 3).as_deref(), Some(expected));
        }
    }

    #[test]
    fn describes_spent_value_with_secondary_move_blame_and_copy_fix() {
        let source = "func consume (xs: own #[U32]): U32 => return xs.length\n#entry func main (): U32 =>\n  let a = #[1, 2, 3]\n  let n = consume a\n  return n + a.length\n";
        let move_start = u32::try_from(source.find("consume a").unwrap()).unwrap();
        let move_end = move_start + u32::try_from("consume a".len()).unwrap();
        let subj_start = u32::try_from(source.find("a =").unwrap()).unwrap();
        let subj_end = subj_start + 1;
        let use_start = u32::try_from(source.find("a.length").unwrap()).unwrap();
        let use_end = use_start + u32::try_from("a.length".len()).unwrap();
        let diagnostic = Diagnostic {
            file: "/tmp/spent.blot".to_owned(),
            stage: "lower".to_owned(),
            code: "spent_value".to_owned(),
            start: use_start,
            end: use_end,
            message: "This value was given away".to_owned(),
            details: Some(format!(
                "{{\"origin_kind\":\"moved\",\"origin_start\":{move_start},\"origin_end\":{move_end},\"subject_start\":{subj_start},\"subject_end\":{subj_end},\"holder_start\":0,\"holder_end\":0}}"
            )),
        };
        let rendered = describe(&diagnostic, Some(("/tmp/spent.blot", source)));
        assert!(rendered.starts_with("/tmp/spent.blot:5:14: [SPENT_VALUE] (spent_value)"), "{rendered}");
        assert!(rendered.contains("in spent.blot › #entry func main"), "{rendered}");
        assert!(rendered.contains("--------- `a` moved here"), "{rendered}");
        assert!(rendered.contains("^^^^^^^^ `a` used after move"), "{rendered}");
        assert!(rendered.contains("fix: Pass `a.copy` at line 4"), "{rendered}");
        assert!(rendered.contains("doc: blot explain SPENT_VALUE"), "{rendered}");
    }
}
