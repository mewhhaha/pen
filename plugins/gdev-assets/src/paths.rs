//! Absolute paths as Node's `path` treats them on POSIX, by their text alone:
//! the plugin cannot resolve symbolic links, which assets.ts did.

/// `path` from `base` (absolute), with `.` and `..` gone and no trailing
/// slash.
pub(crate) fn resolve(base: &str, path: &str) -> String {
    let joined = if path.starts_with('/') { path.to_string() } else { format!("{base}/{path}") };
    let mut segments: Vec<&str> = Vec::new();
    for segment in joined.split('/') {
        match segment {
            "" | "." => {}
            ".." => {
                segments.pop();
            }
            segment => segments.push(segment),
        }
    }
    format!("/{}", segments.join("/"))
}

/// The directory `path` (resolved) is in; the root's is the root.
pub(crate) fn dirname(path: &str) -> String {
    match path.rfind('/') {
        Some(0) | None => "/".to_string(),
        Some(index) => path[..index].to_string(),
    }
}

/// `to` as a path from `from` (both resolved): `text.json`,
/// `../src/text.json`, or empty when they are the same.
pub(crate) fn relative(from: &str, to: &str) -> String {
    let from: Vec<&str> = from.split('/').filter(|segment| !segment.is_empty()).collect();
    let to: Vec<&str> = to.split('/').filter(|segment| !segment.is_empty()).collect();
    let shared = from.iter().zip(&to).take_while(|(left, right)| left == right).count();
    let mut path: Vec<&str> = vec![".."; from.len() - shared];
    path.extend(&to[shared..]);
    path.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_as_node_does() {
        assert_eq!(resolve("/a/b", "./c.json"), "/a/b/c.json");
        assert_eq!(resolve("/a/b", "../src/c.json"), "/a/src/c.json");
        assert_eq!(resolve("/a/b", "/x/./y/../z/"), "/x/z");
        assert_eq!(resolve("/", "../.."), "/");
        assert_eq!(dirname("/a/b/main.blot"), "/a/b");
        assert_eq!(dirname("/a"), "/");
        assert_eq!(dirname("/"), "/");
        assert_eq!(relative("/a/devtools", "/a/src/text.json"), "../src/text.json");
        assert_eq!(relative("/a/src", "/a/src/assets/x.wgsl"), "assets/x.wgsl");
        assert_eq!(relative("/a/src", "/a/src"), "");
        assert_eq!(relative("/", "/a"), "a");
    }
}
