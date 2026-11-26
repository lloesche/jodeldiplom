//! `INTERACTIONS.md` names the tests that hold each rule: every name it
//! gives is a function somewhere in the workspace, so a test renamed or
//! taken out without the document is caught.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// The names in back-ticks that read as a test's: lower-case words joined
/// by underscores, three of them at least (a setting such as
/// `karaoke_echo` is two).
fn cited(document: &str) -> BTreeSet<&str> {
    document
        .split('`')
        .skip(1)
        .step_by(2)
        .filter(|name| {
            name.split('_').count() >= 3
                && name.split('_').all(|w| {
                    !w.is_empty()
                        && w.bytes()
                            .all(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
                })
        })
        .collect()
}

/// Every `.rs` file under `dir`, past the build's output and the hidden
/// folders.
fn sources(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = entry.file_name();
        let name = name.to_string_lossy();
        if path.is_dir() {
            if name != "target" && !name.starts_with('.') {
                sources(&path, out);
            }
        } else if name.ends_with(".rs") {
            out.push(path);
        }
    }
}

/// The functions defined in `source`: the word after each `fn `.
fn functions(source: &str, into: &mut BTreeSet<String>) {
    for (at, _) in source.match_indices("fn ") {
        let before = source[..at].chars().next_back();
        if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
            continue;
        }
        let name: String = source[at + 3..]
            .chars()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            into.insert(name);
        }
    }
}

#[test]
fn every_test_the_rules_name_exists() {
    let player = Path::new(env!("CARGO_MANIFEST_DIR"));
    let document = std::fs::read_to_string(player.join("INTERACTIONS.md")).unwrap();
    let names = cited(&document);
    assert!(names.len() > 100, "only {} names found", names.len());
    let mut files = Vec::new();
    sources(player.parent().unwrap(), &mut files);
    let mut defined = BTreeSet::new();
    for file in &files {
        functions(&std::fs::read_to_string(file).unwrap(), &mut defined);
    }
    let missing: Vec<_> = names
        .iter()
        .filter(|name| !defined.contains(**name))
        .collect();
    assert!(
        missing.is_empty(),
        "INTERACTIONS.md names tests that are not in the workspace: {missing:#?}"
    );
}
