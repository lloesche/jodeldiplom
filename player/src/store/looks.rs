//! The looks kept on the Shader tab ([`crate::core::shader_tab`]): the
//! user's own presets, in `looks` in the data folder
//! ([`crate::store::dirs::Data::looks`]), a file each named after its id
//! (`my-noir.json`), as JSON ([`jodel_looks::Look::to_json`]), loaded at the
//! start ([`load_all`]). Written whole or not at all ([`durable`]), where
//! the data folder is now (it may be moving: [`crate::store::mover`]), on a
//! thread of their own, one at a time and in order ([`LookStore`]).

use std::fmt::Write as _;
use std::io;
use std::path::{Path, PathBuf};

use jodel_looks::Look;

use crate::store::writer::Writer;
use crate::work::outbox::Outbox;

/// The file of the look `id` in `dir`: the id as it is where it is made of
/// lowercase letters, digits, `-` and `_`; every other byte of it as `%`
/// and its two lowercase hex digits (`My/A` as `%4dy%2f%41`). So two ids
/// never share a file, also where the file system takes `A` for `a`.
fn file(dir: &Path, id: &str) -> PathBuf {
    let mut name = String::with_capacity(id.len());
    for b in id.bytes() {
        if b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_' {
            name.push(char::from(b));
        } else {
            let _ = write!(name, "%{b:02x}");
        }
    }
    dir.join(format!("{name}.json"))
}

/// Every look kept in `dir` (where each is now: while the data folder
/// moves, one not moved yet is where it was), by id; a file that does not
/// read as a look is passed over. Where two files hold the same id (one
/// kept under an older naming), the one at the id's file now ([`file`],
/// the one [`save`] writes) is it, else the first by name.
pub fn load_all(dir: &Path) -> Vec<Look> {
    let mut names: Vec<PathBuf> = Vec::new();
    for at in [crate::store::mover::current(dir), dir.to_owned()] {
        if let Ok(entries) = std::fs::read_dir(&at) {
            for e in entries.flatten() {
                let p = e.path();
                let name = p.file_name().map(|n| dir.join(n));
                if p.extension().is_some_and(|x| x == "json") {
                    names.extend(name.filter(|n| !names.contains(n)));
                }
            }
        }
    }
    names.sort();
    let mut looks: Vec<(bool, Look)> = (names.iter())
        .filter_map(|n| {
            let bytes = crate::store::mover::read(n).ok()?;
            let look = Look::from_json(&String::from_utf8(bytes).ok()?).ok()?;
            Some((*n != file(dir, &look.id), look))
        })
        .collect();
    looks.sort_by(|(a_old, a), (b_old, b)| a.id.cmp(&b.id).then(a_old.cmp(b_old)));
    looks.dedup_by(|(_, a), (_, b)| a.id == b.id);
    looks.into_iter().map(|(_, l)| l).collect()
}

/// Keeps `look` in `dir`, whole or not at all.
pub fn save(dir: &Path, look: &Look) -> io::Result<()> {
    let file = file(dir, &look.id);
    durable::write_atomic(
        &crate::store::mover::current(&file),
        look.to_json().as_bytes(),
    )
}

/// Keeps looks in a folder, on a thread of its own: in the order asked,
/// one at a time ([`Writer`]). A failure is said with
/// [`crate::core::Action::Failed`].
pub struct LookStore {
    writer: Writer<Look>,
}

impl LookStore {
    /// A store in `dir`, saying failures to `outbox`.
    pub fn new(dir: PathBuf, outbox: Outbox) -> LookStore {
        let writer = Writer::new(outbox, move |look: Look| {
            save(&dir, &look).map_err(|e| format!("The look {} was not kept: {e}", look.id))
        });
        LookStore { writer }
    }

    /// Keeps `look`.
    pub fn keep(&self, look: Look) {
        self.writer.send(look);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A look kept is found again as it was, also while the data folder
    /// moves; kept again under its id, it replaces the one before.
    #[test]
    fn looks_are_kept_and_found_while_the_data_folder_moves() {
        let dir = jodel_testkit::TempDir::new("looks");
        let (old, new) = (dir.join("old/looks"), dir.join("new/looks"));
        assert!(load_all(&old).is_empty());
        let mut look = jodel_looks::presets::all()[0].clone();
        look.id = "my-first/one".into();
        save(&old, &look).unwrap();
        assert_eq!(load_all(&old), vec![look.clone()]);
        let settings = dir.join("settings");
        crate::store::mover::begin(&settings, &[(dir.join("old"), dir.join("new"))]).unwrap();
        assert_eq!(load_all(&old), vec![look.clone()]);
        look.layers.pop();
        save(&old, &look).unwrap();
        crate::store::mover::finish(&settings);
        assert_eq!(load_all(&new), vec![look]);
    }

    /// A look also kept under another file's name (an older naming of its
    /// id) is the one at its id's file now, whatever order the folder
    /// lists them in.
    #[test]
    fn of_two_files_with_one_id_the_one_at_its_file_is_it() {
        let dir = jodel_testkit::TempDir::new("looks-twice");
        let mut look = jodel_looks::presets::all()[0].clone();
        look.id = "my-Twice".into();
        save(&dir, &look).unwrap();
        let mut stale = look.clone();
        stale.layers.pop();
        for name in ["a-my-Twice.json", "my-Twice.json", "z.json"] {
            durable::write_atomic(&dir.join(name), stale.to_json().as_bytes()).unwrap();
        }
        assert_eq!(load_all(&dir), vec![look]);
    }

    /// Two ids never share a file, not even on a file system that takes
    /// `A` for `a`; an id of lowercase letters, digits, `-` and `_` is its
    /// file's name as it is.
    #[test]
    fn distinct_looks_are_kept_in_distinct_files() {
        let dir = Path::new("looks");
        assert_eq!(file(dir, "my-noir_2"), dir.join("my-noir_2.json"));
        let ids = [
            "my-a/b",
            "my-a_b",
            "my-a%2fb",
            "my-a.b",
            "my-a b",
            "My-A",
            "my-a",
            "MY-A",
            "my-\u{e9}",
            "my-e",
            "my-ab",
            "my-a_",
        ];
        let names: Vec<String> = ids
            .iter()
            .map(|id| file(dir, id).to_string_lossy().to_lowercase())
            .collect();
        for (i, a) in names.iter().enumerate() {
            for (j, b) in names.iter().enumerate() {
                assert!(i == j || a != b, "{} and {} share {a}", ids[i], ids[j]);
            }
        }
    }
}
