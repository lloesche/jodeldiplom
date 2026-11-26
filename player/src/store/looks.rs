//! The looks kept on the Shader tab ([`crate::core::shader_tab`]): the
//! user's own presets, in `looks` in the data folder
//! ([`crate::store::dirs::Data::looks`]), a file each named after its id
//! (`my-noir.json`), as JSON ([`jodel_looks::Look::to_json`]), loaded at the
//! start ([`load_all`]). Written whole or not at all ([`durable`]), where
//! the data folder is now (it may be moving: [`crate::store::mover`]), on a
//! thread of their own, one at a time and in order ([`LookStore`]).

use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Sender};
use std::thread;

use jodel_looks::Look;

use crate::core::Action;
use crate::work::outbox::Outbox;

/// The file of the look `id` in `dir` (an id's characters other than
/// letters, digits, `-` and `_` as `_`).
fn file(dir: &Path, id: &str) -> PathBuf {
    let name: String = id
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect();
    dir.join(format!("{name}.json"))
}

/// Every look kept in `dir` (where each is now: while the data folder
/// moves, one not moved yet is where it was), by id; a file that does not
/// read as a look is passed over.
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
    let mut looks: Vec<Look> = names
        .iter()
        .filter_map(|n| crate::store::mover::read(n).ok())
        .filter_map(|b| Look::from_json(&String::from_utf8(b).ok()?).ok())
        .collect();
    looks.sort_by(|a, b| a.id.cmp(&b.id));
    looks.dedup_by(|a, b| a.id == b.id);
    looks
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
/// one at a time. A failure is said with [`Action::Failed`].
pub struct LookStore {
    jobs: Sender<Look>,
}

impl LookStore {
    /// A store in `dir`, saying failures to `outbox`.
    pub fn new(dir: PathBuf, outbox: Outbox) -> LookStore {
        let (jobs, rx) = mpsc::channel::<Look>();
        thread::spawn(move || {
            for look in rx {
                if let Err(e) = save(&dir, &look) {
                    let text = format!("The look {} was not kept: {e}", look.id);
                    if !outbox.send(Action::Failed(text)) {
                        return;
                    }
                }
            }
        });
        LookStore { jobs }
    }

    /// Keeps `look`.
    pub fn keep(&self, look: Look) {
        let _ = self.jobs.send(look);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A look kept is found again as it was, also while the data folder
    /// moves; kept again under its id, it replaces the one before.
    #[test]
    fn looks_are_kept_and_found_while_the_data_folder_moves() {
        let dir = std::env::temp_dir().join(format!("jodel-looks-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
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
        let _ = std::fs::remove_dir_all(&dir);
    }
}
