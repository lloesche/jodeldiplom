//! The game's sounds' patches as kept on the Synth tab
//! ([`crate::core::synth_tab`]): the user's own work, in `patches` in the
//! data folder ([`crate::store::dirs::Data::patches`]), loaded over the
//! built-in palette at the start ([`load_all`]).
//!
//! A file a theme's atom, named after both (`classic-PopBody.txt`), in
//! [`cache::PATCH`]'s schema:
//!
//! ```text
//! jodeldiplom-cache patch 1 by jodeldiplom 0.1.0
//! theme classic
//! atom Pop
//! engine analog
//! envelope attack = 0
//! envelope decay = 0.09
//! osc wave = sine
//! osc 2 sync = off
//! ```
//!
//! `theme` and `atom` say whose it is; the rest is the patch as text
//! ([`Patch::to_text`]: `engine` its engine, then each parameter of the
//! patch that applies, its own then its engine's, in their tables' order,
//! as `KEY = VALUE`: the key its group and name, a number in its unit, a
//! choice by its option's name, a switch `on` or `off`). It is read over
//! the theme's built-in patch of the atom ([`Patch::from_text`]; its
//! engine's defaults, for another engine): a parameter left out, or one
//! this player does not know, is the built-in one's.
//!
//! The files are written whole or not at all ([`durable`]), where the
//! data folder is now (it may be moving: [`crate::store::mover`]), on a thread
//! of their own, one at a time and in order ([`PatchStore`]).

use std::io;
use std::path::{Path, PathBuf};

use jodel_synth::Patch;

use crate::core::patch_book::PatchBook;
use crate::sfx::conductor::Atom;
use crate::store::cache;
use crate::store::writer::Writer;
use crate::theme::{Theme, THEMES};
use crate::work::outbox::Outbox;

/// The file of `theme`'s `atom` in `dir`.
fn file(dir: &Path, theme: &str, atom: Atom) -> PathBuf {
    dir.join(format!("{theme}-{}.txt", atom.name()))
}

/// `patch` as kept for `theme`'s `atom`, in [`cache::PATCH`]'s version.
pub fn to_text(theme: &str, atom: Atom, patch: &Patch) -> String {
    let head = format!("theme {theme}\natom {}\n", atom.name());
    cache::PATCH.wrap(&(head + &patch.to_text()))
}

/// The patch in a file's `text` and whose it is (a theme and an atom this
/// player knows), read over that theme's built-in patch of the atom; in
/// any version of [`cache::PATCH`] up to this player's (the user's work is
/// kept).
pub fn from_text(text: &str) -> Option<(&'static Theme, Atom, Patch)> {
    let (version, body) = cache::PATCH.open(text)?;
    if version > cache::PATCH.version {
        return None;
    }
    let (theme, body) = body.split_once('\n')?;
    let (atom, patch) = body.split_once('\n')?;
    let theme = Theme::named(theme.strip_prefix("theme ")?)?;
    let atom = Atom::named(atom.strip_prefix("atom ")?)?;
    let patch = Patch::from_text(theme.patch(atom), patch)?;
    Some((theme, atom, patch))
}

/// Every patch kept in `dir` (where each is now: while the data folder
/// moves, one not moved yet is where it was), a book of them as kept.
pub fn load_all(dir: &Path) -> PatchBook {
    let mut kept = Vec::new();
    for theme in THEMES {
        for atom in Atom::ALL {
            let Ok(bytes) = crate::store::mover::read(&file(dir, theme.name, atom)) else {
                continue;
            };
            let read = String::from_utf8(bytes).ok();
            let Some((t, a, patch)) = read.as_deref().and_then(from_text) else {
                continue;
            };
            if t == theme && a == atom {
                kept.push((theme, atom, patch));
            }
        }
    }
    PatchBook::of_kept(kept)
}

/// Keeps `patch` as `theme`'s `atom`'s in `dir`, whole or not at all;
/// `None` forgets it (the built-in one again).
pub fn save(dir: &Path, theme: &str, atom: Atom, patch: Option<&Patch>) -> io::Result<()> {
    let file = file(dir, theme, atom);
    match patch {
        Some(p) => durable::write_atomic(
            &crate::store::mover::current(&file),
            to_text(theme, atom, p).as_bytes(),
        ),
        None => crate::store::mover::remove(&file),
    }
}

/// What the store's thread does next: a patch kept, or forgotten.
type Job = (&'static Theme, Atom, Option<Patch>);

/// Keeps and forgets patches in a folder, on a thread of its own: in the
/// order asked, one at a time ([`Writer`]). A failure is said with
/// [`crate::core::Action::Failed`].
pub struct PatchStore {
    writer: Writer<Job>,
}

impl PatchStore {
    /// A store in `dir`, saying failures to `outbox`.
    pub fn new(dir: PathBuf, outbox: Outbox) -> PatchStore {
        let writer = Writer::new(outbox, move |(theme, atom, patch): Job| {
            save(&dir, theme.name, atom, patch.as_ref()).map_err(|e| {
                let what = if patch.is_some() { "kept" } else { "forgotten" };
                format!("The patch of {} was not {what}: {e}", atom.name())
            })
        });
        PatchStore { writer }
    }

    /// Keeps `patch` as `theme`'s `atom`'s (`None`: forgets it).
    pub fn keep(&self, theme: &'static Theme, atom: Atom, patch: Option<Patch>) {
        self.writer.send((theme, atom, patch));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::{CLASSIC, SPOOKY};

    /// A patch edited: its envelope, a parameter of its engine, a choice,
    /// a switch; and one moved to another engine.
    fn edited() -> [(&'static Theme, Atom, Patch); 2] {
        let mut pop = CLASSIC.patch(Atom::Pop);
        pop.adsr.decay = 0.4;
        pop.reverb = 0.31;
        pop.mono = true;
        pop.slide = 0.05;
        let mut chime = SPOOKY.patch(Atom::Chime);
        chime.engine = jodel_synth::engines::AnyParams::of_engine("lofi").unwrap();
        let wave = chime.find("chip wave").unwrap();
        chime.set(wave.k, 5.0);
        [(&CLASSIC, Atom::Pop, pop), (&SPOOKY, Atom::Chime, chime)]
    }

    #[test]
    fn a_patch_reads_back_as_it_was_written() {
        for (theme, atom, patch) in edited() {
            let text = to_text(theme.name, atom, &patch);
            assert_eq!(from_text(&text), Some((theme, atom, patch)), "{text}");
        }
        // Built in, it reads back built in; a key it does not know, a value
        // out of its range, are let be.
        let built = CLASSIC.patch(Atom::Glint);
        let text = to_text("classic", Atom::Glint, &built);
        assert_eq!(from_text(&text).map(|t| t.2), Some(built));
        let odd = text.replace("sends reverb = 0.15", "sends reverb = 7\nsparkle = 3");
        let read = from_text(&odd).unwrap().2;
        assert_eq!(read.reverb, 1.0);
        // Another theme, another atom, a later version: none.
        assert_eq!(
            from_text(&text.replace("theme classic", "theme fabulous")),
            None
        );
        assert_eq!(from_text(&text.replace("atom Glint", "atom Kazoo")), None);
        let later = text.replace("patch 1 by", "patch 9 by");
        assert_eq!(from_text(&later), None);
    }

    /// Kept patches are found again, also while the data folder moves
    /// (where they still are until they come over); new ones go where the
    /// folder goes, and forgotten ones are gone.
    #[test]
    fn patches_are_kept_and_found_while_the_data_folder_moves() {
        let dir = std::env::temp_dir().join(format!("jodel-patches-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let (old, new) = (dir.join("old/patches"), dir.join("new/patches"));
        assert!(load_all(&old).is_empty());
        let [(t, a, pop), (u, b, chime)] = edited();
        save(&old, t.name, a, Some(&pop)).unwrap();
        save(&old, u.name, b, Some(&chime)).unwrap();
        let found = |at: &Path| -> Vec<(&str, Atom, Patch)> {
            let book = load_all(at);
            let mut kept = Vec::new();
            for theme in THEMES {
                for atom in Atom::ALL {
                    if let Some(patch) = book.kept(theme, atom) {
                        assert_eq!(book.patch(theme, atom), patch);
                        kept.push((theme.name, atom, patch));
                    }
                }
            }
            kept
        };
        let both = vec![("classic", a, pop), ("spooky", b, chime)];
        assert_eq!(found(&old), both);
        let settings = dir.join("settings");
        let (from, to) = (dir.join("old"), dir.join("new"));
        crate::store::mover::begin(&settings, &[(from, to)]).unwrap();
        assert_eq!(found(&old), both);
        crate::store::mover::finish(&settings);
        assert_eq!(found(&new), both);
        // Forgotten where it is now: gone from both places.
        save(&old, t.name, a, None).unwrap();
        assert_eq!(found(&new), vec![("spooky", b, chime)]);
        assert_eq!(found(&old), vec![("spooky", b, chime)]);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
