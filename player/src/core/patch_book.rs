//! The game's sounds' patches as edited on the Synth tab and kept in the
//! data folder ([`PatchBook`]): by a theme and an atom, each the theme's
//! built-in one unless it is edited or kept. The tab edits, keeps and
//! reverts them ([`crate::core::synth_tab`]), the store loads and writes
//! what is kept ([`crate::store::patches`]), and the effects play every
//! theme's as the book has them ([`crate::sfx`]: its revision says when
//! one changed).

use std::sync::atomic::{AtomicU64, Ordering};

use jodel_synth::Patch;

use crate::sfx::conductor::Atom;
use crate::theme::Theme;

/// Where an atom's patch stands.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PatchStatus {
    /// The theme's own.
    AsBuilt,
    /// Edited, nothing kept.
    Edited,
    /// Kept, as it is.
    Kept,
    /// Kept, and edited since.
    KeptEdited,
}

impl PatchStatus {
    /// It said.
    pub fn name(self) -> &'static str {
        match self {
            PatchStatus::AsBuilt => "as built",
            PatchStatus::Edited => "edited",
            PatchStatus::Kept => "kept",
            PatchStatus::KeptEdited => "kept, edited since",
        }
    }
}

/// A patch off the theme's own: whose, what it is now, and what is kept
/// of it (none: nothing, the built-in one at the start).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Page {
    theme: &'static Theme,
    atom: Atom,
    patch: Patch,
    kept: Option<Patch>,
}

/// The patches edited or kept, by theme and atom.
#[derive(Clone, Debug, PartialEq)]
pub struct PatchBook {
    pages: Vec<Page>,
    /// Its revision: new at every change, never one another book had.
    revision: u64,
}

/// A revision no book has had before (of all the player's books).
fn next_revision() -> u64 {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

impl Default for PatchBook {
    /// None edited or kept.
    fn default() -> PatchBook {
        PatchBook {
            pages: Vec::new(),
            revision: next_revision(),
        }
    }
}

impl PatchBook {
    /// A book of patches kept (each a theme's atom's, as the data folder
    /// has it).
    pub fn of_kept(kept: impl IntoIterator<Item = (&'static Theme, Atom, Patch)>) -> PatchBook {
        PatchBook {
            pages: (kept.into_iter())
                .map(|(theme, atom, patch)| Page {
                    theme,
                    atom,
                    patch,
                    kept: Some(patch),
                })
                .collect(),
            revision: next_revision(),
        }
    }

    /// Where `theme`'s `atom`'s page is, if it has one.
    fn find(&self, theme: &Theme, atom: Atom) -> Option<usize> {
        (self.pages.iter()).position(|p| p.theme == theme && p.atom == atom)
    }

    /// `theme`'s `atom`'s patch: edited, kept, or built in.
    pub fn patch(&self, theme: &Theme, atom: Atom) -> Patch {
        self.find(theme, atom)
            .map_or_else(|| theme.patch(atom), |k| self.pages[k].patch)
    }

    /// Where `theme`'s `atom`'s patch stands.
    pub fn status(&self, theme: &Theme, atom: Atom) -> PatchStatus {
        match self.find(theme, atom).map(|k| self.pages[k]) {
            None => PatchStatus::AsBuilt,
            Some(p) if p.kept == Some(p.patch) => PatchStatus::Kept,
            Some(p) if p.kept.is_some() => PatchStatus::KeptEdited,
            Some(_) => PatchStatus::Edited,
        }
    }

    /// What is kept of `theme`'s `atom`'s patch, if anything.
    pub fn kept(&self, theme: &Theme, atom: Atom) -> Option<Patch> {
        self.find(theme, atom).and_then(|k| self.pages[k].kept)
    }

    /// Whether nothing is edited or kept.
    pub fn is_empty(&self) -> bool {
        self.pages.is_empty()
    }

    /// Its revision, new at each change (an edit, a keep, a revert) and
    /// never one of another book's: what plays from it looks again when it
    /// moves.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// `theme`'s `atom`'s patch changed by `change` (none made of a
    /// change that leaves it the theme's own).
    pub fn edit(&mut self, theme: &'static Theme, atom: Atom, change: impl FnOnce(&mut Patch)) {
        let mut patch = self.patch(theme, atom);
        change(&mut patch);
        match self.find(theme, atom) {
            Some(k) if self.pages[k].patch == patch => return,
            Some(k) => self.pages[k].patch = patch,
            None if patch == theme.patch(atom) => return,
            None => self.pages.push(Page {
                theme,
                atom,
                patch,
                kept: None,
            }),
        }
        self.revision = next_revision();
    }

    /// `theme`'s `atom`'s patch kept as it is now: what to write, or where
    /// it stands when there is nothing to (as built, or kept already).
    pub fn keep(&mut self, theme: &Theme, atom: Atom) -> Result<Patch, PatchStatus> {
        let status = self.status(theme, atom);
        let Some(k) = self
            .find(theme, atom)
            .filter(|_| status != PatchStatus::Kept)
        else {
            return Err(status);
        };
        let page = &mut self.pages[k];
        page.kept = Some(page.patch);
        self.revision = next_revision();
        Ok(page.patch)
    }

    /// `theme`'s `atom`'s patch back to the theme's own, what was kept of
    /// it forgotten: whether something was kept (none: it was as built).
    pub fn revert(&mut self, theme: &Theme, atom: Atom) -> Option<bool> {
        let k = self.find(theme, atom)?;
        let gone = self.pages.remove(k);
        self.revision = next_revision();
        Some(gone.kept.is_some())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::theme::{CLASSIC, SPOOKY};

    /// An edit is its theme's alone; kept, it stands kept until edited
    /// again; reverted, the theme's own is back; each change moves the
    /// revision, an edit to the same patch does not.
    #[test]
    fn a_patch_is_edited_kept_and_reverted_by_theme() {
        let mut book = PatchBook::default();
        let louder = |p: &mut Patch| p.level += 3.0;
        book.edit(&CLASSIC, Atom::Pop, louder);
        let edited = book.patch(&CLASSIC, Atom::Pop);
        assert_eq!(edited.level, CLASSIC.patch(Atom::Pop).level + 3.0);
        assert_eq!(book.patch(&SPOOKY, Atom::Pop), SPOOKY.patch(Atom::Pop));
        assert_eq!(book.status(&CLASSIC, Atom::Pop), PatchStatus::Edited);
        let at = book.revision();
        book.edit(&CLASSIC, Atom::Pop, |_| {});
        assert_eq!(book.revision(), at);
        assert_eq!(book.keep(&CLASSIC, Atom::Pop), Ok(edited));
        assert_eq!(book.keep(&CLASSIC, Atom::Pop), Err(PatchStatus::Kept));
        assert_eq!(book.keep(&SPOOKY, Atom::Pop), Err(PatchStatus::AsBuilt));
        book.edit(&CLASSIC, Atom::Pop, louder);
        assert_eq!(book.status(&CLASSIC, Atom::Pop), PatchStatus::KeptEdited);
        assert_eq!(book.kept(&CLASSIC, Atom::Pop), Some(edited));
        assert_eq!(book.revert(&CLASSIC, Atom::Pop), Some(true));
        assert_eq!(book.revert(&CLASSIC, Atom::Pop), None);
        assert_eq!(book.patch(&CLASSIC, Atom::Pop), CLASSIC.patch(Atom::Pop));
        assert!(book.is_empty());
        assert!(book.revision() > at);
        // Another book, at whatever revision, is never at this one's.
        let other = PatchBook::of_kept([]);
        assert_ne!(other.revision(), book.revision());
        assert_ne!(PatchBook::default().revision(), book.revision());
    }
}
