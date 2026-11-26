//! The command palette's model ([`Palette`]): what is typed, and the
//! commands it matches in view, each with its keys, the one selected
//! marked. The window draws it over everything
//! ([`crate::gui::screens::palette`]).

use std::fmt;

use crate::core::keymap::{self, Context, Op};
use crate::core::{palette_window, State};

/// The keys that work the palette, and what they do (from the keymap).
pub fn hints() -> [(String, &'static str); 3] {
    let key = |op| keymap::hint(Context::Palette, op);
    [
        (format!("{} {}", key(Op::Up), key(Op::Down)), "move"),
        (key(Op::Accept), "run"),
        (key(Op::Back), "close"),
    ]
}

/// The palette as it is now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Palette {
    /// What is typed.
    pub typed: String,
    /// The commands it matches in view.
    pub matches: Vec<Match>,
    /// How many it matches in all.
    pub total: usize,
}

/// A command the palette matches.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Match {
    /// Its index among the matches.
    pub index: usize,
    /// Its name.
    pub name: &'static str,
    /// Its keys, each as written for people (`N`, `shift+→`).
    pub keys: Vec<String>,
    /// The one selected.
    pub selected: bool,
}

impl Palette {
    /// The palette for `state`, where it is open, `rows` matches in view
    /// at most (the one selected among them).
    pub fn of(state: &State, rows: usize) -> Option<Palette> {
        let palette = state.palette.as_ref()?;
        let (matches, start) = palette_window(state, rows.max(1) + 1);
        let total = matches.len();
        let matches = matches
            .into_iter()
            .enumerate()
            .skip(start)
            .take(rows.max(1))
            .map(|(index, entry)| Match {
                index,
                name: entry.name,
                keys: entry.keys.split(" / ").map(str::to_owned).collect(),
                selected: index == palette.cursor,
            })
            .collect();
        Some(Palette {
            typed: palette.text.clone(),
            matches,
            total,
        })
    }
}

/// An outline of the palette, one line a piece with its tone in
/// brackets: for the snapshots.
impl fmt::Display for Palette {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[palette] : {}▏", self.typed)?;
        for m in &self.matches {
            let selected = if m.selected { " selected" } else { "" };
            writeln!(f, "[match{selected}] {}  {}", m.name, m.keys.join(" / "))?;
        }
        if self.total == 0 {
            writeln!(f, "[dim] no command matches")?;
        }
        Ok(())
    }
}
