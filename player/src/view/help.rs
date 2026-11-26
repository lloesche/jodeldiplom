//! The help screen's model ([`Page`]): every key by context, as its
//! search leaves them, a line at a time from where it is scrolled to: a
//! section's title, its keys with what they do, a gap after it (the same
//! lines, and as many, as [`crate::core::help_line_count`] counts); the
//! search while there is one; and the keys that move about it. The window
//! draws it (`gui::screens::help`): the sections as cards, the keys as
//! key caps.
//!
//! How many lines show is the window's ([`crate::core::Fit::help_rows`]),
//! measured on the page's measures in points ([`measure`], [`rows`]): every
//! line is one [`measure::LINE`] high, so the core's scrolling and paging
//! in lines are the page's.

use std::fmt;

use crate::core::keymap::{self, Context, Op};
use crate::core::{help_shown, State};

/// The page's measures, in points, as the window draws it.
pub mod measure {
    /// A line: a section's title, a key's line, the gap after a section.
    pub const LINE: f32 = 26.0;
    /// The room above the header.
    pub const TOP: f32 = 20.0;
    /// The header: the title, the search, the keys that move about.
    pub const HEADER: f32 = 32.0;
    /// Between the header and the lines.
    pub const UNDER_HEADER: f32 = 12.0;
    /// The room under the lines.
    pub const BOTTOM: f32 = 12.0;
}

/// The lines a page `height` points high shows (the header and its room
/// taken off), and the search's line while there is one (the core takes
/// it off again: [`crate::core::Fit::help_rows`] counts it).
pub fn rows(height: f32, state: &State) -> u16 {
    let room = height - measure::TOP - measure::HEADER - measure::UNDER_HEADER - measure::BOTTOM;
    let lines = (room / measure::LINE)
        .floor()
        .clamp(0.0, f32::from(u16::MAX - 1)) as u16;
    lines + u16::from(state.help_search.is_some())
}

/// The screen's title.
pub const TITLE: &str = "Keys";

/// The keys that move about the help, and what they do (from the keymap).
pub fn hints() -> [(String, &'static str); 3] {
    let key = |op| keymap::hint(Context::Help, op);
    [
        (key(Op::Up) + &key(Op::Down), "scroll"),
        (key(Op::Filter), "search"),
        (key(Op::Back), "back"),
    ]
}

/// A line of the help.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Line {
    /// A section's title: where its keys work.
    Title(&'static str),
    /// Keys (each a key cap; any of them does it) and what they do.
    Keys {
        /// The keys, one by one (`Q`, `ctrl+c`).
        keys: Vec<String>,
        /// What they do.
        what: &'static str,
    },
    /// The gap after a section.
    Gap,
}

/// The search of the help, as it stands.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Search {
    /// What is typed.
    pub typed: String,
    /// It is being typed (the caret shows).
    pub editing: bool,
    /// How many keys it leaves.
    pub shown: usize,
}

/// What the help screen shows now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Page {
    /// The lines shown, from where the help is scrolled to.
    pub lines: Vec<Line>,
    /// The search, while there is one.
    pub search: Option<Search>,
    /// What shows where the search leaves no key.
    pub empty: Option<String>,
}

impl Page {
    /// The page for `state`: as many lines as the window shows
    /// ([`crate::core::Fit::help_rows`], but for the search's).
    pub fn of(state: &State) -> Page {
        let sections = help_shown(state);
        let search = state.help_search.as_ref().map(|s| Search {
            typed: s.text.clone(),
            editing: s.editing,
            shown: sections.iter().map(|s| s.lines.len()).sum(),
        });
        let empty = search
            .as_ref()
            .filter(|s| s.shown == 0)
            .map(|s| format!("No key or text has “{}”", s.typed));
        let shown = usize::from(state.fit.help_rows).saturating_sub(usize::from(search.is_some()));
        let lines = sections
            .into_iter()
            .flat_map(|section| {
                let keys = section.lines.into_iter().map(|(keys, what)| Line::Keys {
                    keys: keys.split(" / ").map(str::to_owned).collect(),
                    what,
                });
                std::iter::once(Line::Title(section.title))
                    .chain(keys)
                    .chain(std::iter::once(Line::Gap))
            })
            .skip(state.help_scroll)
            .take(shown)
            .collect();
        Page {
            lines,
            search,
            empty,
        }
    }
}

/// An outline of the page, one line a piece with its kind in brackets:
/// for the snapshots.
impl fmt::Display for Page {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[title] {TITLE}")?;
        if let Some(s) = &self.search {
            let caret = if s.editing { "▏" } else { "" };
            writeln!(f, "[search] / {}{caret}  {} shown", s.typed, s.shown)?;
        }
        let hints: Vec<String> = hints().iter().map(|(k, w)| format!("{k} {w}")).collect();
        writeln!(f, "[hints] {}", hints.join("   "))?;
        if let Some(empty) = &self.empty {
            writeln!(f, "[empty] {empty}")?;
        }
        for line in &self.lines {
            match line {
                Line::Title(title) => writeln!(f, "[section] {title}")?,
                Line::Keys { keys, what } => writeln!(f, "[keys] {}  {what}", keys.join(" / "))?,
                Line::Gap => writeln!(f, "[gap]")?,
            }
        }
        Ok(())
    }
}
