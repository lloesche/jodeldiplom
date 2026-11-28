//! A list opened under the value it chooses, its entries each taking a
//! click: the output devices and the data folder's folders on the
//! settings, the microphones on the Karaoke tab and the Mixer tab. The
//! window draws it (`gui::widgets::choices`).

use std::fmt;

use crate::core::State;

/// How an entry of a list is marked.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mark {
    /// The device chosen.
    Chosen,
    /// Another device.
    Device,
    /// Make the data folder in the folder shown.
    Here,
    /// Up a folder.
    Up,
    /// A folder to open.
    Folder,
}

impl Mark {
    /// The mark as the outlines write it.
    pub fn symbol(self) -> &'static str {
        match self {
            Mark::Chosen => "●",
            Mark::Device => "○",
            Mark::Here => "✓",
            Mark::Up => "↑",
            Mark::Folder => "▸",
        }
    }
}

/// An entry of a list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// Its place in the whole list (what a click chooses).
    pub index: usize,
    /// Its mark.
    pub mark: Mark,
    /// Its name.
    pub name: String,
    /// What it takes, in a few words (a device's formats).
    pub detail: String,
    /// The one the keys are on.
    pub selected: bool,
}

/// A list under the value it chooses: the entries shown and what shows
/// while it is listed or empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct List {
    /// The entries shown.
    pub entries: Vec<Entry>,
    /// Still listing, or nothing found.
    pub note: Option<String>,
}

/// The microphones' list, where it is open and shows: the one in use (or
/// the settings name) marked and selected.
pub fn mics(state: &State) -> Option<List> {
    let open = state
        .karaoke
        .mic_list
        .as_ref()
        .filter(|l| state.showing(l.on))?;
    let Some(mics) = &open.mics else {
        return Some(List {
            entries: Vec::new(),
            note: Some("listing microphones…".to_owned()),
        });
    };
    if mics.is_empty() {
        return Some(List {
            entries: Vec::new(),
            note: Some("no microphones found".to_owned()),
        });
    }
    let current = crate::core::mics::current(state);
    let entries = mics
        .iter()
        .enumerate()
        .map(|(index, mic)| {
            let chosen = current == Some(mic.id.as_str());
            Entry {
                index,
                mark: if chosen { Mark::Chosen } else { Mark::Device },
                name: mic.name.clone(),
                detail: String::new(),
                selected: chosen,
            }
        })
        .collect();
    Some(List {
        entries,
        note: None,
    })
}

impl List {
    /// The list as the outlines write it, a line an entry (then the
    /// note), each set in by `indent`.
    pub fn outline(&self, f: &mut fmt::Formatter, indent: &str) -> fmt::Result {
        for e in &self.entries {
            let selected = if e.selected { " selected" } else { "" };
            write!(
                f,
                "{indent}  [entry{selected}] {} {}",
                e.mark.symbol(),
                e.name
            )?;
            if !e.detail.is_empty() {
                write!(f, "  {}", e.detail)?;
            }
            writeln!(f)?;
        }
        if let Some(note) = &self.note {
            writeln!(f, "{indent}  [note] {note}")?;
        }
        Ok(())
    }
}

/// The list as the outlines write it, not set in.
impl fmt::Display for List {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.outline(f, "")
    }
}
