//! The settings screen's model ([`Settings`]): the rows in their sections
//! (each row's label, what it is set to in words, and its control: a
//! switch, a check, a choice of a few, or a value that opens a list), the
//! device list or the data folder's list under its row while it is open,
//! the notes (where the settings are saved, what went wrong) and the keys
//! that move about. The window draws it (`gui::screens::settings`); a
//! click on a row is [`crate::core::Action::SettingsClick`] with the step
//! its control asks for ([`Control::step_to`]).

use std::fmt;

use jodel_playback::Headroom;

use crate::core::transition::Preset;
use crate::core::{device_window, folder_window, FolderRow, Gpu, SettingsRow, State};

/// The rows a device list or the data folder's list shows at most.
pub const LIST_ROWS: usize = 8;

/// The screen's title.
pub const TITLE: &str = "Settings";

/// The keys that move about the settings, and what they do.
pub const HINTS: [(&str, &str); 4] = [
    ("↑↓", "select"),
    ("←→", "change"),
    ("enter", "open"),
    ("esc", "back"),
];

/// How a row is set.
#[derive(Clone, Debug, PartialEq)]
pub enum Control {
    /// On or off.
    Switch {
        /// It is on.
        on: bool,
        /// It can be turned on (stems cannot with no GPU).
        usable: bool,
    },
    /// One of the beat methods.
    Check {
        /// It is checked: it votes, or it is the one that decides.
        on: bool,
        /// A radio button (one decides), not a box (they vote).
        round: bool,
    },
    /// One of a few.
    Choice {
        /// What each is, in words.
        options: Vec<String>,
        /// The one set (none: a value of the config file's that is not
        /// among them).
        chosen: Option<usize>,
    },
    /// A value that opens a list to choose another in.
    Open(String),
    /// The data folder moving.
    Moving {
        /// The share of it moved.
        share: f64,
        /// It stopped.
        failed: bool,
    },
}

impl Control {
    /// The step [`crate::core::Action::SettingsClick`] takes to get from
    /// what a choice is now to its option `option` (the steps go round);
    /// `None` for the chosen one, where there is nothing to do.
    pub fn step_to(&self, option: usize) -> Option<i32> {
        let Control::Choice { options, chosen } = self else {
            return None;
        };
        match *chosen {
            Some(c) if c == option => None,
            Some(c) => Some(option as i32 - c as i32),
            // Not among the options: the steps count from the first, and
            // the first is the whole way round (a step of none opens).
            None if option == 0 => Some(options.len() as i32),
            None => Some(option as i32),
        }
    }
}

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

/// The device list or the data folder's list, under its row: the
/// entries shown ([`LIST_ROWS`] at most, the selected one in view) and
/// what shows while it is listed or empty.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct List {
    /// The entries shown.
    pub entries: Vec<Entry>,
    /// Still listing, or nothing found.
    pub note: Option<String>,
}

/// A row of the settings.
#[derive(Clone, Debug, PartialEq)]
pub struct Row {
    /// Its place among [`SettingsRow::ALL`].
    pub index: usize,
    /// Its label.
    pub label: &'static str,
    /// It belongs to the row above it (set in under it).
    pub under: bool,
    /// The keys are on it.
    pub selected: bool,
    /// What it is set to, in words.
    pub about: Option<String>,
    /// How it is set.
    pub control: Control,
    /// Its list, while open.
    pub list: Option<List>,
}

/// A section: its title and rows.
#[derive(Clone, Debug, PartialEq)]
pub struct Section {
    /// Its title.
    pub title: &'static str,
    /// Its rows, in order.
    pub rows: Vec<Row>,
}

/// How a note is coloured.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    /// Where the settings are saved.
    Dim,
    /// What needs knowing (this run plays into the null output).
    Warn,
    /// What went wrong.
    Fail,
}

/// What the settings screen shows now.
#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// The sections, top to bottom.
    pub sections: Vec<Section>,
    /// The notes under them.
    pub notes: Vec<(String, Tone)>,
}

/// The section `row` is in.
fn section(row: SettingsRow) -> &'static str {
    match row {
        SettingsRow::Device | SettingsRow::Headroom => "Output",
        SettingsRow::Tempo | SettingsRow::Voting | SettingsRow::Method(_) => "Tempo and beats",
        SettingsRow::Stems | SettingsRow::StemSwitch => "Stems",
        SettingsRow::SkipMix | SettingsRow::SkipPreset => "Next track",
        SettingsRow::DataFolder => "Data",
    }
}

impl Settings {
    /// The settings for `state`.
    pub fn of(state: &State) -> Settings {
        let mut sections: Vec<Section> = Vec::new();
        for (index, &row) in SettingsRow::ALL.iter().enumerate() {
            let (about, control) = set(state, row);
            let selected = index == state.settings.row
                && !state.settings.picking
                && state.settings.folders.is_none();
            let list = match row {
                SettingsRow::Device if state.settings.picking => Some(devices(state)),
                SettingsRow::DataFolder => folders(state),
                _ => None,
            };
            let made = Row {
                index,
                label: row.label(),
                under: row.under(),
                selected,
                about,
                control,
                list,
            };
            let title = section(row);
            match sections.last_mut().filter(|s| s.title == title) {
                Some(s) => s.rows.push(made),
                None => sections.push(Section {
                    title,
                    rows: vec![made],
                }),
            }
        }
        let mut notes = vec![(
            match &state.config_file {
                Some(path) => format!("Saved to {}", path.display()),
                None => "Not saved: no config file".to_owned(),
            },
            Tone::Dim,
        )];
        if let Some(why) = state.data_move.as_ref().and_then(|m| m.failure.as_ref()) {
            notes.push((why.clone(), Tone::Fail));
        }
        if state.null_output {
            notes.push((
                "--null: this run plays into the null output, whatever device is set".to_owned(),
                Tone::Warn,
            ));
        }
        Settings { sections, notes }
    }
}

/// What `row` is set to, in words, and how it is set.
fn set(state: &State, row: SettingsRow) -> (Option<String>, Control) {
    let c = &state.config;
    let switch = |on: bool, yes: &str, no: &str| {
        (
            Some(if on { yes } else { no }.to_owned()),
            Control::Switch { on, usable: true },
        )
    };
    let choice = |options: Vec<String>, chosen: Option<usize>, about: String| {
        (Some(about), Control::Choice { options, chosen })
    };
    match row {
        SettingsRow::Device => (None, Control::Open(state.device_label())),
        SettingsRow::Headroom => {
            let about = match c.headroom {
                Headroom::NONE => "lossy files -1.5 dB, for their overs",
                h if h < jodel_playback::LOSSY_HEADROOM => {
                    "where the samples change, lossy files -1.5 dB"
                }
                _ => "where the samples change",
            };
            let options = Headroom::CHOICES
                .iter()
                .map(|h| match *h {
                    Headroom::NONE => "none".to_owned(),
                    h => format!("{} dB", h.value()),
                })
                .collect();
            let chosen = Headroom::CHOICES.iter().position(|h| *h == c.headroom);
            choice(options, chosen, about.to_owned())
        }
        SettingsRow::Tempo => switch(
            c.bpm,
            "tempo, beat pulse and beat analysis",
            "no tempo, no beat pulse, nothing analysed",
        ),
        SettingsRow::Voting => switch(
            c.beat_voting,
            "the checked methods vote",
            "one method decides",
        ),
        SettingsRow::Method(method) => {
            let on = if c.beat_voting {
                c.beat_methods.contains(&method)
            } else {
                c.beat_method == method
            };
            let what = match method {
                jodel_rhythm::Method::Flux => "onsets through two trackers",
                jodel_rhythm::Method::Network => "a neural network, our own inference",
            };
            (
                Some(what.to_owned()),
                Control::Check {
                    on,
                    round: !c.beat_voting,
                },
            )
        }
        SettingsRow::Stems => {
            let (about, on, usable) = match &state.gpu {
                Gpu::Present(name) if c.stems => (
                    format!("the cues listen to stems, separated on {name}"),
                    true,
                    true,
                ),
                Gpu::Present(_) => ("the cues listen to the mix".to_owned(), false, true),
                Gpu::Absent => (
                    "unavailable: no GPU, the cues listen to the mix".to_owned(),
                    false,
                    false,
                ),
                Gpu::Looking => ("looking for a GPU…".to_owned(), false, false),
            };
            (Some(about), Control::Switch { on, usable })
        }
        SettingsRow::StemSwitch => choice(
            vec!["on the next bar".to_owned(), "instantly".to_owned()],
            Some(usize::from(!c.stems_on_bar)),
            if c.stems_on_bar {
                "stems go on and off on the beat"
            } else {
                "within a second or so"
            }
            .to_owned(),
        ),
        SettingsRow::SkipMix => choice(
            vec!["cut".to_owned(), "mix in".to_owned()],
            Some(usize::from(c.skip_mix)),
            if c.skip_mix {
                "beat-matched, by its transition below"
            } else {
                "the next track starts at once"
            }
            .to_owned(),
        ),
        SettingsRow::SkipPreset => {
            let options = std::iter::once("the tab's".to_owned())
                .chain(Preset::ALL.iter().map(|p| p.name().to_owned()))
                .collect();
            let chosen = match c.skip_preset {
                None => Some(0),
                Some(p) => Preset::ALL.iter().position(|q| *q == p).map(|i| i + 1),
            };
            let about = match c.skip_preset {
                None => format!(
                    "the Transition tab's: {}",
                    state.transition.draft.preset.name()
                ),
                Some(p) => format!("{} (its own)", p.name()),
            };
            choice(options, chosen, about)
        }
        SettingsRow::DataFolder => match (&state.data_move, &state.data_dir) {
            (Some(m), _) if m.failure.is_some() => (
                Some("the move stopped: enter tries again".to_owned()),
                Control::Moving {
                    share: m.share,
                    failed: true,
                },
            ),
            (Some(m), _) if !m.finished => (
                Some(format!(
                    "{:.0} % moving to {}",
                    m.share * 100.0,
                    m.to.display()
                )),
                Control::Moving {
                    share: m.share,
                    failed: false,
                },
            ),
            (_, Some(dir)) => (None, Control::Open(dir.display().to_string())),
            (_, None) => (None, Control::Open("none".to_owned())),
        },
    }
}

/// The output devices to choose from, the chosen one marked.
fn devices(state: &State) -> List {
    let Some(list) = &state.settings.devices else {
        return List {
            entries: Vec::new(),
            note: Some("listing devices…".to_owned()),
        };
    };
    if list.is_empty() {
        return List {
            entries: Vec::new(),
            note: Some("no output devices found".to_owned()),
        };
    }
    let start = device_window(state, LIST_ROWS);
    let entries = list
        .iter()
        .enumerate()
        .skip(start)
        .take(LIST_ROWS)
        .map(|(index, choice)| Entry {
            index,
            mark: if choice.id == state.config.device {
                Mark::Chosen
            } else {
                Mark::Device
            },
            name: choice.name.clone(),
            detail: choice.summary.clone(),
            selected: index == state.settings.device_cursor,
        })
        .collect();
    List {
        entries,
        note: None,
    }
}

/// The data folder's list, while open: the choice of the folder shown,
/// up, its subfolders.
fn folders(state: &State) -> Option<List> {
    let pick = state.settings.folders.as_ref()?;
    let start = folder_window(state, LIST_ROWS);
    let entries = pick
        .rows()
        .into_iter()
        .enumerate()
        .skip(start)
        .take(LIST_ROWS)
        .map(|(index, row)| {
            let (mark, name) = match row {
                FolderRow::Choose(dir) => (
                    Mark::Here,
                    format!(
                        "move the data folder here: {}",
                        dir.join(crate::dirs::DATA).display()
                    ),
                ),
                FolderRow::Up => (Mark::Up, "..".to_owned()),
                FolderRow::Folder(dir) => (
                    Mark::Folder,
                    dir.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_else(|| dir.display().to_string()),
                ),
            };
            Entry {
                index,
                mark,
                name,
                detail: String::new(),
                selected: index == pick.cursor,
            }
        })
        .collect();
    Some(List {
        entries,
        note: pick.folders.is_none().then(|| "listing…".to_owned()),
    })
}

impl Mark {
    /// The mark as the outline writes it.
    fn symbol(self) -> &'static str {
        match self {
            Mark::Chosen => "●",
            Mark::Device => "○",
            Mark::Here => "✓",
            Mark::Up => "↑",
            Mark::Folder => "▸",
        }
    }
}

/// An outline of the screen, one line a piece with its kind in brackets:
/// for the snapshots.
impl fmt::Display for Settings {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[title] {TITLE}")?;
        for section in &self.sections {
            writeln!(f, "[section] {}", section.title)?;
            for row in &section.rows {
                let selected = if row.selected { " selected" } else { "" };
                let under = if row.under { "  " } else { "" };
                write!(f, "{under}[row{selected}] {}", row.label)?;
                match &row.control {
                    Control::Switch { on, usable } => {
                        let state = match (on, usable) {
                            (_, false) => "unusable",
                            (true, true) => "on",
                            (false, true) => "off",
                        };
                        write!(f, " [switch {state}]")?;
                    }
                    Control::Check { on, round } => {
                        let mark = match (on, round) {
                            (true, true) => "(●)",
                            (false, true) => "( )",
                            (true, false) => "[x]",
                            (false, false) => "[ ]",
                        };
                        write!(f, " {mark}")?;
                    }
                    Control::Choice { options, chosen } => {
                        let options: Vec<String> = options
                            .iter()
                            .enumerate()
                            .map(|(i, o)| {
                                if Some(i) == *chosen {
                                    format!("<{o}>")
                                } else {
                                    o.clone()
                                }
                            })
                            .collect();
                        write!(f, " [choice] {}", options.join(" | "))?;
                    }
                    Control::Open(value) => write!(f, " [open] {value} ▾")?,
                    Control::Moving { share, failed } => {
                        let failed = if *failed { " stopped" } else { "" };
                        write!(f, " [moving{failed}] {:.0} %", share * 100.0)?;
                    }
                }
                writeln!(f)?;
                if let Some(about) = &row.about {
                    writeln!(f, "{under}  {about}")?;
                }
                if let Some(list) = &row.list {
                    for e in &list.entries {
                        let selected = if e.selected { " selected" } else { "" };
                        write!(
                            f,
                            "{under}  [entry{selected}] {} {}",
                            e.mark.symbol(),
                            e.name
                        )?;
                        if !e.detail.is_empty() {
                            write!(f, "  {}", e.detail)?;
                        }
                        writeln!(f)?;
                    }
                    if let Some(note) = &list.note {
                        writeln!(f, "{under}  [note] {note}")?;
                    }
                }
            }
        }
        for (note, tone) in &self.notes {
            let tone = match tone {
                Tone::Dim => "dim",
                Tone::Warn => "warn",
                Tone::Fail => "fail",
            };
            writeln!(f, "[{tone}] {note}")?;
        }
        let hints: Vec<String> = HINTS.iter().map(|(k, w)| format!("{k} {w}")).collect();
        writeln!(f, "[hints] {}", hints.join("   "))
    }
}
