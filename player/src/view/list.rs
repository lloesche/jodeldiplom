//! The list pane's model ([`Pane`]): the tabs over it (the browser, the
//! queue with its length, Transition, Karaoke, the Mixer) and, on the
//! browser's or the queue's side, what it shows of them: where the
//! browser is, the rows in view (what each is, its title, format and
//! length; in the queue how far each is got ready, and the icons of what
//! is known of it), a note where there is nothing to list, and the
//! filter line. The window draws it ([`crate::gui::screens::list`]),
//! row by row from the first in view: a folder of thousands costs what a
//! folder of ten does.

use std::fmt;
use std::path::{Path, PathBuf};

use jodel_playback::PlayState;

use crate::browser::{self, EntryKind};
use crate::core::karaoke::SongLyrics;
use crate::core::keymap::{self, Context, Op};
use crate::core::text::{self, clock};
use crate::core::{
    browser_rows, list_cursor, list_height, readiness, row_entry, Job, LyricsFacts, QueueFacts,
    Readiness, Rhythm, RhythmFacts, Row, Side, State,
};

/// What the empty queue says, and how to fill it.
pub const QUEUE_EMPTY: &str = "The queue is empty.";
/// How to fill the queue, with the browser's keys from the keymap.
pub fn queue_hint() -> String {
    let key = |op| keymap::hint(Context::List, op);
    format!(
        "{} queues a file or playlist, {} plays it too; {} plays a file with its whole folder.",
        key(Op::Open),
        key(Op::QueuePlay),
        key(Op::OpenReplace)
    )
}

/// The narrowest pane, in points, whose queue rows show the icons of what
/// is known of each track.
pub const FACTS_WIDTH: f32 = 448.0;

/// The list pane as it is now.
#[derive(Clone, Debug, PartialEq)]
pub struct Pane {
    /// The tabs, left to right.
    pub tabs: Vec<Tab>,
    /// What the pane shows under them.
    pub body: Body,
}

/// A tab over the pane.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tab {
    /// The side it shows.
    pub side: Side,
    /// Its name.
    pub label: &'static str,
    /// The count beside its name (the queue's length).
    pub count: Option<usize>,
    /// It is the side shown.
    pub shown: bool,
}

/// What the pane shows under its tabs.
#[derive(Clone, Debug, PartialEq)]
pub enum Body {
    /// The browser.
    Browser(Browser),
    /// The queue.
    Queue(Queue),
    /// A tab of its own (Transition, Karaoke, the Mixer), drawn apart.
    Tab,
}

/// The browser's side.
#[derive(Clone, Debug, PartialEq)]
pub struct Browser {
    /// Where it is.
    pub place: Place,
    /// The rows in view.
    pub rows: Vec<BrowserRow>,
    /// What there is to say where there is nothing (else) to list.
    pub note: Option<Note>,
    /// The filter line, while the folder is filtered.
    pub filter: Option<FilterLine>,
}

/// Where the browser is: its folder (or the drives), and whether it is
/// being listed or searched below.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Place {
    /// The folder, as written for people (`Drives` for the drives).
    pub path: String,
    /// Its parts, from the root.
    pub crumbs: Vec<String>,
    /// The folder each part names (none for the drives).
    pub dirs: Vec<PathBuf>,
    /// While a search below it lists what it found: how many, and
    /// whether it is still at it.
    pub found: Option<(usize, bool)>,
    /// The folder is being listed.
    pub listing: bool,
}

/// What a browser row is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// The folder above (`..`).
    Parent,
    /// A folder.
    Folder,
    /// A playlist.
    Playlist,
    /// A music file.
    File,
    /// "Search all subfolders for …", under a filtered folder.
    SearchBelow,
}

/// A file's format and rate, as its chip says it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chip {
    /// `FLAC 44.1`, `WAV 96`; `?` when the file could not be read.
    pub text: String,
    /// The file could not be read.
    pub failed: bool,
}

/// What a row knows of a file: its chip and length, or that they are
/// still being read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Known {
    /// Not a file.
    Nothing,
    /// Still being read.
    Pending,
    /// Read: its chip, and its length where it has one.
    Read(Chip, Option<String>),
}

/// A row of the browser.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrowserRow {
    /// Its index in the browser's rows.
    pub index: usize,
    /// What it is.
    pub kind: Kind,
    /// Its title: the folder's or file's name, the title tag, the path
    /// below the folder searched; what is searched for, for
    /// [`Kind::SearchBelow`].
    pub title: String,
    /// A file's chip and length.
    pub known: Known,
    /// The row selected.
    pub selected: bool,
    /// The file playing.
    pub playing: bool,
}

/// What there is to say where there is nothing (else) to list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Note {
    /// The folder could not be listed: why.
    Failed(String),
    /// It is being listed.
    Listing,
    /// A search found nothing yet.
    NothingFound,
    /// Nothing to play here.
    Empty,
}

impl Note {
    /// What it says.
    pub fn text(&self) -> &str {
        match self {
            Note::Failed(why) => why,
            Note::Listing => "listing…",
            Note::NothingFound => "nothing found yet",
            Note::Empty => "no folders or music files here",
        }
    }
}

/// The filter line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FilterLine {
    /// What is typed.
    pub typed: String,
    /// It is being typed (the caret shows).
    pub editing: bool,
    /// How many rows it leaves (the search row not counted).
    pub shown: usize,
}

/// The queue's side.
#[derive(Clone, Debug, PartialEq)]
pub struct Queue {
    /// Which track plays of how many ("playing 2 of 5"), or how many
    /// there are.
    pub summary: String,
    /// The rows in view (none: the queue is empty).
    pub rows: Vec<QueueRow>,
}

/// A thing known of a track in the queue, as it is got ready.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fact {
    /// Its key (found as soon as the file is decoded).
    Key,
    /// Its stems (separated alongside, on the GPU).
    Stems,
    /// Its tempo (decided last, from the beats and the stems).
    Tempo,
    /// Its lyrics (looked up and timed alongside).
    Lyrics,
    /// Its cover, taken apart.
    Cover,
}

impl Fact {
    /// Its sign in the outline.
    fn sign(self) -> &'static str {
        match self {
            Fact::Key => "♯",
            Fact::Stems => "▥",
            Fact::Tempo => "♩",
            Fact::Lyrics => "¶",
            Fact::Cover => "◐",
        }
    }
}

/// A fact as a queue row shows it: lit once it is known; the lyrics
/// `heard` when no trusted source had them and machine learning heard
/// them from the voice.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Status {
    /// What.
    pub fact: Fact,
    /// Known.
    pub lit: bool,
    /// Heard by machine learning.
    pub heard: bool,
}

/// A row of the queue.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueRow {
    /// Its index in the queue.
    pub index: usize,
    /// Its title.
    pub title: String,
    /// The track playing (or paused, or stopped on it).
    pub current: Option<PlayState>,
    /// How far it is got ready.
    pub readiness: Readiness,
    /// What its preparation is doing, or found just now
    /// ([`crate::view::prep::caption`]).
    pub caption: Option<crate::view::prep::Caption>,
    /// What is known of it, where the pane is wide enough (empty
    /// otherwise), in the order the work finds it.
    pub facts: Vec<Status>,
    /// Its chip and length.
    pub known: Known,
    /// The row selected.
    pub selected: bool,
}

impl Pane {
    /// The pane for `state`, `width` points wide.
    pub fn of(state: &State, width: f32) -> Pane {
        let tabs = [
            (Side::Browser, "Browser"),
            (Side::Queue, "Queue"),
            (Side::Transition, "Transition"),
            (Side::Karaoke, "Karaoke"),
            (Side::Mixer, "Mixer"),
            (Side::Synth, "Synth"),
        ]
        .into_iter()
        .map(|(side, label)| Tab {
            side,
            label,
            count: (side == Side::Queue).then_some(state.queue.paths.len()),
            shown: state.side == side,
        })
        .collect();
        let body = match state.side {
            Side::Browser => Body::Browser(Browser::of(state)),
            Side::Queue => Body::Queue(Queue::of(state, width)),
            Side::Transition | Side::Karaoke | Side::Mixer | Side::Synth => Body::Tab,
        };
        Pane { tabs, body }
    }
}

/// The rows of the browser from `first`, `count` of them at most: found
/// one by one (the folder's own and the search's), the folder's filtered
/// rows listed whole.
fn browser_window(state: &State, first: usize, count: usize) -> Vec<(usize, Row)> {
    let b = &state.browser;
    let filtered = b.search.is_none() && b.filter.as_ref().is_some_and(|f| !f.text.is_empty());
    if filtered {
        return browser_rows(state)
            .into_iter()
            .enumerate()
            .skip(first)
            .take(count)
            .collect();
    }
    let len = match &b.search {
        Some(search) => search.hits.len(),
        None => b.entries.len(),
    };
    (first..len.min(first.saturating_add(count)))
        .map(|i| {
            let row = if b.search.is_some() {
                Row::Hit(i)
            } else {
                Row::Entry(i)
            };
            (i, row)
        })
        .collect()
}

/// What a row knows of the file at `path`.
fn known(state: &State, path: &Path) -> Known {
    match state.details.get(path) {
        Some(d) => Known::Read(
            Chip {
                text: d.chip.clone(),
                failed: d.error.is_some(),
            },
            d.seconds.map(clock),
        ),
        None => Known::Pending,
    }
}

impl Browser {
    fn of(state: &State) -> Browser {
        let b = &state.browser;
        let (cursor, scroll) = list_cursor(state, Side::Browser);
        let window = browser_window(state, scroll, list_height(state, Side::Browser));
        let playing = state.playing.track.as_ref().map(|t| t.info.path.as_path());
        let rows: Vec<BrowserRow> = window
            .into_iter()
            .map(|(index, row)| {
                let (kind, title, known, path) = match (row, row_entry(state, row)) {
                    (Row::SearchBelow, _) | (_, None) => (
                        Kind::SearchBelow,
                        b.filter.as_ref().map_or("", |f| f.text.as_str()).to_owned(),
                        Known::Nothing,
                        None,
                    ),
                    (_, Some(e)) => match e.kind {
                        EntryKind::Parent => (Kind::Parent, "..".to_owned(), Known::Nothing, None),
                        EntryKind::Dir => {
                            (Kind::Folder, e.label().to_owned(), Known::Nothing, None)
                        }
                        EntryKind::Playlist => {
                            (Kind::Playlist, e.label().to_owned(), Known::Nothing, None)
                        }
                        EntryKind::File => {
                            let title = if e.relative.is_some() {
                                e.label().to_owned()
                            } else {
                                state
                                    .details
                                    .get(&e.path)
                                    .and_then(|d| d.title.clone())
                                    .unwrap_or_else(|| e.name.clone())
                            };
                            (Kind::File, title, known(state, &e.path), Some(&e.path))
                        }
                    },
                };
                BrowserRow {
                    index,
                    kind,
                    title,
                    known,
                    selected: index == cursor,
                    playing: path.is_some_and(|p| Some(p.as_path()) == playing),
                }
            })
            .collect();
        // Only the `..` row: nothing to play here (the row stays usable).
        let only_parent = || {
            let parent = |e: &browser::Entry| e.kind == EntryKind::Parent;
            match b.filter.as_ref().filter(|f| !f.text.is_empty()) {
                Some(_) => browser_rows(state)
                    .iter()
                    .all(|&r| row_entry(state, r).is_some_and(parent)),
                None => b.entries.iter().all(parent),
            }
        };
        let note = if let Some(error) = &b.error {
            Some(Note::Failed(error.clone()))
        } else if b.loading {
            Some(Note::Listing)
        } else if b.search.as_ref().is_some_and(|s| s.hits.is_empty()) {
            Some(Note::NothingFound)
        } else if b.search.is_none() && only_parent() {
            Some(Note::Empty)
        } else {
            None
        };
        let filter = b.filter.as_ref().map(|f| FilterLine {
            typed: f.text.clone(),
            editing: f.editing,
            shown: browser_rows(state)
                .iter()
                .filter(|r| !matches!(r, Row::SearchBelow))
                .count(),
        });
        Browser {
            place: Place::of(state),
            rows,
            note,
            filter,
        }
    }
}

impl Place {
    fn of(state: &State) -> Place {
        let b = &state.browser;
        let drives = browser::is_drives(&b.dir);
        let path = if drives {
            "Drives".to_owned()
        } else {
            b.dir.to_string_lossy().into_owned()
        };
        let (crumbs, dirs) = if drives {
            (vec![path.clone()], Vec::new())
        } else {
            crumbs(&b.dir).into_iter().unzip()
        };
        Place {
            path,
            crumbs,
            dirs,
            found: b.search.as_ref().map(|s| (s.hits.len(), !s.done)),
            listing: b.loading,
        }
    }
}

/// `dir`'s parts, from its root (a drive's root as `C:`, the root as
/// `/`), each with the folder it names.
fn crumbs(dir: &Path) -> Vec<(String, PathBuf)> {
    use std::path::Component;
    let mut out: Vec<(String, PathBuf)> = Vec::new();
    let mut at = PathBuf::new();
    for part in dir.components() {
        at.push(part);
        match part {
            Component::Prefix(prefix) => {
                // A drive's part names its root, not its current folder.
                at.push(std::path::MAIN_SEPARATOR_STR);
                let name = prefix.as_os_str().to_string_lossy().into_owned();
                out.push((name, at.clone()));
            }
            Component::RootDir if out.is_empty() => out.push(("/".to_owned(), at.clone())),
            Component::Normal(name) => {
                out.push((name.to_string_lossy().into_owned(), at.clone()));
            }
            _ => {}
        }
    }
    out
}

/// The facts of `path` as its queue row shows them, for the jobs the
/// queue is got ready by, in the order they come.
fn facts(state: &State, path: &Path) -> Vec<Status> {
    let mut facts = state.facts.get(path).cloned().unwrap_or_default();
    // Known here already (the songs wanted now), though not told yet.
    if let Some(Rhythm::Ready(r)) = state.rhythm.get(path) {
        if facts.tempo.is_none() {
            facts.analysed(&RhythmFacts::of(&r.analysis));
        }
    }
    let QueueFacts {
        key,
        stems,
        tempo,
        lyrics,
        cover,
    } = facts;
    let known = state.karaoke.lyrics.get(path);
    let heard = lyrics == Some(LyricsFacts::Heard) || known.is_some_and(SongLyrics::heard);
    let lyrics = heard
        || matches!(lyrics, Some(LyricsFacts::Timed | LyricsFacts::Untimed))
        || matches!(known, Some(SongLyrics::Found { .. }));
    let status = |fact, lit: Option<bool>| Status {
        fact,
        lit: lit == Some(true),
        heard: false,
    };
    // A job's last icon lights only once it is done (none of its work to
    // come, none under way): all lit, nothing is left for the track.
    let done = |job| crate::core::job_done(state, path, job);
    let mut out = Vec::new();
    for job in [Job::Rhythm, Job::Lyrics, Job::Cover] {
        if !state.preparing.contains(&job) {
            continue;
        }
        match job {
            Job::Rhythm => out.extend([
                status(Fact::Key, key),
                status(Fact::Stems, stems),
                status(Fact::Tempo, tempo.filter(|_| done(job))),
            ]),
            Job::Lyrics => out.push(Status {
                fact: Fact::Lyrics,
                lit: lyrics && done(job),
                heard,
            }),
            Job::Cover => out.push(status(Fact::Cover, cover.filter(|_| done(job)))),
        }
    }
    out
}

impl Queue {
    fn of(state: &State, width: f32) -> Queue {
        let q = &state.queue;
        let summary = match q.current {
            Some(i) => format!("playing {} of {}", i + 1, q.paths.len()),
            None => format!("{} in the queue", q.paths.len()),
        };
        let (cursor, scroll) = list_cursor(state, Side::Queue);
        let rows = q
            .paths
            .iter()
            .enumerate()
            .skip(scroll)
            .take(list_height(state, Side::Queue))
            .map(|(i, path)| QueueRow {
                index: i,
                title: state
                    .details
                    .get(path)
                    .and_then(|d| d.title.clone())
                    .unwrap_or_else(|| {
                        path.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_default()
                    }),
                current: (q.current == Some(i)).then_some(state.playing.state),
                readiness: readiness(state, path),
                caption: crate::view::prep::caption(state, path),
                facts: if width >= FACTS_WIDTH {
                    facts(state, path)
                } else {
                    Vec::new()
                },
                known: known(state, path),
                selected: i == cursor,
            })
            .collect();
        Queue { summary, rows }
    }
}

/// An outline of the pane, one line a piece with its tone in brackets:
/// for the snapshots.
impl fmt::Display for Pane {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let names: Vec<String> = self
            .tabs
            .iter()
            .map(|t| match t.count {
                Some(n) => format!("{} ({n})", t.label),
                None => t.label.to_owned(),
            })
            .collect();
        writeln!(f, "[tabs] {}", names.join(" │ "))?;
        if let Some(shown) = self.tabs.iter().find(|t| t.shown) {
            writeln!(f, "[shown] {}", shown.label)?;
        }
        match &self.body {
            Body::Browser(b) => b.fmt(f),
            Body::Queue(q) => q.fmt(f),
            Body::Tab => Ok(()),
        }
    }
}

/// A row's chip and length in the outline.
fn known_text(known: &Known) -> String {
    match known {
        Known::Nothing => String::new(),
        Known::Pending => "  …".to_owned(),
        Known::Read(chip, length) => {
            let failed = if chip.failed { " failed" } else { "" };
            match length {
                Some(length) => format!("  [chip{failed}] {}  {length}", chip.text),
                None => format!("  [chip{failed}] {}", chip.text),
            }
        }
    }
}

impl fmt::Display for Browser {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let place = &self.place;
        let mut path = match place.found {
            Some((n, _)) => format!("{n} found below {}", place.path),
            None => place.path.clone(),
        };
        if place.found.is_some_and(|(_, searching)| searching) {
            path.push_str(", searching…");
        } else if place.listing {
            path.push_str(" (listing)");
        }
        writeln!(f, "[path] {path}")?;
        for row in &self.rows {
            let mut marks = Vec::new();
            if row.selected {
                marks.push(" selected");
            }
            if row.playing {
                marks.push(" playing");
            }
            let marks = marks.concat();
            let known = known_text(&row.known);
            match row.kind {
                Kind::Parent => writeln!(f, "[row{marks}] ..  up")?,
                Kind::Folder => writeln!(f, "[row{marks}] ▸ {}/", row.title)?,
                Kind::Playlist => writeln!(f, "[row{marks}] ≡ {}  playlist", row.title)?,
                Kind::File => {
                    let mark = if row.playing { "▶" } else { "♪" };
                    writeln!(f, "[row{marks}] {mark} {}{known}", row.title)?;
                }
                Kind::SearchBelow => writeln!(
                    f,
                    "[row{marks}] » search all subfolders for “{}”  (enter)",
                    row.title
                )?,
            }
        }
        if let Some(note) = &self.note {
            let tone = match note {
                Note::Failed(_) => "fail",
                _ => "dim",
            };
            writeln!(f, "[{tone}] {}", note.text())?;
        }
        if let Some(filter) = &self.filter {
            let caret = if filter.editing { "▏" } else { "" };
            writeln!(
                f,
                "[filter] / {}{caret}  {} shown",
                filter.typed, filter.shown
            )?;
        }
        Ok(())
    }
}

impl fmt::Display for Queue {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "[summary] {}", self.summary)?;
        if self.rows.is_empty() {
            writeln!(f, "[dim] {QUEUE_EMPTY}")?;
            writeln!(f, "[dim] {}", queue_hint())?;
        }
        for row in &self.rows {
            let selected = if row.selected { " selected" } else { "" };
            let readiness = match row.readiness {
                Readiness::Working => " working",
                Readiness::Waiting => " waiting",
                Readiness::Done => "",
            };
            let symbol = match row.current {
                Some(play) => format!("{} ", text::play_symbol(play)),
                None => String::new(),
            };
            let facts: Vec<String> = row
                .facts
                .iter()
                .map(|s| {
                    let lit = match (s.lit, s.heard) {
                        (true, true) => "!",
                        (true, false) => "*",
                        (false, _) => "",
                    };
                    format!("{}{lit}", s.fact.sign())
                })
                .collect();
            let facts = if facts.is_empty() {
                String::new()
            } else {
                format!("  {}", facts.join(" "))
            };
            let caption = match &row.caption {
                Some(c) => format!("  ‹{}›", c.text),
                None => String::new(),
            };
            writeln!(
                f,
                "[row{selected}{readiness}] {symbol}{} {}{caption}{facts}{}",
                row.index + 1,
                row.title,
                known_text(&row.known)
            )?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each part of the path names its own folder: the first a root, each
    /// the one above the next, the last the folder shown.
    #[test]
    fn each_part_of_the_path_names_its_folder() {
        let dir = std::env::temp_dir().join("Artist").join("Album");
        let (names, dirs): (Vec<String>, Vec<PathBuf>) = crumbs(&dir).into_iter().unzip();
        assert_eq!(names.last().map(String::as_str), Some("Album"));
        assert_eq!(dirs.last(), Some(&dir));
        assert_eq!(dirs[0].parent(), None, "{:?}", dirs[0]);
        for pair in dirs.windows(2) {
            assert_eq!(pair[1].parent(), Some(pair[0].as_path()));
        }
    }
}
