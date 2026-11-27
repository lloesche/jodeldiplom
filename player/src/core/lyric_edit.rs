//! The lyrics editor's actions: a song's words selected, moved, made
//! longer or shorter, marked as not to be sung, the voice's level set over
//! them, their text edited, and every change undone and redone. The
//! editor ([`LyricEditor`]) holds the song's lyrics as edited, its
//! selection, its history and its screen; it is pure.
//!
//! - **Times:** a word moved or made longer or shorter is set by hand
//!   ([`jodel_lyrics::WordEdit::pin`]): never across a word set by hand
//!   before it or after it, never shorter than [`SHORTEST`]; the words
//!   not set by hand that it now covers make room at once (the aligner fits
//!   them around it when it times the song again). Its letters keep their
//!   places in it, stretched with it.
//! - **Marks:** a word is marked as not to be sung, or not, and the voice's
//!   level over it set (dB against the voice as recorded) or cleared.
//! - **Text:** a word or a line replaced (several words where the text
//!   has several), put in before or after the one selected, or taken out.
//!   A word replaced keeps its times and marks; words put in are timed
//!   between their neighbours until the song is timed again; a line
//!   replaced by as many words keeps their times, else its new words are
//!   spread over its time.
//! - **History:** each change is a step; undo goes back one, redo forward
//!   again, until a new change.
//!
//! In the app ([`act`]): the editor opens on the song karaoke is about as
//! its screen ([`Screen::Lyrics`], which holds it: [`screen`]), and each
//! change is shown at once (the song's lyrics as shown are the editor's, so
//! its bubbles' notes are read from the melody over the new times) and kept
//! ([`Effect::SaveLyricEdits`]): the lyrics' worker keeps the edits and
//! times the song again with them, and the song timed comes back as any
//! lyrics do.

use std::path::PathBuf;
use std::sync::Arc;

use jodel_lyrics::{Line, Lyrics, Word, WordAt};

use super::karaoke::{self, SongLyrics};
use super::{Effect, Screen, State};
use crate::core::timeview::PanInput;
use screen::{Ctx, Draft, Held};

pub mod screen;

/// The shortest a word can be made by hand, in seconds.
pub const SHORTEST: f64 = 0.05;

/// What a press on a word holds: its start, its middle (all of it) or its
/// end; what a key nudges.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    /// Its start: dragged, its start moves.
    Start,
    /// All of it: dragged, it moves.
    Middle,
    /// Its end: dragged, its end moves.
    End,
}

/// Another word selected.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Selection {
    /// This one.
    Word(WordAt),
    /// The next word (over lines).
    Next,
    /// The word before.
    Previous,
}

/// A change to the lyrics at the word selected (or its line).
#[derive(Clone, Debug, PartialEq)]
pub enum Change {
    /// Moves its start by this many seconds (its end stays).
    NudgeStart(f64),
    /// Moves its end by this many seconds (its start stays).
    NudgeEnd(f64),
    /// Moves it whole by this many seconds.
    Move(f64),
    /// Sets its start and end, in seconds into the song.
    SetTimes(f64, f64),
    /// Marks it as not to be sung, or to be sung.
    NotSung(bool),
    /// Sets the voice's level over it, in dB against the voice as recorded
    /// (`None`: as the karaoke's voice level has it).
    Voice(Option<f32>),
    /// Replaces its text (several words where there are several; none takes
    /// it out).
    ReplaceWord(String),
    /// Puts the words of a text in after it, or before it.
    InsertWord {
        /// After it, else before.
        after: bool,
        /// The text.
        text: String,
    },
    /// Takes it out (its line too, when it was the last word).
    DeleteWord,
    /// Replaces its line's words with those of a text (none takes the line
    /// out).
    ReplaceLine(String),
    /// Puts a line of a text's words in after its line, or before it.
    InsertLine {
        /// After it, else before.
        after: bool,
        /// The text.
        text: String,
    },
    /// Takes its line out.
    DeleteLine,
}

impl Change {
    /// The change that moves `part` of a word by `by` seconds: its start,
    /// all of it, or its end.
    pub fn nudge(part: Part, by: f64) -> Change {
        match part {
            Part::Start => Change::NudgeStart(by),
            Part::Middle => Change::Move(by),
            Part::End => Change::NudgeEnd(by),
        }
    }
}

/// One action of the editor's.
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// Selects another word.
    Select(Selection),
    /// Changes the lyrics, one step of the history.
    Change(Change),
    /// Goes back a change.
    Undo,
    /// Goes forward a change undone.
    Redo,
}

/// The editor on one song's lyrics.
#[derive(Clone, Debug, PartialEq)]
pub struct LyricEditor {
    /// The song.
    pub path: PathBuf,
    /// Its lyrics as edited, timed as shown.
    pub lyrics: Lyrics,
    /// The word selected.
    pub selected: Option<WordAt>,
    /// What its screen shows and holds.
    pub screen: screen::EditorScreen,
    undo: Vec<Lyrics>,
    redo: Vec<Lyrics>,
}

/// The places of every word of `lyrics`, in order.
fn places(lyrics: &Lyrics) -> Vec<WordAt> {
    lyrics.words().map(|(at, _)| at).collect()
}

/// The words of `text`, as written, untimed.
fn words_of(text: &str) -> Vec<Word> {
    text.split_whitespace().map(Word::new).collect()
}

impl LyricEditor {
    /// The editor on `lyrics` (the song at `path`'s, as shown), its first
    /// word selected.
    pub fn open(path: PathBuf, lyrics: &Lyrics) -> LyricEditor {
        let first = lyrics.words().next();
        LyricEditor {
            path,
            selected: first.map(|(at, _)| at),
            screen: screen::EditorScreen::at(first.and_then(|(_, w)| w.start)),
            lyrics: lyrics.clone(),
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// The word selected.
    pub fn word(&self) -> Option<&Word> {
        self.lyrics.word(self.selected?)
    }

    /// Whether there is a change to undo.
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }

    /// Whether there is a change undone to redo.
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }

    /// Takes the song as timed again with its edits (`timed`): the edits
    /// are kept as they were, the times the aligner fitted around them in
    /// place of those shown. Not when they are other edits (made since).
    pub fn timed_again(&mut self, timed: &Lyrics) -> bool {
        if timed.edits() != self.lyrics.edits() {
            return false;
        }
        self.lyrics = timed.clone();
        true
    }

    /// Carries out `edit`. Whether the lyrics changed (a selection does
    /// not change them).
    pub fn apply(&mut self, edit: Edit) -> bool {
        match edit {
            Edit::Select(selection) => {
                self.select(selection);
                false
            }
            Edit::Undo => self.step(true),
            Edit::Redo => self.step(false),
            Edit::Change(change) => {
                let before = (self.lyrics.clone(), self.selected);
                if self.change(change) && self.lyrics != before.0 {
                    self.undo.push(before.0);
                    self.redo.clear();
                    true
                } else {
                    (self.lyrics, self.selected) = before;
                    false
                }
            }
        }
    }

    /// Selects another word.
    fn select(&mut self, selection: Selection) {
        let all = places(&self.lyrics);
        let k = self.selected.and_then(|s| all.iter().position(|w| *w == s));
        self.selected = match selection {
            Selection::Word(at) => all.contains(&at).then_some(at).or(self.selected),
            Selection::Next => k
                .map_or(all.first(), |k| all.get(k + 1).or(all.get(k)))
                .copied(),
            Selection::Previous => k
                .map_or(all.first(), |k| all.get(k.saturating_sub(1)))
                .copied(),
        };
    }

    /// While a word is dragged: `change` (a nudge or a move of the word
    /// selected) made to `from` (the lyrics as the drag found them) in
    /// place of what is shown, out of the history until the drag ends
    /// ([`LyricEditor::dragged`]). Whether it could be made; where it could
    /// not, the lyrics are `from`.
    pub fn drag(&mut self, from: &Lyrics, change: Change) -> bool {
        self.lyrics = from.clone();
        let made = self.change(change);
        if !made {
            self.lyrics = from.clone();
        }
        made
    }

    /// The drag that found the lyrics as `from` ended: what it changed one
    /// step of the history. Whether it changed anything.
    pub fn dragged(&mut self, from: Lyrics) -> bool {
        if self.lyrics == from {
            return false;
        }
        self.undo.push(from);
        self.redo.clear();
        true
    }

    /// Undoes (`back`) or redoes a change.
    fn step(&mut self, back: bool) -> bool {
        let (from, to) = if back {
            (&mut self.undo, &mut self.redo)
        } else {
            (&mut self.redo, &mut self.undo)
        };
        let Some(lyrics) = from.pop() else {
            return false;
        };
        to.push(std::mem::replace(&mut self.lyrics, lyrics));
        let all = places(&self.lyrics);
        if !self.selected.is_some_and(|s| all.contains(&s)) {
            self.selected = all.last().copied();
        }
        true
    }

    /// A change to the lyrics (not a step of the history): whether it could
    /// be made.
    fn change(&mut self, change: Change) -> bool {
        let Some(at) = self.selected else {
            return false;
        };
        match change {
            Change::NudgeStart(by) => {
                self.retime(at, |a, b, lo, _| ((a + by).min(b - SHORTEST).max(lo), b))
            }
            Change::NudgeEnd(by) => {
                self.retime(at, |a, b, _, hi| (a, (b + by).max(a + SHORTEST).min(hi)))
            }
            Change::Move(by) => self.retime(at, |a, b, lo, hi| {
                let by = by.min(hi - b).max(lo - a);
                (a + by, b + by)
            }),
            Change::SetTimes(a, b) => self.retime(at, |_, _, lo, hi| (a.max(lo), b.min(hi))),
            Change::NotSung(on) => {
                self.lyrics[at].edit.not_sung = on;
                true
            }
            Change::Voice(db) => {
                self.lyrics[at].edit.voice_db = db.filter(|d| d.is_finite());
                true
            }
            Change::ReplaceWord(text) => {
                let mut new = words_of(&text);
                if new.is_empty() {
                    return self.change(Change::DeleteWord);
                }
                let line = &mut self.lyrics.lines[at.line];
                line.words[at.word].retext(new.remove(0).text);
                line.words.splice(at.word + 1..at.word + 1, new);
                self.fill_untimed();
                true
            }
            Change::InsertWord { after, text } => {
                let new = words_of(&text);
                if new.is_empty() {
                    return false;
                }
                let word = at.word + usize::from(after);
                self.lyrics.lines[at.line].words.splice(word..word, new);
                self.selected = Some(WordAt { word, ..at });
                self.fill_untimed();
                true
            }
            Change::DeleteWord => {
                let k = places(&self.lyrics).iter().position(|w| *w == at);
                let line = &mut self.lyrics.lines[at.line];
                line.words.remove(at.word);
                if line.words.is_empty() {
                    self.lyrics.lines.remove(at.line);
                }
                let all = places(&self.lyrics);
                self.selected = k.and_then(|k| all.get(k).or(all.last())).copied();
                self.fill_untimed();
                true
            }
            Change::ReplaceLine(text) => {
                let mut new = words_of(&text);
                if new.is_empty() {
                    return self.change(Change::DeleteLine);
                }
                let line = &mut self.lyrics.lines[at.line];
                if new.len() == line.words.len() {
                    for (w, n) in line.words.iter_mut().zip(new) {
                        w.retext(n.text);
                    }
                } else {
                    let start = line.words.iter().find_map(|w| w.start);
                    let end = line.words.iter().rev().find_map(|w| w.end);
                    if let (Some(a), Some(b)) = (start, end) {
                        // By their letters.
                        let weights: Vec<f64> =
                            new.iter().map(|w| w.text.chars().count() as f64).collect();
                        jodel_lyrics::spread::across(&mut new, a, b - a, &weights);
                    }
                    line.words = new;
                }
                line.is_break = false;
                self.selected = Some(WordAt { word: 0, ..at });
                self.fill_untimed();
                true
            }
            Change::InsertLine { after, text } => {
                let words = words_of(&text);
                if words.is_empty() {
                    return false;
                }
                let line = at.line + usize::from(after);
                let new = Line {
                    words,
                    ..Line::default()
                };
                self.lyrics.lines.insert(line, new);
                self.selected = Some(WordAt { line, word: 0 });
                self.fill_untimed();
                true
            }
            Change::DeleteLine => {
                self.lyrics.lines.remove(at.line);
                let all = places(&self.lyrics);
                self.selected = all
                    .iter()
                    .find(|w| w.line >= at.line)
                    .or(all.last())
                    .copied();
                true
            }
        }
    }

    /// Sets the times of the word at `at` (a timed one) to what `to` makes
    /// of its start, its end, and the earliest and latest it may reach (the
    /// end of the word set by hand before it, the start of the one after):
    /// set by hand, its letters stretched with it, the words not set by
    /// hand under it making room. Not when it would be shorter than
    /// [`SHORTEST`].
    fn retime(&mut self, at: WordAt, to: impl FnOnce(f64, f64, f64, f64) -> (f64, f64)) -> bool {
        let all = places(&self.lyrics);
        let k = all.iter().position(|w| *w == at).unwrap_or(0);
        let word = |w: &WordAt| &self.lyrics[*w];
        let Some((a, b)) = word(&at).start.zip(word(&at).end) else {
            return false;
        };
        let lo = all[..k]
            .iter()
            .rev()
            .find_map(|w| word(w).edit.pin.map(|p| p.1))
            .unwrap_or(0.0);
        let hi = all[k + 1..]
            .iter()
            .find_map(|w| word(w).edit.pin.map(|p| p.0))
            .unwrap_or(f64::INFINITY);
        let (a2, b2) = to(a, b, lo, hi);
        let fits = a2 >= lo - 1e-9 && b2 <= hi + 1e-9 && b2 - a2 >= SHORTEST - 1e-9;
        if !(a2.is_finite() && b2.is_finite() && fits) {
            return false;
        }
        let w = &mut self.lyrics[at];
        let stretch = ((b2 - a2) / (b - a).max(1e-9)) as f32;
        for t in &mut w.letters {
            *t *= stretch;
        }
        w.pin_to(a2, b2);
        make_room(&mut self.lyrics, &all, k, (a2, b2));
        true
    }

    /// After a change of the text: each line's backing words marked again,
    /// and the words without times timed between their neighbours.
    fn fill_untimed(&mut self) {
        for line in &mut self.lyrics.lines {
            line.mark_backing();
        }
        jodel_lyrics::spread::interpolate(&mut self.lyrics.lines, 0.0);
    }

    /// The word selected brought into the screen's view where it is not in
    /// it.
    fn reveal(&mut self, ctx: &Ctx) {
        if let Some(times) = self.word().and_then(|w| w.start.zip(w.end)) {
            self.screen.reveal(ctx, times);
        }
    }
}

/// The words not set by hand around word `k` of `all` (the words of
/// `lyrics`, in order), now from `a` to `b` seconds, made to end by `a`
/// (those before it) or start from `b` (those after), as far as they
/// reach into it; their letters kept where they were sung, within them.
fn make_room(lyrics: &mut Lyrics, all: &[WordAt], k: usize, (a, b): (f64, f64)) {
    for at in all[..k].iter().rev() {
        let w = &mut lyrics[*at];
        let Some(end) = w.end.filter(|_| w.edit.pin.is_none()) else {
            continue;
        };
        if end <= a {
            break;
        }
        let start = w.start.map(|s| s.min(a));
        (w.start, w.end) = (start, Some(a));
        w.clip_letters();
    }
    for at in &all[k + 1..] {
        let w = &mut lyrics[*at];
        let Some(start) = w.start.filter(|_| w.edit.pin.is_none()) else {
            continue;
        };
        if start >= b {
            break;
        }
        // Its letters as far after its new start as they were after b.
        let later = (b - start) as f32;
        for t in &mut w.letters {
            *t = (*t - later).max(0.0);
        }
        let end = w.end.map(|e| e.max(b));
        (w.start, w.end) = (Some(b), end);
        w.clip_letters();
    }
}

/// What the app does with the editor open ([`open`]).
#[derive(Clone, Debug, PartialEq)]
pub enum LyricsEdit {
    /// Closes it, back to the main screen.
    Close,
    /// An action of the editor's.
    Edit(Edit),
    /// Asks whether every edit is to be forgotten (`true`), or calls the
    /// question off.
    AskRevert(bool),
    /// Forgets every edit of the song: its text as found, timed again
    /// (the editor closes).
    Revert,
    /// A press on `word`'s `part`, `time` seconds into the song under the
    /// pointer: it is selected and held; the pointer may go `slop` seconds
    /// from there and change nothing.
    Hold {
        /// The word.
        word: WordAt,
        /// What of it.
        part: Part,
        /// Under the pointer.
        time: f64,
        /// How far it may tremble.
        slop: f64,
    },
    /// The pointer holding a word is `time` seconds into the song.
    Drag(f64),
    /// It let go of it.
    Drop,
    /// A press on the waveform off the words: the playhead goes to `time`
    /// seconds (the song playing, or paused).
    Seek(f64),
    /// Write the word selected's text, or its whole `line`'s.
    Write {
        /// Its line's.
        line: bool,
    },
    /// The text as written now.
    Draft(String),
    /// What was written kept.
    Keep,
    /// What was written left, the text as it was.
    Discard,
    /// The wheel `steps` up (in) or down (out) over the lane, the time `at`
    /// (seconds) under the pointer staying where it is.
    Zoom {
        /// Steps, in while positive.
        steps: i32,
        /// Under the pointer.
        at: f64,
    },
}

/// A change, as the app's action.
impl From<Change> for LyricsEdit {
    fn from(change: Change) -> LyricsEdit {
        LyricsEdit::Edit(Edit::Change(change))
    }
}

/// The lyrics editor, while it is open (its screen shows).
pub fn editor(state: &State) -> Option<&LyricEditor> {
    match &state.screen {
        Screen::Lyrics(editor) => Some(editor),
        _ => None,
    }
}

/// The lyrics editor, to change, while it is open.
fn editor_mut(state: &mut State) -> Option<&mut LyricEditor> {
    match &mut state.screen {
        Screen::Lyrics(editor) => Some(editor),
        _ => None,
    }
}

/// Carries out `what` for the app: a change shown at once on the song's
/// lyrics and kept ([`Effect::SaveLyricEdits`]).
pub(crate) fn act(state: &mut State, what: LyricsEdit, fx: &mut Vec<Effect>) {
    let Some(ctx) = Ctx::of(state) else {
        return;
    };
    // A press anywhere on the lane while a text is written keeps it.
    let writing = editor(state).is_some_and(|e| e.screen.writing());
    if writing && matches!(what, LyricsEdit::Hold { .. } | LyricsEdit::Seek(_)) {
        return act(state, LyricsEdit::Keep, fx);
    }
    let Some(editor) = editor_mut(state) else {
        return;
    };
    // Anything else calls the question of forgetting every edit off.
    if !matches!(what, LyricsEdit::AskRevert(_) | LyricsEdit::Revert) {
        editor.screen.asking = false;
    }
    match what {
        LyricsEdit::Close => state.screen = Screen::Main,
        LyricsEdit::AskRevert(on) => editor.screen.asking = on,
        LyricsEdit::Revert => {
            fx.push(Effect::SaveLyricEdits(editor.path.clone(), None));
            state.screen = Screen::Main;
            state.info("Every edit forgotten: the lyrics as found, timed again");
        }
        LyricsEdit::Edit(edit) => {
            let changed = editor.apply(edit);
            editor.reveal(&ctx);
            if changed {
                kept(state, fx);
            }
        }
        LyricsEdit::Hold {
            word,
            part,
            time,
            slop,
        } => {
            editor.screen.hold_view(&ctx);
            editor.apply(Edit::Select(Selection::Word(word)));
            if editor.selected == Some(word) {
                editor.screen.held = Some(Held {
                    part,
                    from: editor.lyrics.clone(),
                    time,
                    slop,
                    moved: false,
                });
            }
        }
        LyricsEdit::Drag(time) => {
            let Some(held) = &mut editor.screen.held else {
                return;
            };
            let by = time - held.time;
            if !held.moved && by.abs() < held.slop {
                return;
            }
            held.moved = true;
            let (change, from) = (Change::nudge(held.part, by), held.from.clone());
            // Where the pointer leads nowhere it may go (it is not timed),
            // the drag changes nothing.
            editor.drag(&from, change);
        }
        LyricsEdit::Drop => {
            let Some(held) = editor.screen.held.take() else {
                return;
            };
            if held.moved {
                if editor.dragged(held.from) {
                    kept(state, fx);
                }
                return;
            }
            // A click: a second one on the word soon after writes it.
            let clicked = editor
                .selected
                .is_some_and(|w| editor.screen.clicked.press(w, ctx.now));
            if clicked {
                act(state, LyricsEdit::Write { line: false }, fx);
            }
        }
        LyricsEdit::Seek(time) => {
            if ctx.playhead.is_some() {
                let t = time.max(0.0);
                state.playing.position = t;
                fx.push(Effect::Engine(jodel_playback::Command::Seek(t)));
            }
        }
        LyricsEdit::Write { line } => {
            let Some(at) = editor.selected else {
                return;
            };
            let text = if line {
                editor.lyrics.lines[at.line].text()
            } else {
                editor.lyrics[at].text.clone()
            };
            editor.screen.draft = Some(Draft { line, text });
        }
        LyricsEdit::Draft(text) => {
            if let Some(draft) = &mut editor.screen.draft {
                draft.text = text;
            }
        }
        LyricsEdit::Keep => {
            let Some(draft) = editor.screen.draft.take() else {
                return;
            };
            let change = if draft.line {
                Change::ReplaceLine(draft.text)
            } else {
                Change::ReplaceWord(draft.text)
            };
            act(state, change.into(), fx);
        }
        LyricsEdit::Discard => editor.screen.draft = None,
        LyricsEdit::Zoom { steps, at } => editor.screen.zoom(&ctx, steps, at),
    }
}

/// What the secondary button does to the editor's view (open); as any
/// action there, it calls the question of forgetting every edit off.
pub(crate) fn pan(state: &mut State, input: PanInput) {
    let Some(ctx) = Ctx::of(state) else {
        return;
    };
    if let Some(editor) = editor_mut(state) {
        editor.screen.asking = false;
        editor.screen.pan(&ctx, input);
    }
}

/// Opens the editor on the song karaoke is about, as the screen, where its
/// lyrics are found.
pub(crate) fn open(state: &mut State) {
    let Some(path) = karaoke::song_index(state).and_then(|i| state.queue.paths.get(i)) else {
        return state.info("No song to edit the lyrics of");
    };
    match state.karaoke.lyrics.get(path) {
        Some(SongLyrics::Found { shown, .. }) => {
            let editor = LyricEditor::open(path.clone(), shown);
            state.screen = Screen::Lyrics(Box::new(editor));
        }
        _ => {
            let title = super::shown_title(state, path);
            state.info(format!("No lyrics of {title} to edit yet"));
        }
    }
}

/// The editor's lyrics changed: shown at once as the song's, and kept.
fn kept(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(editor) = editor(state) else {
        return;
    };
    let (path, lyrics) = (editor.path.clone(), editor.lyrics.clone());
    if let Some(SongLyrics::Found { shown, .. }) = state.karaoke.lyrics.get_mut(&path) {
        *shown = Arc::new(lyrics.clone());
    }
    fx.push(Effect::SaveLyricEdits(path, Some(lyrics.edits())));
}

/// A song's lyrics arrived: the editor on it takes them when they are its
/// edits timed again ([`LyricEditor::timed_again`]).
pub(crate) fn arrived(state: &mut State, path: &std::path::Path) {
    let Some(SongLyrics::Found { shown, .. }) = state.karaoke.lyrics.get(path) else {
        return;
    };
    let shown = Arc::clone(shown);
    if let Some(editor) = editor_mut(state).filter(|e| e.path == path) {
        editor.timed_again(&shown);
    }
}

#[cfg(test)]
mod tests;
