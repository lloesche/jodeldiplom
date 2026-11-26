//! The lyrics editor's actions, without its screen: a song's words
//! selected, moved, made longer or shorter, marked as not to be sung, the
//! voice's level set over them, their text edited, and every change undone
//! and redone. Pure: the editor ([`LyricEditor`]) holds the song's lyrics
//! as edited, its selection and its history.
//!
//! - **Times:** a word moved or made longer or shorter is set by hand
//!   ([`jodel_lyrics::Word::pinned`]): never across a word set by hand
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
//! In the app ([`act`]): the editor opens on the song karaoke is about, and
//! each change is shown at once (the song's lyrics as shown are the
//! editor's, so its bubbles' notes are read from the melody over the new
//! times) and kept ([`Effect::SaveLyricEdits`]): the lyrics' worker keeps
//! the edits and times the song again with them, and the song timed comes
//! back as any lyrics do.

use std::path::PathBuf;
use std::sync::Arc;

use jodel_lyrics::{Line, Lyrics, Word, GIVEN};

use super::karaoke::{self, SongLyrics};
use super::{Effect, State};
use crate::lyrics::edits::edits_of;

/// The shortest a word can be made by hand, in seconds.
pub const SHORTEST: f64 = 0.05;

/// A word of the lyrics: its line, and its place in the line.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordAt {
    /// The line.
    pub line: usize,
    /// The word in it.
    pub word: usize,
}

/// One action of the editor's on the word selected (or its line).
#[derive(Clone, Debug, PartialEq)]
pub enum Edit {
    /// Selects a word.
    Select(WordAt),
    /// Selects the next word (over lines).
    Next,
    /// Selects the word before.
    Previous,
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
    undo: Vec<Lyrics>,
    redo: Vec<Lyrics>,
}

/// Every word of `lyrics`, in order.
fn words_in(lyrics: &Lyrics) -> Vec<WordAt> {
    (lyrics.lines.iter().enumerate())
        .flat_map(|(line, l)| (0..l.words.len()).map(move |word| WordAt { line, word }))
        .collect()
}

/// The words of `text`, as written, untimed.
fn words_of(text: &str) -> Vec<Word> {
    text.split_whitespace().map(Word::new).collect()
}

impl LyricEditor {
    /// The editor on `lyrics` (the song at `path`'s, as shown), its first
    /// word selected.
    pub fn open(path: PathBuf, lyrics: &Lyrics) -> LyricEditor {
        LyricEditor {
            path,
            selected: words_in(lyrics).first().copied(),
            lyrics: lyrics.clone(),
            undo: Vec::new(),
            redo: Vec::new(),
        }
    }

    /// The word selected.
    pub fn word(&self) -> Option<&Word> {
        let at = self.selected?;
        self.lyrics.lines.get(at.line)?.words.get(at.word)
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
        if edits_of(timed) != edits_of(&self.lyrics) {
            return false;
        }
        self.lyrics = timed.clone();
        true
    }

    /// Carries out `edit`. Whether the lyrics changed (a selection does
    /// not change them).
    pub fn apply(&mut self, edit: Edit) -> bool {
        let all = words_in(&self.lyrics);
        let k = self.selected.and_then(|s| all.iter().position(|w| *w == s));
        match edit {
            Edit::Select(at) => {
                self.selected = all.contains(&at).then_some(at).or(self.selected);
                false
            }
            Edit::Next => {
                self.selected = k
                    .map_or(all.first(), |k| all.get(k + 1).or(all.get(k)))
                    .copied();
                false
            }
            Edit::Previous => {
                self.selected = k
                    .map_or(all.first(), |k| all.get(k.saturating_sub(1)))
                    .copied();
                false
            }
            Edit::Undo => self.step(true),
            Edit::Redo => self.step(false),
            edit => {
                let before = (self.lyrics.clone(), self.selected);
                if self.change(edit) && self.lyrics != before.0 {
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
        let all = words_in(&self.lyrics);
        if !self.selected.is_some_and(|s| all.contains(&s)) {
            self.selected = all.last().copied();
        }
        true
    }

    /// A change to the lyrics (not a step of the history): whether it could
    /// be made.
    fn change(&mut self, edit: Edit) -> bool {
        let Some(at) = self.selected else {
            return false;
        };
        match edit {
            Edit::NudgeStart(by) => {
                self.retime(at, |a, b, lo, _| ((a + by).min(b - SHORTEST).max(lo), b))
            }
            Edit::NudgeEnd(by) => {
                self.retime(at, |a, b, _, hi| (a, (b + by).max(a + SHORTEST).min(hi)))
            }
            Edit::Move(by) => self.retime(at, |a, b, lo, hi| {
                let by = by.min(hi - b).max(lo - a);
                (a + by, b + by)
            }),
            Edit::SetTimes(a, b) => self.retime(at, |_, _, lo, hi| (a.max(lo), b.min(hi))),
            Edit::NotSung(on) => {
                self.lyrics.lines[at.line].words[at.word].not_sung = on;
                true
            }
            Edit::Voice(db) => {
                self.lyrics.lines[at.line].words[at.word].voice_db = db.filter(|d| d.is_finite());
                true
            }
            Edit::ReplaceWord(text) => {
                let mut new = words_of(&text);
                if new.is_empty() {
                    return self.change(Edit::DeleteWord);
                }
                let line = &mut self.lyrics.lines[at.line];
                let w = &mut line.words[at.word];
                let first = new.remove(0);
                if w.text != first.text {
                    w.text = first.text;
                    w.letters.clear();
                }
                line.words.splice(at.word + 1..at.word + 1, new);
                self.settle();
                true
            }
            Edit::InsertWord { after, text } => {
                let new = words_of(&text);
                if new.is_empty() {
                    return false;
                }
                let word = at.word + usize::from(after);
                self.lyrics.lines[at.line].words.splice(word..word, new);
                self.selected = Some(WordAt { word, ..at });
                self.settle();
                true
            }
            Edit::DeleteWord => {
                let k = words_in(&self.lyrics).iter().position(|w| *w == at);
                let line = &mut self.lyrics.lines[at.line];
                line.words.remove(at.word);
                if line.words.is_empty() {
                    self.lyrics.lines.remove(at.line);
                }
                let all = words_in(&self.lyrics);
                self.selected = k.and_then(|k| all.get(k).or(all.last())).copied();
                self.settle();
                true
            }
            Edit::ReplaceLine(text) => {
                let mut new = words_of(&text);
                if new.is_empty() {
                    return self.change(Edit::DeleteLine);
                }
                let line = &mut self.lyrics.lines[at.line];
                if new.len() == line.words.len() {
                    for (w, n) in line.words.iter_mut().zip(new) {
                        if w.text != n.text {
                            w.text = n.text;
                            w.letters.clear();
                        }
                    }
                } else {
                    let start = line.words.iter().find_map(|w| w.start);
                    let end = line.words.iter().rev().find_map(|w| w.end);
                    if let (Some(a), Some(b)) = (start, end) {
                        spread(&mut new, a, b);
                    }
                    line.words = new;
                }
                line.is_break = false;
                self.selected = Some(WordAt { word: 0, ..at });
                self.settle();
                true
            }
            Edit::InsertLine { after, text } => {
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
                self.settle();
                true
            }
            Edit::DeleteLine => {
                self.lyrics.lines.remove(at.line);
                let all = words_in(&self.lyrics);
                self.selected = all
                    .iter()
                    .find(|w| w.line >= at.line)
                    .or(all.last())
                    .copied();
                true
            }
            Edit::Select(_) | Edit::Next | Edit::Previous | Edit::Undo | Edit::Redo => false,
        }
    }

    /// Sets the times of the word at `at` (a timed one) to what `to` makes
    /// of its start, its end, and the earliest and latest it may reach (the
    /// end of the word set by hand before it, the start of the one after):
    /// set by hand, its letters stretched with it, the words not set by
    /// hand under it making room. Not when it would be shorter than
    /// [`SHORTEST`].
    fn retime(&mut self, at: WordAt, to: impl FnOnce(f64, f64, f64, f64) -> (f64, f64)) -> bool {
        let all = words_in(&self.lyrics);
        let k = all.iter().position(|w| *w == at).unwrap_or(0);
        let word = |w: &WordAt| &self.lyrics.lines[w.line].words[w.word];
        let Some((a, b)) = word(&at).start.zip(word(&at).end) else {
            return false;
        };
        let lo = all[..k]
            .iter()
            .rev()
            .find_map(|w| word(w).pin().map(|p| p.1))
            .unwrap_or(0.0);
        let hi = all[k + 1..]
            .iter()
            .find_map(|w| word(w).pin().map(|p| p.0))
            .unwrap_or(f64::INFINITY);
        let (a2, b2) = to(a, b, lo, hi);
        let fits = a2 >= lo - 1e-9 && b2 <= hi + 1e-9 && b2 - a2 >= SHORTEST - 1e-9;
        if !(a2.is_finite() && b2.is_finite() && fits) {
            return false;
        }
        let w = &mut self.lyrics.lines[at.line].words[at.word];
        let stretch = ((b2 - a2) / (b - a).max(1e-9)) as f32;
        for t in &mut w.letters {
            *t *= stretch;
        }
        (w.start, w.end, w.confidence, w.pinned) = (Some(a2), Some(b2), GIVEN, true);
        make_room(&mut self.lyrics, &all, k, (a2, b2));
        true
    }

    /// After a change of the text: each line's backing words marked again,
    /// and the words without times timed between their neighbours.
    fn settle(&mut self) {
        for line in &mut self.lyrics.lines {
            line.mark_backing();
        }
        jodel_lyrics::spread::interpolate(&mut self.lyrics.lines, 0.0);
    }
}

/// `words` spread from `a` to `b` seconds, by their letters.
fn spread(words: &mut [Word], a: f64, b: f64) {
    let total: usize = words.iter().map(|w| w.text.chars().count()).sum();
    let mut at = 0;
    for w in words {
        let from = a + (b - a) * at as f64 / total.max(1) as f64;
        at += w.text.chars().count();
        let to = a + (b - a) * at as f64 / total.max(1) as f64;
        (w.start, w.end) = (Some(from), Some(to));
    }
}

/// The words not set by hand around word `k` of `all` (the words of
/// `lyrics`, in order), now from `a` to `b` seconds, made to end by `a`
/// (those before it) or start from `b` (those after), as far as they
/// reach into it.
fn make_room(lyrics: &mut Lyrics, all: &[WordAt], k: usize, (a, b): (f64, f64)) {
    for at in all[..k].iter().rev() {
        let w = &mut lyrics.lines[at.line].words[at.word];
        let Some(end) = w.end.filter(|_| !w.pinned) else {
            continue;
        };
        if end <= a {
            break;
        }
        let start = w.start.map(|s| s.min(a));
        (w.start, w.end) = (start, Some(a));
        clip_letters(w);
    }
    for at in &all[k + 1..] {
        let w = &mut lyrics.lines[at.line].words[at.word];
        let Some(start) = w.start.filter(|_| !w.pinned) else {
            continue;
        };
        if start >= b {
            break;
        }
        let end = w.end.map(|e| e.max(b));
        (w.start, w.end) = (Some(b), end);
        w.letters.clear();
    }
}

/// A word's letters kept within its length.
fn clip_letters(w: &mut Word) {
    let most = w.start.zip(w.end).map_or(0.0, |(s, e)| (e - s).max(0.0)) as f32;
    for t in &mut w.letters {
        *t = t.min(most);
    }
}

/// What the app does with the editor.
#[derive(Clone, Debug, PartialEq)]
pub enum LyricsEdit {
    /// Opens it on the song karaoke is about (its lyrics found).
    Open,
    /// Closes it.
    Close,
    /// An action of the editor's.
    Edit(Edit),
    /// Forgets every edit of the song: its text as found, timed again
    /// (the editor closes).
    Revert,
}

/// Carries out `what` for the app: a change shown at once on the song's
/// lyrics and kept ([`Effect::SaveLyricEdits`]).
pub(crate) fn act(state: &mut State, what: LyricsEdit, fx: &mut Vec<Effect>) {
    match what {
        LyricsEdit::Open => {
            let Some(path) = karaoke::song_index(state).and_then(|i| state.queue.paths.get(i))
            else {
                return;
            };
            if let Some(SongLyrics::Found { shown, .. }) = state.karaoke.lyrics.get(path) {
                state.karaoke.editor = Some(LyricEditor::open(path.clone(), shown));
            }
        }
        LyricsEdit::Close => state.karaoke.editor = None,
        LyricsEdit::Revert => {
            if let Some(editor) = state.karaoke.editor.take() {
                fx.push(Effect::SaveLyricEdits(editor.path, None));
            }
        }
        LyricsEdit::Edit(edit) => {
            let Some(editor) = &mut state.karaoke.editor else {
                return;
            };
            if !editor.apply(edit) {
                return;
            }
            let (path, lyrics) = (editor.path.clone(), editor.lyrics.clone());
            if let Some(SongLyrics::Found { shown, .. }) = state.karaoke.lyrics.get_mut(&path) {
                *shown = Arc::new(lyrics.clone());
            }
            fx.push(Effect::SaveLyricEdits(path, Some(edits_of(&lyrics))));
        }
    }
}

/// A song's lyrics arrived: the editor on it takes them when they are its
/// edits timed again ([`LyricEditor::timed_again`]).
pub(crate) fn arrived(state: &mut State, path: &std::path::Path) {
    let Some(editor) = state.karaoke.editor.as_mut().filter(|e| e.path == path) else {
        return;
    };
    if let Some(SongLyrics::Found { shown, .. }) = state.karaoke.lyrics.get(path) {
        editor.timed_again(shown);
    }
}

#[cfg(test)]
mod tests;
