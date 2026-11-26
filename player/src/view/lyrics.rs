//! The lyrics editor as the window shows it ([`crate::gui`]'s lyrics
//! screen): its header (the song, whether its lyrics are edited by hand,
//! undo and redo, Revert and Done), the time across the view, the words in
//! view over the voice's waveform as boxes ([`WordBox`]), the word
//! selected ([`Selected`]) and its line ([`LineWord`]), the text being
//! written, and the keys. Its outline ([`Editor`]'s `Display`) is what the
//! shots write beside the picture.

use std::fmt;
use std::sync::Arc;

use jodel_lyrics::Line;
use jodel_playback::Stem;
use jodel_rhythm::waveform::Waveform;

use super::karaoke::Hint;
use crate::core::karaoke::SongLyrics;
use crate::core::keymap::{self, Context, Op};
use crate::core::lanes::Lane;
use crate::core::lyric_edit::screen::{self, Draft, Part};
use crate::core::lyric_edit::{LyricEditor, WordAt};
use crate::core::{sliders, text, voice_curve, Rhythm, State};

/// The button that closes the editor.
pub const DONE: &str = "Done";
/// The button that forgets every edit.
pub const REVERT: &str = "Revert";
/// What the editor asks before it forgets every edit.
pub const ASK: &str = "Forget every edit of this song: its text as found, timed again?";

/// A word in view, as a box over the waveform.
#[derive(Clone, Debug, PartialEq)]
pub struct WordBox {
    /// The word.
    pub at: WordAt,
    /// Its text.
    pub text: String,
    /// When it starts and ends, in seconds into the song.
    pub start: f64,
    /// Ends.
    pub end: f64,
    /// It is selected.
    pub selected: bool,
    /// The pointer holds it, by this part.
    pub held: Option<Part>,
    /// Its times were set by hand.
    pub pinned: bool,
    /// It is not to be sung.
    pub not_sung: bool,
    /// It is heard now (the playhead in it).
    pub heard: bool,
    /// The voice's level over it in Karaoke mode, in dB against the voice
    /// as recorded.
    pub level: f64,
    /// That level is its own (set over it, or raised as it is not sung),
    /// not the slider's.
    pub own: bool,
}

/// The word selected, as the panel under the waveform says it.
#[derive(Clone, Debug, PartialEq)]
pub struct Selected {
    /// Its text.
    pub text: String,
    /// Its times, as `1:23.45`, and its length (`0.62 s`).
    pub times: String,
    /// Its times were set by hand.
    pub pinned: bool,
    /// It is not to be sung.
    pub not_sung: bool,
    /// The voice over it in Karaoke mode: `−6 dB, set here`.
    pub level: String,
    /// A level is set over it.
    pub own_level: bool,
}

/// A word of the selected word's line.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LineWord {
    /// The word.
    pub at: WordAt,
    /// Its text.
    pub text: String,
    /// It is the one selected.
    pub selected: bool,
    /// It is not to be sung.
    pub not_sung: bool,
}

/// The voice's waveform under the words: the track's waveform and which
/// lane of it (the whole song where it has no stems).
#[derive(Clone, Debug)]
pub struct Wave {
    /// The waveform.
    pub waveform: Arc<Waveform>,
    /// Its lane.
    pub lane: Lane,
    /// The bars' first beats, in seconds.
    pub downbeats: Vec<f64>,
}

/// The editor's screen as it is now.
#[derive(Clone, Debug, PartialEq)]
pub struct Editor {
    /// The song's title.
    pub title: String,
    /// Its artist.
    pub artist: String,
    /// Its lyrics are edited by hand.
    pub edited: bool,
    /// There is a change to undo, one undone to redo.
    pub undo: bool,
    /// Redo.
    pub redo: bool,
    /// Whether every edit is to be forgotten is asked.
    pub asking: bool,
    /// The seconds in view.
    pub window: (f64, f64),
    /// The playhead, while the song is played.
    pub playhead: Option<f64>,
    /// The time across the view: where, and what.
    pub ticks: Vec<(f64, String)>,
    /// The words in view.
    pub words: Vec<WordBox>,
    /// The word selected.
    pub selected: Option<Selected>,
    /// Its line's words.
    pub line: Vec<LineWord>,
    /// The text being written.
    pub draft: Option<Draft>,
    /// What stands in for the voice's waveform while there is none, or says
    /// it is the whole song's.
    pub note: Option<&'static str>,
    /// The keys, for the foot.
    pub keys: Vec<Hint>,
}

/// The keys of `ops` in the editor, doing `what`.
fn hint(ops: &[Op], what: &'static str) -> Hint {
    Hint::new(keymap::keys(Context::LyricsEditor, ops), what)
}

/// The editor's keys, for its foot (the text field's while writing).
fn keys(writing: bool) -> Vec<Hint> {
    if writing {
        let key = |op, what| Hint::new(keymap::hint(Context::LyricsText, op), what);
        return vec![key(Op::Accept, "keep"), key(Op::Back, "leave as it was")];
    }
    vec![
        hint(&[Op::LyricsPrevious, Op::LyricsNext], "word"),
        Hint::new(
            format!(
                "{} {}",
                keymap::hint(Context::LyricsEditor, Op::LyricsStartEarlier),
                keymap::hint(Context::LyricsEditor, Op::LyricsStartLater)
            ),
            "start",
        ),
        Hint::new(
            format!(
                "{} {}",
                keymap::hint(Context::LyricsEditor, Op::LyricsEndEarlier),
                keymap::hint(Context::LyricsEditor, Op::LyricsEndLater)
            ),
            "end",
        ),
        hint(&[Op::LyricsPlay], "play"),
        hint(&[Op::LyricsNotSung], "not sung"),
        hint(&[Op::LyricsLouder, Op::LyricsQuieter], "voice"),
        hint(&[Op::LyricsWrite], "text"),
        hint(&[Op::LyricsUndo], "undo"),
        hint(&[Op::Back], "done"),
    ]
}

/// The step between the time's marks for a view `span` seconds across:
/// eight or more marks.
fn tick_step(span: f64) -> f64 {
    const STEPS: [f64; 10] = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 30.0, 60.0];
    STEPS
        .iter()
        .copied()
        .rev()
        .find(|&s| span / s >= 8.0)
        .unwrap_or(STEPS[0])
}

/// `t` seconds as the time's marks say it: `1:23`, with tenths where the
/// marks are less than a second apart (`1:23.4`).
fn tick_text(t: f64, step: f64) -> String {
    if step >= 1.0 {
        return text::clock(t + 1e-6);
    }
    let tenths = ((t * 10.0).round() as u64) % 10;
    format!("{}.{tenths}", text::clock(t + 1e-6))
}

/// `t` seconds to the hundredth: `1:23.45`.
fn exact(t: f64) -> String {
    let hundredths = ((t.max(0.0) * 100.0).round() as u64) % 100;
    format!("{}.{hundredths:02}", text::clock(t + 1e-6))
}

/// `db` as the editor writes a level: `−6 dB`, `+3 dB`, `off`.
pub fn db_text(db: f64) -> String {
    if !db.is_finite() || db <= f64::from(screen::LEVEL_LEAST) {
        return "off".to_owned();
    }
    let r = db.round() as i64;
    match r.signum() {
        1 => format!("+{r} dB"),
        -1 => format!("−{} dB", r.unsigned_abs()),
        _ => "0 dB".to_owned(),
    }
}

/// Whether the editor's lyrics are edited by hand: a change made in it, a
/// word set by hand or marked, or a text other than the one found.
fn edited(state: &State, editor: &LyricEditor) -> bool {
    let marked = editor
        .lyrics
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .any(|w| w.pinned || w.not_sung || w.voice_db.is_some());
    let found = match state.karaoke.lyrics.get(&editor.path) {
        Some(SongLyrics::Found { lyrics, .. }) => {
            let texts = |l: &jodel_lyrics::Lyrics| -> Vec<String> {
                l.lines
                    .iter()
                    .filter(|l| !l.words.is_empty())
                    .map(Line::text)
                    .collect()
            };
            texts(lyrics) != texts(&editor.lyrics)
        }
        _ => false,
    };
    editor.can_undo() || marked || found
}

/// The voice's waveform for the editor's song, where its analysis is in:
/// its vocals' lane, else (no stems) the whole song's.
pub fn wave(state: &State) -> Option<Wave> {
    let editor = state.karaoke.editor.as_ref()?;
    let Some(Rhythm::Ready(r)) = state.rhythm.get(&editor.path) else {
        return None;
    };
    let waveform = r.waveform.clone()?;
    let lane = if waveform.stems.is_some() {
        Lane::Stem(Stem::Vocals)
    } else {
        Lane::Song
    };
    let a = &r.analysis;
    let downbeats = a
        .downbeats
        .iter()
        .filter_map(|&d| a.beats.get(d).copied())
        .collect();
    Some(Wave {
        waveform,
        lane,
        downbeats,
    })
}

impl Editor {
    /// The screen for `state`, while the editor is open.
    pub fn of(state: &State) -> Option<Editor> {
        let editor = state.karaoke.editor.as_ref()?;
        let window = screen::window(state)?;
        let playhead = screen::playhead(state);
        let slider = sliders::voice_db(state.config.karaoke_voice);
        let s = &editor.screen;
        let held = s.held.as_ref().map(|h| h.part);
        let mut words = Vec::new();
        for (line, l) in editor.lyrics.lines.iter().enumerate() {
            for (word, w) in l.words.iter().enumerate() {
                let (Some(start), Some(end)) = (w.start, w.end) else {
                    continue;
                };
                if end < window.0 || start > window.1 {
                    continue;
                }
                let at = WordAt { line, word };
                let selected = editor.selected == Some(at);
                let own = voice_curve::own_level(w, slider);
                words.push(WordBox {
                    at,
                    text: w.text.clone(),
                    start,
                    end,
                    selected,
                    held: held.filter(|_| selected),
                    pinned: w.pinned,
                    not_sung: w.not_sung,
                    heard: playhead.is_some_and(|p| p >= start && p < end),
                    level: own.unwrap_or(slider),
                    own: own.is_some(),
                });
            }
        }
        let selected = editor.word().map(|w| {
            let times = match (w.start, w.end) {
                (Some(a), Some(b)) => format!("{} – {}  ·  {:.2} s", exact(a), exact(b), b - a),
                _ => "not timed".to_owned(),
            };
            let level = match (w.voice_db, w.not_sung) {
                (Some(db), _) => format!("{}, set here", db_text(f64::from(db))),
                (None, true) => format!(
                    "{}, raised: not sung",
                    db_text(voice_curve::own_level(w, slider).unwrap_or(slider))
                ),
                (None, false) => format!("{}, as the Voice slider", db_text(slider)),
            };
            Selected {
                text: w.text.clone(),
                times,
                pinned: w.pinned,
                not_sung: w.not_sung,
                level,
                own_level: w.voice_db.is_some(),
            }
        });
        let line = editor
            .selected
            .and_then(|at| editor.lyrics.lines.get(at.line).map(|l| (at, l)))
            .map(|(at, l)| {
                l.words
                    .iter()
                    .enumerate()
                    .map(|(word, w)| LineWord {
                        at: WordAt {
                            line: at.line,
                            word,
                        },
                        text: w.text.clone(),
                        selected: word == at.word,
                        not_sung: w.not_sung,
                    })
                    .collect()
            })
            .unwrap_or_default();
        let step = tick_step(window.1 - window.0);
        let first = (window.0 / step).ceil() as i64;
        let ticks = (first..)
            .map(|k| k as f64 * step)
            .take_while(|&t| t <= window.1)
            .map(|t| (t, tick_text(t, step)))
            .collect();
        let note = match wave(state) {
            None => Some("the voice's waveform comes once the song is analysed"),
            Some(w) if w.lane == Lane::Song => {
                Some("no stems for this song: the whole song's waveform")
            }
            Some(_) => None,
        };
        Some(Editor {
            title: crate::core::shown_title(state, &editor.path),
            artist: state
                .details
                .get(&editor.path)
                .and_then(|d| d.artist.clone())
                .unwrap_or_default(),
            edited: edited(state, editor),
            undo: editor.can_undo(),
            redo: editor.can_redo(),
            asking: s.asking,
            window,
            playhead,
            ticks,
            words,
            selected,
            line,
            draft: s.draft.clone(),
            note,
            keys: keys(s.draft.is_some()),
        })
    }
}

/// An outline of the screen, a line a piece: for the shots.
impl fmt::Display for Editor {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let was = if self.edited {
            "edited by hand"
        } else {
            "as found"
        };
        writeln!(f, "[title] Lyrics · {}  [chip] {was}", self.title)?;
        if !self.artist.is_empty() {
            writeln!(f, "[dim] {}", self.artist)?;
        }
        let on = |b: bool| if b { "" } else { " (greyed)" };
        writeln!(
            f,
            "[buttons] undo{} · redo{} · {REVERT} · {DONE}",
            on(self.undo),
            on(self.redo)
        )?;
        if self.asking {
            writeln!(f, "[ask] {ASK}")?;
        }
        let (a, b) = self.window;
        writeln!(f, "[view] {} – {}", exact(a), exact(b))?;
        if let Some(p) = self.playhead {
            writeln!(f, "[playhead] {}", exact(p))?;
        }
        if let Some(note) = self.note {
            writeln!(f, "[note] {note}")?;
        }
        for w in &self.words {
            let mut tone = String::from("word");
            for (on, said) in [
                (w.selected, " selected"),
                (w.held.is_some(), " held"),
                (w.pinned, " set"),
                (w.not_sung, " not-sung"),
                (w.heard, " heard"),
            ] {
                if on {
                    tone.push_str(said);
                }
            }
            let level = if w.own {
                format!("  voice {}", db_text(w.level))
            } else {
                String::new()
            };
            writeln!(
                f,
                "[{tone}] {} {}–{}{level}",
                w.text,
                exact(w.start),
                exact(w.end)
            )?;
        }
        if let Some(s) = &self.selected {
            writeln!(f, "[selected] {}  {}  voice {}", s.text, s.times, s.level)?;
        }
        let line: Vec<String> = self
            .line
            .iter()
            .map(|w| {
                if w.selected {
                    format!("[{}]", w.text)
                } else {
                    w.text.clone()
                }
            })
            .collect();
        writeln!(f, "[line] {}", line.join(" "))?;
        if let Some(d) = &self.draft {
            let what = if d.line { "line" } else { "word" };
            writeln!(f, "[writing {what}] {}", d.text)?;
        }
        writeln!(f, "[keys] {}", super::karaoke::keys_line(&self.keys))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The time's marks come eight or more to a view, at a round step, said
    /// with tenths only where they are closer than a second.
    #[test]
    fn the_times_marks_come_at_a_round_step() {
        assert_eq!(tick_step(10.0), 1.0);
        assert_eq!(tick_step(3.0), 0.2);
        assert_eq!(tick_step(240.0), 30.0);
        assert_eq!(tick_text(83.0, 1.0), "1:23");
        assert_eq!(tick_text(83.4, 0.2), "1:23.4");
        assert_eq!(exact(83.456), "1:23.46");
        assert_eq!(db_text(-6.2), "−6 dB");
        assert_eq!(db_text(3.0), "+3 dB");
        assert_eq!(db_text(f64::NEG_INFINITY), "off");
    }
}
