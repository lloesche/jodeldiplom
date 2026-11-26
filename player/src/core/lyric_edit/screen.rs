//! The lyrics editor's screen ([`Screen::Lyrics`]): what it shows of the
//! song, and what its keys, its pointer and its text field do.
//!
//! - **The view:** a span of the song over the voice's waveform
//!   ([`EditorScreen::start`], [`EditorScreen::span`]): the wheel zooms it
//!   round the time under the pointer, the secondary button drags it, and
//!   the word the keys select is brought into it. While the song plays from
//!   a word (`space`) it pages after the playhead ([`window`]) until it is
//!   moved by hand.
//! - **The pointer:** a press on a word selects it and holds it ([`Held`]):
//!   dragged by its middle it moves, by an edge that end moves, set by hand
//!   (a tremble within [`Input::Hold`]'s slop changes nothing); let go of,
//!   the drag is one step of the history, kept. A second click on it soon
//!   after writes its text. A press on the waveform off the words puts the
//!   playhead there (the song playing).
//! - **The text:** a word's or its line's written ([`Draft`]): kept with
//!   `enter`, left as it was with `esc`; a press on the lane keeps it too.
//! - **The keys** ([`apply`]): the word before or after, the line above or
//!   below, its start, end or whole a step ([`NUDGE`]) earlier or later,
//!   the song played from it, not to be sung, the voice's level over it,
//!   its text, taken out, undo and redo, `esc` closing; quitting, the
//!   volume, the stems and playback as anywhere; nothing that leaves the
//!   screen (the palette, the settings, the help, the tabs).

use jodel_lyrics::Lyrics;
use jodel_playback::{Command, PlayState};

use super::{act as edit_act, kept, Edit, LyricEditor, LyricsEdit, WordAt};
use crate::core::keymap::Op;
use crate::core::{sliders, voice_curve, DoubleClick, Effect, Screen, State};

/// How far the keys move a word's start or end, or the whole word, in
/// seconds.
pub const NUDGE: f64 = 0.02;
/// How far `+` and `-` move the voice's level over a word, in dB.
pub const LEVEL_STEP: f32 = 3.0;
/// The quietest the voice's level over a word is set, in dB against the
/// voice as recorded.
pub const LEVEL_LEAST: f32 = -45.0;
/// The loudest.
pub const LEVEL_MOST: f32 = 6.0;
/// The seconds across the view as the editor opens.
pub const SPAN: f64 = 10.0;
/// The fewest seconds across the view, zoomed in all the way.
pub const LEAST_SPAN: f64 = 1.0;
/// The most, where the song's length is not known.
const MOST_SPAN: f64 = 600.0;
/// How much a step of the wheel zooms.
const ZOOM: f64 = 1.25;
/// The share of the view a page moves on while it follows the playhead
/// (the rest of the page before stays in view).
pub const PAGE: f64 = 0.9;
/// How long before a word the song plays from it, in seconds: its onset is
/// heard.
pub const PREROLL: f64 = 0.3;

/// What a press on a word holds: its start, its middle (all of it) or its
/// end.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// Its start: dragged, its start moves.
    Start,
    /// All of it: dragged, it moves.
    Middle,
    /// Its end: dragged, its end moves.
    End,
}

/// A word held by the pointer.
#[derive(Clone, Debug, PartialEq)]
pub struct Held {
    /// What of it.
    pub part: Part,
    /// The lyrics as the press found them.
    pub from: Lyrics,
    /// The seconds into the song under the pointer as it was pressed.
    pub time: f64,
    /// How far the pointer may go from there and still change nothing, in
    /// seconds.
    pub slop: f64,
    /// It went that far.
    pub moved: bool,
}

/// A text being written: the word's, or its whole line's.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Draft {
    /// The line's.
    pub line: bool,
    /// The text as written so far.
    pub text: String,
}

/// The view dragged with the secondary button.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pan {
    /// Where the pointer was pressed across the window, in points.
    pub at: f32,
    /// The view's start then.
    pub start: f64,
    /// The lane's width, in points.
    pub width: f32,
}

/// What the screen shows and holds.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorScreen {
    /// The seconds into the song at the view's left edge.
    pub start: f64,
    /// The seconds across it.
    pub span: f64,
    /// It pages after the playhead ([`window`]).
    pub follow: bool,
    /// The word the pointer holds.
    pub held: Option<Held>,
    /// The view the secondary button holds.
    pub pan: Option<Pan>,
    /// The text being written.
    pub draft: Option<Draft>,
    /// The last click on a word: a second one soon after writes it.
    pub clicked: DoubleClick<WordAt>,
    /// Whether every edit is to be forgotten is asked.
    pub asking: bool,
}

impl EditorScreen {
    /// The screen as the editor opens, its first word (sung from `first`
    /// seconds) a third of the way in.
    pub fn at(first: Option<f64>) -> EditorScreen {
        EditorScreen {
            start: first.map_or(0.0, |t| (t - SPAN / 3.0).max(0.0)),
            span: SPAN,
            follow: false,
            held: None,
            pan: None,
            draft: None,
            clicked: DoubleClick::default(),
            asking: false,
        }
    }
}

/// What the screen's pointer and its text field do.
#[derive(Clone, Debug, PartialEq)]
pub enum Input {
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
    /// The wheel `steps` up (in) or down (out), the time `at` (seconds)
    /// under the pointer staying where it is.
    Zoom {
        /// Steps, in while positive.
        steps: i32,
        /// Under the pointer.
        at: f64,
    },
    /// The secondary button pressed `at` points across the window, over a
    /// lane `width` points wide: the view held.
    PanStart {
        /// Where.
        at: f32,
        /// The lane's width.
        width: f32,
    },
    /// The pointer holding the view is `at` points across the window.
    Pan(f32),
    /// It let go of it.
    PanEnd,
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
}

/// Whether a text is being written (the text field takes the keys).
pub fn writing(state: &State) -> bool {
    state.screen == Screen::Lyrics
        && state
            .karaoke
            .editor
            .as_ref()
            .is_some_and(|e| e.screen.draft.is_some())
}

/// Whether what the screen shows moves with every frame: the song playing
/// under it (the playhead), a word or the view held by the pointer.
pub fn moving(state: &State) -> bool {
    let Some(editor) = state.karaoke.editor.as_ref() else {
        return false;
    };
    let s = &editor.screen;
    state.screen == Screen::Lyrics
        && (s.held.is_some()
            || s.pan.is_some()
            || (is_played(state, editor) && state.playing.state == PlayState::Playing))
}

/// Whether the editor's song is the track played (playing, paused, or
/// stopped on it).
fn is_played(state: &State, editor: &LyricEditor) -> bool {
    state
        .playing
        .track
        .as_ref()
        .is_some_and(|t| state.queue.paths.get(t.index) == Some(&editor.path))
}

/// Where the playhead is in the editor's song while it is played (playing
/// or paused), in seconds.
pub fn playhead(state: &State) -> Option<f64> {
    let editor = state.karaoke.editor.as_ref()?;
    (is_played(state, editor) && state.playing.state != PlayState::Stopped).then(|| state.heard())
}

/// The editor's song's length, where it is known.
pub fn duration(state: &State) -> Option<f64> {
    let editor = state.karaoke.editor.as_ref()?;
    let analysed = match state.rhythm.get(&editor.path) {
        Some(crate::core::Rhythm::Ready(r)) => Some(
            r.waveform
                .as_ref()
                .map_or(0.0, |w| w.duration())
                .max(r.analysis.duration),
        ),
        _ => None,
    };
    analysed
        .filter(|d| *d > 0.0)
        .or_else(|| {
            is_played(state, editor)
                .then(|| state.playing.duration())
                .flatten()
        })
        .or_else(|| state.details.get(&editor.path).and_then(|d| d.seconds))
}

/// The seconds the view shows (from, to): its span from its start, paged
/// after the playhead while it follows it (a page on each time the
/// playhead reaches [`PAGE`] of the way across).
pub fn window(state: &State) -> Option<(f64, f64)> {
    let s = &state.karaoke.editor.as_ref()?.screen;
    let start = match playhead(state).filter(|_| s.follow) {
        Some(p) => {
            let page = PAGE * s.span;
            s.start + ((p - s.start) / page).floor() * page
        }
        None => s.start,
    };
    Some((start, start + s.span))
}

/// The view held where it shows now, no longer following the playhead (the
/// pointer moves it by hand).
fn settle(state: &mut State) {
    let Some((start, _)) = window(state) else {
        return;
    };
    if let Some(editor) = &mut state.karaoke.editor {
        editor.screen.start = start;
        editor.screen.follow = false;
    }
}

/// The word selected brought into view where it is not in it: a third of
/// the way in.
pub(super) fn reveal(state: &mut State) {
    let Some((start, end)) = window(state) else {
        return;
    };
    let Some(editor) = &mut state.karaoke.editor else {
        return;
    };
    let Some((a, b)) = editor.word().and_then(|w| w.start.zip(w.end)) else {
        return;
    };
    if a >= start && b <= end {
        return;
    }
    let s = &mut editor.screen;
    s.start = (a - s.span / 3.0).max(0.0);
    s.follow = false;
}

/// The first word of the line `by` lines above (negative) or below the one
/// selected that has words, if any.
fn line_word(editor: &LyricEditor, by: isize) -> Option<WordAt> {
    let at = editor.selected?;
    let lines = &editor.lyrics.lines;
    let mut line = at.line as isize;
    loop {
        line += by.signum();
        let l = lines.get(usize::try_from(line).ok()?)?;
        if !l.words.is_empty() {
            return Some(WordAt {
                line: line as usize,
                word: 0,
            });
        }
    }
}

/// The voice's level over the word selected in the mode, as `+` and `-`
/// start from: its own, else the slider's.
fn level(state: &State, editor: &LyricEditor) -> f32 {
    let slider = sliders::voice_db(state.config.karaoke_voice);
    let own = editor
        .word()
        .and_then(|w| voice_curve::own_level(w, slider));
    own.unwrap_or(slider)
        .clamp(f64::from(LEVEL_LEAST), f64::from(LEVEL_MOST)) as f32
}

/// Carries out `op` on the editor's screen; `false` when it is not the
/// screen's (it is not shown, or the op is one of those done as anywhere).
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if state.screen != Screen::Lyrics || state.palette.is_some() {
        return false;
    }
    let Some(editor) = &state.karaoke.editor else {
        // Nothing to edit: back to the main screen.
        state.screen = Screen::Main;
        return false;
    };
    let edit = LyricsEdit::Edit;
    let input = LyricsEdit::Screen;
    let word = editor.word();
    let what = match op {
        Op::Back if editor.screen.draft.is_some() => input(Input::Discard),
        Op::Back if editor.screen.asking => LyricsEdit::AskRevert(false),
        Op::Back => LyricsEdit::Close,
        Op::Accept => input(Input::Keep),
        Op::LyricsPrevious => edit(Edit::Previous),
        Op::LyricsNext => edit(Edit::Next),
        Op::Up | Op::Down => {
            let by = if op == Op::Up { -1 } else { 1 };
            match line_word(editor, by) {
                Some(at) => edit(Edit::Select(at)),
                None => return true,
            }
        }
        Op::LyricsStartEarlier => edit(Edit::NudgeStart(-NUDGE)),
        Op::LyricsStartLater => edit(Edit::NudgeStart(NUDGE)),
        Op::LyricsEndEarlier => edit(Edit::NudgeEnd(-NUDGE)),
        Op::LyricsEndLater => edit(Edit::NudgeEnd(NUDGE)),
        Op::LyricsMoveEarlier => edit(Edit::Move(-NUDGE)),
        Op::LyricsMoveLater => edit(Edit::Move(NUDGE)),
        Op::LyricsNotSung => edit(Edit::NotSung(!word.is_some_and(|w| w.not_sung))),
        Op::LyricsLouder | Op::LyricsQuieter => {
            let by = if op == Op::LyricsLouder {
                LEVEL_STEP
            } else {
                -LEVEL_STEP
            };
            let db = (level(state, editor) + by).clamp(LEVEL_LEAST, LEVEL_MOST);
            edit(Edit::Voice(Some(db)))
        }
        Op::LyricsLevelReset => edit(Edit::Voice(None)),
        Op::LyricsWrite => input(Input::Write { line: false }),
        Op::LyricsWriteLine => input(Input::Write { line: true }),
        Op::LyricsDelete => edit(Edit::DeleteWord),
        Op::LyricsUndo => edit(Edit::Undo),
        Op::LyricsRedo => edit(Edit::Redo),
        Op::LyricsPlay => {
            play(state, fx);
            return true;
        }
        // As anywhere: they leave the screen as it is.
        Op::Quit
        | Op::VolumeUp
        | Op::VolumeDown
        | Op::PlayPause
        | Op::Next
        | Op::Previous
        | Op::Stop
        | Op::ToggleDrums
        | Op::ToggleBass
        | Op::ToggleVocals
        | Op::ToggleOther
        | Op::AllStems => return false,
        _ => return true,
    };
    edit_act(state, what, fx);
    true
}

/// `space`: the song paused while it plays; else played from just before
/// the word selected ([`PREROLL`]), the view following it.
fn play(state: &mut State, fx: &mut Vec<Effect>) {
    let Some(editor) = &state.karaoke.editor else {
        return;
    };
    let played = is_played(state, editor);
    if played && state.playing.state == PlayState::Playing {
        fx.push(Effect::Engine(Command::Pause));
        return;
    }
    let Some(from) = editor.word().and_then(|w| w.start) else {
        return;
    };
    let path = editor.path.clone();
    let t = (from - PREROLL).max(0.0);
    reveal(state);
    if let Some(editor) = &mut state.karaoke.editor {
        editor.screen.follow = true;
    }
    if played && state.playing.state == PlayState::Paused {
        state.playing.position = t;
        fx.push(Effect::Engine(Command::Seek(t)));
        fx.push(Effect::Engine(Command::Resume));
        return;
    }
    let Some(index) = state.queue.paths.iter().position(|p| *p == path) else {
        return;
    };
    let command = match duration(state) {
        Some(d) if d > 0.0 => Command::PlayAt {
            index,
            share: (t / d).clamp(0.0, 1.0),
        },
        _ => Command::Play(index),
    };
    fx.push(Effect::Engine(command));
}

/// Carries out what the screen's pointer or its text field did.
pub(super) fn act(state: &mut State, what: Input, fx: &mut Vec<Effect>) {
    // A press anywhere on the lane while a text is written keeps it.
    if writing(state) && matches!(what, Input::Hold { .. } | Input::Seek(_)) {
        return act(state, Input::Keep, fx);
    }
    let now = state.now_ms;
    let most = duration(state).unwrap_or(MOST_SPAN).max(LEAST_SPAN);
    match what {
        Input::Hold {
            word,
            part,
            time,
            slop,
        } => {
            settle(state);
            let Some(editor) = &mut state.karaoke.editor else {
                return;
            };
            editor.apply(Edit::Select(word));
            if editor.selected != Some(word) {
                return;
            }
            editor.screen.held = Some(Held {
                part,
                from: editor.lyrics.clone(),
                time,
                slop,
                moved: false,
            });
        }
        Input::Drag(time) => {
            let Some(editor) = &mut state.karaoke.editor else {
                return;
            };
            let Some(held) = &mut editor.screen.held else {
                return;
            };
            let by = time - held.time;
            if !held.moved && by.abs() < held.slop {
                return;
            }
            held.moved = true;
            let edit = match held.part {
                Part::Start => Edit::NudgeStart(by),
                Part::Middle => Edit::Move(by),
                Part::End => Edit::NudgeEnd(by),
            };
            let from = held.from.clone();
            // Back to where it was where the pointer leads nowhere it may go
            // (it is not timed): the drag changes nothing.
            if !editor.drag(&from, Some(edit)) {
                editor.drag(&from, None);
            }
        }
        Input::Drop => {
            let Some(editor) = &mut state.karaoke.editor else {
                return;
            };
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
            let Some(word) = editor.selected else {
                return;
            };
            if editor.screen.clicked.press(word, now) {
                act(state, Input::Write { line: false }, fx);
            }
        }
        Input::Zoom { steps, at } => {
            settle(state);
            let Some(editor) = &mut state.karaoke.editor else {
                return;
            };
            let s = &mut editor.screen;
            let span = (s.span * ZOOM.powi(-steps)).clamp(LEAST_SPAN, most);
            let share = ((at - s.start) / s.span).clamp(0.0, 1.0);
            s.start = (at - share * span).max(0.0);
            s.span = span;
        }
        Input::PanStart { at, width } => {
            settle(state);
            if let Some(editor) = &mut state.karaoke.editor {
                let s = &mut editor.screen;
                s.pan = Some(Pan {
                    at,
                    start: s.start,
                    width: width.max(1.0),
                });
            }
        }
        Input::Pan(x) => {
            if let Some(editor) = &mut state.karaoke.editor {
                let s = &mut editor.screen;
                if let Some(pan) = s.pan {
                    let by = f64::from((pan.at - x) / pan.width) * s.span;
                    s.start = (pan.start + by).max(0.0);
                }
            }
        }
        Input::PanEnd => {
            if let Some(editor) = &mut state.karaoke.editor {
                editor.screen.pan = None;
            }
        }
        Input::Seek(time) => {
            if playhead(state).is_some() {
                let t = time.max(0.0);
                state.playing.position = t;
                fx.push(Effect::Engine(Command::Seek(t)));
            }
        }
        Input::Write { line } => {
            let Some(editor) = &mut state.karaoke.editor else {
                return;
            };
            let Some(at) = editor.selected else {
                return;
            };
            let text = if line {
                editor.lyrics.lines[at.line].text()
            } else {
                editor.lyrics.lines[at.line].words[at.word].text.clone()
            };
            editor.screen.draft = Some(Draft { line, text });
        }
        Input::Draft(text) => {
            if let Some(draft) = state
                .karaoke
                .editor
                .as_mut()
                .and_then(|e| e.screen.draft.as_mut())
            {
                draft.text = text;
            }
        }
        Input::Keep => {
            let Some(draft) = state
                .karaoke
                .editor
                .as_mut()
                .and_then(|e| e.screen.draft.take())
            else {
                return;
            };
            let edit = if draft.line {
                Edit::ReplaceLine(draft.text)
            } else {
                Edit::ReplaceWord(draft.text)
            };
            edit_act(state, LyricsEdit::Edit(edit), fx);
        }
        Input::Discard => {
            if let Some(editor) = &mut state.karaoke.editor {
                editor.screen.draft = None;
            }
        }
    }
}

#[cfg(test)]
mod tests;
