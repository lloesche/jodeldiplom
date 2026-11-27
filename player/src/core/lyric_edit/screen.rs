//! The lyrics editor's screen ([`crate::core::Screen::Lyrics`]): what it
//! shows of the song, and what its keys do.
//!
//! - **The view** ([`EditorScreen::view`], a [`TimeView`] as the Transition
//!   tab's blocks have): a span of the song over the voice's waveform: the
//!   wheel zooms it round the time under the pointer, the secondary button
//!   drags it, and the word the keys select is brought into it. While the
//!   song plays from a word (`space`) it pages after the playhead
//!   ([`TimeView::paged`]) until it is moved by hand; `.` has it follow
//!   again. The secondary button's drag goes through the app's one
//!   [`crate::core::Action::Pan`], as the Transition tab's does. `z` zooms it to the widest span at which every word shows its
//!   text whole, as the Transition tab's `z` does its lanes
//!   ([`fit_lyrics`]).
//! - **The pointer** ([`super::LyricsEdit`]): a press on a word selects it
//!   and holds it ([`Held`]): dragged by its middle it moves, by an edge
//!   that end moves, set by hand (a tremble within its slop changes
//!   nothing); let go of, the drag is one step of the history, kept. A
//!   second click on it soon after writes its text. A press on the
//!   waveform off the words puts the playhead there (the song playing).
//! - **The text:** a word's or its line's written ([`Draft`]): kept with
//!   `enter`, left as it was with `esc`; a press on the lane keeps it too.
//! - **The keys** ([`apply`]): the word before or after, the line above or
//!   below, its start, end or whole a step ([`NUDGE`]) earlier or later,
//!   the song played from it, the view fitted to the lyrics or following
//!   the playhead again, not to be sung, the voice's level over it, its
//!   text, taken out, undo and redo, `esc` closing; playback, the volume
//!   and the stems as anywhere ([`Context::Playback`]); nothing else (the
//!   palette, the settings, the help, the tabs).

use jodel_lyrics::{Lyrics, WordAt};
use jodel_playback::{Command, PlayState};

use super::{
    act as edit_act, editor, editor_mut, Change, Edit, LyricEditor, LyricsEdit, Part, Selection,
};
use crate::core::keymap::{self, Context, Op};
use crate::core::timeview::{self, lead_start, zoomed_start, Fitted, Floor, LyricsFit, PanInput};
use crate::core::timeview::{TimeView, LEAD};
use crate::core::{lanes, sliders, text, voice_curve, DoubleClick, Effect, State};

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
/// The letters a word's box takes beyond its text (its insets and the gaps
/// between boxes): a box shows its text whole where it is its letters and
/// these long ([`timeview::words_cut`]).
pub const BOX_PAD: f64 = 2.0;

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

/// What the screen shows and holds.
#[derive(Clone, Debug, PartialEq)]
pub struct EditorScreen {
    /// Where the view is: held, or paging after the playhead (from
    /// `space` or `.` until it is moved by hand), and its drag by the
    /// secondary button.
    pub view: TimeView,
    /// The seconds across it.
    pub span: f64,
    /// The word the pointer holds.
    pub held: Option<Held>,
    /// The text being written.
    pub draft: Option<Draft>,
    /// The last click on a word: a second one soon after writes it.
    pub clicked: DoubleClick<WordAt>,
    /// Whether every edit is to be forgotten is asked.
    pub asking: bool,
}

/// What the screen's work needs of the app, taken once: where the playhead
/// is in the editor's song while it is played, the song's length where
/// known, and the clock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ctx {
    /// The playhead, in seconds, while the song is played (playing or
    /// paused).
    pub playhead: Option<f64>,
    /// The song's length.
    pub duration: Option<f64>,
    /// The app's clock, in milliseconds.
    pub now: u64,
}

impl Ctx {
    /// What the editor's screen needs of `state`, while it is open.
    pub fn of(state: &State) -> Option<Ctx> {
        let editor = editor(state)?;
        let played = is_played(state, editor);
        let playhead = (played && state.playing.state != PlayState::Stopped).then(|| state.heard());
        let analysed = match state.rhythm.get(&editor.path) {
            Some(crate::core::Rhythm::Ready(r)) => Some(r.length()),
            _ => None,
        };
        let duration = analysed
            .filter(|d| *d > 0.0)
            .or_else(|| played.then(|| state.playing.duration()).flatten())
            .or_else(|| state.details.get(&editor.path).and_then(|d| d.seconds));
        Some(Ctx {
            playhead,
            duration,
            now: state.now_ms,
        })
    }

    /// The most seconds across the view: the song's length, where known.
    fn most(&self) -> f64 {
        self.duration.unwrap_or(MOST_SPAN).max(LEAST_SPAN)
    }
}

impl EditorScreen {
    /// The screen as the editor opens, its first word (sung from `first`
    /// seconds) a third of the way in.
    pub fn at(first: Option<f64>) -> EditorScreen {
        EditorScreen {
            view: TimeView::held(first.map_or(0.0, |t| lead_start(t, SPAN))),
            span: SPAN,
            held: None,
            draft: None,
            clicked: DoubleClick::default(),
            asking: false,
        }
    }

    /// The seconds the view shows (from, to): its span from where it is,
    /// paged after the playhead while it follows it (a page on each time
    /// the playhead reaches [`PAGE`] of the way across).
    pub fn window(&self, ctx: &Ctx) -> (f64, f64) {
        let start = self.view.paged(self.span, ctx.playhead, PAGE);
        (start, start + self.span)
    }

    /// Whether a text is being written (the text field takes the keys).
    pub fn writing(&self) -> bool {
        self.draft.is_some()
    }

    /// The view held where it shows now, no longer following the playhead
    /// (the pointer moves it by hand).
    pub(super) fn hold_view(&mut self, ctx: &Ctx) {
        let (start, _) = self.window(ctx);
        self.view.hold(start);
    }

    /// The seconds from `a` to `b` brought into view where they are not in
    /// it: a third of the way in, held there.
    pub(super) fn reveal(&mut self, ctx: &Ctx, (a, b): (f64, f64)) {
        let (start, end) = self.window(ctx);
        if a < start || b > end {
            self.view.lead_to(a, self.span);
        }
    }

    /// The view `span` seconds across, `at` (seconds into the song) where
    /// it was across it (a third of the way in where it was not in view).
    /// A view following the playhead goes on following it.
    fn zoom_round(&mut self, ctx: &Ctx, span: f64, at: f64) {
        self.view
            .move_to(zoomed_start(self.window(ctx), span, at, LEAD));
        self.span = span;
    }

    /// `.`: the view follows the playhead again, the playhead (where the
    /// song is played) a third of the way in.
    fn follow(&mut self, ctx: &Ctx) {
        match ctx.playhead {
            Some(p) => self.view.follow_from(lead_start(p, self.span)),
            None => self.view.follow(),
        }
    }

    /// The wheel `steps` up (in) or down (out), the time `at` (seconds)
    /// under the pointer staying where it is: the view held there.
    pub(super) fn zoom(&mut self, ctx: &Ctx, steps: i32, at: f64) {
        self.hold_view(ctx);
        let span = (self.span * ZOOM.powi(-steps)).clamp(LEAST_SPAN, ctx.most());
        self.zoom_round(ctx, span, at);
    }

    /// What the secondary button does to the view: pressed, it is held
    /// where it shows and dragged by the span over the lane's width a
    /// point.
    pub(super) fn pan(&mut self, ctx: &Ctx, input: PanInput) {
        match input {
            PanInput::Start { at, width } => {
                self.hold_view(ctx);
                let (start, _) = self.window(ctx);
                let per_unit = self.span / f64::from(width.max(1.0));
                self.view.start_pan(at, start, per_unit);
            }
            PanInput::To(at) => self.view.pan_to(at),
            PanInput::End => self.view.end_pan(),
        }
    }
}

/// Whether a text is being written (the text field takes the keys).
pub fn writing(state: &State) -> bool {
    editor(state).is_some_and(|e| e.screen.writing())
}

/// Whether what the screen shows moves with every frame: the song playing
/// under it (the playhead), a word or the view held by the pointer.
pub fn moving(state: &State) -> bool {
    editor(state).is_some_and(|editor| {
        let s = &editor.screen;
        s.held.is_some()
            || s.view.panning()
            || (is_played(state, editor) && state.playing.state == PlayState::Playing)
    })
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
    Ctx::of(state)?.playhead
}

/// The seconds the view shows (from, to), while the editor is open
/// ([`EditorScreen::window`]).
pub fn window(state: &State) -> Option<(f64, f64)> {
    Some(editor(state)?.screen.window(&Ctx::of(state)?))
}

/// The spans `z` picks from, the narrowest first: [`LEAST_SPAN`], then a
/// step of the wheel wider each, up to `most` (the song's length).
fn fit_spans(most: f64) -> Vec<f64> {
    let mut spans: Vec<f64> = std::iter::successors(Some(LEAST_SPAN), |s| Some(s * ZOOM))
        .take_while(|&s| s < most)
        .collect();
    spans.push(most);
    spans
}

/// `z`: the view zoomed to the widest span (of [`fit_spans`]) at which every
/// word's box shows its text whole, but those no span shows so
/// ([`timeview::widest_fit`], as the Transition tab's lanes fit theirs),
/// round the playhead while the view follows it, else round the word
/// selected; and said.
fn fit_lyrics(state: &mut State, ctx: &Ctx) {
    let Some(editor) = editor(state) else {
        return;
    };
    let words = lanes::fitted_words(&editor.lyrics);
    let spans = fit_spans(ctx.most());
    let shown = |k: usize| {
        let span = spans[k];
        if span < 60.0 {
            format!("{span:.1} s")
        } else {
            text::clock(span)
        }
    };
    let fit = LyricsFit {
        what: "the song",
        words: &words,
        letters: state.fit.lyric_letters,
        spans: &spans,
        floor: Floor::Narrowest,
        cut: &|words, per| timeview::words_cut(words, per, BOX_PAD),
        shown: &shown,
    };
    let Some(fitted) = timeview::fit_lyrics(&fit) else {
        return;
    };
    let s = &editor.screen;
    let (start, end) = s.window(ctx);
    let word = editor.word().and_then(|w| w.start.zip(w.end));
    let at = match (ctx.playhead.filter(|_| s.view.is_following()), word) {
        (Some(p), _) => p,
        (None, Some((a, b))) => 0.5 * (a + b),
        (None, None) => 0.5 * (start + end),
    };
    if let (Fitted::Zoom { index, .. }, Some(editor)) = (&fitted, editor_mut(state)) {
        editor.screen.zoom_round(ctx, spans[*index], at);
    }
    state.info(fitted.said());
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
    let level = editor
        .word()
        .map_or(slider, |w| voice_curve::voice_level(w, slider).db());
    level.clamp(f64::from(LEVEL_LEAST), f64::from(LEVEL_MOST)) as f32
}

/// Carries out `op` on the editor's screen; `false` when it is not the
/// screen's: not shown, or one done as anywhere ([`Context::Playback`]'s).
/// Any other op leaves the screen as it is.
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    let (Some(editor), Some(ctx)) = (editor(state), Ctx::of(state)) else {
        return false;
    };
    let edit = LyricsEdit::Edit;
    let word = editor.word();
    let what = match op {
        Op::Back if editor.screen.writing() => LyricsEdit::Discard,
        Op::Back if editor.screen.asking => LyricsEdit::AskRevert(false),
        Op::Back => LyricsEdit::Close,
        Op::Accept => LyricsEdit::Keep,
        Op::LyricsPrevious => edit(Edit::Select(Selection::Previous)),
        Op::LyricsNext => edit(Edit::Select(Selection::Next)),
        Op::Up | Op::Down => {
            let by = if op == Op::Up { -1 } else { 1 };
            match line_word(editor, by) {
                Some(at) => edit(Edit::Select(Selection::Word(at))),
                None => return true,
            }
        }
        Op::LyricsNudge(part, by) => Change::nudge(part, f64::from(by) * NUDGE).into(),
        Op::LyricsNotSung => {
            let not_sung = word.is_some_and(|w| w.edit.not_sung);
            Change::NotSung(!not_sung).into()
        }
        Op::LyricsLevel(by) => {
            let db = level(state, editor) + f32::from(by) * LEVEL_STEP;
            Change::Voice(Some(db.clamp(LEVEL_LEAST, LEVEL_MOST))).into()
        }
        Op::LyricsLevelReset => Change::Voice(None).into(),
        Op::LyricsWrite => LyricsEdit::Write { line: false },
        Op::LyricsWriteLine => LyricsEdit::Write { line: true },
        Op::LyricsDelete => Change::DeleteWord.into(),
        Op::LyricsUndo => edit(Edit::Undo),
        Op::LyricsRedo => edit(Edit::Redo),
        Op::LyricsPlay => {
            play(state, &ctx, fx);
            return true;
        }
        Op::FitLyrics => {
            fit_lyrics(state, &ctx);
            return true;
        }
        Op::Follow => {
            if let Some(editor) = editor_mut(state) {
                editor.screen.follow(&ctx);
            }
            return true;
        }
        op => return !keymap::binds(Context::Playback, op),
    };
    edit_act(state, what, fx);
    true
}

/// `space`: the song paused while it plays; else played from just before
/// the word selected ([`PREROLL`]), the view following it.
fn play(state: &mut State, ctx: &Ctx, fx: &mut Vec<Effect>) {
    let Some(editor) = editor(state) else {
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
    if let Some(editor) = editor_mut(state) {
        editor.reveal(ctx);
        editor.screen.view.follow();
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
    let command = match ctx.duration {
        Some(d) if d > 0.0 => Command::PlayAt {
            index,
            share: (t / d).clamp(0.0, 1.0),
        },
        _ => Command::Play(index),
    };
    fx.push(Effect::Engine(command));
}

#[cfg(test)]
mod tests;
