//! The Mix tab: the track playing and the next one, their waveforms and
//! bars over time, a cursor on the beats, a selection of bars, and the
//! fixes to a beat grid ([`super::grid`]).
//!
//! Both blocks show the same span of time, so their tempi compare at a
//! glance: [`ZOOMS`] bars of the playing track, or the whole of the longer
//! track. The playing track's block follows what is heard (a third from
//! its left) until its cursor moves; the next one's starts at its first
//! bar. The cursor is a beat; it moves by bars (or beats), and the
//! selection runs from where it was when a selecting move began to where
//! it is, in whole bars.

use std::path::PathBuf;
use std::sync::Arc;

use jodel_playback::Command;
use jodel_rhythm::Analysis;

use super::grid::Grid;
use super::keymap::Op;
use super::transition::{self as tr, Draft, Edit, Status};
use super::{Effect, Rhythm, Rhythmed, Screen, Side, State};
use jodel_playback::stems::STEMS;
use jodel_playback::{TempoMode, TransitionPlan, TransitionState};

/// The spans the view shows, in bars of the playing track; 0 is the whole
/// of the longer track.
pub const ZOOMS: [usize; 8] = [1, 2, 4, 8, 16, 32, 64, 0];
/// The zoom at the start (16 bars).
pub const DEFAULT_ZOOM: usize = 4;
/// Bars a page moves.
pub const PAGE_BARS: isize = 8;
/// What a nudge moves the grid by, in seconds.
pub const NUDGE: f64 = 0.005;
/// A bar's length where the playing track's is not known, in seconds.
const FALLBACK_BAR: f64 = 2.0;
/// The share of the span shown before the playhead while following, and
/// before the cursor when it is brought into view.
pub const LEAD: f64 = 1.0 / 3.0;

/// A block of the tab.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Block {
    /// The track playing.
    #[default]
    Now,
    /// The next one in the queue.
    Next,
    /// The one before the playing track: shown only while it slides away
    /// after a transition into the playing one ([`slide`]).
    Prev,
}

impl Block {
    /// The two the keys act on, top first.
    pub const BOTH: [Block; 2] = [Block::Now, Block::Next];
}

/// How long the next track's block takes to slide up into the playing
/// track's after a transition hands over, in milliseconds.
pub const SLIDE_MS: u64 = 700;

/// A drag of a block's view with the secondary button.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pan {
    /// The block.
    pub block: Block,
    /// Seconds a cell column is worth (negative: the view follows the
    /// pointer, as on the whole track; positive: the waveform does).
    pub per_column: f64,
    /// The column it started at.
    pub column: u16,
    /// The view's start then.
    pub start: f64,
}

/// A transition's cue on a track: the bars it is to mix over there.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cue {
    /// The first beat of its first bar.
    pub beat: usize,
    /// Its bars (at least one).
    pub bars: usize,
    /// Its bars were selected (else [`tr::DEFAULT_BARS`] from the cursor).
    pub chosen: bool,
}

/// What one block is set to, for the track it was set for.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MixBlock {
    /// The track (a block for another track starts afresh).
    pub path: Option<PathBuf>,
    /// The beat the cursor is on; `None` until it moves (at the playhead's
    /// bar for the playing track, the first bar for the next).
    pub cursor: Option<usize>,
    /// Where a selection started (a beat), while there is one.
    pub anchor: Option<usize>,
    /// The time the view starts at; `None`: following (or, for the next
    /// track, from its first bar).
    pub start: Option<f64>,
    /// The bars a transition is to mix over on this track, once set
    /// (`enter`).
    pub cue: Option<Cue>,
    /// Its own zoom (an index into [`ZOOMS`]), set by the wheel over it;
    /// `None`: the tab's ([`MixState::zoom`]).
    pub zoom: Option<usize>,
}

/// The Mix tab's state.
#[derive(Clone, Debug, PartialEq)]
pub struct MixState {
    /// The block the keys act on.
    pub focus: Block,
    /// The playing track's block.
    pub now: MixBlock,
    /// The next track's block.
    pub next: MixBlock,
    /// The index into [`ZOOMS`].
    pub zoom: usize,
    /// The last press on the lanes: the block, the beat and when
    /// ([`State::now_ms`]), to tell a double click.
    pub last_press: Option<(Block, usize, u64)>,
    /// The transition being set up.
    pub draft: Draft,
    /// The track before the playing one, as its block was when a
    /// transition handed over from it (its view held where it was).
    pub prev: MixBlock,
    /// When ([`frame_ms`]) a transition handed over: the next track's
    /// block slides up into the playing track's place from then.
    pub slide: Option<u64>,
    /// A drag of a view with the secondary button, under way.
    pub pan: Option<Pan>,
}

impl Default for MixState {
    fn default() -> MixState {
        MixState {
            focus: Block::Now,
            now: MixBlock::default(),
            next: MixBlock::default(),
            zoom: DEFAULT_ZOOM,
            last_press: None,
            draft: Draft::default(),
            prev: MixBlock::default(),
            slide: None,
            pan: None,
        }
    }
}

impl MixState {
    /// A Mix tab whose transitions start from `preset` and `tempo` (as the
    /// settings remember them).
    pub fn with(preset: super::transition::Preset, tempo: TempoMode) -> MixState {
        let mut mix = MixState::default();
        mix.draft.preset = preset;
        mix.draft.tempo = tempo;
        mix
    }
}

impl MixState {
    /// A block's state, as stored (see [`block`] for it as it applies).
    pub fn get(&self, block: Block) -> &MixBlock {
        match block {
            Block::Now => &self.now,
            Block::Next => &self.next,
            Block::Prev => &self.prev,
        }
    }

    fn get_mut(&mut self, block: Block) -> &mut MixBlock {
        match block {
            Block::Now => &mut self.now,
            Block::Next => &mut self.next,
            Block::Prev => &mut self.prev,
        }
    }
}

/// The track a block shows: the one playing, the one after it, or the one
/// before it.
pub fn path(state: &State, block: Block) -> Option<&PathBuf> {
    state.queue.paths.get(index(state, block)?)
}

/// The queue index of the track a block shows (it may be past the queue).
pub fn index(state: &State, block: Block) -> Option<usize> {
    let index = state.playing.track.as_ref()?.index;
    match block {
        Block::Now => Some(index),
        Block::Next => Some(index + 1),
        Block::Prev => index.checked_sub(1),
    }
}

/// The milliseconds of the frame being drawn (the clock between ticks).
pub fn frame_ms(state: &State) -> u64 {
    state.drawn_ms.unwrap_or(state.now_ms)
}

/// How far the slide after a hand-over has come, 0 to 1 (eased), while
/// it is under way.
pub fn slide(state: &State) -> Option<f64> {
    let since = state.mix.slide?;
    let p = frame_ms(state).saturating_sub(since) as f64 / SLIDE_MS as f64;
    (p < 1.0).then(|| 1.0 - (1.0 - p).powi(3))
}

/// The transition armed or under way from the playing track into the
/// next, as sent.
pub fn transition_into(state: &State) -> Option<&tr::Sent> {
    let d = &state.mix.draft;
    let sent = d.sent.as_ref()?;
    let playing = state.playing.track.as_ref()?.index;
    (matches!(d.status, Status::Armed | Status::Mixing(_)) && sent.plan.from == playing)
        .then_some(sent)
}

/// What is heard of a block's track, with `heard` the position heard in
/// the playing track: that; the next track's while a transition mixes it
/// in.
pub fn playhead(state: &State, which: Block, heard: f64) -> Option<f64> {
    state.playing.track.as_ref()?;
    match which {
        Block::Now => Some(heard),
        Block::Next => {
            let sent = transition_into(state)?;
            matches!(state.mix.draft.status, Status::Mixing(_)).then(|| sent.map.at(heard))
        }
        Block::Prev => None,
    }
}

/// The analysis and waveform of a block's track, once in.
pub fn rhythmed(state: &State, block: Block) -> Option<&Arc<Rhythmed>> {
    match state.rhythm.get(path(state, block)?)? {
        Rhythm::Ready(r) => Some(r),
        _ => None,
    }
}

/// A block's state as it applies to the track it shows now: what was set
/// for that track, else nothing yet.
pub fn block(state: &State, which: Block) -> MixBlock {
    let stored = state.mix.get(which);
    let path = path(state, which);
    if stored.path.as_ref() == path {
        stored.clone()
    } else {
        MixBlock {
            path: path.cloned(),
            ..MixBlock::default()
        }
    }
}

/// A block's state to change, started afresh if it was for another track.
fn block_mut(state: &mut State, which: Block) -> &mut MixBlock {
    let path = path(state, which).cloned();
    let stored = state.mix.get_mut(which);
    if stored.path != path {
        *stored = MixBlock {
            path,
            ..MixBlock::default()
        };
    }
    stored
}

/// The median length of the bars of `a`, in seconds.
fn bar_seconds(a: &Analysis) -> Option<f64> {
    let mut lens: Vec<f64> = a
        .downbeats
        .windows(2)
        .map(|w| a.beats[w[1]] - a.beats[w[0]])
        .collect();
    if lens.is_empty() {
        return None;
    }
    lens.sort_by(f64::total_cmp);
    Some(lens[lens.len() / 2])
}

/// The length of a block's track, as far as known.
fn duration(state: &State, which: Block) -> f64 {
    rhythmed(state, which).map_or(0.0, |r| {
        r.analysis
            .duration
            .max(r.waveform.as_ref().map_or(0.0, |w| w.duration()))
    })
}

/// A block's zoom: its own (the wheel's), else the tab's.
pub fn zoom_of(state: &State, which: Block) -> usize {
    block(state, which)
        .zoom
        .unwrap_or(state.mix.zoom)
        .min(ZOOMS.len() - 1)
}

/// Whether a block shows its whole track.
pub fn whole(state: &State, which: Block) -> bool {
    ZOOMS[zoom_of(state, which)] == 0
}

/// The seconds a block shows.
pub fn span(state: &State, which: Block) -> f64 {
    span_at(state, zoom_of(state, which))
}

/// The seconds zoom `zoom` (an index into [`ZOOMS`]) shows.
fn span_at(state: &State, zoom: usize) -> f64 {
    let bars = ZOOMS[zoom.min(ZOOMS.len() - 1)];
    if bars == 0 {
        return duration(state, Block::Now)
            .max(duration(state, Block::Next))
            .max(1.0);
    }
    let bar = Block::BOTH
        .iter()
        .find_map(|&b| rhythmed(state, b).and_then(|r| bar_seconds(&r.analysis)))
        .unwrap_or(FALLBACK_BAR);
    bars as f64 * bar
}

/// The time of a block's first bar, 0 without one.
fn first_bar(a: &Analysis) -> f64 {
    a.downbeats
        .first()
        .and_then(|&d| a.beats.get(d))
        .copied()
        .unwrap_or(0.0)
}

/// The seconds a block shows, `(start, end)`, with `heard` the position
/// heard in the playing track.
pub fn window(state: &State, which: Block, heard: f64) -> (f64, f64) {
    let span = span(state, which);
    if whole(state, which) {
        return (0.0, span);
    }
    let b = block(state, which);
    let into = transition_into(state).map(|s| s.map.at(heard));
    let start = match (b.start, which, into) {
        (Some(start), _, _) => start,
        (None, Block::Now, _) => heard - LEAD * span,
        // Mixed in (or about to be): it follows its own playhead, which
        // stays under the playing track's.
        (None, Block::Next, Some(at)) => at - LEAD * span,
        (None, Block::Next | Block::Prev, _) => {
            rhythmed(state, which).map_or(0.0, |r| first_bar(&r.analysis) - 0.05 * span)
        }
    };
    let start = start.max(0.0);
    (start, start + span)
}

/// The bar a beat is in, counted from 1 (0 before the first bar).
pub fn bar_of(a: &Analysis, beat: usize) -> usize {
    a.downbeats.partition_point(|&d| d <= beat)
}

/// The beat the cursor is on: where it was put, else the playing track's
/// bar under the playhead, or the next track's first bar.
pub fn cursor(state: &State, which: Block, heard: f64) -> Option<usize> {
    let a = &rhythmed(state, which)?.analysis;
    if a.beats.is_empty() {
        return None;
    }
    let b = block(state, which);
    let beat = match (b.cursor, which) {
        (Some(c), _) => c,
        (None, Block::Now) => {
            let at = a.beat_at(heard).unwrap_or(0);
            let bar = bar_of(a, at);
            bar.checked_sub(1).map_or(at, |k| a.downbeats[k])
        }
        (None, Block::Next) => a.downbeats.first().copied().unwrap_or(0),
        (None, Block::Prev) => return None,
    };
    Some(beat.min(a.beats.len() - 1))
}

/// The seconds a block's selection covers, in whole bars, if it has one.
pub fn selection(state: &State, which: Block, heard: f64) -> Option<(f64, f64)> {
    let a = &rhythmed(state, which)?.analysis;
    let b = block(state, which);
    let (anchor, cursor) = (b.anchor?, cursor(state, which, heard)?);
    let (lo, hi) = (anchor.min(cursor), anchor.max(cursor));
    let start = match bar_of(a, lo).checked_sub(1) {
        Some(k) => a.beats[a.downbeats[k]],
        None => *a.beats.first()?,
    };
    let end = a
        .downbeats
        .get(bar_of(a, hi))
        .and_then(|&d| a.beats.get(d))
        .copied()
        .unwrap_or(a.duration);
    Some((start, end))
}

/// The start of the bar `offset` bars from the one heard (0: its own
/// start) and the start of the next bar, in seconds of the playing track:
/// past the bars found, at the bar's length.
pub fn bar_jump(a: &Analysis, heard: f64, offset: isize) -> Option<(f64, f64)> {
    let starts: Vec<f64> = a
        .downbeats
        .iter()
        .filter_map(|&d| a.beats.get(d).copied())
        .collect();
    let n = starts.len();
    if n < 2 {
        return None;
    }
    let bar = (starts[n - 1] - starts[0]) / (n - 1) as f64;
    let at = |i: isize| -> f64 {
        if i < 0 {
            starts[0] + i as f64 * bar
        } else if i as usize >= n {
            starts[n - 1] + (i - (n as isize - 1)) as f64 * bar
        } else {
            starts[i as usize]
        }
    };
    // The bar heard (before the first bar found: a bar's length before it).
    let k = starts.partition_point(|&t| t <= heard) as isize - 1;
    let to = at(k + offset);
    (to >= 0.0 && to < a.duration).then(|| (to, at(k + 1)))
}

/// The playing track jumps to the start of the bar `offset` bars from the
/// one heard (0: its own start): at once, or (`at_end`) when the bar heard
/// ends. Without a gap: the engine keeps the bars around what is heard.
pub(crate) fn jump_bar(state: &mut State, offset: isize, at_end: bool, fx: &mut Vec<Effect>) {
    let (Some(index), Some(_)) = (state.queue.current, state.playing.track.as_ref()) else {
        return;
    };
    if state.mix.draft.status != Status::Draft {
        return super::update::say_info(state, "No bar jumps while a transition is set up");
    }
    let Some(r) = rhythmed(state, Block::Now).cloned() else {
        return super::update::say_info(state, "No bars yet: the track is being analysed");
    };
    let Some((to, end)) = bar_jump(&r.analysis, state.heard(), offset) else {
        return super::update::say_info(state, "No bar there");
    };
    let at = if at_end {
        jodel_playback::JumpAt::At(end)
    } else {
        jodel_playback::JumpAt::Now
    };
    fx.push(Effect::Engine(Command::Jump { index, to, at }));
}

/// The beat `bars` bars from `from` (on a bar's first beat: whole bars;
/// within a bar: to the bar's first beat first, going back). Without bars,
/// four beats a bar.
fn by_bars(a: &Analysis, from: usize, bars: isize) -> usize {
    let n = a.beats.len();
    if a.downbeats.is_empty() {
        return (from as isize + 4 * bars).clamp(0, n as isize - 1) as usize;
    }
    let started = a.downbeats.partition_point(|&d| d <= from) as isize;
    let on_bar = started > 0 && a.downbeats[started as usize - 1] == from;
    // Back from within a bar, its own first beat is the first step.
    let target = started - 1 + bars + isize::from(bars < 0 && !on_bar);
    let k = target.clamp(0, a.downbeats.len() as isize - 1) as usize;
    a.downbeats[k]
}

/// Puts a block's cursor on `beat` (selecting from where it was when
/// `select`), and brings it into view.
fn put_cursor(state: &mut State, which: Block, beat: usize, select: bool, heard: f64) {
    let old = cursor(state, which, heard);
    let span = span(state, which);
    let (start, end) = window(state, which, heard);
    let Some(r) = rhythmed(state, which).cloned() else {
        return;
    };
    let time = r.analysis.beats.get(beat).copied().unwrap_or(0.0);
    let whole = whole(state, which);
    let following = which == Block::Now || transition_into(state).is_some();
    let b = block_mut(state, which);
    if select {
        if b.anchor.is_none() {
            b.anchor = old;
        }
    } else {
        b.anchor = None;
    }
    b.cursor = Some(beat);
    if !whole && (time < start || time >= end) {
        b.start = Some((time - LEAD * span).max(0.0));
    } else if b.start.is_none() && following {
        // The view stops following where it is.
        b.start = Some(start);
    }
}

/// Moves a block's cursor by bars or beats.
fn step(state: &mut State, bars: isize, beats: isize, select: bool, heard: f64) {
    let which = state.mix.focus;
    let Some(r) = rhythmed(state, which).cloned() else {
        return;
    };
    let a = &r.analysis;
    let Some(from) = cursor(state, which, heard) else {
        return;
    };
    // By whole bars of the slower track where the beats pair two to one.
    let per = pairing(state);
    let every = match which {
        Block::Now => per.bars_per_step(),
        _ => per.next_bars_per_step(),
    } as isize;
    let to = if bars != 0 {
        by_bars(a, from, bars * every)
    } else {
        (from as isize + beats).clamp(0, a.beats.len() as isize - 1) as usize
    };
    put_cursor(state, which, to, select, heard);
}

/// Zooms the tab by `by` levels ([`set_zoom`]).
fn zoom(state: &mut State, by: isize, heard: f64) {
    let z = (state.mix.zoom as isize + by).clamp(0, ZOOMS.len() as isize - 1) as usize;
    set_zoom(state, z, heard);
}

/// Sets the tab's zoom (both blocks, their own zooms dropped), keeping the
/// middle of each block's view where it is.
fn set_zoom(state: &mut State, z: usize, heard: f64) {
    let old: Vec<(f64, f64)> = Block::BOTH
        .iter()
        .map(|&b| window(state, b, heard))
        .collect();
    state.mix.zoom = z;
    for (i, which) in Block::BOTH.into_iter().enumerate() {
        block_mut(state, which).zoom = None;
        let span = span(state, which);
        let mid = 0.5 * (old[i].0 + old[i].1);
        let b = block_mut(state, which);
        if b.start.is_some() {
            b.start = Some((mid - 0.5 * span).max(0.0));
        }
    }
}

/// The widest zoom at which the focused track's lanes leave out no word of
/// its lyrics (no more than they must at any zoom): set as that track's own
/// zoom (the other keeps its own), and said. A view following the playhead
/// goes on following it; one moved by hand keeps its middle.
fn fit_lyrics(state: &mut State, heard: f64) {
    let which = state.mix.focus;
    let Some(width) = lane_width(state) else {
        return;
    };
    let lyrics = lyrics_of_block(state, which);
    let words = lyrics.as_deref().map(fitted_words).unwrap_or_default();
    if words.is_empty() {
        super::update::say_info(state, "No timed lyrics to fit the zoom to");
        return;
    }
    let least = words_left_out(&words, LYRICS_MOST_PER, 1);
    let fits = |z: usize| {
        let per = f64::from(width) / span_at(state, z).max(1e-3);
        words_left_out(&words, per, 1) <= least
    };
    let fitting = (0..ZOOMS.len()).rev().find(|&z| fits(z));
    let z = fitting.unwrap_or(0);
    let from = zoom_of(state, which);
    // A view following the playhead (or, the next track's, at its first
    // bar) goes on doing so; one moved by hand stays around its middle.
    let following = block(state, which).start.is_none();
    zoom_block(state, which, z as isize - from as isize, None, heard);
    if following {
        block_mut(state, which).start = None;
    }
    let shown = match ZOOMS[z] {
        0 => "the whole track".to_owned(),
        1 => "1 bar".to_owned(),
        bars => format!("{bars} bars"),
    };
    let track = match which {
        Block::Next => "the next track",
        _ => "the playing track",
    };
    let said = if fitting.is_some() {
        format!("Zoomed {track} to fit its lyrics: {shown}")
    } else {
        format!("Zoomed {track} in all the way ({shown}); some words still do not fit")
    };
    super::update::say_info(state, said);
}
/// The width of the Mix tab's lanes, in cells.
fn lane_width(state: &State) -> Option<u16> {
    let super::layout::Layout::Main(m) = super::layout::compute(state) else {
        return None;
    };
    let b = m.mix?.blocks.into_iter().next()?;
    let width = b.lanes.first().map_or(b.overview.width, |l| l.2.width);
    (width > 0).then_some(width)
}

/// The lyrics of a block's track, once found.
pub fn lyrics_of_block(state: &State, which: Block) -> Option<Arc<jodel_lyrics::Lyrics>> {
    use super::karaoke::{lyrics_of, SongLyrics};
    match index(state, which).and_then(|i| lyrics_of(state, i)) {
        Some(SongLyrics::Found { shown, .. }) => Some(Arc::clone(shown)),
        _ => None,
    }
}

/// The timed words of `lyrics`, in order.
pub fn timed_words(lyrics: &jodel_lyrics::Lyrics) -> Vec<&jodel_lyrics::Word> {
    lyrics
        .lines
        .iter()
        .flat_map(|l| &l.words)
        .filter(|w| w.is_timed())
        .collect()
}

/// The words of `lyrics` a zoom is fitted to: those timed where they were
/// heard (or given their times by the source), not those the aligner did
/// not hear and timed between their neighbours, often squeezed into a gap
/// (an echo of backing words in a fifth of a second would set the zoom of
/// the whole song). They are still drawn where there is room. All the
/// timed words when none were heard.
pub fn fitted_words(lyrics: &jodel_lyrics::Lyrics) -> Vec<&jodel_lyrics::Word> {
    let timed = timed_words(lyrics);
    let heard: Vec<&jodel_lyrics::Word> = timed
        .iter()
        .copied()
        .filter(|w| w.confidence >= jodel_lyrics::CONFIDENT)
        .collect();
    if heard.is_empty() {
        timed
    } else {
        heard
    }
}

/// The most cells a second a lane's words are laid out at to fit them:
/// faster, they would scroll by too fast to read.
pub const LYRICS_MOST_PER: f64 = 40.0;
/// The least (a lane over a song's quiet stretch shows no more than this).
pub const LYRICS_LEAST_PER: f64 = 3.0;

/// The lines a lane's words go on, and whether a word with no room is
/// left out or goes on later.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordLines {
    /// Lines of words: the first, then each under the one before.
    pub rows: usize,
    /// A word with no room anywhere near where it is sung still goes in,
    /// after the one before (late), rather than being left out.
    pub keep_all: bool,
}

impl WordLines {
    /// One line; words with no room left out (the Mix lanes).
    pub const ONE: WordLines = WordLines {
        rows: 1,
        keep_all: false,
    };
    /// One line; every word goes in, each where it is sung (at the zoom
    /// that fits them all), but for words sung at the same moment, which
    /// go on after the one before (the Karaoke tab's voice lane).
    pub const KARAOKE: WordLines = WordLines {
        rows: 1,
        keep_all: true,
    };
}

/// Lays `words` (timed, in order: the whole song's) out along a lane: each
/// at `x_of(its start)` (the lane's units, cells or pixels, from a fixed
/// origin), a character `cell` units wide at `per` units a second, as wide
/// as it is sung (up to the next word's start, never cut below its
/// characters), on the first of `lines.rows` lines where its place is free
/// (a word crowding the one before on the first line goes under it, where
/// it is sung). Where it is taken on every line, it goes after the word on
/// the line that frees first (a character between them), or is left out
/// once that is past where it ends (unless `lines.keep_all`).
/// `place(k, at, wide, row)` is given each word laid out (its index, where
/// it starts, its width in characters and its line) and says whether to go
/// on. Returns how many were left out (or, with `keep_all`, went in late).
/// The words are laid out from the song's start, so where one stands
/// never depends on which others are in view.
pub fn lay_out_words(
    words: &[&jodel_lyrics::Word],
    x_of: &dyn Fn(f64) -> f64,
    per: f64,
    cell: f64,
    lines: WordLines,
    place: &mut dyn FnMut(usize, f64, usize, usize) -> bool,
) -> usize {
    let rows = lines.rows.clamp(1, 4);
    let chars_per_second = per / cell.max(1e-9);
    let mut free = [f64::NEG_INFINITY; 4];
    let mut left_out = 0;
    for (k, w) in words.iter().enumerate() {
        let (Some(start), Some(end)) = (w.start, w.end) else {
            continue;
        };
        let letters = w.text.chars().count();
        if letters == 0 {
            continue;
        }
        let x = x_of(start);
        // The first line free where it is sung, else the one that frees
        // first.
        let (row, at) = match (0..rows).find(|&r| free[r] <= x) {
            Some(r) => (r, x),
            None => {
                let r = (0..rows)
                    .min_by(|&a, &b| free[a].total_cmp(&free[b]))
                    .unwrap_or(0);
                (r, free[r])
            }
        };
        if at > x_of(end) + cell {
            left_out += 1;
            if !lines.keep_all {
                continue;
            }
        }
        let next = words
            .get(k + 1)
            .and_then(|n| n.start)
            .map_or(f64::INFINITY, x_of);
        let room = ((next - at) / cell - 1.0).floor().clamp(0.0, 1e6) as usize;
        let sung = ((end - start) * chars_per_second).floor().max(0.0) as usize;
        let wide = sung.min(room).max(letters);
        free[row] = at + (wide + 1) as f64 * cell;
        if !place(k, at, wide, row) {
            break;
        }
    }
    left_out
}

/// How many of `words` a lane of `rows` lines leaves out at `per` cells a
/// second.
pub fn words_left_out(words: &[&jodel_lyrics::Word], per: f64, rows: usize) -> usize {
    let lines = WordLines {
        rows,
        keep_all: false,
    };
    lay_out_words(
        words,
        &|t| (t * per).floor(),
        per,
        1.0,
        lines,
        &mut |_, _, _, _| true,
    )
}

/// The fewest cells a second at which a lane of `rows` lines leaves out no
/// more of `words` than it must (those it leaves out even at
/// [`LYRICS_MOST_PER`]: words sung at the same moment), a little more for
/// room; `None` without words.
pub fn lyrics_per_second(words: &[&jodel_lyrics::Word], rows: usize) -> Option<f64> {
    if words.is_empty() {
        return None;
    }
    let least = words_left_out(words, LYRICS_MOST_PER, rows);
    let fits = |per: f64| words_left_out(words, per, rows) <= least;
    if fits(LYRICS_LEAST_PER) {
        return Some(LYRICS_LEAST_PER);
    }
    let (mut lo, mut hi) = (LYRICS_LEAST_PER, LYRICS_MOST_PER);
    while hi / lo > 1.01 {
        let mid = (lo * hi).sqrt();
        if fits(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    let roomy = (hi * 1.1).min(LYRICS_MOST_PER);
    Some(if fits(roomy) { roomy } else { hi })
}
/// Zooms one block (the wheel over it), keeping `at` (seconds into its
/// track, under the pointer) where it is on screen; without `at`, its
/// middle.
fn zoom_block(state: &mut State, which: Block, by: isize, at: Option<f64>, heard: f64) {
    let (start, end) = window(state, which, heard);
    let from = zoom_of(state, which);
    let z = (from as isize + by).clamp(0, ZOOMS.len() as isize - 1) as usize;
    if z == from {
        return;
    }
    let at = at.unwrap_or(0.5 * (start + end)).clamp(start, end);
    // Where the time is across the view, 0 to 1.
    let share = if end > start {
        (at - start) / (end - start)
    } else {
        0.5
    };
    block_mut(state, which).zoom = Some(z);
    let span = span(state, which);
    if !whole(state, which) {
        block_mut(state, which).start = Some((at - share * span).max(0.0));
    }
}

/// A grid fix on the focused block's track: the analysis in the state
/// takes it at once, and the fixed grid is saved.
fn fix(state: &mut State, fx: &mut Vec<Effect>, heard: f64, how: impl Fn(&Grid, usize) -> Grid) {
    let which = state.mix.focus;
    let (Some(path), Some(r)) = (path(state, which).cloned(), rhythmed(state, which).cloned())
    else {
        super::update::say_info(state, "No beats to fix here yet");
        return;
    };
    let Some(at) = cursor(state, which, heard) else {
        return;
    };
    let grid = how(&Grid::of(&r.analysis), at);
    let analysis = Arc::new(grid.apply(&r.analysis));
    let (old, new) = (&r.analysis.beats, analysis.beats.clone());
    // A beat of the old grid as the beat of the new one nearest its time
    // (half and double tempo number the beats afresh).
    let same = |beat: Option<usize>| beat.and_then(|b| old.get(b)).map(|&t| nearest(&new, t));
    state.rhythm.insert(
        path.clone(),
        Rhythm::Ready(Arc::new(Rhythmed {
            analysis,
            found: Arc::clone(&r.found),
            waveform: r.waveform.clone(),
        })),
    );
    let b = block_mut(state, which);
    b.cursor = same(b.cursor);
    b.cue = b
        .cue
        .and_then(|c| same(Some(c.beat)).map(|beat| Cue { beat, ..c }));
    b.anchor = None;
    fx.push(Effect::SaveGrid(path, grid));
}

/// The index of the beat of `beats` nearest `time` (0 without beats).
fn nearest(beats: &[f64], time: f64) -> usize {
    let i = beats.partition_point(|&b| b < time);
    match (i.checked_sub(1), beats.get(i)) {
        (Some(before), Some(&after)) if time - beats[before] <= after - time => before,
        (Some(before), None) => before,
        _ => i.min(beats.len().saturating_sub(1)),
    }
}

/// Back to the grid the analysis found.
fn reset(state: &mut State, fx: &mut Vec<Effect>) {
    let which = state.mix.focus;
    let (Some(path), Some(r)) = (path(state, which).cloned(), rhythmed(state, which).cloned())
    else {
        return;
    };
    if !r.fixed() {
        super::update::say_info(state, "The grid is as the analysis found it");
        return;
    }
    state.rhythm.insert(
        path.clone(),
        Rhythm::Ready(Arc::new(Rhythmed {
            analysis: Arc::clone(&r.found),
            found: Arc::clone(&r.found),
            waveform: r.waveform.clone(),
        })),
    );
    let (old, new) = (&r.analysis.beats, &r.found.beats);
    let same = |beat: Option<usize>| beat.and_then(|b| old.get(b)).map(|&t| nearest(new, t));
    let b = block_mut(state, which);
    b.cursor = same(b.cursor);
    b.cue = b
        .cue
        .and_then(|c| same(Some(c.beat)).map(|beat| Cue { beat, ..c }));
    b.anchor = None;
    fx.push(Effect::ResetGrid(path));
    super::update::say_info(state, "The grid is back as the analysis found it");
}

/// Whether the Mix tab is what keys act on.
pub fn active(state: &State) -> bool {
    state.screen == Screen::Main && state.side == Side::Mix && state.palette.is_none()
}

/// Carries out `op` on the Mix tab; `false` when it is not one of the
/// tab's (or the tab is not shown).
pub(crate) fn apply(state: &mut State, op: Op, fx: &mut Vec<Effect>) -> bool {
    if !active(state) {
        return false;
    }
    let heard = state.heard();
    if let Some(at) = state.mix.draft.edit.as_mut() {
        match op {
            Op::Up => {
                at.lane = at.lane.saturating_sub(1);
                return true;
            }
            Op::Down => {
                at.lane = (at.lane + 1).min(2 * STEMS - 1);
                return true;
            }
            Op::Back => {
                state.mix.draft.edit = None;
                return true;
            }
            _ => {}
        }
    }
    match op {
        Op::Up => state.mix.focus = Block::Now,
        Op::Down => state.mix.focus = Block::Next,
        Op::MixBarBack => step(state, -1, 0, false, heard),
        Op::MixBarForward => step(state, 1, 0, false, heard),
        Op::MixBeatBack => step(state, 0, -1, false, heard),
        Op::MixBeatForward => step(state, 0, 1, false, heard),
        Op::MixSelectBack => step(state, -1, 0, true, heard),
        Op::MixSelectForward => step(state, 1, 0, true, heard),
        Op::PageUp => step(state, -PAGE_BARS, 0, false, heard),
        Op::PageDown => step(state, PAGE_BARS, 0, false, heard),
        Op::MixZoomIn => zoom(state, -1, heard),
        Op::MixZoomOut => zoom(state, 1, heard),
        Op::MixFitLyrics => fit_lyrics(state, heard),
        Op::MixLyrics => {
            state.config.mix_lyrics = !state.config.mix_lyrics;
            fx.push(Effect::SaveConfig);
            let now = if state.config.mix_lyrics { "on" } else { "off" };
            super::update::say_info(state, format!("Lyrics on the vocals lane: {now}"));
        }
        Op::MixFollow => {
            let b = block_mut(state, Block::Now);
            b.start = None;
            b.cursor = None;
            b.anchor = None;
            state.mix.focus = Block::Now;
        }
        // Enter never moves playback: it sets the focused track's cue.
        Op::Accept => set_cue(state, fx, heard),
        // Esc stays on the tab (tab and its buttons leave it): it clears
        // the selection, else calls off a transition armed, else clears the
        // focused track's cue.
        Op::Back => {
            let which = state.mix.focus;
            let b = block_mut(state, which);
            if b.anchor.is_some() {
                b.anchor = None;
            } else {
                match state.mix.draft.status {
                    Status::Armed => call_off(state, fx),
                    Status::Mixing(_) => {
                        super::update::say_info(state, "A transition is under way");
                    }
                    Status::Draft => {
                        let b = block_mut(state, which);
                        if b.cue.take().is_some() {
                            state.mix.draft.edit = None;
                            super::update::say_info(state, "Cue cleared");
                        }
                    }
                }
            }
        }
        Op::GridEarlier => fix(state, fx, heard, |g, _| g.shifted(-1)),
        Op::GridLater => fix(state, fx, heard, |g, _| g.shifted(1)),
        Op::GridNudgeBack => fix(state, fx, heard, |g, _| g.nudged(-NUDGE)),
        Op::GridNudgeForward => fix(state, fx, heard, |g, _| g.nudged(NUDGE)),
        Op::GridHalf => fix(state, fx, heard, |g, _| g.halved()),
        Op::GridDouble => fix(state, fx, heard, |g, _| g.doubled()),
        Op::GridOne => fix(state, fx, heard, |g, at| g.with_one_at(at)),
        Op::GridReset => reset(state, fx),
        Op::TransitionGo => go(state, fx),
        // During the mix the plan heard stays as it is.
        Op::TransitionPreset
        | Op::TransitionTempo
        | Op::TransitionPairing
        | Op::TransitionEdit
        | Op::EditBack
        | Op::EditForward
        | Op::EditRaise
        | Op::EditLower
        | Op::EditOff
        | Op::EditFull
        | Op::EditCut
            if matches!(state.mix.draft.status, Status::Mixing(_)) =>
        {
            super::update::say_info(state, "A transition is under way: it plays as it is");
        }
        Op::TransitionPreset => {
            let d = &mut state.mix.draft;
            d.preset = d.preset.next();
            d.lanes = None;
            let name = d.preset.name();
            // Remembered: the next run starts from it.
            state.config.mix_preset = state.mix.draft.preset;
            fx.push(Effect::SaveConfig);
            super::update::say_info(state, format!("Transition: {name}"));
            rearm(state, fx, true);
        }
        Op::TransitionTempo => {
            let d = &mut state.mix.draft;
            d.tempo = match d.tempo {
                TempoMode::Varispeed => TempoMode::Stretch,
                TempoMode::Stretch => TempoMode::Varispeed,
            };
            let what = tempo_name(d.tempo);
            state.config.mix_tempo = state.mix.draft.tempo;
            fx.push(Effect::SaveConfig);
            super::update::say_info(state, format!("Transition: {what}"));
            rearm(state, fx, false);
        }
        Op::TransitionPairing => {
            let d = &mut state.mix.draft;
            d.pairing = match d.pairing {
                None => Some(tr::Pairing::One),
                Some(tr::Pairing::One) => Some(tr::Pairing::Two),
                Some(tr::Pairing::Two) => Some(tr::Pairing::Half),
                Some(tr::Pairing::Half) => None,
            };
            let label = pairing(state).label();
            let what = match state.mix.draft.pairing {
                None => format!("{label} (by the tempi)"),
                Some(_) => label.to_owned(),
            };
            super::update::say_info(state, format!("Transition: beats paired {what}"));
            repair(state, fx);
        }
        Op::TransitionEdit => {
            let cued = Block::BOTH.iter().any(|&b| block(state, b).cue.is_some());
            if !cued && state.mix.draft.status == Status::Draft {
                super::update::say_info(state, "Set a cue first: enter on the bars to mix over");
                return true;
            }
            let bars = draft_bars(state);
            let d = &mut state.mix.draft;
            if d.edit.take().is_none() {
                d.lanes = Some(d.lanes(bars));
                d.edit = Some(Edit::default());
            }
        }
        Op::EditBack
        | Op::EditForward
        | Op::EditRaise
        | Op::EditLower
        | Op::EditOff
        | Op::EditFull
        | Op::EditCut => edit(state, op, fx),
        _ => return false,
    }
    true
}

/// The tempo mode's name.
pub fn tempo_name(tempo: TempoMode) -> &'static str {
    match tempo {
        TempoMode::Varispeed => "varispeed",
        TempoMode::Stretch => "time-stretch",
    }
}

/// The cue `enter` sets on a block: its selection's bars, else
/// [`tr::DEFAULT_BARS`] from its cursor's bar (the playing track's, while
/// its cursor follows the playhead: from the first bar the engine can still
/// get ready for, [`GO_AFTER`] on).
pub fn cue_here(state: &State, which: Block, heard: f64) -> Option<Cue> {
    let a = &rhythmed(state, which)?.analysis;
    let b = block(state, which);
    let cursor = cursor(state, which, heard)?;
    let bar_start = |beat: usize| {
        bar_of(a, beat)
            .checked_sub(1)
            .map_or(beat, |k| a.downbeats[k])
    };
    match (b.anchor, b.cursor, which) {
        (Some(anchor), _, _) => {
            let (lo, hi) = (anchor.min(cursor), anchor.max(cursor));
            let bars = (bar_of(a, hi) + 1)
                .saturating_sub(bar_of(a, lo).max(1))
                .max(1);
            Some(Cue {
                beat: bar_start(lo),
                bars,
                chosen: true,
            })
        }
        (None, None, Block::Now) => Some(Cue {
            beat: first_bar_after(a, heard + GO_AFTER)?,
            bars: tr::DEFAULT_BARS,
            chosen: false,
        }),
        _ => Some(Cue {
            beat: bar_start(cursor),
            bars: tr::DEFAULT_BARS,
            chosen: false,
        }),
    }
}

/// The seconds a block's cue covers on its own track, and its bars.
pub fn cue_window(state: &State, which: Block) -> Option<(f64, f64, usize)> {
    let a = &rhythmed(state, which)?.analysis;
    let cue = block(state, which).cue?;
    tr::window(a, cue.beat, cue.bars)
}

/// The bars the transition mixes over, in the playing track's bars, with
/// `now` the playing track's cue: the longer of the two cues (the next
/// track's counted in the playing track's bars, by how the beats pair),
/// of those selected if either was; in whole bars of the slower track.
fn mixed_bars(state: &State, now: Option<Cue>) -> usize {
    let per = pairing(state);
    let next = block(state, Block::Next).cue.map(|c| Cue {
        bars: (c.bars * per.bars_per_step()).div_ceil(per.next_bars_per_step()),
        ..c
    });
    let cues = [now, next];
    let chosen = cues.iter().flatten().any(|c| c.chosen);
    let bars = cues
        .iter()
        .flatten()
        .filter(|c| c.chosen || !chosen)
        .map(|c| c.bars)
        .max()
        .unwrap_or(tr::DEFAULT_BARS);
    let every = per.bars_per_step();
    bars.max(1).div_ceil(every) * every
}

/// The window on the playing track: from its cue's first bar, over the
/// bars of the longer cue ([`mixed_bars`]). `None` without a cue there.
pub fn draft_window(state: &State) -> Option<(f64, f64, usize)> {
    let a = &rhythmed(state, Block::Now)?.analysis;
    let cue = block(state, Block::Now).cue?;
    tr::window(a, cue.beat, mixed_bars(state, Some(cue)))
}

/// `enter`: the focused block's cue, from its selection or its cursor
/// ([`cue_here`]). Playback stays as it is; a transition armed takes the
/// new cue.
fn set_cue(state: &mut State, fx: &mut Vec<Effect>, heard: f64) {
    let which = state.mix.focus;
    if which == Block::Prev {
        return;
    }
    if matches!(state.mix.draft.status, Status::Mixing(_)) {
        return super::update::say_info(state, "A transition is under way: it plays as it is");
    }
    let Some(cue) = cue_here(state, which, heard) else {
        return super::update::say_info(state, "No bars here yet");
    };
    let bar = rhythmed(state, which).map_or(0, |r| bar_of(&r.analysis, cue.beat));
    let b = block_mut(state, which);
    b.cue = Some(cue);
    b.anchor = None;
    let track = match which {
        Block::Now => "the playing track",
        _ => "the next track",
    };
    let bars = cue.bars;
    let last = bar + bars - 1;
    let what = if bars == 1 {
        format!("bar {bar}")
    } else {
        format!("bars {bar}–{last}")
    };
    let other = match which {
        Block::Now => Block::Next,
        _ => Block::Now,
    };
    let then = if block(state, other).cue.is_none() {
        let there = match other {
            Block::Now => "the playing track",
            _ => "the next track",
        };
        format!("; set one in {there} too")
    } else if state.mix.draft.status == Status::Draft {
        "; g arms the transition".to_owned()
    } else {
        String::new()
    };
    super::update::say_info(state, format!("Cue on {track}: {what}{then}"));
    if state.mix.draft.status == Status::Armed {
        // The transition armed moves to the new cue (if it can).
        match cued_plan(state) {
            Ok(sent) => send(state, fx, sent),
            Err(why) => super::update::say_info(state, format!("The armed one stays: {why}")),
        }
    }
}

/// How the beats pair: for the transition sent while it is armed or
/// mixing, else as `r` set it, else by the two tracks' tempi.
pub fn pairing(state: &State) -> tr::Pairing {
    let d = &state.mix.draft;
    if let (Status::Armed | Status::Mixing(_), Some(sent)) = (&d.status, &d.sent) {
        return sent.pairing;
    }
    chosen_pairing(state)
}

/// How the beats are to pair: as `r` set it, else by the two tracks'
/// tempi.
fn chosen_pairing(state: &State) -> tr::Pairing {
    if let Some(p) = state.mix.draft.pairing {
        return p;
    }
    let (a, b) = meeting_tempi(state);
    tr::pairing_for(a, b)
}

/// The two tracks' tempi where they meet: the playing track's from its
/// cursor's bar (or what is heard), the next one's from its cue (or its
/// cursor, or its first bar), over [`tr::TEMPO_SPAN`].
pub fn meeting_tempi(state: &State) -> (Option<f64>, Option<f64>) {
    let heard = state.heard();
    let near = |which: Block, at: Option<usize>, or: f64| {
        rhythmed(state, which).and_then(|r| {
            let a = &r.analysis;
            let from = at.and_then(|i| a.beats.get(i).copied()).unwrap_or(or);
            tr::beat_tempo_near(a, from, tr::TEMPO_SPAN)
        })
    };
    let now = block(state, Block::Now);
    let next = block(state, Block::Next);
    let first_b = rhythmed(state, Block::Next)
        .and_then(|r| r.analysis.downbeats.first().map(|&d| r.analysis.beats[d]))
        .unwrap_or(0.0);
    let cued = |b: &MixBlock| b.cue.map(|c| c.beat).or(b.anchor).or(b.cursor);
    (
        near(Block::Now, cued(&now), heard),
        near(Block::Next, cued(&next), first_b),
    )
}

/// The first beat of the first bar at or after `t`.
fn first_bar_after(a: &Analysis, t: f64) -> Option<usize> {
    a.downbeats.iter().copied().find(|&d| a.beats[d] >= t)
}

/// The bars of the transition shown: the plan sent while it is armed or
/// mixing, else the window set up.
pub fn draft_bars(state: &State) -> usize {
    let d = &state.mix.draft;
    match (&d.status, &d.sent) {
        (Status::Armed | Status::Mixing(_), Some(sent)) => sent.bars,
        _ => match draft_window(state) {
            Some(w) => w.2,
            None if block(state, Block::Next).cue.is_some() => mixed_bars(state, None),
            None => tr::DEFAULT_BARS,
        },
    }
}

/// The transition shown and its bars: the plan sent while it is armed or
/// mixing, else the one set up.
pub fn shown_plan(state: &State) -> Option<(TransitionPlan, usize)> {
    let d = &state.mix.draft;
    match (&d.status, &d.sent) {
        (Status::Armed | Status::Mixing(_), Some(sent)) => Some((sent.plan.clone(), sent.bars)),
        _ => draft_plan(state, None).ok(),
    }
}

/// The plan of the transition set up and its window's bars: the window
/// ([`draft_window`], or from the first bar at least `soon` seconds after
/// what is heard), the next track's cue (or, with `soon`, its cursor or
/// first bar), the lanes and the tempo mode.
pub fn draft_plan(state: &State, soon: Option<f64>) -> Result<(TransitionPlan, usize), String> {
    let from = state
        .playing
        .track
        .as_ref()
        .ok_or("nothing is playing")?
        .index;
    let a = &rhythmed(state, Block::Now)
        .ok_or("the playing track is not analysed yet")?
        .analysis;
    let b = &rhythmed(state, Block::Next)
        .ok_or("the next track is not analysed yet")?
        .analysis;
    if b.beats.is_empty() || a.downbeats.is_empty() {
        return Err("no bars to mix on".into());
    }
    let window = match soon {
        Some(after) => {
            let bars = draft_window(state).map_or(tr::DEFAULT_BARS, |w| w.2);
            let first = first_bar_after(a, state.heard() + after)
                .ok_or("the playing track has no bar left to start on")?;
            tr::window(a, first, bars)
        }
        None => Some(draft_window(state).ok_or("no cue in the playing track")?),
    }
    .ok_or("no bars to mix on")?;
    let next = block(state, Block::Next);
    let entry = match (next.cue, soon) {
        (Some(cue), _) => cue.beat,
        (None, None) => return Err("no cue in the next track".into()),
        // A skip: where its cursor was put, else its first bar.
        (None, Some(_)) => next
            .cursor
            .unwrap_or_else(|| b.downbeats.first().copied().unwrap_or(0)),
    };
    let lanes = state.mix.draft.lanes(window.2);
    let per = pairing(state);
    let plan = tr::plan(from, a, b, window, entry, lanes, state.mix.draft.tempo, per);
    Ok((plan, window.2))
}

/// The least time before a cue's first bar that `g` arms a transition at
/// it, in seconds after what is heard (the engine writes ahead of it).
pub const SOONEST: f64 = 1.0;

/// The transition at the two cues, ready to send.
fn cued_plan(state: &State) -> Result<tr::Sent, String> {
    let missing = |b: Block| block(state, b).cue.is_none();
    match (missing(Block::Now), missing(Block::Next)) {
        (true, true) => return Err("set a cue in both tracks first (enter on their bars)".into()),
        (true, false) => return Err("set a cue in the playing track first".into()),
        (false, true) => return Err("set a cue in the next track first".into()),
        (false, false) => {}
    }
    let (plan, bars) = draft_plan(state, None)?;
    if plan.start < state.heard() + SOONEST {
        return Err("the playing track's cue has passed: set a later one".into());
    }
    Ok(tr::Sent::new(
        plan,
        bars,
        state.mix.draft.preset,
        pairing(state),
    ))
}

/// `g`: arms the transition at the two cues, or calls the armed one off.
fn go(state: &mut State, fx: &mut Vec<Effect>) {
    match state.mix.draft.status {
        // Calling it off does not cut a mix under way off (moving playback
        // does).
        Status::Mixing(_) => super::update::say_info(state, "A transition is under way"),
        Status::Armed => call_off(state, fx),
        Status::Draft => match cued_plan(state) {
            Ok(sent) => {
                let at = sent.plan.start - state.heard();
                send(state, fx, sent);
                super::update::say_info(state, format!("Transition armed: it starts in {at:.0} s"));
            }
            Err(why) => super::update::say_info(state, format!("No transition: {why}")),
        },
    }
}

/// Calls off the transition armed (its cues stay).
fn call_off(state: &mut State, fx: &mut Vec<Effect>) {
    fx.push(Effect::Engine(Command::CancelTransition));
    state.mix.draft.status = Status::Draft;
    state.mix.draft.sent = None;
    super::update::say_info(state, "Transition called off");
}

/// Sends `sent` to the engine, and keeps it.
fn send(state: &mut State, fx: &mut Vec<Effect>, sent: tr::Sent) {
    fx.push(Effect::Engine(Command::Transition(Box::new(
        sent.plan.clone(),
    ))));
    if state.mix.draft.status == Status::Draft {
        // Newly armed: the next track's view runs along with it.
        block_mut(state, Block::Next).start = None;
    }
    state.mix.draft.sent = Some(sent);
}

/// Seconds after what is heard that a skip mixes in at the soonest, and
/// where `enter` cues the playing track while its cursor follows: the
/// engine gets the tracks ready over [`jodel_playback`]'s lead.
pub const GO_AFTER: f64 = 4.5;

/// The fades (`lanes`) or the tempo mode changed while armed: the engine
/// takes the plan it has with them (its window and entry stay).
fn rearm(state: &mut State, fx: &mut Vec<Effect>, lanes: bool) {
    let d = &state.mix.draft;
    if d.status != Status::Armed {
        return;
    }
    let Some(mut sent) = d.sent.clone() else {
        return;
    };
    if lanes {
        sent.plan.lanes = d.lanes(sent.bars);
        sent.preset = d.preset;
    }
    sent.plan.tempo = d.tempo;
    send(state, fx, sent);
}

/// The pairing changed (`r`): an armed transition is sent again with its
/// beats paired the new way, from the same window start and entry (the
/// window made whole bars of the slower track).
fn repair(state: &mut State, fx: &mut Vec<Effect>) {
    let d = &state.mix.draft;
    if let (Status::Armed, Some(sent)) = (&d.status, d.sent.clone()) {
        let per = chosen_pairing(state);
        resend(state, fx, sent, per);
    }
}

/// `sent` again, its beats paired by `per`.
fn resend(state: &mut State, fx: &mut Vec<Effect>, sent: tr::Sent, per: tr::Pairing) {
    let (Some(a), Some(b)) = (rhythmed(state, Block::Now), rhythmed(state, Block::Next)) else {
        return;
    };
    let (a, b) = (&a.analysis, &b.analysis);
    let first = a.beats.partition_point(|&t| t < sent.plan.start - 1e-6);
    let entry = b.beats.partition_point(|&t| t < sent.plan.entry - 1e-6);
    let bars = sent.bars.div_ceil(per.bars_per_step()) * per.bars_per_step();
    let Some(window) = tr::window(a, first, bars) else {
        return;
    };
    let lanes = state.mix.draft.lanes(window.2);
    let plan = tr::plan(
        sent.plan.from,
        a,
        b,
        window,
        entry,
        lanes,
        state.mix.draft.tempo,
        per,
    );
    send(state, fx, tr::Sent::new(plan, window.2, sent.preset, per));
}

/// An editing key on the lanes.
fn edit(state: &mut State, op: Op, fx: &mut Vec<Effect>) {
    let bars = draft_bars(state);
    let d = &mut state.mix.draft;
    let Some(mut at) = d.edit else {
        return;
    };
    let lanes = d.lanes.get_or_insert_with(|| d.preset.lanes(bars));
    let share = at.bar as f64 / bars as f64;
    let now = lanes[at.lane].gain(share);
    let cut = tr::point_at(lanes, at.lane, at.bar, bars).is_some_and(|p| p.cut);
    match op {
        Op::EditBack => at.bar = at.bar.saturating_sub(1),
        Op::EditForward => at.bar = (at.bar + 1).min(bars),
        Op::EditRaise => tr::set_point(lanes, at.lane, at.bar, bars, now + tr::GAIN_STEP, cut),
        Op::EditLower => tr::set_point(lanes, at.lane, at.bar, bars, now - tr::GAIN_STEP, cut),
        Op::EditOff => tr::set_point(lanes, at.lane, at.bar, bars, 0.0, cut),
        Op::EditFull => tr::set_point(lanes, at.lane, at.bar, bars, 1.0, cut),
        Op::EditCut => tr::set_point(lanes, at.lane, at.bar, bars, now, !cut),
        _ => {}
    }
    d.edit = Some(at);
    rearm(state, fx, true);
}

/// A new engine: nothing of a transition the old one had is left (it will
/// say nothing more of it).
pub fn engine_restarted(state: &mut State) {
    let d = &mut state.mix.draft;
    d.status = Status::Draft;
    d.sent = None;
}

/// The engine says how the transition goes.
pub(crate) fn transition_event(state: &mut State, event: TransitionState) {
    let d = &mut state.mix.draft;
    match event {
        TransitionState::Armed => d.status = Status::Armed,
        // Only for one it has: a late word of one called off is not taken.
        TransitionState::Started { output_frame } => {
            if d.sent.is_some() {
                d.status = Status::Mixing(output_frame);
            }
        }
        TransitionState::Finished => {
            *d = Draft {
                preset: d.preset,
                tempo: d.tempo,
                ..Draft::default()
            };
            super::update::say_info(state, "Transition done");
        }
        TransitionState::Cancelled { reason } => {
            d.status = Status::Draft;
            d.sent = None;
            // Playback moved on by the user's own key or click: no news.
            if reason != "playback moved elsewhere" {
                super::update::say_info(state, format!("Transition called off: {reason}"));
            }
        }
    }
}

/// A point of a block's lanes was pressed (`extend`: dragged to, which
/// selects): the cursor goes to the beat at or before `time`. Whether it
/// was a double click (a second press on that beat within
/// [`super::DOUBLE_CLICK_MS`]).
pub(crate) fn point(state: &mut State, which: Block, time: f64, extend: bool) -> bool {
    state.mix.focus = which;
    let heard = state.heard();
    let Some(r) = rhythmed(state, which).cloned() else {
        return false;
    };
    let a = &r.analysis;
    if a.beats.is_empty() {
        return false;
    }
    let beat = a.beat_at(time).unwrap_or(0);
    let now = state.now_ms;
    let double = !extend
        && state.mix.last_press.is_some_and(|(b, at, ms)| {
            b == which && at == beat && now.saturating_sub(ms) <= super::DOUBLE_CLICK_MS
        });
    state.mix.last_press = (!extend && !double).then_some((which, beat, now));
    if extend {
        let b = block_mut(state, which);
        if b.anchor.is_none() {
            b.anchor = b.cursor;
        }
    }
    put_cursor(state, which, beat, extend, heard);
    double
}

/// The playing track's overview was clicked at `time`: playback goes there
/// (unless Karaoke mode holds it), and the view follows the playhead.
/// With nothing to seek, the view goes there.
pub(crate) fn seek(state: &mut State, fx: &mut Vec<Effect>, time: f64) {
    let Some(duration) = state.playing.duration() else {
        return jump(state, Block::Now, time);
    };
    if super::karaoke::seek_refused(state) {
        return jump(state, Block::Now, time);
    }
    let seconds = time.clamp(0.0, duration);
    state.playing.position = seconds;
    fx.push(Effect::Engine(Command::Seek(seconds)));
    block_mut(state, Block::Now).start = None;
}

/// The next track's overview was clicked at `time`: its view goes there.
pub(crate) fn jump(state: &mut State, which: Block, time: f64) {
    if whole(state, which) {
        return;
    }
    let span = span(state, which);
    block_mut(state, which).start = Some((time - 0.5 * span).max(0.0));
}

/// The view scrolls by a share of the span (the wheel with shift).
pub(crate) fn scroll(state: &mut State, which: Block, share: f64) {
    if whole(state, which) {
        return;
    }
    let heard = state.heard();
    let (start, _) = window(state, which, heard);
    let span = span(state, which);
    block_mut(state, which).start = Some((start + share * span).max(0.0));
}

/// The next track key: with a transition armed, cut to the next track
/// now (the engine drops the transition); with one mixing, cut into the
/// next track where it is heard (the engine plays it on alone from there,
/// the playing track cut off); else, by the setting, mix into it from the
/// first bar the engine can get ready for, or cut to it.
pub(crate) fn skip(state: &mut State, fx: &mut Vec<Effect>) {
    match state.mix.draft.status {
        Status::Mixing(_) if transition_into(state).is_some() => {
            super::update::say_info(state, "Cut into the next track where it is");
        }
        Status::Armed => {}
        _ => {
            if state.config.skip_mix {
                match skip_plan(state, state.config.skip_preset) {
                    Ok(sent) => {
                        let at = sent.plan.start - state.heard();
                        let name = super::update::skip_preset_name(state);
                        super::update::say_info(
                            state,
                            format!(
                                "Mixing into the next track in {at:.0} s ({name}); N again cuts"
                            ),
                        );
                        return send(state, fx, sent);
                    }
                    Err(why) => {
                        super::update::say_info(state, format!("Cut to the next track: {why}"));
                    }
                }
            }
        }
    }
    fx.push(Effect::Engine(Command::Next));
}

/// The transition a skip starts: from the first bar at least [`GO_AFTER`]
/// on, over the bars set up (or [`tr::DEFAULT_BARS`]), into the next
/// track's cue (or first bar), faded by `preset` (`None`: the Mix tab's
/// preset as it is set, its fades as edited).
fn skip_plan(state: &State, preset: Option<tr::Preset>) -> Result<tr::Sent, String> {
    let (mut plan, bars) = draft_plan(state, Some(GO_AFTER))?;
    let d = &state.mix.draft;
    let (lanes, preset) = match preset {
        Some(p) => (p.lanes(bars), p),
        None => (d.lanes(bars), d.preset),
    };
    plan.lanes = lanes;
    Ok(tr::Sent::new(plan, bars, preset, pairing(state)))
}

/// Track `index` starts: when it is the next track (handed over to, or
/// skipped to), its block (its own zoom with it) becomes the playing
/// track's, following the playhead, its cue (where it was mixed in) gone.
/// When a transition hands over to it, the playing track's block is kept
/// as it is seen now (its view held), and the next track's block slides up
/// into its place ([`slide`]).
pub(crate) fn track_started(state: &mut State, index: usize) {
    let carried = (self::index(state, Block::Next) == Some(index)).then(|| MixBlock {
        start: None,
        cursor: None,
        anchor: None,
        cue: None,
        ..block(state, Block::Next)
    });
    let handing_over = transition_into(state).is_some_and(|s| s.plan.into == index)
        && matches!(state.mix.draft.status, Status::Mixing(_));
    if handing_over {
        hand_over(state);
    }
    if let Some(now) = carried {
        state.mix.now = now;
        state.mix.next = MixBlock::default();
    }
}

/// A transition hands over: the playing track's block kept as it is seen
/// now, the next one's sliding up into its place.
fn hand_over(state: &mut State) {
    let heard = state.heard();
    let (start, _) = window(state, Block::Now, heard);
    state.mix.prev = MixBlock {
        start: Some(start),
        cursor: None,
        anchor: None,
        ..block(state, Block::Now)
    };
    state.mix.slide = Some(frame_ms(state));
    state.mix.focus = Block::Now;
    state.mix.pan = None;
}

/// The clock: a slide that has ended is forgotten.
pub(crate) fn tick(state: &mut State) {
    if state.mix.slide.is_some() && slide(state).is_none() {
        state.mix.slide = None;
    }
}

/// The secondary button went down over a block (`overview`: over its
/// whole track) at column `column`, of an area `width` columns wide: a
/// drag of its view starts.
pub(crate) fn pan_start(state: &mut State, which: Block, column: u16, width: u16, overview: bool) {
    if whole(state, which) || width == 0 || which == Block::Prev {
        return;
    }
    let heard = state.heard();
    let (start, end) = window(state, which, heard);
    let per_column = if overview {
        -duration(state, which) / f64::from(width)
    } else {
        (end - start) / f64::from(width)
    };
    state.mix.focus = which;
    state.mix.pan = Some(Pan {
        block: which,
        per_column,
        column,
        start,
    });
}

/// The pointer is at `column` with a drag of a view under way: the view
/// moves with it (dragged right, the waveform goes right, to earlier
/// times; on the whole track, the span shown follows the pointer).
pub(crate) fn pan_to(state: &mut State, column: u16) {
    let Some(pan) = state.mix.pan else {
        return;
    };
    let moved = f64::from(column) - f64::from(pan.column);
    let start = (pan.start - moved * pan.per_column).max(0.0);
    block_mut(state, pan.block).start = Some(start);
}

/// The drag of a view ends.
pub(crate) fn pan_end(state: &mut State) {
    state.mix.pan = None;
}

/// The wheel without shift over a block: that block zooms in (`by` < 0)
/// or out, around `at` (seconds into its track, under the pointer).
pub(crate) fn wheel_zoom(state: &mut State, which: Block, by: isize, at: Option<f64>) {
    let heard = state.heard();
    zoom_block(state, which, by, at, heard);
}
