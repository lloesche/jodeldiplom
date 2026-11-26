//! The Transition tab as the window shows it ([`crate::gui::transition`]):
//! per block (the playing track's, the next one's, the one left while it
//! slides away) a header of facts ([`Header`]), the lanes' labels
//! ([`LaneTag`]) and what the lanes draw ([`Plots`]: the whole track in one
//! row, the bar numbers, and the lanes, the whole song and each stem, over
//! the span shown, drawn as the lanes are, [`super::lanes`]); and the
//! footer with the keys ([`Footer`]). `L` (the `transition_lyrics` setting)
//! turns the lyrics on the voice's lane off. Its outline ([`Tab`]) is what
//! the snapshots compare.

use std::fmt;
use std::sync::Arc;

use jodel_playback::Stem;

use super::lanes::{lane_color, WavePlot, OFF};
use super::palette;
use crate::core::keymap::{self, Context, Op};
use crate::core::lanes::{self, Block, Lane, WordLines};
use crate::core::transition::{self, draft, tab, viewport, Pairing};
use crate::core::{text, Rhythm, State};

/// The lanes' labels' column, left of the lanes, in points.
pub const LABELS: f32 = 64.0;
/// The tab's content in from its panel's edge, in points.
pub const PAD: f32 = 12.0;

/// The width of the lanes, in points, in the tab's pane (its panel)
/// `pane` points wide: the pane less its padding and the labels' column.
pub fn lane_width(pane: f32) -> f32 {
    (pane - 2.0 * PAD - LABELS).max(0.0)
}

/// The width of the lanes in letters of the words along them, a letter
/// `letter` points wide, in a pane `pane` points wide: what fits the
/// lyrics' words ([`crate::core::Fit::lane_letters`]).
pub fn lane_letters(pane: f32, letter: f32) -> u16 {
    if letter <= 0.0 {
        return 0;
    }
    (lane_width(pane) / letter)
        .floor()
        .clamp(0.0, f32::from(u16::MAX)) as u16
}

/// The lanes a block shows: every stem's where its track has them (or,
/// before its analysis is in, where stems are asked for), else the whole
/// song's.
pub fn block_lanes(state: &State, block: Block) -> &'static [Lane] {
    let stems = match lanes::rhythmed(state, block) {
        Some(r) => r.waveform.as_ref().is_some_and(|w| w.stems.is_some()),
        None => state.rhythm_recipe.stems,
    };
    if stems {
        &Lane::ALL
    } else {
        &Lane::ALL[..1]
    }
}

/// Whether a lane is a stem turned off (its strip on the desk muted).
fn is_off(state: &State, lane: Lane) -> bool {
    match lane {
        Lane::Song => false,
        Lane::Stem(stem) => crate::core::stems::is_off(state, stem),
    }
}

/// The seconds of the bars' first beats of `a`.
fn downbeats(a: &jodel_rhythm::Analysis) -> Vec<f64> {
    a.downbeats
        .iter()
        .filter_map(|&d| a.beats.get(d).copied())
        .collect()
}

/// Where the beat is unclear in `a`, as stretches of seconds.
fn unclear(a: &jodel_rhythm::Analysis) -> Vec<(f64, f64)> {
    a.regions
        .iter()
        .filter(|r| r.confidence < jodel_rhythm::CLEAR)
        .map(|r| (r.start, r.end))
        .collect()
}

/// How a fact in a header is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tone {
    /// On a quiet chip: a fact of the track.
    Chip,
    /// Lit in the accent: the transition armed or mixing.
    Lit,
    /// A warning's tone: no stems, no analysis.
    Warn,
    /// Dim words, no chip: what to do next, what is under way.
    Hint,
}

/// One fact of a block's header: the tempo, the bar, the transition's
/// preset.
#[derive(Clone, Debug, PartialEq)]
pub struct Fact {
    /// What it is, before it (`transition`, `next`, `tempo`).
    pub label: Option<&'static str>,
    /// What it says.
    pub text: String,
    /// How it is shown.
    pub tone: Tone,
    /// How long it stays where there is not room for all (higher: longer).
    pub keep: u8,
}

impl Fact {
    fn new(text: impl Into<String>, tone: Tone, keep: u8) -> Fact {
        Fact {
            label: None,
            text: text.into(),
            tone,
            keep,
        }
    }

    fn labelled(label: &'static str, text: impl Into<String>, tone: Tone, keep: u8) -> Fact {
        Fact {
            label: Some(label),
            ..Fact::new(text, tone, keep)
        }
    }
}

impl fmt::Display for Fact {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.label {
            Some(label) => write!(f, "{label} {}", self.text),
            None => f.write_str(&self.text),
        }
    }
}

/// A block's header: which track, whether it has the keys, its title (or
/// why there is none), and its facts.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    /// The block.
    pub block: Block,
    /// `Now`, `Next`, `Last`.
    pub name: &'static str,
    /// The keys move it.
    pub focused: bool,
    /// The track's title, or what there is instead (`nothing playing`).
    pub title: Result<String, &'static str>,
    /// Its tempo, key, bar and the rest, in order.
    pub facts: Vec<Fact>,
}

impl Header {
    /// The header of block `which`, with `heard` seconds of the playing
    /// track heard.
    pub fn of(state: &State, which: Block, heard: f64) -> Header {
        let name = match which {
            Block::Now => "Now",
            Block::Next => "Next",
            Block::Prev => "Last",
        };
        let focused = state.transition.focus == which;
        let Some(path) = lanes::path(state, which) else {
            let what = match which {
                Block::Now => "nothing playing",
                Block::Next => "nothing after it in the queue",
                Block::Prev => "",
            };
            return Header {
                block: which,
                name,
                focused,
                title: Err(what),
                facts: Vec::new(),
            };
        };
        let mut facts = Vec::new();
        if !state.config.bpm {
            let key = keymap::hint(Context::Global, Op::ToggleBpm);
            facts.push(Fact::new(format!("tempo is off ({key})"), Tone::Hint, 9));
        } else {
            match state.rhythm.get(path) {
                None | Some(Rhythm::Pending) => facts.push(Fact::new("analysing…", Tone::Hint, 9)),
                Some(Rhythm::Failed) => {
                    facts.push(Fact::new("could not be analysed", Tone::Warn, 9))
                }
                Some(Rhythm::Ready(r)) => {
                    let a = &r.analysis;
                    if let Some(sum) = a.summary {
                        let bpm = format!("{} BPM", text::bpm_text(sum.bpm));
                        facts.push(Fact::new(bpm, Tone::Chip, 8));
                    }
                    if let Some(key) = a.key {
                        facts.push(Fact::new(text::key_text(&key), Tone::Chip, 3));
                    }
                    let bars = a.downbeats.len();
                    if let Some(c) = viewport::cursor(state, which, heard) {
                        let bar = format!("bar {} / {bars}", viewport::bar_of(a, c));
                        facts.push(Fact::new(bar, Tone::Chip, 7));
                    }
                    if r.waveform.as_ref().is_some_and(|w| w.stems.is_none()) {
                        facts.push(Fact::new("no stems", Tone::Warn, 5));
                    }
                    if r.fixed() {
                        facts.push(Fact::new("grid fixed by hand", Tone::Chip, 2));
                    }
                    if which == Block::Now {
                        transition_facts(state, &mut facts);
                    }
                }
            }
        }
        Header {
            block: which,
            name,
            focused,
            title: Ok(crate::core::shown_title(state, path)),
            facts,
        }
    }
}

impl fmt::Display for Header {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mark = if self.focused { "▸ " } else { "" };
        write!(f, "{mark}{}", self.name)?;
        match &self.title {
            Ok(title) => write!(f, " · {title}")?,
            Err(what) if !what.is_empty() => write!(f, " · {what}")?,
            Err(_) => {}
        }
        for fact in &self.facts {
            write!(f, " · {fact}")?;
        }
        Ok(())
    }
}

/// The transition set up, as facts: its preset and tempo, where it stands,
/// and the tempi where the tracks meet.
fn transition_facts(state: &State, facts: &mut Vec<Fact>) {
    use crate::core::transition::Status;
    let d = &state.transition.draft;
    // The preset of the one sent (a skip sends its own), else the one set.
    let preset = d.in_force().map_or(d.preset, |sent| sent.preset);
    let mut set = format!("{}, {}", preset.name(), transition::tempo_name(d.tempo));
    let bars = draft::draft_bars(state);
    if d.lanes.as_ref().is_some_and(|l| *l != d.preset.lanes(bars)) {
        set.push_str(" (edited)");
    }
    facts.push(Fact::labelled("transition", set, Tone::Chip, 6));
    let cued = |b: Block| tab::block(state, b).cue.is_some();
    facts.push(match &d.status {
        Status::Draft if cued(Block::Now) && cued(Block::Next) => {
            Fact::new("g arms", Tone::Hint, 1)
        }
        Status::Draft => Fact::new("enter sets the cues", Tone::Hint, 1),
        Status::Armed => Fact::new("armed", Tone::Lit, 6),
        Status::Mixing(_) => Fact::new("mixing", Tone::Lit, 6),
    });
    // The tempi of the beats that are paired, where they meet.
    let (Some(a), Some(b)) = draft::meeting_tempi(state) else {
        return;
    };
    // The next track's tempo in the playing track's beats, by the pairing
    // (2:1: its beat is two of these), and the glide to it.
    let per = draft::pairing(state);
    let to = b * per.ratio();
    // Set by hand: what the transition set up (or armed) takes; one
    // already mixing plays what it was sent with.
    let set = d.pairing.is_some() && !matches!(d.status, Status::Mixing(_));
    let mut next = String::new();
    if per != Pairing::One || set {
        next.push_str(per.label());
        next.push_str(if set { " (set) " } else { " " });
    }
    next.push_str(&format!("{:+.1} %", 100.0 * (to / a - 1.0)));
    facts.push(Fact::labelled("next", next, Tone::Chip, 4));
    if (to / a - 1.0).abs() >= 0.002 {
        // The glide goes as far as the engine takes it; the next track
        // makes up the rest after the window.
        let most = jodel_playback::warp::GLIDE_MOST;
        let reach = a * (to / a).clamp(1.0 / most, most);
        let mut glide = format!("{a:.0} → {reach:.0} across the window");
        if (reach - to).abs() >= 0.5 {
            glide.push_str(&format!(", {to:.0} after it"));
        }
        facts.push(Fact::labelled("tempo", glide, Tone::Chip, 3));
    }
}

/// A lane's label: which lane, turned off (grey) or its fades being edited
/// (lit in its colour).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LaneTag {
    /// The lane.
    pub lane: Lane,
    /// Its stem is not heard.
    pub off: bool,
    /// Its fade is the one being edited.
    pub edited: bool,
    /// Its colour.
    pub color: [u8; 3],
}

/// The labels of block `which`'s lanes, top first.
pub fn lane_tags(state: &State, which: Block) -> Vec<LaneTag> {
    let edited = shown(state, which).and_then(|s| s.edit).map(|e| e.0);
    block_lanes(state, which)
        .iter()
        .map(|&lane| LaneTag {
            lane,
            off: is_off(state, lane),
            edited: edited == Some(lane),
            color: lane_color(lane),
        })
        .collect()
}

/// What a block shows of the transition: its window on that track, each
/// lane's gain curve over it, and the edited bar.
struct Shown {
    window: (f64, f64),
    curves: Vec<(Lane, Vec<(f64, f32)>)>,
    edit: Option<(Lane, f64)>,
}

/// The transition as block `which` shows it: the one armed or mixing, else
/// the one set up at the two cues, else the fades of this track's side
/// over its own cue (one cue alone).
fn shown(state: &State, which: Block) -> Option<Shown> {
    use crate::core::transition::Status;
    use jodel_playback::stems::STEMS;
    use jodel_playback::transition::Side;
    let d = &state.transition.draft;
    // The playing track's seconds of the window, its bars, the side, this
    // track's time at the playing track's, and each stem's gains at a
    // share of the window.
    type Gains = Box<dyn Fn(f64) -> [f64; STEMS]>;
    type Here = Box<dyn Fn(f64) -> f64>;
    let (start, end, bars, side, here, gains): (f64, f64, usize, Side, Here, Gains) =
        match draft::shown_plan(state) {
            Some((plan, bars)) => {
                // The side by the track (after the hand-over the track
                // mixed in is the playing one, the one left slides away
                // above it).
                let index = lanes::index(state, which)?;
                let side = if index == plan.from {
                    Side::From
                } else if index == plan.into {
                    Side::Into
                } else {
                    return None;
                };
                let map = plan.time_map();
                let here: Here = match side {
                    Side::From => Box::new(|t| t),
                    Side::Into => Box::new(move |t| map.at(t)),
                };
                let (start, end) = (plan.start, plan.start + plan.length);
                let gains: Gains = Box::new(move |share| plan.gains(side, share));
                (start, end, bars, side, here, gains)
            }
            None if d.status == Status::Draft => {
                let (start, end, _) = draft::cue_window(state, which)?;
                let side = match which {
                    Block::Now => Side::From,
                    Block::Next => Side::Into,
                    Block::Prev => return None,
                };
                let bars = draft::draft_bars(state);
                let lanes = d.lanes(bars);
                let base = if side == Side::From { 0 } else { STEMS };
                let gains: Gains =
                    Box::new(move |share| std::array::from_fn(|s| lanes[base + s].gain(share)));
                (start, end, bars, side, Box::new(|t| t), gains)
            }
            None => return None,
        };
    let window = (here(start), here(end));
    let steps = 96;
    let curves = Lane::ALL
        .iter()
        .map(|&lane| {
            let points = (0..=steps)
                .map(|k| {
                    let share = f64::from(k) / f64::from(steps);
                    let t = start + (end - start) * share;
                    let g = gains(share);
                    let gain = match lane.stem_index() {
                        Some(i) => g[i],
                        None => g.iter().sum::<f64>() / g.len() as f64,
                    };
                    (here(t), gain as f32)
                })
                .collect();
            (lane, points)
        })
        .collect();
    let edit = d.edit.and_then(|e| {
        let mine = match side {
            Side::From => e.lane < STEMS,
            Side::Into => e.lane >= STEMS,
        };
        let bars = bars.max(1);
        let lane = Lane::ALL
            .iter()
            .copied()
            .find(|l| l.stem_index() == Some(e.lane % STEMS))?;
        mine.then(|| {
            (
                lane,
                here(start + (end - start) * e.bar as f64 / bars as f64),
            )
        })
    });
    Some(Shown {
        window,
        curves,
        edit,
    })
}

/// The bar numbers over a block's lanes, over `start..end` seconds.
#[derive(Clone, Debug, PartialEq)]
pub struct RulerPlot {
    /// The seconds shown.
    pub start: f64,
    /// Up to.
    pub end: f64,
    /// Each bar's first beat and its number: the whole track's, the
    /// numbers laid out from the first so they stand where they are as the
    /// view moves.
    pub bars: Vec<(f64, usize)>,
    /// The cursor's beat.
    pub cursor: Option<f64>,
    /// The next track's cue bar.
    pub cue: Option<f64>,
    /// It follows the playhead with the lanes under it
    /// ([`WavePlot::follow`]).
    pub follow: Option<f64>,
}

impl RulerPlot {
    /// Where `time` falls, in units of `width` across the span.
    pub fn x_of(&self, time: f64, width: f64) -> f64 {
        (time - self.start) / (self.end - self.start).max(1e-9) * width
    }
}

/// What a block's lanes draw: the whole track, the bar numbers and each
/// lane ([`block_lanes`]' order).
#[derive(Clone, Debug)]
pub struct Plots {
    /// The whole track, the span shown lit.
    pub overview: WavePlot,
    /// The bar numbers.
    pub ruler: RulerPlot,
    /// The lanes, top first.
    pub lanes: Vec<WavePlot>,
}

/// What block `which` draws with `heard` seconds of the playing track
/// heard (none before its track's waveform is in); `follow`: the playing
/// track's block may follow the playhead (not while its top is out of
/// view).
pub fn plots(state: &State, which: Block, heard: f64, follow: bool) -> Option<Plots> {
    let r = lanes::rhythmed(state, which)?;
    let waveform = r.waveform.clone()?;
    let a = &r.analysis;
    let focused = state.transition.focus == which;
    let (start, end) = viewport::window(state, which, heard);
    let beat_time = |i: Option<usize>| i.and_then(|i| a.beats.get(i).copied());
    let block = tab::block(state, which);
    let cursor = focused
        .then(|| beat_time(viewport::cursor(state, which, heard)))
        .flatten();
    let cue = beat_time(block.cue.map(|c| c.beat));
    let selection = viewport::selection(state, which, heard);
    let playhead = viewport::playhead(state, which, heard);
    let duration = waveform.duration().max(a.duration).max(1e-3);
    let shown = shown(state, which);
    let transition = shown.as_ref().map(|s| s.window);
    let plot = |lane: Lane| WavePlot {
        waveform: Arc::clone(&waveform),
        lane,
        start,
        end,
        color: palette::DIM,
        downbeats: Vec::new(),
        unclear: Vec::new(),
        selection,
        shown: None,
        playhead,
        cursor,
        cue,
        transition,
        gain: Vec::new(),
        edit: None,
        lyrics: None,
        text_scale: 1,
        lines: WordLines::ONE,
        follow: None,
        deep: false,
        key: None,
    };
    let overview = WavePlot {
        start: 0.0,
        end: duration,
        shown: (!viewport::whole(state, which)).then_some((start, end)),
        cursor: None,
        ..plot(Lane::Song)
    };
    // The playing track's block following the playhead (not while blocks
    // slide): the window scrolls its ruler and lanes by itself. Every lane
    // carries the whole track's bars, so a view moved is the same lane
    // moved.
    let follow = (which == Block::Now
        && block.start.is_none()
        && viewport::slide(state).is_none()
        && follow)
        .then_some(viewport::LEAD);
    let bars = a
        .downbeats
        .iter()
        .enumerate()
        .filter_map(|(k, &d)| a.beats.get(d).map(|&t| (t, k + 1)))
        .collect();
    let ruler = RulerPlot {
        start,
        end,
        bars,
        cursor,
        cue,
        follow,
    };
    let downbeats = downbeats(a);
    let unclear = unclear(a);
    let lyrics = state
        .config
        .transition_lyrics
        .then(|| lanes::lyrics(state, which))
        .flatten();
    let lanes = block_lanes(state, which)
        .iter()
        .map(|&lane| {
            let curve = shown
                .as_ref()
                .and_then(|s| s.curves.iter().find(|c| c.0 == lane));
            WavePlot {
                color: if is_off(state, lane) {
                    OFF
                } else {
                    lane_color(lane)
                },
                downbeats: downbeats.clone(),
                unclear: unclear.clone(),
                gain: curve.map(|c| c.1.clone()).unwrap_or_default(),
                edit: shown
                    .as_ref()
                    .and_then(|s| s.edit)
                    .filter(|e| e.0 == lane)
                    .map(|e| e.1),
                lyrics: if lane == Lane::Stem(Stem::Vocals) {
                    lyrics.clone()
                } else {
                    None
                },
                follow,
                ..plot(lane)
            }
        })
        .collect();
    Some(Plots {
        overview,
        ruler,
        lanes,
    })
}

/// What a key does, for the footer: its key (as the keymap writes it),
/// then what.
pub type Hint = (String, &'static str);

/// The Transition tab's footer: the focused track's zoom and the main
/// keys, or, while a fade is edited, which and the keys that edit it; and
/// the button that zooms to fit the lyrics.
#[derive(Clone, Debug, PartialEq)]
pub struct Footer {
    /// The focused track's zoom (`16 bars`), or the fade edited (`now
    /// bass at bar 0`).
    pub zoom: String,
    /// A fade is being edited.
    pub editing: bool,
    /// The keys, in order.
    pub keys: Vec<Hint>,
}

/// The keys the footer names (from the keymap).
fn keys() -> Vec<Hint> {
    let key = |ops: &[Op], what| (keymap::keys(Context::Transition, ops), what);
    let jumps = format!(
        "{}-{}",
        keymap::hint(Context::Transition, Op::JumpBarAtEnd(-4)),
        keymap::hint(Context::Transition, Op::JumpBarAtEnd(4))
    );
    vec![
        key(&[Op::TransitionBarBack, Op::TransitionBarForward], "bars"),
        (jumps, "jump"),
        key(&[Op::Accept], "cue"),
        key(&[Op::TransitionPreset], "preset"),
        key(&[Op::TransitionPairing], "pairing"),
        key(&[Op::TransitionGo], "go"),
        key(&[Op::TransitionEdit], "edit"),
        key(&[Op::Help], "keys"),
    ]
}

/// The keys while a fade is edited (from the keymap).
fn edit_keys() -> Vec<Hint> {
    let key = |ops: &[Op], what| (keymap::keys(Context::Fades, ops), what);
    let gain = format!(
        "{}/{} {}",
        keymap::hint(Context::Fades, Op::EditRaise),
        keymap::hint(Context::Fades, Op::EditLower),
        keymap::keys(Context::Fades, &[Op::EditOff, Op::EditFull])
    );
    vec![
        key(&[Op::Up, Op::Down], "lane"),
        key(&[Op::EditBack, Op::EditForward], "bar"),
        (gain, "gain"),
        key(&[Op::EditCut], "cut"),
        key(&[Op::Back], "done"),
    ]
}

/// What the button that zooms to fit the lyrics says, and its key (from
/// the keymap).
pub fn fit_lyrics() -> Hint {
    (
        keymap::hint(Context::Transition, Op::TransitionFitLyrics),
        "fit lyrics",
    )
}

impl Footer {
    /// The footer for `state`.
    pub fn of(state: &State) -> Footer {
        if let Some(e) = state.transition.draft.edit {
            return Footer {
                zoom: format!(
                    "{} at bar {}",
                    crate::core::transition::lane_name(e.lane),
                    e.bar
                ),
                editing: true,
                keys: edit_keys(),
            };
        }
        // The focused track's zoom (its own, set by the wheel or `z`, else
        // the tab's).
        let zoom = match viewport::ZOOMS[viewport::zoom_of(state, state.transition.focus)] {
            0 => "whole tracks".to_owned(),
            1 => "1 bar".to_owned(),
            bars => format!("{bars} bars"),
        };
        Footer {
            zoom,
            editing: false,
            keys: keys(),
        }
    }
}

impl fmt::Display for Footer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.editing {
            write!(f, "editing ")?;
        }
        write!(f, "{}", self.zoom)?;
        for (key, what) in &self.keys {
            write!(f, " · {key} {what}")?;
        }
        Ok(())
    }
}

/// The tab's outline: each block that shows (the one left while it slides
/// away first), its header and its lanes' labels, then the footer.
pub struct Tab {
    /// The blocks' headers and their lanes' labels.
    pub blocks: Vec<(Header, Vec<LaneTag>)>,
    /// The footer.
    pub footer: Footer,
}

impl Tab {
    /// The tab for `state`, `heard` seconds of the playing track heard.
    pub fn of(state: &State, heard: f64) -> Tab {
        let sliding = viewport::slide(state).is_some();
        let blocks = [Block::Prev, Block::Now, Block::Next]
            .into_iter()
            .filter(|&b| b != Block::Prev || sliding)
            .map(|b| (Header::of(state, b, heard), lane_tags(state, b)))
            .collect();
        Tab {
            blocks,
            footer: Footer::of(state),
        }
    }
}

impl fmt::Display for Tab {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (header, tags) in &self.blocks {
            writeln!(f, "[block] {header}")?;
            let names: Vec<String> = tags
                .iter()
                .map(|t| {
                    let name = t.lane.label();
                    match (t.edited, t.off) {
                        (true, _) => format!("[{name}]"),
                        (false, true) => format!("{name} (off)"),
                        (false, false) => name.to_owned(),
                    }
                })
                .collect();
            writeln!(f, "[lanes] {}", names.join(" "))?;
        }
        writeln!(f, "[footer] {}", self.footer)?;
        let (key, what) = fit_lyrics();
        writeln!(f, "[button] {key} {what}")
    }
}
