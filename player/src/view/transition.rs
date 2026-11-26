//! The Transition tab: per block a header, the whole track in one row, the
//! bar numbers and the lanes (the whole song and each stem) over the span
//! shown, drawn as the lanes are ([`super::lanes`]). `L` (the
//! `transition_lyrics` setting) turns the lyrics on the voice's lane off.

use std::sync::Arc;

use jodel_playback::Stem;
use ratatui::layout::Rect;
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;
use ratatui::Frame;

use super::colors::{ACCENT, DIM, TEXT};
use super::lanes::{lane_color, WavePlot, OFF};
use super::palette;
use super::{Pictures, Scene};
use crate::core::lanes::{self, Block, Lane, WordLines};
use crate::core::transition::{self, draft, tab, viewport, Pairing};
use crate::core::{text, Rhythm, State};
use crate::view::layout::{TransitionBlockLayout, TransitionLayout};

/// Text on a coloured label.
const BG_TEXT: Color = Color::Rgb(20, 20, 24);

/// Whether a lane is a stem turned off (its strip on the desk muted).
fn is_off(state: &State, lane: Lane) -> bool {
    match lane {
        Lane::Song => false,
        Lane::Stem(stem) => crate::core::stems::is_off(state, stem),
    }
}

/// The seconds of the bars' first beats of `a` within `start..end`.
fn downbeats_within(a: &jodel_rhythm::Analysis, start: f64, end: f64) -> Vec<f64> {
    a.downbeats
        .iter()
        .filter_map(|&d| a.beats.get(d).copied())
        .filter(|&t| t >= start && t < end)
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

/// The header line of a block.
fn header(state: &State, which: Block, focused: bool, heard: f64) -> Line<'static> {
    let mark = if focused { "▸ " } else { "  " };
    let name = match which {
        Block::Now => "Now",
        Block::Next => "Next",
        Block::Prev => "Last",
    };
    let style = if focused {
        Style::new().fg(ACCENT).add_modifier(Modifier::BOLD)
    } else {
        Style::new().fg(DIM)
    };
    let mut spans = vec![Span::styled(format!("{mark}{name:<5}"), style)];
    let Some(path) = lanes::path(state, which) else {
        let what = match which {
            Block::Now => "nothing playing",
            Block::Next => "nothing after it in the queue",
            Block::Prev => "",
        };
        spans.push(Span::styled(what, Style::new().fg(DIM)));
        return Line::from(spans);
    };
    spans.push(Span::styled(
        crate::core::shown_title(state, path),
        Style::new().fg(TEXT),
    ));
    let status = if !state.config.bpm {
        " · tempo is off (M)".to_owned()
    } else {
        match state.rhythm.get(path) {
            None | Some(Rhythm::Pending) => " · analysing…".to_owned(),
            Some(Rhythm::Failed) => " · could not be analysed".to_owned(),
            Some(Rhythm::Ready(r)) => {
                let a = &r.analysis;
                let mut s = String::new();
                if let Some(sum) = a.summary {
                    s.push_str(&format!(" · {} BPM", text::bpm_text(sum.bpm)));
                }
                if let Some(key) = a.key {
                    s.push_str(&format!(" · {}", text::key_text(&key)));
                }
                let bars = a.downbeats.len();
                if let Some(c) = viewport::cursor(state, which, heard) {
                    s.push_str(&format!(" · bar {} / {bars}", viewport::bar_of(a, c)));
                }
                if r.waveform.as_ref().is_some_and(|w| w.stems.is_none()) {
                    s.push_str(" · no stems");
                }
                if r.fixed() {
                    s.push_str(" · grid fixed by hand");
                }
                if which == Block::Now {
                    s.push_str(&transition_text(state));
                }
                s
            }
        }
    };
    spans.push(Span::styled(status, Style::new().fg(DIM)));
    Line::from(spans)
}

/// The transition set up, in a few words.
fn transition_text(state: &State) -> String {
    use crate::core::transition::Status;
    let d = &state.transition.draft;
    // The preset of the one sent (a skip sends its own), else the one set.
    let preset = d.in_force().map_or(d.preset, |sent| sent.preset);
    let mut s = format!(
        " · transition: {}, {}",
        preset.name(),
        transition::tempo_name(d.tempo)
    );
    let bars = draft::draft_bars(state);
    if d.lanes.as_ref().is_some_and(|l| *l != d.preset.lanes(bars)) {
        s.push_str(" (edited)");
    }
    let cued = |b: Block| tab::block(state, b).cue.is_some();
    match &d.status {
        Status::Draft if cued(Block::Now) && cued(Block::Next) => s.push_str(", g arms"),
        Status::Draft => s.push_str(", enter sets the cues"),
        Status::Armed => s.push_str(", armed"),
        Status::Mixing(_) => s.push_str(", mixing"),
    }
    // The tempi of the beats that are paired, where they meet.
    if let (Some(a), Some(b)) = draft::meeting_tempi(state) {
        // The next track's tempo in the playing track's beats, by the
        // pairing (2:1: its beat is two of these), and the glide to it.
        let per = draft::pairing(state);
        let to = b * per.ratio();
        // Set by hand: what the transition set up (or armed) takes; one
        // already mixing plays what it was sent with.
        let set = d.pairing.is_some() && !matches!(d.status, Status::Mixing(_));
        s.push_str(" · next ");
        if per != Pairing::One || set {
            s.push_str(per.label());
            s.push_str(if set { " (set) " } else { " " });
        }
        s.push_str(&format!("{:+.1} %", 100.0 * (to / a - 1.0)));
        if (to / a - 1.0).abs() >= 0.002 {
            // The glide goes as far as the engine takes it; the next track
            // makes up the rest after the window.
            let most = jodel_playback::warp::GLIDE_MOST;
            let reach = a * (to / a).clamp(1.0 / most, most);
            s.push_str(&format!(" · tempo {a:.0} → {reach:.0} across the window"));
            if (reach - to).abs() >= 0.5 {
                s.push_str(&format!(", {to:.0} after it"));
            }
        }
    }
    s
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
    /// Where it goes (one row).
    pub area: Rect,
    /// The seconds shown.
    pub start: f64,
    /// Up to.
    pub end: f64,
    /// Each bar's first beat in the span, and its number.
    pub bars: Vec<(f64, usize)>,
    /// The cursor's beat.
    pub cursor: Option<f64>,
    /// The next track's cue bar.
    pub cue: Option<f64>,
    /// It follows the playhead with the lanes under it
    /// ([`WavePlot::follow`]); its bars are then the whole track's.
    pub follow: Option<f64>,
}

impl RulerPlot {
    /// Where `time` falls, in units of `width` across the span.
    pub fn x_of(&self, time: f64, width: f64) -> f64 {
        (time - self.start) / (self.end - self.start).max(1e-9) * width
    }
}

/// One block: its header, overview, ruler and lanes.
fn draw_block(
    frame: &mut Frame,
    scene: &Scene,
    b: &TransitionBlockLayout,
    pictures: &mut dyn Pictures,
) {
    let state = scene.state;
    let heard = scene.heard;
    let focused = state.transition.focus == b.block;
    frame.render_widget(
        Paragraph::new(header(state, b.block, focused, heard)),
        b.header,
    );
    for (lane, label, _) in &b.lanes {
        frame.render_widget(
            Paragraph::new(Span::styled(
                lane.label(),
                Style::new().fg(if is_off(state, *lane) {
                    DIM
                } else {
                    Color::from(lane_color(*lane))
                }),
            )),
            *label,
        );
    }
    let Some(r) = lanes::rhythmed(state, b.block) else {
        return;
    };
    let Some(waveform) = r.waveform.clone() else {
        return;
    };
    let a = &r.analysis;
    let (start, end) = viewport::window(state, b.block, heard);
    let beat_time = |i: Option<usize>| i.and_then(|i| a.beats.get(i).copied());
    let block = tab::block(state, b.block);
    let cursor = focused
        .then(|| beat_time(viewport::cursor(state, b.block, heard)))
        .flatten();
    let cue = beat_time(block.cue.map(|c| c.beat));
    let selection = viewport::selection(state, b.block, heard);
    let playhead = viewport::playhead(state, b.block, heard);
    let duration = waveform.duration().max(a.duration).max(1e-3);
    let shown = shown(state, b.block);
    let transition = shown.as_ref().map(|s| s.window);
    pictures.wave(
        frame,
        &WavePlot {
            area: b.overview,
            waveform: Arc::clone(&waveform),
            lane: Lane::Song,
            start: 0.0,
            end: duration,
            color: palette::DIM,
            downbeats: Vec::new(),
            unclear: Vec::new(),
            selection,
            shown: (!viewport::whole(state, b.block)).then_some((start, end)),
            playhead,
            cursor: None,
            cue,
            transition,
            gain: Vec::new(),
            edit: None,
            rows: (0, 1),
            lyrics: None,
            text_scale: 1,
            lines: WordLines::ONE,
            follow: None,
            deep: false,
            key: None,
        },
    );
    // The playing track's block following the playhead (not while blocks
    // slide): a frontend may scroll its ruler and lanes by itself, so they
    // carry the whole track's bars.
    let follow = (b.block == Block::Now
        && block.start.is_none()
        && viewport::slide(state).is_none()
        && b.top_cut == 0)
        .then_some(viewport::LEAD);
    let bars = a
        .downbeats
        .iter()
        .enumerate()
        .filter_map(|(k, &d)| a.beats.get(d).map(|&t| (t, k + 1)))
        .filter(|&(t, _)| follow.is_some() || (t >= start && t < end))
        .collect();
    pictures.ruler(
        frame,
        &RulerPlot {
            area: b.ruler,
            start,
            end,
            bars,
            cursor,
            cue,
            follow,
        },
    );
    let downbeats = if follow.is_some() {
        downbeats_within(a, f64::NEG_INFINITY, f64::INFINITY)
    } else {
        downbeats_within(a, start, end)
    };
    let unclear = unclear(a);
    let lyrics = state
        .config
        .transition_lyrics
        .then(|| lanes::lyrics(state, b.block))
        .flatten();
    for (i, (lane, label, area)) in b.lanes.iter().enumerate() {
        let skip = if i == 0 { b.top_cut } else { 0 };
        let color = if is_off(state, *lane) {
            OFF
        } else {
            lane_color(*lane)
        };
        let gain = shown
            .as_ref()
            .and_then(|s| s.curves.iter().find(|c| c.0 == *lane))
            .map(|c| c.1.clone())
            .unwrap_or_default();
        let edit = shown
            .as_ref()
            .and_then(|s| s.edit)
            .filter(|e| e.0 == *lane)
            .map(|e| e.1);
        if edit.is_some() {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    lane.label(),
                    Style::new().fg(BG_TEXT).bg(Color::from(lane_color(*lane))),
                )),
                *label,
            );
        }
        pictures.wave(
            frame,
            &WavePlot {
                area: *area,
                waveform: Arc::clone(&waveform),
                lane: *lane,
                start,
                end,
                color,
                downbeats: downbeats.clone(),
                unclear: unclear.clone(),
                selection,
                shown: None,
                playhead,
                cursor,
                cue,
                transition,
                gain,
                edit,
                rows: (skip, b.lane_rows.max(area.height)),
                lyrics: if *lane == Lane::Stem(Stem::Vocals) {
                    lyrics.clone()
                } else {
                    None
                },
                text_scale: 1,
                lines: WordLines::ONE,
                follow,
                deep: false,
                key: None,
            },
        );
    }
}

/// The Transition tab.
pub fn draw(frame: &mut Frame, scene: &Scene, m: &TransitionLayout, pictures: &mut dyn Pictures) {
    for b in &m.blocks {
        draw_block(frame, scene, b, pictures);
    }
}

/// The Transition tab's footer: the focused track's zoom and the main keys.
pub fn footer(state: &State) -> String {
    // The focused track's zoom (its own, set by the wheel or `z`, else the
    // tab's).
    let zoom = match viewport::ZOOMS[viewport::zoom_of(state, state.transition.focus)] {
        0 => "whole tracks".to_owned(),
        1 => "1 bar".to_owned(),
        bars => format!("{bars} bars"),
    };
    if let Some(e) = state.transition.draft.edit {
        return format!(
            "editing {} at bar {} · ↑↓ lane · ←→ bar · +/- 0 1 gain · c cut · esc done",
            crate::core::transition::lane_name(e.lane),
            e.bar
        );
    }
    format!(
        "{zoom} · ←→ bars · 1-9 jump · enter cue · p preset · r pairing · g go · e edit · ? keys"
    )
}
