//! Karaoke mode's stage, made frame by frame for the GPU
//! ([`gpu`]): the song's words as tubes through their notes
//! on a lane that runs by a playhead (by time: [`AHEAD`] seconds of the
//! song ahead of it), their text riding in them, filled where the singer
//! hit them ([`words`]); the shots rescuing a word missed ([`shots`]); the
//! singer as an arrow at the playhead, gliding to their pitch, its trail
//! behind it and sparks flying off it ([`sparks`]); the streak's
//! multiplier over the playhead; the line sung and the next at the bottom;
//! and the song's progress, its breaks apart, along the bottom edge;
//! "AUTO" in a corner while the demo sings ([`tag`]). The tiers' effects
//! draw at their points of it ([`super::fx`]: the rainbow behind the
//! tubes, the bursts from the playhead, the light round the lane, the
//! screen passes over it all, a word slammed down over everything, the
//! screen shaken with it), as strong as the board eases them from the
//! singing and the music. Its text is in Karaoke mode's own faces
//! ([`Face`]), its colours its own ([`palette`]). Where it draws no lane (the notes
//! off, a song without notes), the lyrics are set in the middle instead
//! ([`lyrics`]); with the lane, only the lines that are no lyric (the
//! song's title as it starts, a note in a pause). Under it all the
//! count-in ([`count`]) and the score ([`score`]). While a song's result
//! shows, the stage is the result screen instead ([`result`]). The stage
//! is all of Karaoke mode the window draws over its backdrop, but the
//! controls' bar.
//!
//! What moves by itself (the arrow, the sparks, the multiplier's pop) is
//! stepped by the time between frames, so it moves the same at any frame
//! rate.

use std::collections::HashMap;

use super::big_text::{Face, Faces, COVER_PAD};
use super::fx::{self, At};

mod arrow;
mod atlas;
mod badge;
mod comet;
pub mod count;
mod effects;
pub mod gpu;
pub(crate) mod lines;
pub mod lyrics;
mod meter;
mod next;
pub mod palette;
mod result;
mod scene;
pub mod score;
mod shade;
pub mod shapes;
mod shots;
mod sparks;
mod tag;
mod words;
use self::arrow::Arrow;
use self::atlas::Atlas;
use self::gpu::{Instance, Shape, StageFrame};
use self::palette::{rgba, Rgb, ACTIVE, CORAL, GOLD, GOLD_DEEP, GOLD_TOP, SMOKE, WHITE};
use self::shade::Shade;
pub use result::Tint;

use crate::core::State;
use crate::view::karaoke::KaraokePicture;

/// Where the playhead is, a share of the width from the left.
pub const HEAD: f32 = 0.28;
/// The seconds of song shown ahead of the playhead.
pub const AHEAD: f64 = 4.0;
/// The lane's band: its top and bottom, shares of the height.
const BAND: (f32, f32) = (0.18, 0.56);
/// A pill's height, a share of the window's height.
const PILL: f32 = 0.052;
/// The air round the HUD's band ([`Lane::hud`]), a share of the height.
const AIR: f32 = 0.008;
/// The HUD's band at its tallest ([`Lane::hud`]), a share of the height.
const HUD: f32 = 0.0495;
/// The progress bar: its middle, a share of the height.
const PROGRESS: f32 = 0.935;
/// A gap between notes this long or longer is a break on the progress bar.
const BREAK: f64 = 4.0;
/// How soon after its last pitch the arrow fades, in seconds.
const ARROW_HOLD: f64 = 0.25;

/// Karaoke mode's stage between frames.
#[derive(Default)]
pub struct Stage {
    /// Its text, by face.
    faces: Faces,
    atlas: Atlas,
    arrow: Arrow,
    /// The arrow's comet tail.
    comet: comet::Comet,
    /// The sparks off the arrow ([`sparks`]).
    sparks: sparks::Sparks,
    /// The streak: its shine, the words scored ([`super::streak`]).
    streak: super::streak::Streak,
    /// Where it is in the singing's moments.
    cursor: crate::core::singing::Cursor,
    /// The multiplier's badge ([`badge`]).
    badge: badge::Badge,
    /// Where the capitals' middle is in text of a face and size
    /// ([`Stage::cap_middle`]).
    caps: HashMap<(Face, u32), f32>,
    /// The song's time at the frame before: what moves goes by the song.
    before: Option<f64>,
    /// The words scored bursting, handed to the GPU ([`super::particles`]).
    sprays: Vec<super::particles::Spray>,
    /// The shine's meter ([`meter`]).
    meter: meter::Meter,
    /// The effects' strengths and what they keep ([`fx::Board`]).
    board: fx::Board,
    /// How far the effects shake the screen this frame, all of them.
    shake: [f32; 2],
    /// The bursts that are light (the angels, the jackpots) showing.
    lights: Vec<super::particles::Spray>,
    /// What the result screen keeps between frames ([`result::Kept`]).
    kept: result::Kept,
    /// What takes clicks as drawn last ([`Stage::targets`]).
    targets: Vec<([f32; 4], crate::core::keymap::Op)>,
    /// The score's counter ([`score`]).
    odometer: score::Odometer,
    /// The bursts a step up throws from the bar under the word sung,
    /// while the lyrics are set without the lane ([`lyrics`]).
    ups: Vec<super::particles::Spray>,
    /// The theme it was last drawn in ([`Stage::scene`]: the song's).
    theme: &'static crate::theme::Theme,
}

/// What the stage is made from, besides the picture.
pub struct Song<'a> {
    /// The singing, scored.
    pub singing: &'a crate::core::singing::Singing,
    /// The song's sung lines (those with words), as its notes are of.
    pub sung: Vec<&'a jodel_lyrics::Line>,
    /// Whether the game is on.
    pub game: bool,
    /// How long the song is, in seconds, where known.
    pub length: Option<f64>,
    /// How late the voice comes, in seconds: the singing's time is the song's less it.
    pub late: f64,
    /// Whether the demo key is held ([`crate::core::singing::Demo`]).
    pub demo: bool,
    /// The song as it is heard, once analysed (the effects' music).
    pub heard: Option<&'a super::backdrop::music::Song>,
    /// The game's theme ([`crate::theme`]).
    pub theme: &'static crate::theme::Theme,
}

impl<'a> Song<'a> {
    /// The song in Karaoke mode in `state`, `heard` as it is heard (once
    /// analysed).
    pub fn of(state: &'a State, heard: Option<&'a super::backdrop::music::Song>) -> Song<'a> {
        let sung = crate::core::karaoke::song_index(state)
            .and_then(|i| crate::core::karaoke::lyrics_of(state, i))
            .and_then(|l| match l {
                crate::core::karaoke::SongLyrics::Found { shown, .. } => {
                    Some(shown.lines.iter().filter(|l| !l.words.is_empty()).collect())
                }
                _ => None,
            })
            .unwrap_or_default();
        Song {
            singing: &state.karaoke.singing,
            sung,
            game: state.config.karaoke_game,
            length: state
                .playing
                .track
                .as_ref()
                .and_then(|t| t.info.duration())
                .map(|d| d.as_secs_f64())
                .filter(|&d| d > 0.0),
            late: state.karaoke.lag.late,
            demo: state.karaoke.demo.held(),
            heard,
            theme: state.theme(),
        }
    }

    /// The song of `singing` (its `sung` lines, `length` long) with the
    /// game on, the voice taken as it comes (no delay) and the classic
    /// theme: for a stage made outside the player.
    pub fn new(
        singing: &'a crate::core::singing::Singing,
        sung: Vec<&'a jodel_lyrics::Line>,
        length: Option<f64>,
    ) -> Song<'a> {
        Song {
            singing,
            sung,
            game: true,
            length,
            late: 0.0,
            demo: false,
            heard: None,
            theme: &crate::theme::CLASSIC,
        }
    }
}

/// Whether the stage draws the song's lane for `state`: Karaoke mode, its
/// notes shown, the song in it with notes made; or a song's result shown
/// in it (no lane then: the result screen). Otherwise it sets all the
/// lyrics in the middle ([`lyrics`]).
pub fn wanted(state: &State) -> bool {
    if !crate::core::karaoke::in_mode(state) {
        return false;
    }
    if crate::core::karaoke::result_shown(state).is_some() {
        return true;
    }
    if !state.config.karaoke_notes {
        return false;
    }
    let Some(index) = crate::core::karaoke::song_index(state) else {
        return false;
    };
    let singing = &state.karaoke.singing;
    state
        .queue
        .paths
        .get(index)
        .is_some_and(|p| singing.is_of(p) && singing.words() > 0)
}

/// The shapes of `out` from `from` on kept to `rect` (left, top, right,
/// bottom): their quads cut to it, a run of text's atlas texels with its
/// quad; those left with nothing gone.
fn clip(out: &mut Vec<Instance>, from: usize, [l, t, r, b]: [f32; 4]) {
    let mut k = from;
    while k < out.len() {
        let i = &mut out[k];
        let q = i.quad;
        let cut = [q[0].max(l), q[1].max(t), q[2].min(r), q[3].min(b)];
        if cut[2] <= cut[0] || cut[3] <= cut[1] {
            out.remove(k);
            continue;
        }
        if i.is(Shape::Text) {
            let along = |v: f32, (a, b): (f32, f32), (ua, ub): (f32, f32)| {
                ua + (v - a) / (b - a).max(1e-6) * (ub - ua)
            };
            let (across, down) = ((q[0], q[2]), (q[1], q[3]));
            let (u, v) = ((i.uv[0], i.uv[2]), (i.uv[1], i.uv[3]));
            i.uv = [
                along(cut[0], across, u),
                along(cut[1], down, v),
                along(cut[2], across, u),
                along(cut[3], down, v),
            ];
        }
        i.quad = cut;
        k += 1;
    }
}

/// The lane's geometry in a window: the playhead, pixels a second, the
/// band, a pill's height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lane {
    /// The playhead, pixels across.
    pub head: f32,
    /// Pixels a second.
    pub pps: f32,
    /// The band's top and bottom.
    pub band: (f32, f32),
    /// A pill's height.
    pub pill: f32,
    /// The time at the playhead.
    pub now: f64,
}

impl Lane {
    /// For a window `(x, y, w, h)` at `now` seconds into the song.
    pub fn new((x, y, w, h): (f32, f32, f32, f32), now: f64) -> Lane {
        let head = x + w * HEAD;
        Lane {
            head,
            pps: (x + w * 0.975 - head) / AHEAD as f32,
            band: (y + h * BAND.0, y + h * BAND.1),
            pill: (h * PILL).clamp(16.0, 90.0),
            now,
        }
    }

    /// Where `t` seconds into the song is across.
    pub fn x(&self, t: f64) -> f32 {
        self.head + ((t - self.now) as f32) * self.pps
    }

    /// Where a note `height` high (-1 to 1) is down: its middle.
    pub fn y(&self, height: f32) -> f32 {
        let mid = (self.band.0 + self.band.1) / 2.0;
        let span = (self.band.1 - self.band.0) / 2.0 - self.pill / 2.0;
        mid - height.clamp(-1.0, 1.0) * span
    }

    /// The seconds of song behind the playhead that show.
    pub fn behind(&self) -> f64 {
        f64::from((self.head / self.pps).max(0.0))
    }

    /// The band over the lane the HUD keeps to in the window's `area` (the
    /// multiplier's badge, the meter and its verdict, the score's corner),
    /// its top and bottom: from a little under the stage's top, [`HUD`] of
    /// the height down, never past a little over the highest the marquee
    /// bulbs round the glass light ([`super::fx::bulbs::lit_top`]), whether
    /// they show or not, so nothing moves as they come and go (the air
    /// left between it and their light is the marquee's). What is in it is
    /// made smaller where it is lower than that.
    pub fn hud(&self, area: (f32, f32, f32, f32)) -> (f32, f32) {
        let air = area.3 * AIR;
        let top = area.1 + air;
        let bulbs = super::fx::bulbs::lit_top(self.glass(area).rect(), super::fx::far_corner(area));
        (top, (bulbs - air).min(top + area.3 * HUD).max(top))
    }

    /// The lane's glass in the window's `area` (round the band, a little
    /// more than a pill over).
    pub fn glass(&self, (x, _, w, _): (f32, f32, f32, f32)) -> LaneBox {
        let middle = (x + w * 0.5, (self.band.0 + self.band.1) / 2.0);
        let half_high = (self.band.1 - self.band.0) / 2.0 + self.pill * 0.55;
        LaneBox {
            middle,
            half: (w * 0.4875, half_high),
        }
    }
}

/// The lane's glass: its middle and half its width and height.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LaneBox {
    /// Its middle.
    pub middle: (f32, f32),
    /// Half its width and height.
    pub half: (f32, f32),
}

impl LaneBox {
    /// Its left, top, right and bottom.
    pub fn rect(&self) -> [f32; 4] {
        let ((cx, cy), (hx, hy)) = (self.middle, self.half);
        [cx - hx, cy - hy, cx + hx, cy + hy]
    }
}

fn pill(
    c: (f32, f32),
    half: (f32, f32),
    body: [f32; 4],
    edge: [f32; 4],
    glow: f32,
    outline: f32,
    radius: f32,
) -> Instance {
    let m = glow + 1.0;
    let quad = [
        c.0 - half.0 - m,
        c.1 - half.1 - m,
        c.0 + half.0 + m,
        c.1 + half.1 + m,
    ];
    Instance::of(Shape::Pill, quad, [c.0, c.1, half.0, half.1])
        .colors(body, edge)
        .params([glow, outline, radius])
}

/// The part of the pill at `c`, `half` between `span` (pixels across),
/// filled: cut square at the span's ends, its glow fading past them; none
/// where they do not meet.
fn fill(
    c: (f32, f32),
    half: (f32, f32),
    span: (f32, f32),
    top: [f32; 4],
    bottom: [f32; 4],
    glow: f32,
) -> Option<Instance> {
    let x0 = (c.0 - half.0).max(span.0);
    let x1 = (c.0 + half.0).min(span.1);
    if x1 - x0 < 0.5 {
        return None;
    }
    let m = glow + 1.0;
    let quad = [x0 - m, c.1 - half.1 - m, x1 + m, c.1 + half.1 + m];
    let fill = Instance::of(Shape::Fill, quad, [c.0, c.1, half.0, half.1]);
    Some(
        fill.colors(top, bottom)
            .params([glow, 0.0, 0.0])
            .uv([x0, x1, 0.0, 0.0]),
    )
}

/// How far in from each edge of the window nothing shows, a share of its
/// width ([`edge_fade`]).
const EDGE: f32 = 0.012;

/// How much shows at `x` across the window `(x0, w)`: fading out at its
/// edges.
fn edge_fade(x: f32, (x0, w): (f32, f32)) -> f32 {
    let fade = w * 0.05;
    let left = ((x - (x0 + w * EDGE)) / fade).clamp(0.0, 1.0);
    let right = ((x0 + w * (1.0 - EDGE) - x) / fade).clamp(0.0, 1.0);
    left.min(right)
}

fn segment(
    a: (f32, f32),
    b: (f32, f32),
    thick: f32,
    core: [f32; 4],
    glow_color: [f32; 4],
    glow: f32,
) -> Instance {
    let m = thick + glow + 1.0;
    let quad = [
        a.0.min(b.0) - m,
        a.1.min(b.1) - m,
        a.0.max(b.0) + m,
        a.1.max(b.1) + m,
    ];
    Instance::of(Shape::Segment, quad, [a.0, a.1, b.0, b.1])
        .colors(core, glow_color)
        .params([thick, glow, 0.0])
}

fn disc(c: (f32, f32), r: f32, core: [f32; 4], glow_color: [f32; 4], glow: f32) -> Instance {
    let m = r + glow + 1.0;
    let quad = [c.0 - m, c.1 - m, c.0 + m, c.1 + m];
    Instance::of(Shape::Disc, quad, [c.0, c.1, r, glow]).colors(core, glow_color)
}

impl Stage {
    /// The width of `text` in `face` at `px`.
    fn measure(&mut self, face: Face, text: &str, px: f32) -> f32 {
        self.faces.get(face).measure(text, px)
    }

    /// A run of text: `text` in `face` at `px` (drawn at `scale` of that)
    /// with its line's top-left at `at`; left of `split` (pixels across)
    /// in `filled`, right of it in `rest`. What it put into `out`, if any.
    #[allow(clippy::too_many_arguments)]
    fn text<'o>(
        &mut self,
        out: &'o mut Vec<Instance>,
        face: Face,
        text: &str,
        (px, scale): (f32, f32),
        at: (f32, f32),
        split: f32,
        (rest, filled): ([f32; 4], [f32; 4]),
    ) -> Option<&'o mut Instance> {
        if text.trim().is_empty() {
            return None;
        }
        let placed = self.atlas.place(self.faces.get(face), text, px)?;
        let (u, v, w, h) = placed.rect;
        let scale = scale / placed.made;
        let pad = COVER_PAD as f32 * scale;
        let (x0, y0) = (at.0 - pad, at.1 - pad);
        let quad = [x0, y0, x0 + w as f32 * scale, y0 + h as f32 * scale];
        let texels = [u as f32, v as f32, (u + w) as f32, (v + h) as f32];
        let run = Instance::of(Shape::Text, quad, [0.0; 4]).colors(rest, filled);
        out.push(run.params([split, 0.0, 0.0]).uv(texels));
        out.last_mut()
    }

    /// [`Stage::text`] over the lyrics' dark outline ([`Shade::OUTLINE`],
    /// as opaque as its more opaque colour), so it reads over any cover
    /// and any look.
    #[allow(clippy::too_many_arguments)]
    fn outlined_text<'o>(
        &mut self,
        out: &'o mut Vec<Instance>,
        face: Face,
        text: &str,
        (px, scale): (f32, f32),
        at: (f32, f32),
        split: f32,
        (rest, filled): ([f32; 4], [f32; 4]),
    ) -> Option<&'o mut Instance> {
        let rim = (Shade::OUTLINE, rest[3].max(filled[3]));
        self.shade(out, (face, text), (px, scale), (at, f32::INFINITY), rim);
        self.text(out, face, text, (px, scale), at, split, (rest, filled))
    }

    /// How far down from the top of text in `face` made at `px` for the
    /// atlas ([`super::big_text::BigText::coverage`]) the middle of its capitals is, between
    /// their top and the baseline: kept by face and size.
    fn cap_middle(&mut self, face: Face, px: f32) -> f32 {
        let key = (face, (px * 4.0).round() as u32);
        if let Some(&m) = self.caps.get(&key) {
            return m;
        }
        let (w, h, coverage) = self.faces.get(face).coverage("H", px);
        let rows: Vec<usize> = (0..h)
            .filter(|&r| coverage[r * w..(r + 1) * w].iter().any(|&a| a > 64))
            .collect();
        let m = match (rows.first(), rows.last()) {
            (Some(&a), Some(&b)) => (a + b + 1) as f32 / 2.0,
            _ => COVER_PAD as f32 + px * 0.6,
        };
        self.caps.insert(key, m);
        m
    }

    /// The stage for `picture` (all its lines, as the view made it) in the
    /// window's `area` (pixels), at `now` seconds into the song, `dt`
    /// seconds after the last frame.
    pub fn frame(
        &mut self,
        song: &Song,
        picture: &KaraokePicture,
        area: (f32, f32, f32, f32),
        now: f64,
        dt: f32,
    ) -> StageFrame {
        let dt = dt.clamp(0.0, 0.1);
        let clock = crate::core::lag::sung_clock(now, song.late);
        let before = self.before.replace(now);
        let song_dt = before.map_or(0.0, |b| (now - b).clamp(0.0, 0.1)) as f32;
        let mut out: Vec<Instance> = Vec::new();
        let lane = Lane::new(area, now);
        let (x, _, w, _) = area;
        let singing = song.singing;
        let game = song.game;
        let p = lane.pill;
        let behind = lane.behind();
        let (from, to) = (now - behind - 0.5, now + AHEAD + 0.5);

        // The lane: a band of smoked glass (the bubbles read over any
        // cover), the playhead across it.
        let lane_box = lane.glass(area);
        out.push(pill(
            lane_box.middle,
            lane_box.half,
            rgba(SMOKE, 0.34),
            rgba(WHITE, 0.09),
            0.0,
            1.0,
            p * 0.7,
        ));
        let (top, bottom) = (lane.band.0 - p * 0.45, lane.band.1 + p * 0.45);
        out.push(segment(
            (lane.head, top),
            (lane.head, bottom),
            1.0,
            rgba(WHITE, 0.55),
            rgba(WHITE, 0.18),
            p * 0.25,
        ));
        // The effects as strong as the singing and the music hold them;
        // those behind the tubes (the streak's rainbow bands).
        let since = singing.moments_since(&mut self.cursor);
        let step = self.streak.step(singing, game, &since, clock);
        let music = fx::music::read(&fx::music::Heard {
            song: song.heard,
            singing,
            now,
            clock,
            length: song.length,
        });
        let effects = self.board.frame(song.theme, &step, &music, clock);
        self.shake = [0.0; 2];
        // Over everything, lettered before any other text of the frame: the
        // effects' big words and their light.
        let over_at = self.over_at(area, &lane);
        let (over, over_light) = self.over(&over_at, &effects);
        let under = At::Under {
            head: (lane.head, lane_box.middle.1),
            lane: lane_box,
        };
        self.effects_at(&under, &effects, (&mut out, &mut Vec::new()));

        // The words, each a tube through its notes, its bubble filling as it
        // was hit; the shots rescuing those missed.
        let mut paths = Vec::new();
        let words = self.lay_words(song, &lane, (from, to));
        let mut sprays = Vec::new();
        let labels = self.draw_words(
            song,
            (&lane, (x, w)),
            &words,
            (&mut out, &mut paths),
            &mut sprays,
            (now, clock),
        );
        let mut shots = shots::shots(song, &lane, &words, clock);
        out.append(&mut shots.out);
        sprays.append(&mut shots.sprays);
        let y_of = |h: &crate::core::singing::Heard| words.y_of(song, &lane, h);
        let recent: Vec<_> = singing
            .heard
            .iter()
            .rev()
            .take_while(|h| h.t >= clock - behind)
            .filter(|h| h.t <= clock)
            .collect();
        // The pitch heard last, where it is on the lane.
        let last_point = recent
            .iter()
            .find_map(|h| y_of(h).map(|y| ((lane.x(h.t), y), h.hit, h.t)));
        self.draw_labels(&mut out, &words, labels, now);

        // The singer: where they are now, a comet's tail behind them.
        let fresh = last_point.filter(|(_, _, t)| clock - t < ARROW_HOLD);
        self.arrow.step(fresh.map(|(p, _, _)| p.1), dt);
        let hitting = game && fresh.is_some_and(|f| f.1);
        let arrow_color = if hitting { GOLD } else { CORAL };
        let tip = (lane.head + p * 0.42, self.arrow.y);
        let at = (self.arrow.shown > 0.0).then_some((now, self.arrow.y));
        self.comet.step(at);
        self.comet
            .draw(&mut out, &lane, (arrow_color, p), self.arrow.shown);

        // Stardust along the tail, and sparks off the arrow on the note: owed
        // by the time gone by.
        let off = (self.arrow.shown > 0.5).then_some((lane.head, self.arrow.y));
        self.sparks.fly(off, arrow_color, p, hitting, song_dt);
        // A word scored (once): a burst where it hit; then the effects'
        // bursts from the playhead (each tier the streak went up, a rainbow
        // burst), the multiplier popping.
        let at = |end: f64| (lane.x(end).max(x), self.arrow.y);
        self.sprays
            .extend(step.word_spray(at, (p, clock, lane.pps)));
        let playhead = At::Playhead {
            tip,
            pill: p,
            pps: lane.pps,
            word: step.word.map(at),
            area,
        };
        let mut bursts = std::mem::take(&mut self.sprays);
        self.effects_at(&playhead, &effects, (&mut Vec::new(), &mut bursts));
        self.sprays = bursts;
        self.sparks.step(song_dt, lane.pps);
        self.sparks.draw(&mut out);
        if self.arrow.shown > 0.0 {
            let a = self.arrow.shown;
            out.push(disc(
                (lane.head, self.arrow.y),
                p * 0.11,
                rgba([1.0, 0.98, 0.92], a),
                rgba(arrow_color, 0.4 * a),
                p * 0.45,
            ));
            // Slim, and beating gently while on the note.
            let beat = if hitting {
                1.0 + 0.08 * (clock as f32 * 14.0).sin()
            } else {
                1.0
            };
            let len = p * 0.9 * beat;
            let quad = [
                tip.0 - len * 1.6,
                tip.1 - len * 1.2,
                tip.0 + len * 0.6,
                tip.1 + len * 1.2,
            ];
            let arrow = Instance::of(Shape::Arrow, quad, [tip.0, tip.1, len, 0.0]);
            let arrow = arrow.colors(rgba(arrow_color, a), rgba(arrow_color, 0.6 * a));
            out.push(arrow.params([p * 0.35, 0.0, 0.0]));
            let arrow = At::Arrow {
                at: (lane.head, self.arrow.y),
                radius: p * 0.9,
            };
            self.effects_at(&arrow, &effects, (&mut out, &mut Vec::new()));
        }

        // The multiplier, over the playhead ([`badge`]): over everything.
        let mut top = Vec::new();
        if song.demo {
            self.demo_tag(&mut top, area);
        }
        let mut multiplier = Vec::new();
        self.draw_badge(
            &mut multiplier,
            (singing, game),
            (&lane, area),
            dt,
            &effects,
        );

        // The line sung and the next ([`lines`]).
        self.lines(&mut out, picture, area, &effects);

        // The shine: the meter, the tiers' looks, the song's progress.
        let mut frame = StageFrame {
            instances: out,
            top,
            clock: clock as f32,
            ..StageFrame::default()
        };
        let at = meter::Where {
            area,
            lane,
            now,
            clock,
            gone: picture.controls.map_or(0.0, |fade| {
                crate::core::karaoke::controls::handover(fade).0 as f32
            }),
        };
        self.shine(
            &mut frame,
            (singing, game, song.length),
            &at,
            (&step, &effects),
        );
        frame.top.extend(over);
        frame.top.append(&mut multiplier);
        frame.lights.extend(over_light);
        // The screen passes over it all, clear of the lyric band and what
        // the singer reads.
        let at = self.screen_at(area, &lane);
        self.effects_at(&at, &effects, (&mut frame.screen, &mut Vec::new()));
        frame.lights.append(&mut shots.lights);
        // The bursts that are light, kept until they are over.
        self.lights.retain(|s| s.alive(clock));
        self.lights.append(&mut frame.light_sprays);

        self.sprays.retain(|s| s.alive(clock));
        sprays.extend_from_slice(&self.sprays);
        StageFrame {
            uploads: std::mem::take(&mut self.atlas.uploads),
            paths,
            sprays,
            light_sprays: self.lights.clone(),
            music,
            looks: effects.looks(),
            shake: self.shake,
            ..frame
        }
    }
}

/// The runs of text in `out`, not their outline's dark copies
/// ([`Stage::outlined_text`]).
#[cfg(test)]
fn runs(out: &[Instance]) -> Vec<Instance> {
    out.iter().filter(|i| !Shade::is_copy(i)).copied().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_lane_puts_a_second_ahead_one_second_s_pixels_right_of_the_playhead() {
        let lane = Lane::new((0.0, 0.0, 1600.0, 900.0), 10.0);
        assert_eq!(lane.head, 1600.0 * HEAD);
        assert!((lane.x(11.0) - (lane.head + lane.pps)).abs() < 1e-3);
        assert!((lane.x(10.0 + AHEAD) - 1600.0 * 0.975).abs() < 0.01);
        // Higher up the higher, the middle in the band's.
        assert!(lane.y(1.0) < lane.y(0.0) && lane.y(0.0) < lane.y(-1.0));
        assert_eq!(lane.y(0.0), (lane.band.0 + lane.band.1) / 2.0);
    }

    #[test]
    fn the_capitals_middle_is_inside_the_text() {
        let mut stage = Stage::default();
        let m = stage.cap_middle(Face::LINE, 40.0);
        // Below the top, above the baseline (the ascent, about 0.9 of it).
        assert!(
            m > COVER_PAD as f32 + 4.0 && m < COVER_PAD as f32 + 40.0,
            "{m}"
        );
    }
}
