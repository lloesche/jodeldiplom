//! Karaoke mode's stage, made frame by frame for the GPU
//! ([`super::stage_gpu`]): the song's notes as pills on a lane that runs
//! by a playhead (by time: [`AHEAD`] seconds of the song ahead of it), the
//! words in them, filled where the singer hit them; the singer as an
//! arrow at the playhead, gliding to their pitch, its trail behind it and
//! sparks flying off it; the streak's multiplier over the playhead; the
//! line sung and the next at the bottom; and the song's progress, its
//! breaks apart, along the bottom edge.
//!
//! What moves by itself (the arrow, the sparks, the multiplier's pop) is
//! stepped by the time between frames, so it moves the same at any frame
//! rate.

use std::collections::HashMap;

use super::karaoke::{BigText, COVER_PAD};
use super::sparks::noise;
use super::stage_gpu::{kind, AtlasUpload, Instance, Palette, Skin, StageFrame, Tint, ATLAS};
use crate::core::singing::{multiplier, Hit};
use crate::core::State;
use crate::view::karaoke::{KaraokePicture, LineKind, LyricLine};

/// Where the playhead is, a share of the width from the left.
pub const HEAD: f32 = 0.28;
/// The seconds of song shown ahead of the playhead.
pub const AHEAD: f64 = 4.0;
/// The lane's band: its top and bottom, shares of the height.
const BAND: (f32, f32) = (0.17, 0.55);
/// A pill's height, a share of the window's height.
const PILL: f32 = 0.052;
/// The line sung and the next: their tops, shares of the height.
const LINES: (f32, f32) = (0.68, 0.78);
/// The progress bar: its middle, a share of the height.
const PROGRESS: f32 = 0.935;
/// A gap between notes this long or longer is a break on the progress bar.
const BREAK: f64 = 4.0;
/// How soon after its last pitch the arrow fades, in seconds.
const ARROW_HOLD: f64 = 0.25;
/// How fast the arrow follows the pitch (a critically damped spring, per
/// second).
const SPRING: f32 = 55.0;
/// The least a word's text is made, a share of its size, to fit before
/// the next word.
const SMALLEST_TEXT: f32 = 0.75;
/// The most sparks at once.
const MOST_SPARKS: usize = 700;

/// sRGB colours, 0 to 1.
type Rgb = [f32; 3];
const WHITE: Rgb = [1.0, 1.0, 1.0];
const GOLD_TOP: Rgb = [1.0, 0.93, 0.6];
const GOLD: Rgb = [1.0, 0.72, 0.2];
const GOLD_DEEP: Rgb = [0.98, 0.55, 0.12];
const PALE_TOP: Rgb = [0.9, 0.84, 0.7];
const PALE: Rgb = [0.72, 0.6, 0.4];
const CORAL: Rgb = [0.95, 0.43, 0.31];
const ACTIVE: Rgb = [1.0, 0.86, 0.5];

fn rgba(c: Rgb, a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], a]
}

/// Text kept in the atlas, shelf by shelf: where each (text, size) is.
#[derive(Default)]
struct Atlas {
    places: HashMap<(String, u32), (u32, u32, u32, u32)>,
    /// Where the next goes: across, the shelf's top, the shelf's height.
    pen: (u32, u32, u32),
    uploads: Vec<AtlasUpload>,
}

impl Atlas {
    /// Where `text` at `px` is in the atlas (put there now if it is not),
    /// its quad's size; none where it does not fit even in an empty one.
    fn place(&mut self, big: &mut BigText, text: &str, px: f32) -> Option<(u32, u32, u32, u32)> {
        let key = (text.to_owned(), (px * 4.0).round() as u32);
        if let Some(&place) = self.places.get(&key) {
            return Some(place);
        }
        let (w, h, coverage) = big.coverage(text, px);
        let (w, h) = (w as u32, h as u32);
        if w > ATLAS || h > ATLAS {
            return None;
        }
        if self.pen.0 + w > ATLAS {
            self.pen = (0, self.pen.1 + self.pen.2 + 2, 0);
        }
        if self.pen.1 + h > ATLAS {
            // Full: started again (what shows is put back as it is asked
            // for).
            self.places.clear();
            self.pen = (0, 0, 0);
        }
        let place = (self.pen.0, self.pen.1, w, h);
        self.pen.0 += w + 2;
        self.pen.2 = self.pen.2.max(h);
        self.places.insert(key, place);
        self.uploads.push(AtlasUpload {
            rect: place,
            coverage,
        });
        Some(place)
    }
}

/// A spark flying off the arrow.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spark {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    age: f32,
    life: f32,
    size: f32,
    turn: f32,
    spin: f32,
    color: Rgb,
    phase: f32,
}

/// The singer's arrow: where it is (pixels down) and how fast it moves,
/// and how much it shows.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Arrow {
    y: f32,
    speed: f32,
    shown: f32,
    placed: bool,
}

impl Arrow {
    /// `dt` seconds on, gliding to `to` (none: fading).
    fn step(&mut self, to: Option<f32>, dt: f32) {
        match to {
            Some(to) => {
                if !self.placed {
                    self.y = to;
                    self.speed = 0.0;
                    self.placed = true;
                }
                // A critically damped spring, in steps short enough to
                // stay put at any frame rate.
                let steps = (dt / 0.004).ceil().max(1.0) as usize;
                let h = dt / steps as f32;
                for _ in 0..steps {
                    let pull = -2.0 * SPRING * self.speed - SPRING * SPRING * (self.y - to);
                    self.speed += pull * h;
                    self.y += self.speed * h;
                }
                self.shown = (self.shown + dt / 0.08).min(1.0);
            }
            None => {
                self.shown = (self.shown - dt / 0.15).max(0.0);
                if self.shown == 0.0 {
                    self.placed = false;
                }
            }
        }
    }
}

/// Karaoke mode's stage between frames.
#[derive(Default)]
pub struct Stage {
    text: BigText,
    atlas: Atlas,
    sparks: std::collections::VecDeque<Spark>,
    arrow: Arrow,
    seed: u32,
    /// Sparks owed to the next frame (a share of one).
    owed: f32,
    /// The last word scored, by its end.
    scored: Option<f64>,
    /// The multiplier shown, how much it shows, and how long since it
    /// went up.
    badge: (u32, f32, f32),
    /// Where the capitals' middle is in text of a size ([`Stage::cap_middle`]).
    caps: HashMap<u32, f32>,
    /// The song's time at the frame before: what moves goes by the song.
    before: Option<f64>,
    /// The words scored bursting, handed to the GPU ([`super::particles`]).
    sprays: Vec<super::particles::Spray>,
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
}

impl<'a> Song<'a> {
    /// The song in Karaoke mode in `state`.
    pub fn of(state: &'a State) -> Song<'a> {
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
        }
    }
}

/// Whether the stage shows for `state`: Karaoke mode, its notes shown, the
/// song in it with notes made.
pub fn wanted(state: &State) -> bool {
    if !crate::core::karaoke::in_mode(state) || !state.config.karaoke_notes {
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

/// `picture` without what the stage draws instead: its lyric lines (and
/// their notes).
pub fn without_lyrics(picture: &KaraokePicture) -> KaraokePicture {
    let keep = |l: &&LyricLine| l.id.is_none();
    KaraokePicture {
        lines: picture.lines.iter().filter(keep).cloned().collect(),
        before: picture.before.iter().filter(keep).cloned().collect(),
        ..picture.clone()
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
    Instance {
        quad: [
            c.0 - half.0 - m,
            c.1 - half.1 - m,
            c.0 + half.0 + m,
            c.1 + half.1 + m,
        ],
        shape: [c.0, c.1, half.0, half.1],
        color: body,
        color2: edge,
        params: [kind::PILL, glow, outline, radius],
        uv: [0.0; 4],
        strain: [0.0; 4],
    }
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
    Some(Instance {
        quad: [x0 - m, c.1 - half.1 - m, x1 + m, c.1 + half.1 + m],
        shape: [c.0, c.1, half.0, half.1],
        color: top,
        color2: bottom,
        params: [kind::FILL, glow, 0.0, 0.0],
        uv: [x0, x1, 0.0, 0.0],
        strain: [0.0; 4],
    })
}

/// How much shows at `x` across the window `(x0, w)`: fading out at its
/// edges.
fn edge_fade(x: f32, (x0, w): (f32, f32)) -> f32 {
    let fade = w * 0.05;
    let left = ((x - (x0 + w * 0.012)) / fade).clamp(0.0, 1.0);
    let right = ((x0 + w * 0.988 - x) / fade).clamp(0.0, 1.0);
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
    Instance {
        quad: [
            a.0.min(b.0) - m,
            a.1.min(b.1) - m,
            a.0.max(b.0) + m,
            a.1.max(b.1) + m,
        ],
        shape: [a.0, a.1, b.0, b.1],
        color: core,
        color2: glow_color,
        params: [kind::SEGMENT, thick, glow, 0.0],
        uv: [0.0; 4],
        strain: [0.0; 4],
    }
}

fn disc(c: (f32, f32), r: f32, core: [f32; 4], glow_color: [f32; 4], glow: f32) -> Instance {
    let m = r + glow + 1.0;
    Instance {
        quad: [c.0 - m, c.1 - m, c.0 + m, c.1 + m],
        shape: [c.0, c.1, r, glow],
        color: core,
        color2: glow_color,
        params: [kind::DISC, 0.0, 0.0, 0.0],
        uv: [0.0; 4],
        strain: [0.0; 4],
    }
}

impl Stage {
    /// A new stage.
    pub fn new() -> Stage {
        Stage {
            text: BigText::lyrics(),
            ..Stage::default()
        }
    }

    /// A run of text: `text` at `px` (drawn at `scale` of that) with its
    /// line's top-left at `at`; left of `split` (pixels across) in
    /// `filled`, right of it in `rest`.
    #[allow(clippy::too_many_arguments)]
    fn text(
        &mut self,
        out: &mut Vec<Instance>,
        text: &str,
        px: f32,
        scale: f32,
        at: (f32, f32),
        split: f32,
        rest: [f32; 4],
        filled: [f32; 4],
    ) {
        if text.trim().is_empty() {
            return;
        }
        let Some((u, v, w, h)) = self.atlas.place(&mut self.text, text, px) else {
            return;
        };
        let pad = COVER_PAD as f32 * scale;
        let (x0, y0) = (at.0 - pad, at.1 - pad);
        out.push(Instance {
            quad: [x0, y0, x0 + w as f32 * scale, y0 + h as f32 * scale],
            shape: [0.0; 4],
            color: rest,
            color2: filled,
            params: [kind::TEXT, split, 0.0, 0.0],
            uv: [u as f32, v as f32, (u + w) as f32, (v + h) as f32],
            strain: [0.0; 4],
        });
    }

    /// How far down from the top of text made at `px` for the atlas
    /// ([`BigText::coverage`]) the middle of its capitals is, between
    /// their top and the baseline: kept by size.
    fn cap_middle(&mut self, px: f32) -> f32 {
        let key = (px * 4.0).round() as u32;
        if let Some(&m) = self.caps.get(&key) {
            return m;
        }
        let (w, h, coverage) = self.text.coverage("H", px);
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

    fn rand(&mut self) -> f32 {
        self.seed = self.seed.wrapping_add(1);
        noise(self.seed)
    }

    /// `n` sparks off (`x`, `y`): `color`, `size` across, `spread` how
    /// far they fly from there, `reach` pixels a second at 1.
    fn emit(
        &mut self,
        n: usize,
        (x, y): (f32, f32),
        color: Rgb,
        size: f32,
        spread: f32,
        reach: f32,
    ) {
        for _ in 0..n {
            if self.sparks.len() >= MOST_SPARKS {
                self.sparks.pop_front();
            }
            let (a, b, c, d, e) = (
                self.rand(),
                self.rand(),
                self.rand(),
                self.rand(),
                self.rand(),
            );
            let white = self.rand() < 0.3;
            let (f, g, i, j) = (self.rand(), self.rand(), self.rand(), self.rand());
            self.sparks.push_back(Spark {
                x: x + (a - 0.5) * size * 0.6,
                y: y + (b - 0.5) * size * 0.6,
                vx: (c - 0.65) * reach * 4.0 * spread,
                vy: (d - 0.5) * reach * 4.0 * spread,
                age: 0.0,
                life: 0.4 + 0.7 * e,
                size: size * (0.35 + 0.65 * f),
                turn: g * std::f32::consts::TAU,
                spin: (i - 0.5) * 6.0,
                color: if white { [1.0, 0.97, 0.88] } else { color },
                phase: j * std::f32::consts::TAU,
            });
        }
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
        let (x, y, w, h) = area;
        let singing = song.singing;
        let game = song.game;
        let p = lane.pill;
        let behind = lane.behind();
        let (from, to) = (now - behind - 0.5, now + AHEAD + 0.5);
        let sung = &song.sung;

        // The lane: a band of glass, the playhead across it.
        let lane_half = ((lane.band.1 - lane.band.0) / 2.0 + p * 0.55, w * 0.4875);
        let lane_mid = ((x + w * 0.5), (lane.band.0 + lane.band.1) / 2.0);
        out.push(pill(
            lane_mid,
            (lane_half.1, lane_half.0),
            rgba(WHITE, 0.035),
            rgba(WHITE, 0.07),
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

        // The words, line by line: a pill each, as high as its longest
        // note, as long as it is sung (or its text, where longer: the text
        // made smaller to fit before the next word, down to
        // [`SMALLEST_TEXT`] of its size), filled where it was hit.
        let text_px = (p * 0.56).round().max(9.0);
        let pad = p * 0.34;
        let mut words: Vec<(usize, usize, f64, f64, f32)> = Vec::new();
        for (l, notes) in singing.lines.iter().enumerate() {
            let (Some(first), Some(last)) = (notes.notes.first(), notes.notes.last()) else {
                continue;
            };
            if last.end < from || first.start > to {
                continue;
            }
            let at = words.len();
            for n in &notes.notes {
                match words.last_mut() {
                    Some(wd) if wd.0 == l && wd.1 == n.word => {
                        wd.3 = wd.3.max(n.end);
                    }
                    _ => words.push((l, n.word, n.start, n.end, n.height)),
                }
            }
            // Each word as high as its longest note.
            for wd in &mut words[at..] {
                if let Some(n) = notes
                    .notes
                    .iter()
                    .filter(|n| n.word == wd.1)
                    .max_by(|a, b| (a.end - a.start).total_cmp(&(b.end - b.start)))
                {
                    wd.4 = n.height;
                }
            }
        }
        /// A word's text: it, its size, where (its left, its pill's
        /// middle), and when it ends, whether it was hit, how much shows.
        type Label = (String, f32, f32, f32, (f64, bool, f32));
        let mut texts: Vec<Label> = Vec::new();
        let word_text = |l: usize, wi: usize| -> &str {
            sung.get(l)
                .and_then(|line| line.words.get(wi))
                .map(|w| w.text.as_str())
                .unwrap_or_default()
        };
        // Laid out in time, one after another: no two pills meet. Each
        // starts when it is sung, or just after the one before ends where
        // that is later; it lasts as long as it is sung, or as its text
        // (made smaller to fit before the next word, down to
        // [`SMALLEST_TEXT`]), or a circle, whichever is longest.
        let gap = f64::from(p * 0.15 / lane.pps);
        let mut laid: Vec<(f32, f32, f64, f64)> = Vec::with_capacity(words.len());
        let mut last_end = f64::NEG_INFINITY;
        for (k, &(l, wi, start, end, _)) in words.iter().enumerate() {
            let text = word_text(l, wi);
            let t0 = start.max(last_end + gap);
            let room = words.get(k + 1).map_or(f32::INFINITY, |next| {
                ((next.2 - t0) as f32) * lane.pps - p * 0.15
            });
            let mut px = text_px;
            let mut tw = self.text.measure(text, px);
            if tw + 2.0 * pad > room {
                let fit = ((room - 2.0 * pad) / tw.max(1.0)).clamp(SMALLEST_TEXT, 1.0);
                // In steps of two pixels, so that the atlas keeps few sizes.
                px = ((text_px * fit) / 2.0).floor().max(4.0) * 2.0;
                tw = self.text.measure(text, px);
            }
            let least = (tw + 2.0 * pad).max(p);
            let t1 = end.max(t0 + f64::from(least / lane.pps));
            last_end = t1;
            laid.push((px, tw, t0, t1));
        }
        let mut pills = super::pop::Pills::new(singing, clock);
        let mut sprays = Vec::new();
        for (k, &(l, wi, start, end, height)) in words.iter().enumerate() {
            let text = word_text(l, wi);
            let (px, tw, t0, t1) = laid[k];
            let (x0, x1) = (lane.x(t0), lane.x(t1));
            if x1 < x - p || x0 > x + w + p {
                continue;
            }
            // Its sung time across all of it.
            let across = |t: f64| x0 + ((t - start) / (end - start).max(1e-3)) as f32 * (x1 - x0);
            let c = ((x0 + x1) / 2.0, lane.y(height));
            let half = ((x1 - x0) / 2.0, p / 2.0);
            let fade = edge_fade(x0, (x, w)).min(edge_fade(x1, (x, w)));
            if fade <= 0.0 {
                continue;
            }
            // Its bubbles, a row along it (the game on), each with its
            // liquid, straining, bursting as it fills and gone after, its
            // bursts handed to the GPU.
            let bub = pills.pill(game, l, wi);
            sprays.extend(bub.sprays(&across, (c, half, p), (clock, lane.pps)));
            let (body, edge, glow) = if end <= now {
                (rgba(WHITE, 0.07), rgba(WHITE, 0.22), 0.0)
            } else if start <= now {
                (rgba(WHITE, 0.16), rgba(ACTIVE, 0.95), p * 0.4)
            } else {
                (rgba(WHITE, 0.12), rgba(WHITE, 0.5), 0.0)
            };
            let faded = |c: [f32; 4]| [c[0], c[1], c[2], c[3] * fade];
            let outline = (p * 0.05).max(1.2);
            let skin = Skin {
                body: faded(body),
                edge: faded(edge),
                glow,
                outline,
            };
            // Their liquid, as the time went by under the playhead.
            let palette = Palette {
                full: Tint {
                    top: rgba(GOLD_TOP, fade),
                    bottom: rgba(GOLD_DEEP, 0.8 * fade),
                    glow: p * 0.35,
                },
                half: Tint {
                    top: rgba(PALE_TOP, 0.85 * fade),
                    bottom: rgba(PALE, 0.45 * fade),
                    glow: p * 0.15,
                },
            };
            let filled = bub.draw(&mut out, (c, half), (&skin, &palette), &across, p);
            if !text.is_empty() {
                let hit = filled > 0.5 * (end - start);
                texts.push((text.to_owned(), px, c.0 - tw / 2.0, c.1, (end, hit, fade)));
            }
        }
        let word_y: HashMap<(usize, usize), f32> =
            words.iter().map(|w| ((w.0, w.1), lane.y(w.4))).collect();
        let snaps = singing.level.snaps();
        let y_of = |hf: &crate::core::singing::Heard| -> Option<f32> {
            let l = hf.line?;
            let notes = singing.lines.get(l)?;
            let n = notes
                .notes
                .iter()
                .find(|n| hf.t >= n.start && hf.t < n.end)?;
            let on = *word_y.get(&(l, n.word))?;
            if snaps && hf.hit == Some(Hit::Full) {
                return Some(on);
            }
            Some(on + lane.y(notes.height(hf.cents)) - lane.y(n.height))
        };
        let recent: Vec<_> = singing
            .heard
            .iter()
            .rev()
            .take_while(|h| h.t >= clock - behind)
            .filter(|h| h.t <= clock)
            .collect();
        let mut last_point: Option<((f32, f32), Option<Hit>, f64)> = None;
        for pair in recent.windows(2).rev() {
            let (a, b) = (pair[1], pair[0]);
            let (Some(ya), Some(yb)) = (y_of(a), y_of(b)) else {
                continue;
            };
            let pa = (lane.x(a.t), ya);
            let pb = (lane.x(b.t), yb);
            last_point = Some((pb, b.hit, b.t));
            if b.t - a.t > 0.035 {
                continue;
            }
            let hit = if game { b.hit } else { None };
            // On a level that snaps, a hit is the fill itself; and from a
            // hit to a miss the line would jump off the pill: neither drawn.
            let full = |h: Option<Hit>| game && h == Some(Hit::Full);
            if snaps && (full(a.hit) || full(b.hit)) {
                continue;
            }
            let (core, glow_c, thick, g) = match hit {
                // On hard, as sung over the pills: thin, the words read.
                Some(Hit::Full) => ([1.0, 0.9, 0.55, 0.95], GOLD, p * 0.07, p * 0.25),
                Some(Hit::Half) => ([0.95, 0.88, 0.72, 0.9], PALE_TOP, p * 0.11, p * 0.25),
                None => (rgba(CORAL, 0.85), CORAL, p * 0.06, p * 0.22),
            };
            // A glow the length of the piece, so that pieces side by side
            // add up to one line's.
            let len = ((pb.0 - pa.0).powi(2) + (pb.1 - pa.1).powi(2)).sqrt();
            let share = (len / (g * 0.9)).min(1.0);
            out.push(segment(pa, pb, thick, core, rgba(glow_c, 0.5 * share), g));
        }
        if recent.len() == 1 {
            let h = recent[0];
            if let Some(yh) = y_of(h) {
                last_point = Some(((lane.x(h.t), yh), h.hit, h.t));
            }
        }
        // The words over the trail.
        for (word, px, tx, ty, (end, hit, fade)) in texts {
            let alpha = if end <= now && !hit { 0.45 } else { 0.95 } * fade;
            // Its capitals' middle on the pill's.
            let at = (tx, ty - self.cap_middle(px) + COVER_PAD as f32);
            // Dark on the gold where it was hit, up to the playhead.
            let (split, done) = if hit {
                (lane.head, [0.27, 0.15, 0.03, 0.95 * fade])
            } else {
                (f32::NEG_INFINITY, rgba(WHITE, alpha))
            };
            self.text(
                &mut out,
                &word,
                px,
                1.0,
                at,
                split,
                rgba(WHITE, alpha),
                done,
            );
        }

        // The singer: their trail, and where they are now.
        let fresh = last_point.filter(|(_, _, t)| clock - t < ARROW_HOLD);
        self.arrow.step(fresh.map(|(p, _, _)| p.1), dt);
        let hitting = if game { fresh.and_then(|f| f.1) } else { None };
        let arrow_color = match hitting {
            Some(Hit::Full) => GOLD,
            Some(Hit::Half) => PALE_TOP,
            None => CORAL,
        };
        let tip = (lane.head + p * 0.42, self.arrow.y);
        if self.arrow.shown > 0.0 {
            // The trail on to the arrow.
            if let Some((pt, _, _)) = fresh {
                let thick = if hitting == Some(Hit::Full) {
                    p * 0.16
                } else {
                    p * 0.07
                };
                out.push(segment(
                    pt,
                    (lane.head, self.arrow.y),
                    thick,
                    rgba(arrow_color, 0.85 * self.arrow.shown),
                    rgba(arrow_color, 0.35 * self.arrow.shown),
                    p * 0.3,
                ));
            }
        }

        // The sparks: owed by the time gone by, more on the note.
        if self.arrow.shown > 0.5 {
            let rate = match hitting {
                Some(Hit::Full) => 150.0,
                Some(Hit::Half) => 80.0,
                None => 28.0,
            };
            self.owed += rate * song_dt;
            let n = self.owed.floor() as usize;
            self.owed -= n as f32;
            let color = if hitting.is_some() { GOLD } else { CORAL };
            self.emit(n, (lane.head, self.arrow.y), color, p * 0.55, 1.0, p);
        } else {
            self.owed = 0.0;
        }
        // A word scored: a burst where it hit; more as the multiplier
        // goes up.
        // Once a word (not again sought back a little), any again started over.
        let over = before.is_some_and(|b| now < b - 1.0) || singing.heard.is_empty();
        self.scored = self.scored.filter(|_| !over);
        let last = singing.tally.last_scored;
        if let Some(scored) = last.filter(|s| self.scored.is_none_or(|e| s.end > e)) {
            self.scored = Some(scored.end);
            if game && scored.hit == Some(Hit::Full) {
                let up = multiplier(scored.streak) > multiplier(scored.streak.saturating_sub(1));
                let at = (lane.x(scored.end).max(x), self.arrow.y);
                let from = if up { tip } else { at };
                self.sprays
                    .push(super::pop::scored(from, p, clock, lane.pps, up, scored.end));
                if up {
                    self.badge.2 = 0.0;
                }
            }
        }
        for s in &mut self.sparks {
            s.age += song_dt;
            // Their own way, and by with the lane.
            s.x += (s.vx - lane.pps) * song_dt;
            s.y += s.vy * song_dt;
            let drag = (-2.4 * song_dt).exp();
            s.vx *= drag;
            s.vy *= drag;
            s.turn += s.spin * song_dt;
        }
        self.sparks.retain(|s| s.age < s.life);
        for s in &self.sparks {
            let left = 1.0 - s.age / s.life;
            let twinkle = 0.6 + 0.4 * (s.age * 26.0 + s.phase).sin();
            let size = s.size * (0.45 + 0.55 * left);
            let m = size + 1.0;
            out.push(Instance {
                quad: [s.x - m, s.y - m, s.x + m, s.y + m],
                shape: [s.x, s.y, size, s.turn],
                color: rgba(s.color, left.powf(1.2) * twinkle),
                color2: [0.0; 4],
                params: [kind::SPARK, 0.0, 0.0, 0.0],
                uv: [0.0; 4],
                strain: [0.0; 4],
            });
        }
        if self.arrow.shown > 0.0 {
            let a = self.arrow.shown;
            out.push(disc(
                (lane.head, self.arrow.y),
                p * 0.16,
                rgba([1.0, 0.98, 0.92], a),
                rgba(arrow_color, 0.4 * a),
                p * 0.6,
            ));
            let len = p * 0.95;
            out.push(Instance {
                quad: [
                    tip.0 - len * 1.6,
                    tip.1 - len * 1.2,
                    tip.0 + len * 0.6,
                    tip.1 + len * 1.2,
                ],
                shape: [tip.0, tip.1, len, 0.0],
                color: rgba(arrow_color, a),
                color2: rgba(arrow_color, 0.6 * a),
                params: [kind::ARROW, p * 0.35, 0.0, 0.0],
                uv: [0.0; 4],
                strain: [0.0; 4],
            });
        }

        // The multiplier, over the playhead: popping as it goes up.
        let m = if game { singing.multiplier() } else { 1 };
        let shown_target = if m > 1 { 1.0 } else { 0.0 };
        if m > 1 {
            self.badge.0 = m;
        }
        self.badge.1 += (shown_target - self.badge.1) * (1.0 - (-dt / 0.12).exp());
        self.badge.2 += dt;
        if self.badge.1 > 0.01 && self.badge.0 > 1 {
            let pop = 1.0 + 0.45 * (-self.badge.2 * 7.0).exp();
            let a = self.badge.1;
            let px = (p * 1.05).round();
            let label = format!("×{}", self.badge.0);
            let tw = self.text.measure(&label, px) * pop;
            let cy = lane.band.0 - p * 1.55;
            out.push(disc(
                (lane.head, cy),
                0.0,
                [0.0; 4],
                rgba(GOLD, (0.25 + 0.1 * self.badge.0 as f32) * a * pop),
                p * 1.4 * pop,
            ));
            self.text(
                &mut out,
                &label,
                px,
                pop,
                (lane.head - tw / 2.0, cy - px * 0.62 * pop),
                f32::NEG_INFINITY,
                rgba(GOLD_TOP, a),
                rgba(GOLD_TOP, a),
            );
            if game && singing.tally.streak > 0 {
                let small = (p * 0.36).round().max(9.0);
                let streak = format!("{} in a row", singing.tally.streak);
                let sw = self.text.measure(&streak, small);
                self.text(
                    &mut out,
                    &streak,
                    small,
                    1.0,
                    (lane.head - sw / 2.0, cy + px * 0.62),
                    f32::NEG_INFINITY,
                    rgba(WHITE, 0.7 * a),
                    rgba(WHITE, 0.7 * a),
                );
            }
        }

        // The line sung and the next, moving up as the lines change.
        let big = (h / 15.0).clamp(18.0, 120.0).round();
        let slot = |kind: LineKind| -> Option<(f32, f32, f32)> {
            let (cur, next) = (y + h * LINES.0, y + h * LINES.1);
            match kind {
                LineKind::Current => Some((cur, 1.0, 1.0)),
                LineKind::Next => Some((next, 0.66, 0.6)),
                LineKind::Previous => Some((cur - (next - cur), 0.66, 0.0)),
                _ => None,
            }
        };
        let moved = picture.moved.clamp(0.0, 1.0) as f32;
        let mut drawn: Vec<(&LyricLine, f32, f32, f32)> = Vec::new();
        // Only the first line coming (the view gives two).
        let first_next = picture
            .lines
            .iter()
            .position(|l| l.kind == LineKind::Next && l.id.is_some());
        for (i, line) in picture
            .lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.id.is_some())
        {
            if line.kind == LineKind::Next && Some(i) != first_next {
                continue;
            }
            let Some((ty, scale, alpha)) = slot(line.kind) else {
                continue;
            };
            let from = picture
                .before
                .iter()
                .find(|b| b.id == line.id)
                .and_then(|b| slot(b.kind))
                .unwrap_or((ty + (y + h * LINES.1 - y - h * LINES.0), scale, 0.0));
            let lerp = |a: f32, b: f32| a + (b - a) * moved;
            drawn.push((
                line,
                lerp(from.0, ty),
                lerp(from.1, scale),
                lerp(from.2, alpha),
            ));
        }
        if moved < 1.0 {
            for b in picture.before.iter().filter(|b| b.id.is_some()) {
                if picture.lines.iter().any(|l| l.id == b.id) {
                    continue;
                }
                let (Some(from), Some(to)) = (slot(b.kind), slot(LineKind::Previous)) else {
                    continue;
                };
                let lerp = |a: f32, b: f32| a + (b - a) * moved;
                drawn.push((
                    b,
                    lerp(from.0, to.0),
                    lerp(from.1, to.1),
                    lerp(from.2, to.2),
                ));
            }
        }
        for (line, ty, scale, alpha) in drawn {
            let alpha = alpha * line.alpha as f32;
            if alpha <= 0.01 {
                continue;
            }
            // Smaller where it would not fit.
            let width = self.text.measure(&line.text, big);
            let scale = scale * (w * 0.92 / width.max(1.0)).min(1.0);
            let lw = width * scale;
            let lx = x + (w - lw) / 2.0;
            let split = if line.kind == LineKind::Current {
                let n = line.text.chars().count();
                let at = line.fill.clamp(0.0, 1.0) * n as f64;
                let whole = at.floor() as usize;
                let before: String = line.text.chars().take(whole).collect();
                let mut sx = self.text.measure(&before, big);
                if let Some(c) = line.text.chars().nth(whole) {
                    sx += (at - whole as f64) as f32 * self.text.measure(&c.to_string(), big);
                }
                lx + sx * scale
            } else {
                f32::NEG_INFINITY
            };
            self.text(
                &mut out,
                &line.text,
                big,
                scale,
                (lx, ty),
                split,
                rgba(WHITE, 0.95 * alpha),
                rgba(GOLD, alpha),
            );
        }

        // The song's progress: its sung stretches apart from its breaks,
        // filled up to now.
        if let Some(length) = song.length {
            let (bx0, bx1) = (x + w * 0.06, x + w * 0.9);
            let by = y + h * PROGRESS;
            let bh = (h * 0.0045).max(2.5);
            let bx = |t: f64| bx0 + (bx1 - bx0) * (t / length).clamp(0.0, 1.0) as f32;
            out.push(pill(
                ((bx0 + bx1) / 2.0, by),
                ((bx1 - bx0) / 2.0, bh * 0.6),
                rgba(WHITE, 0.12),
                rgba(WHITE, 0.0),
                0.0,
                0.0,
                0.0,
            ));
            let mut stretches: Vec<(f64, f64)> = Vec::new();
            for n in singing.lines.iter().flat_map(|l| l.notes.iter()) {
                match stretches.last_mut() {
                    Some(s) if n.start - s.1 < BREAK => s.1 = s.1.max(n.end),
                    _ => stretches.push((n.start, n.end)),
                }
            }
            stretches.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (a, b) in &stretches {
                let (xa, xb) = (bx(*a), bx(*b));
                out.push(pill(
                    ((xa + xb) / 2.0, by),
                    (((xb - xa) / 2.0).max(bh), bh),
                    rgba(WHITE, 0.32),
                    rgba(WHITE, 0.0),
                    0.0,
                    0.0,
                    0.0,
                ));
            }
            let at = bx(now);
            out.extend(fill(
                ((bx0 + bx1) / 2.0, by),
                ((bx1 - bx0) / 2.0, bh),
                (bx0, at),
                rgba(GOLD_TOP, 0.95),
                rgba(GOLD_DEEP, 0.8),
                bh * 3.0,
            ));
            out.push(disc(
                (at, by),
                bh * 1.7,
                rgba([1.0, 0.98, 0.9], 1.0),
                rgba(GOLD, 0.6),
                bh * 5.0,
            ));
            // The time left, right of it.
            let left = (length - now.max(0.0)).max(0.0).ceil() as u64;
            let label = format!("{}:{:02}", left / 60, left % 60);
            let px = (h * 0.026).round().clamp(10.0, 48.0);
            let at = (bx1 + w * 0.018, by - self.cap_middle(px) + COVER_PAD as f32);
            self.text(
                &mut out,
                &label,
                px,
                1.0,
                at,
                f32::NEG_INFINITY,
                rgba(WHITE, 0.85),
                rgba(WHITE, 0.85),
            );
        }

        self.sprays.retain(|s| s.alive(clock));
        sprays.extend_from_slice(&self.sprays);
        StageFrame {
            instances: out,
            uploads: std::mem::take(&mut self.atlas.uploads),
            sprays,
            clock: clock as f32,
        }
    }
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
    fn the_arrow_glides_to_the_pitch_alike_at_any_frame_rate() {
        for fps in [60.0f32, 144.0] {
            let mut arrow = Arrow::default();
            arrow.step(Some(100.0), 1.0 / fps);
            let frames = (0.08 * fps).round() as usize;
            for _ in 0..frames {
                arrow.step(Some(200.0), 1.0 / fps);
            }
            // Most of the way within 80 ms, not past it.
            assert!(arrow.y > 190.0 && arrow.y <= 200.5, "{fps}: {}", arrow.y);
        }
        // Without a pitch it fades, and starts afresh where the next is.
        let mut arrow = Arrow::default();
        arrow.step(Some(100.0), 0.016);
        for _ in 0..20 {
            arrow.step(None, 0.016);
        }
        assert_eq!(arrow.shown, 0.0);
        arrow.step(Some(300.0), 0.016);
        assert_eq!(arrow.y, 300.0);
    }

    #[test]
    fn sparks_fly_and_go() {
        let mut stage = Stage::new();
        stage.emit(10, (100.0, 100.0), GOLD, 20.0, 1.0, 40.0);
        assert_eq!(stage.sparks.len(), 10);
        // Most fly back, and all go by with the lane.
        assert!(stage.sparks.iter().filter(|s| s.vx < 0.0).count() >= 5);
        for s in &mut stage.sparks {
            s.age = s.life;
        }
        stage.sparks.retain(|s| s.age < s.life);
        assert!(stage.sparks.is_empty());
    }

    #[test]
    fn the_capitals_middle_is_inside_the_text() {
        let mut stage = Stage::new();
        let m = stage.cap_middle(40.0);
        // Below the top, above the baseline (the ascent, about 0.9 of it).
        assert!(
            m > COVER_PAD as f32 + 4.0 && m < COVER_PAD as f32 + 40.0,
            "{m}"
        );
    }

    #[test]
    fn the_atlas_keeps_text_once_and_starts_again_when_full() {
        let mut big = BigText::default();
        let mut atlas = Atlas::default();
        let a = atlas.place(&mut big, "hello", 40.0).unwrap();
        assert_eq!(atlas.place(&mut big, "hello", 40.0), Some(a));
        assert_eq!(atlas.uploads.len(), 1);
        let b = atlas.place(&mut big, "world", 40.0).unwrap();
        assert!(b.0 >= a.0 + a.2);
        // Full: started again from the top.
        atlas.pen = (0, ATLAS - 2, 0);
        let c = atlas.place(&mut big, "again", 40.0).unwrap();
        assert_eq!((c.0, c.1), (0, 0));
        assert!(!atlas.places.contains_key(&("hello".to_owned(), 160)));
    }
}
