//! Karaoke mode's stage, made frame by frame for the GPU
//! ([`super::stage_gpu`]): the song's words as tubes through their notes
//! on a lane that runs by a playhead (by time: [`AHEAD`] seconds of the
//! song ahead of it), their text riding in them, filled where the singer
//! hit them ([`words`]); the shots rescuing a word missed ([`shots`]); the
//! singer as an arrow at the playhead, gliding to their pitch, its trail
//! behind it and sparks flying off it; the streak's multiplier over the
//! playhead; the line sung and the next at the bottom; and the song's
//! progress, its breaks apart, along the bottom edge; "AUTO" in a corner
//! while the demo sings ([`tag`]). Its text is in
//! Karaoke mode's own faces ([`Face`]). While a song's result shows, the
//! stage is the result screen instead ([`result`]).
//!
//! What moves by itself (the arrow, the sparks, the multiplier's pop) is
//! stepped by the time between frames, so it moves the same at any frame
//! rate.

use std::collections::HashMap;

use super::big_text::{BigText, Face, Faces, COVER_PAD};
use super::rainbow;

mod comet;
mod lines;
mod meter;
mod result;
mod scene;
mod shade;
mod shots;
mod start;
mod tag;
mod words;
use self::shade::Shade;
use super::sparks::noise;
use super::stage_gpu::{kind, AtlasUpload, Instance, StageFrame, ATLAS};
pub use result::Tint;

use crate::core::State;
use crate::view::karaoke::{KaraokePicture, LyricLine};

/// Where the playhead is, a share of the width from the left.
pub const HEAD: f32 = 0.28;
/// The seconds of song shown ahead of the playhead.
pub const AHEAD: f64 = 4.0;
/// The lane's band: its top and bottom, shares of the height.
const BAND: (f32, f32) = (0.17, 0.55);
/// A pill's height, a share of the window's height.
const PILL: f32 = 0.052;
/// The progress bar: its middle, a share of the height.
const PROGRESS: f32 = 0.935;
/// A gap between notes this long or longer is a break on the progress bar.
const BREAK: f64 = 4.0;
/// How soon after its last pitch the arrow fades, in seconds.
const ARROW_HOLD: f64 = 0.25;
/// How fast the arrow follows the pitch: a weight on a rubber band, its
/// natural frequency (radians a second), damped just enough never to swing
/// past (critically). Most of the way to a new pitch in about a tenth of a
/// second.
const SPRING: f32 = 45.0;
/// What the arrow goes for: the middle (median) of the pitches heard this
/// long back, so a stray one never pulls it.
const STEADY: f32 = 0.035;
/// The most sparks at once.
const MOST_SPARKS: usize = 700;

/// sRGB colours, 0 to 1.
type Rgb = [f32; 3];
const WHITE: Rgb = [1.0, 1.0, 1.0];
/// The lane's glass, dark and a little cool.
const SMOKE: Rgb = [0.02, 0.025, 0.05];
const GOLD_TOP: Rgb = [1.0, 0.93, 0.6];
const GOLD: Rgb = [1.0, 0.72, 0.2];
const GOLD_DEEP: Rgb = [0.98, 0.55, 0.12];
const CORAL: Rgb = [0.95, 0.43, 0.31];
const ACTIVE: Rgb = [1.0, 0.86, 0.5];

fn rgba(c: Rgb, a: f32) -> [f32; 4] {
    [c[0], c[1], c[2], a]
}

/// Text in a face at a size (a quarter pixel's steps).
type Key = (Face, String, u32);

/// Text kept in the atlas, shelf by shelf: where each (face, text, size)
/// is.
#[derive(Default)]
struct Atlas {
    places: HashMap<Key, Placed>,
    /// Where the next goes: across, the shelf's top, the shelf's height.
    pen: (u32, u32, u32),
    uploads: Vec<AtlasUpload>,
}

impl Atlas {
    /// Where `text` at `px` in `big`'s face is in the atlas (put there now
    /// if it is not): its place, and the size it was made at as a share of
    /// `px`. Text wider than the atlas (a long lyric line on a large
    /// screen) is made smaller to fit, its quad drawn larger by as much.
    fn place(&mut self, big: &mut BigText, text: &str, px: f32) -> Option<Placed> {
        let key = (big.face(), text.to_owned(), (px * 4.0).round() as u32);
        if let Some(&place) = self.places.get(&key) {
            return Some(place);
        }
        let (mut w, mut h, mut coverage) = big.coverage(text, px);
        let mut made = 1.0;
        if w.max(h) > ATLAS as usize {
            made = (ATLAS - 8) as f32 / w.max(h) as f32;
            (w, h, coverage) = big.coverage(text, px * made);
        }
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
        let rect = (self.pen.0, self.pen.1, w, h);
        self.pen.0 += w + 2;
        self.pen.2 = self.pen.2.max(h);
        let place = Placed { rect, made };
        self.places.insert(key, place);
        self.uploads.push(AtlasUpload { rect, coverage });
        Some(place)
    }
}

/// Text's place in the atlas: across, down, wide, high (atlas pixels), and
/// the size it was made at as a share of the size asked for.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Placed {
    rect: (u32, u32, u32, u32),
    made: f32,
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
/// and how much it shows; the pitches it was given lately (how long ago,
/// where), to go for their middle.
#[derive(Clone, Debug, Default, PartialEq)]
struct Arrow {
    y: f32,
    speed: f32,
    shown: f32,
    placed: bool,
    lately: std::collections::VecDeque<(f32, f32)>,
}

impl Arrow {
    /// `dt` seconds on, gliding to `to` (none: fading).
    fn step(&mut self, to: Option<f32>, dt: f32) {
        for heard in &mut self.lately {
            heard.0 += dt;
        }
        self.lately.retain(|&(ago, _)| ago <= STEADY);
        match to {
            Some(to) => {
                self.lately.push_back((0.0, to));
                let to = self.steady();
                if !self.placed {
                    self.y = to;
                    self.speed = 0.0;
                    self.placed = true;
                }
                // A critically damped spring, in steps short enough to move
                // alike at any frame rate.
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
                    self.lately.clear();
                }
            }
        }
    }

    /// The middle of the pitches given lately; where they are even, of the
    /// two middle ones the nearer to where the arrow is (a new pitch moves
    /// it only once it is most of those heard).
    fn steady(&self) -> f32 {
        let mut ys: Vec<f32> = self.lately.iter().map(|&(_, y)| y).collect();
        ys.sort_by(f32::total_cmp);
        let upper = ys[ys.len() / 2];
        if ys.len() % 2 == 1 || !self.placed {
            return upper;
        }
        let lower = ys[ys.len() / 2 - 1];
        if (lower - self.y).abs() < (upper - self.y).abs() {
            lower
        } else {
            upper
        }
    }
}

/// Karaoke mode's stage between frames.
#[derive(Default)]
pub struct Stage {
    /// Its text, by face.
    faces: Faces,
    atlas: Atlas,
    sparks: std::collections::VecDeque<Spark>,
    arrow: Arrow,
    /// The arrow's comet tail, and the stardust owed to it.
    comet: comet::Comet,
    dust: f32,
    seed: u32,
    /// Sparks owed to the next frame (a share of one).
    owed: f32,
    /// The streak: its shine, the words scored ([`super::streak`]).
    streak: super::streak::Streak,
    /// Where it is in the singing's moments.
    cursor: crate::core::singing::Cursor,
    /// The multiplier shown, how much it shows, and how long since it
    /// went up.
    badge: (u32, f32, f32),
    /// Where the capitals' middle is in text of a face and size
    /// ([`Stage::cap_middle`]).
    caps: HashMap<(Face, u32), f32>,
    /// The song's time at the frame before: what moves goes by the song.
    before: Option<f64>,
    /// The words scored bursting, handed to the GPU ([`super::particles`]).
    sprays: Vec<super::particles::Spray>,
    /// The shine's meter ([`meter`]).
    meter: meter::Meter,
    /// The bursts that are light (the angels, the jackpots) showing.
    lights: Vec<super::particles::Spray>,
    /// What the result screen keeps between frames ([`result::Kept`]).
    kept: result::Kept,
    /// What takes clicks as drawn last ([`Stage::targets`]).
    targets: Vec<([f32; 4], crate::core::mouse::Target)>,
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
            demo: state.karaoke.demo.held(),
        }
    }
}

/// Whether the stage shows for `state`: Karaoke mode, its notes shown, the
/// song in it with notes made; or a song's result shown in it.
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

/// `picture` without what the stage draws instead: its lyric lines (and
/// their notes); while a result shows, all its lines (the result's words)
/// and the result.
pub fn without_lyrics(picture: &KaraokePicture) -> KaraokePicture {
    let keep = |l: &&LyricLine| l.id.is_none() && picture.result.is_none();
    KaraokePicture {
        lines: picture.lines.iter().filter(keep).cloned().collect(),
        before: picture.before.iter().filter(keep).cloned().collect(),
        result: None,
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
        out.push(Instance {
            quad: [x0, y0, x0 + w as f32 * scale, y0 + h as f32 * scale],
            shape: [0.0; 4],
            color: rest,
            color2: filled,
            params: [kind::TEXT, split, 0.0, 0.0],
            uv: [u as f32, v as f32, (u + w) as f32, (v + h) as f32],
            strain: [0.0; 4],
        });
        out.last_mut()
    }

    /// How far down from the top of text in `face` made at `px` for the
    /// atlas ([`BigText::coverage`]) the middle of its capitals is, between
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
        let (x, _, w, _) = area;
        let singing = song.singing;
        let game = song.game;
        let p = lane.pill;
        let behind = lane.behind();
        let (from, to) = (now - behind - 0.5, now + AHEAD + 0.5);

        // The lane: a band of smoked glass (the bubbles read over any
        // cover), the playhead across it.
        let lane_half = ((lane.band.1 - lane.band.0) / 2.0 + p * 0.55, w * 0.4875);
        let lane_mid = ((x + w * 0.5), (lane.band.0 + lane.band.1) / 2.0);
        out.push(pill(
            lane_mid,
            (lane_half.1, lane_half.0),
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
        // The streak's rainbow bands from the playhead, behind the tubes.
        let since = singing.moments_since(&mut self.cursor);
        let step = self.streak.step(singing, game, &since, clock);
        let lane_box = (lane_mid, (lane_half.1, lane_half.0));
        let bands = rainbow::bands_of((lane.head, lane_mid.1), lane_box, &step.glow, clock);
        out.extend(bands);

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
        if self.arrow.shown > 0.5 {
            let at = (lane.head, self.arrow.y);
            self.dust += comet::DUST * song_dt;
            let n = self.dust.floor() as usize;
            self.dust -= n as f32;
            self.emit(n, at, arrow_color, p * 0.26, 0.12, p);
            self.owed += if hitting { 40.0 } else { 0.0 } * song_dt;
            let n = self.owed.floor() as usize;
            self.owed -= n as f32;
            self.emit(n, at, arrow_color, p * 0.5, 1.0, p);
        } else {
            (self.owed, self.dust) = (0.0, 0.0);
        }
        // A word scored (once): a burst where it hit; each level it took the
        // streak up, a rainbow burst from the playhead, the multiplier popping.
        let at = |end: f64| (lane.x(end).max(x), self.arrow.y);
        self.sprays
            .extend(step.sprays(at, tip, (p, clock, lane.pps)));
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
            let swirl = rainbow::swirl_of((lane.head, self.arrow.y), p * 0.9, &step.glow, clock);
            out.extend(swirl);
        }

        // The multiplier, over the playhead: popping (in rainbow) as it goes
        // up; the streak beside it, over the light (Amazing's flames blaze
        // up there).
        let mut top = Vec::new();
        if song.demo {
            self.demo_tag(&mut top, area);
        }
        let m = if game { singing.multiplier() } else { 1 };
        let shown_target = if m > 1 { 1.0 } else { 0.0 };
        if m > 1 {
            if m != self.badge.0 {
                self.badge.2 = 0.0;
            }
            self.badge.0 = m;
        }
        self.badge.1 += (shown_target - self.badge.1) * (1.0 - (-dt / 0.12).exp());
        self.badge.2 += dt;
        if self.badge.1 > 0.01 && self.badge.0 > 1 {
            let pop = 1.0 + 0.45 * (-self.badge.2 * 7.0).exp();
            let a = self.badge.1;
            let px = (p * 1.05).round();
            let label = format!("×{}", self.badge.0);
            let tw = self.measure(Face::SHOUT, &label, px) * pop;
            let cy = lane.band.0 - p * 1.55;
            out.push(disc(
                (lane.head, cy),
                0.0,
                [0.0; 4],
                rgba(GOLD, (0.25 + 0.1 * self.badge.0 as f32) * a * pop),
                p * 1.4 * pop,
            ));
            let number = self.text(
                &mut out,
                Face::SHOUT,
                &label,
                (px, pop),
                (lane.head - tw / 2.0, cy - px * 0.62 * pop),
                f32::NEG_INFINITY,
                (rgba(GOLD_TOP, a), rgba(GOLD_TOP, a)),
            );
            // In rainbow while the singing shines.
            number
                .filter(|_| step.glow.shows(clock as f32))
                .into_iter()
                .for_each(|t| rainbow::in_rainbow(t, &step.glow));
            if game && singing.tally.streak > 0 {
                let small = (p * 0.36).round().max(9.0);
                let streak = format!("{} in a row", singing.tally.streak);
                let ty = cy - self.cap_middle(Face::NUMBER, small) + COVER_PAD as f32;
                let (sx, sw) = (
                    lane.head + tw / pop / 2.0 + p * 0.4,
                    self.measure(Face::NUMBER, &streak, small),
                );
                // On a dark pill of its own, to read over the flames.
                top.push(pill(
                    (sx + sw / 2.0, cy),
                    (sw / 2.0 + small * 0.45, small * 0.7),
                    [0.04, 0.03, 0.09, 0.6 * a],
                    [0.0; 4],
                    0.0,
                    0.0,
                    0.0,
                ));
                self.text(
                    &mut top,
                    Face::NUMBER,
                    &streak,
                    (small, 1.0),
                    (sx, ty),
                    f32::NEG_INFINITY,
                    (rgba(WHITE, 0.8 * a), rgba(WHITE, 0.8 * a)),
                );
            }
        }

        // The line sung and the next ([`lines`]).
        self.lines(&mut out, picture, area);

        // The shine: the meter, the tiers' looks, the song's progress.
        let mut drawn = meter::Drawn {
            out,
            top,
            ..Default::default()
        };
        let at = meter::Where {
            area,
            lane,
            lane_box,
            now,
            clock,
            gone: picture.controls.map_or(0.0, |fade| {
                crate::core::karaoke::controls::handover(fade).0 as f32
            }),
        };
        self.shine(&mut drawn, (singing, game, song.length), &at, &step);
        let (out, top, mut lights, mut light_sprays) =
            (drawn.out, drawn.top, drawn.lights, drawn.light_sprays);
        lights.append(&mut shots.lights);
        self.lights.retain(|s| s.alive(clock));
        self.lights.append(&mut light_sprays);

        self.sprays.retain(|s| s.alive(clock));
        sprays.extend_from_slice(&self.sprays);
        StageFrame {
            instances: out,
            top,
            uploads: std::mem::take(&mut self.atlas.uploads),
            paths,
            sprays,
            clock: clock as f32,
            lights,
            light_sprays: self.lights.clone(),
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
        let mut ends = Vec::new();
        for fps in [60.0f32, 144.0] {
            let mut arrow = Arrow::default();
            arrow.step(Some(100.0), 1.0 / fps);
            let mut ys = Vec::new();
            for _ in 0..(0.5 * fps).round() as usize {
                arrow.step(Some(200.0), 1.0 / fps);
                ys.push(arrow.y);
            }
            let at = |s: f32| ys[(s * fps).round() as usize - 1];
            // Put for its first moments (the new pitch not yet the middle
            // of those heard), most of the way within 0.3 s, never past it,
            // and still by half a second.
            assert!(at(0.01) < 110.0, "{fps}: {}", at(0.01));
            assert!(at(0.3) > 190.0, "{fps}: {}", at(0.3));
            let most = ys.iter().copied().fold(0.0, f32::max);
            assert!(most <= 200.01, "{fps}: {most}");
            assert!((at(0.5) - 200.0).abs() < 0.5, "{fps}: {}", at(0.5));
            ends.push(at(0.3));
        }
        assert!((ends[0] - ends[1]).abs() < 3.0, "{ends:?}");
        // Stray pitches, a few in a row, do not move it at all.
        let mut arrow = Arrow::default();
        for _ in 0..20 {
            arrow.step(Some(100.0), 0.004);
        }
        for _ in 0..3 {
            arrow.step(Some(300.0), 0.004);
        }
        let mut most: f32 = 0.0;
        for _ in 0..50 {
            arrow.step(Some(100.0), 0.004);
            most = most.max((arrow.y - 100.0).abs());
        }
        assert!(most < 0.01, "{most}");
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
        let mut stage = Stage::default();
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
        let mut stage = Stage::default();
        let m = stage.cap_middle(Face::LINE, 40.0);
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
        assert_eq!(a.made, 1.0);
        let b = atlas.place(&mut big, "world", 40.0).unwrap().rect;
        assert!(b.0 >= a.rect.0 + a.rect.2);
        // Full: started again from the top.
        atlas.pen = (0, ATLAS - 2, 0);
        let c = atlas.place(&mut big, "again", 40.0).unwrap().rect;
        assert_eq!((c.0, c.1), (0, 0));
        let gone = (Face::Main, "hello".to_owned(), 160);
        assert!(!atlas.places.contains_key(&gone));
    }

    /// A lyric line wider than the atlas at its size (a large screen's)
    /// is made smaller to fit, not left out.
    #[test]
    fn text_wider_than_the_atlas_is_made_smaller() {
        let mut big = BigText::default();
        let mut atlas = Atlas::default();
        let line = "Makes you feel like every corner is black and white tonight";
        let p = atlas.place(&mut big, line, 120.0).unwrap();
        assert!(p.made < 1.0, "{p:?}");
        assert!(p.rect.2 <= ATLAS && p.rect.2 > ATLAS / 2, "{p:?}");
    }
}
