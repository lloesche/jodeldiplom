//! The lyrics set in the middle of the screen ([`Stage::lyrics`]): the
//! line before, the line sung (large, filling as it is sung, in the
//! accent), the lines coming (smaller, dimmer), sizes by what each line
//! is (smaller where too wide), stacked and centred; moving into their
//! places as the lines change (their size and colour too), new ones rising
//! in as they fade in, those gone rising away as they fade out. Without
//! the song's lane (the notes off, or a song without notes) these are all
//! the lines; with it only those that are no lyric (the song's title and
//! artist as it starts, a note in a long pause, a warning), the lane's
//! own lines at the bottom ([`super::lines`]).
//!
//! Under the line being sung, where its notes are known: a bar under the
//! letters each is sung on, as high as it goes against the line's middle,
//! those sung in the notes' colour, those to come darker; the singer's
//! pitch along them, sparkles shed behind it and a dot where they are now;
//! each word's bubble's liquid over its bars (straining, pumping,
//! overfilling and bursting as the stage's bubbles do), and a rainbow edge
//! on the bar under the word sung while the singing shines, a burst from
//! it a step up.

use super::super::big_text::{BigText, Face};
use super::super::fx::rainbow::rim_of;
use super::super::fx::step_up::burst;
use super::super::particles::{Kind, Spray};
use super::super::pixels::mix;
use super::super::sparks::noise;
use super::super::stage_gpu::{Instance, Pour, Shape, Tint, ATLAS};
use super::score::rgba8;
use super::{disc, pill, segment, Song, Stage};
use crate::view::karaoke::{
    line_color, KaraokePicture, LineKind, LyricLine, HIT, NOTE_AHEAD, SINGER,
};
use crate::view::palette;

/// The room under the line being sung for its notes, as a share of its
/// size.
const NOTES_ROOM: f32 = 1.45;
/// Where the notes go in it: their highest's top, from the text's top...
const NOTES_TOP: f32 = 1.3;
/// ... and their lowest's bottom.
const NOTES_BOTTOM: f32 = 2.5;
/// A note's thickness, a share of the size.
const NOTE_THICK: f32 = 0.09;

/// The singer's last frames that shed sparkles behind them (0.5 s).
const SPARKLE_FRAMES: usize = 50;
/// The share of those frames that shed one.
const SPARKLE_SHARE: f32 = 0.4;

/// The widest run of text put into the atlas at once, pixels: a line
/// wider than that goes in word by word, so a long line on a large screen
/// is drawn at its own size.
const RUN: f32 = (ATLAS / 2) as f32;

/// The line sung's size in a picture `height` pixels high.
pub fn big_of(height: f32) -> f32 {
    (height / 11.0).clamp(20.0, 160.0).round()
}

/// How far down a line's text is from the next one's: its size and more.
pub fn gap(px: f32) -> f32 {
    px * 1.45
}

/// Whether `line` gets room for notes under it: a lyric being sung.
fn noted(line: &LyricLine) -> bool {
    line.kind == LineKind::Current && line.id.is_some()
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// Where a line goes: its size and the top of its text.
#[derive(Clone, Copy)]
struct Slot {
    px: f32,
    y: f32,
}

/// The lines' sizes (by what they are, smaller where too wide) and places,
/// stacked and centred in `room` rows of pixels from `top`, `width` wide.
fn slots(
    text: &mut BigText,
    lines: &[&LyricLine],
    width: f32,
    (top, room): (f32, f32),
    big: f32,
) -> Vec<Slot> {
    let widest = width * 0.92;
    let sizes: Vec<f32> = lines
        .iter()
        .map(|l| {
            let px = match l.kind {
                LineKind::Current => big,
                LineKind::Next | LineKind::Previous => big * 0.62,
                LineKind::Info => big * 0.42,
                LineKind::Warning => big * 0.36,
            };
            let w = text.measure(&l.text, px);
            if w > widest && w > 0.0 {
                (px * widest / w).max(10.0).round()
            } else {
                px.round()
            }
        })
        .collect();
    let below = |l: &LyricLine, px: f32| if noted(l) { px * NOTES_ROOM } else { 0.0 };
    let total: f32 = lines
        .iter()
        .zip(&sizes)
        .map(|(l, &px)| gap(px) + below(l, px))
        .sum();
    let mut y = top + (room - total).max(0.0) / 2.0;
    lines
        .iter()
        .zip(sizes)
        .map(|(l, px)| {
            let slot = Slot {
                px,
                y: y + (gap(px) - px) / 2.0,
            };
            y += gap(px) + below(l, px);
            slot
        })
        .collect()
}

/// How far into `line` (at `px`, glyphs made at `base`) character `at` is,
/// in pixels (a fraction through one, through its width).
fn x_of_char(text: &mut BigText, line: &str, base: f32, px: f32, at: f32) -> f32 {
    let at = at.max(0.0);
    let whole = at.floor() as usize;
    let before: String = line.chars().take(whole).collect();
    let x = text.measure_from(&before, base, px);
    match line.chars().nth(whole) {
        Some(c) => x + (at - whole as f32) * text.measure_from(&c.to_string(), base, px),
        None => x,
    }
}

/// A note's bar from `x0` to `x1` under text `px` high: a hair apart from
/// the next.
fn span(px: f32, (x0, x1): (f32, f32)) -> (f32, f32) {
    let gap = (px * 0.04).min((x1 - x0) * 0.2);
    (x0 + gap, x1 - gap)
}

/// A note's bar under text `px` high whose top is at `y`, `height` high
/// (-1 to 1): its top, and how thick it is.
fn bar(px: f32, y: f32, height: f32) -> (f32, f32) {
    let thick = (px * NOTE_THICK).max(3.0);
    let (top, bottom) = (px * NOTES_TOP, px * NOTES_BOTTOM - thick);
    (y + bottom - (height + 1.0) / 2.0 * (bottom - top), thick)
}

/// Where each of the lines `now` was among the lines `before` (the same
/// lyric line: the same id), and which of `before` are gone from `now`:
/// how lines move as they change.
pub(super) fn moved(now: &[&LyricLine], before: &[&LyricLine]) -> (Vec<Option<usize>>, Vec<usize>) {
    let same = |a: &LyricLine, b: &LyricLine| a.id.is_some() && a.id == b.id;
    let was = now
        .iter()
        .map(|l| before.iter().position(|b| same(l, b)))
        .collect();
    let gone = (0..before.len())
        .filter(|&j| !now.iter().any(|l| same(l, before[j])))
        .collect();
    (was, gone)
}

/// A line as it is set.
struct Set<'a> {
    /// The line.
    pub line: &'a LyricLine,
    /// The size its glyphs are made at.
    pub base: f32,
    /// The size it is drawn at.
    pub px: f32,
    /// Its text's top-left.
    pub at: (f32, f32),
    /// Where its fill ends, pixels across.
    pub fill_x: f32,
    /// Its colours: sung, the rest.
    pub colors: ([u8; 3], [u8; 3]),
    /// How much of it shows.
    pub alpha: f32,
}

impl Set<'_> {
    /// How far into its text character `at` is (a fraction through one),
    /// in pixels, its glyphs in `text` (the lines' face).
    pub fn x_of(&self, text: &mut BigText, at: f32) -> f32 {
        x_of_char(text, &self.line.text, self.base, self.px, at)
    }
}

/// `picture`'s lines that `keep` keeps (those it moves from too), set in
/// the window's `area` (left, top, width, height), their glyphs in `text`
/// (the lines' face): centred, moving into their places while they
/// scroll; none that does not show.
fn set<'a>(
    text: &mut BigText,
    picture: &'a KaraokePicture,
    keep: impl Fn(&LyricLine) -> bool,
    (x, y, w, h): (f32, f32, f32, f32),
) -> Vec<Set<'a>> {
    let big = big_of(h);
    let lines: Vec<&LyricLine> = picture.lines.iter().filter(|l| keep(l)).collect();
    let before: Vec<&LyricLine> = picture.before.iter().filter(|l| keep(l)).collect();
    let now = slots(text, &lines, w, (y, h), big);
    let moving = picture.moved < 1.0 && !before.is_empty();
    let then = if moving {
        slots(text, &before, w, (y, h), big)
    } else {
        Vec::new()
    };
    let m = picture.moved.clamp(0.0, 1.0) as f32;
    let faded = |kind: LineKind| match kind {
        LineKind::Previous => 0.5,
        _ => 1.0,
    };
    let mut out = Vec::new();
    let mut put = |text: &mut BigText,
                   line: &'a LyricLine,
                   base: f32,
                   slot: Slot,
                   rest: [u8; 3],
                   alpha: f32| {
        let alpha = alpha * line.alpha.clamp(0.0, 1.0) as f32;
        if alpha <= 0.004 {
            return;
        }
        let width = text.measure_from(&line.text, base, slot.px);
        let left = x + (w - width) / 2.0;
        // The fill to the character sung, measured (the font's letters
        // are not all as wide).
        let chars = line.text.chars().count() as f32;
        let sung = chars * line.fill.clamp(0.0, 1.0) as f32;
        let fill_x = left + x_of_char(text, &line.text, base, slot.px, sung);
        let sung_color = match line.kind {
            LineKind::Current => palette::ACCENT,
            _ => rest,
        };
        out.push(Set {
            line,
            base,
            px: slot.px,
            at: (left, slot.y),
            fill_x,
            colors: (sung_color, rest),
            alpha,
        });
    };
    let (was, gone) = moved(&lines, &before);
    for (k, &line) in lines.iter().enumerate() {
        let slot = now[k];
        let rest = line_color(line.kind);
        match (moving, was[k]) {
            (false, _) => put(text, line, slot.px, slot, rest, faded(line.kind)),
            // From where it was to where it goes, its size and colour too;
            // its glyphs made at the larger size, drawn smaller.
            (true, Some(j)) => {
                let old = then[j];
                let at = Slot {
                    px: lerp(old.px, slot.px, m),
                    y: lerp(old.y, slot.y, m),
                };
                let kind = before[j].kind;
                let color = mix(line_color(kind), rest, m);
                let alpha = lerp(faded(kind), faded(line.kind), m);
                put(text, line, old.px.max(slot.px), at, color, alpha);
            }
            // New: rising into its place as it fades in.
            (true, None) => {
                let at = Slot {
                    y: slot.y + (1.0 - m) * gap(slot.px) * 0.5,
                    ..slot
                };
                put(text, line, slot.px, at, rest, faded(line.kind) * m);
            }
        }
    }
    // Lines that went: rising away as they fade out.
    if moving {
        for j in gone {
            let old = then[j];
            let at = Slot {
                y: old.y - m * gap(old.px) * 0.5,
                ..old
            };
            let line = before[j];
            let alpha = faded(line.kind) * (1.0 - m);
            put(text, line, old.px, at, line_color(line.kind), alpha);
        }
    }
    out
}

/// Which of a picture's lines the stage sets in the middle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Which {
    /// All of them: no lane.
    All,
    /// Those that are no lyric: the lane draws the lyrics.
    Info,
}

impl Which {
    fn keeps(self, line: &LyricLine) -> bool {
        self == Which::All || line.id.is_none()
    }
}

impl Stage {
    /// `text` in `face` (glyphs made at `base`, drawn `px` high) with its
    /// line's top-left at `at`, left of `split` (pixels across) in
    /// `filled`, right of it in `rest`, into `out`: in runs of words no
    /// wider than [`RUN`] (one where it is narrower), on whole pixels where
    /// it is drawn at the size it is made at, so it is as crisp as its
    /// glyphs.
    pub(super) fn set_text(
        &mut self,
        out: &mut Vec<Instance>,
        (face, text): (Face, &str),
        (base, px): (f32, f32),
        at: (f32, f32),
        split: f32,
        (rest, filled): ([f32; 4], [f32; 4]),
    ) {
        let scale = px / base.max(1.0);
        let whole = (scale - 1.0).abs() < 1e-3;
        let snap = |v: f32| if whole { v.floor() } else { v };
        let big = self.faces.get(face);
        let mut runs: Vec<(String, f32)> = Vec::new();
        let (mut run, mut from, mut pen) = (String::new(), 0.0, 0.0);
        for word in text.split_inclusive(' ') {
            let wide = big.measure(word, base);
            if !run.is_empty() && pen + wide - from > RUN {
                runs.push((std::mem::take(&mut run), from));
                from = pen;
            }
            run.push_str(word);
            pen += wide;
        }
        runs.push((run, from));
        for (run, from) in runs {
            let at = (snap(at.0 + from * scale), snap(at.1));
            let colors = (rest, filled);
            self.text(out, face, &run, (base, scale), at, split, colors);
        }
    }

    /// `picture`'s lines that `which` says set in the middle of the
    /// window's `area` into `out`, with their notes and the singing on
    /// them; their bubbles' liquid into `over` (over everything the stage
    /// draws of the picture) and its bursts into `sprays`. Gives back the
    /// bar under the word being sung (its centre and half size), where
    /// the singing's shine goes ([`Stage::rim_shine`]).
    pub(super) fn lyrics(
        &mut self,
        (out, over, sprays): (&mut Vec<Instance>, &mut Vec<Instance>, &mut Vec<Spray>),
        picture: &KaraokePicture,
        (area, which): ((f32, f32, f32, f32), Which),
    ) -> Option<((f32, f32), (f32, f32))> {
        let face = self.faces.get(Face::LINE);
        let keep = |l: &LyricLine| which.keeps(l) && picture.result.is_none();
        let sets = set(face, picture, keep, area);
        let mut rim = None;
        for s in &sets {
            let (sung, rest) = (rgba8(s.colors.0, s.alpha), rgba8(s.colors.1, s.alpha));
            let line = (Face::LINE, s.line.text.as_str());
            self.set_text(out, line, (s.base, s.px), s.at, s.fill_x, (rest, sung));
            self.notes(out, s);
            rim = self.bubbles((over, sprays), s).or(rim);
        }
        rim
    }

    /// The singing's shine on the bar under the word sung (`rim`: its
    /// centre and half size, where there is one) for `song` at `heard`
    /// seconds into it: its rainbow edge into `over`, and a burst from it
    /// each step up into `sprays`. Stepped only where the stage draws no
    /// lane (the lane's own effects step it there).
    pub(super) fn rim_shine(
        &mut self,
        (over, sprays): (&mut Vec<Instance>, &mut Vec<Spray>),
        (song, heard): (&Song, f64),
        rim: Option<((f32, f32), (f32, f32))>,
    ) {
        let clock = crate::core::lag::sung_clock(heard, song.late);
        let since = song.singing.moments_since(&mut self.cursor);
        let step = self.streak.step(song.singing, song.game, &since, clock);
        if let Some((c, half)) = rim {
            let seed = |k: u32| (clock * 1000.0) as u32 + 7919 * k;
            let ups = (0..step.ups).map(|k| burst(c, half.1 * 6.0, clock, 0.0, seed(k)));
            self.ups.extend(ups);
        }
        self.ups.retain(|s| s.alive(clock));
        sprays.extend_from_slice(&self.ups);
        over.extend(rim.and_then(|(c, half)| rim_of(c, half, (step.glow, 1.0), clock)));
    }

    /// The notes under the line `s` (where it has them) into `out`: their
    /// bars (those burst or bursting gone), the singer's pitch along them,
    /// the sparkles it sheds and the singer now.
    fn notes(&mut self, out: &mut Vec<Instance>, s: &Set) {
        let line = s.line;
        if line.notes.is_empty() || s.alpha <= 0.004 {
            return;
        }
        let (px, at, alpha) = (s.px, s.at, s.alpha);
        let face = self.faces.get(Face::LINE);
        let begun = |k: usize| line.liquid.iter().any(|b| b.has(k) && b.begun());
        let thick = bar(px, at.1, 0.0).1;
        let y_of = |height: f32| bar(px, at.1, height).0;
        let (sung_c, ahead) = (crate::view::lanes::NOTE, NOTE_AHEAD);
        // Each bar square, the part sung in the notes' colour, the rest
        // darker.
        for (_, n) in line.notes.iter().enumerate().filter(|&(k, _)| !begun(k)) {
            let (x0, x1) = span(px, (s.x_of(face, n.from), s.x_of(face, n.to)));
            let (x0, x1) = ((at.0 + x0).round(), (at.0 + x1).round());
            let y = y_of(n.height);
            let (y0, y1) = (y.round(), (y + thick).round());
            let split = s.fill_x.ceil().clamp(x0, x1);
            for ((a, b), color) in [((x0, split), sung_c), ((split, x1), ahead)] {
                if b - a >= 0.5 {
                    let c = ((a + b) / 2.0, (y0 + y1) / 2.0);
                    let half = ((b - a) / 2.0, (y1 - y0) / 2.0);
                    out.push(pill(c, half, rgba8(color, alpha), [0.0; 4], 0.0, 0.0, 0.5));
                }
            }
        }
        // The singer's pitch: a thin line along their frames (apart where
        // they were not singing), then a dot where they are now.
        let trail: Vec<(f32, f32)> = (line.trail.iter())
            .map(|&(c, h)| (at.0 + s.x_of(face, c), y_of(h) + thick / 2.0))
            .collect();
        let wide = (thick * 0.45).max(2.0) / 2.0;
        let singer = rgba8(SINGER, alpha * 0.9);
        for pair in trail.windows(2) {
            if (pair[1].0 - pair[0].0).abs() <= px * 0.6 {
                out.push(segment(pair[0], pair[1], wide, singer, [0.0; 4], 0.0));
            }
        }
        let Some((c, height, way)) = line.voice else {
            return;
        };
        // Sparkles on, white hot near the singer: gold on the note, coral
        // off it; the last of their frames shed them, which drift back and
        // apart, shrink, twinkle and fade.
        let glint = if way == 0 {
            mix([255, 255, 255], HIT, 0.45)
        } else {
            mix([255, 255, 255], SINGER, 0.55)
        };
        let n = trail.len();
        for k in 0..SPARKLE_FRAMES.min(n) {
            let i = n - 1 - k;
            let r = |j: u32| noise(i as u32 * 5 + j);
            if r(0) > SPARKLE_SHARE {
                continue;
            }
            let age = k as f32 / SPARKLE_FRAMES as f32;
            let (tx, ty) = trail[i];
            let sx = tx - age * px * (0.3 + 0.6 * r(1));
            let sy = ty + (r(2) - 0.5) * age * px * 1.1;
            let size = px * (0.34 - 0.22 * age) * (0.55 + 0.6 * r(3));
            let twinkle = 0.55 + 0.45 * (age * 28.0 + r(4) * std::f32::consts::TAU).sin();
            let a = (1.0 - age).powf(1.4) * twinkle * alpha;
            let m = size + 1.0;
            let quad = [sx - m, sy - m, sx + m, sy + m];
            let spark = Instance::of(Shape::Spark, quad, [sx, sy, size, 0.0]);
            out.push(spark.colors(rgba8(glint, a), [0.0; 4]));
        }
        let at_now = (at.0 + s.x_of(face, c), y_of(height) + thick / 2.0);
        let core = rgba8(SINGER, alpha);
        let glow = rgba8(glint, 0.45 * alpha);
        out.push(disc(at_now, thick * 1.2, core, glow, thick * 1.8));
    }

    /// The bubbles of the line `s` over their bars into `over`: the
    /// liquid of each not torn, in gold, its burst into `sprays` while it
    /// bursts. Gives back the bar under the word being sung (its centre
    /// and half size), the last begun on the line being sung and not
    /// burst, where the singing's shine goes.
    fn bubbles(
        &mut self,
        (over, sprays): (&mut Vec<Instance>, &mut Vec<Spray>),
        s: &Set,
    ) -> Option<((f32, f32), (f32, f32))> {
        let line = s.line;
        let (px, at) = (s.px, s.at);
        let face = self.faces.get(Face::LINE);
        // Its colour at its top, a little deeper at its bottom.
        let tint = |thick: f32| Tint {
            top: rgba8(HIT, 1.0),
            bottom: rgba8(mix(HIT, [0, 0, 0], 0.2), 1.0),
            glow: thick,
        };
        for b in &line.liquid {
            // Each part over its note's bar: where, and how thick.
            let bars: Vec<((f32, f32), (f32, f32))> = (b.parts.iter())
                .map(|p| {
                    let (x0, x1) = (s.x_of(face, p.from), s.x_of(face, p.to));
                    let (x0, x1) = span(px, (at.0 + x0, at.0 + x1));
                    ((x0, x1), bar(px, at.1, p.height))
                })
                .collect();
            let Some(&(_, (_, thick))) = bars.first() else {
                continue;
            };
            // Held full, it pumps (thicker as it grows) and bursts the
            // bigger for it.
            let pump = b.pump() as f32;
            if b.bursting().is_some() {
                // All of it at once, from round its bars.
                let top = bars.iter().map(|(_, y)| y.0).fold(f32::MAX, f32::min);
                let bottom = bars.iter().map(|(_, y)| y.0 + y.1).fold(f32::MIN, f32::max);
                let (left, right) = (bars[0].0 .0, bars[bars.len() - 1].0 .1);
                let top_color = tint(thick).top;
                sprays.push(Spray {
                    at: ((left + right) / 2.0, (top + bottom) / 2.0),
                    half: ((right - left) / 2.0, pump * (bottom - top) / 2.0),
                    start: b.pops_at.unwrap_or_default() as f32,
                    drift: 0.0,
                    size: px * (1.0 + super::super::pop::BIGGER * b.size() as f32),
                    seed: b.seed,
                    rgb: [top_color[0], top_color[1], top_color[2]],
                    kind: Kind::Burst,
                });
            }
            if b.torn() {
                continue;
            }
            // Its ends as square as its bars'.
            let strain = super::super::pop::strain(b);
            for (p, &((x0, x1), (y, thick))) in b.parts.iter().zip(&bars) {
                let (c, half) = (
                    ((x0 + x1) / 2.0, y + thick / 2.0),
                    ((x1 - x0) / 2.0, pump * thick / 2.0),
                );
                let pour = Pour {
                    span: (x0, x0 + (x1 - x0) * p.level as f32),
                    front: p.level < 1.0,
                    jumped: p.jumped.max(-1e6) as f32,
                    tint: tint(thick),
                };
                over.extend(Instance::liquid(c, half, thick * 0.2, &pour, &strain));
            }
        }
        if line.kind != LineKind::Current {
            return None;
        }
        let sung = line.fill as f32 * line.text.chars().count() as f32;
        let burst = |k: usize| line.liquid.iter().any(|b| b.has(k) && b.begun());
        let mut notes = line.notes.iter().enumerate();
        let (_, n) = notes.rfind(|(k, n)| n.from <= sung && !burst(*k))?;
        let (x0, x1) = span(px, (at.0 + s.x_of(face, n.from), at.0 + s.x_of(face, n.to)));
        let (y, thick) = bar(px, at.1, n.height);
        Some((
            ((x0 + x1) / 2.0, y + thick / 2.0),
            ((x1 - x0) / 2.0, thick / 2.0),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gui::big_text::COVER_PAD;

    fn lyric(text: &str, fill: f64, kind: LineKind, id: Option<usize>) -> LyricLine {
        LyricLine {
            text: text.into(),
            fill,
            kind,
            id,
            alpha: 1.0,
            notes: Vec::new(),
            hits: Vec::new(),
            liquid: Vec::new(),
            trail: Vec::new(),
            voice: None,
        }
    }

    fn picture(lines: Vec<LyricLine>) -> KaraokePicture {
        KaraokePicture {
            bar: 5.0 / 24.0,
            lines,
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: None,
            result: None,
        }
    }

    /// The lines set as the processor paints them: centred, the line sung
    /// largest, its fill where its characters sung end; on the stage each
    /// a run of text at its own size on whole pixels, the sung part in the
    /// accent; with the lane, only the lines that are no lyric.
    #[test]
    fn the_lines_are_set_in_the_middle_crisp_and_filled() {
        let area = (0.0, 0.0, 1600.0, 900.0);
        let p = picture(vec![
            lyric("the line before", 1.0, LineKind::Previous, Some(0)),
            lyric("the line sung now", 0.5, LineKind::Current, Some(1)),
            lyric("♪", 0.0, LineKind::Info, None),
        ]);
        let mut big = BigText::of(Face::LINE);
        let sets = set(&mut big, &p, |_| true, area);
        assert_eq!(sets.len(), 3);
        let sung = &sets[1];
        assert_eq!(sung.px, big_of(900.0));
        assert!(sets[0].px < sung.px && sets[2].px < sets[0].px);
        let width = big.measure(&sung.line.text, sung.px);
        assert!((sung.at.0 + width / 2.0 - 800.0).abs() < 0.5);
        assert!(sung.fill_x > sung.at.0 && sung.fill_x < sung.at.0 + width);
        assert_eq!(sung.colors, (palette::ACCENT, palette::TEXT));
        let mut stage = Stage::default();
        let (mut out, mut over, mut sprays) = (Vec::new(), Vec::new(), Vec::new());
        let into = (&mut out, &mut over, &mut sprays);
        assert_eq!(stage.lyrics(into, &p, (area, Which::All)), None);
        assert_eq!(out.len(), 3, "{out:?}");
        for i in &out {
            assert!(i.is(Shape::Text));
            assert_eq!((i.quad[0].fract(), i.quad[1].fract()), (0.0, 0.0));
            assert!((i.quad[2] - i.quad[0] - (i.uv[2] - i.uv[0])).abs() < 1e-3);
        }
        assert_eq!(out[1].params[1], sung.fill_x);
        assert_eq!(out[1].color2, rgba8(palette::ACCENT, 1.0));
        let mut info = Vec::new();
        let only = (&mut info, &mut Vec::new(), &mut Vec::new());
        stage.lyrics(only, &p, (area, Which::Info));
        assert_eq!(info.len(), 1);
    }

    /// A line wider than the atlas holds at once goes in word by word,
    /// each run at its own size, side by side as the whole line.
    #[test]
    fn a_long_line_goes_in_runs_at_its_own_size() {
        let mut stage = Stage::default();
        let mut out = Vec::new();
        let line = "Makes you feel like every corner is black and white tonight";
        let white = [1.0; 4];
        let text = (Face::LINE, line);
        stage.set_text(
            &mut out,
            text,
            (160.0, 160.0),
            (10.5, 20.5),
            0.0,
            (white, white),
        );
        assert!(out.len() > 1, "{}", out.len());
        let whole = stage.faces.get(Face::LINE).measure(line, 160.0);
        let pad = COVER_PAD as f32;
        assert_eq!(out[0].quad[0], 10.0 - pad);
        let end = out.last().unwrap().quad[2] - pad;
        assert!((end - (10.0 + whole)).abs() < 3.0, "{end} {whole}");
        for i in &out {
            assert!((i.quad[2] - i.quad[0] - (i.uv[2] - i.uv[0])).abs() < 1e-3);
            let high = 160.0 * crate::gui::big_text::LINE + 2.0 * pad;
            assert!(i.quad[3] - i.quad[1] <= high + 1.0);
        }
    }
}
