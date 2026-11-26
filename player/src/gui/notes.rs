//! The notes under the line being sung in Karaoke mode's picture
//! ([`super::karaoke`]), painted on the processor: a bar under the letters
//! each is sung on, as high as it goes against the line's middle; those
//! sung already in the notes' colour, those to come darker; the singer's
//! pitch along them as a line, sparkles shed behind it and a dot where
//! they are now. Each word's bubble's liquid, its strain (held full,
//! pumping thicker) and its overfilling are drawn by the GPU over its
//! notes' bars, as the stage's
//! bubbles are ([`Over`], [`super::stage_gpu::Instance::liquid`]), its
//! burst too ([`super::particles`]); a word's bars whose bubble bursts are
//! no longer painted from the moment it starts to (the picture changes
//! once). Without a GPU
//! (`Over::gpu` off) the bars are painted filled flat, gold as far as
//! their level (a paler gold where close), and a bubble burst is simply
//! gone.

use super::karaoke::{rgb, NOTES_BOTTOM, NOTES_TOP, NOTE_THICK};
use super::particles::{Kind, Spray};
use super::raster::{mix, over, PixelRect, Target};
use super::sparks::{noise, sparkle};
use super::stage_gpu::{Instance, Pour, Tint};
use crate::view::karaoke::{LineKind, LyricLine, HIT, NOTE_AHEAD, SINGER};

/// The notes under a line and the singing on them, in pixels from the
/// text's left.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SungOp {
    /// Each note's left and right, and its height (-1 to 1).
    notes: Vec<(f32, f32, f32)>,
    /// Where they are filled (painted flat, without a GPU): left and right.
    hits: Vec<(f32, f32)>,
    /// The singer's pitch along them: where and how high.
    trail: Vec<(f32, f32)>,
    /// The singer now: where, how high, and which way to go (0 on the
    /// note).
    voice: Option<(f32, f32, i8)>,
}

impl SungOp {
    /// `line`'s notes and the singing on them, at text `px` high, `x_of`
    /// giving where a character (a fraction through one) is; their fill
    /// painted only without a GPU (`gpu` off); none without notes.
    pub(super) fn of(
        line: &LyricLine,
        px: f32,
        gpu: bool,
        mut x_of: impl FnMut(f32) -> f32,
    ) -> Option<SungOp> {
        if line.notes.is_empty() {
            return None;
        }
        // Those burst or bursting are gone from the picture.
        let begun = |k: usize| line.liquid.iter().any(|b| b.has(k) && b.begun());
        let notes = line.notes.iter().enumerate().filter(|&(k, _)| !begun(k));
        let notes = notes
            .map(|(_, n)| {
                let (x0, x1) = span(px, (x_of(n.from), x_of(n.to)));
                (x0, x1, n.height)
            })
            .collect();
        let hits = line.hits.iter().filter(|_| !gpu);
        Some(SungOp {
            notes,
            hits: hits.map(|&(a, b)| (x_of(a), x_of(b))).collect(),
            trail: line.trail.iter().map(|&(at, h)| (x_of(at), h)).collect(),
            voice: line.voice.map(|(at, h, way)| (x_of(at), h, way)),
        })
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

/// What the GPU draws over the picture ([`super::stage_gpu::with_picture`]):
/// its bubbles' liquid (with a GPU, `gpu`) and their bursts.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Over {
    /// Whether there is a GPU to draw it: the bars' fill is not painted.
    pub gpu: bool,
    /// The bubbles' liquid, as the stage's.
    pub bubbles: Vec<Instance>,
    /// The bubbles bursting.
    pub sprays: Vec<Spray>,
    /// The bar under the word being sung (its centre and half size), where
    /// a streak's rainbow edge goes ([`super::streak::Streak::picture`]).
    pub rim: Option<((f32, f32), (f32, f32))>,
}

impl Over {
    /// Nothing yet, for a window with a GPU (`gpu`) or without.
    pub fn new(gpu: bool) -> Over {
        Over {
            gpu,
            ..Over::default()
        }
    }

    /// The bubbles of `line`, over their bars under text `px` high with its
    /// top-left at `at`, `x_of` giving where a character (a fraction
    /// through one) is across from it: the liquid of each not torn (with a
    /// GPU), in gold, a paler gold where only close; the burst of each
    /// bursting.
    pub(super) fn add(
        &mut self,
        line: &LyricLine,
        at: (f32, f32),
        px: f32,
        mut x_of: impl FnMut(f32) -> f32,
    ) {
        // Its colour at its top, a little deeper at its bottom.
        let tint = |c: ratatui::style::Color, thick: f32| {
            let opaque =
                |c: [u8; 3]| [0, 1, 2, 3].map(|i| c.get(i).map_or(1.0, |&v| f32::from(v) / 255.0));
            Tint {
                top: opaque(rgb(c)),
                bottom: opaque(mix(rgb(c), [0, 0, 0], 0.2)),
                glow: thick,
            }
        };
        for b in &line.liquid {
            // Each part over its note's bar: where, and how thick.
            let bars: Vec<((f32, f32), (f32, f32))> = (b.parts.iter())
                .map(|p| {
                    let (x0, x1) = span(px, (at.0 + x_of(p.from), at.0 + x_of(p.to)));
                    let (y, thick) = bar(px, at.1, p.height);
                    ((x0, x1), (y, thick))
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
                let tint = tint(HIT, thick);
                self.sprays.push(Spray {
                    at: ((left + right) / 2.0, (top + bottom) / 2.0),
                    half: ((right - left) / 2.0, pump * (bottom - top) / 2.0),
                    start: b.pops_at.unwrap_or_default() as f32,
                    drift: 0.0,
                    size: px * (1.0 + super::pop::BIGGER * b.size() as f32),
                    seed: b.seed,
                    rgb: [tint.top[0], tint.top[1], tint.top[2]],
                    kind: Kind::Burst,
                });
            }
            if !self.gpu || b.torn() {
                continue;
            }
            // Its ends as square as its bars'.
            let strain = super::pop::strain(b);
            for (p, &((x0, x1), (y, thick))) in b.parts.iter().zip(&bars) {
                let (c, half) = (
                    ((x0 + x1) / 2.0, y + thick / 2.0),
                    ((x1 - x0) / 2.0, pump * thick / 2.0),
                );
                let pour = Pour {
                    span: (x0, x0 + (x1 - x0) * p.level as f32),
                    front: p.level < 1.0,
                    jumped: p.jumped.max(-1e6) as f32,
                    tint: tint(HIT, thick),
                };
                self.bubbles
                    .extend(Instance::liquid(c, half, thick * 0.2, &pour, &strain));
            }
        }
        // The bar under the word being sung (the last begun on the line
        // being sung, not burst).
        if line.kind != LineKind::Current {
            return;
        }
        let sung = line.fill as f32 * line.text.chars().count() as f32;
        let burst = |k: usize| line.liquid.iter().any(|b| b.has(k) && b.begun());
        let mut notes = line.notes.iter().enumerate();
        let under = notes.rfind(|(k, n)| n.from <= sung && !burst(*k));
        self.rim = under.map(|(_, n)| {
            let (x0, x1) = span(px, (at.0 + x_of(n.from), at.0 + x_of(n.to)));
            let (y, thick) = bar(px, at.1, n.height);
            (
                ((x0 + x1) / 2.0, y + thick / 2.0),
                ((x1 - x0) / 2.0, thick / 2.0),
            )
        });
    }
}

/// The singer's last frames that shed sparkles behind them (0.5 s).
const SPARKLE_FRAMES: usize = 50;
/// The share of those frames that shed one.
const SPARKLE_SHARE: f32 = 0.4;

/// Draws `sung` under text `px` high with its top-left at `at` (sung up to
/// `fill_x`, pixels across) at `alpha`, only within `clip`.
pub(super) fn draw(
    target: &mut Target,
    sung: &SungOp,
    at: (f32, f32),
    px: f32,
    fill_x: f32,
    alpha: f32,
    clip: PixelRect,
) {
    if alpha <= 0.004 {
        return;
    }
    let (cx, cy, cw, ch) = clip;
    let x1c = (cx + cw).min(target.width);
    let y1c = (cy + ch).min(target.height);
    let thick = bar(px, at.1, 0.0).1;
    // A note's top at `height`.
    let y_of = |height: f32| bar(px, at.1, height).0;
    let (sung_c, ahead, hit) = (rgb(crate::view::lanes::NOTE), rgb(NOTE_AHEAD), rgb(HIT));
    let mut put = |px: usize, py: usize, color: [u8; 3], a: f32| {
        if px < cx || px >= x1c || py < cy || py >= y1c {
            return;
        }
        let i = py * target.width + px;
        target.pixels[i] = over(target.pixels[i], color, a);
    };
    for &(x0, x1, height) in &sung.notes {
        let y = y_of(height);
        let (x0, x1) = (at.0 + x0, at.0 + x1);
        let (px0, px1) = (x0.round().max(0.0) as usize, x1.round().max(0.0) as usize);
        let (py0, py1) = (
            y.round().max(0.0) as usize,
            (y + thick).round().max(0.0) as usize,
        );
        for qx in px0..px1 {
            let off = qx as f32 - at.0;
            let color = match sung.hits.iter().any(|h| off >= h.0 && off < h.1) {
                true => hit,
                false if (qx as f32) < fill_x => sung_c,
                false => ahead,
            };
            for qy in py0..py1 {
                put(qx, qy, color, alpha);
            }
        }
    }

    // The singer's pitch: a thin line along their frames (apart where they
    // were not singing), then a dot where they are now.
    let singer = rgb(SINGER);
    let line = (thick * 0.45).max(2.0);
    for pair in sung.trail.windows(2) {
        let ((xa, ha), (xb, hb)) = (pair[0], pair[1]);
        if (xb - xa).abs() > px * 0.6 {
            continue;
        }
        let (ya, yb) = (y_of(ha) + thick / 2.0, y_of(hb) + thick / 2.0);
        let steps = ((xb - xa).abs().max((yb - ya).abs()).ceil() as usize).max(1);
        for k in 0..=steps {
            let f = k as f32 / steps as f32;
            let (x, y) = (at.0 + xa + f * (xb - xa), ya + f * (yb - ya));
            let (y0, y1) = (
                (y - line / 2.0).round() as i64,
                (y + line / 2.0).round() as i64,
            );
            for qy in y0..y1 {
                if qy >= 0 && x >= 0.0 {
                    put(x.round() as usize, qy as usize, singer, alpha * 0.9);
                }
            }
        }
    }
    let Some((x, height, way)) = sung.voice else {
        return;
    };
    // Sparkles on, white hot near the singer: gold on the note, coral off
    // it.
    let glint = if way == 0 {
        mix([255, 255, 255], hit, 0.45)
    } else {
        mix([255, 255, 255], singer, 0.55)
    };
    // The tail: the last of their frames shed sparkles, which drift back
    // and apart, shrink, twinkle and fade.
    let n = sung.trail.len();
    for k in 0..SPARKLE_FRAMES.min(n) {
        let i = n - 1 - k;
        let r = |j: u32| noise(i as u32 * 5 + j);
        if r(0) > SPARKLE_SHARE {
            continue;
        }
        let age = k as f32 / SPARKLE_FRAMES as f32;
        let (tx, th) = sung.trail[i];
        let sx = at.0 + tx - age * px * (0.3 + 0.6 * r(1));
        let sy = y_of(th) + thick / 2.0 + (r(2) - 0.5) * age * px * 1.1;
        let size = px * (0.34 - 0.22 * age) * (0.55 + 0.6 * r(3));
        let twinkle = 0.55 + 0.45 * (age * 28.0 + r(4) * std::f32::consts::TAU).sin();
        let a = (1.0 - age).powf(1.4) * twinkle * alpha;
        sparkle(&mut |qx, qy, c| put(qx, qy, glint, c), sx, sy, size, a);
    }
    let (x, y) = (at.0 + x, y_of(height) + thick / 2.0);
    // A glow around the singer now.
    let glow = thick * 3.0;
    for qy in ((y - glow).floor() as i64).max(0)..(y + glow).ceil() as i64 {
        for qx in ((x - glow).floor() as i64).max(0)..(x + glow).ceil() as i64 {
            let (dx, dy) = (qx as f32 + 0.5 - x, qy as f32 + 0.5 - y);
            let f = 1.0 - (dx * dx + dy * dy).sqrt() / glow;
            if f > 0.0 {
                put(qx as usize, qy as usize, glint, alpha * 0.45 * f * f);
            }
        }
    }
    let r = thick * 1.2;
    let (x0, x1) = ((x - r).floor() as i64, (x + r).ceil() as i64);
    let (y0, y1) = ((y - r).floor() as i64, (y + r).ceil() as i64);
    for qy in y0.max(0)..y1 {
        for qx in x0.max(0)..x1 {
            let (dx, dy) = (qx as f32 + 0.5 - x, qy as f32 + 0.5 - y);
            let cover = (r - (dx * dx + dy * dy).sqrt() + 0.5).clamp(0.0, 1.0);
            if cover > 0.0 {
                put(qx as usize, qy as usize, singer, alpha * cover);
            }
        }
    }
}
