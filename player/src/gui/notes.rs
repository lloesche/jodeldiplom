//! The notes under the line being sung in Karaoke mode's picture
//! ([`super::karaoke`]), painted on the processor for a window without a
//! GPU (with one, the stage draws them: [`super::stage::lyrics`]): a bar
//! under the letters each is sung on, as high as it goes against the
//! line's middle ([`bar`]); those sung already in the notes' colour, those
//! to come darker, those hit filled flat in gold; the singer's pitch along
//! them as a line, sparkles shed behind it and a dot where they are now. A
//! word's bars whose bubble bursts are no longer painted from the moment
//! it starts to (the picture changes once), and a bubble burst is simply
//! gone.

use super::pixels::{mix, over, PixelRect, Target};
use super::sparks::{noise, sparkle};
use super::stage::lyrics::{bar, span, SPARKLE_FRAMES, SPARKLE_SHARE};
use crate::view::karaoke::{LyricLine, HIT, NOTE_AHEAD, SINGER};

/// The notes under a line and the singing on them, in pixels from the
/// text's left.
#[derive(Clone, Debug, PartialEq)]
pub(super) struct SungOp {
    /// Each note's left and right, and its height (-1 to 1).
    notes: Vec<(f32, f32, f32)>,
    /// Where they are filled: left and right.
    hits: Vec<(f32, f32)>,
    /// The singer's pitch along them: where and how high.
    trail: Vec<(f32, f32)>,
    /// The singer now: where, how high, and which way to go (0 on the
    /// note).
    voice: Option<(f32, f32, i8)>,
}

impl SungOp {
    /// `line`'s notes and the singing on them, at text `px` high, `x_of`
    /// giving where a character (a fraction through one) is; none without
    /// notes.
    pub(super) fn of(
        line: &LyricLine,
        px: f32,
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
        Some(SungOp {
            notes,
            hits: line.hits.iter().map(|&(a, b)| (x_of(a), x_of(b))).collect(),
            trail: line.trail.iter().map(|&(at, h)| (x_of(at), h)).collect(),
            voice: line.voice.map(|(at, h, way)| (x_of(at), h, way)),
        })
    }
}

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
    let (sung_c, ahead, hit) = (crate::view::lanes::NOTE, NOTE_AHEAD, HIT);
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
    let singer = SINGER;
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
