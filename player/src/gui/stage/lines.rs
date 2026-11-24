//! The line sung and the next at the bottom of the stage, in Karaoke
//! mode's words' face ([`Face::LINE`]): the line sung filling as it is
//! sung, the next smaller and dimmer under it, both moving up as the
//! lines change (the one sung leaving upwards as it fades).

use super::super::stage_gpu::Instance;
use super::{rgba, Face, Shade, Stage, GOLD, WHITE};
use crate::view::karaoke::{KaraokePicture, LineKind, LyricLine};

/// The line sung and the next: their tops, shares of the height.
const LINES: (f32, f32) = (0.68, 0.78);

impl Stage {
    /// Draws `picture`'s line sung and the next into `out`, in the window's
    /// `area`.
    pub(super) fn lines(
        &mut self,
        out: &mut Vec<Instance>,
        picture: &KaraokePicture,
        (x, y, w, h): (f32, f32, f32, f32),
    ) {
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
            let width = self.measure(Face::LINE, &line.text, big);
            let scale = scale * (w * 0.92 / width.max(1.0)).min(1.0);
            let lw = width * scale;
            let lx = x + (w - lw) / 2.0;
            let split = if line.kind == LineKind::Current {
                let n = line.text.chars().count();
                let at = line.fill.clamp(0.0, 1.0) * n as f64;
                let whole = at.floor() as usize;
                let before: String = line.text.chars().take(whole).collect();
                let mut sx = self.measure(Face::LINE, &before, big);
                if let Some(c) = line.text.chars().nth(whole) {
                    let wide = self.measure(Face::LINE, &c.to_string(), big);
                    sx += (at - whole as f64) as f32 * wide;
                }
                lx + sx * scale
            } else {
                f32::NEG_INFINITY
            };
            // A soft dark rim, so it reads over a bright cover.
            let rim = (Shade::RIM, alpha);
            let whole = ((lx, ty), f32::INFINITY);
            self.shade(out, (Face::LINE, &line.text), (big, scale), whole, rim);
            self.text(
                out,
                Face::LINE,
                &line.text,
                (big, scale),
                (lx, ty),
                split,
                (rgba(WHITE, 0.95 * alpha), rgba(GOLD, alpha)),
            );
        }
    }
}
