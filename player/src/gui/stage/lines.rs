//! The line sung and the next at the bottom of the stage, in Karaoke
//! mode's words' face ([`Face::LINE`]): the line sung filling as it is
//! sung, the next smaller and dimmer under it, both moving up as the
//! lines change (the one sung leaving upwards as it fades); the line sung
//! in its lyric treatment, if the effects have one on it
//! ([`crate::gui::fx::Text::Sung`]). Where they are is the lyric band
//! ([`band`]), which the screen passes keep mostly clear.

use super::gpu::Instance;
use super::{rgba, Face, Shade, Stage, GOLD, WHITE};
use crate::gui::fx::{self, Frame, Text};
use crate::view::karaoke::{KaraokePicture, LineKind, LyricLine};

/// The line sung and the next: their tops, shares of the height.
const LINES: (f32, f32) = (0.68, 0.78);

/// The lines' size in the window's `h` (pixels high): the line sung's.
fn big(h: f32) -> f32 {
    (h / 15.0).clamp(18.0, 120.0).round()
}

/// The lyric band in the window's `area`: from the line sung's top to the
/// next's bottom (the next a little smaller: [`LINES`]).
pub(crate) fn band((_, y, _, h): (f32, f32, f32, f32)) -> (f32, f32) {
    (y + h * LINES.0, y + h * LINES.1 + big(h) * 0.66 * 1.3)
}

impl Stage {
    /// Draws `picture`'s line sung and the next into `out`, in the window's
    /// `area`, the line sung in the lyric treatment `effects` have on it.
    pub(super) fn lines(
        &mut self,
        out: &mut Vec<Instance>,
        picture: &KaraokePicture,
        (x, y, w, h): (f32, f32, f32, f32),
        effects: &Frame,
    ) {
        // The line sung and the next, moving up as the lines change.
        let big = big(h);
        let treat = fx::treatment(Text::Sung, effects);
        let slot = |kind: LineKind| -> Option<(f32, f32, f32)> {
            let (cur, next) = (y + h * LINES.0, y + h * LINES.1);
            match kind {
                LineKind::Current => Some((cur, 1.0, 1.0)),
                LineKind::Next => Some((next, 0.66, 0.75)),
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
        let now: Vec<&LyricLine> = picture.lines.iter().collect();
        let before: Vec<&LyricLine> = picture.before.iter().filter(|b| b.id.is_some()).collect();
        let (was, gone) = super::lyrics::moved(&now, &before);
        for (i, &line) in now.iter().enumerate().filter(|(_, l)| l.id.is_some()) {
            if line.kind == LineKind::Next && Some(i) != first_next {
                continue;
            }
            let Some((ty, scale, alpha)) = slot(line.kind) else {
                continue;
            };
            let from = was[i].and_then(|j| slot(before[j].kind)).unwrap_or((
                ty + (y + h * LINES.1 - y - h * LINES.0),
                scale,
                0.0,
            ));
            let lerp = |a: f32, b: f32| a + (b - a) * moved;
            drawn.push((
                line,
                lerp(from.0, ty),
                lerp(from.1, scale),
                lerp(from.2, alpha),
            ));
        }
        if moved < 1.0 {
            for b in gone.into_iter().map(|j| before[j]) {
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
            // A dark outline, so it reads over any cover as it scrolls.
            let rim = (Shade::OUTLINE, alpha);
            let whole = ((lx, ty), f32::INFINITY);
            self.shade(out, (Face::LINE, &line.text), (big, scale), whole, rim);
            let run = self.text(
                out,
                Face::LINE,
                &line.text,
                (big, scale),
                (lx, ty),
                split,
                (rgba(WHITE, 0.95 * alpha), rgba(GOLD, alpha)),
            );
            let sung = run.filter(|_| line.kind == LineKind::Current);
            if let (Some(run), Some(t)) = (sung, treat) {
                t.apply(run);
            }
        }
    }
}
