//! Next on the stage: a gold button, "Next" and an arrow on it, the only
//! Next there is to click (its keys are `enter` and `space`), at the
//! picture's right edge a little above its middle while Karaoke mode waits
//! for it: on entering the mode, before each song, and beside a song's
//! result ([`super::result`], which names the song it plays). Its bottom
//! stays above where the pointer brings the controls up
//! ([`crate::view::karaoke::KaraokePicture::reach`]), so going to it
//! never does, and it is greyed while the song it plays cannot be sung
//! yet. Where it is drawn is
//! the stage's target for the window's pointer ([`Stage::targets`]): one
//! place says where it is, for the eye and the click alike.

use super::gpu::Instance;
use super::result::Pen;
use super::{pill, rgba, segment, Face, Stage, GOLD_DEEP, GOLD_TOP, WHITE};
use crate::core::keymap::Op;

/// The button's middle: a share of the height down.
const HIGH: f32 = 0.42;
/// The air round its word and arrow, across and up and down, in its
/// text's size.
const PAD: (f32, f32) = (0.7, 0.85);
/// How much larger it grows at the height of its pulse.
const PULSE: f32 = 0.06;
/// Its word.
const WORD: &str = "Next";
/// The gap between its word and its arrow, the arrow's depth (half of it)
/// and its half height, in its text's size.
const ARROW: (f32, f32, f32) = (0.32, 0.24, 0.34);

/// The size of the stage's small text in the window's `area` (pixels):
/// 2.9 % of its height, never under 10 pixels (Next's word, the line
/// naming the song it plays).
pub(super) fn text_px((_, _, _, h): (f32, f32, f32, f32)) -> f32 {
    let u = h / 100.0;
    (2.9 * u).round().max(10.0)
}

/// Where Next's button goes in the window's area (pixels): its middle,
/// half its width and height, its text's size and its word's width.
#[derive(Clone, Copy, Debug)]
pub(super) struct Place {
    pub c: (f32, f32),
    pub half: (f32, f32),
    pub px: f32,
    word: f32,
}

impl Place {
    /// The button's place in the window's `area` (pixels) for `stage`'s
    /// faces: at its right edge, a little above its middle, its bottom
    /// (at the height of its pulse) above the last `reach` of the height
    /// (the controls' reach:
    /// [`crate::view::karaoke::KaraokePicture::reach`]); its text
    /// [`text_px`], smaller where the window is narrow; as wide as its
    /// word and arrow.
    pub(super) fn of(stage: &mut Stage, area: (f32, f32, f32, f32), reach: f32) -> Place {
        let (x, y, w, h) = area;
        let u = h / 100.0;
        let px = text_px(area).min((w * 0.04).round()).max(10.0);
        let word = stage.measure(Face::SHOUT, WORD, px);
        let inner = word + px * (ARROW.0 + 2.0 * ARROW.1);
        let half = (inner / 2.0 + px * PAD.0, px * PAD.1);
        let air = (1.6 * u).max(8.0);
        let lowest = y + h * (1.0 - reach) - half.1 * (1.0 + PULSE);
        Place {
            c: (x + w - air - half.0, (y + h * HIGH).min(lowest)),
            half,
            px,
            word,
        }
    }

    /// How far left of the button what goes beside it keeps (pixels
    /// across): its left, pulsing, and some air.
    pub(super) fn clear_left(&self) -> f32 {
        self.c.0 - self.half.0 * (1.0 + PULSE) - self.px
    }
}

/// How Next's button shows: how much (0 to 1, fading in), how far through
/// its pulse (0 to 1), how lit its gold is, and whether the song it plays
/// can be sung (else greyed).
#[derive(Clone, Copy, Debug)]
pub(super) struct Look {
    pub shows: f32,
    pub pulse: f32,
    pub lit: f32,
    pub ready: bool,
}

impl Look {
    /// Shown whole and still, `ready` or greyed.
    pub(super) fn still(ready: bool) -> Look {
        Look {
            shows: 1.0,
            pulse: 0.0,
            lit: 1.0,
            ready,
        }
    }
}

impl Stage {
    /// Next's button at `place` into `out` as `look` says: a gold pill,
    /// "Next" and an arrow pointing right on it; where it is, a target
    /// ([`Stage::targets`]) while it shows at all.
    pub(super) fn next_button(&mut self, out: &mut Vec<Instance>, place: &Place, look: Look) {
        let (c, px) = (place.c, place.px);
        let (shows, grey) = (look.shows, if look.ready { 1.0 } else { 0.4 });
        let scale = 1.0 + PULSE * look.pulse;
        let half = (place.half.0 * scale, place.half.1 * scale);
        out.push(pill(
            c,
            half,
            rgba(GOLD_DEEP, 0.9 * shows * look.lit * grey),
            rgba(GOLD_TOP, (0.5 + 0.5 * look.pulse) * shows * grey),
            0.0,
            1.5,
            0.0,
        ));
        // The word, then the arrow, centred together.
        let word = place.word * scale;
        let [gap, deep, tall] = [ARROW.0, ARROW.1, ARROW.2].map(|a| px * a * scale);
        let left = c.0 - (word + gap + 2.0 * deep) / 2.0;
        let white = rgba(WHITE, shows * (0.5 + 0.5 * grey));
        let at = ((left, c.1), 0.0);
        self.written(
            out,
            (Face::SHOUT, WORD),
            (px, scale),
            at,
            white,
            Pen::SHADED,
        );
        let tip = (left + word + gap + 2.0 * deep, c.1);
        let thick = px * 0.075 * scale;
        for end in [-tall, tall] {
            let from = (tip.0 - 2.0 * deep, c.1 + end);
            out.push(segment(from, tip, thick, white, [0.0; 4], 0.0));
        }
        if shows > 0.0 {
            let at = [c.0 - half.0, c.1 - half.1, c.0 + half.0, c.1 + half.1];
            self.targets.push((at, Op::KaraokeNext));
        }
    }
}
