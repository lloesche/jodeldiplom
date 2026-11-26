//! START on the stage: a gold pill, the only START there is to click (its
//! keys are `enter` and `space`), under the picture while Karaoke mode
//! waits for it: on entering the mode, before each song, and under a
//! song's result (the song it plays named before it, [`super::result`]).
//! It stands above the controls' bar whether that shows or not
//! ([`KaraokePicture::bar`]), so it never moves for it, and greyed while the
//! song it plays cannot be sung yet. Where it is drawn is the stage's
//! target for the window's pointer ([`Stage::targets`]): one place says
//! where it is, for the eye and the click alike.

use super::gpu::Instance;
use super::result::Pen;
use super::{pill, rgba, Face, Stage, GOLD_DEEP, GOLD_TOP, WHITE};
use crate::core::keymap::Op;
use crate::view::karaoke::KaraokePicture;

/// The middle of START's line: a share of the height, unless the controls'
/// bar is higher.
const LINE: f32 = 0.935;

/// Where START's line goes in the window's area (pixels): across `x` to
/// `x + w`, its middle at `y`, its text `px` high.
#[derive(Clone, Copy, Debug)]
pub(super) struct Line {
    pub x: f32,
    pub w: f32,
    pub y: f32,
    pub px: f32,
}

impl Line {
    /// START's line for `picture` in the window's `area` (pixels): low,
    /// but clear above its controls' bar (shown now or not).
    pub(super) fn of(picture: &KaraokePicture, area: (f32, f32, f32, f32)) -> Line {
        Line::above(area, area.1 + (area.3 - area.3 * picture.bar))
    }

    /// START's line in the window's area (pixels), clear above `bottom`.
    pub(super) fn above((x, y, w, h): (f32, f32, f32, f32), bottom: f32) -> Line {
        let u = h / 100.0;
        Line {
            x,
            w,
            y: (y + h * LINE).min(bottom - 3.2 * u),
            px: (2.9 * u).round().max(10.0),
        }
    }
}

/// How START's pill shows: how much (0 to 1, fading in), how far through
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
    /// START's pill on `line`, `words` before it (fitted to the room left,
    /// in their colour) where there are any, centred together, into `out`
    /// as `look` says; where it is, a target ([`Stage::targets`]) while it
    /// shows at all.
    pub(super) fn start_pill(
        &mut self,
        out: &mut Vec<Instance>,
        line: &Line,
        words: Option<(&str, [f32; 4])>,
        look: Look,
    ) {
        let px = line.px;
        let start = (Face::SHOUT, "START");
        let sw = self.measure(start.0, start.1, px);
        let (words, gap) = match words {
            Some((text, color)) => {
                let room = line.w * 0.92 - sw - px * 2.4;
                let (text, _) = self.fitted(Face::NUMBER, text, px, room);
                (Some((text, color)), px * 1.4)
            }
            None => (None, 0.0),
        };
        let tw = words
            .as_ref()
            .map_or(0.0, |(text, _)| self.measure(Face::NUMBER, text, px));
        let left = line.x + line.w / 2.0 - (tw + gap + sw + px) / 2.0;
        if let Some((text, color)) = &words {
            let next = (Face::NUMBER, &**text);
            let at = ((left, line.y), 0.0);
            self.written(out, next, (px, 1.0), at, *color, Pen::SHADED);
        }
        let (shows, grey) = (look.shows, if look.ready { 1.0 } else { 0.4 });
        let scale = 1.0 + 0.06 * look.pulse;
        let c = (left + tw + gap + (sw + px) / 2.0, line.y);
        let half = (sw / 2.0 + px * 0.5, px * 0.8);
        out.push(pill(
            c,
            (half.0 * scale, half.1 * scale),
            rgba(GOLD_DEEP, 0.9 * shows * look.lit * grey),
            rgba(GOLD_TOP, (0.5 + 0.5 * look.pulse) * shows * grey),
            0.0,
            1.5,
            0.0,
        ));
        let white = rgba(WHITE, shows * (0.5 + 0.5 * grey));
        self.written(out, start, (px, scale), (c, 0.5), white, Pen::SHADED);
        if shows > 0.0 {
            let at = [c.0 - half.0, c.1 - half.1, c.0 + half.0, c.1 + half.1];
            self.targets.push((at, Op::KaraokeStart));
        }
    }
}
