//! The count-in on the stage ([`Stage::count`]): its number in the lane
//! behind the playhead, where the singing starts and no bubble is yet
//! (clear of the meter and badges above the lane and of the lines below
//! it; it never moves with them), shrinking and fading over its beat; the
//! cue before the singing comes back after a break half as large, in the
//! same place. Where it goes ([`placed`]) is the same for the stage and for
//! the window without a GPU, which paints it into its pixmap.

use super::super::big_text::{BigText, Face};
use super::super::stage_gpu::Instance;
use super::score::rgba8;
use super::{Lane, Stage};
use crate::view::karaoke::KaraokePicture;

/// How large the count-in's number is at most, a share of the screen's
/// height; the cue's.
const COUNT: f32 = 0.26;
const CUE: f32 = 0.13;

/// The count-in's number as it is drawn.
#[derive(Clone, Debug, PartialEq)]
pub struct Count {
    /// The number.
    pub text: String,
    /// The size its glyphs are made at.
    pub base: f32,
    /// The size it is drawn at.
    pub px: f32,
    /// Its line's top-left.
    pub at: (f32, f32),
    /// How much of it shows.
    pub alpha: f32,
}

/// `picture`'s count-in (or its cue), none without one, in the window's
/// `area` (left, top, width, height) `room` pixels high, its glyphs in
/// `text` (the shouts' face) made at the size `base` gives for the size it
/// is drawn at and the largest it is drawn at.
pub fn placed(
    text: &mut BigText,
    picture: &KaraokePicture,
    (x, y, w, h): (f32, f32, f32, f32),
    room: f32,
    base: impl Fn(f32, f32) -> f32,
) -> Option<Count> {
    let count = picture.count.map(|c| (c, COUNT));
    let ((n, through), size) = count.or(picture.cue.map(|c| ((c.number, c.through), CUE)))?;
    let lane = Lane::new((x, y, w, h), 0.0);
    let behind = lane.head - x;
    // Smaller at once (ease-out), fading late, so it reads through most of
    // its beat.
    let t = through.clamp(0.0, 1.0) as f32;
    let shrink = 1.0 - (1.0 - t).powi(3);
    let most = (room * size).min(behind * 0.9);
    let px = most * (1.0 - 0.4 * shrink);
    let base = base(px, most);
    let digit = n.to_string();
    let wide = text.measure_from(&digit, base, px);
    Some(Count {
        text: digit,
        base,
        px,
        at: (
            x + (behind - wide) / 2.0,
            (lane.band.0 + lane.band.1 - px) / 2.0,
        ),
        alpha: 1.0 - t * t * t,
    })
}

impl Stage {
    /// `picture`'s count-in into `out`, in the window's `area`: its glyphs
    /// made once at the largest it is drawn (the atlas holds a number a
    /// size), drawn smaller from there.
    pub(super) fn count(
        &mut self,
        out: &mut Vec<Instance>,
        picture: &KaraokePicture,
        area: (f32, f32, f32, f32),
    ) {
        let shout = self.faces.get(Face::SHOUT);
        let Some(c) = placed(shout, picture, area, area.3, |_, most| most.round()) else {
            return;
        };
        let accent = rgba8(self.theme.accent(), c.alpha);
        let number = (Face::SHOUT, c.text.as_str());
        let split = f32::INFINITY;
        self.set_text(out, number, (c.base, c.px), c.at, split, (accent, accent));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::karaoke::{Cue, LineKind, LyricLine};

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

    fn picture(lines: Vec<LyricLine>, before: Vec<LyricLine>, moved: f64) -> KaraokePicture {
        KaraokePicture {
            bar: 5.0 / 24.0,
            lines,
            count: None,
            cue: None,
            controls: None,
            start: None,
            before,
            moved,
            score: None,
            result: None,
        }
    }

    /// The count before the singing comes back sits in the lane behind
    /// the playhead, the same whatever lines show or how they move: never
    /// part of them.
    #[test]
    fn the_break_count_stays_put_while_the_lines_move() {
        let area = (0.0, 0.0, 640.0, 360.0);
        let cue = Some(Cue {
            number: 3,
            through: 0.2,
            line: 1,
        });
        let mut stage = Stage::default();
        let mut digit_at = |p: KaraokePicture| {
            let mut out = Vec::new();
            stage.count(&mut out, &p, area);
            assert_eq!(out.len(), 1, "{out:?}");
            let at = (out[0].quad[0], out[0].quad[1]);
            let lane = Lane::new(area, 0.0);
            assert!(out[0].quad[2] < lane.head, "behind the playhead: {at:?}");
            let middle = (out[0].quad[1] + out[0].quad[3]) / 2.0;
            assert!(
                middle > lane.band.0 && middle < lane.band.1,
                "in the lane: {at:?}"
            );
            at
        };
        let pausing = vec![
            lyric("sung before", 1.0, LineKind::Previous, Some(0)),
            lyric("♪", 0.0, LineKind::Info, None),
            lyric("coming back", 0.0, LineKind::Next, Some(1)),
        ];
        let coming = vec![
            lyric("sung before", 1.0, LineKind::Previous, Some(0)),
            lyric("coming back", 0.0, LineKind::Current, Some(1)),
            lyric("after it", 0.0, LineKind::Next, Some(2)),
        ];
        let a = digit_at(KaraokePicture {
            cue,
            ..picture(pausing.clone(), Vec::new(), 1.0)
        });
        let b = digit_at(KaraokePicture {
            cue,
            ..picture(coming.clone(), pausing, 0.5)
        });
        let c = digit_at(KaraokePicture {
            cue,
            ..picture(coming, Vec::new(), 1.0)
        });
        assert_eq!((a, b), (c, c));
    }
}
