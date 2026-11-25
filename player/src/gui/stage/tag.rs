//! The demo's tag: "AUTO", small in the stage's top left corner while the
//! demo key is held ([`super::Song::demo`]), so a demo never passes for
//! singing.

use super::super::stage_gpu::Instance;
use super::result::Pen;
use super::{pill, rgba, Face, Stage, GOLD_TOP, WHITE};

/// What it says.
pub const TAG: &str = "AUTO";

impl Stage {
    /// The tag into `out`, in the top left corner of `area`.
    pub(super) fn demo_tag(&mut self, out: &mut Vec<Instance>, (x, y, _, h): (f32, f32, f32, f32)) {
        let px = (h * 0.024).round().clamp(10.0, 44.0);
        let w = self.measure(Face::SHOUT, TAG, px);
        let half = (w / 2.0 + px * 0.6, px * 0.75);
        let c = (x + px + half.0, y + px + half.1);
        out.push(pill(
            c,
            half,
            [0.04, 0.03, 0.09, 0.6],
            rgba(GOLD_TOP, 0.7),
            0.0,
            1.5,
            0.0,
        ));
        let text = (Face::SHOUT, TAG);
        self.written(
            out,
            text,
            (px, 1.0),
            (c, 0.5),
            rgba(WHITE, 0.9),
            Pen::SHADED,
        );
    }
}
