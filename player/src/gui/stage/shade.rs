//! Dark copies of the stage's text under it ([`Shade`]), so it reads over
//! whatever is behind: a soft shadow dropped under the result's words, a
//! soft dark rim round the lyrics and a lit bubble's letters over the
//! cover.

use super::super::stage_gpu::Instance;
use super::{Face, Stage};

/// Dark copies of text under it, so it reads over whatever is behind:
/// where each lies from the text (shares of its size; at least a pixel)
/// and how dark each is.
#[derive(Clone, Copy, Debug)]
pub(super) struct Shade {
    at: &'static [(f32, f32)],
    dark: f32,
}

impl Shade {
    /// A soft shadow dropped under it.
    pub(super) const DROP: Shade = Shade {
        at: &[(0.0, 0.05)],
        dark: 0.55,
    };
    /// A soft dark rim round it (text over the cover): copies to either side
    /// a little down, and one up.
    pub(super) const RIM: Shade = Shade {
        at: &[(-0.03, 0.02), (0.03, 0.02), (0.0, -0.03)],
        dark: 0.3,
    };
}

impl Stage {
    /// `shade`'s dark copies of a run of text as [`Stage::text`] puts it
    /// (`text` in `face` at `px`, drawn at `scale`, its line's top-left at
    /// `at`, as far as `split`), under it: where they start in `out`. Copies,
    /// not a blur in the shader: the atlas has two pixels round each run.
    pub(super) fn shade(
        &mut self,
        out: &mut Vec<Instance>,
        (face, text): (Face, &str),
        (px, scale): (f32, f32),
        (at, split): ((f32, f32), f32),
        (shade, alpha): (Shade, f32),
    ) -> usize {
        let from = out.len();
        let dark = [0.02, 0.01, 0.05, shade.dark * alpha];
        let off = |k: f32| {
            if k == 0.0 {
                0.0
            } else {
                (k * px).abs().max(1.0).copysign(k) * scale
            }
        };
        for &(dx, dy) in shade.at {
            let at = (at.0 + off(dx), at.1 + off(dy));
            self.text(out, face, text, (px, scale), at, split, ([0.0; 4], dark));
        }
        from
    }
}
