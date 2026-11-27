//! What the guards measure on the Shader tab's picture, frame by frame
//! ([`Readout`]): the look drawn on the processor over the song's picture
//! made small ([`jodel_looks::render`]), with the frame's own numbers; the
//! light of each frame of the last few seconds (the flashes counted in
//! it), the lyric band's contrast, the frame's exposure and the strength
//! the look draws at. What the stack costs and drops is the look's own
//! ([`jodel_looks::Look::check`]), shown by the tab from the look.

use std::collections::VecDeque;
use std::time::Instant;

use jodel_looks::guard;
use jodel_looks::render::Image;
use jodel_looks::{Ctx, Pass};

use crate::core::shader_tab;
use crate::gui::backdrop::BackdropFrame;

/// How long the flashes are counted over, seconds.
const COUNTED: f64 = 3.0;

/// The light of the frames measured lately (each with when it was drawn)
/// and the measures last said.
#[derive(Default)]
pub struct Readout {
    lights: VecDeque<(Instant, f32)>,
    shown: Option<shader_tab::Readout>,
}

impl Readout {
    /// The measures after the frame `frame` drawn at `now` over the song's
    /// picture made small (`small`: [`crate::gui::backdrop::Visualiser::picture`]):
    /// new ones where they changed (each to the precision the tab shows).
    pub fn step(
        &mut self,
        (frame, small): (Option<&BackdropFrame>, &Image),
        now: Instant,
    ) -> Option<shader_tab::Readout> {
        let (w, h) = (small.w as f32, small.h as f32);
        let mut contrast = guard::band_contrast(&small.px, small.w, jodel_looks::BAND);
        let mut exposure = guard::exposure(&small.px);
        let mut strength = 1.0;
        if let Some(look) = frame.and_then(|f| f.look.as_ref()) {
            strength = look.cue.strength;
            let passes: Vec<(&'static Pass, Ctx)> = (look.passes.iter())
                .map(|&(p, c)| {
                    let mut c = c;
                    c.size = [w, h, 1.0 / w, 1.0 / h];
                    (p, c)
                })
                .collect();
            let out = jodel_looks::render::draw(&passes, small, None);
            self.lights.push_back((now, guard::light(&out.px)));
            exposure = guard::exposure(&out.px);
            if let Some((_, c)) = passes.first() {
                contrast =
                    guard::band_contrast(&out.px, out.w, (c.band[0] + 0.01, c.band[1] - 0.01));
            }
        }
        while (self.lights.front())
            .is_some_and(|(t, _)| now.saturating_duration_since(*t).as_secs_f64() > COUNTED)
        {
            self.lights.pop_front();
        }
        let span = (self.lights.front()).map_or(0.0, |(t, _)| {
            now.saturating_duration_since(*t).as_secs_f64()
        });
        let lights: Vec<f32> = self.lights.iter().map(|&(_, l)| l).collect();
        let fps = if span > 0.1 {
            (lights.len() as f64 / span) as f32
        } else {
            60.0
        };
        let r = shader_tab::Readout {
            flashes: guard::flashes(&lights, fps),
            contrast: (contrast * 10.0).round() / 10.0,
            clipped: (exposure.clipped * 100.0).round() / 100.0,
            mean: (exposure.mean * 100.0).round() / 100.0,
            strength,
        };
        (self.shown.as_ref() != Some(&r)).then(|| {
            self.shown = Some(r.clone());
            r
        })
    }
}
