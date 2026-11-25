//! Karaoke mode in the canvas: its picture painted into the pixmap (the
//! backdrop, the lyrics, the count-in and the score, the controls' bar
//! over them) only where it changed, or moves by itself (the score rolling
//! up); and the stage the GPU draws over it, from the same picture.

use std::time::Instant;

use super::Canvas;
use crate::art::Art;
use crate::gui::karaoke::{self, Backdrop};
use crate::gui::raster::{PixelRect, Target};
use crate::gui::{notes, stage, stage_gpu};
use crate::view::karaoke::KaraokePicture;

impl Canvas {
    /// Karaoke mode's stage for the frame just drawn, at `heard` seconds
    /// into the song (as `song` has it heard, once analysed), `dt` seconds
    /// after the last; the result screen in `tint` while one shows; none
    /// where the stage does not show.
    pub fn stage_frame(
        &mut self,
        (state, song): (
            &crate::core::State,
            Option<&crate::gui::backdrop::music::Song>,
        ),
        heard: f64,
        dt: f32,
        tint: &stage::Tint,
    ) -> Option<stage_gpu::StageFrame> {
        let (picture, rect) = self.staged.as_ref()?;
        let (x, y, w, h) = *rect;
        let area = (x as f32, y as f32, w as f32, h as f32);
        let song = (&stage::Song::of(state, song), stage::wanted(state));
        self.stage.scene(song, picture, area, (heard, dt), tint)
    }

    /// The target the stage drew in the frame shown at (`x`, `y`) (window
    /// pixels), if any: START's pill.
    pub fn target_at(&self, x: f32, y: f32) -> Option<crate::core::mouse::Target> {
        self.staged.as_ref()?;
        let inside = |[x0, y0, x1, y1]: [f32; 4]| (x0..=x1).contains(&x) && (y0..=y1).contains(&y);
        let mut targets = self.stage.targets().iter();
        targets
            .find(|(at, _)| inside(*at))
            .map(|&(_, target)| target)
    }

    /// What the GPU draws over Karaoke mode's picture as last painted (none
    /// without it): its bubbles' liquid and bursts.
    pub fn picture_over(&self) -> Option<&notes::Over> {
        self.karaoke.as_ref().map(|_| &self.karaoke_layer.over)
    }

    /// Whether Karaoke mode's picture, shown, moves by itself: its score
    /// rolling up.
    pub(super) fn karaoke_moving(&self) -> bool {
        self.karaoke.is_some() && self.karaoke_layer.odometer.moving()
    }

    /// Karaoke mode's picture `full` in the frame at `now` (`last`: the
    /// one in the frame before, if there was one; `wanted`: the stage draws
    /// the song's lane, so its lyrics are not in the pixmap). On the GPU
    /// the stage is drawn over it, from all of it. Painted again only
    /// where it changed, the cells under it were redrawn, or it moves by
    /// itself; the mode just come, its score shows at once. Returns the
    /// pixels it changed.
    pub(super) fn karaoke_picture(
        &mut self,
        full: &KaraokePicture,
        last: Option<&KaraokePicture>,
        wanted: bool,
        art: Option<&Art>,
        now: Instant,
    ) -> Vec<PixelRect> {
        if self.gpu {
            let rect = self.pixel_rect(full.area);
            self.staged = Some((full.clone(), rect));
        }
        let picture = if !self.gpu {
            // No stage, no pill: START said in words.
            crate::view::karaoke::with_start_cue(full)
        } else if wanted {
            stage::without_lyrics(full)
        } else {
            full.clone()
        };
        if last.is_none() {
            self.karaoke_layer.odometer.step(None, 0.0);
        }
        let damaged = self.terminal.backend().damage.touches(picture.area);
        let kept = last == Some(&picture) && !damaged && !self.karaoke_layer.odometer.moving();
        let changed = if kept {
            Vec::new()
        } else {
            // Afresh where the cells under it were redrawn or its frame
            // (the area, the controls) changed; else only where its text
            // was and is.
            let afresh = damaged
                || last.is_none_or(|k| (k.area, k.controls) != (picture.area, picture.controls));
            self.karaoke_layer.at(now);
            self.paint_karaoke(&picture, art, afresh)
        };
        self.karaoke = Some(picture);
        changed
    }

    /// Karaoke mode's picture: the backdrop (made again for another cover
    /// or size), the lyrics, the count-in and the score, then the
    /// controls' text over it. `full`: all of it; else only the text,
    /// where it was and is (the backdrop put back under the last frame's).
    /// Returns the pixels it changed.
    fn paint_karaoke(
        &mut self,
        picture: &KaraokePicture,
        art: Option<&Art>,
        full: bool,
    ) -> Vec<PixelRect> {
        let area = self.pixel_rect(picture.area);
        let bar = picture.controls.map(|_| self.pixel_rect(picture.bar));
        // On the GPU the ground is its: all see-through here.
        let art_id = (!self.gpu).then(|| art.map_or(0, |a| a.id));
        let fits = self
            .backdrop
            .as_ref()
            .is_some_and(|b| (b.art, b.size) == (art_id, (area.2, area.3)));
        if !fits {
            self.backdrop = Some(if self.gpu {
                Backdrop::through(area.2, area.3)
            } else {
                Backdrop::new(art, area.2, area.3)
            });
        }
        let full = full || !fits;
        self.karaoke_layer.over.gpu = self.gpu;
        let Some(backdrop) = &self.backdrop else {
            return Vec::new();
        };
        let backend = self.terminal.backend_mut();
        let (width, height) = (backend.width, backend.height);
        let mut target = Target {
            pixels: &mut backend.pixmap,
            width,
            height,
        };
        let changed = karaoke::paint(
            &mut target,
            &mut self.faces,
            &mut self.karaoke_layer,
            backdrop,
            picture,
            area,
            bar,
            full,
        );
        if full {
            self.overlay_controls(picture);
        }
        changed
    }

    /// Karaoke mode's controls over its picture, where they are laid out:
    /// their cells' text and their bar, faded in as far as the bar shows
    /// ([`crate::core::karaoke::controls::handover`]).
    fn overlay_controls(&mut self, picture: &KaraokePicture) {
        let (Some(fade), Some(backdrop)) = (picture.controls, &self.backdrop) else {
            return;
        };
        let (area, bar) = (self.pixel_rect(picture.area), self.pixel_rect(picture.bar));
        let backend = self.terminal.backend_mut();
        backend.overlay(picture.bar);
        let (width, height) = (backend.width, backend.height);
        let mut target = Target {
            pixels: &mut backend.pixmap,
            width,
            height,
        };
        let shown = crate::core::karaoke::controls::handover(fade).1;
        backdrop.fade_bar(&mut target, area, bar, shown as f32);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::view::karaoke::Score;
    use ratatui::layout::Rect;
    use std::time::Duration;

    /// The picture with the score `points` and nothing else.
    fn scored(points: u32) -> KaraokePicture {
        KaraokePicture {
            area: Rect::new(0, 0, 80, 24),
            bar: Rect::new(0, 21, 80, 3),
            lines: Vec::new(),
            count: None,
            cue: None,
            controls: None,
            start: None,
            before: Vec::new(),
            moved: 1.0,
            score: Some(Score { points, streak: 0 }),
            result: None,
        }
    }

    /// A canvas of 640 × 384 pixels, and frames 1/60 s apart from now.
    fn canvas() -> (Canvas, impl Fn(usize) -> Instant) {
        let mut canvas = Canvas::new(80, 24, crate::gui::font_px(1.0));
        canvas.fit(640, 384);
        let t0 = Instant::now();
        (canvas, move |f| {
            t0 + Duration::from_secs_f32(f as f32 / 60.0)
        })
    }

    /// The same picture frame after frame is not painted again; points
    /// added, it is painted every frame while they roll up (moving), and
    /// no more once they are at rest.
    #[test]
    fn the_picture_is_painted_again_while_its_score_rolls_and_then_not() {
        let (mut canvas, at) = canvas();
        let mut last = None;
        let mut frame = |canvas: &mut Canvas, points: u32, f: usize| {
            // As each frame starts: no cells drawn under it yet.
            canvas.terminal.backend_mut().reset_damage();
            let picture = scored(points);
            let changed = canvas.karaoke_picture(&picture, last.as_ref(), false, None, at(f));
            last = canvas.karaoke.clone();
            !changed.is_empty()
        };
        assert!(frame(&mut canvas, 12_400, 0));
        assert!(!frame(&mut canvas, 12_400, 1));
        assert!(!canvas.moving());
        assert!(frame(&mut canvas, 13_200, 2));
        let mut painted = 1;
        let mut f = 3;
        while canvas.moving() {
            assert!(frame(&mut canvas, 13_200, f), "frame {f}");
            painted += 1;
            f += 1;
            assert!(f < 120, "still moving");
        }
        // Rolled in and its glow gone: about 0.4 s and 0.3 s of frames.
        assert!(painted > 30, "{painted}");
        assert!(!frame(&mut canvas, 13_200, f));
    }

    /// Points that came while the mode was left (20 000, then 30 000 on
    /// coming back) show at once: nothing rolls up.
    #[test]
    fn a_score_come_while_the_mode_was_left_shows_at_once() {
        let (mut canvas, at) = canvas();
        canvas.karaoke_picture(&scored(20_000), None, false, None, at(0));
        // Left: a frame without the picture, the layer kept.
        canvas.karaoke = None;
        canvas.karaoke_picture(&scored(30_000), None, false, None, at(600));
        assert!(!canvas.moving());
        let again = canvas.karaoke.clone();
        canvas.terminal.backend_mut().reset_damage();
        let changed = canvas.karaoke_picture(&scored(30_000), again.as_ref(), false, None, at(601));
        assert!(changed.is_empty());
    }
}
